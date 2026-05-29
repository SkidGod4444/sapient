//! Ollama-style single-line download progress tracker.
//!
//! Shows a clean, unified bar for the entire model download:
//!   pulling openhorizon/phi-2  ████████████░░░░░░  2.3 GB / 5.2 GB  4.1 MB/s  eta 44s
//!
//! After all bytes land the bar switches to "verifying checksum…" while hf-hub
//! does its post-download SHA256 check, so users see meaningful feedback instead
//! of a confusing near-zero speed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

/// Walk a directory tree and sum the sizes of all data files, including
/// in-progress `.sync.part` downloads (written by hf-hub during active transfers).
/// Only advisory `.lock` files are excluded since they contain no model data.
fn dir_downloaded_bytes(root: &PathBuf) -> u64 {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Skip only advisory lock files — NOT .sync.part, which are the active
        // download temp files hf-hub writes during transfers. Excluding them makes
        // the bar appear stuck at 0 B/s while data is actually being written.
        if name.ends_with(".lock") {
            continue;
        }
        if path.is_dir() {
            total += dir_downloaded_bytes(&path);
        } else if let Ok(meta) = entry.metadata() {
            total += meta.len();
        }
    }
    total
}

/// A handle that owns the background progress-polling task.
/// Drop it (or call `finish`) to stop polling and complete the bar.
pub struct DownloadProgressHandle {
    done: Arc<AtomicBool>,
    bar: ProgressBar,
}

impl DownloadProgressHandle {
    /// Mark the download as finished and finalize the progress bar.
    pub fn finish_success(self, model: &str) {
        self.done.store(true, Ordering::Relaxed);
        self.bar.finish_and_clear();
        println!("✓ {model} pulled successfully");
    }

    /// Mark the download as finished with an error and clear the bar.
    pub fn finish_error(self) {
        self.done.store(true, Ordering::Relaxed);
        self.bar.abandon();
    }
}

/// Start a background task that polls the HuggingFace blobs directory every 250ms and renders
/// a single Ollama-style progress line. Returns a handle; call `finish_*` when done.
///
/// `total_bytes = 0` means the total is unknown (shows a spinner instead of a bar).
pub fn start_download_progress(
    model: &str,
    blobs_dir: Option<PathBuf>,
    total_bytes: u64,
) -> DownloadProgressHandle {
    let done = Arc::new(AtomicBool::new(false));
    let downloaded = Arc::new(AtomicU64::new(0));

    // ── Style ────────────────────────────────────────────────────────────────
    let download_style = ProgressStyle::with_template(
        "  {msg:30} {bar:40.cyan/237} {bytes:>10} / {total_bytes:<10} {bytes_per_sec:>10}  eta {eta}",
    )
    .unwrap()
    .progress_chars("█▉▊▋▌▍▎▏░");

    let verify_style = ProgressStyle::with_template(
        "  {msg:30} {bar:40.yellow/237} {bytes:>10} / {total_bytes:<10}  {spinner:.yellow}",
    )
    .unwrap()
    .progress_chars("█░░░░░░░░");

    let bar = if total_bytes > 0 {
        let pb = ProgressBar::new(total_bytes);
        pb.set_style(download_style.clone());
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template(
                "  {msg:30} {spinner:.cyan} {bytes} downloaded  {bytes_per_sec}",
            )
            .unwrap(),
        );
        pb
    };

    let model_name = model.to_string();
    bar.set_message(format!("pulling {model_name}"));
    bar.enable_steady_tick(Duration::from_millis(120));

    // ── Background polling task ───────────────────────────────────────────────
    let done_flag = done.clone();
    let dl_bytes = downloaded.clone();
    let blobs = blobs_dir.clone();
    let bar_clone = bar.clone();

    // We intentionally use std::thread so we don't need an active tokio runtime
    // during the monitor loop (the download itself runs on the tokio runtime).
    std::thread::spawn(move || {
        let poll = Duration::from_millis(250);
        // Track when bytes last changed — used to detect the post-download
        // SHA256 verification phase where hf-hub is CPU-busy but the file
        // has stopped growing, which otherwise shows near-zero B/s.
        let mut last_change_bytes = 0u64;
        let mut last_change_time = Instant::now();
        let mut in_verify_phase = false;

        while !done_flag.load(Ordering::Relaxed) {
            std::thread::sleep(poll);

            let on_disk = if let Some(ref dir) = blobs {
                dir_downloaded_bytes(dir)
            } else {
                dl_bytes.load(Ordering::Relaxed)
            };

            bar_clone.set_position(on_disk);

            // Detect the verification phase: bytes have been stable for ≥ 2 seconds
            // AND we've downloaded a meaningful fraction of the total (≥ 50%).
            // hf-hub verifies with SHA256 after the transfer completes — for a 1.7 GiB
            // file that can take 5–30 s. Switch the style + message so users see
            // "verifying…" rather than a confusing 300 B/s near-stall.
            if on_disk != last_change_bytes {
                last_change_bytes = on_disk;
                last_change_time = Instant::now();
                if in_verify_phase {
                    // Bytes changed again — back to downloading (resumed shard).
                    in_verify_phase = false;
                    bar_clone.set_style(download_style.clone());
                    bar_clone.set_message(format!("pulling {model_name}"));
                }
            } else if !in_verify_phase
                && total_bytes > 0
                && on_disk >= total_bytes / 2          // downloaded ≥ half
                && last_change_time.elapsed() >= Duration::from_secs(2)
            {
                in_verify_phase = true;
                bar_clone.set_style(verify_style.clone());
                bar_clone.set_message(format!("verifying {model_name}"));
            }
        }
    });

    DownloadProgressHandle { done, bar }
}
