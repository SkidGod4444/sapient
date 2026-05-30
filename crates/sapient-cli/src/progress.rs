//! Ollama-style single-line download progress tracker.
//!
//! Shows a clean, unified bar for the entire model download:
//!   pulling openhorizon/phi-2  ████████████░░░░░░  44%   2.3 GB / 5.2 GB  4.1 MB/s  eta 44s
//!
//! After all bytes land the bar switches to a spinner + "verifying checksum…" while
//! hf-hub does its post-download SHA256 check.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

/// Walk a directory tree and sum the sizes of all data files, including
/// in-progress `.sync.part` downloads (written by hf-hub during active transfers).
fn dir_downloaded_bytes(root: &PathBuf) -> u64 {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
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
pub struct DownloadProgressHandle {
    done: Arc<AtomicBool>,
    bar: ProgressBar,
}

impl DownloadProgressHandle {
    pub fn finish_success(self, model: &str) {
        self.done.store(true, Ordering::Relaxed);
        self.bar.finish_and_clear();
        println!("✓ {model} pulled successfully");
    }

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

    // ── Styles ────────────────────────────────────────────────────────────────

    // Active download: filled bar + percentage + speed + ETA
    let download_style = ProgressStyle::with_template(
        "  {msg:32} {bar:38.cyan/237} {percent:>3}%  {bytes:>10} / {total_bytes:<10}  {bytes_per_sec:>10}  eta {eta}",
    )
    .unwrap()
    .progress_chars("█▉▊▋▌▍▎▏░");

    // Post-download SHA256 verification: spinner + size (no bar — bar was confusing at 100%)
    let verify_style = ProgressStyle::with_template(
        "  {msg:32} {spinner:.yellow}  {bytes:>10} / {total_bytes:<10}",
    )
    .unwrap();

    let bar = if total_bytes > 0 {
        let pb = ProgressBar::new(total_bytes);
        pb.set_style(download_style.clone());
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template(
                "  {msg:32} {spinner:.cyan}  {bytes} downloaded  {bytes_per_sec}",
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

    std::thread::spawn(move || {
        let poll = Duration::from_millis(250);
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

            // Detect the SHA256 verification phase: bytes stable for ≥ 2s AND
            // we've downloaded a meaningful fraction (≥ 50%). Switch from the
            // bar+percentage style to a spinner so users see clear feedback
            // instead of a bar frozen at some percentage.
            if on_disk != last_change_bytes {
                last_change_bytes = on_disk;
                last_change_time = Instant::now();
                if in_verify_phase {
                    in_verify_phase = false;
                    bar_clone.set_style(download_style.clone());
                    bar_clone.set_message(format!("pulling {model_name}"));
                }
            } else if !in_verify_phase
                && last_change_time.elapsed() >= Duration::from_secs(2)
                && ((total_bytes > 0 && on_disk >= total_bytes / 2)
                    || (total_bytes == 0 && on_disk >= 50 * 1024 * 1024))
            {
                in_verify_phase = true;
                bar_clone.set_style(verify_style.clone());
                bar_clone.set_message(format!("verifying {model_name}"));
            }
        }
    });

    DownloadProgressHandle { done, bar }
}
