//! Ollama-style single-line download progress tracker.
//!
//! Shows a clean, unified bar for the entire model download:
//!   pulling openhorizon/phi-2  ████████████░░░░░░  2.3 GB / 5.2 GB  4.1 MB/s  eta 44s

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
    let bar = if total_bytes > 0 {
        let pb = ProgressBar::new(total_bytes);
        pb.set_style(
            ProgressStyle::with_template(
                "  {msg:30} {bar:40.cyan/237} {bytes:>10} / {total_bytes:<10} {bytes_per_sec:>10}  eta {eta}",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏░"),
        );
        pb
    } else {
        // Unknown size: use a spinner
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template(
                "  {msg:30} {spinner:.cyan} {bytes} downloaded  {bytes_per_sec}",
            )
            .unwrap(),
        );
        pb
    };

    bar.set_message(format!("pulling {model}"));
    bar.enable_steady_tick(Duration::from_millis(120));

    // ── Background polling task ───────────────────────────────────────────────
    let done_flag = done.clone();
    let dl_bytes = downloaded.clone();
    let blobs = blobs_dir.clone();
    let bar_clone = bar.clone();

    // We intentionally use std::thread so we don't need an active tokio runtime
    // during the monitor loop (the download itself runs on the tokio runtime).
    std::thread::spawn(move || {
        let mut last_sample_time = Instant::now();
        let mut last_sample_bytes = 0u64;
        let poll = Duration::from_millis(250);

        while !done_flag.load(Ordering::Relaxed) {
            std::thread::sleep(poll);

            // Sum all non-temp blobs on disk
            let on_disk = if let Some(ref dir) = blobs {
                dir_downloaded_bytes(dir)
            } else {
                dl_bytes.load(Ordering::Relaxed)
            };

            bar_clone.set_position(on_disk);

            // Update speed sample every second for stable rate display
            let elapsed = last_sample_time.elapsed();
            if elapsed >= Duration::from_secs(1) {
                let _delta = on_disk.saturating_sub(last_sample_bytes);
                last_sample_bytes = on_disk;
                last_sample_time = Instant::now();
            }
        }
    });

    DownloadProgressHandle { done, bar }
}
