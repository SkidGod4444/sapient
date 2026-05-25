//! Download tuning — parallel chunks, concurrent files, env overrides.
//!
//! Hugging Face Python clients can use `hf_xet` / legacy `hf_transfer`. Sapient uses
//! the Rust `hf-hub` crate, which speeds downloads via HTTP range requests and
//! concurrent connections (`ApiBuilder::high()`).

use hf_hub::api::tokio::ApiBuilder;

use crate::LoadOptions;

/// Default chunk size for resumable range downloads (10 MiB, matches `hf-hub::high()`).
pub const DEFAULT_CHUNK_SIZE: usize = 10_000_000;

/// Read max concurrent file/chunk workers from the environment.
pub fn max_parallel_downloads() -> usize {
    std::env::var("SAPIENT_HUB_MAX_PARALLEL")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            std::env::var("HF_HUB_MAX_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or_else(|| num_cpus::get().clamp(2, 8))
}

/// Read HTTP range chunk size in bytes.
pub fn chunk_size_bytes() -> usize {
    std::env::var("SAPIENT_HUB_CHUNK_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1_048_576) // minimum 1 MiB
        .unwrap_or(DEFAULT_CHUNK_SIZE)
}

/// Whether fast parallel downloads are enabled for this request.
pub fn fast_download_enabled(opts: &LoadOptions) -> bool {
    if std::env::var("SAPIENT_FAST_DOWNLOAD")
        .ok()
        .is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
    {
        return false;
    }
    if std::env::var("HF_HUB_ENABLE_HF_TRANSFER")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        return true;
    }
    opts.fast_download
}

/// Apply sapient download settings to an `hf-hub` API builder.
pub fn configure_api_builder(mut builder: ApiBuilder, opts: &LoadOptions) -> ApiBuilder {
    if opts.quiet {
        builder = builder.with_progress(false);
    }
    if fast_download_enabled(opts) {
        builder = builder
            .with_max_files(max_parallel_downloads())
            .with_chunk_size(Some(chunk_size_bytes()));
    }
    builder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_size_defaults_to_ten_mib() {
        assert_eq!(chunk_size_bytes(), DEFAULT_CHUNK_SIZE);
    }

    #[test]
    fn max_parallel_is_reasonable() {
        let n = max_parallel_downloads();
        assert!((2..=64).contains(&n));
    }

    #[test]
    fn fast_download_respects_option() {
        let mut opts = LoadOptions {
            fast_download: false,
            ..Default::default()
        };
        // Only assert when env doesn't force enable
        if std::env::var("HF_HUB_ENABLE_HF_TRANSFER").ok().as_deref() != Some("1") {
            assert!(!fast_download_enabled(&opts));
        }
        opts.fast_download = true;
        assert!(fast_download_enabled(&opts));
    }
}
