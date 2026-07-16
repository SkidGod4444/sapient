// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Local cache management for downloaded models.

use std::path::PathBuf;

/// Returns the SAPIENT cache root: `$XDG_CACHE_HOME/sapient/hub` or `~/.cache/sapient/hub`.
pub fn cache_dir() -> PathBuf {
    std::env::var("SAPIENT_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("sapient")
                .join("hub")
        })
}

/// Return the on-disk path where a model would be cached.
pub fn model_cache_path(model_id: &str) -> PathBuf {
    // Replace `/` with `--` to keep it as a flat directory name.
    let safe = model_id.replace('/', "--");
    cache_dir().join(safe)
}
