//! HuggingFace Hub helpers for the CLI.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sapient_hub::{HubClient, LoadOptions, ModelFiles, ModelInfo};

/// Returns true for HuggingFace model IDs like `meta-llama/Llama-3.2-1B-Instruct`.
pub fn looks_like_hub_model_id(s: &str) -> bool {
    if s.starts_with('.') || s.starts_with('/') || s.contains('\\') {
        return false;
    }
    if Path::new(s).exists() {
        return false;
    }
    let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
    parts.len() >= 2
}

/// Download a model from the Hub (same as `sapient pull`).
pub async fn pull_model(model_id: &str) -> Result<ModelFiles> {
    pull_model_with_options(
        model_id,
        LoadOptions {
            quiet: false,
            ..Default::default()
        },
    )
    .await
}

/// Download a model with custom Hub options.
pub async fn pull_model_with_options(model_id: &str, opts: LoadOptions) -> Result<ModelFiles> {
    let hub = HubClient::with_options(opts)?;
    hub.download(model_id)
        .await
        .with_context(|| format!("failed to download '{model_id}'"))
}

/// Fetch model metadata from the Hub.
pub async fn fetch_model_info(model_id: &str) -> Result<ModelInfo> {
    let hub = HubClient::new()?;
    hub.model_info(model_id).await
}

/// Resolve a CLI model argument to a local file path (downloads Hub models first).
pub async fn resolve_model_path(model: &str) -> Result<PathBuf> {
    if looks_like_hub_model_id(model) {
        let files = pull_model(model).await?;
        pick_graph_weight(&files)
    } else {
        Ok(PathBuf::from(model))
    }
}

/// Pick a weight file that `Model::load` understands (ONNX / GGUF).
pub fn pick_graph_weight(files: &ModelFiles) -> Result<PathBuf> {
    for path in &files.weight_paths {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "onnx" || ext == "gguf" {
            return Ok(path.clone());
        }
    }
    anyhow::bail!(
        "No ONNX/GGUF weights found for file-based commands.\n\
         Use `sapient chat {model}` or `sapient run {model} --prompt \"...\"` instead.",
        model = files.model_id
    );
}

/// Root of the shared HuggingFace Hub model cache (`~/.cache/huggingface/hub`).
pub fn hub_cache_root() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let hub_cache = home.join(".cache/huggingface/hub");
    hub_cache.is_dir().then_some(hub_cache)
}

/// Convert a HuggingFace model ID to its on-disk cache directory name.
fn model_cache_dir_name(model_id: &str) -> String {
    format!("models--{}", model_id.replace('/', "--"))
}

/// Returns true if a model's blobs directory contains any `.sync.part` files,
/// which means a download was interrupted and the model is not fully cached.
pub fn has_stale_downloads(model_id: &str) -> bool {
    let actual_id = sapient_hub::registry::resolve_model_alias(model_id)
        .unwrap_or_else(|_| model_id.to_string());
    let Some(hub_cache) = hub_cache_root() else {
        return false;
    };
    let blobs_dir = hub_cache
        .join(format!("models--{}", actual_id.replace('/', "--")))
        .join("blobs");
    has_part_files(&blobs_dir)
}

/// Walk the blobs directory and return true if any `.sync.part` file exists.
fn has_part_files(dir: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries
        .flatten()
        .any(|e| e.file_name().to_string_lossy().ends_with(".sync.part"))
}

/// List models fully cached by the HuggingFace Hub client.
/// Models with interrupted downloads (`.sync.part` files present) are excluded —
/// they are partial and must be re-downloaded before use.
pub fn list_cached_models() -> Result<Vec<String>> {
    let Some(hub_cache) = hub_cache_root() else {
        return Ok(vec![]);
    };

    // Build reverse map: hf_repo_id -> alias (e.g. "microsoft/phi-2" -> "openhorizon/phi-2")
    let reverse_aliases = sapient_hub::registry::reverse_alias_map();

    let mut models = Vec::new();
    for entry in std::fs::read_dir(&hub_cache)? {
        let entry = entry?;
        let path = entry.path();

        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(rest) = name.strip_prefix("models--") {
            let snapshots_dir = path.join("snapshots");
            if snapshots_dir.is_dir() {
                let has_commits = std::fs::read_dir(&snapshots_dir)
                    .map(|mut dirs| dirs.next().is_some())
                    .unwrap_or(false);

                if has_commits {
                    // Skip models whose blobs directory still contains .sync.part
                    // files — these are interrupted downloads, not usable models.
                    let blobs_dir = path.join("blobs");
                    if has_part_files(&blobs_dir) {
                        continue;
                    }

                    let hf_id = rest.replace("--", "/");
                    if let Some(alias) = reverse_aliases.get(hf_id.to_lowercase().as_str()) {
                        models.push(alias.clone());
                    }
                }
            }
        }
    }
    models.sort();
    Ok(models)
}

/// Remove incomplete HuggingFace Hub downloads (`.sync.part` and stale `.lock` files).
pub fn clear_stale_downloads() -> Result<u64> {
    let Some(hub_cache) = hub_cache_root() else {
        return Ok(0);
    };

    let mut freed = 0u64;
    for entry in walk_files(&hub_cache)? {
        let name = entry.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".sync.part") || name.ends_with(".lock") {
            freed += entry.metadata()?.len();
            let _ = std::fs::remove_file(&entry);
        }
    }
    Ok(freed)
}

/// Delete one model from the local HuggingFace Hub cache.
pub fn clear_cached_model(model_id: &str) -> Result<u64> {
    let hub_cache = hub_cache_root().context("HuggingFace Hub cache directory not found")?;

    // Resolve the registry alias to the actual HuggingFace repo_id.
    // "openhorizon/deepseek-r1-8b" → "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF"
    // The cache directory is keyed on the real repo_id, not the alias.
    let actual_id = sapient_hub::registry::resolve_model_alias(model_id)
        .unwrap_or_else(|_| model_id.to_string());

    let dir = hub_cache.join(model_cache_dir_name(&actual_id));
    if !dir.is_dir() {
        anyhow::bail!("Model '{model_id}' is not in the local cache");
    }
    let bytes = dir_size(&dir)?;
    std::fs::remove_dir_all(&dir).with_context(|| format!("failed to remove '{model_id}'"))?;
    Ok(bytes)
}

/// Delete all models from the local HuggingFace Hub cache.
pub fn clear_all_cached_models() -> Result<(usize, u64)> {
    let models = list_cached_models()?;
    if models.is_empty() {
        return Ok((0, 0));
    }

    let mut total_bytes = 0u64;
    for model_id in &models {
        total_bytes += clear_cached_model(model_id)?;
    }
    let stale = clear_stale_downloads()?;
    Ok((models.len(), total_bytes + stale))
}

fn walk_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    if path.is_file() {
        return Ok(path.metadata()?.len());
    }
    for file in walk_files(path)? {
        total += file.metadata()?.len();
    }
    Ok(total)
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Save a HuggingFace token to the standard cache location.
pub fn save_hf_token(token: &str) -> Result<PathBuf> {
    let token = token.trim();
    if token.is_empty() {
        anyhow::bail!("token must not be empty");
    }

    let path = dirs::home_dir()
        .context("could not determine home directory")?
        .join(".cache/huggingface/token");

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, format!("{token}\n"))?;
    Ok(path)
}
