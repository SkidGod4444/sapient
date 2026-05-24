//! HuggingFace Hub helpers for the CLI.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sapient_hub::{HubClient, ModelFiles, ModelInfo};

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
    let hub = HubClient::new()?;
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

/// List models cached by the HuggingFace Hub client.
pub fn list_cached_models() -> Result<Vec<String>> {
    let Some(home) = dirs::home_dir() else {
        return Ok(vec![]);
    };

    let hub_cache = home.join(".cache/huggingface/hub");
    if !hub_cache.is_dir() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();
    for entry in std::fs::read_dir(&hub_cache)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(rest) = name.strip_prefix("models--") {
            models.push(rest.replace("--", "/"));
        }
    }
    models.sort();
    Ok(models)
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
