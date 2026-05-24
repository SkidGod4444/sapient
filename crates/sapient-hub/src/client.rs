//! HuggingFace Hub API client.

use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use tracing::{debug, info};

use crate::model_info::ModelInfo;
use crate::resolver::ModelFiles;

// ── LoadOptions ───────────────────────────────────────────────────────────────

/// Options for model loading from the Hub.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// HuggingFace access token. If `None`, reads `HF_TOKEN` env var, then
    /// `~/.cache/huggingface/token`.
    pub token: Option<String>,

    /// Preferred weight format, in priority order.
    /// Defaults to `["gguf", "safetensors", "bin"]`.
    pub formats: Vec<String>,

    /// If true, always re-download even if cached.
    pub force_download: bool,

    /// Maximum file size to download in bytes (0 = unlimited).
    pub max_size_bytes: u64,

    /// Model revision / branch (default: `"main"`).
    pub revision: String,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            token: None,
            formats: vec!["gguf".into(), "safetensors".into(), "bin".into()],
            force_download: false,
            max_size_bytes: 0,
            revision: "main".into(),
        }
    }
}

// ── HubClient ─────────────────────────────────────────────────────────────────

/// Client for the HuggingFace Hub REST API.
pub struct HubClient {
    api: Api,
    opts: LoadOptions,
}

impl HubClient {
    /// Create a new client, auto-reading the HF token from the environment.
    pub fn new() -> Result<Self> {
        Self::with_options(LoadOptions::default())
    }

    /// Create a new client with custom options.
    pub fn with_options(opts: LoadOptions) -> Result<Self> {
        let token = opts
            .token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .or_else(Self::read_cached_token);

        let mut builder = ApiBuilder::new();
        if let Some(t) = token {
            builder = builder.with_token(Some(t));
        }
        let api = builder
            .build()
            .context("Failed to build HF Hub API client")?;
        Ok(Self { api, opts })
    }

    /// Download a model by its HuggingFace model ID (e.g. `"meta-llama/Llama-3.2-1B"`).
    ///
    /// Returns the resolved local file paths — no Python, no git-lfs required.
    pub async fn download(&self, model_id: &str) -> Result<ModelFiles> {
        info!("Downloading model: {model_id}");
        let repo = self.api.model(model_id.to_owned());
        let files = self.resolve_files(&repo, model_id).await?;
        Ok(files)
    }

    /// Fetch model info / architecture type from the Hub (reads `config.json`).
    pub async fn model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let repo = self.api.model(model_id.to_owned());
        let config_path = repo
            .get("config.json")
            .await
            .context("Failed to fetch config.json")?;
        ModelInfo::from_config_file(&config_path)
    }

    // ── Internals ──────────────────────────────────────────────────────────────

    async fn resolve_files(&self, repo: &ApiRepo, model_id: &str) -> Result<ModelFiles> {
        // Always fetch config.json and tokenizer.json.
        let config_path = repo
            .get("config.json")
            .await
            .context("config.json not found — is this a valid model repo?")?;
        debug!("config.json cached at: {}", config_path.display());

        let tokenizer_path = repo.get("tokenizer.json").await.ok();

        let tokenizer_config_path = repo.get("tokenizer_config.json").await.ok();

        // Try each weight format in priority order.
        let weight_paths = self.fetch_weights(repo).await?;

        Ok(ModelFiles {
            model_id: model_id.to_owned(),
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            weight_paths,
        })
    }

    async fn fetch_weights(&self, repo: &ApiRepo) -> Result<Vec<PathBuf>> {
        let repo_info = repo
            .info()
            .await
            .context("Failed to fetch model file listing from HuggingFace Hub")?;

        let mut filenames: Vec<String> = repo_info
            .siblings
            .iter()
            .map(|s| s.rfilename.clone())
            .collect();
        filenames.sort();

        for fmt in &self.opts.formats {
            match fmt.as_str() {
                "gguf" => {
                    for name in &filenames {
                        if name.ends_with(".gguf") {
                            let path = repo.get(name).await.with_context(|| {
                                format!("Failed to download GGUF weights '{name}'")
                            })?;
                            info!("Found GGUF weights: {}", path.display());
                            return Ok(vec![path]);
                        }
                    }
                }
                "safetensors" => {
                    let shards: Vec<&String> = filenames
                        .iter()
                        .filter(|n| n.ends_with(".safetensors"))
                        .collect();
                    if !shards.is_empty() {
                        let mut paths = Vec::with_capacity(shards.len());
                        for name in shards {
                            let path = repo.get(name).await.with_context(|| {
                                format!("Failed to download safetensors shard '{name}'")
                            })?;
                            paths.push(path);
                        }
                        info!("Found {} safetensors shard(s)", paths.len());
                        return Ok(paths);
                    }
                }
                "bin" => {
                    for candidate in &[
                        "pytorch_model.bin",
                        "pytorch_model.bin.index.json",
                    ] {
                        if filenames.iter().any(|n| n == candidate) {
                            let path = repo.get("pytorch_model.bin").await.with_context(|| {
                                "Failed to download pytorch_model.bin".to_string()
                            })?;
                            info!("Found PyTorch bin weights: {}", path.display());
                            return Ok(vec![path]);
                        }
                    }
                }
                _ => {}
            }
        }

        anyhow::bail!(
            "No supported weight files found. Tried: {:?}",
            self.opts.formats
        )
    }

    fn read_cached_token() -> Option<String> {
        let path = dirs::home_dir()?.join(".cache/huggingface/token");
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_owned())
    }
}
