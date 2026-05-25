//! HuggingFace Hub API client.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use tokio::sync::Semaphore;
use tracing::debug;

use crate::download::{configure_api_builder, max_parallel_downloads};
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

    /// Hide HuggingFace Hub progress bars (filenames, byte counts on stderr).
    pub quiet: bool,

    /// Parallel HTTP range downloads + concurrent shard fetches (recommended).
    ///
    /// Disable with `SAPIENT_FAST_DOWNLOAD=0`. Tune workers with
    /// `SAPIENT_HUB_MAX_PARALLEL` (default: min(CPU cores, 8)).
    pub fast_download: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            token: None,
            formats: vec!["gguf".into(), "safetensors".into(), "bin".into()],
            force_download: false,
            max_size_bytes: 0,
            revision: "main".into(),
            quiet: false,
            fast_download: true,
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

        let builder = ApiBuilder::new();
        let builder = if let Some(t) = token {
            builder.with_token(Some(t))
        } else {
            builder
        };
        let builder = configure_api_builder(builder, &opts);
        let api = builder
            .build()
            .context("Failed to build HF Hub API client")?;
        Ok(Self { api, opts })
    }

    /// Download a model by its HuggingFace model ID (e.g. `"meta-llama/Llama-3.2-1B"`).
    ///
    /// Returns the resolved local file paths — no Python, no git-lfs required.
    pub async fn download(&self, model_id: &str) -> Result<ModelFiles> {
        debug!("Downloading model: {model_id}");
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
        let (config_path, tokenizer_path, tokenizer_config_path, weight_paths) = tokio::join!(
            async {
                repo.get("config.json")
                    .await
                    .context("config.json not found — is this a valid model repo?")
            },
            async { repo.get("tokenizer.json").await.ok() },
            async { repo.get("tokenizer_config.json").await.ok() },
            self.fetch_weights(repo, model_id),
        );

        let config_path = config_path?;
        debug!("config.json cached for {model_id}");

        Ok(ModelFiles {
            model_id: model_id.to_owned(),
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            weight_paths: weight_paths?,
        })
    }

    async fn fetch_weights(&self, repo: &ApiRepo, model_id: &str) -> Result<Vec<PathBuf>> {
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
                            debug!("Found GGUF weights for model");
                            return Ok(vec![path]);
                        }
                    }
                }
                "safetensors" => {
                    let shards: Vec<String> = filenames
                        .iter()
                        .filter(|n| n.ends_with(".safetensors"))
                        .cloned()
                        .collect();
                    if !shards.is_empty() {
                        let paths = if shards.len() > 1 && self.opts.fast_download {
                            self.download_files_parallel(model_id, &shards).await?
                        } else {
                            self.download_files_sequential(repo, &shards).await?
                        };
                        debug!("Found {} safetensors shard(s)", paths.len());
                        return Ok(paths);
                    }
                }
                "bin" => {
                    for candidate in &["pytorch_model.bin", "pytorch_model.bin.index.json"] {
                        if filenames.iter().any(|n| n == candidate) {
                            let path = repo.get("pytorch_model.bin").await.with_context(|| {
                                "Failed to download pytorch_model.bin".to_string()
                            })?;
                            debug!("Found PyTorch bin weights for model");
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

    async fn download_files_sequential(
        &self,
        repo: &ApiRepo,
        names: &[String],
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::with_capacity(names.len());
        for name in names {
            let path = repo
                .get(name)
                .await
                .with_context(|| format!("Failed to download '{name}'"))?;
            paths.push(path);
        }
        Ok(paths)
    }

    async fn download_files_parallel(
        &self,
        model_id: &str,
        names: &[String],
    ) -> Result<Vec<PathBuf>> {
        let workers = max_parallel_downloads();
        let semaphore = Arc::new(Semaphore::new(workers));
        let mut handles = Vec::with_capacity(names.len());

        for name in names {
            let api = self.api.clone();
            let model_id = model_id.to_owned();
            let name = name.clone();
            let sem = semaphore.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|e| anyhow::anyhow!("download worker failed: {e}"))?;
                api.model(model_id)
                    .get(&name)
                    .await
                    .with_context(|| format!("Failed to download '{name}'"))
            }));
        }

        let mut paths = Vec::with_capacity(handles.len());
        for handle in handles {
            paths.push(handle.await.context("parallel download task panicked")??);
        }
        paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        Ok(paths)
    }

    fn read_cached_token() -> Option<String> {
        let path = dirs::home_dir()?.join(".cache/huggingface/token");
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_owned())
    }
}
