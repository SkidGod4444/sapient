//! HuggingFace Hub API client.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use tokio::sync::Semaphore;
use tracing::debug;

use crate::download::{configure_api_builder, max_parallel_downloads};
use crate::gguf::select_best_gguf;
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
    pub async fn download(&self, model_alias: &str) -> Result<ModelFiles> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias)?;
        debug!("Downloading model: {model_alias} (resolved to {actual_repo})");
        let repo = self.api.model(actual_repo.clone());
        let mut files = match self.resolve_files(&repo, &actual_repo).await {
            Ok(f) => f,
            Err(e) => {
                if self.opts.fast_download {
                    eprintln!("Warning: Fast download failed ({}). Retrying in safe mode...", e);
                    let mut safe_opts = self.opts.clone();
                    safe_opts.fast_download = false;
                    let safe_client = Self::with_options(safe_opts)?;
                    let safe_repo = safe_client.api.model(actual_repo.clone());
                    safe_client.resolve_files(&safe_repo, &actual_repo).await?
                } else {
                    return Err(e);
                }
            }
        };
        files.model_id = model_alias.to_owned();
        Ok(files)
    }

    /// Fetch model info / architecture type from the Hub (reads `config.json`).
    pub async fn model_info(&self, model_alias: &str) -> Result<ModelInfo> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias)?;
        let repo = self.api.model(actual_repo);
        let config_path = repo
            .get("config.json")
            .await
            .context("Failed to fetch config.json")?;
        ModelInfo::from_config_file(&config_path)
    }

    /// Returns the total download size (in bytes) for all files in the model repo,
    /// by querying the HuggingFace REST API for file metadata.
    pub async fn repo_total_bytes(&self, model_alias: &str) -> Result<u64> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias)?;
        // Use the HF REST API which returns sibling sizes
        let url = format!("https://huggingface.co/api/models/{actual_repo}");
        let client = reqwest::Client::new();
        let mut req = client.get(&url);
        // Forward auth token if available
        let token = self.opts.token.clone()
            .or_else(|| std::env::var("HF_TOKEN").ok());
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp: serde_json::Value = req
            .send()
            .await
            .context("Failed to query HF API for model metadata")?
            .json()
            .await
            .context("Failed to parse HF API response")?;
        let total = resp["siblings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s["size"].as_u64())
            .sum();
        Ok(total)
    }

    /// Returns the on-disk blobs directory for a HuggingFace model, used to poll download progress.
    /// The path follows HF hub cache conventions: `~/.cache/huggingface/hub/models--<org>--<name>/blobs/`.
    pub fn blobs_dir_for_model(model_alias: &str) -> Option<std::path::PathBuf> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias).ok()?;
        let cache_root = dirs::home_dir()?.join(".cache/huggingface/hub");
        let dir_name = format!("models--{}", actual_repo.replace('/', "--"));
        Some(cache_root.join(dir_name).join("blobs"))
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
                    if let Some(name) = select_best_gguf(&filenames) {
                        let path = repo
                            .get(name)
                            .await
                            .with_context(|| format!("Failed to download GGUF weights '{name}'"))?;
                        debug!("Found GGUF weights: {}", path.display());
                        return Ok(vec![path]);
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
            let mut retries = 5;
            let mut backoff = 2;
            let path = loop {
                match repo.get(name).await {
                    Ok(p) => break p,
                    Err(e) if retries > 0 => {
                        debug!("Retry downloading '{}' ({} retries left) due to: {}", name, retries, e);
                        retries -= 1;
                        tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                        backoff = std::cmp::min(backoff * 2, 10);
                    }
                    Err(e) => return Err(anyhow::anyhow!("Failed to download '{}': {}", name, e)),
                }
            };
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
                
                let mut retries = 5;
                let mut backoff = 2;
                loop {
                    match api.model(model_id.clone()).get(&name).await {
                        Ok(p) => return Ok(p),
                        Err(e) if retries > 0 => {
                            debug!("Retry parallel download '{}' ({} retries left) due to: {}", name, retries, e);
                            retries -= 1;
                            tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                            backoff = std::cmp::min(backoff * 2, 10);
                        }
                        Err(e) => return Err(anyhow::anyhow!("Failed to download '{}': {}", name, e)),
                    }
                }
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
