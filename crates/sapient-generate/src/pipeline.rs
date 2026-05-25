//! `Pipeline` — the main user-facing LLM inference API.
//!
//! One line to load any HuggingFace model, one line to generate text.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_hub::resolver::ModelFiles;
use sapient_hub::{HubClient, LoadOptions as HubOptions, tokenizer_fallback_model};
use sapient_models::ForwardEngine;
use sapient_tokenizers::{
    chat::{builtin, ChatMessage, ChatTemplate},
    tokenizer::{SapientTokenizer, TokenizerOptions},
};

use crate::sampler::{Sampler, SamplingStrategy};

// ── GenerationConfig ──────────────────────────────────────────────────────────

/// Controls how text is generated.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Stop generating when this token ID is produced (usually EOS).
    pub eos_token_id: Option<u32>,
    /// Sampling strategy (default: greedy).
    pub strategy: SamplingStrategy,
    /// Stop strings — generation ends if any of these appear in output.
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            eos_token_id: None,
            strategy: SamplingStrategy::default(),
            stop_sequences: vec![],
        }
    }
}

// ── LoadOptions ───────────────────────────────────────────────────────────────

/// Options for loading a model from HuggingFace Hub or local disk.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// HuggingFace Hub options.
    pub hub: HubOptions,
    /// Override the generation config.
    pub generation: GenerationConfig,
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// A fully loaded LLM ready for inference.
pub struct Pipeline {
    tokenizer: Arc<SapientTokenizer>,
    chat_template: Option<ChatTemplate>,
    model_info: ModelInfo,
    weight_paths: Vec<PathBuf>,
    engine: Mutex<ForwardEngine>,
    config: GenerationConfig,
}

impl Pipeline {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Load any model from the HuggingFace Hub by model ID.
    pub async fn from_pretrained(model_id: &str) -> Result<Self> {
        Self::from_pretrained_with_opts(model_id, LoadOptions::default()).await
    }

    /// Load with custom hub and generation options.
    pub async fn from_pretrained_with_opts(model_id: &str, opts: LoadOptions) -> Result<Self> {
        debug!("Loading model: {model_id}");

        let mut hub_opts = opts.hub.clone();
        if hub_opts.formats == LoadOptions::default().hub.formats {
            // Prefer full-precision safetensors for native forward passes.
            hub_opts.formats = vec!["safetensors".into(), "bin".into(), "gguf".into()];
        }

        let hub = HubClient::with_options(hub_opts)?;
        let model_files = hub
            .download(model_id)
            .await
            .with_context(|| format!("Failed to download model '{model_id}'"))?;

        ensure_weights_present(&model_files)?;

        let model_info = ModelInfo::from_config_file(&model_files.config_path)
            .context("Failed to parse config.json")?;
        debug!("Detected architecture: {:?}", model_info.arch);

        if model_info.raw.get("vision_config").is_some() {
            debug!("Vision tower present — text-only mode (images not supported yet)");
        }

        let tok_opts = TokenizerOptions {
            add_bos: true,
            ..Default::default()
        };
        let tokenizer = if let Some(tok_path) = &model_files.tokenizer_path {
            Arc::new(
                SapientTokenizer::from_file(tok_path, tok_opts)
                    .context("Failed to load tokenizer")?,
            )
        } else if let Some(fallback_id) = tokenizer_fallback_model(model_id) {
            debug!("No local tokenizer — loading from fallback Hub model '{fallback_id}'");
            Arc::new(
                SapientTokenizer::from_pretrained(fallback_id).with_context(|| {
                    format!(
                        "Failed to load tokenizer from fallback model '{fallback_id}' \
                         (GGUF repos often omit tokenizer files)"
                    )
                })?,
            )
        } else {
            Arc::new(
                SapientTokenizer::from_pretrained(model_id)
                    .context("Failed to load tokenizer from Hub")?,
            )
        };

        let chat_template = model_files
            .tokenizer_config_path
            .as_ref()
            .and_then(|p| ChatTemplate::from_tokenizer_config(p).ok())
            .or_else(|| Some(builtin_template_for(&model_info.arch, model_id, &model_info.model_type)));

        let engine = ForwardEngine::from_weight_paths(model_info.clone(), &model_files.weight_paths)
            .context("Failed to initialize inference engine")?;

        let mut config = opts.generation;
        if config.eos_token_id.is_none() {
            config.eos_token_id = tokenizer.eos_id;
        }

        debug!(
            "Pipeline ready — vocab_size={} layers={}",
            model_info.vocab_size, model_info.num_hidden_layers
        );

        Ok(Self {
            tokenizer,
            chat_template,
            model_info,
            weight_paths: model_files.weight_paths.clone(),
            engine: Mutex::new(engine),
            config,
        })
    }

    /// Load a GGUF model directly from a local file path.
    pub async fn from_gguf(_path: impl Into<PathBuf>) -> Result<Self> {
        anyhow::bail!(
            "Direct GGUF loading is not yet supported. \
             Download a safetensors model from HuggingFace Hub instead."
        )
    }

    // ── Inference API ─────────────────────────────────────────────────────────

    /// Generate a completion for `prompt`.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self.generate_from_tokens(input_ids).await?;
        self.tokenizer.decode(&output_ids, true)
    }

    /// Generate with a custom generation config.
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self
            .generate_from_tokens_with_config(input_ids, config)
            .await?;
        self.tokenizer.decode(&output_ids, true)
    }

    /// Render the chat prompt string for a message history.
    pub fn format_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String> {
        if let Some(tmpl) = &self.chat_template {
            tmpl.render(messages, true)
                .context("Failed to render chat template")
        } else {
            Ok(messages
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n"))
        }
    }

    /// Chat with the model (for instruct/chat tuned models).
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.format_chat_prompt(messages)?;
        self.generate(&prompt).await
    }

    /// Stream a chat reply token-by-token.
    pub async fn chat_stream(&self, messages: &[ChatMessage]) -> ReceiverStream<String> {
        match self.format_chat_prompt(messages) {
            Ok(prompt) => self.generate_stream(&prompt).await,
            Err(e) => {
                let (tx, rx) = mpsc::channel(1);
                let _ = tx.try_send(format!("Error: {e}"));
                ReceiverStream::new(rx)
            }
        }
    }

    /// Stream tokens as they are generated.
    pub async fn generate_stream(&self, prompt: &str) -> ReceiverStream<String> {
        let (tx, rx) = mpsc::channel(64);
        let input_ids = self.tokenizer.encode(prompt).unwrap_or_default();
        let eos = self.config.eos_token_id;
        let max_new = self.config.max_new_tokens;
        let strategy = self.config.strategy.clone();
        let stop = self.config.stop_sequences.clone();
        let tok = Arc::clone(&self.tokenizer);
        let model_info = self.model_info.clone();
        let weight_paths = self.weight_paths.clone();

        tokio::task::spawn_blocking(move || {
            let mut engine = match ForwardEngine::from_weight_paths(model_info, &weight_paths) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Error: {e}"));
                    return;
                }
            };
            let mut sampler = Sampler::new(strategy);
            let mut all_tokens = input_ids;
            let mut generated: Vec<u32> = Vec::new();

            engine.reset_cache();
            for step in 0..max_new {
                let chunk = if step == 0 {
                    all_tokens.clone()
                } else {
                    vec![*all_tokens.last().unwrap()]
                };
                let logits = match engine.forward_logits(&chunk, true) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.blocking_send(format!("Error: {e}"));
                        break;
                    }
                };

                let next = match sampler.sample(&logits, &all_tokens) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(format!("Error: {e}"));
                        break;
                    }
                };

                generated.push(next);
                all_tokens.push(next);

                if let Ok(piece) = tok.decode_token(next) {
                    if tx.blocking_send(piece).is_err() {
                        break;
                    }
                }

                if eos == Some(next) {
                    break;
                }

                if !stop.is_empty() {
                    if let Ok(text) = tok.decode(&generated, true) {
                        if stop.iter().any(|s| text.contains(s.as_str())) {
                            break;
                        }
                    }
                }
            }
        });

        ReceiverStream::new(rx)
    }

    /// Compute sentence embeddings via mean-pooled hidden states.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode(text)?;
        let mut engine = self.engine.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        engine.embed(&ids)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    async fn generate_from_tokens(&self, input_ids: Vec<u32>) -> Result<Vec<u32>> {
        self.generate_from_tokens_with_config(input_ids, &self.config)
            .await
    }

    async fn generate_from_tokens_with_config(
        &self,
        input_ids: Vec<u32>,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let mut engine = self.engine.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut sampler = Sampler::new(config.strategy.clone());
        let mut generated: Vec<u32> = Vec::new();
        let mut all_tokens = input_ids;

        engine.reset_cache();

        // Prefill must use the KV cache so decode steps see correct positions and context.
        let logits = engine.forward_logits(&all_tokens, true)?;
        let mut next = sampler.sample(&logits, &all_tokens)?;
        generated.push(next);
        all_tokens.push(next);

        if config.eos_token_id == Some(next) {
            return Ok(generated);
        }

        for step in 1..config.max_new_tokens {
            let logits = engine.forward_logits(&[next], true)?;
            next = sampler.sample(&logits, &all_tokens)?;
            generated.push(next);
            all_tokens.push(next);

            if config.eos_token_id == Some(next) {
                debug!("EOS token generated at step {step}");
                break;
            }

            if !config.stop_sequences.is_empty() {
                let decoded = self.tokenizer.decode(&generated, true).unwrap_or_default();
                if config
                    .stop_sequences
                    .iter()
                    .any(|s| decoded.contains(s.as_str()))
                {
                    break;
                }
            }
        }

        Ok(generated)
    }

    pub fn tokenizer(&self) -> &SapientTokenizer {
        &self.tokenizer
    }
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }
    pub fn arch(&self) -> &ArchType {
        &self.model_info.arch
    }
}

fn ensure_weights_present(files: &ModelFiles) -> Result<()> {
    if files.weight_paths.is_empty() {
        anyhow::bail!("No weight files found for this model");
    }
    Ok(())
}

fn builtin_template_for(arch: &ArchType, model_id: &str, model_type: &str) -> ChatTemplate {
    let id = model_id.to_ascii_lowercase();
    let mt = model_type.to_ascii_lowercase();
    match arch {
        ArchType::Llama if id.contains("tinyllama") => {
            ChatTemplate::from_template(builtin::ZEPHYR)
        }
        ArchType::Llama if id.contains("llama-2")
            || id.contains("llama2")
            || (mt.contains("llama") && !id.contains("llama-3") && !id.contains("llama3")) =>
        {
            ChatTemplate::from_template(builtin::LLAMA2)
        }
        ArchType::Llama => ChatTemplate::from_template(builtin::LLAMA3),
        ArchType::Phi => ChatTemplate::from_template(builtin::CHATML),
        ArchType::Gemma => ChatTemplate::from_template(builtin::GEMMA),
        ArchType::Qwen => ChatTemplate::from_template(builtin::CHATML),
        _ => ChatTemplate::from_template(builtin::CHATML),
    }
}
