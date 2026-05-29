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
use sapient_hub::{tokenizer_fallback_model, HubClient, LoadOptions as HubOptions};
use sapient_io::GgufLoader;
use sapient_models::{ForwardEngine, LlmBackendKind};
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
    /// Native LLM backend for Hub generation.
    pub backend: LlmBackendKind,
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
    backend: LlmBackendKind,
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
        let backend = opts.backend;

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

        // GGUF-only repos: the hub's config_path is a sentinel pointing at the
        // GGUF file itself.  Route directly to from_gguf_with_backend instead of
        // trying to parse a config.json that doesn't exist.
        let single_gguf = model_files.weight_paths.len() == 1
            && model_files.weight_paths[0]
                .extension()
                .and_then(|e| e.to_str())
                == Some("gguf");
        if single_gguf {
            return Self::from_gguf_with_backend(&model_files.weight_paths[0], backend).await;
        }

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

        // Prefer the model's own chat template; otherwise fall back to a builtin
        // and remember the stop string(s) that builtin uses to end a turn.
        let mut builtin_stops: Vec<String> = Vec::new();
        let chat_template = match model_files
            .tokenizer_config_path
            .as_ref()
            .and_then(|p| ChatTemplate::from_tokenizer_config(p).ok())
        {
            Some(tmpl) => Some(tmpl),
            None => {
                let (tmpl, stops) =
                    builtin_template_for(&model_info.arch, model_id, &model_info.model_type);
                builtin_stops = stops;
                Some(tmpl)
            }
        };

        validate_tokenizer_model_compat(model_id, &model_info, &tokenizer)?;

        let engine = ForwardEngine::from_weight_paths_with_backend(
            model_info.clone(),
            &model_files.weight_paths,
            backend,
        )
        .context("Failed to initialize inference engine")?;

        let mut config = opts.generation;
        if config.eos_token_id.is_none() {
            config.eos_token_id = tokenizer.eos_id;
        }
        // Register the builtin template's turn-terminator(s) as stop sequences.
        for s in builtin_stops {
            if !config.stop_sequences.contains(&s) {
                config.stop_sequences.push(s);
            }
        }

        debug!(
            "Pipeline ready — vocab_size={} layers={} backend={}",
            model_info.vocab_size, model_info.num_hidden_layers, backend
        );

        Ok(Self {
            tokenizer,
            chat_template,
            model_info,
            weight_paths: model_files.weight_paths.clone(),
            engine: Mutex::new(engine),
            config,
            backend,
        })
    }

    /// Load a GGUF model from a local `.gguf` file.
    ///
    /// Weights are kept quantized in memory (Q4_0/Q8_0 as packed blocks, no F32
    /// expansion) so RAM ≈ file size.  The tokenizer is loaded from the embedded
    /// GGUF vocabulary; if unavailable, a HuggingFace Hub fallback is fetched.
    pub async fn from_gguf(path: impl Into<PathBuf>) -> Result<Self> {
        Self::from_gguf_with_backend(path, LlmBackendKind::Auto).await
    }

    pub async fn from_gguf_with_backend(
        path: impl Into<PathBuf>,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        let path = path.into();
        debug!("Loading GGUF: {}", path.display());

        // Parse metadata + load tensors (quantized types stay quantized).
        let (metadata, _) = GgufLoader::load_tensors_with_metadata(&path)
            .with_context(|| format!("failed to load GGUF: {}", path.display()))?;

        // Build ModelInfo from GGUF KV metadata (no config.json needed).
        let model_info = ModelInfo::from_gguf_metadata(&metadata)
            .context("failed to build ModelInfo from GGUF metadata")?;

        // Build the forward engine directly from the file (it calls
        // load_gguf_hf_weights internally, which keeps quantized types intact).
        let engine = ForwardEngine::from_gguf_with_backend(model_info.clone(), &path, backend)
            .context("failed to initialise ForwardEngine from GGUF")?;

        // Tokenizer: try the model ID from GGUF metadata, else arch-based fallback.
        let model_id = metadata
            .get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let tokenizer = if let Some(fallback) = tokenizer_fallback_model(model_id)
            .or_else(|| tokenizer_fallback_model(model_info.model_type.as_str()))
        {
            Arc::new(
                SapientTokenizer::from_pretrained(fallback)
                    .with_context(|| format!("failed to load tokenizer from '{fallback}'"))?,
            )
        } else {
            anyhow::bail!(
                "Cannot determine tokenizer for GGUF model '{}' (arch: {}). \
                 Load via `Pipeline::from_pretrained` with a registry alias instead.",
                path.display(),
                model_info.model_type
            );
        };

        let (chat_template, builtin_stops) =
            builtin_template_for(&model_info.arch, model_id, &model_info.model_type);

        let mut config = GenerationConfig::default();
        if config.eos_token_id.is_none() {
            config.eos_token_id = tokenizer.eos_id;
        }
        for s in builtin_stops {
            if !config.stop_sequences.contains(&s) {
                config.stop_sequences.push(s);
            }
        }

        validate_tokenizer_model_compat(model_id, &model_info, &tokenizer)?;

        Ok(Self {
            tokenizer,
            chat_template: Some(chat_template),
            model_info,
            weight_paths: vec![path],
            engine: Mutex::new(engine),
            config,
            backend,
        })
    }

    // ── Inference API ─────────────────────────────────────────────────────────

    /// Generate a completion for `prompt`.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self.generate_from_tokens(input_ids).await?;
        let text = self.tokenizer.decode(&output_ids, true)?;
        Ok(self.trim_stop_sequences(text))
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
        let text = self.tokenizer.decode(&output_ids, true)?;
        Ok(self.trim_stop_sequences(text))
    }

    /// All token ids that should terminate generation: the configured EOS plus
    /// every end-of-turn marker the tokenizer knows (e.g. Qwen's `<|im_end|>`,
    /// which `decode` strips, so it can't be caught as a stop *string*).
    fn eos_token_ids(&self) -> Vec<u32> {
        let mut ids = self.tokenizer.eos_ids.clone();
        if let Some(e) = self.config.eos_token_id {
            if !ids.contains(&e) {
                ids.push(e);
            }
        }
        ids
    }

    /// Cut the reply at the first stop sequence (for non-streaming callers).
    fn trim_stop_sequences(&self, text: String) -> String {
        match earliest_stop(&text, &self.config.stop_sequences) {
            Some(idx) => text[..idx].to_string(),
            None => text,
        }
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
        let eos_ids = self.eos_token_ids();
        let max_new = self.config.max_new_tokens;
        let strategy = self.config.strategy.clone();
        let stop = self.config.stop_sequences.clone();
        let tok = Arc::clone(&self.tokenizer);
        let model_info = self.model_info.clone();
        let weight_paths = self.weight_paths.clone();
        let backend = self.configured_backend();

        tokio::task::spawn_blocking(move || {
            let mut engine = match ForwardEngine::from_weight_paths_with_backend(
                model_info,
                &weight_paths,
                backend,
            ) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Error: {e}"));
                    return;
                }
            };
            let mut sampler = Sampler::new(strategy);
            let mut all_tokens = input_ids;
            let mut generated: Vec<u32> = Vec::new();
            // Bytes of the decoded reply already streamed to the caller. We decode
            // the whole reply each step (stable, unlike per-token pieces) and only
            // emit text that cannot be part of a stop marker, so markers like
            // `<|im_end|>` never leak even though they span several tokens.
            let mut emitted = 0usize;
            let mut clean_stop = false;

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

                if eos_ids.contains(&next) {
                    clean_stop = true;
                    break;
                }

                let text = match tok.decode(&generated, true) {
                    Ok(t) => t,
                    Err(_) => continue,
                };

                // A stop sequence appeared: emit everything before it, then stop.
                if let Some(idx) = earliest_stop(&text, &stop) {
                    if idx > emitted {
                        let _ = tx.blocking_send(text[emitted..idx].to_string());
                    }
                    clean_stop = true;
                    break;
                }

                // Emit all but a trailing tail that could still grow into a stop.
                let safe = safe_emit_end(&text, &stop);
                if safe > emitted {
                    if tx.blocking_send(text[emitted..safe].to_string()).is_err() {
                        break;
                    }
                    emitted = safe;
                }
            }

            // Reached max tokens without hitting a stop: flush the held-back tail.
            if !clean_stop {
                if let Ok(text) = tok.decode(&generated, true) {
                    if text.len() > emitted {
                        let _ = tx.blocking_send(text[emitted..].to_string());
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
        let eos_ids = self.eos_token_ids();

        engine.reset_cache();

        // Prefill must use the KV cache so decode steps see correct positions and context.
        let logits = engine.forward_logits(&all_tokens, true)?;
        let mut next = sampler.sample(&logits, &all_tokens)?;
        generated.push(next);
        all_tokens.push(next);

        if eos_ids.contains(&next) {
            return Ok(generated);
        }

        for step in 1..config.max_new_tokens {
            let logits = engine.forward_logits(&[next], true)?;
            next = sampler.sample(&logits, &all_tokens)?;
            generated.push(next);
            all_tokens.push(next);

            if eos_ids.contains(&next) {
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

    fn configured_backend(&self) -> LlmBackendKind {
        self.backend
    }
}

fn ensure_weights_present(files: &ModelFiles) -> Result<()> {
    if files.weight_paths.is_empty() {
        anyhow::bail!("No weight files found for this model");
    }
    Ok(())
}

fn validate_tokenizer_model_compat(
    model_id: &str,
    model_info: &ModelInfo,
    tokenizer: &SapientTokenizer,
) -> Result<()> {
    let tokenizer_vocab = tokenizer.vocab_size();
    if tokenizer_vocab > model_info.vocab_size {
        anyhow::bail!(
            "tokenizer/model vocab mismatch for '{model_id}': tokenizer has {tokenizer_vocab} tokens but model config vocab_size is {}",
            model_info.vocab_size
        );
    }

    if let Some(eos) = tokenizer.eos_id {
        if eos as usize >= model_info.vocab_size {
            anyhow::bail!(
                "tokenizer/model EOS mismatch for '{model_id}': eos_token_id {eos} is outside model vocab_size {}",
                model_info.vocab_size
            );
        }
    } else {
        tracing::warn!(
            model = model_id,
            "tokenizer has no recognized EOS token; generation will stop only by max_new_tokens or stop strings"
        );
    }

    Ok(())
}

/// Byte index of the earliest stop-sequence occurrence in `text`, if any.
fn earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// Largest byte index (a char boundary) up to which `text` is safe to emit
/// without streaming a partial stop marker. Holds back the longest suffix of
/// `text` that is a proper prefix of any stop sequence.
fn safe_emit_end(text: &str, stops: &[String]) -> usize {
    let mut hold = 0usize;
    for s in stops {
        let max_k = s.len().min(text.len());
        for k in (1..max_k).rev() {
            if !s.is_char_boundary(k) {
                continue;
            }
            if text.ends_with(&s[..k]) {
                hold = hold.max(k);
                break;
            }
        }
    }
    text.len() - hold
}

/// Pick a builtin chat template and the stop string(s) that terminate an
/// assistant turn under that template. When we fall back to a builtin template
/// (because the model ships no `chat_template`), the model's plain EOS often
/// isn't what the template trains the turn to end with (e.g. ChatML uses
/// `<|im_end|>`), so these stops must be registered or the marker leaks into
/// the output.
fn builtin_template_for(
    arch: &ArchType,
    model_id: &str,
    model_type: &str,
) -> (ChatTemplate, Vec<String>) {
    let id = model_id.to_ascii_lowercase();
    let mt = model_type.to_ascii_lowercase();
    let chatml = || {
        (
            ChatTemplate::from_template(builtin::CHATML),
            vec!["<|im_end|>".to_string()],
        )
    };
    match arch {
        ArchType::Llama if id.contains("tinyllama") => (
            ChatTemplate::from_template(builtin::ZEPHYR),
            vec!["</s>".to_string()],
        ),
        ArchType::Llama
            if id.contains("llama-2")
                || id.contains("llama2")
                || (mt.contains("llama") && !id.contains("llama-3") && !id.contains("llama3")) =>
        {
            (
                ChatTemplate::from_template(builtin::LLAMA2),
                vec!["</s>".to_string()],
            )
        }
        ArchType::Llama => (
            ChatTemplate::from_template(builtin::LLAMA3),
            vec!["<|eot_id|>".to_string()],
        ),
        ArchType::Gemma => (
            ChatTemplate::from_template(builtin::GEMMA),
            vec!["<end_of_turn>".to_string()],
        ),
        ArchType::Phi | ArchType::Qwen => chatml(),
        _ => chatml(),
    }
}
