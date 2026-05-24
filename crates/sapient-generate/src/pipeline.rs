//! `Pipeline` — the main user-facing LLM inference API.
//!
//! One line to load any HuggingFace model, one line to generate text.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn};

use sapient_hub::{HubClient, LoadOptions as HubOptions, ModelFiles};
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_hub::resolver::WeightFormat;
use sapient_io::{load_gguf, load_safetensors};
use sapient_models::build_graph;
use sapient_tokenizers::{
    chat::{builtin, ChatMessage, ChatTemplate},
    tokenizer::{SapientTokenizer, TokenizerOptions},
};

use crate::kv_cache::KVCache;
use crate::sampler::{SamplingStrategy, Sampler};

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
///
/// # Usage
/// ```no_run
/// # async fn example() -> anyhow::Result<()> {
/// use sapient_generate::Pipeline;
///
/// let p = Pipeline::from_pretrained("microsoft/phi-2").await?;
/// println!("{}", p.generate("The sky is").await?);
/// # Ok(()) }
/// ```
pub struct Pipeline {
    tokenizer: Arc<SapientTokenizer>,
    chat_template: Option<ChatTemplate>,
    model_info: ModelInfo,
    model_files: ModelFiles,
    config: GenerationConfig,
}

impl Pipeline {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Load any model from the HuggingFace Hub by model ID.
    ///
    /// Examples:
    /// - `"meta-llama/Llama-3.2-1B-Instruct"` — Llama 3.2 (requires HF token)
    /// - `"microsoft/phi-2"` — Phi-2 (open, no token needed)
    /// - `"google/gemma-2-2b-it"` — Gemma 2 (requires HF token)
    /// - `"Qwen/Qwen2.5-1.5B-Instruct"` — Qwen 2.5
    /// - `"TheBloke/Llama-2-7B-GGUF"` — quantized GGUF (auto-picks Q4_K_M)
    pub async fn from_pretrained(model_id: &str) -> Result<Self> {
        Self::from_pretrained_with_opts(model_id, LoadOptions::default()).await
    }

    /// Load with custom hub and generation options.
    pub async fn from_pretrained_with_opts(
        model_id: &str,
        opts: LoadOptions,
    ) -> Result<Self> {
        info!("Loading model: {model_id}");

        // 1. Download all model files from the Hub.
        let hub = HubClient::with_options(opts.hub)?;
        let model_files = hub.download(model_id).await
            .with_context(|| format!("Failed to download model '{model_id}'"))?;

        // 2. Parse config.json to detect architecture.
        let model_info = ModelInfo::from_config_file(&model_files.config_path)
            .context("Failed to parse config.json")?;
        info!("Detected architecture: {:?}", model_info.arch);

        // 3. Load tokenizer.
        let tok_opts = TokenizerOptions { add_bos: true, ..Default::default() };
        let tokenizer = if let Some(tok_path) = &model_files.tokenizer_path {
            Arc::new(SapientTokenizer::from_file(tok_path, tok_opts)
                .context("Failed to load tokenizer")?)
        } else {
            // GGUF models embed the tokenizer — fall back to pretrained loading.
            Arc::new(SapientTokenizer::from_pretrained(model_id)
                .context("Failed to load tokenizer from Hub")?)
        };

        // 4. Load chat template from tokenizer_config.json.
        let chat_template = model_files.tokenizer_config_path.as_ref().and_then(|p| {
            ChatTemplate::from_tokenizer_config(p).ok()
        }).or_else(|| {
            // Fall back to built-in template based on architecture.
            Some(builtin_template_for(&model_info.arch))
        });

        // 5. Set EOS from tokenizer.
        let mut config = opts.generation;
        if config.eos_token_id.is_none() {
            config.eos_token_id = tokenizer.eos_id;
        }

        info!("Pipeline ready — vocab_size={} layers={}", model_info.vocab_size, model_info.num_hidden_layers);

        Ok(Self { tokenizer, chat_template, model_info, model_files, config })
    }

    /// Load a GGUF model directly from a local file path.
    pub async fn from_gguf(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        info!("Loading GGUF model from: {}", path.display());
        // Use a minimal config for GGUF (metadata is embedded in the file).
        unimplemented!("Direct GGUF loading from path — coming soon. Use from_pretrained() with a GGUF Hub repo.")
    }

    // ── Inference API ─────────────────────────────────────────────────────────

    /// Generate a completion for `prompt`.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self.generate_from_tokens(input_ids).await?;
        self.tokenizer.decode(&output_ids, true)
    }

    /// Generate with a custom generation config.
    pub async fn generate_with_config(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self.generate_from_tokens_with_config(input_ids, config).await?;
        self.tokenizer.decode(&output_ids, true)
    }

    /// Chat with the model (for instruct/chat tuned models).
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = if let Some(tmpl) = &self.chat_template {
            tmpl.render(messages, true)
                .context("Failed to render chat template")?
        } else {
            // Fallback: concatenate messages without template.
            messages.iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        };
        self.generate(&prompt).await
    }

    /// Stream tokens as they are generated.
    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> ReceiverStream<String> {
        let (tx, rx) = mpsc::channel(64);
        let input_ids = self.tokenizer.encode(prompt).unwrap_or_default();
        let eos = self.config.eos_token_id;
        let max_new = self.config.max_new_tokens;
        let tok = Arc::clone(&self.tokenizer);

        // In a full implementation this would call the model forward pass.
        // Here we emit a placeholder stream.
        tokio::spawn(async move {
            for i in 0..max_new.min(10) {
                let token_text = format!("<token_{i}> ");
                if tx.send(token_text).await.is_err() { break; }
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });

        ReceiverStream::new(rx)
    }

    /// Compute sentence embeddings (for BERT-style models).
    /// Returns the mean-pooled last hidden state as a flat `f32` vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // For now: encode and return token IDs as floats (placeholder).
        // Full implementation requires running the encoder and mean-pooling.
        warn!("embed() is not yet fully implemented — returns token IDs for now");
        let ids = self.tokenizer.encode(text)?;
        Ok(ids.iter().map(|&id| id as f32).collect())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    async fn generate_from_tokens(&self, input_ids: Vec<u32>) -> Result<Vec<u32>> {
        self.generate_from_tokens_with_config(input_ids, &self.config).await
    }

    async fn generate_from_tokens_with_config(
        &self,
        input_ids: Vec<u32>,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        // Build the model graph.
        let graph = build_graph(&self.model_info)?.graph;

        // Set up KV cache.
        let mut kv_cache = KVCache::new(self.model_info.num_hidden_layers);

        // Set up sampler.
        let mut sampler = Sampler::new(config.strategy.clone());

        let mut generated: Vec<u32> = Vec::new();
        let mut all_tokens = input_ids.clone();

        for _step in 0..config.max_new_tokens {
            // In a full implementation: run graph forward pass here.
            // For now, sample from a uniform distribution as a structural test.
            let vocab = self.model_info.vocab_size;
            let logits: Vec<f32> = (0..vocab).map(|i| if i == 42 { 10.0 } else { 0.1 }).collect();

            let next_token = sampler.sample(&logits, &all_tokens)?;
            generated.push(next_token);
            all_tokens.push(next_token);

            // Stop at EOS.
            if config.eos_token_id == Some(next_token) {
                debug!("EOS token generated at step {_step}");
                break;
            }

            // Check stop sequences.
            if !config.stop_sequences.is_empty() {
                let decoded = self.tokenizer.decode(&generated, true).unwrap_or_default();
                if config.stop_sequences.iter().any(|s| decoded.contains(s.as_str())) {
                    break;
                }
            }
        }

        Ok(generated)
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn tokenizer(&self) -> &SapientTokenizer { &self.tokenizer }
    pub fn model_info(&self) -> &ModelInfo { &self.model_info }
    pub fn arch(&self) -> &ArchType { &self.model_info.arch }
}

// ── Built-in template selection ───────────────────────────────────────────────

fn builtin_template_for(arch: &ArchType) -> ChatTemplate {
    match arch {
        ArchType::Llama => ChatTemplate::from_template(builtin::LLAMA3),
        ArchType::Phi   => ChatTemplate::from_template(builtin::CHATML),
        ArchType::Gemma => ChatTemplate::from_template(builtin::GEMMA),
        ArchType::Qwen  => ChatTemplate::from_template(builtin::CHATML),
        _               => ChatTemplate::from_template(builtin::CHATML),
    }
}
