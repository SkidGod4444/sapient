//! `TranscribePipeline` — speech-to-text via Whisper, parallel to [`crate::Pipeline`].
//!
//! Loads a Whisper checkpoint (HF safetensors + `config.json` + `tokenizer.json`),
//! builds the mel front-end ([`sapient_audio`]) and the [`AudioEngine`], then
//! drives: decode audio → 16 kHz mono → 30 s chunks → log-mel → encoder →
//! cached cross-attention → forced-prompt greedy decode → text.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use sapient_audio::{load_audio, MelConfig, MelFrontend};
use sapient_core::Tensor;
use sapient_hub::client::LoadOptions;
use sapient_hub::whisper_config::WhisperConfig;
use sapient_hub::HubClient;
use sapient_models::forward::{AudioEngine, WhisperForward};
use sapient_models::{weights, LlmBackendKind};
use sapient_tokenizers::WhisperTokenizer;
use tracing::debug;

const SAMPLE_RATE: u32 = 16_000;

/// Build the audio engine for the chosen backend. `--backend wgpu` (when the
/// `wgpu` feature is compiled in) runs the transformer body on the GPU from raw
/// f32 weights; every other backend uses the CPU/Metal `WhisperForward` (which
/// online-quantizes its linears to Q8_0).
fn build_audio_engine(
    cfg: WhisperConfig,
    weights: HashMap<String, Tensor>,
    backend: LlmBackendKind,
) -> Result<AudioEngine> {
    #[cfg(feature = "wgpu")]
    if matches!(backend, LlmBackendKind::Wgpu) {
        use sapient_models::forward::WhisperWgpuEngine;
        return Ok(AudioEngine::WhisperWgpu(Box::new(
            WhisperWgpuEngine::from_weights(cfg, weights)?,
        )));
    }
    Ok(AudioEngine::Whisper(Box::new(
        WhisperForward::from_weights_with_backend(cfg, weights, backend)?,
    )))
}

/// Per-call transcription options.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Source language code (e.g. `"en"`). `None` → auto-detect per chunk.
    pub language: Option<String>,
    /// Translate to English (`<|translate|>`) instead of transcribing.
    pub translate: bool,
    /// Emit timestamp tokens (Phase 1 leaves this off and strips them).
    pub timestamps: bool,
    /// Maximum new tokens decoded per 30 s chunk.
    pub max_new_tokens: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            timestamps: false,
            max_new_tokens: 224,
        }
    }
}

/// A loaded Whisper speech-to-text pipeline.
pub struct TranscribePipeline {
    engine: Arc<Mutex<AudioEngine>>,
    tokenizer: Arc<WhisperTokenizer>,
    mel: MelFrontend,
    cfg: WhisperConfig,
    backend: LlmBackendKind,
}

impl TranscribePipeline {
    /// Load a Whisper model by alias/repo id (CPU backend).
    pub async fn from_pretrained(model_id: &str) -> Result<Self> {
        Self::from_pretrained_with_backend(model_id, LlmBackendKind::Cpu).await
    }

    /// Load a Whisper model with a chosen backend.
    pub async fn from_pretrained_with_backend(
        model_id: &str,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        debug!("Loading Whisper model: {model_id}");

        let hub_opts = LoadOptions {
            formats: vec!["safetensors".into(), "bin".into()],
            ..LoadOptions::default()
        };
        let hub = HubClient::with_options(hub_opts)?;
        let files = hub
            .download(model_id)
            .await
            .with_context(|| format!("downloading Whisper model '{model_id}'"))?;

        let cfg = WhisperConfig::from_config_file(&files.config_path)
            .context("parsing Whisper config.json")?;

        let tokenizer = match &files.tokenizer_path {
            Some(p) => WhisperTokenizer::from_file(p)?,
            None => WhisperTokenizer::from_pretrained(model_id)?,
        };

        let raw =
            weights::load_hf_weights(&files.weight_paths).context("loading Whisper weights")?;
        let engine = build_audio_engine(cfg.clone(), raw, backend)?;
        let mel = MelFrontend::new(MelConfig::with_n_mels(cfg.num_mel_bins));

        debug!(
            "Whisper ready — d_model={} enc_layers={} dec_layers={} vocab={} n_mels={}",
            cfg.d_model, cfg.encoder_layers, cfg.decoder_layers, cfg.vocab_size, cfg.num_mel_bins
        );

        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            tokenizer: Arc::new(tokenizer),
            mel,
            cfg,
            backend,
        })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    pub fn backend(&self) -> LlmBackendKind {
        self.backend
    }

    /// Transcribe an audio file with default options.
    pub async fn transcribe(&self, path: impl AsRef<Path>) -> Result<String> {
        self.transcribe_with(path, TranscribeOptions::default())
            .await
    }

    /// Transcribe an audio file with explicit options.
    pub async fn transcribe_with(
        &self,
        path: impl AsRef<Path>,
        opts: TranscribeOptions,
    ) -> Result<String> {
        let samples = load_audio(path, SAMPLE_RATE).context("decoding audio")?;
        // Inference is CPU-bound and synchronous; keep it off the async reactor.
        tokio::task::block_in_place(|| self.transcribe_samples(&samples, &opts))
    }

    /// Transcribe already-decoded mono 16 kHz samples (synchronous).
    pub fn transcribe_samples(&self, samples: &[f32], opts: &TranscribeOptions) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| anyhow!("audio engine mutex poisoned"))?;

        let chunk_len = self.mel.config().n_samples();
        let mut out = String::new();
        let mut start = 0usize;
        while start < samples.len() {
            let end = (start + chunk_len).min(samples.len());
            let text = transcribe_chunk(
                &mut engine,
                &self.tokenizer,
                &self.mel,
                &samples[start..end],
                opts,
                &self.cfg,
            )?;
            let text = text.trim();
            if !text.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(text);
            }
            start += chunk_len;
        }
        Ok(out)
    }
}

/// Transcribe one ≤30 s chunk: log-mel → encode → forced-prompt greedy decode.
fn transcribe_chunk(
    engine: &mut AudioEngine,
    tok: &WhisperTokenizer,
    mel: &MelFrontend,
    chunk: &[f32],
    opts: &TranscribeOptions,
    cfg: &WhisperConfig,
) -> Result<String> {
    let mel_t = mel.log_mel(chunk)?;
    engine.encode(&mel_t)?; // runs encoder + caches cross-attention K/V

    // Language: explicit, or detect from the first decode step after <|sot|>.
    let lang: Option<String> = match &opts.language {
        Some(l) => Some(l.clone()),
        None => {
            engine.reset_decoder();
            let logits = engine.decode_step(&[tok.sot])?;
            let lang_ids = tok.language_token_ids();
            argmax_restricted(&logits, &lang_ids)
                .and_then(|id| tok.language_code(id).map(str::to_string))
        }
    };

    // Forced prompt: [<|sot|>, <|lang|>, <|task|>, <|notimestamps|>].
    engine.reset_decoder();
    let prompt = tok.sot_sequence(lang.as_deref(), opts.translate, opts.timestamps);
    let mut logits = engine.decode_step(&prompt)?;

    let budget = opts
        .max_new_tokens
        .min(cfg.max_target_positions.saturating_sub(prompt.len()).max(1));
    let mut out_tokens = Vec::new();
    for _ in 0..budget {
        let next = argmax(&logits);
        if tok.is_eot(next) {
            break;
        }
        // Keep only emitted text tokens; drop control/timestamp tokens but still
        // feed them back so positions advance correctly.
        if !tok.is_special(next) {
            out_tokens.push(next);
        }
        logits = engine.decode_step(&[next])?;
    }

    tok.decode(&out_tokens, true)
}

/// Index of the maximum logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Argmax restricted to a candidate id set (for language detection).
fn argmax_restricted(logits: &[f32], ids: &[u32]) -> Option<u32> {
    ids.iter()
        .copied()
        .filter(|&id| (id as usize) < logits.len())
        .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]))
}
