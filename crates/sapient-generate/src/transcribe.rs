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
use sapient_hub::whisper_config::{WhisperConfig, WhisperGenConfig};
use sapient_hub::HubClient;
use sapient_models::forward::{AudioEngine, WhisperForward};
use sapient_models::{weights, LlmBackendKind};
use sapient_tokenizers::WhisperTokenizer;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

const SAMPLE_RATE: u32 = 16_000;
/// Seconds of audio per timestamp token: `input_stride(2) * hop(160) / sr(16k)`.
const TIME_PRECISION: f32 = 0.02;

/// Seconds represented by a timestamp token id.
fn timestamp_seconds(timestamp_begin: u32, id: u32) -> f32 {
    id.saturating_sub(timestamp_begin) as f32 * TIME_PRECISION
}

/// Whisper's ApplyTimestampRules, applied to `logits` before argmax in the
/// timestamped decode path. `generated` is the tokens decoded *after* the forced
/// prompt (text + timestamp tokens). Timestamp ids are `>= timestamp_begin`.
///
/// Rules (a subset of openai-whisper's, sufficient for greedy long-form):
/// - (a) at the first position, force a timestamp (suppress all non-timestamps);
/// - (b) after a lone timestamp, the next must be text/eot (suppress timestamps);
///   after a closed timestamp pair, the next must be a timestamp (suppress text);
/// - (c) timestamps must be monotonic non-decreasing.
fn apply_timestamp_rules(logits: &mut [f32], timestamp_begin: u32, generated: &[u32]) {
    let ts0 = timestamp_begin as usize;
    if ts0 >= logits.len() {
        return;
    }
    let neg = f32::NEG_INFINITY;

    // (a) open every segment with a timestamp.
    if generated.is_empty() {
        for v in &mut logits[..ts0] {
            *v = neg;
        }
        return;
    }

    let last = generated[generated.len() - 1];
    let penult = generated.len().checked_sub(2).map(|i| generated[i]);
    let last_ts = last >= timestamp_begin;
    let penult_ts = penult.is_some_and(|p| p >= timestamp_begin);

    // (b) pairing.
    if last_ts && !penult_ts {
        // lone timestamp → force text/eot
        for v in &mut logits[ts0..] {
            *v = neg;
        }
    } else if last_ts && penult_ts {
        // closed pair → force the next timestamp
        for v in &mut logits[..ts0] {
            *v = neg;
        }
    }

    // (c) monotonicity: forbid timestamps earlier than the last emitted one.
    if let Some(&min_ts) = generated.iter().rev().find(|&&t| t >= timestamp_begin) {
        let upper = (min_ts as usize).min(logits.len());
        if upper > ts0 {
            for v in &mut logits[ts0..upper] {
                *v = neg;
            }
        }
    }
}

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
    gen_cfg: WhisperGenConfig,
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

        // Optional suppress-token lists; default empty (no suppression) when the
        // repo ships no generation_config.json.
        let gen_cfg = files
            .generation_config_path
            .as_deref()
            .and_then(|p| WhisperGenConfig::from_config_file(p).ok())
            .unwrap_or_default();

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
            gen_cfg,
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
            // Timestamped path advances by the last decoded segment's end time
            // (long-form re-seek); the default path hops a fixed 30 s window.
            let (text, advance) = if opts.timestamps {
                transcribe_chunk_timestamped(
                    &mut engine,
                    &self.tokenizer,
                    &self.mel,
                    &samples[start..end],
                    opts,
                    &self.cfg,
                    &self.gen_cfg,
                )?
            } else {
                let t = transcribe_chunk(
                    &mut engine,
                    &self.tokenizer,
                    &self.mel,
                    &samples[start..end],
                    opts,
                    &self.cfg,
                    &self.gen_cfg,
                    &mut |_| {},
                )?;
                (t, chunk_len)
            };
            let text = text.trim();
            if !text.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(text);
            }
            start += advance.min(chunk_len).max(SAMPLE_RATE as usize);
            if end == samples.len() {
                break; // processed the tail
            }
        }
        Ok(out)
    }

    /// Transcribe an audio file, streaming decoded text as it is produced.
    pub async fn transcribe_stream(
        self: &Arc<Self>,
        path: impl AsRef<Path>,
        opts: TranscribeOptions,
    ) -> Result<ReceiverStream<String>> {
        let samples = load_audio(path, SAMPLE_RATE).context("decoding audio")?;
        Ok(self.transcribe_samples_stream(samples, opts))
    }

    /// Stream decoded text for already-decoded mono 16 kHz samples. The CPU-bound
    /// decode runs on a blocking task; text deltas arrive on the returned stream.
    pub fn transcribe_samples_stream(
        self: &Arc<Self>,
        samples: Vec<f32>,
        opts: TranscribeOptions,
    ) -> ReceiverStream<String> {
        let (tx, rx) = mpsc::channel::<String>(64);
        let me = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            if samples.is_empty() {
                return;
            }
            let mut engine = match me.engine.lock() {
                Ok(e) => e,
                Err(_) => return,
            };
            let chunk_len = me.mel.config().n_samples();
            let mut start = 0usize;
            let mut first_chunk = true;
            while start < samples.len() {
                let end = (start + chunk_len).min(samples.len());
                // Separate chunks with a space in the stream.
                if !first_chunk {
                    let _ = tx.blocking_send(" ".to_string());
                }
                first_chunk = false;
                let r = transcribe_chunk(
                    &mut engine,
                    &me.tokenizer,
                    &me.mel,
                    &samples[start..end],
                    &opts,
                    &me.cfg,
                    &me.gen_cfg,
                    &mut |delta| {
                        let _ = tx.blocking_send(delta.to_string());
                    },
                );
                if r.is_err() {
                    break; // surface nothing further; partial text already sent
                }
                start += chunk_len;
            }
        });
        ReceiverStream::new(rx)
    }
}

/// Transcribe one ≤30 s chunk: log-mel → encode → forced-prompt greedy decode.
/// `on_text` receives the newly-decoded text after each token (the stable delta),
/// enabling streaming; the full chunk text is returned regardless.
#[allow(clippy::too_many_arguments)]
fn transcribe_chunk(
    engine: &mut AudioEngine,
    tok: &WhisperTokenizer,
    mel: &MelFrontend,
    chunk: &[f32],
    opts: &TranscribeOptions,
    cfg: &WhisperConfig,
    gen_cfg: &WhisperGenConfig,
    on_text: &mut dyn FnMut(&str),
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
    // First sampled step: mask both the always-suppressed set and the
    // begin-only set (blank + eot). Subsequent steps mask only `suppress_tokens`.
    apply_suppress(&mut logits, &gen_cfg.suppress_tokens);
    apply_suppress(&mut logits, &gen_cfg.begin_suppress_tokens);

    let budget = opts
        .max_new_tokens
        .min(cfg.max_target_positions.saturating_sub(prompt.len()).max(1));
    let mut out_tokens = Vec::new();
    let mut emitted = String::new();
    for _ in 0..budget {
        let next = argmax(&logits);
        if tok.is_eot(next) {
            break;
        }
        // Keep only emitted text tokens; drop control/timestamp tokens but still
        // feed them back so positions advance correctly.
        if !tok.is_special(next) {
            out_tokens.push(next);
            // Emit the stable delta: re-decode the running tokens and send the
            // suffix beyond what we've already emitted (handles BPE re-merges).
            let full = tok.decode(&out_tokens, true)?;
            if let Some(delta) = full.strip_prefix(&emitted) {
                if !delta.is_empty() {
                    on_text(delta);
                }
                emitted = full;
            } else {
                emitted = full; // rare re-tokenization shift — resync silently
            }
        }
        logits = engine.decode_step(&[next])?;
        apply_suppress(&mut logits, &gen_cfg.suppress_tokens);
    }

    tok.decode(&out_tokens, true)
}

/// Timestamped chunk decode for long-form re-seeking. Returns the chunk text and
/// the number of samples to advance the window (the last decoded segment's end
/// timestamp, or the full chunk when no usable timestamp was produced).
#[allow(clippy::too_many_arguments)]
fn transcribe_chunk_timestamped(
    engine: &mut AudioEngine,
    tok: &WhisperTokenizer,
    mel: &MelFrontend,
    chunk: &[f32],
    opts: &TranscribeOptions,
    cfg: &WhisperConfig,
    gen_cfg: &WhisperGenConfig,
) -> Result<(String, usize)> {
    let chunk_len = mel.config().n_samples();
    let mel_t = mel.log_mel(chunk)?;
    engine.encode(&mel_t)?;

    let lang: Option<String> = match &opts.language {
        Some(l) => Some(l.clone()),
        None => {
            engine.reset_decoder();
            let logits = engine.decode_step(&[tok.sot])?;
            argmax_restricted(&logits, &tok.language_token_ids())
                .and_then(|id| tok.language_code(id).map(str::to_string))
        }
    };

    engine.reset_decoder();
    let prompt = tok.sot_sequence(lang.as_deref(), opts.translate, true); // timestamps on
    let mut logits = engine.decode_step(&prompt)?;
    apply_suppress(&mut logits, &gen_cfg.suppress_tokens);
    apply_suppress(&mut logits, &gen_cfg.begin_suppress_tokens);

    let budget = opts
        .max_new_tokens
        .min(cfg.max_target_positions.saturating_sub(prompt.len()).max(1));
    let mut generated: Vec<u32> = Vec::new(); // text + timestamp tokens
    let mut text_tokens: Vec<u32> = Vec::new();
    for _ in 0..budget {
        apply_timestamp_rules(&mut logits, tok.timestamp_begin, &generated);
        let next = argmax(&logits);
        if tok.is_eot(next) {
            break;
        }
        generated.push(next);
        if !tok.is_special(next) {
            text_tokens.push(next);
        }
        logits = engine.decode_step(&[next])?;
        apply_suppress(&mut logits, &gen_cfg.suppress_tokens);
    }

    let text = tok.decode(&text_tokens, true)?;
    // Advance by the last emitted timestamp (clamped to the window in the caller).
    let advance = generated
        .iter()
        .rev()
        .find(|&&t| tok.is_timestamp(t))
        .map(|&t| (timestamp_seconds(tok.timestamp_begin, t) * SAMPLE_RATE as f32) as usize)
        .filter(|&s| s > 0)
        .unwrap_or(chunk_len);
    Ok((text, advance))
}

/// Mask the given token ids to -inf so they can never be sampled.
fn apply_suppress(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        if let Some(v) = logits.get_mut(id as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const TB: u32 = 100; // synthetic timestamp_begin; ids < 100 are text, ≥100 are timestamps
    const VOCAB: usize = 160;

    fn fresh() -> Vec<f32> {
        vec![0.0f32; VOCAB]
    }

    #[test]
    fn timestamp_seconds_maps_correctly() {
        assert!((timestamp_seconds(TB, TB) - 0.0).abs() < 1e-6);
        assert!((timestamp_seconds(TB, TB + 50) - 1.0).abs() < 1e-6); // 50 * 0.02
    }

    #[test]
    fn first_position_forces_a_timestamp() {
        let mut l = fresh();
        apply_timestamp_rules(&mut l, TB, &[]);
        // All text logits (< TB) suppressed; timestamps untouched.
        assert!(l[..TB as usize].iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(l[TB as usize..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn after_lone_timestamp_forces_text() {
        let mut l = fresh();
        // generated ends with a lone timestamp (penult is text) → suppress timestamps.
        apply_timestamp_rules(&mut l, TB, &[5, TB + 10]);
        assert!(l[..TB as usize].iter().all(|&v| v == 0.0)); // text allowed
        assert!(l[TB as usize..].iter().all(|&v| v == f32::NEG_INFINITY));
    }

    #[test]
    fn after_closed_pair_forces_timestamp() {
        let mut l = fresh();
        // ...text, ts, ts (closed pair) → next must be a timestamp.
        apply_timestamp_rules(&mut l, TB, &[5, TB + 10, TB + 20]);
        assert!(l[..TB as usize].iter().all(|&v| v == f32::NEG_INFINITY));
        // monotonic: timestamps below the last (TB+20) also suppressed.
        assert!(l[TB as usize..(TB + 20) as usize]
            .iter()
            .all(|&v| v == f32::NEG_INFINITY));
        assert!(l[(TB + 20) as usize..].iter().all(|&v| v == 0.0));
    }
}
