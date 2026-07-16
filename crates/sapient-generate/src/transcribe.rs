// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

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

/// Build the audio engine for the chosen backend. The wgpu GPU engine (when the
/// `wgpu` feature is compiled in) runs the transformer body on the GPU from raw
/// f32 weights — selected by an explicit `--backend wgpu`, or by `auto` when a
/// GPU adapter exists and MLX/Metal doesn't take precedence on Apple Silicon
/// (Phase 10.4). Every other backend uses the CPU/Metal `WhisperForward` (which
/// online-quantizes its linears to Q8_0).
fn build_audio_engine(
    cfg: WhisperConfig,
    weights: HashMap<String, Tensor>,
    backend: LlmBackendKind,
) -> Result<AudioEngine> {
    #[cfg(feature = "wgpu")]
    if sapient_models::forward::whisper_wants_wgpu(backend) {
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
    /// Emit timestamp tokens and re-seek long audio by segment end times.
    pub timestamps: bool,
    /// Maximum new tokens decoded per 30 s chunk.
    pub max_new_tokens: usize,
    /// Beam width. `<= 1` = greedy (default). Larger trades cost for quality
    /// (each step replays every beam's prefix — O(beam·tokens) forwards).
    pub beam_size: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            timestamps: false,
            max_new_tokens: 224,
            beam_size: 1,
        }
    }
}

/// A loaded Whisper speech-to-text pipeline.
#[derive(Clone)]
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
            } else if opts.beam_size > 1 {
                let t = decode_chunk_beam(
                    &mut engine,
                    &self.tokenizer,
                    &self.mel,
                    &samples[start..end],
                    opts,
                    &self.cfg,
                    &self.gen_cfg,
                )?;
                (t, chunk_len)
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

/// Beam-search chunk decode (prefix-replay; no engine cache snapshot needed).
/// Each step replays every live beam's prefix through `decode_step` and keeps the
/// `beam_size` best continuations by length-normalized log-probability.
fn decode_chunk_beam(
    engine: &mut AudioEngine,
    tok: &WhisperTokenizer,
    mel: &MelFrontend,
    chunk: &[f32],
    opts: &TranscribeOptions,
    cfg: &WhisperConfig,
    gen_cfg: &WhisperGenConfig,
) -> Result<String> {
    let beam_size = opts.beam_size.max(1);
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
    let prompt = tok.sot_sequence(lang.as_deref(), opts.translate, opts.timestamps);
    let budget = opts
        .max_new_tokens
        .min(cfg.max_target_positions.saturating_sub(prompt.len()).max(1));

    // Each beam: tokens decoded after the prompt + cumulative log-prob.
    let mut beams: Vec<Beam> = vec![Beam {
        tokens: Vec::new(),
        logprob: 0.0,
    }];
    let mut finished: Vec<Beam> = Vec::new();

    for _ in 0..budget {
        if beams.is_empty() {
            break;
        }
        let mut cands: Vec<Beam> = Vec::new();
        for beam in &beams {
            engine.reset_decoder();
            let full: Vec<u32> = prompt.iter().chain(&beam.tokens).copied().collect();
            let mut logits = engine.decode_step(&full)?;
            apply_suppress(&mut logits, &gen_cfg.suppress_tokens);
            if beam.tokens.is_empty() {
                apply_suppress(&mut logits, &gen_cfg.begin_suppress_tokens);
            }
            log_softmax_inplace(&mut logits);
            for (id, lp) in top_k(&logits, beam_size) {
                let mut tokens = beam.tokens.clone();
                tokens.push(id);
                cands.push(Beam {
                    tokens,
                    logprob: beam.logprob + lp,
                });
            }
        }
        beams = prune_beams(cands, beam_size, tok.eot, &mut finished);
    }
    finished.extend(beams);

    // Best by length-normalized log-prob.
    let best = finished
        .into_iter()
        .max_by(|a, b| a.norm_score().total_cmp(&b.norm_score()));
    let Some(best) = best else {
        return Ok(String::new());
    };
    let text_tokens: Vec<u32> = best
        .tokens
        .into_iter()
        .filter(|&t| !tok.is_special(t))
        .collect();
    tok.decode(&text_tokens, true)
}

struct Beam {
    tokens: Vec<u32>,
    logprob: f32,
}

impl Beam {
    /// Length-normalized log-prob (avoids the bias toward shorter sequences).
    fn norm_score(&self) -> f32 {
        self.logprob / (self.tokens.len().max(1) as f32)
    }
}

/// In-place log-softmax over a logits row (ignores -inf entries correctly).
fn log_softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let sum_exp: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    let log_z = max + sum_exp.ln();
    for v in logits.iter_mut() {
        *v -= log_z;
    }
}

/// Indices + values of the top-`k` entries (descending).
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let len = logits.len();
    let k = k.min(len);
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<u32> = (0..len as u32).collect();
    let mid = (k - 1).min(len - 1);
    idx.select_nth_unstable_by(mid, |&a, &b| {
        logits[b as usize].total_cmp(&logits[a as usize])
    });
    let mut top: Vec<(u32, f32)> = idx[..k].iter().map(|&i| (i, logits[i as usize])).collect();
    top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    top
}

/// Keep the `beam_size` best candidates by length-normalized score; candidates
/// ending in `eot` move to `finished` (with the eot trimmed).
fn prune_beams(
    mut cands: Vec<Beam>,
    beam_size: usize,
    eot: u32,
    finished: &mut Vec<Beam>,
) -> Vec<Beam> {
    cands.sort_unstable_by(|a, b| b.norm_score().total_cmp(&a.norm_score()));
    let mut kept = Vec::new();
    for mut c in cands {
        if kept.len() >= beam_size {
            break;
        }
        if c.tokens.last() == Some(&eot) {
            c.tokens.pop();
            finished.push(c);
        } else {
            kept.push(c);
        }
    }
    kept
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

    #[test]
    fn log_softmax_sums_to_one() {
        let mut l = vec![1.0f32, 2.0, 3.0, f32::NEG_INFINITY];
        log_softmax_inplace(&mut l);
        let sum: f32 = l.iter().filter(|v| v.is_finite()).map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sums to {sum}");
        assert!(l[3] == f32::NEG_INFINITY); // -inf stays -inf
    }

    #[test]
    fn top_k_picks_largest_descending() {
        let l = vec![0.1f32, 0.9, 0.3, 0.7, 0.2];
        let top = top_k(&l, 3);
        assert_eq!(
            top.iter().map(|&(i, _)| i).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
    }

    #[test]
    fn prune_keeps_best_and_finishes_eot() {
        let eot = 99u32;
        let cands = vec![
            Beam {
                tokens: vec![1, 2],
                logprob: -1.0,
            }, // norm -0.5
            Beam {
                tokens: vec![3, eot],
                logprob: -0.4,
            }, // ends in eot → finished
            Beam {
                tokens: vec![4, 5],
                logprob: -3.0,
            }, // norm -1.5 (worst)
        ];
        let mut finished = Vec::new();
        let kept = prune_beams(cands, 2, eot, &mut finished);
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].tokens, vec![3]); // eot trimmed
                                                 // Best live beam kept; the worst is dropped only if over capacity.
        assert!(kept.iter().any(|b| b.tokens == vec![1, 2]));
    }
}
