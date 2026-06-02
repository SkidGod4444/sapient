//! Orpheus → SNAC text-to-speech pipeline (Phase 6d, LM-codec TTS).
//!
//! `sapient speak` synthesises speech with a two-stage, fully pure-Rust path:
//!
//! 1. **LM (Orpheus-3B)** — a Llama-3.2-3B fine-tune run by the existing
//!    [`Pipeline`]/`LlamaForward` engine. Given a voice-prefixed prompt it emits
//!    SNAC neural-audio-codec **token ids** (not text). No grapheme-to-phoneme
//!    front-end is needed (raw-text BPE), so there is no GPLv3 `espeak` dep.
//! 2. **SNAC decoder** — the pure-Rust [`SnacDecoder`] turns those codec tokens
//!    back into a 24 kHz waveform, written to a WAV file.
//!
//! Orpheus packs **7 codes per audio frame**, each offset by its position
//! (`code = token − 128266 − position·4096`), interleaving SNAC's 3-level RVQ
//! hierarchy; [`orpheus_codes_to_snac`] de-frames them. The control tokens that
//! wrap the prompt are injected as literal ids (the text body is BPE-encoded
//! with no special tokens), and the generated codec ids are read raw — so the
//! tokenizer only ever handles ordinary text.

use std::path::Path;

use anyhow::{Context, Result};
use sapient_audio::write_wav;
use sapient_hub::snac_config::SnacConfig;
use sapient_hub::HubClient;
use sapient_models::forward::{normalize_snac_weights, orpheus_codes_to_snac, SnacDecoder};
use sapient_models::{weights, LlmBackendKind};

use crate::pipeline::{LoadOptions, Pipeline};
use crate::sampler::SamplingStrategy;

// ── Orpheus-3B control / audio token ids (Llama-3.2 vocab extended to 156940) ──
const START_HUMAN: u32 = 128259; // opens the prompt
const END_TEXT: u32 = 128009; // closes the text turn
const START_AI: u32 = 128260;
const START_SPEECH: u32 = 128261;
const BEGIN_AUDIO: u32 = 128257; // last prompt token before generation
const END_SPEECH: u32 = 128258; // stop token
const AUDIO_BASE: u32 = 128266; // id of SNAC code 0 at frame position 0
const CODE_SPAN: u32 = 7 * 4096; // 28672 ids span one 7-position frame

/// Voices the Orpheus-3B fine-tune was trained on (best quality, in order).
pub const ORPHEUS_VOICES: &[&str] = &["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];
/// Default voice when none is requested.
pub const DEFAULT_ORPHEUS_VOICE: &str = "tara";

/// SNAC codec repo (safetensors + config.json). The `mlx-community` mirror is
/// ungated and stores plain f32 weights; [`normalize_snac_weights`] adapts its
/// layout. Override the *weights source* with `SAPIENT_SNAC_DIR` (a local dir).
const DEFAULT_SNAC_REPO: &str = "mlx-community/snac_24khz";

/// Text-to-speech pipeline: Orpheus LM + SNAC codec decoder.
pub struct SpeakPipeline {
    lm: Pipeline,
    snac: SnacDecoder,
    sample_rate: u32,
}

impl SpeakPipeline {
    /// Load the Orpheus LM (`model` alias/repo) on `backend` plus the default
    /// SNAC codec.
    pub async fn from_pretrained_with_backend(
        model: &str,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        Self::from_pretrained_with_snac(model, backend, DEFAULT_SNAC_REPO).await
    }

    /// Like [`from_pretrained_with_backend`](Self::from_pretrained_with_backend)
    /// but with an explicit SNAC codec repo id.
    pub async fn from_pretrained_with_snac(
        model: &str,
        backend: LlmBackendKind,
        snac_repo: &str,
    ) -> Result<Self> {
        let opts = LoadOptions {
            backend,
            ..Default::default()
        };
        let lm = Pipeline::from_pretrained_with_opts(model, opts)
            .await
            .with_context(|| format!("loading Orpheus TTS model '{model}'"))?;
        let snac = load_snac(snac_repo).await?;
        let sample_rate = snac.config().sampling_rate;
        Ok(Self {
            lm,
            snac,
            sample_rate,
        })
    }

    /// Output sample rate of the synthesised waveform (24 kHz for `snac_24khz`).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Build the Orpheus prompt token sequence for `text` in `voice`:
    /// `[START_HUMAN] + tokenizer("{voice}: {text}") + [END_TEXT, START_AI,
    /// START_SPEECH, BEGIN_AUDIO]`.
    ///
    /// The body is encoded **with** the tokenizer's special tokens — the
    /// Llama-3.2 post-processor prepends BOS (`128000`), so the realized prefix
    /// is `[128259, 128000, …text…]`, exactly matching the reference Orpheus
    /// implementation (`tokenizer(prompt).input_ids`). Dropping the BOS yields
    /// fluent-but-wrong speech, so it is required.
    fn build_prompt_ids(&self, text: &str, voice: &str) -> Result<Vec<u32>> {
        let body = self
            .lm
            .tokenizer()
            .encode_ids(&format!("{voice}: {text}"), true)?;
        let mut ids = Vec::with_capacity(body.len() + 5);
        ids.push(START_HUMAN);
        ids.extend_from_slice(&body);
        ids.extend_from_slice(&[END_TEXT, START_AI, START_SPEECH, BEGIN_AUDIO]);
        Ok(ids)
    }

    /// Synthesise `text` in `voice`, returning the mono 24 kHz waveform in
    /// `[-1, 1]`. Runs synchronously (CPU-bound LM decode + SNAC); call from a
    /// blocking context.
    pub fn speak(&self, text: &str, voice: &str) -> Result<Vec<f32>> {
        let text = text.trim();
        if text.is_empty() {
            anyhow::bail!("nothing to speak (empty text)");
        }
        let voice = if voice.is_empty() {
            DEFAULT_ORPHEUS_VOICE
        } else {
            voice
        };
        let prompt = self.build_prompt_ids(text, voice)?;

        // Orpheus needs sampling (greedy degenerates) with a repetition penalty
        // ≥ 1.1; ~83 codec tokens ≈ 1 s of speech, so scale the cap to the word
        // count and rely on the stop token for the true end.
        let strategy = SamplingStrategy::Combined {
            top_k: 0,
            top_p: 0.9,
            temperature: 0.6,
            repetition_penalty: 1.1,
        };
        let words = text.split_whitespace().count().max(1);
        let max_new = (words * 90).clamp(256, 4096);

        let generated = self
            .lm
            .generate_token_ids(&prompt, max_new, &[END_SPEECH], strategy)?;

        // Keep only audio-code tokens, rebased to the 0..CODE_SPAN range that
        // `orpheus_codes_to_snac` expects, then drop any trailing partial frame.
        let mut codes: Vec<u32> = generated
            .iter()
            .filter(|&&t| (AUDIO_BASE..AUDIO_BASE + CODE_SPAN).contains(&t))
            .map(|&t| t - AUDIO_BASE)
            .collect();
        codes.truncate(codes.len() / 7 * 7);
        if codes.is_empty() {
            anyhow::bail!(
                "Orpheus emitted no usable audio frames ({} tokens generated) — \
                 the model may not be the TTS fine-tune",
                generated.len()
            );
        }

        let levels = orpheus_codes_to_snac(&codes)?;
        self.snac.decode(&levels)
    }

    /// Synthesise `text` in `voice` and write it to `out` as a 16-bit PCM WAV.
    /// Returns the number of samples written.
    pub fn speak_to_wav(&self, text: &str, voice: &str, out: &Path) -> Result<usize> {
        let wav = self.speak(text, voice)?;
        write_wav(out, &wav, self.sample_rate)
            .with_context(|| format!("writing WAV to {}", out.display()))?;
        Ok(wav.len())
    }

    /// Waveform samples produced per Orpheus frame (`vq_strides[0] × ∏ decoder_rates`).
    fn samples_per_frame(&self) -> usize {
        let cfg = self.snac.config();
        cfg.vq_strides.first().copied().unwrap_or(4) * cfg.decoder_rates.iter().product::<usize>()
    }

    /// Streaming synthesis: emits audio chunks via `on_audio(samples, rate)` as
    /// the codec LM decodes, so playback can begin long before the clip is done.
    /// Used by `sapient converse` for lower-latency spoken replies.
    ///
    /// It re-decodes the running code sequence every [`STREAM_CHUNK_FRAMES`] and
    /// emits only the *stable* prefix — holding back [`STREAM_MARGIN_FRAMES`] of
    /// look-ahead so an already-played sample never changes when more right-context
    /// arrives (SNAC's convs have a finite right receptive field) — then flushes
    /// the tail at the end.
    pub fn speak_streaming(
        &self,
        text: &str,
        voice: &str,
        on_audio: &mut dyn FnMut(&[f32], u32),
    ) -> Result<()> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        let voice = if voice.is_empty() {
            DEFAULT_ORPHEUS_VOICE
        } else {
            voice
        };
        let prompt = self.build_prompt_ids(text, voice)?;
        let strategy = SamplingStrategy::Combined {
            top_k: 0,
            top_p: 0.9,
            temperature: 0.6,
            repetition_penalty: 1.1,
        };
        let words = text.split_whitespace().count().max(1);
        let max_new = (words * 90).clamp(256, 4096);

        let spf = self.samples_per_frame();
        let sr = self.sample_rate;
        let mut codes: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let mut since = 0usize;

        self.lm
            .generate_token_ids_streaming(&prompt, max_new, &[END_SPEECH], strategy, |tok| {
                if (AUDIO_BASE..AUDIO_BASE + CODE_SPAN).contains(&tok) {
                    codes.push(tok - AUDIO_BASE);
                    if codes.len() % 7 == 0 {
                        since += 1;
                        if since >= STREAM_CHUNK_FRAMES {
                            since = 0;
                            let _ = emit_stable(
                                &self.snac,
                                &codes,
                                &mut emitted,
                                spf,
                                sr,
                                false,
                                on_audio,
                            );
                        }
                    }
                }
                true
            })?;
        // Flush whatever remains (the true tail of the clip).
        emit_stable(&self.snac, &codes, &mut emitted, spf, sr, true, on_audio)
    }
}

/// Decode every ~8 frames (~0.34 s of audio) during streaming.
const STREAM_CHUNK_FRAMES: usize = 8;
/// Look-ahead frames held back so an emitted sample never changes on re-decode.
const STREAM_MARGIN_FRAMES: usize = 8;

/// Decode the running code sequence and emit the newly-*stable* samples (or the
/// whole tail when `flush`). Re-decoding from the start each call keeps full
/// left-context (no boundary clicks); holding back [`STREAM_MARGIN_FRAMES`] keeps
/// already-emitted samples byte-stable across calls.
#[allow(clippy::too_many_arguments)]
fn emit_stable(
    snac: &SnacDecoder,
    codes: &[u32],
    emitted: &mut usize,
    spf: usize,
    sr: u32,
    flush: bool,
    on_audio: &mut dyn FnMut(&[f32], u32),
) -> Result<()> {
    let usable = codes.len() / 7 * 7;
    if usable == 0 {
        return Ok(());
    }
    let levels = orpheus_codes_to_snac(&codes[..usable])?;
    let wave = snac.decode(&levels)?;
    let total_frames = usable / 7;
    let upto = if flush {
        wave.len()
    } else {
        (total_frames.saturating_sub(STREAM_MARGIN_FRAMES) * spf).min(wave.len())
    };
    if upto > *emitted {
        on_audio(&wave[*emitted..upto], sr);
        *emitted = upto;
    }
    Ok(())
}

/// Use Orpheus TTS as the synthesizer in the speech-to-speech cascade
/// (`sapient converse --speak`). Note: a 3B LM decode is slow on CPU, so spoken
/// replies lag well behind text — keep replies short.
impl crate::converse::Tts for SpeakPipeline {
    fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        self.speak(text, DEFAULT_ORPHEUS_VOICE)
    }
    fn synthesize_streaming(
        &self,
        text: &str,
        on_audio: &mut dyn FnMut(&[f32], u32),
    ) -> Result<()> {
        self.speak_streaming(text, DEFAULT_ORPHEUS_VOICE, on_audio)
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Load the SNAC decoder: a local dir (`SAPIENT_SNAC_DIR`) if set, else the
/// `snac_repo` from the Hub.
async fn load_snac(snac_repo: &str) -> Result<SnacDecoder> {
    if let Ok(dir) = std::env::var("SAPIENT_SNAC_DIR") {
        return load_snac_from_dir(Path::new(&dir));
    }
    let hub = HubClient::new()?;
    let files = hub
        .download_files(snac_repo, &["config.json", "model.safetensors"])
        .await
        .with_context(|| format!("downloading SNAC codec '{snac_repo}'"))?;
    let cfg = SnacConfig::from_config_file(&files[0])?;
    let raw = weights::load_hf_weights(&[files[1].clone()])?;
    Ok(SnacDecoder::from_weights(cfg, normalize_snac_weights(raw)?))
}

/// Load SNAC weights + config from a local directory (accepts either the
/// `mlx-community` `model.safetensors` or a converted `snac.safetensors`).
fn load_snac_from_dir(dir: &Path) -> Result<SnacDecoder> {
    let cfg = SnacConfig::from_config_file(&dir.join("config.json"))?;
    let st = ["model.safetensors", "snac.safetensors"]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no SNAC safetensors (model.safetensors / snac.safetensors) in {}",
                dir.display()
            )
        })?;
    let raw = weights::load_hf_weights(&[st])?;
    Ok(SnacDecoder::from_weights(cfg, normalize_snac_weights(raw)?))
}
