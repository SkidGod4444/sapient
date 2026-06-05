//! `KokoroModel` — the full StyleTTS2 + ISTFTNet forward pass, tying the ALBERT
//! encoder, the prosody predictor, the prosodic text encoder, and the ISTFTNet
//! decoder into `phonemes (input_ids) + voice → 24 kHz waveform`. Non-autoregressive
//! (one forward pass, no codec-token loop) — the real-time TTS path for `converse`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use sapient_core::{Shape, Tensor};

use super::super::conv::conv1d;
use super::albert::albert_encode;
use super::decoder::decode;
use super::loader::{load_from_dir, KokoroConfig};
use super::ops::{
    layer_norm_rows, leaky_relu_inplace, length_regulate, linear2d, lstm_bidirectional, LstmParams,
};
use super::predictor::{f0_n_train, predict_prosody};

/// 24 kHz — Kokoro's fixed output sample rate.
pub const KOKORO_SAMPLE_RATE: u32 = 24_000;

fn get<'a>(w: &'a HashMap<String, Tensor>, k: &str) -> Result<&'a Tensor> {
    w.get(k)
        .ok_or_else(|| anyhow!("kokoro: missing weight {k}"))
}

fn transpose(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            o[j * r + i] = x[i * c + j];
        }
    }
    o
}

/// Prosodic `TextEncoder`: embedding → 3×(Conv1d k5 + channel-LayerNorm +
/// LeakyReLU) → BiLSTM. Returns `t_en` laid out `[hidden, L]`.
pub(super) fn text_encode(
    w: &HashMap<String, Tensor>,
    input_ids: &[u32],
    cfg: &KokoroConfig,
) -> Result<Vec<f32>> {
    let l = input_ids.len();
    let h = cfg.hidden_dim; // 512
    let emb = get(w, "text_encoder.embedding.weight")?.to_f32_vec(); // [n_token, 512]
    let n_rows = emb.len() / h;
    // embedding → [L, h] → transpose → [h, L]. Clamp the id so a stray out-of-range
    // token (see `phonemes_to_ids`) can't index past the table and panic.
    let mut e = vec![0.0f32; l * h];
    for (i, &id) in input_ids.iter().enumerate() {
        let row = (id as usize).min(n_rows - 1);
        e[i * h..i * h + h].copy_from_slice(&emb[row * h..row * h + h]);
    }
    let mut x = transpose(&e, l, h); // [h, L]

    let k = cfg.text_encoder_kernel_size; // 5
    let pad = (k - 1) / 2;
    for i in 0..cfg.n_layer {
        let xt = Tensor::from_f32(&x, Shape::new([1, h, l])).map_err(|e| anyhow!("{e}"))?;
        let conv = conv1d(
            &xt,
            get(w, &format!("text_encoder.cnn.{i}.0.weight"))?,
            Some(get(w, &format!("text_encoder.cnn.{i}.0.bias"))?),
            pad,
            1,
            1,
            1,
        )?;
        // channel-axis LayerNorm: [h,L] → [L,h] → LN(gamma,beta) → [h,L]
        let cv = conv.to_f32_vec();
        let mut rows = transpose(&cv, h, l); // [L, h]
        let gamma = get(w, &format!("text_encoder.cnn.{i}.1.gamma"))?.to_f32_vec();
        let beta = get(w, &format!("text_encoder.cnn.{i}.1.beta"))?.to_f32_vec();
        layer_norm_rows(&mut rows, l, h, &gamma, &beta, 1e-5);
        leaky_relu_inplace(&mut rows, 0.2);
        x = transpose(&rows, l, h); // [h, L]
    }

    // BiLSTM: [h,L] → [L,h] → BiLSTM(h→h) → [L,h] → [h,L]
    let xt = transpose(&x, h, l);
    let xt = Tensor::from_f32(&xt, Shape::new([l, h])).map_err(|e| anyhow!("{e}"))?;
    let fwd = LstmParams {
        weight_ih: get(w, "text_encoder.lstm.weight_ih_l0")?,
        weight_hh: get(w, "text_encoder.lstm.weight_hh_l0")?,
        bias_ih: Some(get(w, "text_encoder.lstm.bias_ih_l0")?),
        bias_hh: Some(get(w, "text_encoder.lstm.bias_hh_l0")?),
    };
    let bwd = LstmParams {
        weight_ih: get(w, "text_encoder.lstm.weight_ih_l0_reverse")?,
        weight_hh: get(w, "text_encoder.lstm.weight_hh_l0_reverse")?,
        bias_ih: Some(get(w, "text_encoder.lstm.bias_ih_l0_reverse")?),
        bias_hh: Some(get(w, "text_encoder.lstm.bias_hh_l0_reverse")?),
    };
    let y = lstm_bidirectional(&xt, &fwd, &bwd)?; // [L, h]
    Ok(transpose(&y.to_f32_vec(), l, h)) // [h, L]
}

/// A loaded Kokoro-82M model ready to synthesize.
pub struct KokoroModel {
    weights: HashMap<String, Tensor>,
    config: KokoroConfig,
    voices: HashMap<String, Tensor>,
}

impl KokoroModel {
    /// Load from a converted-weights directory (`config.json`,
    /// `model.safetensors`, `voices.safetensors`).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let a = load_from_dir(dir)?;
        Ok(Self {
            weights: a.weights,
            config: a.config,
            voices: a.voices,
        })
    }

    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    /// Largest token id the embedding tables can actually index. The two
    /// embeddings keyed by `input_ids` — ALBERT's `word_embeddings` and the
    /// text-encoder embedding — must both accept the id, so the safe bound is
    /// the smaller row count. Used to drop ids the model can't embed.
    fn embedding_vocab_len(&self) -> usize {
        let rows = |k: &str| {
            self.weights
                .get(k)
                .map(|t| t.shape().dims()[0])
                .unwrap_or(usize::MAX)
        };
        rows("bert.embeddings.word_embeddings.weight").min(rows("text_encoder.embedding.weight"))
    }

    /// Max inner phoneme count a single forward pass can encode. ALBERT's learned
    /// position embedding has a fixed capacity (~512 rows); the pad-wrapped id
    /// sequence (`+2`) must fit inside it, so longer inputs are truncated rather
    /// than indexing past the table and panicking. (The reference Kokoro chunks
    /// long text into ≤510-phoneme segments — truncation is the minimal guard.)
    fn max_inner_phonemes(&self) -> usize {
        self.weights
            .get("bert.embeddings.position_embeddings.weight")
            .map(|t| t.shape().dims()[0].saturating_sub(2))
            .unwrap_or(usize::MAX)
    }

    /// Map a phoneme string to token ids: `[0, vocab[c]…, 0]` (pad-wrapped),
    /// skipping characters absent from the vocab. Returns the inner phoneme count
    /// too (for voice-pack indexing).
    ///
    /// Also drops any id the embedding tables can't index. A mirror's `config.json`
    /// `vocab` can map a rare IPA symbol to an id ≥ the embedding row count
    /// (observed: id 512 with a 512-row table), which previously panicked with
    /// "index out of bounds" mid-synthesis; skipping it (like an unknown char)
    /// just omits that one phoneme.
    pub fn phonemes_to_ids(&self, phonemes: &str) -> (Vec<u32>, usize) {
        let vocab_len = self.embedding_vocab_len() as u32;
        let mut dropped = 0usize;
        let mut inner: Vec<u32> = phonemes
            .chars()
            .filter_map(|c| self.config.vocab.get(&c.to_string()).copied())
            .filter(|&id| {
                let ok = id < vocab_len;
                if !ok {
                    dropped += 1;
                }
                ok
            })
            .collect();
        if dropped > 0 {
            tracing::warn!("kokoro: dropped {dropped} phoneme id(s) ≥ embedding vocab {vocab_len}");
        }
        // Cap to the position-embedding capacity so a long reply can't overflow it.
        let max_inner = self.max_inner_phonemes();
        if inner.len() > max_inner {
            tracing::warn!(
                "kokoro: truncating {} phonemes to {max_inner} (position-embedding limit)",
                inner.len()
            );
            inner.truncate(max_inner);
        }
        let n = inner.len();
        let mut ids = Vec::with_capacity(n + 2);
        ids.push(0);
        ids.extend_from_slice(&inner);
        ids.push(0);
        (ids, n)
    }

    /// The style/ref vector `[256]` for a voice at a given phoneme count, indexed
    /// like `KPipeline` (`pack[len(ps)-1]`, clamped to the pack range).
    pub fn ref_s(&self, voice: &str, phoneme_count: usize) -> Result<Vec<f32>> {
        let pack = self
            .voices
            .get(voice)
            .ok_or_else(|| anyhow!("kokoro: voice '{voice}' not loaded"))?;
        let d = pack.shape().dims().to_vec(); // [510, 256]
        let rows = d[0];
        let cols = d[1];
        let idx = phoneme_count.saturating_sub(1).min(rows - 1);
        let v = pack.to_f32_vec();
        Ok(v[idx * cols..idx * cols + cols].to_vec())
    }

    /// Synthesize a waveform from pad-wrapped token ids and a `[256]` ref vector.
    pub fn synthesize_ids(&self, input_ids: &[u32], ref_s: &[f32], speed: f32) -> Result<Vec<f32>> {
        let w = &self.weights;
        let cfg = &self.config;
        let l = input_ids.len();
        let h = cfg.hidden_dim;
        let timing = std::env::var("SAPIENT_KOKORO_TIMING").is_ok();
        macro_rules! stage {
            ($name:expr, $e:expr) => {{
                let t = std::time::Instant::now();
                let r = $e;
                if timing {
                    eprintln!(
                        "  [kokoro] {:<14} {:>7.1} ms",
                        $name,
                        t.elapsed().as_secs_f32() * 1000.0
                    );
                }
                r
            }};
        }

        // ALBERT → bert_encoder → d_en [h, L]
        let bert = stage!("albert", albert_encode(w, input_ids, &cfg.plbert)?);
        let be_w = get(w, "bert_encoder.weight")?.to_f32_vec();
        let be_b = get(w, "bert_encoder.bias")?.to_f32_vec();
        let be = linear2d(&bert, l, cfg.plbert.hidden_size, &be_w, Some(&be_b), h);
        let d_en = transpose(&be, l, h); // [h, L]

        let s_pred = &ref_s[cfg.style_dim..]; // predictor style (128:)
        let s_dec = &ref_s[..cfg.style_dim]; // decoder style (:128)

        let prosody = stage!("prosody", predict_prosody(w, cfg, &d_en, l, s_pred, speed)?);
        let (f0, n) = stage!("f0/n", f0_n_train(w, cfg, &prosody.en, s_pred)?);

        let t_en = stage!("text_encoder", text_encode(w, input_ids, cfg)?); // [h, L]
        let t_en = Tensor::from_f32(&t_en, Shape::new([1, h, l])).map_err(|e| anyhow!("{e}"))?;
        let asr = length_regulate(&t_en, &prosody.pred_dur)?; // [1, h, T]

        stage!("decoder", decode(w, cfg, &asr, &f0, &n, s_dec))
    }

    /// Convenience: synthesize from a phoneme string + a loaded voice name.
    pub fn synthesize(&self, phonemes: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let (ids, n) = self.phonemes_to_ids(phonemes);
        let ref_s = self
            .ref_s(voice, n)
            .with_context(|| format!("voice {voice}"))?;
        self.synthesize_ids(&ids, &ref_s, speed)
    }
}
