//! Coherence test for `WhisperForward` on a synthetic tiny model.
//!
//! Builds a 2-mel / d=8 / 1-encoder / 2-decoder-layer Whisper with deterministic
//! F32 weights (F32 ⇒ no online Q8_0 quantization, so this is an exact path) and
//! asserts the structural invariants that the #1 correctness risks would break:
//!
//! 1. **Encoder** produces a finite `[1, n_audio_ctx, d]` context.
//! 2. **Batch == incremental decode**: feeding `[a, b, c]` in one call yields the
//!    same final-position logits as feeding `a`, then `b`, then `c` one at a time.
//!    This is the gold-standard check for the self-attention KV cache, the causal
//!    mask offset, and the learned-positional-embedding offset all lining up.
//! 3. **Cross-attention actually depends on the audio context** (a different
//!    encoder output changes the logits), proving the cross-attn path is wired.
//! 4. Decoding before `set_audio_context` is an error, not garbage.

use std::collections::HashMap;

use sapient_core::Tensor;
use sapient_hub::whisper_config::WhisperConfig;
use sapient_models::forward::WhisperForward;

const D: usize = 8;
const N_MELS: usize = 4;
const N_HEAD: usize = 2;
const FFN: usize = 16;
const VOCAB: usize = 20;
const ENC_LAYERS: usize = 1;
const DEC_LAYERS: usize = 2;
const N_AUDIO_CTX: usize = 8; // 16 mel frames → conv2 stride 2 → 8
const MAX_TARGET: usize = 16;

fn cfg() -> WhisperConfig {
    WhisperConfig {
        num_mel_bins: N_MELS,
        d_model: D,
        encoder_layers: ENC_LAYERS,
        encoder_attention_heads: N_HEAD,
        encoder_ffn_dim: FFN,
        decoder_layers: DEC_LAYERS,
        decoder_attention_heads: N_HEAD,
        decoder_ffn_dim: FFN,
        vocab_size: VOCAB,
        max_target_positions: MAX_TARGET,
        max_source_positions: N_AUDIO_CTX,
    }
}

/// Deterministic pseudo-random F32 tensor, small magnitude for numerical sanity.
fn rand_tensor(seed: usize, dims: Vec<usize>) -> Tensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i + seed * 977) as f32 * 0.123 + 0.7).sin() * 0.2)
        .collect();
    Tensor::from_f32(&data, dims).unwrap()
}

fn ln_weights(w: &mut HashMap<String, Tensor>, seed: usize, prefix: &str) {
    // LayerNorm weight near 1.0, bias near 0.0.
    let g: Vec<f32> = (0..D).map(|i| 1.0 + 0.01 * (i + seed) as f32).collect();
    w.insert(
        format!("{prefix}.weight"),
        Tensor::from_f32(&g, vec![D]).unwrap(),
    );
    w.insert(format!("{prefix}.bias"), rand_tensor(seed + 1, vec![D]));
}

fn build_weights() -> HashMap<String, Tensor> {
    let mut w = HashMap::new();
    let mut s = 1usize;
    let mut next = || {
        s += 1;
        s
    };

    // Encoder conv stem.
    w.insert(
        "encoder.conv1.weight".into(),
        rand_tensor(next(), vec![D, N_MELS, 3]),
    );
    w.insert("encoder.conv1.bias".into(), rand_tensor(next(), vec![D]));
    w.insert(
        "encoder.conv2.weight".into(),
        rand_tensor(next(), vec![D, D, 3]),
    );
    w.insert("encoder.conv2.bias".into(), rand_tensor(next(), vec![D]));
    w.insert(
        "encoder.embed_positions.weight".into(),
        rand_tensor(next(), vec![N_AUDIO_CTX, D]),
    );

    for li in 0..ENC_LAYERS {
        let p = format!("encoder.layers.{li}");
        ln_weights(&mut w, next(), &format!("{p}.self_attn_layer_norm"));
        for proj in ["q_proj", "v_proj", "out_proj"] {
            w.insert(
                format!("{p}.self_attn.{proj}.weight"),
                rand_tensor(next(), vec![D, D]),
            );
            w.insert(
                format!("{p}.self_attn.{proj}.bias"),
                rand_tensor(next(), vec![D]),
            );
        }
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(next(), vec![D, D]),
        );
        ln_weights(&mut w, next(), &format!("{p}.final_layer_norm"));
        w.insert(format!("{p}.fc1.weight"), rand_tensor(next(), vec![FFN, D]));
        w.insert(format!("{p}.fc1.bias"), rand_tensor(next(), vec![FFN]));
        w.insert(format!("{p}.fc2.weight"), rand_tensor(next(), vec![D, FFN]));
        w.insert(format!("{p}.fc2.bias"), rand_tensor(next(), vec![D]));
    }
    ln_weights(&mut w, next(), "encoder.layer_norm");

    // Decoder.
    w.insert(
        "decoder.embed_tokens.weight".into(),
        rand_tensor(next(), vec![VOCAB, D]),
    );
    w.insert(
        "decoder.embed_positions.weight".into(),
        rand_tensor(next(), vec![MAX_TARGET, D]),
    );
    for li in 0..DEC_LAYERS {
        let p = format!("decoder.layers.{li}");
        ln_weights(&mut w, next(), &format!("{p}.self_attn_layer_norm"));
        ln_weights(&mut w, next(), &format!("{p}.encoder_attn_layer_norm"));
        ln_weights(&mut w, next(), &format!("{p}.final_layer_norm"));
        for attn in ["self_attn", "encoder_attn"] {
            for proj in ["q_proj", "v_proj", "out_proj"] {
                w.insert(
                    format!("{p}.{attn}.{proj}.weight"),
                    rand_tensor(next(), vec![D, D]),
                );
                w.insert(
                    format!("{p}.{attn}.{proj}.bias"),
                    rand_tensor(next(), vec![D]),
                );
            }
            w.insert(
                format!("{p}.{attn}.k_proj.weight"),
                rand_tensor(next(), vec![D, D]),
            );
        }
        w.insert(format!("{p}.fc1.weight"), rand_tensor(next(), vec![FFN, D]));
        w.insert(format!("{p}.fc1.bias"), rand_tensor(next(), vec![FFN]));
        w.insert(format!("{p}.fc2.weight"), rand_tensor(next(), vec![D, FFN]));
        w.insert(format!("{p}.fc2.bias"), rand_tensor(next(), vec![D]));
    }
    ln_weights(&mut w, next(), "decoder.layer_norm");

    w
}

fn mel_input(seed: usize) -> Tensor {
    // [1, n_mels, 16 frames] → conv2 stride 2 → 8 audio-context frames.
    rand_tensor(seed, vec![1, N_MELS, 16])
}

#[test]
fn encoder_output_is_finite_and_correct_shape() {
    let mut w = WhisperForward::from_weights(cfg(), build_weights()).unwrap();
    let ctx = w.encode(&mel_input(7)).unwrap();
    assert_eq!(ctx.shape().dims(), &[1, N_AUDIO_CTX, D]);
    assert!(ctx.as_f32_slice().iter().all(|v| v.is_finite()));
}

#[test]
fn batch_equals_incremental_decode() {
    let weights = build_weights();
    let tokens = [3u32, 11, 5];

    // Batch: feed all tokens at once.
    let mut wb = WhisperForward::from_weights(cfg(), weights.clone()).unwrap();
    let ctx = wb.encode(&mel_input(7)).unwrap();
    wb.set_audio_context(&ctx).unwrap();
    let batch_logits = wb.decode_step(&tokens).unwrap();

    // Incremental: one token per step, reusing the KV cache.
    let mut wi = WhisperForward::from_weights(cfg(), weights).unwrap();
    let ctx = wi.encode(&mel_input(7)).unwrap();
    wi.set_audio_context(&ctx).unwrap();
    wi.decode_step(&tokens[0..1]).unwrap();
    wi.decode_step(&tokens[1..2]).unwrap();
    let inc_logits = wi.decode_step(&tokens[2..3]).unwrap();

    assert_eq!(batch_logits.len(), VOCAB);
    let max_err = batch_logits
        .iter()
        .zip(&inc_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "batch vs incremental decode diverged: max_err={max_err}"
    );
}

#[test]
fn cross_attention_depends_on_audio_context() {
    let weights = build_weights();
    let tokens = [3u32, 11, 5];

    let logits_for = |mel_seed: usize| {
        let mut w = WhisperForward::from_weights(cfg(), weights.clone()).unwrap();
        let ctx = w.encode(&mel_input(mel_seed)).unwrap();
        w.set_audio_context(&ctx).unwrap();
        w.decode_step(&tokens).unwrap()
    };

    let a = logits_for(7);
    let b = logits_for(42);
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-5,
        "logits identical for different audio — cross-attention not wired"
    );
}

#[test]
fn decode_before_set_audio_context_errors() {
    let mut w = WhisperForward::from_weights(cfg(), build_weights()).unwrap();
    // No encode / set_audio_context.
    assert!(w.decode_step(&[3u32]).is_err());
}
