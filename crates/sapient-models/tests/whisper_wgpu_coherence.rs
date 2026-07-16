// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Coherence test: the wgpu Whisper engine must match the CPU `WhisperForward`.
//!
//! Builds a synthetic tiny Whisper with F32 weights (F32 ⇒ neither engine
//! quantizes, so this is an exact f32-vs-f32 compare modulo GPU reduction order)
//! and asserts the GPU encoder output and decoder logits match the CPU engine.
//! Runs on whatever adapter wgpu picks (Metal locally, Vulkan/DX12 in CI); skips
//! cleanly when no GPU is available. Gated on the `wgpu` feature.

#![cfg(feature = "wgpu")]

use std::collections::HashMap;

use sapient_core::Tensor;
use sapient_hub::whisper_config::WhisperConfig;
use sapient_models::forward::{WhisperForward, WhisperWgpuEngine};

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

fn rand_tensor(seed: usize, dims: Vec<usize>) -> Tensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i + seed * 977) as f32 * 0.123 + 0.7).sin() * 0.2)
        .collect();
    Tensor::from_f32(&data, dims).unwrap()
}

fn ln_weights(w: &mut HashMap<String, Tensor>, seed: usize, prefix: &str) {
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

fn mel_input() -> Tensor {
    rand_tensor(7, vec![1, N_MELS, 16])
}

#[test]
fn wgpu_matches_cpu_encode_and_decode() {
    let weights = build_weights();

    // CPU reference.
    let mut cpu = WhisperForward::from_weights(cfg(), weights.clone()).unwrap();
    let cpu_ctx = cpu.encode(&mel_input()).unwrap();
    cpu.set_audio_context(&cpu_ctx).unwrap();

    // GPU engine (skip if no adapter).
    let mut gpu = match WhisperWgpuEngine::from_weights(cfg(), weights) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no wgpu adapter — skipping ({e})");
            return;
        }
    };
    let gpu_ctx = gpu.encode(&mel_input()).unwrap();

    // Encoder output must match.
    let (cd, gd) = (cpu_ctx.as_f32_slice(), gpu_ctx.as_f32_slice());
    assert_eq!(cd.len(), gd.len());
    let enc_err = cd
        .iter()
        .zip(gd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(enc_err < 5e-3, "encoder output diverged: max_err={enc_err}");

    // Decoder logits over a forced prompt must match.
    let tokens = [3u32, 11, 5];
    let cpu_logits = cpu.decode_step(&tokens).unwrap();
    let gpu_logits = gpu.decode_step(&tokens).unwrap();
    let dec_err = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(dec_err < 5e-3, "decoder logits diverged: max_err={dec_err}");

    // Argmax (the only thing greedy decoding depends on) must agree.
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };
    assert_eq!(amax(&cpu_logits), amax(&gpu_logits), "argmax disagrees");
}
