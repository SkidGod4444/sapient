//! End-to-end coherence: the wgpu GPU forward engine must produce the same logits as
//! the proven CPU `LlamaForward` for an identical (synthetic) Llama-family model. This
//! is the gating test for the cross-platform GPU path — if a kernel (RoPE axis, GQA
//! mapping, KV-cache layout, reduction) were subtly wrong, the per-token logits would
//! diverge and the model would emit token-salad (exactly the class of bug this guards).
//!
//! Pure f32 throughout: F32 weights aren't online-quantized by the CPU path, and a
//! head_dim not divisible by 32 keeps the CPU KV cache in F32 too — so the only
//! differences are floating-point reduction order (tiny).
#![cfg(feature = "wgpu")]

use std::collections::HashMap;

use sapient_core::{Shape, Tensor};
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_models::forward::{LlamaForward, WgpuForwardEngine};

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // ~U(-1,1)
    }
}

fn tiny_llama() -> (ModelInfo, HashMap<String, Tensor>) {
    let hidden = 64usize;
    let n_heads = 4usize;
    let n_kv = 2usize;
    let head_dim = 16usize; // not a multiple of 32 → CPU keeps an F32 KV cache
    let inter = 128usize;
    let layers = 2usize;
    let vocab = 48usize;

    let info = ModelInfo {
        arch: ArchType::Llama,
        model_type: "llama".into(),
        vocab_size: vocab,
        hidden_size: hidden,
        num_hidden_layers: layers,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv,
        intermediate_size: inter,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-5,
        hidden_act: "silu".into(),
        rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        head_dim,
        raw: serde_json::Value::Null,
    };

    let mut next = lcg(0xC0FFEE);
    let mut w = HashMap::new();
    let mat = |rows: usize, cols: usize, scale: f32, n: &mut dyn FnMut() -> f32| {
        let data: Vec<f32> = (0..rows * cols).map(|_| n() * scale).collect();
        Tensor::from_f32_vec(data, Shape::new([rows, cols])).unwrap()
    };
    let norm = |dim: usize, n: &mut dyn FnMut() -> f32| {
        let data: Vec<f32> = (0..dim).map(|_| 1.0 + n() * 0.05).collect();
        Tensor::from_f32_vec(data, Shape::new([dim])).unwrap()
    };

    w.insert(
        "model.embed_tokens.weight".into(),
        mat(vocab, hidden, 0.1, &mut next),
    );
    w.insert("model.norm.weight".into(), norm(hidden, &mut next));
    w.insert("lm_head.weight".into(), mat(vocab, hidden, 0.1, &mut next));

    let qd = n_heads * head_dim;
    let kvd = n_kv * head_dim;
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        w.insert(
            format!("{p}.input_layernorm.weight"),
            norm(hidden, &mut next),
        );
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            norm(hidden, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            mat(qd, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            mat(kvd, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            mat(kvd, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            mat(hidden, qd, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            mat(inter, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            mat(inter, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            mat(hidden, inter, 0.1, &mut next),
        );
    }
    (info, w)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
fn wgpu_logits_match_cpu_llama() {
    let (info, weights) = tiny_llama();

    let mut cpu =
        LlamaForward::from_weights(info.clone(), weights.clone()).expect("build CPU LlamaForward");
    let mut gpu = match WgpuForwardEngine::from_weights(info, weights) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no wgpu GPU adapter ({e}) — skipping coherence test");
            return;
        }
    };

    let tokens: Vec<u32> = vec![1, 5, 9, 3, 7, 2, 11];

    // Full-prompt last-token logits.
    let cpu_logits = cpu.forward_logits(&tokens, false).unwrap();
    let gpu_logits = gpu.forward_logits(&tokens, false).unwrap();
    assert_eq!(cpu_logits.len(), gpu_logits.len());

    let max_err = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 5e-3,
        "wgpu vs cpu prompt logits max_err={max_err} (argmax cpu={} gpu={})",
        argmax(&cpu_logits),
        argmax(&gpu_logits)
    );
    assert_eq!(
        argmax(&cpu_logits),
        argmax(&gpu_logits),
        "greedy next-token must match"
    );

    // Incremental decode (use_cache=true) must match a fresh CPU run of prompt+token.
    let next_tok = argmax(&gpu_logits) as u32;
    let gpu_step = gpu.forward_logits(&[next_tok], true).unwrap();
    let mut full = tokens.clone();
    full.push(next_tok);
    let cpu_step = cpu.forward_logits(&full, false).unwrap();
    let step_err = cpu_step
        .iter()
        .zip(&gpu_step)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        step_err < 5e-3,
        "wgpu incremental-decode logits max_err={step_err}"
    );
    assert_eq!(
        argmax(&cpu_step),
        argmax(&gpu_step),
        "decode greedy must match"
    );
}
