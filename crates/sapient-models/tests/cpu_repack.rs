//! Engine-level gate for the Q4_K_R4 multi-row repack: a pure-CPU `LlamaForward`
//! must produce IDENTICAL logits with the repack enabled (the aarch64 default)
//! and disabled (`SAPIENT_NO_REPACK=1`) — the repacked kernel's per-row math is
//! bit-for-bit the single-row kernel's, so any divergence is a layout bug.
//! On non-aarch64 the repack never engages and the test degenerates to
//! self-consistency (still a valid smoke test of the Q4_K CPU path).

use std::collections::HashMap;

use sapient_core::{DType, Shape, Tensor};
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_models::forward::LlamaForward;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Random-but-valid raw Q4_K blocks with small positive d/dmin (magnitudes ~0.1).
fn q4_k_random_tensor(shape: [usize; 2], next: &mut dyn FnMut() -> f32) -> Tensor {
    let numel = shape[0] * shape[1];
    assert_eq!(numel % 256, 0);
    let mut blocks = Vec::with_capacity(numel / 256 * 144);
    for _ in 0..numel / 256 {
        let d = half::f16::from_f32(1.0e-4 * (1.0 + next().abs()));
        let dmin = half::f16::from_f32(1.0e-4 * (1.0 + next().abs()));
        blocks.extend_from_slice(&d.to_le_bytes());
        blocks.extend_from_slice(&dmin.to_le_bytes());
        for _ in 0..140 {
            blocks.push((next().abs() * 255.0) as u8);
        }
    }
    Tensor::from_quant_bytes(&blocks, shape.to_vec(), DType::Q4_K).unwrap()
}

fn tiny_q4_k_llama() -> (ModelInfo, HashMap<String, Tensor>) {
    // All matmul dims are multiples of 4 (and k of 256) so every 2-D weight is
    // repack-eligible except the (skipped) embedding.
    let hidden = 256usize;
    let n_heads = 4usize;
    let n_kv = 2usize;
    let head_dim = 24usize; // f32 CPU KV cache (not a multiple of 32)
    let inter = 512usize;
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

    let mut next = lcg(0x4B1D);
    let mut w = HashMap::new();
    let norm = |dim: usize, n: &mut dyn FnMut() -> f32| {
        let data: Vec<f32> = (0..dim).map(|_| 1.0 + n() * 0.05).collect();
        Tensor::from_f32_vec(data, Shape::new([dim])).unwrap()
    };

    let qd = n_heads * head_dim; // 96
    let kvd = n_kv * head_dim; // 48
    w.insert(
        "model.embed_tokens.weight".into(),
        q4_k_random_tensor([vocab, hidden], &mut next),
    );
    w.insert("model.norm.weight".into(), norm(hidden, &mut next));
    w.insert(
        "lm_head.weight".into(),
        q4_k_random_tensor([vocab, hidden], &mut next),
    );
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
        for (suffix, rows, cols) in [
            ("self_attn.q_proj", qd, hidden),
            ("self_attn.k_proj", kvd, hidden),
            ("self_attn.v_proj", kvd, hidden),
        ] {
            w.insert(
                format!("{p}.{suffix}.weight"),
                q4_k_random_tensor([rows, cols], &mut next),
            );
        }
        // o_proj has k = 96 (not %256) → stays plain Q4_K? No — k must be %256
        // for Q4_K at all; use f32 for o_proj like real mixed files use other
        // quants there.
        let o: Vec<f32> = (0..hidden * qd).map(|_| next() * 0.1).collect();
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            Tensor::from_f32_vec(o, Shape::new([hidden, qd])).unwrap(),
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            q4_k_random_tensor([inter, hidden], &mut next),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            q4_k_random_tensor([inter, hidden], &mut next),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            q4_k_random_tensor([hidden, inter], &mut next),
        );
    }
    (info, w)
}

#[test]
fn repacked_engine_logits_are_bit_identical() {
    let (info, weights) = tiny_q4_k_llama();

    // Engine A: repack disabled via env (read at engine construction).
    std::env::set_var("SAPIENT_NO_REPACK", "1");
    let mut plain = LlamaForward::from_weights(info.clone(), weights.clone()).expect("plain");
    std::env::remove_var("SAPIENT_NO_REPACK");
    // Engine B: default path (repacks on aarch64+dotprod pure-CPU).
    let mut repacked = LlamaForward::from_weights(info, weights).expect("repacked");

    let tokens: Vec<u32> = vec![2, 40, 17, 5, 33, 8, 21];
    let a = plain.forward_logits(&tokens, false).unwrap();
    let b = repacked.forward_logits(&tokens, false).unwrap();
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "logit {i} differs: {x} vs {y} — repack layout bug"
        );
    }

    // And through the KV cache on a decode step.
    let t = a
        .iter()
        .enumerate()
        .max_by(|p, q| p.1.partial_cmp(q.1).unwrap())
        .unwrap()
        .0 as u32;
    let a2 = plain.forward_logits(&[t], true).unwrap();
    let b2 = repacked.forward_logits(&[t], true).unwrap();
    for (x, y) in a2.iter().zip(&b2) {
        assert_eq!(x.to_bits(), y.to_bits(), "decode-step logits differ");
    }
}
