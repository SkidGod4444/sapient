//! End-to-end forward pass with random tiny Llama weights.

use std::collections::HashMap;

use sapient_core::Tensor;
use sapient_hub::model_info::ModelInfo;
use sapient_models::forward::{LlamaForward, LlmBackendKind};

const TINY: &str = r#"{
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 128,
    "head_dim": 8,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "rope_theta": 10000.0
}"#;

fn rand_tensor(shape: Vec<usize>, seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = (seed.wrapping_mul(1103515245).wrapping_add(12345 + i as u64)) as f32;
            (x % 1000.0) / 1000.0 - 0.5
        })
        .collect();
    Tensor::from_f32(&data, shape).unwrap()
}

fn build_tiny_weights(info: &ModelInfo) -> HashMap<String, Tensor> {
    let mut w = HashMap::new();
    let h = info.hidden_size;
    let vocab = info.vocab_size;
    let inter = info.intermediate_size;
    let n_heads = info.num_attention_heads;
    let n_kv = info.num_key_value_heads;
    let head_dim = info.head_dim;

    w.insert(
        "model.embed_tokens.weight".into(),
        rand_tensor(vec![vocab, h], 1),
    );
    w.insert("model.norm.weight".into(), rand_tensor(vec![h], 2));
    w.insert("lm_head.weight".into(), rand_tensor(vec![vocab, h], 3));

    for i in 0..info.num_hidden_layers {
        let p = format!("model.layers.{i}");
        w.insert(
            format!("{p}.input_layernorm.weight"),
            rand_tensor(vec![h], 10 + i as u64),
        );
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            rand_tensor(vec![h], 20 + i as u64),
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor(vec![n_heads * head_dim, h], 30 + i as u64),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(vec![n_kv * head_dim, h], 40 + i as u64),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rand_tensor(vec![n_kv * head_dim, h], 50 + i as u64),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor(vec![h, n_heads * head_dim], 60 + i as u64),
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor(vec![inter, h], 70 + i as u64),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor(vec![inter, h], 80 + i as u64),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor(vec![h, inter], 90 + i as u64),
        );
    }
    w
}

#[test]
fn tiny_llama_forward_produces_logits() {
    let info = ModelInfo::from_json_str(TINY).unwrap();
    let weights = build_tiny_weights(&info);
    let mut fwd = LlamaForward::from_weights(info, weights).unwrap();

    let logits = fwd.forward_logits(&[1u32, 2, 3], false).unwrap();
    assert_eq!(logits.len(), 64);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn tiny_llama_forward_accepts_explicit_cpu_backend() {
    let info = ModelInfo::from_json_str(TINY).unwrap();
    let weights = build_tiny_weights(&info);
    let mut fwd =
        LlamaForward::from_weights_with_backend(info, weights, LlmBackendKind::Cpu).unwrap();

    let logits = fwd.forward_logits(&[1u32, 2, 3], false).unwrap();
    assert_eq!(logits.len(), 64);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn tiny_llama_metal_backend_matches_cpu_reference() {
    let info = ModelInfo::from_json_str(TINY).unwrap();
    let weights = build_tiny_weights(&info);

    let mut cpu =
        LlamaForward::from_weights_with_backend(info.clone(), weights.clone(), LlmBackendKind::Cpu)
            .unwrap();
    let mut metal =
        LlamaForward::from_weights_with_backend(info, weights, LlmBackendKind::Metal).unwrap();

    let cpu_logits = cpu.forward_logits(&[1u32, 2, 3], false).unwrap();
    let metal_logits = metal.forward_logits(&[1u32, 2, 3], false).unwrap();

    for (a, b) in cpu_logits.iter().zip(metal_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "metal backend diverges from CPU reference: {a} vs {b}"
        );
    }
}

#[test]
fn tiny_llama_auto_backend_generates_logits() {
    let info = ModelInfo::from_json_str(TINY).unwrap();
    let weights = build_tiny_weights(&info);
    let mut fwd =
        LlamaForward::from_weights_with_backend(info, weights, LlmBackendKind::Auto).unwrap();

    let logits = fwd.forward_logits(&[1u32, 2, 3], false).unwrap();
    assert_eq!(logits.len(), 64);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn tiny_llama_decode_step_uses_cache() {
    let info = ModelInfo::from_json_str(TINY).unwrap();
    let weights = build_tiny_weights(&info);
    let mut fwd = LlamaForward::from_weights(info, weights).unwrap();

    // Prefill must populate KV cache (use_cache=true), otherwise decode is wrong.
    let _ = fwd.forward_logits(&[1, 2, 3], true).unwrap();
    let logits = fwd.forward_logits(&[4], true).unwrap();
    assert_eq!(logits.len(), 64);
}

#[test]
fn prefill_with_cache_matches_full_forward() {
    let info = ModelInfo::from_json_str(TINY).unwrap();
    let weights = build_tiny_weights(&info);
    let prompt = [1u32, 2, 3, 4];

    let mut cached = LlamaForward::from_weights(info.clone(), weights.clone()).unwrap();
    cached.reset_cache();
    let _ = cached.forward_logits(&prompt, true).unwrap();
    let cached_logits = cached.forward_logits(&[5], true).unwrap();

    let mut full = LlamaForward::from_weights(info, weights).unwrap();
    full.reset_cache();
    let full_logits = full.forward_logits(&[1, 2, 3, 4, 5], false).unwrap();

    for (a, b) in cached_logits.iter().zip(full_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "cached decode logits diverge from full forward: {a} vs {b}"
        );
    }
}
