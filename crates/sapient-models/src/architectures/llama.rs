//! LlamaForCausalLM graph builder.
//!
//! Covers: Llama 1/2/3, Mistral, Mistral-Instruct, CodeLlama, Vicuna,
//!         WizardLM, Orca-2, OpenHermes, Zephyr, and any derivative.
//!
//! Architecture:
//!   embed_tokens → N × DecoderLayer → rms_norm → lm_head → logits
//!
//! DecoderLayer:
//!   input_rms_norm → QKV projections → RoPE → GQA → out_proj
//!   + gate/up projections → SiLU+mul → down_proj (SwiGLU FFN)
//!   + residual connections

use anyhow::Result;
use ordered_float::OrderedFloat;
use sapient_ir::{
    graph::Graph,
    op::OpType,
};
use sapient_hub::model_info::ModelInfo;

pub fn build(info: &ModelInfo) -> Result<Graph> {
    let mut g = Graph::new(format!("llama_{}", info.model_type));

    // ── Inputs ────────────────────────────────────────────────────────────────
    let input_ids = g.add_input("input_ids", None, None);          // (batch, seq)
    let _attn_mask = g.add_input("attention_mask", None, None);    // (batch, seq) optional
    let _pos_ids   = g.add_input("position_ids", None, None);      // (batch, seq) optional

    // ── Token Embedding (embed_tokens) ────────────────────────────────────────
    let mut x = g.add_op(
        OpType::Embedding { vocab_size: info.vocab_size, dim: info.hidden_size },
        vec![input_ids],
        1,
        Some("embed_tokens".into()),
    );

    // ── N decoder layers ─────────────────────────────────────────────────────
    for layer_idx in 0..info.num_hidden_layers {
        x = build_decoder_layer(&mut g, x, info, layer_idx);
    }

    // ── Final RMSNorm ─────────────────────────────────────────────────────────
    let normed = g.add_op(
        OpType::RmsNorm { epsilon: OrderedFloat(info.rms_norm_eps) },
        vec![x],
        1,
        Some("norm".into()),
    );

    // ── LM Head (linear projection → vocab) ───────────────────────────────────
    let logits = g.add_op(
        OpType::MatMul,
        vec![normed],
        1,
        Some("lm_head".into()),
    );

    g.mark_output(logits, "logits");

    Ok(g)
}

/// Build one Llama decoder block and return its output node ID.
fn build_decoder_layer(
    g:   &mut Graph,
    x:   sapient_ir::node::NodeId,
    info: &ModelInfo,
    idx:  usize,
) -> sapient_ir::node::NodeId {
    let pfx = format!("layers.{idx}");
    let eps = OrderedFloat(info.rms_norm_eps);

    // ── Self-attention sub-layer ───────────────────────────────────────────────
    // Pre-norm.
    let attn_norm = g.add_op(
        OpType::RmsNorm { epsilon: eps },
        vec![x],
        1,
        Some(format!("{pfx}.input_layernorm")),
    );

    // QKV projections (separate, as in HF implementation).
    let q = g.add_op(OpType::MatMul, vec![attn_norm], 1, Some(format!("{pfx}.self_attn.q_proj")));
    let k = g.add_op(OpType::MatMul, vec![attn_norm], 1, Some(format!("{pfx}.self_attn.k_proj")));
    let v = g.add_op(OpType::MatMul, vec![attn_norm], 1, Some(format!("{pfx}.self_attn.v_proj")));

    // RoPE applied to Q and K.
    let q_rope = g.add_op(
        OpType::RotaryEmbedding { base: OrderedFloat(info.rope_theta), dim: info.head_dim },
        vec![q],
        1,
        Some(format!("{pfx}.self_attn.q_rope")),
    );
    let k_rope = g.add_op(
        OpType::RotaryEmbedding { base: OrderedFloat(info.rope_theta), dim: info.head_dim },
        vec![k],
        1,
        Some(format!("{pfx}.self_attn.k_rope")),
    );

    // GQA (or MHA if n_kv_heads == n_heads).
    let attn_out = g.add_op(
        OpType::GroupedQueryAttention {
            n_heads:    info.num_attention_heads,
            n_kv_heads: info.num_key_value_heads,
            head_dim:   info.head_dim,
            causal:     true,
        },
        vec![q_rope, k_rope, v],
        1,
        Some(format!("{pfx}.self_attn.gqa")),
    );

    // Output projection.
    let o_proj = g.add_op(
        OpType::MatMul,
        vec![attn_out],
        1,
        Some(format!("{pfx}.self_attn.o_proj")),
    );

    // Residual.
    let x = g.add_op(OpType::Add, vec![x, o_proj], 1, Some(format!("{pfx}.attn_residual")));

    // ── Feed-forward sub-layer (SwiGLU) ───────────────────────────────────────
    // Pre-norm.
    let ffn_norm = g.add_op(
        OpType::RmsNorm { epsilon: eps },
        vec![x],
        1,
        Some(format!("{pfx}.post_attention_layernorm")),
    );

    // Gate and Up projections.
    let gate = g.add_op(OpType::MatMul, vec![ffn_norm], 1, Some(format!("{pfx}.mlp.gate_proj")));
    let up   = g.add_op(OpType::MatMul, vec![ffn_norm], 1, Some(format!("{pfx}.mlp.up_proj")));

    // SwiGLU: SiLU(gate) * up.
    let gate_act = g.add_op(OpType::Silu, vec![gate], 1, Some(format!("{pfx}.mlp.silu")));
    let ffn_mid  = g.add_op(OpType::Mul, vec![gate_act, up], 1, Some(format!("{pfx}.mlp.gate_mul")));

    // Down projection.
    let down = g.add_op(OpType::MatMul, vec![ffn_mid], 1, Some(format!("{pfx}.mlp.down_proj")));

    // Residual.
    g.add_op(OpType::Add, vec![x, down], 1, Some(format!("{pfx}.ffn_residual")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_hub::model_info::ModelInfo;

    const TINY_LLAMA_CFG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "rope_theta": 10000.0
    }"#;

    #[test]
    fn tiny_llama_builds() {
        let info = ModelInfo::from_json_str(TINY_LLAMA_CFG).unwrap();
        let g = build(&info).unwrap();
        assert!(g.node_count() > 10, "graph should have many nodes");
        assert_eq!(g.outputs.len(), 1, "should have one output (logits)");
    }
}
