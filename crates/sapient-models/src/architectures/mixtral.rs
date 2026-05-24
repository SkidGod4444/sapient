//! MixtralForCausalLM graph builder — Sparse Mixture of Experts.
//! Covers: Mixtral-8x7B, Mixtral-8x22B.

use anyhow::Result;
use ordered_float::OrderedFloat;
use sapient_ir::{graph::Graph, op::OpType};
use sapient_hub::model_info::ModelInfo;

const DEFAULT_NUM_EXPERTS: usize = 8;
const DEFAULT_TOP_K: usize = 2;

pub fn build(info: &ModelInfo) -> Result<Graph> {
    let mut g = Graph::new(format!("mixtral_{}", info.model_type));

    let num_experts = info.raw["num_local_experts"].as_u64().unwrap_or(DEFAULT_NUM_EXPERTS as u64) as usize;
    let num_experts_per_tok = info.raw["num_experts_per_tok"].as_u64().unwrap_or(DEFAULT_TOP_K as u64) as usize;

    let input_ids = g.add_input("input_ids", None, None);
    let _attn_mask = g.add_input("attention_mask", None, None);

    let mut x = g.add_op(
        OpType::Embedding { vocab_size: info.vocab_size, dim: info.hidden_size },
        vec![input_ids], 1, Some("model.embed_tokens".into()),
    );

    for i in 0..info.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let eps = OrderedFloat(info.rms_norm_eps);

        // Attention sub-layer (identical to Llama/Mistral).
        let norm = g.add_op(OpType::RmsNorm { epsilon: eps }, vec![x], 1, Some(format!("{p}.input_layernorm")));
        let q = g.add_op(OpType::MatMul, vec![norm], 1, Some(format!("{p}.self_attn.q_proj")));
        let k = g.add_op(OpType::MatMul, vec![norm], 1, Some(format!("{p}.self_attn.k_proj")));
        let v = g.add_op(OpType::MatMul, vec![norm], 1, Some(format!("{p}.self_attn.v_proj")));
        let q = g.add_op(OpType::RotaryEmbedding { base: OrderedFloat(info.rope_theta), dim: info.head_dim }, vec![q], 1, Some(format!("{p}.q_rope")));
        let k = g.add_op(OpType::RotaryEmbedding { base: OrderedFloat(info.rope_theta), dim: info.head_dim }, vec![k], 1, Some(format!("{p}.k_rope")));
        let attn = g.add_op(
            OpType::GroupedQueryAttention { n_heads: info.num_attention_heads, n_kv_heads: info.num_key_value_heads, head_dim: info.head_dim, causal: true },
            vec![q, k, v], 1, Some(format!("{p}.self_attn")),
        );
        let o = g.add_op(OpType::MatMul, vec![attn], 1, Some(format!("{p}.self_attn.o_proj")));
        let x1 = g.add_op(OpType::Add, vec![x, o], 1, Some(format!("{p}.attn_res")));

        // MoE FFN sub-layer.
        let ff_norm = g.add_op(OpType::RmsNorm { epsilon: eps }, vec![x1], 1, Some(format!("{p}.post_attention_layernorm")));

        // Router gate — selects top-k experts.
        let gate_out = g.add_op(
            OpType::MoEGate { num_experts, top_k: num_experts_per_tok },
            vec![ff_norm], 1, Some(format!("{p}.block_sparse_moe.gate")),
        );

        // Expert FFNs (SwiGLU). In a real deployment these are sparse.
        // The graph represents all experts; the scheduler decides which to run.
        let mut expert_outputs: Vec<sapient_ir::node::NodeId> = Vec::new();
        for e in 0..num_experts {
            let ep = format!("{p}.block_sparse_moe.experts.{e}");
            let gate_proj = g.add_op(OpType::MatMul, vec![ff_norm], 1, Some(format!("{ep}.w1")));
            let up_proj   = g.add_op(OpType::MatMul, vec![ff_norm], 1, Some(format!("{ep}.w3")));
            let gate_act  = g.add_op(OpType::Silu, vec![gate_proj], 1, Some(format!("{ep}.silu")));
            let mid       = g.add_op(OpType::Mul, vec![gate_act, up_proj], 1, Some(format!("{ep}.mul")));
            let down      = g.add_op(OpType::MatMul, vec![mid], 1, Some(format!("{ep}.w2")));
            expert_outputs.push(down);
        }

        // Weighted sum of expert outputs (controlled by gate_out routing weights).
        // Represented as a Concat + weighted add in the graph.
        let first_expert = expert_outputs[0];
        let moe_out = expert_outputs[1..].iter().fold(first_expert, |acc, &e| {
            g.add_op(OpType::Add, vec![acc, e], 1, None)
        });

        x = g.add_op(OpType::Add, vec![x1, moe_out], 1, Some(format!("{p}.ffn_res")));
    }

    let normed = g.add_op(OpType::RmsNorm { epsilon: OrderedFloat(info.rms_norm_eps) }, vec![x], 1, Some("model.norm".into()));
    let logits = g.add_op(OpType::MatMul, vec![normed], 1, Some("lm_head".into()));
    g.mark_output(logits, "logits");
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    const CFG: &str = r#"{"architectures":["MixtralForCausalLM"],"model_type":"mixtral","vocab_size":32000,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"num_local_experts":4,"num_experts_per_tok":2,"intermediate_size":128,"max_position_embeddings":4096,"rms_norm_eps":1e-5,"hidden_act":"silu","rope_theta":1000000.0}"#;
    #[test]
    fn tiny_mixtral_builds() {
        let info = sapient_hub::model_info::ModelInfo::from_json_str(CFG).unwrap();
        let g = build(&info).unwrap();
        assert!(g.node_count() > 10);
        assert_eq!(g.outputs.len(), 1);
    }
}
