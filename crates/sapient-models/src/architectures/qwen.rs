//! Qwen2ForCausalLM graph builder. Covers: Qwen, Qwen2, Qwen2.5, Qwen-Coder.

use anyhow::Result;
use ordered_float::OrderedFloat;
use sapient_hub::model_info::ModelInfo;
use sapient_ir::{graph::Graph, op::OpType};

pub fn build(info: &ModelInfo) -> Result<Graph> {
    let mut g = Graph::new(format!("qwen_{}", info.model_type));
    let input_ids = g.add_input("input_ids", None, None);
    let _attn_mask = g.add_input("attention_mask", None, None);

    let mut x = g.add_op(
        OpType::Embedding {
            vocab_size: info.vocab_size,
            dim: info.hidden_size,
        },
        vec![input_ids],
        1,
        Some("model.embed_tokens".into()),
    );

    for i in 0..info.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let eps = OrderedFloat(info.rms_norm_eps);

        let norm = g.add_op(
            OpType::RmsNorm { epsilon: eps },
            vec![x],
            1,
            Some(format!("{p}.input_layernorm")),
        );

        // Qwen uses QKV with separate biases (represented as Add nodes after MatMul).
        let q = g.add_op(
            OpType::MatMul,
            vec![norm],
            1,
            Some(format!("{p}.self_attn.q_proj")),
        );
        let k = g.add_op(
            OpType::MatMul,
            vec![norm],
            1,
            Some(format!("{p}.self_attn.k_proj")),
        );
        let v = g.add_op(
            OpType::MatMul,
            vec![norm],
            1,
            Some(format!("{p}.self_attn.v_proj")),
        );
        // QKV biases (baked into graph as Add with a trainable bias constant input).
        let q = g.add_op(
            OpType::Add,
            vec![q, norm],
            1,
            Some(format!("{p}.self_attn.q_bias")),
        );
        let k = g.add_op(
            OpType::Add,
            vec![k, norm],
            1,
            Some(format!("{p}.self_attn.k_bias")),
        );

        let q = g.add_op(
            OpType::RotaryEmbedding {
                base: OrderedFloat(info.rope_theta),
                dim: info.head_dim,
            },
            vec![q],
            1,
            Some(format!("{p}.q_rope")),
        );
        let k = g.add_op(
            OpType::RotaryEmbedding {
                base: OrderedFloat(info.rope_theta),
                dim: info.head_dim,
            },
            vec![k],
            1,
            Some(format!("{p}.k_rope")),
        );
        let attn = g.add_op(
            OpType::GroupedQueryAttention {
                n_heads: info.num_attention_heads,
                n_kv_heads: info.num_key_value_heads,
                head_dim: info.head_dim,
                causal: true,
            },
            vec![q, k, v],
            1,
            Some(format!("{p}.self_attn")),
        );
        let o = g.add_op(
            OpType::MatMul,
            vec![attn],
            1,
            Some(format!("{p}.self_attn.o_proj")),
        );
        let x1 = g.add_op(OpType::Add, vec![x, o], 1, Some(format!("{p}.attn_res")));

        // SwiGLU FFN (same as Llama).
        let ff_norm = g.add_op(
            OpType::RmsNorm { epsilon: eps },
            vec![x1],
            1,
            Some(format!("{p}.post_attention_layernorm")),
        );
        let gate = g.add_op(
            OpType::MatMul,
            vec![ff_norm],
            1,
            Some(format!("{p}.mlp.gate_proj")),
        );
        let up = g.add_op(
            OpType::MatMul,
            vec![ff_norm],
            1,
            Some(format!("{p}.mlp.up_proj")),
        );
        let gate_act = g.add_op(OpType::Silu, vec![gate], 1, Some(format!("{p}.mlp.silu")));
        let mid = g.add_op(
            OpType::Mul,
            vec![gate_act, up],
            1,
            Some(format!("{p}.mlp.mul")),
        );
        let down = g.add_op(
            OpType::MatMul,
            vec![mid],
            1,
            Some(format!("{p}.mlp.down_proj")),
        );
        x = g.add_op(OpType::Add, vec![x1, down], 1, Some(format!("{p}.ffn_res")));
    }

    let normed = g.add_op(
        OpType::RmsNorm {
            epsilon: OrderedFloat(info.rms_norm_eps),
        },
        vec![x],
        1,
        Some("model.norm".into()),
    );
    let logits = g.add_op(OpType::MatMul, vec![normed], 1, Some("lm_head".into()));
    g.mark_output(logits, "logits");
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    const CFG: &str = r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","vocab_size":151936,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":128,"max_position_embeddings":4096,"rms_norm_eps":1e-6,"hidden_act":"silu","rope_theta":1000000.0}"#;
    #[test]
    fn tiny_qwen_builds() {
        let info = sapient_hub::model_info::ModelInfo::from_json_str(CFG).unwrap();
        let g = build(&info).unwrap();
        assert!(g.node_count() > 5);
    }
}
