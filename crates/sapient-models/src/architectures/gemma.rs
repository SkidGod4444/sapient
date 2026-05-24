//! GemmaForCausalLM graph builder. Covers: Gemma, Gemma 2.

use anyhow::Result;
use ordered_float::OrderedFloat;
use sapient_hub::model_info::ModelInfo;
use sapient_ir::{graph::Graph, op::OpType};

pub fn build(info: &ModelInfo) -> Result<Graph> {
    let mut g = Graph::new(format!("gemma_{}", info.model_type));
    let input_ids = g.add_input("input_ids", None, None);
    let _attn_mask = g.add_input("attention_mask", None, None);

    // Gemma scales embeddings by sqrt(hidden_size).
    let embed = g.add_op(
        OpType::Embedding {
            vocab_size: info.vocab_size,
            dim: info.hidden_size,
        },
        vec![input_ids],
        1,
        Some("embed_tokens".into()),
    );
    // Scale constant is baked in at graph-build time (Mul by scalar).
    let scale = g.add_constant(
        sapient_core::Tensor::scalar_f32((info.hidden_size as f32).sqrt()).unwrap(),
        Some("embed_scale".into()),
    );
    let mut x = g.add_op(
        OpType::Mul,
        vec![embed, scale],
        1,
        Some("embed_scaled".into()),
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

        // GeGLU FFN (GELU gate, unlike Llama's SwiGLU).
        let ff_norm = g.add_op(
            OpType::RmsNorm { epsilon: eps },
            vec![x1],
            1,
            Some(format!("{p}.post_feedforward_layernorm")),
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
        let gate_act = g.add_op(OpType::Gelu, vec![gate], 1, Some(format!("{p}.mlp.gelu")));
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
    const CFG: &str = r#"{"architectures":["GemmaForCausalLM"],"model_type":"gemma","vocab_size":256000,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":128,"max_position_embeddings":512,"rms_norm_eps":1e-6,"hidden_act":"gelu","rope_theta":10000.0}"#;
    #[test]
    fn tiny_gemma_builds() {
        let info = sapient_hub::model_info::ModelInfo::from_json_str(CFG).unwrap();
        let g = build(&info).unwrap();
        assert!(g.node_count() > 5);
    }
}
