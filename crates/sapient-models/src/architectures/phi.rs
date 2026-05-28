//! PhiForCausalLM graph builder.
//! Covers: Phi-1, Phi-1.5, Phi-2, Phi-3, Phi-3.5-Mini.

use anyhow::Result;
use ordered_float::OrderedFloat;
use sapient_hub::model_info::ModelInfo;
use sapient_ir::{graph::Graph, op::OpType};

pub fn build(info: &ModelInfo) -> Result<Graph> {
    let mut g = Graph::new(format!("phi_{}", info.model_type));
    let input_ids = g.add_input("input_ids", None, None);
    let _attn_mask = g.add_input("attention_mask", None, None);

    let mut x = g.add_op(
        OpType::Embedding {
            vocab_size: info.vocab_size,
            dim: info.hidden_size,
        },
        vec![input_ids],
        1,
        Some("embed_tokens".into()),
    );

    for i in 0..info.num_hidden_layers {
        let p = format!("layers.{i}");
        let eps = OrderedFloat(info.rms_norm_eps);

        // Pre-norm (Phi uses LayerNorm, not RMSNorm).
        let norm = g.add_op(
            OpType::LayerNorm {
                axis: -1,
                epsilon: eps,
            },
            vec![x],
            1,
            Some(format!("{p}.input_layernorm")),
        );

        // MHA with RoPE.
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
            Some(format!("{p}.self_attn.q_rope")),
        );
        let k = g.add_op(
            OpType::RotaryEmbedding {
                base: OrderedFloat(info.rope_theta),
                dim: info.head_dim,
            },
            vec![k],
            1,
            Some(format!("{p}.self_attn.k_rope")),
        );
        let attn = g.add_op(
            OpType::MultiHeadAttention {
                num_heads: info.num_attention_heads,
                head_dim: info.head_dim,
                causal: true,
                scale: None,
            },
            vec![q, k, v],
            1,
            Some(format!("{p}.self_attn.mha")),
        );
        let o = g.add_op(
            OpType::MatMul,
            vec![attn],
            1,
            Some(format!("{p}.self_attn.out_proj")),
        );

        let is_parallel = info.model_type == "phi";

        if is_parallel {
            // Parallel structure (Phi-1, Phi-2)
            let ff1 = g.add_op(
                OpType::MatMul,
                vec![norm],
                1,
                Some(format!("{p}.mlp.fc1")),
            );
            let act = g.add_op(OpType::Gelu, vec![ff1], 1, Some(format!("{p}.mlp.gelu")));
            let ff2 = g.add_op(OpType::MatMul, vec![act], 1, Some(format!("{p}.mlp.fc2")));
            
            let parallel_res = g.add_op(OpType::Add, vec![o, ff2], 1, Some(format!("{p}.parallel_res")));
            x = g.add_op(OpType::Add, vec![x, parallel_res], 1, Some(format!("{p}.res")));
        } else {
            // Sequential structure (Phi-3)
            let x_attn = g.add_op(OpType::Add, vec![x, o], 1, Some(format!("{p}.attn_res")));
            let norm2 = g.add_op(
                OpType::LayerNorm {
                    axis: -1,
                    epsilon: eps,
                },
                vec![x_attn],
                1,
                Some(format!("{p}.post_attention_layernorm")),
            );
            
            let ff1 = g.add_op(
                OpType::MatMul,
                vec![norm2],
                1,
                Some(format!("{p}.mlp.gate_up_proj")),
            );
            let act = g.add_op(OpType::Silu, vec![ff1], 1, Some(format!("{p}.mlp.act")));
            let ff2 = g.add_op(OpType::MatMul, vec![act], 1, Some(format!("{p}.mlp.down_proj")));
            x = g.add_op(
                OpType::Add,
                vec![x_attn, ff2],
                1,
                Some(format!("{p}.ffn_res")),
            );
        }
    }

    let normed = g.add_op(
        OpType::LayerNorm {
            axis: -1,
            epsilon: OrderedFloat(info.rms_norm_eps),
        },
        vec![x],
        1,
        Some("final_layernorm".into()),
    );
    let logits = g.add_op(OpType::MatMul, vec![normed], 1, Some("lm_head".into()));
    g.mark_output(logits, "logits");
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    const CFG: &str = r#"{"architectures":["PhiForCausalLM"],"model_type":"phi","vocab_size":1000,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":128,"max_position_embeddings":512,"rms_norm_eps":1e-5,"hidden_act":"gelu","rope_theta":10000.0}"#;
    #[test]
    fn tiny_phi_builds() {
        let info = sapient_hub::model_info::ModelInfo::from_json_str(CFG).unwrap();
        let g = build(&info).unwrap();
        assert!(g.node_count() > 5);
        assert_eq!(g.outputs.len(), 1);
    }
}
