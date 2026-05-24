//! GPT2LMHeadModel graph builder. Covers: GPT-2, CodeGen, GPT-J.

use anyhow::Result;
use ordered_float::OrderedFloat;
use sapient_ir::{graph::Graph, op::OpType};
use sapient_hub::model_info::ModelInfo;

pub fn build(info: &ModelInfo) -> Result<Graph> {
    let mut g = Graph::new(format!("gpt2_{}", info.model_type));
    let input_ids = g.add_input("input_ids", None, None);
    let position_ids = g.add_input("position_ids", None, None);

    // Token + position embeddings.
    let tok_emb = g.add_op(OpType::Embedding { vocab_size: info.vocab_size, dim: info.hidden_size }, vec![input_ids], 1, Some("wte".into()));
    let pos_emb = g.add_op(OpType::Embedding { vocab_size: info.max_position_embeddings, dim: info.hidden_size }, vec![position_ids], 1, Some("wpe".into()));
    let mut x = g.add_op(OpType::Add, vec![tok_emb, pos_emb], 1, Some("embed".into()));

    for i in 0..info.num_hidden_layers {
        let p = format!("h.{i}");
        let eps = OrderedFloat(info.rms_norm_eps.max(1e-5));

        // Pre-norm → MHA → Linear → residual.
        let norm1 = g.add_op(OpType::LayerNorm { axis: -1, epsilon: eps }, vec![x], 1, Some(format!("{p}.ln_1")));
        let q = g.add_op(OpType::MatMul, vec![norm1], 1, Some(format!("{p}.attn.q")));
        let k = g.add_op(OpType::MatMul, vec![norm1], 1, Some(format!("{p}.attn.k")));
        let v = g.add_op(OpType::MatMul, vec![norm1], 1, Some(format!("{p}.attn.v")));
        let attn = g.add_op(
            OpType::MultiHeadAttention { num_heads: info.num_attention_heads, head_dim: info.head_dim, causal: true, scale: None },
            vec![q, k, v], 1, Some(format!("{p}.attn.mha")),
        );
        let proj = g.add_op(OpType::MatMul, vec![attn], 1, Some(format!("{p}.attn.c_proj")));
        let x1 = g.add_op(OpType::Add, vec![x, proj], 1, Some(format!("{p}.attn_res")));

        // Pre-norm → FFN → residual.
        let norm2 = g.add_op(OpType::LayerNorm { axis: -1, epsilon: eps }, vec![x1], 1, Some(format!("{p}.ln_2")));
        let ff1  = g.add_op(OpType::MatMul, vec![norm2], 1, Some(format!("{p}.mlp.c_fc")));
        let act  = g.add_op(OpType::Gelu, vec![ff1], 1, Some(format!("{p}.mlp.act")));
        let ff2  = g.add_op(OpType::MatMul, vec![act], 1, Some(format!("{p}.mlp.c_proj")));
        x = g.add_op(OpType::Add, vec![x1, ff2], 1, Some(format!("{p}.ffn_res")));
    }

    let normed = g.add_op(OpType::LayerNorm { axis: -1, epsilon: OrderedFloat(1e-5) }, vec![x], 1, Some("ln_f".into()));
    let logits = g.add_op(OpType::MatMul, vec![normed], 1, Some("lm_head".into()));
    g.mark_output(logits, "logits");
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    const CFG: &str = r#"{"architectures":["GPT2LMHeadModel"],"model_type":"gpt2","vocab_size":50257,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":256,"max_position_embeddings":1024,"rms_norm_eps":1e-5,"hidden_act":"gelu","rope_theta":10000.0}"#;
    #[test]
    fn tiny_gpt2_builds() {
        let info = sapient_hub::model_info::ModelInfo::from_json_str(CFG).unwrap();
        let g = build(&info).unwrap();
        assert!(g.node_count() > 5);
    }
}
