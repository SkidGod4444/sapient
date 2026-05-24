//! BertForMaskedLM / BertModel graph builder.
//! Covers: BERT, RoBERTa, DistilBERT — for embeddings & classification.

use anyhow::Result;
use ordered_float::OrderedFloat;
use sapient_ir::{graph::Graph, op::OpType};
use sapient_hub::model_info::ModelInfo;

pub fn build(info: &ModelInfo) -> Result<Graph> {
    let mut g = Graph::new(format!("bert_{}", info.model_type));
    let input_ids      = g.add_input("input_ids", None, None);
    let position_ids   = g.add_input("position_ids", None, None);
    let token_type_ids = g.add_input("token_type_ids", None, None);
    let _attn_mask     = g.add_input("attention_mask", None, None);

    // Embeddings: word + position + token_type → LayerNorm.
    let word_emb = g.add_op(OpType::Embedding { vocab_size: info.vocab_size, dim: info.hidden_size }, vec![input_ids], 1, Some("embeddings.word_embeddings".into()));
    let pos_emb  = g.add_op(OpType::Embedding { vocab_size: info.max_position_embeddings, dim: info.hidden_size }, vec![position_ids], 1, Some("embeddings.position_embeddings".into()));
    let tt_emb   = g.add_op(OpType::Embedding { vocab_size: 2, dim: info.hidden_size }, vec![token_type_ids], 1, Some("embeddings.token_type_embeddings".into()));
    let sum1 = g.add_op(OpType::Add, vec![word_emb, pos_emb], 1, Some("embed_sum1".into()));
    let sum2 = g.add_op(OpType::Add, vec![sum1, tt_emb], 1, Some("embed_sum2".into()));
    let eps = OrderedFloat(info.rms_norm_eps.max(1e-12));
    let mut x = g.add_op(OpType::LayerNorm { axis: -1, epsilon: eps }, vec![sum2], 1, Some("embeddings.LayerNorm".into()));

    // Encoder layers.
    for i in 0..info.num_hidden_layers {
        let p = format!("encoder.layer.{i}");
        // Self-attention (non-causal / bidirectional).
        let norm1 = g.add_op(OpType::LayerNorm { axis: -1, epsilon: eps }, vec![x], 1, Some(format!("{p}.attention.output.LayerNorm")));
        let q = g.add_op(OpType::MatMul, vec![norm1], 1, Some(format!("{p}.attention.self.query")));
        let k = g.add_op(OpType::MatMul, vec![norm1], 1, Some(format!("{p}.attention.self.key")));
        let v = g.add_op(OpType::MatMul, vec![norm1], 1, Some(format!("{p}.attention.self.value")));
        let attn = g.add_op(
            OpType::MultiHeadAttention { num_heads: info.num_attention_heads, head_dim: info.head_dim, causal: false, scale: None },
            vec![q, k, v], 1, Some(format!("{p}.attention.self")),
        );
        let out = g.add_op(OpType::MatMul, vec![attn], 1, Some(format!("{p}.attention.output.dense")));
        let x1 = g.add_op(OpType::Add, vec![x, out], 1, Some(format!("{p}.attn_res")));

        // FFN.
        let ff1 = g.add_op(OpType::MatMul, vec![x1], 1, Some(format!("{p}.intermediate.dense")));
        let act = g.add_op(OpType::Gelu, vec![ff1], 1, Some(format!("{p}.intermediate.act")));
        let ff2 = g.add_op(OpType::MatMul, vec![act], 1, Some(format!("{p}.output.dense")));
        let x2  = g.add_op(OpType::Add, vec![x1, ff2], 1, Some(format!("{p}.ffn_res")));
        x = g.add_op(OpType::LayerNorm { axis: -1, epsilon: eps }, vec![x2], 1, Some(format!("{p}.output.LayerNorm")));
    }

    // Two outputs: full hidden states + CLS pooler (for sentence embeddings).
    g.mark_output(x, "last_hidden_state");
    let pooler = g.add_op(OpType::MatMul, vec![x], 1, Some("pooler.dense".into()));
    let pooler_act = g.add_op(OpType::Tanh, vec![pooler], 1, Some("pooler.activation".into()));
    g.mark_output(pooler_act, "pooler_output");

    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    const CFG: &str = r#"{"architectures":["BertForMaskedLM"],"model_type":"bert","vocab_size":30522,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":256,"max_position_embeddings":512,"rms_norm_eps":1e-12,"hidden_act":"gelu","rope_theta":10000.0}"#;
    #[test]
    fn tiny_bert_builds() {
        let info = sapient_hub::model_info::ModelInfo::from_json_str(CFG).unwrap();
        let g = build(&info).unwrap();
        assert_eq!(g.outputs.len(), 2, "BERT should have 2 outputs");
    }
}
