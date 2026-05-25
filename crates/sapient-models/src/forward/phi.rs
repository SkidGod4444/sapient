//! Phi-family causal LM forward pass.

use std::collections::HashMap;

use anyhow::Result;
use sapient_core::Tensor;
use sapient_hub::model_info::ModelInfo;

use super::common::{
    add, apply_rope_positions, concat_seq, embed_tokens, gelu, gqa_attention, layer_norm,
    linear_3d, logits_from_hidden, mean_pool_hidden, merge_heads, split_heads,
};
use crate::weights::{
    detect_weight_prefix, load_hf_weights, resolve_lm_head, resolve_weight,
    tie_word_embeddings_from_config,
};

#[derive(Debug, Default, Clone)]
struct LayerCache {
    keys: Option<Tensor>,
    values: Option<Tensor>,
}

pub struct PhiForward {
    info: ModelInfo,
    prefix: String,
    weights: HashMap<String, Tensor>,
    embed_key: String,
    lm_head: Tensor,
    cache: Vec<LayerCache>,
}

impl PhiForward {
    pub fn from_files(info: ModelInfo, weight_paths: &[std::path::PathBuf]) -> Result<Self> {
        let weights = load_hf_weights(weight_paths)?;
        Self::from_weights(info, weights)
    }

    pub fn from_weights(info: ModelInfo, weights: HashMap<String, Tensor>) -> Result<Self> {
        let prefix = detect_weight_prefix(&weights);
        let embed_key = format!("{prefix}embed_tokens.weight");
        let tie = tie_word_embeddings_from_config(&info.raw);
        let lm_head = resolve_lm_head(&weights, &prefix, tie, &embed_key)?.clone();

        Ok(Self {
            cache: vec![LayerCache::default(); info.num_hidden_layers],
            info,
            prefix,
            embed_key,
            lm_head,
            weights,
        })
    }

    pub fn reset_cache(&mut self) {
        for layer in &mut self.cache {
            *layer = LayerCache::default();
        }
    }

    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(input_ids, use_cache)?;
        logits_from_hidden(&hidden, &self.lm_head)
    }

    pub fn embed(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.reset_cache();
        let hidden = self.forward_hidden(input_ids, false)?;
        mean_pool_hidden(&hidden)
    }

    fn forward_hidden(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Tensor> {
        let embed = self
            .weights
            .get(&self.embed_key)
            .ok_or_else(|| anyhow::anyhow!("missing embedding weights at '{}'", self.embed_key))?;
        let mut x = embed_tokens(embed, input_ids)?;

        let start_pos = if use_cache {
            self.cache
                .first()
                .and_then(|l| l.keys.as_ref())
                .map(|k| k.shape().dims()[2])
                .unwrap_or(0)
        } else {
            self.reset_cache();
            0
        };
        let seq_len = input_ids.len();
        let positions: Vec<usize> = (start_pos..start_pos + seq_len).collect();

        for layer_idx in 0..self.info.num_hidden_layers {
            x = self.forward_layer(x, layer_idx, &positions, use_cache)?;
        }

        let norm_w = resolve_weight(&self.weights, &self.prefix, "norm")?;
        layer_norm(&x, norm_w, None, self.info.rms_norm_eps as f32)
    }

    fn forward_layer(
        &mut self,
        x: Tensor,
        layer_idx: usize,
        positions: &[usize],
        use_cache: bool,
    ) -> Result<Tensor> {
        let pfx = format!("layers.{layer_idx}");
        let eps = self.info.rms_norm_eps as f32;
        let n_heads = self.info.num_attention_heads;
        let head_dim = self.info.head_dim;

        let norm_w = resolve_weight(
            &self.weights,
            &self.prefix,
            &format!("{pfx}.input_layernorm"),
        )?;
        let h = layer_norm(&x, norm_w, None, eps)?;

        let q = linear_3d(
            &h,
            resolve_weight(
                &self.weights,
                &self.prefix,
                &format!("{pfx}.self_attn.q_proj"),
            )?,
        )?;
        let k = linear_3d(
            &h,
            resolve_weight(
                &self.weights,
                &self.prefix,
                &format!("{pfx}.self_attn.k_proj"),
            )?,
        )?;
        let v = linear_3d(
            &h,
            resolve_weight(
                &self.weights,
                &self.prefix,
                &format!("{pfx}.self_attn.v_proj"),
            )?,
        )?;

        let mut q = split_heads(&q, n_heads, head_dim)?;
        let mut k = split_heads(&k, n_heads, head_dim)?;
        let mut v = split_heads(&v, n_heads, head_dim)?;

        q = apply_rope_positions(&q, positions, self.info.rope_theta as f32)?;
        k = apply_rope_positions(&k, positions, self.info.rope_theta as f32)?;

        let cache = &mut self.cache[layer_idx];
        if use_cache {
            if let (Some(ck), Some(cv)) = (&cache.keys, &cache.values) {
                k = concat_seq(ck, &k)?;
                v = concat_seq(cv, &v)?;
            }
            cache.keys = Some(k.clone());
            cache.values = Some(v.clone());
        }

        let attn = gqa_attention(&q, &k, &v, n_heads, true)?;
        let attn = merge_heads(&attn)?;
        let o = linear_3d(
            &attn,
            resolve_weight(
                &self.weights,
                &self.prefix,
                &format!("{pfx}.self_attn.dense"),
            )
            .or_else(|_| {
                resolve_weight(
                    &self.weights,
                    &self.prefix,
                    &format!("{pfx}.self_attn.o_proj"),
                )
            })?,
        )?;
        let x = add(&x, &o)?;

        let ff1 = linear_3d(
            &x,
            resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.fc1"))?,
        )?;
        let ff1 = gelu(&ff1)?;
        let ff2 = linear_3d(
            &ff1,
            resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.fc2"))?,
        )?;
        add(&x, &ff2)
    }
}
