//! Llama-family causal LM forward pass (Llama, Mistral, Qwen, SmolVLM text backbone).

use std::collections::HashMap;

use anyhow::Result;
use sapient_core::Tensor;
use sapient_hub::model_info::ModelInfo;

use super::backend::{LlmBackend, LlmBackendDispatch, LlmBackendKind};
use super::common::{embed_tokens, mean_pool_hidden, merge_heads, split_heads};
use crate::weights::{
    detect_weight_prefix, load_hf_weights, resolve_bias, resolve_lm_head, resolve_weight,
    tie_word_embeddings_from_config,
};

/// Per-layer KV cache stored as concatenated 4-D tensors.
#[derive(Debug, Default, Clone)]
struct LayerCache {
    keys: Option<Tensor>,
    values: Option<Tensor>,
    seq_len: usize,
}

/// Real Llama-architecture forward engine backed by safetensors weights.
pub struct LlamaForward {
    info: ModelInfo,
    prefix: String,
    weights: HashMap<String, Tensor>,
    embed_key: String,
    lm_head: Tensor,
    cache: Vec<LayerCache>,
    backend: LlmBackendDispatch,
}

impl LlamaForward {
    pub fn from_files(info: ModelInfo, weight_paths: &[std::path::PathBuf]) -> Result<Self> {
        Self::from_files_with_backend(info, weight_paths, LlmBackendKind::Auto)
    }

    pub fn from_files_with_backend(
        info: ModelInfo,
        weight_paths: &[std::path::PathBuf],
        backend: LlmBackendKind,
    ) -> Result<Self> {
        let weights = load_hf_weights(weight_paths)?;
        Self::from_weights_with_backend(info, weights, backend)
    }

    pub fn from_weights(info: ModelInfo, weights: HashMap<String, Tensor>) -> Result<Self> {
        Self::from_weights_with_backend(info, weights, LlmBackendKind::Auto)
    }

    pub fn from_weights_with_backend(
        info: ModelInfo,
        weights: HashMap<String, Tensor>,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        let prefix = detect_weight_prefix(&weights);
        let embed_key = format!("{prefix}embed_tokens.weight");
        let tie = tie_word_embeddings_from_config(&info.raw);
        let lm_head = resolve_lm_head(&weights, &prefix, tie, &embed_key)?.clone();
        validate_core_shapes(&info, &weights, &embed_key, &lm_head)?;
        let backend = LlmBackendDispatch::from_kind(backend)?;
        tracing::debug!(
            backend = backend.name(),
            "initialized Llama forward backend"
        );

        let max_seq = info.max_position_embeddings;
        let n_kv = info.num_key_value_heads;
        let hd = info.head_dim;
        let cache_shape = vec![1, n_kv, max_seq, hd];

        // Allocate KV cache as Q8_0 (4× smaller than F32) when head_dim is a multiple
        // of 32 (the Q8_0 block size).  Fall back to F32 otherwise.
        let use_q8_cache = hd % 32 == 0;

        let cache = (0..info.num_hidden_layers)
            .map(|_| {
                let (keys, values) = if use_q8_cache {
                    // Q8_0: numel/32 blocks × 34 bytes each.
                    let numel = n_kv * max_seq * hd;
                    let kv_bytes = numel / 32 * 34;
                    let k = Tensor::from_quant_bytes(
                        &vec![0u8; kv_bytes],
                        cache_shape.clone(),
                        sapient_core::DType::Q8_0,
                    )
                    .unwrap();
                    let v = Tensor::from_quant_bytes(
                        &vec![0u8; kv_bytes],
                        cache_shape.clone(),
                        sapient_core::DType::Q8_0,
                    )
                    .unwrap();
                    (k, v)
                } else {
                    let k = Tensor::zeros(cache_shape.clone(), sapient_core::DType::F32).unwrap();
                    let v = Tensor::zeros(cache_shape.clone(), sapient_core::DType::F32).unwrap();
                    (k, v)
                };
                LayerCache {
                    keys: Some(keys),
                    values: Some(values),
                    seq_len: 0,
                }
            })
            .collect();

        Ok(Self {
            cache,
            info,
            prefix,
            embed_key,
            lm_head,
            weights,
            backend,
        })
    }

    pub fn reset_cache(&mut self) {
        for layer in &mut self.cache {
            layer.seq_len = 0;
        }
    }

    /// Run forward on token ids and return logits for the last token.
    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(input_ids, use_cache)?;
        self.backend.logits_from_hidden(&hidden, &self.lm_head)
    }

    /// Returns logits for ALL positions without updating the KV cache.
    /// Used by speculative decoding to verify draft tokens in one shot.
    pub fn forward_all_logits(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let hidden = self.forward_hidden(input_ids, false)?;
        self.backend.all_logits_from_hidden(&hidden, &self.lm_head)
    }

    /// Mean-pooled hidden states for embedding models.
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
            self.cache.first().map(|l| l.seq_len).unwrap_or(0)
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
        self.backend
            .rms_norm(&x, norm_w, self.info.rms_norm_eps as f32)
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
        let n_kv = self.info.num_key_value_heads;
        let head_dim = self.info.head_dim;

        let attn_norm_w = resolve_weight(
            &self.weights,
            &self.prefix,
            &format!("{pfx}.input_layernorm"),
        )?;
        let h = self.backend.rms_norm(&x, attn_norm_w, eps)?;

        // Q/K/V projections. Llama/Mistral have no bias; Qwen2 has q/k/v biases —
        // resolve_bias returns None when absent, so this is correct for both.
        let q = self.linear(&h, &format!("{pfx}.self_attn.q_proj"))?;
        let k = self.linear(&h, &format!("{pfx}.self_attn.k_proj"))?;
        let v = self.linear(&h, &format!("{pfx}.self_attn.v_proj"))?;

        let mut q = split_heads(&q, n_heads, head_dim)?;
        let mut k = split_heads(&k, n_kv, head_dim)?;
        let mut v = split_heads(&v, n_kv, head_dim)?;

        q = self
            .backend
            .apply_rope_positions(&q, positions, self.info.rope_theta as f32)?;
        k = self
            .backend
            .apply_rope_positions(&k, positions, self.info.rope_theta as f32)?;

        let cache = &mut self.cache[layer_idx];
        if use_cache {
            let current_seq = cache.seq_len;
            if let (Some(ck), Some(cv)) = (&mut cache.keys, &mut cache.values) {
                k = crate::forward::common::update_kv_cache(ck, current_seq, &k)?;
                v = crate::forward::common::update_kv_cache(cv, current_seq, &v)?;
            }
            cache.seq_len = current_seq + positions.len();
        }

        let attn = self.backend.gqa_attention(&q, &k, &v, n_kv, true)?;
        let attn = merge_heads(&attn)?;
        let o = self.linear(&attn, &format!("{pfx}.self_attn.o_proj"))?;
        let x = self.backend.add(&x, &o)?;

        let ffn_norm_w = resolve_weight(
            &self.weights,
            &self.prefix,
            &format!("{pfx}.post_attention_layernorm"),
        )?;
        let h = self.backend.rms_norm(&x, ffn_norm_w, eps)?;

        let gate = self.backend.linear_3d(
            &h,
            resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.gate_proj"))?,
        )?;
        let up = self.backend.linear_3d(
            &h,
            resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.up_proj"))?,
        )?;
        let gate = self.backend.silu(&gate)?;
        let mid = self.backend.mul(&gate, &up)?;
        let down = self.backend.linear_3d(
            &mid,
            resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.down_proj"))?,
        )?;
        self.backend.add(&x, &down)
    }

    /// Linear projection that automatically applies a bias when the model has one
    /// (Qwen2 q/k/v), and is a plain matmul otherwise (Llama, Mistral).
    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let weight = resolve_weight(&self.weights, &self.prefix, name)?;
        let bias = resolve_bias(&self.weights, &self.prefix, name);
        self.backend.linear_3d_bias(x, weight, bias)
    }
}

fn validate_core_shapes(
    info: &ModelInfo,
    weights: &HashMap<String, Tensor>,
    embed_key: &str,
    lm_head: &Tensor,
) -> Result<()> {
    let embed = weights
        .get(embed_key)
        .ok_or_else(|| anyhow::anyhow!("missing embedding weights at '{embed_key}'"))?;
    let embed_dims = embed.shape().dims();
    if embed_dims.len() != 2 || embed_dims[1] != info.hidden_size {
        anyhow::bail!(
            "embedding shape mismatch at '{embed_key}': expected [vocab, {}], got {:?}",
            info.hidden_size,
            embed_dims
        );
    }
    if embed_dims[0] < info.vocab_size {
        anyhow::bail!(
            "embedding vocab rows {} are smaller than config vocab_size {}",
            embed_dims[0],
            info.vocab_size
        );
    }

    let head_dims = lm_head.shape().dims();
    if head_dims.len() != 2 || head_dims[1] != info.hidden_size {
        anyhow::bail!(
            "lm_head shape mismatch: expected [vocab, {}], got {:?}",
            info.hidden_size,
            head_dims
        );
    }

    Ok(())
}
