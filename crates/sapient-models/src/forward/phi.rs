//! Phi-family causal LM forward pass.

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

#[derive(Debug, Default, Clone)]
struct LayerCache {
    keys: Option<Tensor>,
    values: Option<Tensor>,
    seq_len: usize,
}

pub struct PhiForward {
    info: ModelInfo,
    prefix: String,
    weights: HashMap<String, Tensor>,
    embed_key: String,
    lm_head: Tensor,
    cache: Vec<LayerCache>,
    backend: LlmBackendDispatch,
}

impl PhiForward {
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
        tracing::debug!(backend = backend.name(), "initialized Phi forward backend");

        let max_seq = info.max_position_embeddings;
        let n_kv = info.num_key_value_heads;
        let hd = info.head_dim;
        let cache_shape = vec![1, n_kv, max_seq, hd];

        let cache = (0..info.num_hidden_layers)
            .map(|_| {
                let keys = Tensor::zeros(cache_shape.clone(), sapient_core::DType::F32).unwrap();
                let values = Tensor::zeros(cache_shape.clone(), sapient_core::DType::F32).unwrap();
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

    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(input_ids, use_cache)?;
        let mut logits = self.backend.logits_from_hidden(&hidden, &self.lm_head)?;
        // Phi's lm_head has a bias term; add it if present.
        if let Some(bias) = resolve_bias(&self.weights, &self.prefix, "lm_head") {
            let bias_cow = bias.to_f32_cow();
            for (l, b) in logits.iter_mut().zip(bias_cow.iter()) {
                *l += *b;
            }
        }
        Ok(logits)
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

        // Phi names the final norm `final_layernorm`; fall back to `norm` for other variants.
        let (norm_w, norm_b) =
            match resolve_weight(&self.weights, &self.prefix, "final_layernorm") {
                Ok(w) => (w, resolve_bias(&self.weights, &self.prefix, "final_layernorm")),
                Err(_) => (
                    resolve_weight(&self.weights, &self.prefix, "norm")?,
                    resolve_bias(&self.weights, &self.prefix, "norm"),
                ),
            };
        self.backend
            .layer_norm(&x, norm_w, norm_b, self.info.rms_norm_eps as f32)
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

        // RoPE is applied to only the first `rotary_dim` channels (Phi partial rotary).
        let rotary_dim = ((self.info.partial_rotary_factor * head_dim as f64).round() as usize)
            .clamp(2, head_dim);
        let theta = self.info.rope_theta as f32;

        // Input LayerNorm (Phi uses LayerNorm with a bias term).
        let in_ln = format!("{pfx}.input_layernorm");
        let norm_w = resolve_weight(&self.weights, &self.prefix, &in_ln)?;
        let norm_b = resolve_bias(&self.weights, &self.prefix, &in_ln);
        let h = self.backend.layer_norm(&x, norm_w, norm_b, eps)?;

        // Q/K/V projections (Phi has bias on each).
        let q = self.linear_with_bias(&h, &format!("{pfx}.self_attn.q_proj"), None)?;
        let k = self.linear_with_bias(&h, &format!("{pfx}.self_attn.k_proj"), None)?;
        let v = self.linear_with_bias(&h, &format!("{pfx}.self_attn.v_proj"), None)?;

        let q = split_heads(&q, n_heads, head_dim)?;
        let k = split_heads(&k, n_heads, head_dim)?;
        let mut v = split_heads(&v, n_heads, head_dim)?;

        let q = self.backend.apply_rope_partial(&q, positions, theta, rotary_dim)?;
        let mut k = self.backend.apply_rope_partial(&k, positions, theta, rotary_dim)?;

        if use_cache {
            let current_seq = self.cache[layer_idx].seq_len;
            let cache = &mut self.cache[layer_idx];
            if let (Some(ck), Some(cv)) = (&mut cache.keys, &mut cache.values) {
                k = crate::forward::common::update_kv_cache(ck, current_seq, &k)?;
                v = crate::forward::common::update_kv_cache(cv, current_seq, &v)?;
            }
            cache.seq_len = (current_seq + positions.len()).min(self.info.max_position_embeddings);
        }

        let attn = self.backend.gqa_attention(&q, &k, &v, n_heads, true)?;
        let attn = merge_heads(&attn)?;
        // Attention output projection (Phi-2 calls it `dense`, Phi-3 `o_proj`).
        let o = self.linear_with_bias(
            &attn,
            &format!("{pfx}.self_attn.dense"),
            Some(&format!("{pfx}.self_attn.o_proj")),
        )?;

        // Phi-1/1.5/2 ("phi") use a parallel block: attention and MLP both read the
        // same normalized input `h` and are summed onto the residual.
        if self.info.model_type == "phi" {
            let ff = self.mlp_phi2(&h, &pfx)?;
            let parallel_res = self.backend.add(&o, &ff)?;
            self.backend.add(&x, &parallel_res)
        } else {
            // Phi-3 sequential: residual add, post-attention LayerNorm, then MLP.
            let x = self.backend.add(&x, &o)?;
            let post_ln = format!("{pfx}.post_attention_layernorm");
            let pn_w = resolve_weight(&self.weights, &self.prefix, &post_ln)?;
            let pn_b = resolve_bias(&self.weights, &self.prefix, &post_ln);
            let hn = self.backend.layer_norm(&x, pn_w, pn_b, eps)?;
            let ff = self.mlp_phi3(&hn, &pfx)?;
            self.backend.add(&x, &ff)
        }
    }

    /// Linear projection with optional bias, resolving `name` (or `alt` fallback)
    /// as the weight key and `<name>.bias` as the bias if present.
    fn linear_with_bias(&self, x: &Tensor, name: &str, alt: Option<&str>) -> Result<Tensor> {
        let (weight, bias) = match resolve_weight(&self.weights, &self.prefix, name) {
            Ok(w) => (w, resolve_bias(&self.weights, &self.prefix, name)),
            Err(e) => match alt {
                Some(a) => (
                    resolve_weight(&self.weights, &self.prefix, a)?,
                    resolve_bias(&self.weights, &self.prefix, a),
                ),
                None => return Err(e),
            },
        };
        self.backend.linear_3d_bias(x, weight, bias)
    }

    /// Phi-1/1.5/2 MLP: fc1 → gelu_new → fc2 (both with bias).
    fn mlp_phi2(&self, h: &Tensor, pfx: &str) -> Result<Tensor> {
        let ff1 = self.linear_with_bias(h, &format!("{pfx}.mlp.fc1"), None)?;
        let ff1 = self.backend.gelu(&ff1)?;
        self.linear_with_bias(&ff1, &format!("{pfx}.mlp.fc2"), None)
    }

    /// Phi-3 MLP: fused gate_up_proj → SwiGLU → down_proj.
    fn mlp_phi3(&self, h: &Tensor, pfx: &str) -> Result<Tensor> {
        let gate_up = self.linear_with_bias(h, &format!("{pfx}.mlp.gate_up_proj"), None)?;
        // gate_up is [1, seq, 2*inter] contiguous; split the last dim into the
        // gate and up halves. We copy into contiguous buffers rather than use a
        // strided view, since the elementwise kernels read data contiguously.
        let dims = gate_up.shape().dims().to_vec();
        let last = *dims.last().unwrap();
        let inter = last / 2;
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let src = gate_up.to_f32_cow();
        let mut gate_v = vec![0.0f32; rows * inter];
        let mut up_v = vec![0.0f32; rows * inter];
        for r in 0..rows {
            let base = r * last;
            gate_v[r * inter..(r + 1) * inter].copy_from_slice(&src[base..base + inter]);
            up_v[r * inter..(r + 1) * inter].copy_from_slice(&src[base + inter..base + last]);
        }
        let mut half_dims = dims.clone();
        *half_dims.last_mut().unwrap() = inter;
        let gate = Tensor::from_f32(&gate_v, sapient_core::Shape::new(half_dims.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let up = Tensor::from_f32(&up_v, sapient_core::Shape::new(half_dims))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let gate = self.backend.silu(&gate)?;
        let activated = self.backend.mul(&gate, &up)?;
        self.linear_with_bias(&activated, &format!("{pfx}.mlp.down_proj"), None)
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
