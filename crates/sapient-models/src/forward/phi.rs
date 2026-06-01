//! Phi-family causal LM forward pass.

use std::collections::HashMap;

use anyhow::Result;
use sapient_core::Tensor;
use sapient_hub::model_info::ModelInfo;

#[cfg(all(target_os = "macos", feature = "mlx"))]
use super::backend::mlx_sdpa_supported_head_dim;
use super::backend::{LlmBackend, LlmBackendDispatch, LlmBackendKind};
use super::common::{
    embed_tokens, mean_pool_hidden, merge_heads, quantize_tensor_to_q8_0, should_quantize_online,
    split_heads,
};
use crate::weights::{
    detect_weight_prefix, load_hf_weights, resolve_bias, resolve_lm_head, resolve_weight,
    tie_word_embeddings_from_config,
};

/// Decide backend split for Phi models, respecting Metal SDPA head_dim constraints.
///
/// Returns `(primary_kind, gpu_layers, cpu_fallback)`:
/// - Metal is never selected when `head_dim` is not in MLX's pre-compiled shader set.
/// - `gpu_layers == 0` → single backend, no split.
/// - `gpu_layers > 0`  → first `gpu_layers` on Metal, rest on CPU fallback.
#[allow(unused_variables)]
fn compute_phi_backend_split(
    requested: LlmBackendKind,
    model_bytes: u64,
    num_layers: usize,
    head_dim: usize,
) -> (LlmBackendKind, usize, Option<LlmBackendDispatch>) {
    // If the requested backend is explicitly CPU, honour it immediately.
    if matches!(requested, LlmBackendKind::Cpu) {
        return (LlmBackendKind::Cpu, 0, None);
    }

    // Metal SDPA shader compatibility check: certain head_dims have no precompiled
    // Metal shader.  Phi-2 uses head_dim=80, which is not supported by MLX.
    // Forcing Metal would panic at inference time; silently fall back to CPU.
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        use super::backend::MetalLlmBackend;
        if MetalLlmBackend::is_available() && !mlx_sdpa_supported_head_dim(head_dim) {
            if matches!(requested, LlmBackendKind::Metal) {
                // Explicit --backend metal with an incompatible head_dim:
                // return Metal kind and let from_kind_with_head_dim produce a clear error.
                return (LlmBackendKind::Metal, 0, None);
            }
            tracing::info!(
                head_dim,
                "Phi auto-backend: CPU (Metal SDPA has no shader for head_dim={head_dim}; \
                 supported: 32, 64, 96, 128, 256)"
            );
            return (LlmBackendKind::Cpu, 0, None);
        }
    }

    // From here: Auto or Metal with a compatible head_dim.
    if !matches!(requested, LlmBackendKind::Auto) {
        return (requested, 0, None);
    }

    // Auto with Metal available: apply the same layer-split logic as LlamaForward.
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        use super::backend::{total_system_ram_bytes, MetalLlmBackend};
        if !MetalLlmBackend::is_available() {
            return (LlmBackendKind::Cpu, 0, None);
        }

        let total_ram = total_system_ram_bytes();
        if total_ram == 0 {
            return (LlmBackendKind::Metal, 0, None);
        }

        let os_reserve = 2u64 * 1024 * 1024 * 1024;
        let budget = total_ram.saturating_sub(os_reserve);
        let needed = (model_bytes as f64 * 1.5) as u64;

        if needed <= budget {
            return (LlmBackendKind::Metal, 0, None);
        }

        if num_layers > 0 && model_bytes < total_ram && model_bytes > 0 {
            let bytes_per_layer = model_bytes / num_layers as u64;
            if bytes_per_layer > 0 {
                let gpu_layers =
                    ((budget as f64 / (bytes_per_layer as f64 * 1.5)) as usize).min(num_layers - 1);
                if gpu_layers >= num_layers / 4 {
                    if let Ok(cpu) = LlmBackendDispatch::from_kind(LlmBackendKind::Cpu) {
                        tracing::info!(
                            gpu_layers,
                            total = num_layers,
                            "Phi hybrid Metal+CPU split"
                        );
                        return (LlmBackendKind::Metal, gpu_layers, Some(cpu));
                    }
                }
            }
        }

        return (LlmBackendKind::Cpu, 0, None);
    }

    #[allow(unreachable_code)]
    (LlmBackendKind::Cpu, 0, None)
}

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
    /// Primary backend — Metal GPU on Apple Silicon when head_dim is supported, CPU otherwise.
    backend: LlmBackendDispatch,
    /// CPU fallback used for layers ≥ `gpu_layers` in hybrid mode. None = single backend.
    cpu_fallback: Option<Box<LlmBackendDispatch>>,
    /// Number of leading layers on `backend` (Metal). 0 = all layers on primary backend.
    gpu_layers: usize,
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

        let weights: HashMap<String, Tensor> = weights
            .into_iter()
            .map(|(k, v)| {
                if should_quantize_online(&k, &v) {
                    (k, quantize_tensor_to_q8_0(v))
                } else {
                    (k, v)
                }
            })
            .collect();

        let embed_key = format!("{prefix}embed_tokens.weight");
        let tie = tie_word_embeddings_from_config(&info.raw);
        let lm_head = resolve_lm_head(&weights, &prefix, tie, &embed_key)?.clone();
        validate_core_shapes(&info, &weights, &embed_key, &lm_head)?;

        // Compute model size for backend split heuristics.
        let model_bytes: u64 = weights.values().map(|t| t.byte_size() as u64).sum();
        let num_layers = info.num_hidden_layers;
        let head_dim = info.head_dim;

        let (primary_kind, gpu_layers, cpu_fallback) =
            compute_phi_backend_split(backend, model_bytes, num_layers, head_dim);

        // Use head_dim-aware selection so an explicit --backend metal with an
        // incompatible head_dim gives a user-readable error instead of a panic.
        let backend = LlmBackendDispatch::from_kind_with_head_dim(primary_kind, head_dim)?;
        tracing::debug!(
            backend = backend.name(),
            gpu_layers,
            "initialized Phi forward backend"
        );

        // Cap the pre-allocated cache window (see common::kv_cache_ctx) so large
        // context models don't OOM at load time; longer chats slide the window.
        let max_seq = super::common::kv_cache_ctx(info.max_position_embeddings);
        let n_kv = info.num_key_value_heads;
        let cache_shape = vec![1, n_kv, max_seq, head_dim];
        let use_q8_cache = head_dim % 32 == 0;

        let cache = (0..num_layers)
            .map(|_| {
                let (keys, values) = if use_q8_cache {
                    let numel = n_kv * max_seq * head_dim;
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
            cpu_fallback: cpu_fallback.map(Box::new),
            gpu_layers,
        })
    }

    pub fn reset_cache(&mut self) {
        for layer in &mut self.cache {
            layer.seq_len = 0;
        }
    }

    /// Keep only the first `n` cached positions; returns the actual kept length.
    pub fn truncate_cache(&mut self, n: usize) -> usize {
        let kept = self.cache.first().map(|l| l.seq_len.min(n)).unwrap_or(0);
        for layer in &mut self.cache {
            layer.seq_len = kept;
        }
        kept
    }

    /// True when layers are split between Metal (primary) and CPU (fallback).
    pub fn is_hybrid(&self) -> bool {
        self.gpu_layers > 0 && self.cpu_fallback.is_some()
    }

    /// Human-readable label for the active backend(s).
    pub fn backend_label(&self) -> String {
        if self.is_hybrid() {
            format!(
                "metal+cpu hybrid ({}/{} layers on GPU)",
                self.gpu_layers, self.info.num_hidden_layers
            )
        } else {
            self.backend.name().to_string()
        }
    }

    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(input_ids, use_cache)?;
        let mut logits = self.backend.logits_from_hidden(&hidden, &self.lm_head)?;
        if let Some(bias) = resolve_bias(&self.weights, &self.prefix, "lm_head") {
            let bias_cow = bias.to_f32_cow();
            for (l, b) in logits.iter_mut().zip(bias_cow.iter()) {
                *l += *b;
            }
        }
        Ok(logits)
    }

    /// Returns logits for ALL positions without updating the KV cache.
    pub fn forward_all_logits(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let hidden = self.forward_hidden(input_ids, false)?;
        let mut all = self
            .backend
            .all_logits_from_hidden(&hidden, &self.lm_head)?;
        if let Some(bias) = resolve_bias(&self.weights, &self.prefix, "lm_head") {
            let bias_cow = bias.to_f32_cow();
            for logits in &mut all {
                for (l, b) in logits.iter_mut().zip(bias_cow.iter()) {
                    *l += *b;
                }
            }
        }
        Ok(all)
    }

    /// Returns logits for ALL positions while **appending** `input_ids` to the
    /// KV cache (positions continue from the current cache length). Used by
    /// speculative decoding to verify draft tokens with prompt context.
    pub fn forward_all_logits_cached(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let hidden = self.forward_hidden(input_ids, true)?;
        let mut all = self
            .backend
            .all_logits_from_hidden(&hidden, &self.lm_head)?;
        if let Some(bias) = resolve_bias(&self.weights, &self.prefix, "lm_head") {
            let bias_cow = bias.to_f32_cow();
            for logits in &mut all {
                for (l, b) in logits.iter_mut().zip(bias_cow.iter()) {
                    *l += *b;
                }
            }
        }
        Ok(all)
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
            .ok_or_else(|| anyhow::anyhow!("missing embedding at '{}'", self.embed_key))?;
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

        let (norm_w, norm_b) = match resolve_weight(&self.weights, &self.prefix, "final_layernorm")
        {
            Ok(w) => (
                w,
                resolve_bias(&self.weights, &self.prefix, "final_layernorm"),
            ),
            Err(_) => (
                resolve_weight(&self.weights, &self.prefix, "norm")?,
                resolve_bias(&self.weights, &self.prefix, "norm"),
            ),
        };
        let is_phi2 = matches!(self.info.model_type.as_str(), "phi" | "phi2");
        apply_norm(
            &self.backend,
            &x,
            norm_w,
            norm_b,
            self.info.rms_norm_eps as f32,
            !is_phi2,
        )
    }

    fn forward_layer(
        &mut self,
        x: Tensor,
        layer_idx: usize,
        positions: &[usize],
        use_cache: bool,
    ) -> Result<Tensor> {
        let on_gpu =
            !(self.gpu_layers > 0 && layer_idx >= self.gpu_layers && self.cpu_fallback.is_some());

        let pfx = format!("layers.{layer_idx}");
        let eps = self.info.rms_norm_eps as f32;
        let n_heads = self.info.num_attention_heads;
        let n_kv = self.info.num_key_value_heads;
        let head_dim = self.info.head_dim;
        let rotary_dim = ((self.info.partial_rotary_factor * head_dim as f64).round() as usize)
            .clamp(2, head_dim);
        let theta = self.info.rope_theta as f32;
        // Phi-1/1.5/2 use the parallel block + LayerNorm + fc1/fc2 GELU MLP; Phi-3/4
        // use the sequential block + RMSNorm + fused gate_up SwiGLU. GGUF reports
        // "phi"/"phi2" for the former, "phi3" for the latter.
        let is_phi2 = matches!(self.info.model_type.as_str(), "phi" | "phi2");

        // ── Phase 1: norm, QKV, RoPE (and parallel MLP for Phi-2) ────────────────
        // The backend reference is held only within this inner scope; it is dropped
        // before Phase 2 mutates self.cache, satisfying the borrow checker.
        let (q, mut k, mut v, x_residual, parallel_ff_opt) = {
            let bk: &LlmBackendDispatch = if on_gpu {
                &self.backend
            } else {
                self.cpu_fallback.as_deref().unwrap()
            };
            let weights = &self.weights;
            let prefix = &self.prefix;

            let in_ln = format!("{pfx}.input_layernorm");
            let norm_w = resolve_weight(weights, prefix, &in_ln)?;
            let norm_b = resolve_bias(weights, prefix, &in_ln);
            let h = apply_norm(bk, &x, norm_w, norm_b, eps, !is_phi2)?;

            let q = linear_with_bias_bk(
                bk,
                weights,
                prefix,
                &h,
                &format!("{pfx}.self_attn.q_proj"),
                None,
            )?;
            let k = linear_with_bias_bk(
                bk,
                weights,
                prefix,
                &h,
                &format!("{pfx}.self_attn.k_proj"),
                None,
            )?;
            let v = linear_with_bias_bk(
                bk,
                weights,
                prefix,
                &h,
                &format!("{pfx}.self_attn.v_proj"),
                None,
            )?;

            // GQA: k/v have n_kv heads (== n_heads for Phi-1/2; < for Phi-3/4).
            let q = split_heads(&q, n_heads, head_dim)?;
            let k = split_heads(&k, n_kv, head_dim)?;
            let v = split_heads(&v, n_kv, head_dim)?;

            let q = bk.apply_rope_partial(&q, positions, theta, rotary_dim)?;
            let k = bk.apply_rope_partial(&k, positions, theta, rotary_dim)?;

            // Phi-2 parallel block: compute MLP from the same normalised input h.
            let parallel_ff = if is_phi2 {
                Some(mlp_phi2_bk(bk, weights, prefix, &h, &pfx)?)
            } else {
                None
            };

            (q, k, v, x, parallel_ff)
        }; // bk / weights / prefix borrows released here

        // ── Phase 2: KV cache update (no backend reference held) ─────────────────
        if use_cache {
            let current_seq = self.cache[layer_idx].seq_len;
            let cache = &mut self.cache[layer_idx];
            if let (Some(ck), Some(cv)) = (&mut cache.keys, &mut cache.values) {
                k = crate::forward::common::update_kv_cache(ck, current_seq, &k)?;
                v = crate::forward::common::update_kv_cache(cv, current_seq, &v)?;
            }
            self.cache[layer_idx].seq_len = (current_seq + positions.len()).min(
                super::common::kv_cache_ctx(self.info.max_position_embeddings),
            );
        }

        // ── Phase 3: attention, output projection, FFN ────────────────────────────
        let bk: &LlmBackendDispatch = if on_gpu {
            &self.backend
        } else {
            self.cpu_fallback.as_deref().unwrap()
        };
        let weights = &self.weights;
        let prefix = &self.prefix;

        let attn = bk.gqa_attention(&q, &k, &v, n_kv, true)?;
        let attn = merge_heads(&attn)?;
        let o = linear_with_bias_bk(
            bk,
            weights,
            prefix,
            &attn,
            &format!("{pfx}.self_attn.dense"),
            Some(&format!("{pfx}.self_attn.o_proj")),
        )?;

        if is_phi2 {
            let ff = parallel_ff_opt.unwrap();
            let parallel_res = bk.add(&o, &ff)?;
            bk.add(&x_residual, &parallel_res)
        } else {
            let x = bk.add(&x_residual, &o)?;
            let post_ln = format!("{pfx}.post_attention_layernorm");
            let pn_w = resolve_weight(weights, prefix, &post_ln)?;
            let pn_b = resolve_bias(weights, prefix, &post_ln);
            let hn = apply_norm(bk, &x, pn_w, pn_b, eps, !is_phi2)?;
            let ff = mlp_phi3_bk(bk, weights, prefix, &hn, &pfx)?;
            bk.add(&x, &ff)
        }
    }
}

// ── Free helper functions (explicit backend + weights refs, borrow-checker safe) ──

/// Apply the block's normalization: RMSNorm for Phi-3/4 (`rms = true`), LayerNorm
/// (with optional bias) for Phi-1/1.5/2. Phi-3 switched from LayerNorm to RMSNorm.
fn apply_norm(
    bk: &LlmBackendDispatch,
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
    rms: bool,
) -> Result<Tensor> {
    if rms {
        bk.rms_norm(x, weight, eps)
    } else {
        bk.layer_norm(x, weight, bias, eps)
    }
}

/// Linear projection with optional bias; resolves `name` (or `alt` fallback).
fn linear_with_bias_bk(
    bk: &LlmBackendDispatch,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: &Tensor,
    name: &str,
    alt: Option<&str>,
) -> Result<Tensor> {
    let (weight, bias) = match resolve_weight(weights, prefix, name) {
        Ok(w) => (w, resolve_bias(weights, prefix, name)),
        Err(e) => match alt {
            Some(a) => (
                resolve_weight(weights, prefix, a)?,
                resolve_bias(weights, prefix, a),
            ),
            None => return Err(e),
        },
    };
    bk.linear_3d_bias(x, weight, bias)
}

/// Phi-1/1.5/2 MLP: fc1 → gelu_new → fc2 (both with bias).
fn mlp_phi2_bk(
    bk: &LlmBackendDispatch,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    h: &Tensor,
    pfx: &str,
) -> Result<Tensor> {
    let ff1 = linear_with_bias_bk(bk, weights, prefix, h, &format!("{pfx}.mlp.fc1"), None)?;
    let ff1 = bk.gelu(&ff1)?;
    linear_with_bias_bk(bk, weights, prefix, &ff1, &format!("{pfx}.mlp.fc2"), None)
}

/// Phi-3 MLP: fused gate_up_proj → SwiGLU → down_proj.
fn mlp_phi3_bk(
    bk: &LlmBackendDispatch,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    h: &Tensor,
    pfx: &str,
) -> Result<Tensor> {
    let gate_up = linear_with_bias_bk(
        bk,
        weights,
        prefix,
        h,
        &format!("{pfx}.mlp.gate_up_proj"),
        None,
    )?;
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
    let gate = bk.silu(&gate)?;
    let activated = bk.mul(&gate, &up)?;
    linear_with_bias_bk(
        bk,
        weights,
        prefix,
        &activated,
        &format!("{pfx}.mlp.down_proj"),
        None,
    )
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
            "embedding vocab rows {} < config vocab_size {}",
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
