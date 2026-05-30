//! Llama-family causal LM forward pass (Llama, Mistral, Qwen, SmolVLM text backbone).

use std::collections::HashMap;

use anyhow::Result;
use sapient_core::Tensor;
use sapient_hub::model_info::ModelInfo;

use super::backend::{LlmBackend, LlmBackendDispatch, LlmBackendKind};
use super::common::{
    embed_tokens, mean_pool_hidden, merge_heads, quantize_tensor_to_q8_0, should_quantize_online,
    split_heads,
};
use crate::weights::{
    detect_weight_prefix, load_hf_weights, resolve_bias, resolve_lm_head, resolve_weight,
    tie_word_embeddings_from_config,
};

/// Decide how to split layers between Metal GPU and CPU.
///
/// Returns `(primary_kind, gpu_layers, cpu_fallback_dispatch)`:
/// - `gpu_layers == 0` → single backend, no split
/// - `gpu_layers > 0` → first `gpu_layers` on Metal, rest on CPU fallback
///
/// On Apple Silicon with Auto backend, if the model doesn't fit entirely in the
/// Metal memory budget we split layers proportionally — more layers on GPU than
/// CPU since UMA means zero-copy switching between the two.
#[allow(unused_variables)] // model_bytes / num_layers only used inside cfg(mlx) block
fn compute_backend_split(
    requested: LlmBackendKind,
    model_bytes: u64,
    num_layers: usize,
) -> (LlmBackendKind, usize, Option<LlmBackendDispatch>) {
    // Only apply splitting for Auto — explicit --backend cpu/metal is honoured as-is.
    if !matches!(requested, LlmBackendKind::Auto) {
        return (requested, 0, None);
    }

    // macOS + Metal check
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        use super::backend::{total_system_ram_bytes, MetalLlmBackend};
        if !MetalLlmBackend::is_available() {
            return (LlmBackendKind::Cpu, 0, None);
        }

        let total_ram = total_system_ram_bytes();
        if total_ram == 0 {
            return (LlmBackendKind::Metal, 0, None); // unknown → try Metal
        }

        // Full Metal: model + 1.5× KV-cache headroom fits with 2 GB OS reserve.
        let os_reserve = 2u64 * 1024 * 1024 * 1024;
        let budget = total_ram.saturating_sub(os_reserve);
        let needed = (model_bytes as f64 * 1.5) as u64;

        if needed <= budget {
            return (LlmBackendKind::Metal, 0, None); // entire model on Metal
        }

        // Partial fit: calculate how many layers can live on Metal.
        if num_layers > 0 && model_bytes < total_ram && model_bytes > 0 {
            let bytes_per_layer = model_bytes / num_layers as u64;
            if bytes_per_layer == 0 {
                return (LlmBackendKind::Metal, 0, None);
            }
            // How many layers fit in the Metal budget (including KV headroom)?
            let gpu_layers =
                ((budget as f64 / (bytes_per_layer as f64 * 1.5)) as usize).min(num_layers - 1); // keep at least one CPU layer

            if gpu_layers >= num_layers / 4 {
                // Worthwhile split: at least 25% of layers on GPU.
                if let Ok(cpu) = LlmBackendDispatch::from_kind(LlmBackendKind::Cpu) {
                    tracing::info!(
                        gpu_layers,
                        total = num_layers,
                        model_gb = model_bytes as f64 / 1e9,
                        budget_gb = budget as f64 / 1e9,
                        "hybrid Metal+CPU split"
                    );
                    return (LlmBackendKind::Metal, gpu_layers, Some(cpu));
                }
            }
        }

        // Model doesn't fit at all → CPU
        return (LlmBackendKind::Cpu, 0, None);
    }

    #[allow(unreachable_code)]
    (LlmBackendKind::Cpu, 0, None)
}

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
    /// Primary backend — Metal GPU on Apple Silicon, CPU otherwise.
    backend: LlmBackendDispatch,
    /// CPU fallback used for layers ≥ `gpu_layers` in hybrid mode.
    /// None = single-backend mode (all layers use `backend`).
    cpu_fallback: Option<Box<LlmBackendDispatch>>,
    /// Number of leading layers that run on `backend` (Metal).
    /// 0 means all layers use `backend`.
    gpu_layers: usize,
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

        // Online quantization: convert F16/BF16 projection matrices to Q8_0 at
        // load time.  This is strictly better than expanding to F32:
        //   - F32 expansion: 2 bytes/weight (F16) -> 4 bytes/weight (F32) = 2x RAM
        //   - Q8_0 quantization: 2 bytes/weight (F16) -> ~1.06 bytes/weight = half RAM
        //   - Per-step bandwidth: Q8_0 kernel reads ~1 byte/weight vs 4 for F32
        //   - Quality: Q8_0 is near-lossless (~0.01 PPL increase over F16)
        // Norm weights, biases, and embeddings retain their original dtype since
        // they are accessed differently (row gather, broadcast, etc.).
        // For already-quantized (Q4_0/Q8_0/K-quant) models this is a no-op.
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

        // Determine hybrid split: on Apple Silicon with Auto backend, check if the
        // model fits entirely in Metal's memory budget. If not, split layers.
        let model_bytes: u64 = weights
            .values()
            .map(|t| t.dtype().byte_count(t.numel()) as u64)
            .sum();
        let (primary_kind, gpu_layers, cpu_fallback) =
            compute_backend_split(backend, model_bytes, info.num_hidden_layers);

        let backend = LlmBackendDispatch::from_kind(primary_kind)?;
        if gpu_layers > 0 {
            tracing::info!(
                gpu_layers,
                total = info.num_hidden_layers,
                "hybrid Metal+CPU mode: first {gpu_layers} layers on Metal"
            );
        } else {
            tracing::debug!(
                backend = backend.name(),
                "initialized Llama forward backend"
            );
        }

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
            cpu_fallback: cpu_fallback.map(Box::new),
            gpu_layers,
        })
    }

    /// True when this engine is running in hybrid Metal+CPU mode.
    pub fn is_hybrid(&self) -> bool {
        self.gpu_layers > 0 && self.cpu_fallback.is_some()
    }

    /// For display: "auto", "metal", "cpu", or "metal+cpu (N/T layers on GPU)".
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
        // Select which backend handles this layer.
        // Hybrid mode: layers 0..gpu_layers → Metal, the rest → CPU fallback.
        // Single-backend mode: gpu_layers == 0, always use self.backend.
        //
        // Implementation note: we extract the references BEFORE any &mut borrows
        // of self.cache (Rust allows partial field borrows but not through methods).
        let on_gpu =
            !(self.gpu_layers > 0 && layer_idx >= self.gpu_layers && self.cpu_fallback.is_some());

        let pfx = format!("layers.{layer_idx}");
        let eps = self.info.rms_norm_eps as f32;
        let n_heads = self.info.num_attention_heads;
        let n_kv = self.info.num_key_value_heads;
        let head_dim = self.info.head_dim;
        let rope_theta = self.info.rope_theta as f32;

        // ── Pre-cache phase: norm → QKV → RoPE ───────────────────────────────
        // All operations in this scope use the per-layer backend.
        // We drop the backend reference at the end of the scope before touching cache.
        let (q, mut k, mut v, x_residual) = {
            let bk: &LlmBackendDispatch = if on_gpu {
                &self.backend
            } else {
                self.cpu_fallback.as_deref().unwrap()
            };

            let attn_norm_w = resolve_weight(
                &self.weights,
                &self.prefix,
                &format!("{pfx}.input_layernorm"),
            )?;
            let h = bk.rms_norm(&x, attn_norm_w, eps)?;

            // Q/K/V — parallel on CPU, sequential on Metal/GPU.
            let q_name = format!("{pfx}.self_attn.q_proj");
            let k_name = format!("{pfx}.self_attn.k_proj");
            let v_name = format!("{pfx}.self_attn.v_proj");
            let (q, k, v) = if bk.is_cpu() {
                let ((qr, kr), vr) = rayon::join(
                    || {
                        rayon::join(
                            || self.linear_with(&h, &q_name, bk),
                            || self.linear_with(&h, &k_name, bk),
                        )
                    },
                    || self.linear_with(&h, &v_name, bk),
                );
                (qr?, kr?, vr?)
            } else {
                (
                    self.linear_with(&h, &q_name, bk)?,
                    self.linear_with(&h, &k_name, bk)?,
                    self.linear_with(&h, &v_name, bk)?,
                )
            };

            let mut q = split_heads(&q, n_heads, head_dim)?;
            let mut k = split_heads(&k, n_kv, head_dim)?;
            let v = split_heads(&v, n_kv, head_dim)?;

            q = bk.apply_rope_positions(&q, positions, rope_theta)?;
            k = bk.apply_rope_positions(&k, positions, rope_theta)?;

            // x_residual is needed for the residual add after attention — borrow ends here.
            (q, k, v, x)
        }; // bk reference dropped

        // ── Cache phase: mutate self.cache[layer_idx] (no backend reference) ─
        if use_cache {
            let cache = &mut self.cache[layer_idx];
            let current_seq = cache.seq_len;
            if let (Some(ck), Some(cv)) = (&mut cache.keys, &mut cache.values) {
                k = crate::forward::common::update_kv_cache(ck, current_seq, &k)?;
                v = crate::forward::common::update_kv_cache(cv, current_seq, &v)?;
            }
            cache.seq_len = current_seq + positions.len();
        }

        // ── Post-cache phase: attention → FFN ────────────────────────────────
        // Re-borrow the per-layer backend (cache borrow is over).
        let bk: &LlmBackendDispatch = if on_gpu {
            &self.backend
        } else {
            self.cpu_fallback.as_deref().unwrap()
        };

        let attn = bk.gqa_attention(&q, &k, &v, n_kv, true)?;
        let attn = merge_heads(&attn)?;
        let o = self.linear_with(&attn, &format!("{pfx}.self_attn.o_proj"), bk)?;
        let x = bk.add(&x_residual, &o)?;

        let ffn_norm_w = resolve_weight(
            &self.weights,
            &self.prefix,
            &format!("{pfx}.post_attention_layernorm"),
        )?;
        let h = bk.rms_norm(&x, ffn_norm_w, eps)?;

        // Gate and up projections — parallel on CPU, sequential on Metal.
        let gate_w = resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.gate_proj"))?;
        let up_w = resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.up_proj"))?;
        let (gate, up) = if bk.is_cpu() {
            let (gr, ur) = rayon::join(|| bk.linear_3d(&h, gate_w), || bk.linear_3d(&h, up_w));
            (gr?, ur?)
        } else {
            (bk.linear_3d(&h, gate_w)?, bk.linear_3d(&h, up_w)?)
        };

        let gate = bk.silu(&gate)?;
        let mid = bk.mul(&gate, &up)?;
        let down = bk.linear_3d(
            &mid,
            resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.down_proj"))?,
        )?;
        bk.add(&x, &down)
    }

    /// Linear projection with explicit backend (used in forward_layer for hybrid routing).
    fn linear_with(&self, x: &Tensor, name: &str, bk: &LlmBackendDispatch) -> Result<Tensor> {
        let weight = resolve_weight(&self.weights, &self.prefix, name)?;
        let bias = resolve_bias(&self.weights, &self.prefix, name);
        bk.linear_3d_bias(x, weight, bias)
    }

    /// Linear projection using the model's primary backend.
    #[allow(dead_code)] // kept for non-hybrid callers and potential future use
    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        self.linear_with(x, name, &self.backend)
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
