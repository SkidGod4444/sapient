//! Llama-family causal LM forward pass (Llama, Mistral, Qwen, SmolVLM text backbone).

use std::collections::HashMap;

use anyhow::{bail, Result};
use sapient_core::Tensor;
use sapient_hub::model_info::{ModelInfo, MoeConfig, MoeScoring};

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
    /// Allocated KV-cache context window (≤ model max). `seq_len` is capped here
    /// so the sliding-window update never indexes past the cache.
    kv_ctx: usize,
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

        // MoE support gate: softmax (Mixtral) and sigmoid (GLM-4.5-Air: correction
        // bias + shared expert) routing are implemented; group-limited routing
        // (`n_group > 1`, DeepSeek-V3/GLM-5.2) fails loudly rather than route
        // silently-wrong.
        if let Some(moe) = &info.moe {
            validate_moe_support(moe)?;
        }

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

        // Determine hybrid split: on Apple Silicon with Auto backend, check if the
        // model fits entirely in Metal's memory budget. If not, split layers.
        let model_bytes: u64 = weights
            .values()
            .map(|t| t.dtype().byte_count(t.numel()) as u64)
            .sum();
        let (primary_kind, gpu_layers, cpu_fallback) =
            compute_backend_split(backend, model_bytes, info.num_hidden_layers);

        let backend = LlmBackendDispatch::from_kind(primary_kind)?;

        // Multi-row GEMV repack (llama.cpp-style): interleave heap Q4_K rows so
        // the SDOT kernel reads one contiguous stream per task. Pure-CPU engines
        // only — hybrid Metal layers must keep standard Q4_K.
        #[cfg(target_arch = "aarch64")]
        let weights = if gpu_layers == 0
            && backend.name() == "cpu"
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            super::common::repack_q4_k_weights(weights, &embed_key)
        } else {
            weights
        };

        // Resolve the output head AFTER the repack so an untied lm_head (the
        // single biggest matrix) gets the multi-row layout too; a tied head is
        // the embedding and stays row-major by the embed_key skip above.
        let lm_head = resolve_lm_head(&weights, &prefix, tie, &embed_key)?.clone();
        validate_core_shapes(&info, &weights, &embed_key, &lm_head)?;

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

        // Cap the pre-allocated cache window so 128K-context models don't reserve
        // (and OOM on) gigabytes of KV cache at load time. Longer conversations
        // slide the window. Override with SAPIENT_CTX.
        let max_seq = super::common::kv_cache_ctx(info.max_position_embeddings);
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
            kv_ctx: max_seq,
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

    /// Keep only the first `n` cached positions; returns the actual kept length
    /// (clamped to the current cache length). Used for prompt/prefix reuse.
    pub fn truncate_cache(&mut self, n: usize) -> usize {
        let kept = self.cache.first().map(|l| l.seq_len.min(n)).unwrap_or(0);
        for layer in &mut self.cache {
            layer.seq_len = kept;
        }
        kept
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

    /// Returns logits for ALL positions while **appending** `input_ids` to the
    /// KV cache (positions continue from the current cache length). Used by
    /// speculative decoding to verify draft tokens *with* prompt context; the
    /// caller rolls back rejected tokens via `truncate_cache`.
    pub fn forward_all_logits_cached(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let hidden = self.forward_hidden(input_ids, true)?;
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
        let x = embed_tokens(embed, input_ids)?;
        self.forward_hidden_from_embeds(x, input_ids.len(), use_cache)
    }

    /// Run the transformer from pre-built input embeddings `[1, seq, hidden]`
    /// instead of token ids — the multimodal entry point: a VLM splices visual
    /// token embeddings into the text embedding sequence (at the `<image>`
    /// positions) and prefills through here; subsequent decode steps use the
    /// normal token-id path against the same KV cache.
    fn forward_hidden_from_embeds(
        &mut self,
        mut x: Tensor,
        seq_len: usize,
        use_cache: bool,
    ) -> Result<Tensor> {
        let start_pos = if use_cache {
            self.cache.first().map(|l| l.seq_len).unwrap_or(0)
        } else {
            self.reset_cache();
            0
        };

        let positions: Vec<usize> = (start_pos..start_pos + seq_len).collect();

        for layer_idx in 0..self.info.num_hidden_layers {
            x = self.forward_layer(x, layer_idx, &positions, use_cache)?;
        }

        let norm_w = resolve_weight(&self.weights, &self.prefix, "norm")?;
        self.backend
            .rms_norm(&x, norm_w, self.info.rms_norm_eps as f32)
    }

    /// Gather input embeddings for `input_ids` — `[1, seq, hidden]`. Public for
    /// the VLM path (which overwrites the `<image>` rows with visual tokens).
    pub fn token_embeddings(&self, input_ids: &[u32]) -> Result<Tensor> {
        let embed = self
            .weights
            .get(&self.embed_key)
            .ok_or_else(|| anyhow::anyhow!("missing embedding weights at '{}'", self.embed_key))?;
        embed_tokens(embed, input_ids)
    }

    /// [`forward_logits`](Self::forward_logits) from pre-built embeddings —
    /// last-position logits, appending to the KV cache when `use_cache`.
    pub fn forward_logits_embeds(&mut self, embeds: Tensor, use_cache: bool) -> Result<Vec<f32>> {
        let seq = embeds.shape().dims()[1];
        let hidden = self.forward_hidden_from_embeds(embeds, seq, use_cache)?;
        self.backend.logits_from_hidden(&hidden, &self.lm_head)
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

            // Partial RoPE (GLM4-MoE `partial_rotary_factor` 0.5, Phi 0.4): only
            // the first `rotary_dim` (NEOX) channels rotate. Factor 1.0 (Llama,
            // Qwen, Mistral) keeps the existing full-RoPE path unchanged.
            let pf = self.info.partial_rotary_factor;
            if pf < 1.0 {
                let rotary_dim = ((head_dim as f64 * pf).round() as usize) & !1;
                q = bk.apply_rope_partial(&q, positions, rope_theta, rotary_dim)?;
                k = bk.apply_rope_partial(&k, positions, rope_theta, rotary_dim)?;
            } else {
                q = bk.apply_rope_positions(&q, positions, rope_theta)?;
                k = bk.apply_rope_positions(&k, positions, rope_theta)?;
            }

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
            // Cap at the allocated window so the next sliding-window update stays
            // in bounds (the cache evicts oldest positions beyond kv_ctx).
            cache.seq_len = (current_seq + positions.len()).min(self.kv_ctx);
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

        // FFN sub-layer: sparse MoE (router + top-k experts) or dense SwiGLU.
        let ffn_out = if self.is_moe_layer(layer_idx) {
            self.moe_ffn(&h, layer_idx, bk)?
        } else {
            // Dense SwiGLU — gate and up projections parallel on CPU, sequential on Metal.
            let gate_w =
                resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.gate_proj"))?;
            let up_w = resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.up_proj"))?;
            let (gate, up) = if bk.is_cpu() {
                let (gr, ur) = rayon::join(|| bk.linear_3d(&h, gate_w), || bk.linear_3d(&h, up_w));
                (gr?, ur?)
            } else {
                (bk.linear_3d(&h, gate_w)?, bk.linear_3d(&h, up_w)?)
            };
            let gate = bk.silu(&gate)?;
            let mid = bk.mul(&gate, &up)?;
            bk.linear_3d(
                &mid,
                resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.mlp.down_proj"))?,
            )?
        };
        bk.add(&x, &ffn_out)
    }

    /// True when layer `idx` is a sparse MoE layer (router + experts) rather than
    /// a dense FFN. MoE models replace all layers except the first `first_k_dense`
    /// (0 for Mixtral → every layer is MoE; 3 for DeepSeek/GLM).
    fn is_moe_layer(&self, idx: usize) -> bool {
        self.info
            .moe
            .as_ref()
            .is_some_and(|m| idx >= m.first_k_dense)
    }

    /// Sparse Mixture-of-Experts FFN for one layer.
    ///
    /// `h` is the post-attention-normed hidden state `[1, seq, hidden]`. Returns
    /// the FFN branch output `[1, seq, hidden]` (the residual add happens in the
    /// caller). The router scores every token over all experts, selects the top-k,
    /// (optionally) renormalises their weights, then runs each **active** expert
    /// once over the tokens routed to it (expert-grouped batching), scattering the
    /// weighted results back. For decode (`seq == 1`) this touches exactly `top_k`
    /// experts — the whole point of MoE: `top_k`-expert bandwidth, not all-expert.
    fn moe_ffn(&self, h: &Tensor, layer_idx: usize, bk: &LlmBackendDispatch) -> Result<Tensor> {
        let moe = self
            .info
            .moe
            .as_ref()
            .expect("moe_ffn called on a non-MoE model");
        let hidden = self.info.hidden_size;
        let seq = h.shape().dims()[1];
        let num_experts = moe.num_experts;
        let pfx = format!("layers.{layer_idx}.block_sparse_moe");

        // Router logits: [1, seq, num_experts].
        let gate_w = resolve_weight(&self.weights, &self.prefix, &format!("{pfx}.gate"))?;
        let router_logits = bk.linear_3d(h, gate_w)?;
        let logits = router_logits.as_f32_slice();

        // Sigmoid-gate correction bias (`e_score_correction_bias`, GLM/DeepSeek) —
        // biases WHICH experts win, not their weights. Absent for Mixtral.
        let correction_bias: Option<Vec<f32>> = self
            .weights
            .get(&format!(
                "{}{pfx}.gate.e_score_correction_bias",
                self.prefix
            ))
            .map(|t| t.to_f32_vec());

        // Per-token routing → build each active expert's token list with weights.
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for t in 0..seq {
            let row = &logits[t * num_experts..(t + 1) * num_experts];
            let (idx, wts) = route_topk(
                row,
                moe.top_k,
                moe.scoring_func,
                moe.norm_topk_prob,
                correction_bias.as_deref(),
                moe.routed_scaling_factor,
            )?;
            for (ei, w) in idx.into_iter().zip(wts) {
                expert_tokens[ei].push((t, w));
            }
        }

        let h_data = h.as_f32_slice();
        let mut out = vec![0f32; seq * hidden];
        for (ei, toks) in expert_tokens.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            // Gather this expert's tokens into a contiguous [1, rows, hidden] batch.
            let rows = toks.len();
            let mut hb = vec![0f32; rows * hidden];
            for (i, &(t, _)) in toks.iter().enumerate() {
                hb[i * hidden..(i + 1) * hidden]
                    .copy_from_slice(&h_data[t * hidden..(t + 1) * hidden]);
            }
            let hb = Tensor::from_f32_vec(hb, vec![1, rows, hidden])?;

            // SwiGLU expert: down(silu(w1·h) * w3·h). w1=gate_proj, w3=up_proj, w2=down_proj.
            let ep = format!("{pfx}.experts.{ei}");
            let w1 = resolve_weight(&self.weights, &self.prefix, &format!("{ep}.w1"))?;
            let w3 = resolve_weight(&self.weights, &self.prefix, &format!("{ep}.w3"))?;
            let w2 = resolve_weight(&self.weights, &self.prefix, &format!("{ep}.w2"))?;
            let (g, u) = if bk.is_cpu() {
                let (gr, ur) = rayon::join(|| bk.linear_3d(&hb, w1), || bk.linear_3d(&hb, w3));
                (gr?, ur?)
            } else {
                (bk.linear_3d(&hb, w1)?, bk.linear_3d(&hb, w3)?)
            };
            let mid = bk.mul(&bk.silu(&g)?, &u)?;
            let down = bk.linear_3d(&mid, w2)?;

            // Scatter weighted expert output back to the token rows.
            let d = down.as_f32_slice();
            for (i, &(t, w)) in toks.iter().enumerate() {
                let src = &d[i * hidden..(i + 1) * hidden];
                let dst = &mut out[t * hidden..(t + 1) * hidden];
                for (o, &s) in dst.iter_mut().zip(src) {
                    *o += w * s;
                }
            }
        }

        // Shared expert (GLM-4.5-Air / DeepSeek): an always-on SwiGLU run over
        // EVERY token and added **unweighted** to the routed output.
        if moe.num_shared_experts > 0 {
            let sp = format!("{pfx}.shared_expert");
            let w1 = resolve_weight(&self.weights, &self.prefix, &format!("{sp}.w1"))?;
            let w3 = resolve_weight(&self.weights, &self.prefix, &format!("{sp}.w3"))?;
            let w2 = resolve_weight(&self.weights, &self.prefix, &format!("{sp}.w2"))?;
            let (g, u) = if bk.is_cpu() {
                let (gr, ur) = rayon::join(|| bk.linear_3d(h, w1), || bk.linear_3d(h, w3));
                (gr?, ur?)
            } else {
                (bk.linear_3d(h, w1)?, bk.linear_3d(h, w3)?)
            };
            let mid = bk.mul(&bk.silu(&g)?, &u)?;
            let shared = bk.linear_3d(&mid, w2)?;
            for (o, &s) in out.iter_mut().zip(shared.as_f32_slice()) {
                *o += s;
            }
        }

        Ok(Tensor::from_f32_vec(out, vec![1, seq, hidden])?)
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

/// Reject MoE features this cut does not execute yet, so unsupported models fail
/// at load rather than routing silently-wrong. Softmax (Mixtral) and sigmoid
/// (GLM-4.5-Air: correction bias + shared expert) routing are implemented;
/// **group-limited** routing (`n_group > 1` — DeepSeek-V3, GLM-5.2) is not.
fn validate_moe_support(moe: &MoeConfig) -> Result<()> {
    if moe.n_group > 1 {
        bail!(
            "group-limited MoE routing (n_group={}, DeepSeek-V3/GLM-5.2) is not yet \
             supported — only single-group top-k (Mixtral, GLM-4.5-Air) is implemented",
            moe.n_group
        );
    }
    Ok(())
}

/// Route one token through the MoE gate: score all experts, select the top-k,
/// (optionally) renormalise, and scale.
///
/// Two gate flavours, each order-sensitive (a wrong order degrades quality
/// *silently* — gated numerically, not by "does it emit words"):
/// - **Softmax** (Mixtral): softmax over all experts → top-k by that value →
///   renorm. Selection score == weight score.
/// - **Sigmoid** (GLM-4.5-Air / DeepSeek): `sigmoid(logits)` are the weights;
///   selection uses `sigmoid + e_score_correction_bias` (the aux-loss-free bias
///   biases *which* experts win but NOT their weights); renorm (`+1e-20` denom),
///   then `* routed_scaling_factor`.
fn route_topk(
    logits: &[f32],
    k: usize,
    scoring: MoeScoring,
    norm_topk_prob: bool,
    correction_bias: Option<&[f32]>,
    routed_scaling: f32,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let n = logits.len();
    // `weight_score` becomes the routing weight; `choice_score` selects the top-k
    // (differs only for the sigmoid gate's correction bias).
    let (weight_score, choice_score): (Vec<f32>, Vec<f32>) = match scoring {
        MoeScoring::Softmax => {
            let s = softmax(logits);
            (s.clone(), s)
        }
        MoeScoring::Sigmoid => {
            let s: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();
            let choice = match correction_bias {
                Some(b) => s.iter().zip(b).map(|(v, bb)| v + bb).collect(),
                None => s.clone(),
            };
            (s, choice)
        }
    };
    let k = k.min(n);
    // Select the top-k by choice score (descending); ties break by lower index.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| choice_score[b].total_cmp(&choice_score[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let mut wts: Vec<f32> = idx.iter().map(|&i| weight_score[i]).collect();
    if norm_topk_prob {
        // +1e-20 matches the reference; in f32 it's a no-op for Mixtral's O(1) sums.
        let sum: f32 = wts.iter().sum::<f32>() + 1e-20;
        for w in &mut wts {
            *w /= sum;
        }
    }
    if routed_scaling != 1.0 {
        for w in &mut wts {
            *w *= routed_scaling;
        }
    }
    Ok((idx, wts))
}

/// Logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
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

#[cfg(test)]
mod moe_tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0, 0.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Monotonic in the logits.
        assert!(s[2] > s[1] && s[1] > s[0] && s[0] > s[3]);
    }

    #[test]
    fn route_topk_mixtral_order_matches_hand_computed() {
        // softmax([1,2,3,0]) = [0.087145, 0.236882, 0.643915, 0.032059].
        // top-2 → experts {2, 1}; renorm over their softmax values:
        //   0.643915/(0.643915+0.236882) = 0.731049
        //   0.236882/(0.643915+0.236882) = 0.268951
        let (idx, wts) = route_topk(
            &[1.0, 2.0, 3.0, 0.0],
            2,
            MoeScoring::Softmax,
            true,
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(
            idx,
            vec![2, 1],
            "top-k must pick the two highest-scoring experts"
        );
        assert!((wts[0] - 0.731049).abs() < 1e-4, "wts[0]={}", wts[0]);
        assert!((wts[1] - 0.268951).abs() < 1e-4, "wts[1]={}", wts[1]);
        assert!(
            (wts.iter().sum::<f32>() - 1.0).abs() < 1e-6,
            "renormalised weights must sum to 1"
        );
    }

    #[test]
    fn route_topk_without_renorm_keeps_raw_softmax() {
        let (idx, wts) = route_topk(
            &[1.0, 2.0, 3.0, 0.0],
            2,
            MoeScoring::Softmax,
            false,
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(idx, vec![2, 1]);
        // Raw softmax values (NOT renormalised) → sum < 1.
        assert!((wts[0] - 0.643915).abs() < 1e-4, "wts[0]={}", wts[0]);
        assert!((wts[1] - 0.236882).abs() < 1e-4, "wts[1]={}", wts[1]);
        assert!(wts.iter().sum::<f32>() < 0.99);
    }

    #[test]
    fn route_topk_ties_break_by_lower_index() {
        // Equal logits → equal softmax; the first `k` experts by index must win.
        let (idx, _) = route_topk(
            &[1.0, 1.0, 1.0, 1.0],
            2,
            MoeScoring::Softmax,
            true,
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn route_topk_glm_sigmoid_gate_matches_hand_computed() {
        // GLM-4.5-Air / DeepSeek gate. logits=[1,2,3,0], correction_bias=[0.5,0,-1,0.4].
        //   sigmoid = [0.731059, 0.880797, 0.952574, 0.5]        (these are the WEIGHTS)
        //   choice  = sigmoid + bias = [1.231059, 0.880797, -0.047426, 0.9]  (SELECTION)
        // top-2 by CHOICE → experts {0, 3} — note the bias FLIPS the pick away from
        // {2,1} (the raw-sigmoid top-2), proving the bias steers selection.
        // weights = RAW sigmoid at {0,3} = [0.731059, 0.5], renorm → [0.593846, 0.406154].
        let bias = [0.5f32, 0.0, -1.0, 0.4];
        let (idx, wts) = route_topk(
            &[1.0, 2.0, 3.0, 0.0],
            2,
            MoeScoring::Sigmoid,
            true,
            Some(&bias),
            1.0,
        )
        .unwrap();
        assert_eq!(idx, vec![0, 3], "correction bias must steer SELECTION");
        assert!(
            (wts[0] - 0.593846).abs() < 1e-4,
            "wts[0]={} (raw sigmoid, not bias-corrected)",
            wts[0]
        );
        assert!((wts[1] - 0.406154).abs() < 1e-4, "wts[1]={}", wts[1]);
    }

    #[test]
    fn route_topk_sigmoid_no_bias_selects_by_sigmoid() {
        // Without a correction bias, selection == raw-sigmoid top-k (experts 2,1).
        let (idx, _) = route_topk(
            &[1.0, 2.0, 3.0, 0.0],
            2,
            MoeScoring::Sigmoid,
            true,
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(idx, vec![2, 1]);
    }

    #[test]
    fn route_topk_routed_scaling_multiplies_weights() {
        let (_, base) = route_topk(
            &[1.0, 2.0, 3.0, 0.0],
            2,
            MoeScoring::Softmax,
            true,
            None,
            1.0,
        )
        .unwrap();
        let (_, scaled) = route_topk(
            &[1.0, 2.0, 3.0, 0.0],
            2,
            MoeScoring::Softmax,
            true,
            None,
            2.5,
        )
        .unwrap();
        for (b, s) in base.iter().zip(&scaled) {
            assert!((s - b * 2.5).abs() < 1e-6);
        }
    }
}
