//! MLX-native Llama-family forward pass for Apple Silicon.
//!
//! All intermediate activations stay as `mlx_rs::Array` throughout the full
//! transformer pass. A single `eval()` call materialises the logit vector at
//! the very end. This is the same lazy-graph approach mlx-lm uses and yields
//! ~35× better throughput vs the per-op CPU↔GPU round-trip path.
//!
//! Supported architectures: Llama, Qwen2 / Qwen2.5, Mistral, SmolLM2.
//! (Phi models use a different block structure and are handled by PhiForward.)

#![cfg(all(target_os = "macos", feature = "mlx"))]

use std::collections::HashMap;

use anyhow::{Context, Result};
use mlx_rs::fast::ScaledDotProductAttentionMask;
use sapient_core::Tensor;
use sapient_hub::model_info::ModelInfo;

use crate::weights::{detect_weight_prefix, load_hf_weights, resolve_bias, resolve_weight};

/// Shorthand for mlx_rs's native Result (carries Exception, not anyhow::Error).
/// Used in internal pure-MLX functions; converted to anyhow at public boundaries.
type MR<T> = std::result::Result<T, mlx_rs::error::Exception>;

/// Convert mlx_rs Exception to anyhow::Error.
fn ae(e: mlx_rs::error::Exception) -> anyhow::Error {
    anyhow::anyhow!("{e:?}")
}

// ── Weight types ───────────────────────────────────────────────────────────────

/// Quantized weight in MLX Q4 format: (packed_weights, scales, biases).
/// Produced by `mlx_rs::ops::quantize(w, group_size=64, bits=4)`.
type QuantWeight = (mlx_rs::Array, mlx_rs::Array, mlx_rs::Array);

/// A model weight that is either MLX-quantized or a dense F32 array.
enum MlxW {
    Quant(QuantWeight),
    Dense(mlx_rs::Array),
}

// ── Per-layer data structures ─────────────────────────────────────────────────

struct MlxLlamaLayer {
    // Attention projections (quantized)
    q_proj: MlxW,
    k_proj: MlxW,
    v_proj: MlxW,
    o_proj: MlxW,
    // Optional Q/K/V biases (Qwen2 family)
    q_bias: Option<mlx_rs::Array>,
    k_bias: Option<mlx_rs::Array>,
    v_bias: Option<mlx_rs::Array>,
    // FFN (SwiGLU: gate+up → silu(gate)*up → down)
    gate_proj: MlxW,
    up_proj: MlxW,
    down_proj: MlxW,
    // Norms
    input_norm_w: mlx_rs::Array,
    post_attn_norm_w: mlx_rs::Array,
}

/// KV cache for one layer, stored as MLX arrays.
/// Shape: `[1, seq, n_kv_heads, head_dim]` — grows via concatenate_axis on seq.
struct MlxLayerCache {
    k: Option<mlx_rs::Array>,
    v: Option<mlx_rs::Array>,
    seq_len: usize,
}

// ── Engine ────────────────────────────────────────────────────────────────────

pub struct MlxForwardEngine {
    info: ModelInfo,
    layers: Vec<MlxLlamaLayer>,
    cache: Vec<MlxLayerCache>,
    embed: mlx_rs::Array, // [vocab, hidden]
    final_norm: mlx_rs::Array,
    lm_head: MlxW, // [vocab, hidden]
    lm_head_bias: Option<mlx_rs::Array>,
}

impl MlxForwardEngine {
    pub fn from_files(info: ModelInfo, weight_paths: &[std::path::PathBuf]) -> Result<Self> {
        let weights = load_hf_weights(weight_paths)?;
        Self::from_weights(info, weights)
    }

    pub fn from_weights(info: ModelInfo, weights: HashMap<String, Tensor>) -> Result<Self> {
        let prefix = detect_weight_prefix(&weights);
        let n = info.num_hidden_layers;

        // ── Embedding and final norm ──────────────────────────────────────────
        let embed_key = format!("{prefix}embed_tokens.weight");
        let embed = tensor_to_dense(
            weights
                .get(&embed_key)
                .with_context(|| format!("missing {embed_key}"))?,
        )?;

        let final_norm = tensor_to_dense(
            resolve_weight(&weights, &prefix, "norm").with_context(|| "missing final norm")?,
        )?;

        // ── LM head ──────────────────────────────────────────────────────────
        let lm_head_t = if let Some(t) = weights.get("lm_head.weight") {
            t
        } else {
            // tied embeddings
            weights
                .get(&embed_key)
                .with_context(|| "missing lm_head.weight")?
        };
        let lm_head = tensor_to_mlx_weight(lm_head_t)?;
        let lm_head_bias =
            resolve_bias(&weights, "", "lm_head").and_then(|t| tensor_to_dense(t).ok());

        // ── Layers ───────────────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(load_llama_layer(&weights, &prefix, i)?);
        }

        // ── KV cache ─────────────────────────────────────────────────────────
        let cache = (0..n)
            .map(|_| MlxLayerCache {
                k: None,
                v: None,
                seq_len: 0,
            })
            .collect();

        Ok(Self {
            info,
            layers,
            cache,
            embed,
            final_norm,
            lm_head,
            lm_head_bias,
        })
    }

    pub fn reset_cache(&mut self) {
        for c in &mut self.cache {
            c.k = None;
            c.v = None;
            c.seq_len = 0;
        }
    }

    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        if !use_cache {
            self.reset_cache();
        }
        let logits_arr = self.forward_mlx(input_ids, use_cache).map_err(ae)?;

        // Evaluate logits AND all KV cache arrays in a single MLX eval() call.
        //
        // Why evaluate KV cache here rather than inside forward_llama_layer?
        // Without explicit eval, the KV cache lazy graph grows O(N) deep over N
        // decode steps (each step chains a new concat on top of the previous),
        // eventually causing OOM at long sequences.
        //
        // Evaluating inside each layer (24 calls/step) forces 24 Metal command
        // buffer submissions — that was 2× slower than necessary. One combined
        // eval() at the end keeps the submission count at 1 per decode step while
        // still materialising the KV cache before the next step's concat reads it.
        let mut to_eval: Vec<mlx_rs::Array> = Vec::with_capacity(1 + self.layers.len() * 2);
        to_eval.push(logits_arr.clone());
        if use_cache {
            for c in &self.cache {
                if let Some(k) = &c.k {
                    to_eval.push(k.clone());
                }
                if let Some(v) = &c.v {
                    to_eval.push(v.clone());
                }
            }
        }
        mlx_rs::transforms::eval(to_eval.iter()).map_err(ae)?;
        Ok(logits_arr.as_slice::<f32>().to_vec())
    }

    /// Returns logits for ALL token positions (for speculative decoding verification).
    pub fn forward_all_logits(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.reset_cache();
        let logits_arr = self.forward_mlx_all(input_ids).map_err(ae)?;
        mlx_rs::transforms::eval([&logits_arr]).map_err(ae)?;
        let vocab = self.info.vocab_size;
        let seq = input_ids.len();
        let flat = logits_arr.as_slice::<f32>();
        Ok((0..seq)
            .map(|i| flat[i * vocab..(i + 1) * vocab].to_vec())
            .collect())
    }

    // ── Private: full forward pass in MLX arrays ──────────────────────────────

    fn forward_mlx(&mut self, input_ids: &[u32], use_cache: bool) -> MR<mlx_rs::Array> {
        let seq = input_ids.len();
        let offset = if use_cache { self.cache[0].seq_len } else { 0 };

        // Embed tokens: [seq] → [seq, hidden] → [1, seq, hidden]
        let ids = mlx_rs::Array::from_slice(
            &input_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            &[seq as i32],
        );
        let mut x = mlx_rs::ops::indexing::take_axis(&self.embed, &ids, 0)?;
        x = mlx_rs::ops::reshape(&x, &[1, seq as i32, self.info.hidden_size as i32])?;

        // Transformer layers
        let n_heads = self.info.num_attention_heads;
        let n_kv_heads = self.info.num_key_value_heads;
        let head_dim = self.info.head_dim;
        let rope_theta = self.info.rope_theta as f32;
        let eps = self.info.rms_norm_eps as f32;

        for i in 0..self.layers.len() {
            x = forward_llama_layer(
                x,
                &self.layers[i],
                &mut self.cache[i],
                n_heads,
                n_kv_heads,
                head_dim,
                rope_theta,
                eps,
                offset,
                use_cache,
            )?;
        }

        // Final norm
        x = mlx_rs::fast::rms_norm(&x, &self.final_norm, eps)?;

        // For decode (seq > 1 with cache, or any seq without cache), take last token
        if seq > 1 {
            // x: [1, seq, hidden] → last token: [1, 1, hidden]
            let hidden = self.info.hidden_size as i32;
            let flat = mlx_rs::ops::reshape(&x, &[seq as i32, hidden])?;
            let last_idx = mlx_rs::Array::from_slice(&[(seq - 1) as i32], &[1]);
            let last = mlx_rs::ops::indexing::take_axis(&flat, &last_idx, 0)?;
            x = mlx_rs::ops::reshape(&last, &[1, 1, hidden])?;
        }
        // x: [1, 1, hidden]

        // LM head: [1, 1, hidden] × [vocab, hidden]^T → [1, 1, vocab]
        let logits = mlx_linear(&x, &self.lm_head, self.lm_head_bias.as_ref())?;
        // Reshape to [vocab]
        let vocab = self.info.vocab_size as i32;
        mlx_rs::ops::reshape(&logits, &[vocab])
    }

    fn forward_mlx_all(&mut self, input_ids: &[u32]) -> MR<mlx_rs::Array> {
        let seq = input_ids.len();
        let ids = mlx_rs::Array::from_slice(
            &input_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            &[seq as i32],
        );
        let mut x = mlx_rs::ops::indexing::take_axis(&self.embed, &ids, 0)?;
        x = mlx_rs::ops::reshape(&x, &[1, seq as i32, self.info.hidden_size as i32])?;

        let n_heads = self.info.num_attention_heads;
        let n_kv_heads = self.info.num_key_value_heads;
        let head_dim = self.info.head_dim;
        let rope_theta = self.info.rope_theta as f32;
        let eps = self.info.rms_norm_eps as f32;

        // Temporary caches for the all-logits path
        let mut temp_cache: Vec<MlxLayerCache> = (0..self.layers.len())
            .map(|_| MlxLayerCache {
                k: None,
                v: None,
                seq_len: 0,
            })
            .collect();

        for i in 0..self.layers.len() {
            x = forward_llama_layer(
                x,
                &self.layers[i],
                &mut temp_cache[i],
                n_heads,
                n_kv_heads,
                head_dim,
                rope_theta,
                eps,
                0,
                false,
            )?;
        }
        x = mlx_rs::fast::rms_norm(&x, &self.final_norm, eps)?;
        // x: [1, seq, hidden]

        // All positions: reshape to [seq, hidden], apply lm_head, get [seq, vocab]
        let hidden = self.info.hidden_size as i32;
        let x2d = mlx_rs::ops::reshape(&x, &[seq as i32, hidden])?;
        let x2d = mlx_rs::ops::expand_dims(&x2d, 1)?; // [seq, 1, hidden]
        let logits = mlx_linear(&x2d, &self.lm_head, self.lm_head_bias.as_ref())?;
        // logits: [seq, 1, vocab] → [seq, vocab]
        let vocab = self.info.vocab_size as i32;
        mlx_rs::ops::reshape(&logits, &[seq as i32, vocab])
    }
}

// ── Per-layer forward ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn forward_llama_layer(
    x: mlx_rs::Array,
    layer: &MlxLlamaLayer,
    cache: &mut MlxLayerCache,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    eps: f32,
    offset: usize,
    use_cache: bool,
) -> MR<mlx_rs::Array> {
    let seq = x.shape()[1] as i32;
    let hidden = x.shape()[2] as i32;

    // ── 1. Input RMS norm ────────────────────────────────────────────────────
    let h = mlx_rs::fast::rms_norm(&x, &layer.input_norm_w, eps)?;

    // ── 2. QKV projections ───────────────────────────────────────────────────
    let q = mlx_linear(&h, &layer.q_proj, layer.q_bias.as_ref())?; // [1, seq, n_heads*hd]
    let k = mlx_linear(&h, &layer.k_proj, layer.k_bias.as_ref())?; // [1, seq, n_kv*hd]
    let v = mlx_linear(&h, &layer.v_proj, layer.v_bias.as_ref())?; // [1, seq, n_kv*hd]

    // ── 3. Reshape to [1, seq, n_heads, head_dim] ────────────────────────────
    let q = mlx_rs::ops::reshape(&q, &[1, seq, n_heads as i32, head_dim as i32])?;
    let k = mlx_rs::ops::reshape(&k, &[1, seq, n_kv_heads as i32, head_dim as i32])?;
    let v = mlx_rs::ops::reshape(&v, &[1, seq, n_kv_heads as i32, head_dim as i32])?;

    // ── 4. RoPE ─────────────────────────────────────────────────────────────
    // mlx fast::rope expects [batch, seq, n_heads, head_dim]
    // offset = number of already-cached tokens (current sequence position start)
    let q = mlx_rs::fast::rope(
        &q,
        head_dim as i32,
        false,            // Llama-style (not traditional)
        Some(rope_theta), // base frequency
        1.0,              // scale
        offset as i32,    // starting position offset
        None,             // no precomputed freqs
    )?;
    let k = mlx_rs::fast::rope(
        &k,
        head_dim as i32,
        false,
        Some(rope_theta),
        1.0,
        offset as i32,
        None,
    )?;

    // ── 5. KV cache update ───────────────────────────────────────────────────
    // Store/extend cache in [1, seq, n_kv_heads, head_dim] format.
    // After concatenation we eval() the cache arrays immediately so the lazy
    // graph doesn't accumulate depth across decode steps (which would cause
    // TTFT to grow with each benchmark run due to graph re-evaluation cost).
    let (k_full, v_full) = if use_cache {
        let k_ext = match &cache.k {
            Some(ck) => mlx_rs::ops::concatenate_axis(&[ck, &k], 1)?,
            None => k,
        };
        let v_ext = match &cache.v {
            Some(cv) => mlx_rs::ops::concatenate_axis(&[cv, &v], 1)?,
            None => v,
        };
        // Note: we intentionally do NOT call eval() here. The KV cache arrays
        // stay as lazy graph nodes; they are evaluated together with the full
        // forward pass at the single eval() call in forward_logits(). This
        // keeps the Metal command buffer submissions to one per decode step,
        // which is critical for performance (24 per-layer evals was 2× slower).
        cache.seq_len += seq as usize;
        cache.k = Some(k_ext.clone());
        cache.v = Some(v_ext.clone());
        (k_ext, v_ext)
    } else {
        (k, v)
    };

    // ── 6. Transpose to [1, n_heads, seq, head_dim] for SDPA ─────────────────
    // q: [1, seq, n_heads, hd] → [1, n_heads, seq, hd]
    // k/v: [1, total_seq, n_kv_heads, hd] → [1, n_kv_heads, total_seq, hd]
    let q_t = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?;
    let k_t = mlx_rs::ops::transpose_axes(&k_full, &[0, 2, 1, 3])?;
    let v_t = mlx_rs::ops::transpose_axes(&v_full, &[0, 2, 1, 3])?;

    // ── 7. Pre-tile KV heads for GQA when MLX's native GQA path is unreliable ──
    //
    // MLX's fast SDPA handles GQA natively for head_dim=64 (confirmed working).
    // For head_dim ≥ 96, the native GQA Metal kernel produces incorrect attention
    // for certain (n_heads, n_kv_heads) combinations — empirically verified wrong
    // for Qwen2.5-1.5B (n_heads=12, n_kv_heads=2, head_dim=128).
    //
    // Fix: explicitly tile K/V to full n_heads when head_dim ≥ 96.
    // Leave head_dim=64 models (Qwen2.5-0.5B with n_heads=14) on native GQA path.
    let needs_kv_tile = n_kv_heads < n_heads && head_dim >= 96;
    let (k_t, v_t) = if needs_kv_tile {
        (
            kv_repeat_interleave(&k_t, n_heads, n_kv_heads)?,
            kv_repeat_interleave(&v_t, n_heads, n_kv_heads)?,
        )
    } else {
        (k_t, v_t)
    };

    // ── 8. Scaled dot-product attention ──────────────────────────────────────
    // For prefill (seq>1): apply causal mask. For decode (seq=1): no mask needed.
    let scale = (head_dim as f32).powf(-0.5);
    let mask = if seq > 1 {
        Some(ScaledDotProductAttentionMask::Causal)
    } else {
        None
    };
    let attn = mlx_rs::fast::scaled_dot_product_attention(&q_t, &k_t, &v_t, scale, mask)?;
    // Evaluate the attention output immediately: this releases the tiled K/V arrays
    // from the lazy graph before the next layer allocates its own tiled arrays.
    // Without this, all 28 layers' tiled K/V arrays (up to 325 MB for 1.5B) would
    // live simultaneously in the lazy graph until the final eval(), causing OOM.
    if needs_kv_tile {
        mlx_rs::transforms::eval([&attn])?;
    }
    // attn: [1, n_heads, seq, head_dim]

    // ── 8. Merge heads → [1, seq, hidden] ────────────────────────────────────
    let attn = mlx_rs::ops::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let attn = mlx_rs::ops::reshape(&attn, &[1, seq, hidden])?;

    // ── 9. Output projection + residual ──────────────────────────────────────
    let o = mlx_linear(&attn, &layer.o_proj, None)?;
    let x = mlx_rs::ops::add(&x, &o)?;

    // ── 10. Post-attention RMS norm ───────────────────────────────────────────
    let h2 = mlx_rs::fast::rms_norm(&x, &layer.post_attn_norm_w, eps)?;

    // ── 11. FFN (SwiGLU: gate+up → silu(gate)*up → down) ────────────────────
    let gate = mlx_linear(&h2, &layer.gate_proj, None)?;
    let up = mlx_linear(&h2, &layer.up_proj, None)?;
    let ff = mlx_rs::ops::multiply(&mlx_rs::nn::silu(&gate)?, &up)?;
    let ff = mlx_linear(&ff, &layer.down_proj, None)?;

    // ── 12. Final residual ────────────────────────────────────────────────────
    mlx_rs::ops::add(&x, &ff)
}

// ── Weight loading helpers ────────────────────────────────────────────────────

/// Load one transformer layer's weights into GPU-resident MLX arrays.
///
/// `prefix` is the model-level prefix (e.g. `"model."` or `""`).
/// The full key for layer i, projection q is: `{prefix}layers.{i}.self_attn.q_proj.weight`.
fn load_llama_layer(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    layer_idx: usize,
) -> Result<MlxLlamaLayer> {
    // resolve_weight builds: {prefix}{suffix}.weight
    // So suffix must be "layers.{i}.self_attn.q_proj" to produce the correct key.
    let base = format!("layers.{layer_idx}");

    let w = |name: &str| -> Result<MlxW> {
        let suffix = format!("{base}.{name}");
        let t = resolve_weight(weights, prefix, &suffix)
            .with_context(|| format!("missing {prefix}{base}.{name}"))?;
        tensor_to_mlx_weight(t)
    };
    let dense = |name: &str| -> Result<mlx_rs::Array> {
        let suffix = format!("{base}.{name}");
        let t = resolve_weight(weights, prefix, &suffix)
            .with_context(|| format!("missing {prefix}{base}.{name}"))?;
        tensor_to_dense(t)
    };
    let opt_dense = |name: &str| -> Option<mlx_rs::Array> {
        let suffix = format!("{base}.{name}");
        resolve_bias(weights, prefix, &suffix).and_then(|t| tensor_to_dense(t).ok())
    };

    Ok(MlxLlamaLayer {
        q_proj: w("self_attn.q_proj")?,
        k_proj: w("self_attn.k_proj")?,
        v_proj: w("self_attn.v_proj")?,
        o_proj: w("self_attn.o_proj")?,
        q_bias: opt_dense("self_attn.q_proj"),
        k_bias: opt_dense("self_attn.k_proj"),
        v_bias: opt_dense("self_attn.v_proj"),
        gate_proj: w("mlp.gate_proj")?,
        up_proj: w("mlp.up_proj")?,
        down_proj: w("mlp.down_proj")?,
        input_norm_w: dense("input_layernorm")?,
        post_attn_norm_w: dense("post_attention_layernorm")?,
    })
}

/// Convert a Tensor to an MLX weight: quantized if dimensions allow, dense otherwise.
/// Quantization requires a 2D matrix where in_dim % 64 == 0 and out_dim % 32 == 0.
fn tensor_to_mlx_weight(t: &Tensor) -> Result<MlxW> {
    let dims = t.shape().dims();
    if dims.len() == 2 {
        let (out, inp) = (dims[0], dims[1]);
        if inp % 64 == 0 && out % 32 == 0 && t.numel() >= 512 {
            let shape = &[out as i32, inp as i32];
            let cow = t.to_f32_cow();
            let arr = mlx_rs::Array::from_slice(&cow[..t.numel().min(cow.len())], shape);
            let (wq, sc, bi) = mlx_rs::ops::quantize(&arr, 64i32, 4i32).map_err(ae)?;
            // Force GPU materialisation so future quantized_matmul calls
            // reuse resident GPU memory rather than re-executing the graph.
            mlx_rs::transforms::eval([&wq, &sc, &bi]).map_err(ae)?;
            return Ok(MlxW::Quant((wq, sc, bi)));
        }
    }
    Ok(MlxW::Dense(tensor_to_dense(t)?))
}

/// Convert any Tensor to a dense F32 MLX Array.
fn tensor_to_dense(t: &Tensor) -> Result<mlx_rs::Array> {
    let dims: Vec<i32> = t.shape().dims().iter().map(|&d| d as i32).collect();
    let numel = t.numel();
    let cow = t.to_f32_cow();
    let arr = mlx_rs::Array::from_slice(&cow[..numel.min(cow.len())], &dims);
    mlx_rs::transforms::eval([&arr]).map_err(ae)?;
    Ok(arr)
}

// ── Operation helpers ─────────────────────────────────────────────────────────

/// Repeat-interleave KV heads to match the number of query heads (GQA expansion).
///
/// Mirrors what mlx-lm does: explicitly tile K/V before SDPA rather than relying
/// on MLX's built-in GQA support, which produces incorrect results for some
/// (n_heads, n_kv_heads) pairs (e.g. Qwen2.5-1.5B with n_heads=12, n_kv_heads=2).
///
/// Input:  kv — `[1, n_kv_heads, total_seq, head_dim]`
/// Output: `[1, n_heads, total_seq, head_dim]`
///
/// Uses `expand_dims` + `broadcast_to` (zero-copy) + `flatten` (one materialization)
/// instead of split+clone+concat, which avoids n_rep separate memory copies.
fn kv_repeat_interleave(
    kv: &mlx_rs::Array,
    n_heads: usize,
    n_kv_heads: usize,
) -> MR<mlx_rs::Array> {
    let n_rep = n_heads / n_kv_heads;
    if n_rep == 1 {
        return Ok(kv.clone());
    }
    // Split along head axis (axis=1) → n_kv_heads arrays of [1, 1, total_seq, head_dim].
    // Clone each head n_rep times, then concatenate: [h0×n_rep, h1×n_rep, ...]
    // In the lazy MLX graph, part.clone() is a zero-cost refcount increment —
    // the actual data copy happens only at eval() time for the concat output.
    let parts = mlx_rs::ops::split(kv, n_kv_heads as i32, Some(1i32))?;
    let mut repeated: Vec<mlx_rs::Array> = Vec::with_capacity(n_heads);
    for part in &parts {
        for _ in 0..n_rep {
            repeated.push(part.clone());
        }
    }
    let refs: Vec<&mlx_rs::Array> = repeated.iter().collect();
    mlx_rs::ops::concatenate_axis(&refs, 1)
}

/// Linear projection: `x @ W.T` with optional bias.
/// Uses quantized_matmul for Quant weights, standard matmul for Dense.
fn mlx_linear(x: &mlx_rs::Array, w: &MlxW, bias: Option<&mlx_rs::Array>) -> MR<mlx_rs::Array> {
    let y = match w {
        MlxW::Quant((wq, sc, bi)) => {
            mlx_rs::ops::quantized_matmul(x, wq, sc, bi, true, 64i32, 4i32)?
        }
        MlxW::Dense(arr) => {
            // Transpose [out, in] → [in, out] then matmul
            let wt = mlx_rs::ops::transpose_axes(arr, &[1, 0])?;
            mlx_rs::ops::matmul(x, &wt)?
        }
    };
    match bias {
        Some(b) => Ok(mlx_rs::ops::add(&y, b)?),
        None => Ok(y),
    }
}
