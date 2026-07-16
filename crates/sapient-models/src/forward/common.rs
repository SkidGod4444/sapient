// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Shared tensor ops for transformer forward passes.

use anyhow::Result;
use sapient_backends_cpu::kernels::{self, attention, layernorm, matmul, quant, rope};
use sapient_core::error::SapientError;
use sapient_core::{DType, Shape, Tensor};

fn map_err<T>(result: std::result::Result<T, SapientError>) -> Result<T> {
    result.map_err(|e| anyhow::anyhow!("{e}"))
}

// ── KV-cache context window ──────────────────────────────────────────────────

/// Default cap on the KV-cache context window allocated at load time.
///
/// The KV cache is pre-allocated for `max_seq` positions up front. Modern models
/// advertise enormous context windows (Llama-3.1 / DeepSeek-R1 = 131072), and at
/// 8 KV heads × 128 head_dim × 32 layers that is ~9 GB of Q8_0 cache for an 8B
/// model — enough to OOM-kill a 16 GB device during *load*, before a single token
/// is generated. We cap the allocation to a sane chat window; longer
/// conversations slide the window (see [`update_kv_cache`]).
pub const DEFAULT_KV_CACHE_CTX: usize = 8192;

/// Resolve the KV-cache context window: `min(model_max, cap)`, where `cap`
/// defaults to [`DEFAULT_KV_CACHE_CTX`] and can be overridden (up to the model
/// maximum) via the `SAPIENT_CTX` environment variable.
pub fn kv_cache_ctx(model_max: usize) -> usize {
    let cap = std::env::var("SAPIENT_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_KV_CACHE_CTX);
    model_max.min(cap).max(1)
}

// ── Online F16 → Q8_0 quantization at load time ──────────────────────────────

/// Returns true if a weight tensor should be quantized online to Q8_0.
///
/// Criteria:
/// - Must be a 2-D matrix with at least 32 elements (one Q8_0 block).
/// - Must have dtype F16 or BF16 (safetensors weight matrices).
/// - Must not be a norm weight, bias, embedding table, or lm_head
///   (these have different access patterns or are tiny).
pub fn should_quantize_online(name: &str, t: &Tensor) -> bool {
    let dims = t.shape().dims();
    if dims.len() != 2 {
        return false;
    }
    let numel = dims[0] * dims[1];
    if numel < 32 || numel % 32 != 0 {
        return false;
    }
    // Skip small helper tensors and anything already quantized. The MoE router
    // gate (`block_sparse_moe.gate`) is precision-sensitive — routing decisions
    // flip under Q8_0 rounding — and llama.cpp keeps it full precision, so skip
    // it too. (Dense `mlp.gate_proj` does NOT contain "block_sparse_moe.gate".)
    let skip = ["norm", "bias", "embed", "lm_head", "block_sparse_moe.gate"];
    if skip.iter().any(|s| name.contains(s)) {
        return false;
    }
    matches!(t.dtype(), DType::F16 | DType::BF16)
}

/// Quantize a 2-D F16/BF16 weight tensor to Q8_0 in one pass.
///
/// The F16→F32 dequantization happens once here at load time; all subsequent
/// decode steps use the already-NEON-optimized Q8_0 kernel (~1 byte/weight vs
/// 2 bytes/weight for F16, and avoids the per-step F16→F32 conversion cost).
pub fn quantize_tensor_to_q8_0(t: Tensor) -> Tensor {
    let shape = t.shape().dims().to_vec();
    let numel = shape[0] * shape[1];
    debug_assert_eq!(numel % 32, 0);

    let f32_data = t.to_f32_vec(); // one-time dequantization
    let n_blocks = numel / 32;
    let mut q8_bytes = Vec::with_capacity(n_blocks * 34);
    for block in f32_data.chunks_exact(32) {
        q8_bytes.extend_from_slice(&quant::quantize_q8_0_block(block));
    }

    Tensor::from_quant_bytes(&q8_bytes, shape, DType::Q8_0).unwrap_or(t)
}

/// Repack heap-resident Q4_K weight matrices into the row-interleaved Q4_K_R4
/// layout for the multi-row SDOT GEMV (one contiguous weight stream per task —
/// see `repack_q4_k_rows4`). Eligibility: 2-D, rows % 4 == 0, k % 256 == 0,
/// heap-backed (mmap tensors must stay paged), and NOT the embedding table
/// (row-gathered, must stay row-major; a tied lm_head is the embedding and is
/// therefore skipped too). `SAPIENT_NO_REPACK=1` disables (escape hatch +
/// A/B benching).
#[cfg(target_arch = "aarch64")]
pub fn repack_q4_k_weights(
    weights: std::collections::HashMap<String, Tensor>,
    embed_key: &str,
) -> std::collections::HashMap<String, Tensor> {
    if std::env::var("SAPIENT_NO_REPACK").is_ok_and(|v| v == "1") {
        return weights;
    }
    let mut repacked = 0usize;
    let out = weights
        .into_iter()
        .map(|(name, t)| {
            let dims = t.shape().dims().to_vec();
            // Q4_K repacks by default (measured: Pi +7%, M4 +5% over plain
            // multi-row). Q6_K repack defaults ON where i8mm exists because the
            // SMMLA x2 prefill kernel consumes the R4 layout (M4 prefill 1.16×);
            // decode is NEUTRAL either way on a cool machine (A/B'd — an earlier
            // +13% reading was thermal-order artifact). OFF elsewhere (Pi/A76:
            // neutral-to-slightly-negative, no i8mm to exploit it). Override
            // either way with SAPIENT_REPACK_Q6K=1/0.
            let q6_opt_in = match std::env::var("SAPIENT_REPACK_Q6K").as_deref() {
                Ok("1") => true,
                Ok("0") => false,
                _ => std::arch::is_aarch64_feature_detected!("i8mm"),
            };
            let eligible = (t.dtype() == DType::Q4_K || (t.dtype() == DType::Q6_K && q6_opt_in))
                && !t.is_mmap()
                && dims.len() == 2
                && dims[0] % 4 == 0
                && dims[1] % 256 == 0
                && name != embed_key;
            if eligible {
                let (packed, r4_dtype) = match t.dtype() {
                    DType::Q4_K => (
                        kernels::quant::repack_q4_k_rows4(t.as_quant_blocks(), dims[0], dims[1]),
                        DType::Q4_K_R4,
                    ),
                    _ => (
                        kernels::quant::repack_q6_k_rows4(t.as_quant_blocks(), dims[0], dims[1]),
                        DType::Q6_K_R4,
                    ),
                };
                if let Ok(r4) = Tensor::from_quant_bytes(&packed, dims, r4_dtype) {
                    repacked += 1;
                    return (name, r4);
                }
            }
            (name, t)
        })
        .collect();
    if repacked > 0 {
        tracing::debug!(
            repacked,
            "Q4_K weights repacked to Q4_K_R4 (multi-row GEMV)"
        );
    }
    out
}

/// Gather token embeddings: weight `[vocab, hidden]`, ids `[seq]` → `[1, seq, hidden]`.
///
/// **Row-wise, never whole-table** (Phase 8.3 finding): this runs every decode
/// step, and the old `to_f32_cow()` fast path silently dequantized the ENTIRE
/// table per token for quantized GGUF embeddings — for Llama-3.2-1B (tied Q6_K
/// embed, 128k vocab × 2048) that was a ~1 GB f32 allocation + 262M-element
/// dequant per generated token, the dominant decode cost. Each quantized row is
/// a contiguous `byte_count(hidden)` slice (ggml quantizes per row;
/// `hidden % block == 0` guaranteed for quantized tables), so we dequantize only
/// the `seq_len` rows actually needed.
pub fn embed_tokens(weight: &Tensor, input_ids: &[u32]) -> Result<Tensor> {
    let dims = weight.shape().dims();
    let (vocab, hidden) = (dims[0], dims[1]);
    let seq_len = input_ids.len();
    let mut out = vec![0.0f32; seq_len * hidden];

    for (i, &id) in input_ids.iter().enumerate() {
        if id as usize >= vocab {
            anyhow::bail!("token id {id} out of vocab range");
        }
        let dst = &mut out[i * hidden..(i + 1) * hidden];
        gather_row_f32(weight, id as usize, hidden, dst)?;
    }

    Tensor::from_f32(&out, Shape::new([1, seq_len, hidden])).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Dequantize/convert one row of a `[vocab, hidden]` table into `dst` (len =
/// `hidden`) without touching the rest of the table. Bit-identical to slicing
/// the row out of a full `to_f32_vec()` (per-row block layout for quantized
/// dtypes; plain per-element conversion for floats).
fn gather_row_f32(weight: &Tensor, row: usize, hidden: usize, dst: &mut [f32]) -> Result<()> {
    use sapient_core::DType;
    match weight.dtype() {
        DType::F32 => {
            let w = weight.as_f32_slice();
            dst.copy_from_slice(&w[row * hidden..(row + 1) * hidden]);
        }
        DType::F16 => {
            let bytes = &weight.as_bytes()[row * hidden * 2..(row + 1) * hidden * 2];
            for (d, b) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
                *d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        DType::BF16 => {
            let bytes = &weight.as_bytes()[row * hidden * 2..(row + 1) * hidden * 2];
            for (d, b) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
                *d = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        DType::Q4_K_R4 | DType::Q6_K_R4 => anyhow::bail!(
            "embedding table is row-interleaved (R4) — repacking must skip embeddings"
        ),
        dt if dt.is_quantized() => {
            let row_bytes = dt.byte_count(hidden);
            let blocks = &weight.as_quant_blocks()[row * row_bytes..(row + 1) * row_bytes];
            let t = Tensor::from_quant_bytes(blocks, vec![1, hidden], dt)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            dst.copy_from_slice(&t.to_f32_vec());
        }
        other => anyhow::bail!("unsupported embedding dtype {other}"),
    }
    Ok(())
}

/// Linear on 3-D activations: `[1, seq, in] @ W^T` where W is `[out, in]`.
pub fn linear_3d(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        anyhow::bail!("linear_3d expects [batch, seq, hidden]");
    }
    let (batch, seq, in_dim) = (dims[0], dims[1], dims[2]);
    let w_dims = weight.shape().dims();
    if w_dims.len() != 2 {
        anyhow::bail!("linear weight must be 2-D");
    }
    let out_dim = w_dims[0];
    if w_dims[1] != in_dim {
        anyhow::bail!("linear weight in_dim mismatch: {} vs {in_dim}", w_dims[1]);
    }

    let x2d = map_err(x.reshape(vec![batch * seq, in_dim]))?;
    // weight is [out, in] (PyTorch nn.Linear layout); matmul_nt computes x @ weightᵀ
    // directly, honouring the layout and any F16/BF16 weight dtype.
    let y2d = map_err(matmul::matmul_nt(&x2d, weight))?;
    map_err(y2d.reshape(vec![batch, seq, out_dim]))
}

/// Reshape `[1, seq, n_heads * head_dim]` → `[1, n_heads, seq, head_dim]`.
pub fn split_heads(x: &Tensor, n_heads: usize, head_dim: usize) -> Result<Tensor> {
    let seq = x.shape().dims()[1];
    permute(
        &map_err(x.reshape(vec![1, seq, n_heads, head_dim]))?,
        &[0, 2, 1, 3],
    )
}

/// Merge heads back: `[1, n_heads, seq, head_dim]` → `[1, seq, n_heads * head_dim]`.
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims();
    let (n_heads, seq, head_dim) = (d[1], d[2], d[3]);
    permute(x, &[0, 2, 1, 3])?
        .reshape(vec![1, seq, n_heads * head_dim])
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn permute(x: &Tensor, order: &[usize]) -> Result<Tensor> {
    let dims = x.shape().dims();
    if order.len() != dims.len() {
        anyhow::bail!("permute rank mismatch");
    }
    let new_dims: Vec<usize> = order.iter().map(|&i| dims[i]).collect();
    let src = x.as_f32_slice();
    let mut out = vec![0.0f32; src.len()];

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        dims: &[usize],
        order: &[usize],
        src: &[f32],
        out: &mut [f32],
        src_strides: &[usize],
        dst_strides: &[usize],
        idx: &mut [usize],
        depth: usize,
    ) {
        if depth == dims.len() {
            let src_off: usize = idx
                .iter()
                .zip(src_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            let dst_off: usize = order
                .iter()
                .enumerate()
                .map(|(dst_ax, &src_ax)| idx[src_ax] * dst_strides[dst_ax])
                .sum();
            out[dst_off] = src[src_off];
            return;
        }
        for i in 0..dims[depth] {
            idx[depth] = i;
            recurse(
                dims,
                order,
                src,
                out,
                src_strides,
                dst_strides,
                idx,
                depth + 1,
            );
        }
    }

    let src_strides = strides_for(dims);
    let dst_strides = strides_for(&new_dims);
    let mut idx = vec![0usize; dims.len()];
    recurse(
        dims,
        order,
        src,
        &mut out,
        &src_strides,
        &dst_strides,
        &mut idx,
        0,
    );
    Tensor::from_f32(&out, Shape::new(new_dims)).map_err(|e| anyhow::anyhow!("{e}"))
}

fn strides_for(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Quantize 32 `f32` values into a single Q8_0 block (2-byte f16 scale + 32 × i8).
/// Returns the 34-byte block in ggml layout.
#[inline]
fn quantize_f32_to_q8_0_block(data: &[f32]) -> [u8; 34] {
    debug_assert_eq!(data.len(), 32, "Q8_0 block must have exactly 32 elements");
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = max_abs / 127.0;
    let d = half::f16::from_f32(scale);
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let mut block = [0u8; 34];
    block[0..2].copy_from_slice(&d.to_le_bytes());
    for (i, &v) in data.iter().enumerate() {
        block[2 + i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8 as u8;
    }
    block
}

/// Update the pre-allocated KV cache in place and return a view of length `seq_len + new_seq`.
///
/// When the cache holds Q8_0 blocks (quantized KV cache), the new F32 values are
/// quantized on write and the returned tensor is a freshly-allocated F32 tensor
/// (dequantized from the cache). When the cache is F32, the old in-place path is used.
pub fn update_kv_cache(
    cache: &mut Tensor,
    current_seq_len: usize,
    new_k: &Tensor,
) -> Result<Tensor> {
    let cd = cache.shape().dims().to_vec();
    let nd = new_k.shape().dims().to_vec();

    if cd.len() != 4 || nd.len() != 4 {
        anyhow::bail!("update_kv_cache expects 4-D tensors");
    }
    if cd[0] != nd[0] || cd[1] != nd[1] || cd[3] != nd[3] {
        anyhow::bail!("update_kv_cache shape mismatch");
    }

    // Dispatch to the Q8_0-quantized path when the cache holds packed blocks.
    if cache.dtype() == DType::Q8_0 {
        return update_kv_cache_q8(cache, &cd, &nd, current_seq_len, new_k);
    }

    let max_seq = cd[2];
    let new_seq = nd[2];

    if new_seq > max_seq {
        anyhow::bail!("new tokens {} exceeds max cache size {}", new_seq, max_seq);
    }

    let mut total_seq = current_seq_len + new_seq;
    let shift = total_seq.saturating_sub(max_seq);

    let (b_sz, h, hd) = (cd[0], cd[1], cd[3]);
    let new_k_slice = new_k.as_f32_slice();
    let cache_strides = cache.strides().to_vec();

    {
        let cache_slice = cache.as_f32_slice_mut()?;

        // If we need to shift, move existing elements left
        if shift > 0 {
            let keep_seq = current_seq_len - shift;
            for bi in 0..b_sz {
                for hi in 0..h {
                    let cache_base = bi * cache_strides[0] + hi * cache_strides[1];
                    for si in 0..keep_seq {
                        let src_idx = cache_base + (si + shift) * cache_strides[2];
                        let dst_idx = cache_base + si * cache_strides[2];
                        cache_slice.copy_within(src_idx..src_idx + hd, dst_idx);
                    }
                }
            }
        }

        // Now append the new tokens
        let insert_pos = if shift > 0 {
            current_seq_len - shift
        } else {
            current_seq_len
        };
        for bi in 0..b_sz {
            for hi in 0..h {
                let cache_base =
                    bi * cache_strides[0] + hi * cache_strides[1] + insert_pos * cache_strides[2];
                let new_base = ((bi * h + hi) * new_seq) * hd; // new_k is assumed contiguous from split_heads

                for si in 0..new_seq {
                    let c_idx = cache_base + si * cache_strides[2];
                    let n_idx = new_base + si * hd;

                    // Copy head_dim elements
                    cache_slice[c_idx..c_idx + hd].copy_from_slice(&new_k_slice[n_idx..n_idx + hd]);
                }
            }
        }
    }

    if shift > 0 {
        total_seq = max_seq;
    }

    // Return a sliced view of the cache from 0 to total_seq
    cache
        .slice_axis(2, 0, total_seq)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Q8_0-quantized KV cache update.
///
/// Writes new F32 tokens as Q8_0 blocks into the packed cache buffer by copying the
/// existing bytes, mutating them, then swapping the cache tensor in place.
/// Returns a freshly-allocated contiguous F32 tensor (dequantized from the live prefix)
/// suitable for the attention kernel.
///
/// Buffer layout: flat row-major over [b, h, seq_pos], each position is
/// `blocks_per_head * 34` bytes (one Q8_0 block per 32 head_dim elements).
fn update_kv_cache_q8(
    cache: &mut Tensor,
    cd: &[usize],
    nd: &[usize],
    current_seq_len: usize,
    new_k: &Tensor,
) -> Result<Tensor> {
    let (b_sz, h, max_seq, hd) = (cd[0], cd[1], cd[2], cd[3]);
    let new_seq = nd[2];

    if new_seq > max_seq {
        anyhow::bail!("new tokens {} exceeds max cache size {}", new_seq, max_seq);
    }

    let blocks_per_head = hd / 32;
    let bytes_per_pos = blocks_per_head * 34;
    let mut total_seq = current_seq_len + new_seq;
    let shift = total_seq.saturating_sub(max_seq);

    let pos_off = |bi: usize, hi: usize, si: usize| -> usize {
        (bi * h * max_seq + hi * max_seq + si) * bytes_per_pos
    };

    // In-place mutation via as_bytes_mut — zero allocation, zero copy.
    let cache_bytes = cache.as_bytes_mut()?;

    if shift > 0 {
        let keep_seq = current_seq_len - shift;
        for bi in 0..b_sz {
            for hi in 0..h {
                for si in 0..keep_seq {
                    let src = pos_off(bi, hi, si + shift);
                    let dst = pos_off(bi, hi, si);
                    cache_bytes.copy_within(src..src + bytes_per_pos, dst);
                }
            }
        }
    }

    let insert_pos = if shift > 0 {
        current_seq_len - shift
    } else {
        current_seq_len
    };
    let new_k_f32 = new_k.to_f32_vec();

    for bi in 0..b_sz {
        for hi in 0..h {
            for si in 0..new_seq {
                let dst_start = pos_off(bi, hi, insert_pos + si);
                let src_f32_start = (bi * h * new_seq + hi * new_seq + si) * hd;
                let src_f32 = &new_k_f32[src_f32_start..src_f32_start + hd];
                for blk in 0..blocks_per_head {
                    let encoded = quantize_f32_to_q8_0_block(&src_f32[blk * 32..(blk + 1) * 32]);
                    cache_bytes[dst_start + blk * 34..dst_start + blk * 34 + 34]
                        .copy_from_slice(&encoded);
                }
            }
        }
    }

    if shift > 0 {
        total_seq = max_seq;
    }

    // Dequantize the live prefix to F32 for the attention kernel.
    // Read directly from the (now-updated) in-place cache.
    let cache_ro = cache.as_bytes();
    let out_numel = b_sz * h * total_seq * hd;
    let mut out_f32 = vec![0.0f32; out_numel];

    for bi in 0..b_sz {
        for hi in 0..h {
            for si in 0..total_seq {
                let src_start = pos_off(bi, hi, si);
                let dst_f32_start = (bi * h * total_seq + hi * total_seq + si) * hd;
                for blk in 0..blocks_per_head {
                    let bb = &cache_ro[src_start + blk * 34..src_start + blk * 34 + 34];
                    let d = half::f16::from_le_bytes([bb[0], bb[1]]).to_f32();
                    for j in 0..32 {
                        out_f32[dst_f32_start + blk * 32 + j] = bb[2 + j] as i8 as f32 * d;
                    }
                }
            }
        }
    }

    Tensor::from_f32_vec(out_f32, Shape::new(vec![b_sz, h, total_seq, hd]))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn apply_rope_positions(x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
    map_err(rope::apply_rope(x, positions, base))
}

/// RoPE applied to only the first `rotary_dim` channels (Phi partial rotary).
pub fn apply_rope_partial(
    x: &Tensor,
    positions: &[usize],
    base: f32,
    rotary_dim: usize,
) -> Result<Tensor> {
    map_err(rope::apply_rope_partial(x, positions, base, rotary_dim))
}

/// Add a per-feature bias `[n]` broadcast over the last dimension of `y`
/// (shape `[.., n]`). `y` must be F32; `bias` may be F16/BF16.
pub fn add_bias_last_dim(y: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = y.shape().dims().to_vec();
    let n = *dims.last().ok_or_else(|| anyhow::anyhow!("empty tensor"))?;
    let bias_cow = bias.to_f32_cow();
    let b = bias_cow.as_ref();
    if b.len() != n {
        anyhow::bail!("bias length {} does not match last dim {n}", b.len());
    }
    let mut data = y.as_f32_slice().to_vec();
    for (i, v) in data.iter_mut().enumerate() {
        *v += b[i % n];
    }
    map_err(Tensor::from_f32(&data, Shape::new(dims)))
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    map_err(layernorm::rms_norm(x, Some(weight), eps))
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    map_err(layernorm::layer_norm(x, Some(weight), bias, -1, eps))
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::silu(x))
}

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::gelu(x))
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::add(a, b))
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::mul(a, b))
}

pub fn gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_kv_heads: usize,
    causal: bool,
) -> Result<Tensor> {
    let mask = if causal {
        let sq = q.shape().dims()[2];
        let sk = k.shape().dims()[2];
        Some(attention::causal_mask(sq, sk))
    } else {
        None
    };
    map_err(attention::scaled_dot_product_attention(
        q,
        k,
        v,
        mask.as_ref(),
        None,
        n_kv_heads,
    ))
}

/// Compute logits for ALL positions in the sequence. Used by speculative
/// decoding to verify K draft tokens in a single target-model forward pass.
pub fn all_logits_from_hidden(hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<Vec<f32>>> {
    let dims = hidden.shape().dims();
    let hidden_size = dims[2];
    let seq = dims[1];
    let vocab_size = lm_head.shape().dims()[0];
    let h = hidden.as_f32_slice();
    let h_all =
        Tensor::from_f32(h, Shape::new([seq, hidden_size])).map_err(|e| anyhow::anyhow!("{e}"))?;
    let logits_flat = map_err(matmul::matmul_nt(&h_all, lm_head))?;
    let flat = logits_flat.as_f32_slice();
    let mut all = Vec::with_capacity(seq);
    for i in 0..seq {
        all.push(flat[i * vocab_size..(i + 1) * vocab_size].to_vec());
    }
    Ok(all)
}

pub fn logits_from_hidden(hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>> {
    // hidden: [1, seq, hidden], take last position
    let dims = hidden.shape().dims();
    let hidden_size = dims[2];
    let seq = dims[1];
    let h = hidden.as_f32_slice();
    let last = &h[(seq - 1) * hidden_size..seq * hidden_size];
    let h_last =
        Tensor::from_f32(last, Shape::new([1, hidden_size])).map_err(|e| anyhow::anyhow!("{e}"))?;
    // lm_head is [vocab, hidden]; matmul_nt computes h_last @ lm_headᵀ directly.
    let logits = map_err(matmul::matmul_nt(&h_last, lm_head))?;
    Ok(logits.as_f32_slice().to_vec())
}

pub fn mean_pool_hidden(hidden: &Tensor) -> Result<Vec<f32>> {
    let dims = hidden.shape().dims();
    let (seq, hidden_size) = (dims[1], dims[2]);
    let h = hidden.as_f32_slice();
    let mut out = vec![0.0f32; hidden_size];
    for t in 0..seq {
        for i in 0..hidden_size {
            out[i] += h[t * hidden_size + i];
        }
    }
    let n = seq as f32;
    for v in &mut out {
        *v /= n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_core::DType;

    /// Row-wise embedding gather must be bit-identical to slicing rows out of a
    /// full-table dequant — for every table dtype the GGUF/safetensors paths
    /// ship (the old implementation did the full-table dequant per decode step;
    /// this pins the replacement to the same values).
    #[test]
    fn embed_row_gather_matches_full_dequant() {
        let (vocab, hidden) = (17usize, 256usize);
        let mut s = 0xABCDu64;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let data: Vec<f32> = (0..vocab * hidden).map(|_| next()).collect();

        // Build one table per dtype from the same values.
        let f32_t = Tensor::from_f32(&data, Shape::new([vocab, hidden])).unwrap();
        let f16_bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let f16_t = Tensor::from_f16_bytes(&f16_bytes, Shape::new([vocab, hidden])).unwrap();
        let q8_t = quantize_tensor_to_q8_0(f32_t.clone());
        assert_eq!(q8_t.dtype(), DType::Q8_0);

        for table in [&f32_t, &f16_t, &q8_t] {
            let full = table.to_f32_vec();
            let ids: Vec<u32> = vec![0, 7, 16, 3];
            let got = embed_tokens(table, &ids).unwrap();
            let got = got.as_f32_slice();
            for (i, &id) in ids.iter().enumerate() {
                let want = &full[id as usize * hidden..(id as usize + 1) * hidden];
                assert_eq!(
                    &got[i * hidden..(i + 1) * hidden],
                    want,
                    "row gather differs from full dequant (dtype {:?}, id {id})",
                    table.dtype()
                );
            }
        }

        // Out-of-range id still errors.
        assert!(embed_tokens(&f32_t, &[vocab as u32]).is_err());
    }
}
