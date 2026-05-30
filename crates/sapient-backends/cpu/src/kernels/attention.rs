//! Scaled dot-product attention — Flash-Edge (tiled online-softmax) implementation.
//!
//! Implements the core attention mechanism used by all transformer LLMs:
//!   Attention(Q, K, V) = softmax(QK^T / √d_k + mask) × V
//!
//! Algorithm: Flash-Attention style *online softmax* that never materialises the
//! full seq_q × seq_k score matrix.  For each query row the loop over key/value
//! positions maintains a running max `m` and a running denominator `l`, avoiding
//! the memory blow-up of the naïve implementation (e.g. 2048×2048×4B = 16 MB for
//! a single head at seq=2048).
//!
//! Also implements:
//!   - Causal masking — applied online via -inf injection
//!   - Grouped-Query Attention (Llama2/3, Mistral) — KV head repeat
//!   - NEON SIMD dot-product and saxpby helpers on aarch64

use rayon::prelude::*;
use sapient_core::error::{Result, SapientError};
use sapient_core::{Shape, Tensor};

// ── SIMD helpers ──────────────────────────────────────────────────────────────

/// Dot product of two equal-length f32 slices, NEON-vectorised on aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, vb);
        i += 4;
    }
    let mut s = vaddvq_f32(acc);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// o[i] = alpha * o[i] + beta * v[i], NEON-vectorised on aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn saxpby_neon(o: &mut [f32], v: &[f32], alpha: f32, beta: f32) {
    use std::arch::aarch64::*;
    let va = vdupq_n_f32(alpha);
    let vb = vdupq_n_f32(beta);
    let n = o.len();
    let mut i = 0;
    while i + 4 <= n {
        let vo = vld1q_f32(o.as_ptr().add(i));
        let vv = vld1q_f32(v.as_ptr().add(i));
        let r = vfmaq_f32(vmulq_f32(vo, va), vv, vb);
        vst1q_f32(o.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < n {
        o[i] = alpha * o[i] + beta * v[i];
        i += 1;
    }
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
fn saxpby_neon(o: &mut [f32], v: &[f32], alpha: f32, beta: f32) {
    for (oi, vi) in o.iter_mut().zip(v) {
        *oi = alpha * *oi + beta * vi;
    }
}

// ── Flash-Edge online-softmax attention kernel ────────────────────────────────

/// Single-head flash attention for one query row `qi`.
///
/// Performs the online-softmax loop over keys 0..attend_len and accumulates
/// into `o_row` (length = head_dim).
///
/// `k_head` and `v_head` are **contiguous** slices of shape
/// `[seq_k, head_dim]` for the relevant KV head (already GQA-expanded by the
/// caller selecting the right `kv_h` slice).
///
/// `causal` controls whether keys beyond `qi + offset` are masked.
/// `offset = seq_k - seq_q` is the KV-cache prefix length.
#[inline(always)]
fn flash_attn_row(
    q_row: &[f32],     // [head_dim]
    k_head: &[f32],    // [seq_k * head_dim] contiguous
    v_head: &[f32],    // [seq_k * head_dim] contiguous
    o_row: &mut [f32], // [head_dim] — written in place
    scale: f32,
    _seq_k: usize,
    head_dim: usize,
    attend_len: usize, // how many k/v positions to visit (causal: qi+offset+1)
    mask_row: Option<&[f32]>, // optional additive mask for this query row, length seq_k
) {
    let mut m = f32::NEG_INFINITY; // running max
    let mut l = 0.0f32; // running sum of exp weights

    // Zero the output accumulator.
    for x in o_row.iter_mut() {
        *x = 0.0;
    }

    for ki in 0..attend_len {
        let k_row = &k_head[ki * head_dim..(ki + 1) * head_dim];

        // Score: scale * q·k + optional mask
        #[cfg(target_arch = "aarch64")]
        let raw_s = unsafe { dot_f32_neon(q_row, k_row) } * scale;
        #[cfg(not(target_arch = "aarch64"))]
        let raw_s = dot_f32_neon(q_row, k_row) * scale;

        let s = raw_s + mask_row.map(|m| m[ki]).unwrap_or(0.0);

        // Online softmax update.
        let m_new = if s > m { s } else { m };
        let p = (s - m_new).exp();
        let correction = (m - m_new).exp();

        // O = correction * O + p * v[ki]
        let v_row = &v_head[ki * head_dim..(ki + 1) * head_dim];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            saxpby_neon(o_row, v_row, correction, p);
        }
        #[cfg(not(target_arch = "aarch64"))]
        saxpby_neon(o_row, v_row, correction, p);

        l = correction * l + p;
        m = m_new;
    }

    // Normalize.
    let inv_l = if l == 0.0 {
        1.0 / f32::EPSILON
    } else {
        1.0 / l
    };
    for x in o_row.iter_mut() {
        *x *= inv_l;
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Standard multi-head / grouped-query attention.
///
/// Inputs:
///   q: (batch, n_heads, seq_q, head_dim)
///   k: (batch, n_kv_heads, seq_k, head_dim)
///   v: (batch, n_kv_heads, seq_k, head_dim)
///   mask: optional (seq_q, seq_k) additive mask (−inf for masked positions)
///
/// Output: (batch, n_heads, seq_q, head_dim)
///
/// Implementation: Flash-Edge online-softmax — never materialises the full
/// seq_q × seq_k score matrix; O(seq_q × head_dim) working memory per head.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
    n_kv_heads: usize,
) -> Result<Tensor> {
    let qs = q.shape().dims().to_vec();
    let ks = k.shape().dims().to_vec();

    if qs.len() != 4 {
        return Err(SapientError::RankMismatch {
            expected: 4,
            got: qs.len(),
        });
    }

    let (batch, n_heads, seq_q, head_dim) = (qs[0], qs[1], qs[2], qs[3]);
    let seq_k = ks[2];
    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // KV head repetition for GQA.
    let kv_rep = n_heads / n_kv_heads; // 1 for MHA, >1 for GQA

    // --- Pre-convert K and V to contiguous f32 once ---
    // KV tensors may be non-contiguous views from slice_axis on the KV cache.
    // to_contiguous_f32_vec handles stride-based extraction safely.
    // Layout after conversion: [batch, n_kv_heads, seq_k, head_dim] row-major.
    let k_data: Vec<f32> = k.to_contiguous_f32_vec();
    let v_data: Vec<f32> = v.to_contiguous_f32_vec();

    // Q is typically contiguous; use to_f32_cow for a zero-copy borrow when possible.
    let q_cow = q.to_f32_cow();
    let q_data = q_cow.as_ref();
    let q_strides = q.strides();

    // Pre-fetch optional mask data.
    let mask_cow = mask.map(|m| m.to_f32_cow());
    let mask_data: Option<&[f32]> = mask_cow.as_deref();

    // KV cache prefix offset for causal masking.
    // When seq_k > seq_q the first (seq_k - seq_q) positions are cached tokens.
    let kv_offset = seq_k.saturating_sub(seq_q);

    // Each (b, h) pair writes seq_q * head_dim elements.
    let head_out_size = seq_q * head_dim;
    let kv_head_size = seq_k * head_dim; // elements per (b, kv_h) in k_data / v_data

    let mut out = vec![0.0f32; batch * n_heads * head_out_size];

    out.par_chunks_mut(head_out_size)
        .enumerate()
        .for_each(|(bh, out_chunk)| {
            let b = bh / n_heads;
            let h = bh % n_heads;
            let kv_h = h / kv_rep;

            // Contiguous slice for this (b, kv_h) in K and V.
            let k_base = (b * n_kv_heads + kv_h) * kv_head_size;
            let v_base = (b * n_kv_heads + kv_h) * kv_head_size;
            let k_head = &k_data[k_base..k_base + kv_head_size];
            let v_head = &v_data[v_base..v_base + kv_head_size];

            for qi in 0..seq_q {
                // Build a contiguous q_row slice.
                // Q strides: [batch_s, head_s, seq_s, dim_s].
                // If q is contiguous the slice is zero-copy; otherwise we copy.
                let q_base_elem = b * q_strides[0] + h * q_strides[1] + qi * q_strides[2];

                // Fast path: q strides[3] == 1 (contiguous along head_dim).
                let q_row_owned: Vec<f32>;
                let q_row: &[f32] = if q_strides[3] == 1 {
                    &q_data[q_base_elem..q_base_elem + head_dim]
                } else {
                    q_row_owned = (0..head_dim)
                        .map(|d| q_data[q_base_elem + d * q_strides[3]])
                        .collect();
                    &q_row_owned
                };

                // For causal masking: query at position qi can attend to
                // k[0..=qi+kv_offset].  If an explicit mask tensor is provided,
                // let it govern (it may already encode causality via -inf).
                let attend_len = if mask_data.is_some() {
                    seq_k
                } else {
                    // Built-in causal masking: attend only to past/current tokens.
                    (qi + kv_offset + 1).min(seq_k)
                };

                // Mask row for this query position (shape [seq_k] within mask).
                let mask_row = mask_data.map(|m| &m[qi * seq_k..(qi + 1) * seq_k]);

                let o_row = &mut out_chunk[qi * head_dim..(qi + 1) * head_dim];

                flash_attn_row(
                    q_row, k_head, v_head, o_row, scale, seq_k, head_dim, attend_len, mask_row,
                );
            }
        });

    Tensor::from_f32(&out, Shape::new([batch, n_heads, seq_q, head_dim]))
}

// ── Causal mask ───────────────────────────────────────────────────────────────

/// Build a causal additive mask of shape (seq_q, seq_k):
///   0 for allowed positions, -inf for masked (future) positions.
///
/// For decoding (seq_q=1), this is all zeros (every cached KV is in the past).
pub fn causal_mask(seq_q: usize, seq_k: usize) -> Tensor {
    let mut data = vec![0.0f32; seq_q * seq_k];
    // In a decoder, token at position i can attend to j ≤ i.
    // When seq_k > seq_q we have a prefix (KV cache), so offset accordingly.
    let offset = seq_k.saturating_sub(seq_q);
    for qi in 0..seq_q {
        for ki in 0..seq_k {
            if ki > qi + offset {
                data[qi * seq_k + ki] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_f32(&data, vec![seq_q, seq_k]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mha_output_shape() {
        // batch=1, heads=2, seq=3, dim=4
        let q = Tensor::from_f32(&[0.1f32; 24], vec![1, 2, 3, 4]).unwrap();
        let k = Tensor::from_f32(&[0.1f32; 24], vec![1, 2, 3, 4]).unwrap();
        let v = Tensor::from_f32(&[0.1f32; 24], vec![1, 2, 3, 4]).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v, None, None, 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn gqa_kv_repeat() {
        // batch=1, n_heads=4, n_kv_heads=2, seq=2, dim=4
        let q = Tensor::from_f32(&[0.1f32; 32], vec![1, 4, 2, 4]).unwrap();
        let k = Tensor::from_f32(&[0.1f32; 16], vec![1, 2, 2, 4]).unwrap();
        let v = Tensor::from_f32(&[0.1f32; 16], vec![1, 2, 2, 4]).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v, None, None, 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 4, 2, 4]);
    }

    #[test]
    fn causal_mask_shape() {
        let m = causal_mask(3, 3);
        let d = m.as_f32_slice();
        // Position (0,1) should be -inf (index 1)
        assert!(d[1].is_infinite() && d[1] < 0.0);
        // Position (1,0) should be 0 (index 3)
        assert_eq!(d[3], 0.0);
    }

    /// Smoke-test: attention with all-equal Q/K/V should produce values equal to V rows.
    #[test]
    fn uniform_attention_recovers_v() {
        // When all scores are equal, softmax gives uniform weights and the output
        // is just the average of the V rows.  With identical V rows, output == V row.
        let seq = 4usize;
        let dim = 8usize;
        // V rows: row i has all elements = (i+1) as f32
        let mut v_data = vec![0.0f32; seq * dim];
        for i in 0..seq {
            for d in 0..dim {
                v_data[i * dim + d] = (i + 1) as f32;
            }
        }
        // Q and K are identical (all ones) → uniform scores.
        let q = Tensor::from_f32(&vec![1.0f32; seq * dim], vec![1, 1, seq, dim]).unwrap();
        let k = Tensor::from_f32(&vec![1.0f32; seq * dim], vec![1, 1, seq, dim]).unwrap();
        let v = Tensor::from_f32(&v_data, vec![1, 1, seq, dim]).unwrap();

        // Use an explicit all-zero mask so we attend to ALL keys (no causal masking).
        let mask = Tensor::from_f32(&vec![0.0f32; seq * seq], vec![seq, seq]).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None, 1).unwrap();
        let out_data = out.as_f32_slice();

        // Expected: each output row is the mean of V rows up to that point is not
        // guaranteed here since we use full mask; output for row qi should be average
        // of rows 0..seq which equals (1+2+3+4)/4 = 2.5.
        let expected = (1..=seq).map(|x| x as f32).sum::<f32>() / seq as f32;
        for &val in out_data.iter() {
            let diff = (val - expected).abs();
            assert!(diff < 1e-4, "Expected ~{expected}, got {val}");
        }
    }

    /// Verify that the online-softmax result matches a reference naïve implementation.
    #[test]
    fn flash_matches_naive() {
        use std::f32;
        let batch = 1;
        let n_heads = 2;
        let seq_q = 4;
        let seq_k = 4;
        let head_dim = 8;

        // Random-ish but deterministic data.
        let gen = |i: usize| (i as f32 * 1.3 + 0.7).sin() * 0.5 + 0.5;
        let q_data: Vec<f32> = (0..batch * n_heads * seq_q * head_dim).map(gen).collect();
        let k_data: Vec<f32> = (0..batch * n_heads * seq_k * head_dim)
            .map(|i| gen(i + 100))
            .collect();
        let v_data: Vec<f32> = (0..batch * n_heads * seq_k * head_dim)
            .map(|i| gen(i + 200))
            .collect();

        let q = Tensor::from_f32(&q_data, vec![batch, n_heads, seq_q, head_dim]).unwrap();
        let k = Tensor::from_f32(&k_data, vec![batch, n_heads, seq_k, head_dim]).unwrap();
        let v = Tensor::from_f32(&v_data, vec![batch, n_heads, seq_k, head_dim]).unwrap();

        // Causal mask.
        let mask_t = causal_mask(seq_q, seq_k);
        let flash_out =
            scaled_dot_product_attention(&q, &k, &v, Some(&mask_t), None, n_heads).unwrap();

        // --- Naïve reference ---
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask_data = mask_t.as_f32_slice();
        let mut ref_out = vec![0.0f32; batch * n_heads * seq_q * head_dim];

        for b in 0..batch {
            for h in 0..n_heads {
                let q_off = (b * n_heads + h) * seq_q * head_dim;
                let k_off = (b * n_heads + h) * seq_k * head_dim;
                let v_off = (b * n_heads + h) * seq_k * head_dim;
                let o_off = (b * n_heads + h) * seq_q * head_dim;

                for qi in 0..seq_q {
                    let mut scores = vec![0.0f32; seq_k];
                    for ki in 0..seq_k {
                        let dot: f32 = (0..head_dim)
                            .map(|d| {
                                q_data[q_off + qi * head_dim + d]
                                    * k_data[k_off + ki * head_dim + d]
                            })
                            .sum();
                        scores[ki] = dot * scale + mask_data[qi * seq_k + ki];
                    }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let max_s = if max_s.is_infinite() { 0.0 } else { max_s };
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_s).exp();
                        sum += *s;
                    }
                    if sum < f32::EPSILON {
                        sum = f32::EPSILON;
                    }
                    for d in 0..head_dim {
                        let acc: f32 = (0..seq_k)
                            .map(|ki| scores[ki] / sum * v_data[v_off + ki * head_dim + d])
                            .sum();
                        ref_out[o_off + qi * head_dim + d] = acc;
                    }
                }
            }
        }

        let flash_data = flash_out.as_f32_slice();
        for (i, (&flash, &reference)) in flash_data.iter().zip(ref_out.iter()).enumerate() {
            let diff = (flash - reference).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at index {i}: flash={flash} ref={reference} diff={diff}"
            );
        }
    }

    /// Decode-mode (seq_q=1): verify output shape and no NaN/Inf.
    #[test]
    fn decode_mode_no_nan() {
        let batch = 1;
        let n_heads = 4;
        let seq_q = 1;
        let seq_k = 16; // KV cache has 16 tokens
        let head_dim = 8;

        let q = Tensor::from_f32(
            &vec![0.1f32; batch * n_heads * seq_q * head_dim],
            vec![batch, n_heads, seq_q, head_dim],
        )
        .unwrap();
        let k = Tensor::from_f32(
            &vec![0.1f32; batch * n_heads * seq_k * head_dim],
            vec![batch, n_heads, seq_k, head_dim],
        )
        .unwrap();
        let v = Tensor::from_f32(
            &vec![0.2f32; batch * n_heads * seq_k * head_dim],
            vec![batch, n_heads, seq_k, head_dim],
        )
        .unwrap();

        let out = scaled_dot_product_attention(&q, &k, &v, None, None, n_heads).unwrap();
        assert_eq!(out.shape().dims(), &[batch, n_heads, seq_q, head_dim]);
        for &val in out.as_f32_slice() {
            assert!(val.is_finite(), "NaN/Inf in decode output: {val}");
        }
    }
}
