//! Scaled dot-product attention and Grouped-Query Attention kernels.
//!
//! Implements the core attention mechanism used by all transformer LLMs:
//!   Attention(Q, K, V) = softmax(QK^T / √d_k + mask) × V
//!
//! Also implements:
//!   - Causal masking (upper-triangular -inf)
//!   - Grouped-Query Attention (Llama2/3, Mistral) — KV head repeat
//!   - Rotary Position Embedding (RoPE) inline application

use sapient_core::error::{Result, SapientError};
use sapient_core::{Shape, Tensor};

// ── Scaled dot-product attention ──────────────────────────────────────────────

/// Standard multi-head attention.
///
/// Inputs:
///   q: (batch, n_heads, seq_q, head_dim)
///   k: (batch, n_kv_heads, seq_k, head_dim)
///   v: (batch, n_kv_heads, seq_k, head_dim)
///   mask: optional (seq_q, seq_k) additive mask (−inf for masked positions)
///
/// Output: (batch, n_heads, seq_q, head_dim)
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
    n_kv_heads: usize, // for GQA: repeat KV if n_kv_heads < n_heads
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

    let q_data = q.as_f32_slice();
    let k_data = k.as_f32_slice();
    let v_data = v.as_f32_slice();

    let q_strides = q.strides();
    let k_strides = k.strides();
    let v_strides = v.strides();

    // KV head repetition for GQA.
    let kv_rep = n_heads / n_kv_heads; // 1 for MHA, >1 for GQA

    let mut out = vec![0.0f32; batch * n_heads * seq_q * head_dim];

    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / kv_rep; // which KV head this Q head maps to

            // QK^T: (seq_q, head_dim) × (head_dim, seq_k) → (seq_q, seq_k)
            let mut scores = vec![0.0f32; seq_q * seq_k];

            for qi in 0..seq_q {
                for ki in 0..seq_k {
                    let mut dot = 0.0f32;
                    let q_base = b * q_strides[0] + h * q_strides[1] + qi * q_strides[2];
                    let k_base = b * k_strides[0] + kv_h * k_strides[1] + ki * k_strides[2];
                    
                    for d in 0..head_dim {
                        let q_idx = q_base + d * q_strides[3];
                        let k_idx = k_base + d * k_strides[3];
                        dot += q_data[q_idx] * k_data[k_idx];
                    }
                    scores[qi * seq_k + ki] = dot * scale;
                }
            }

            // Apply optional additive mask.
            if let Some(m) = mask {
                let m_data = m.as_f32_slice();
                for qi in 0..seq_q {
                    for ki in 0..seq_k {
                        scores[qi * seq_k + ki] += m_data[qi * seq_k + ki];
                    }
                }
            }

            // Softmax over seq_k for each query position.
            for qi in 0..seq_q {
                let row = &mut scores[qi * seq_k..(qi + 1) * seq_k];
                let mut max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if max_v == f32::NEG_INFINITY {
                    max_v = 0.0;
                }
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_v).exp();
                    sum += *v;
                }
                if sum == 0.0 {
                    sum = f32::EPSILON;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }

            // scores × V: (seq_q, seq_k) × (seq_k, head_dim) → (seq_q, head_dim)
            for qi in 0..seq_q {
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ki in 0..seq_k {
                        let s_idx = qi * seq_k + ki;
                        let v_idx = b * v_strides[0] + kv_h * v_strides[1] + ki * v_strides[2] + d * v_strides[3];
                        acc += scores[s_idx] * v_data[v_idx];
                    }
                    out[((b * n_heads + h) * seq_q + qi) * head_dim + d] = acc;
                }
            }
        }
    }

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
}
