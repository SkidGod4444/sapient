//! Rotary Position Embedding (RoPE) kernel.
//!
//! RoPE rotates query and key vectors by a position-dependent angle:
//!   x' = x * cos(θ) + rotate_half(x) * sin(θ)
//!
//! Used by: Llama 1/2/3, Mistral, Falcon, Phi, Gemma, Qwen.

use sapient_core::{Tensor, Shape};
use sapient_core::error::{Result, SapientError};

/// Apply RoPE to a tensor of shape (batch, n_heads, seq_len, head_dim).
///
/// `positions` is a 1-D tensor of shape (seq_len,) containing token positions
/// (e.g., [0, 1, 2, ...] for prefill, [n] for single-token decoding).
pub fn apply_rope(
    x: &Tensor,
    positions: &[usize],
    base: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(SapientError::RankMismatch { expected: 4, got: dims.len() });
    }
    let (batch, n_heads, seq_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);

    if head_dim % 2 != 0 {
        return Err(SapientError::internal("RoPE requires even head_dim"));
    }
    if positions.len() != seq_len {
        return Err(SapientError::internal("positions length must match seq_len"));
    }

    let half = head_dim / 2;
    let x_data = x.as_f32_slice();
    let mut out = x_data.to_vec();

    for b in 0..batch {
        for h in 0..n_heads {
            for (s, &pos) in positions.iter().enumerate() {
                let base_idx = ((b * n_heads + h) * seq_len + s) * head_dim;

                for i in 0..half {
                    // θ_i = pos / base^(2i / head_dim)
                    let freq = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
                    let (sin_f, cos_f) = freq.sin_cos();

                    let x0 = x_data[base_idx + i];
                    let x1 = x_data[base_idx + i + half];

                    // Rotate: [x0, x1] → [x0 cos - x1 sin, x1 cos + x0 sin]
                    out[base_idx + i]        = x0 * cos_f - x1 * sin_f;
                    out[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
                }
            }
        }
    }

    Tensor::from_f32(&out, Shape::new([batch, n_heads, seq_len, head_dim]))
}

/// Pre-compute (cos, sin) tables for positions [0..max_seq_len].
/// Returns two tensors of shape (max_seq_len, head_dim/2).
pub fn rope_cos_sin_cache(
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_table = vec![0.0f32; max_seq_len * half];
    let mut sin_table = vec![0.0f32; max_seq_len * half];

    for pos in 0..max_seq_len {
        for i in 0..half {
            let freq = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
            cos_table[pos * half + i] = freq.cos();
            sin_table[pos * half + i] = freq.sin();
        }
    }
    (cos_table, sin_table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_output_shape() {
        let x = Tensor::from_f32(&vec![0.1f32; 1 * 2 * 4 * 8], vec![1, 2, 4, 8]).unwrap();
        let positions: Vec<usize> = (0..4).collect();
        let out = apply_rope(&x, &positions, 10000.0).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        // At position 0, cos(0)=1, sin(0)=0, so the vector is unchanged.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];  // head_dim=4
        let x = Tensor::from_f32(&data, vec![1, 1, 1, 4]).unwrap();
        let out = apply_rope(&x, &[0], 10000.0).unwrap();
        let out_data = out.as_f32_slice();
        for (a, b) in data.iter().zip(out_data.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 should be identity");
        }
    }
}
