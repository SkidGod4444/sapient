//! Rotary Position Embedding (RoPE) kernel.
//!
//! RoPE rotates query and key vectors by a position-dependent angle:
//!   x' = x * cos(θ) + rotate_half(x) * sin(θ)
//!
//! Used by: Llama 1/2/3, Mistral, Falcon, Phi, Gemma, Qwen.

use sapient_core::error::{Result, SapientError};
use sapient_core::{Shape, Tensor};

/// Apply RoPE to a tensor of shape (batch, n_heads, seq_len, head_dim).
///
/// `positions` is a 1-D tensor of shape (seq_len,) containing token positions
/// (e.g., [0, 1, 2, ...] for prefill, [n] for single-token decoding).
pub fn apply_rope(x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(SapientError::RankMismatch {
            expected: 4,
            got: dims.len(),
        });
    }
    let (batch, n_heads, seq_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);

    if head_dim % 2 != 0 {
        return Err(SapientError::internal("RoPE requires even head_dim"));
    }
    if positions.len() != seq_len {
        return Err(SapientError::internal(
            "positions length must match seq_len",
        ));
    }

    let half = head_dim / 2;
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
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
                    out[base_idx + i] = x0 * cos_f - x1 * sin_f;
                    out[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
                }
            }
        }
    }

    Tensor::from_f32(&out, Shape::new([batch, n_heads, seq_len, head_dim]))
}

/// Apply RoPE to only the first `rotary_dim` channels of each head, leaving the
/// remaining `head_dim - rotary_dim` channels unchanged. Used by Phi-family
/// models where `rotary_dim = partial_rotary_factor * head_dim` (e.g. 0.4·80=32).
///
/// When `rotary_dim == head_dim` this is identical to [`apply_rope`].
pub fn apply_rope_partial(
    x: &Tensor,
    positions: &[usize],
    base: f32,
    rotary_dim: usize,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(SapientError::RankMismatch {
            expected: 4,
            got: dims.len(),
        });
    }
    let (batch, n_heads, seq_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);

    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(SapientError::internal(
            "rotary_dim must be in 1..=head_dim",
        ));
    }
    if rotary_dim % 2 != 0 {
        return Err(SapientError::internal("RoPE requires even rotary_dim"));
    }
    if positions.len() != seq_len {
        return Err(SapientError::internal(
            "positions length must match seq_len",
        ));
    }

    // The rotary half-split is taken over `rotary_dim`, not head_dim. Channels
    // [rotary_dim..head_dim] are passed through unchanged.
    let half = rotary_dim / 2;
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let mut out = x_data.to_vec();

    for b in 0..batch {
        for h in 0..n_heads {
            for (s, &pos) in positions.iter().enumerate() {
                let base_idx = ((b * n_heads + h) * seq_len + s) * head_dim;
                for i in 0..half {
                    let freq = (pos as f32) / base.powf(2.0 * i as f32 / rotary_dim as f32);
                    let (sin_f, cos_f) = freq.sin_cos();
                    let x0 = x_data[base_idx + i];
                    let x1 = x_data[base_idx + i + half];
                    out[base_idx + i] = x0 * cos_f - x1 * sin_f;
                    out[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
                }
            }
        }
    }

    Tensor::from_f32(&out, Shape::new([batch, n_heads, seq_len, head_dim]))
}

/// Pre-compute (cos, sin) tables for positions [0..max_seq_len].
/// Returns two tensors of shape (max_seq_len, head_dim/2).
pub fn rope_cos_sin_cache(max_seq_len: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
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
        let x = Tensor::from_f32(&[0.1f32; 64], vec![1, 2, 4, 8]).unwrap();
        let positions: Vec<usize> = (0..4).collect();
        let out = apply_rope(&x, &positions, 10000.0).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn rope_partial_leaves_tail_unchanged() {
        // head_dim=8, rotary_dim=4 → channels [4..8) must pass through unchanged,
        // and at a non-zero position the rotary channels [0..4) must change.
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x = Tensor::from_f32(&data, vec![1, 1, 1, 8]).unwrap();
        let out = apply_rope_partial(&x, &[3], 10000.0, 4).unwrap();
        let o = out.as_f32_slice();
        // Non-rotary tail unchanged.
        for i in 4..8 {
            assert!((o[i] - data[i]).abs() < 1e-6, "tail channel {i} changed");
        }
        // At least one rotary channel changed.
        assert!((0..4).any(|i| (o[i] - data[i]).abs() > 1e-6));
    }

    #[test]
    fn rope_partial_full_matches_apply_rope() {
        // shape [1, 1, 2, 8]: batch=1, heads=1, seq=2, head_dim=8.
        let data: Vec<f32> = (0..16).map(|v| v as f32 * 0.1).collect();
        let x = Tensor::from_f32(&data, vec![1, 1, 2, 8]).unwrap();
        let full = apply_rope(&x, &[2, 5], 10000.0).unwrap();
        let part = apply_rope_partial(&x, &[2, 5], 10000.0, 8).unwrap();
        for (a, b) in full.as_f32_slice().iter().zip(part.as_f32_slice()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        // At position 0, cos(0)=1, sin(0)=0, so the vector is unchanged.
        let data = vec![1.0f32, 2.0, 3.0, 4.0]; // head_dim=4
        let x = Tensor::from_f32(&data, vec![1, 1, 1, 4]).unwrap();
        let out = apply_rope(&x, &[0], 10000.0).unwrap();
        let out_data = out.as_f32_slice();
        for (a, b) in data.iter().zip(out_data.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 should be identity");
        }
    }
}
