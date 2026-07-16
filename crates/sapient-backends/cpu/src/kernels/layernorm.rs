// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! LayerNorm and RMSNorm kernels.
//!
//! Both are numerically stabilised and operate over the last `ndim - axis` axes.

use sapient_core::error::Result;
use sapient_core::Tensor;

// ── LayerNorm ─────────────────────────────────────────────────────────────────

/// Standard layer normalisation:
///   y = (x - mean) / sqrt(var + eps) * weight + bias
///
/// `axis` is the first axis to normalise over (typically -1 for the hidden dim).
pub fn layer_norm(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    axis: i64,
    epsilon: f32,
) -> Result<Tensor> {
    let shape = x.shape();
    let ndim = shape.ndim();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    let outer: usize = shape.dims()[..ax].iter().product();
    let norm_size: usize = shape.dims()[ax..].iter().product();

    let data_cow = x.to_f32_cow();
    let data = data_cow.as_ref();
    let mut out = vec![0.0f32; data.len()];

    let w_cow = weight.map(|t| t.to_f32_cow());
    let w = w_cow.as_ref().map(|c| c.as_ref());
    let b_cow = bias.map(|t| t.to_f32_cow());
    let b = b_cow.as_ref().map(|c| c.as_ref());

    for o in 0..outer {
        let base = o * norm_size;
        let slice = &data[base..base + norm_size];

        // Compute mean.
        let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;

        // Compute variance.
        let var: f32 =
            slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / norm_size as f32;

        let inv_std = 1.0 / (var + epsilon).sqrt();

        for i in 0..norm_size {
            let normed = (slice[i] - mean) * inv_std;
            out[base + i] = match (w, b) {
                (Some(ww), Some(bb)) => normed * ww[i] + bb[i],
                (Some(ww), None) => normed * ww[i],
                (None, Some(bb)) => normed + bb[i],
                (None, None) => normed,
            };
        }
    }

    Tensor::from_f32(&out, shape.clone())
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

/// Root-Mean-Square layer norm (used by LLaMA, Mistral, etc.):
///   y = x / sqrt(mean(x²) + eps) * weight
pub fn rms_norm(x: &Tensor, weight: Option<&Tensor>, epsilon: f32) -> Result<Tensor> {
    let shape = x.shape();
    let ndim = shape.ndim();

    let outer: usize = shape.dims()[..ndim.saturating_sub(1)].iter().product();
    let dim = if ndim > 0 {
        *shape.dims().last().unwrap()
    } else {
        1
    };

    let data_cow = x.to_f32_cow();
    let data = data_cow.as_ref();
    let mut out = vec![0.0f32; data.len()];

    let w_cow = weight.map(|t| t.to_f32_cow());
    let w = w_cow.as_ref().map(|c| c.as_ref());

    for o in 0..outer {
        let base = o * dim;
        let slice = &data[base..base + dim];

        let rms_sq: f32 = slice.iter().map(|&v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (rms_sq + epsilon).sqrt();

        for i in 0..dim {
            out[base + i] = slice[i] * inv_rms * w.map_or(1.0, |ww| ww[i]);
        }
    }

    Tensor::from_f32(&out, shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_core::Tensor;

    #[test]
    fn layernorm_zero_mean_unit_var() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = layer_norm(&x, None, None, -1, 1e-5).unwrap();
        let d = y.as_f32_slice();
        // Each pair: mean=1.5, var=0.25, std=0.5. (1-1.5)/0.5 = -1, (2-1.5)/0.5 = 1.
        assert!((d[0] + 1.0).abs() < 1e-4, "d[0]={}", d[0]);
        assert!((d[1] - 1.0).abs() < 1e-4, "d[1]={}", d[1]);
        assert!((d[2] + 1.0).abs() < 1e-4, "d[2]={}", d[2]);
        assert!((d[3] - 1.0).abs() < 1e-4, "d[3]={}", d[3]);
    }

    #[test]
    fn rmsnorm_identity_weight() {
        // weight = all ones → scale by inv_rms
        // rms = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.5355
        // inv_rms ≈ 0.28284
        // output ≈ [3 * 0.28284, 4 * 0.28284] = [0.84853, 1.13137]
        let x = Tensor::from_f32(&[3.0, 4.0], vec![1, 2]).unwrap();
        let w = Tensor::from_f32(&[1.0, 1.0], vec![2]).unwrap();
        let y = rms_norm(&x, Some(&w), 0.0).unwrap();
        let d = y.as_f32_slice();
        let expected0 = 3.0 / (12.5f32).sqrt();
        let expected1 = 4.0 / (12.5f32).sqrt();
        assert!(
            (d[0] - expected0).abs() < 1e-5,
            "d[0]={} expected {}",
            d[0],
            expected0
        );
        assert!(
            (d[1] - expected1).abs() < 1e-5,
            "d[1]={} expected {}",
            d[1],
            expected1
        );
    }
}
