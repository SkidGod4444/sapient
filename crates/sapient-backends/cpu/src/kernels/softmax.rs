//! Numerically stable softmax kernel.
//!
//! Uses the log-sum-exp trick: subtract max before exp to prevent overflow.

use sapient_core::error::{Result, SapientError};
use sapient_core::Tensor;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Normalise an axis index (negative = count from end).
fn normalise_axis(axis: i64, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────

/// Numerically stable softmax along `axis`.
pub fn softmax(x: &Tensor, axis: i64) -> Result<Tensor> {
    apply_softmax_impl(x, axis, false)
}

/// Numerically stable log-softmax along `axis`.
pub fn log_softmax(x: &Tensor, axis: i64) -> Result<Tensor> {
    apply_softmax_impl(x, axis, true)
}

fn apply_softmax_impl(x: &Tensor, axis: i64, log_mode: bool) -> Result<Tensor> {
    let shape = x.shape();
    let ndim = shape.ndim();
    let ax = normalise_axis(axis, ndim);

    if ax >= ndim {
        return Err(SapientError::internal(format!(
            "softmax axis {axis} out of range for rank {ndim}"
        )));
    }

    let data = x.as_f32_slice();
    let mut out = vec![0.0f32; data.len()];

    // We iterate over slices along the `ax` dimension.
    let outer: usize = shape.dims()[..ax].iter().product();
    let dim_size = shape.dims()[ax];
    let inner: usize = shape.dims()[ax + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            // Gather the slice.
            let slice: Vec<f32> = (0..dim_size)
                .map(|d| data[(o * dim_size + d) * inner + i])
                .collect();

            // Subtract max for numerical stability.
            let max_v = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|&v| (v - max_v).exp()).collect();
            let sum_e: f32 = exps.iter().sum();

            for d in 0..dim_size {
                let idx = (o * dim_size + d) * inner + i;
                out[idx] = if log_mode {
                    (slice[d] - max_v) - sum_e.ln()
                } else {
                    exps[d] / sum_e
                };
            }
        }
    }

    Tensor::from_f32(&out, shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_core::Tensor;

    #[test]
    fn softmax_sums_to_one() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let y = softmax(&x, 1).unwrap();
        let sum: f32 = y.as_f32_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn softmax_stable_large() {
        let x = Tensor::from_f32(&[1000.0, 1001.0, 1002.0], vec![1, 3]).unwrap();
        let y = softmax(&x, 1).unwrap();
        let d = y.as_f32_slice();
        // Should be ~[0.09, 0.24, 0.67] — no NaN/Inf.
        for &v in d {
            assert!(v.is_finite(), "non-finite: {v}");
        }
        let sum: f32 = d.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn log_softmax_finite() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = log_softmax(&x, 1).unwrap();
        for &v in y.as_f32_slice() {
            assert!(v.is_finite());
        }
    }
}
