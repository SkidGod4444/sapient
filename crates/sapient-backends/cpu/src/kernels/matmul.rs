//! Matrix multiplication kernels using the `matrixmultiply` crate.
//!
//! `matrixmultiply` provides a pure-Rust, BLAS-free, AVX2-optimised SGEMM.
//! It beats a naive loop by ~10-30× on modern CPUs.

use sapient_core::error::{Result, SapientError};
use sapient_core::{Shape, Tensor};

// ── matmul ───────────────────────────────────────────────────────────────────

/// 2-D matrix multiply: (M, K) × (K, N) → (M, N).
/// Also handles batched: (..., M, K) × (..., K, N) → (..., M, N).
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.ndim() < 2 || b_shape.ndim() < 2 {
        return Err(SapientError::RankMismatch {
            expected: 2,
            got: a_shape.ndim().min(b_shape.ndim()),
        });
    }

    // Extract M, K, N from last two dims.
    let a_rank = a_shape.ndim();
    let b_rank = b_shape.ndim();
    let m = a_shape.dims()[a_rank - 2];
    let k = a_shape.dims()[a_rank - 1];
    let k2 = b_shape.dims()[b_rank - 2];
    let n = b_shape.dims()[b_rank - 1];

    if k != k2 {
        return Err(SapientError::ShapeMismatch {
            expected: vec![m, k, n],
            got: vec![m, k2, n],
        });
    }

    // Compute batch size from leading dims.
    let batch: usize = a_shape.dims()[..a_rank - 2].iter().product();

    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();

    let out_numel = batch * m * n;
    let mut out_data = vec![0.0f32; out_numel];

    // Stride for each batch element.
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for bi in 0..batch {
        let a_off = bi * a_stride;
        let b_off = bi * b_stride;
        let c_off = bi * c_stride;

        // SAFETY: raw pointers derived from Vec<f32> — valid, non-overlapping.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a_data[a_off..].as_ptr(),
                k as isize,
                1,
                b_data[b_off..].as_ptr(),
                n as isize,
                1,
                0.0,
                out_data[c_off..].as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }

    // Build output shape.
    let mut out_dims: Vec<usize> = if a_rank > 2 {
        a_shape.dims()[..a_rank - 2].to_vec()
    } else {
        vec![]
    };
    out_dims.push(m);
    out_dims.push(n);

    Tensor::from_f32(&out_data, Shape::new(out_dims))
}

// ── gemm ─────────────────────────────────────────────────────────────────────

/// General Matrix Multiply with optional transpose and scaling:
///   C = alpha * op(A) × op(B) + beta * bias
pub fn gemm(
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<Tensor> {
    let a2 = if trans_a { a.t()? } else { a.clone() };
    let b2 = if trans_b { b.t()? } else { b.clone() };

    let a_shape = a2.shape();
    let b_shape = b2.shape();

    let m = a_shape.dims()[0];
    let k = a_shape.dims()[1];
    let k2 = b_shape.dims()[0];
    let n = b_shape.dims()[1];

    if k != k2 {
        return Err(SapientError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![k2, n],
        });
    }

    let a_data = a2.as_f32_slice();
    let b_data = b2.as_f32_slice();
    let mut out = vec![0.0f32; m * n];

    let a_strides = a2.strides();
    let b_strides = b2.strides();

    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a_data.as_ptr(),
            a_strides[0] as isize,
            a_strides[1] as isize,
            b_data.as_ptr(),
            b_strides[0] as isize,
            b_strides[1] as isize,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    // Add bias if present.
    if let Some(bias_t) = bias {
        let bias_data = bias_t.as_f32_slice();
        // Bias shape: [n] or [1, n] — broadcast over rows.
        let b_len = bias_data.len();
        if b_len != n && b_len != 1 {
            return Err(SapientError::ShapeMismatch {
                expected: vec![n],
                got: vec![b_len],
            });
        }
        for i in 0..m {
            for j in 0..n {
                let bv = if b_len == 1 {
                    bias_data[0]
                } else {
                    bias_data[j]
                };
                out[i * n + j] += beta * bv;
            }
        }
    }

    Tensor::from_f32(&out, Shape::new([m, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_core::DType;

    #[test]
    fn matmul_2x2() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        let data = c.as_f32_slice();
        assert!((data[0] - 19.0).abs() < 1e-5);
        assert!((data[1] - 22.0).abs() < 1e-5);
        assert!((data[2] - 43.0).abs() < 1e-5);
        assert!((data[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_rank_mismatch() {
        let a = Tensor::zeros(vec![4], DType::F32).unwrap();
        let b = Tensor::zeros(vec![4], DType::F32).unwrap();
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn gemm_with_bias() {
        let a = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let bias = Tensor::from_f32(&[1.0, 1.0], vec![2]).unwrap();
        let c = gemm(&a, &b, Some(&bias), 1.0, 1.0, false, false).unwrap();
        let d = c.as_f32_slice();
        // Identity × [[2,3],[4,5]] = [[2,3],[4,5]]; + bias [1,1] = [[3,4],[5,6]]
        assert!((d[0] - 3.0).abs() < 1e-5, "got {}", d[0]);
        assert!((d[1] - 4.0).abs() < 1e-5, "got {}", d[1]);
    }
}
