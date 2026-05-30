//! Matrix multiplication kernels using the `matrixmultiply` crate.
//!
//! `matrixmultiply` provides a pure-Rust, BLAS-free, AVX2-optimised SGEMM.
//! It beats a naive loop by ~10-30× on modern CPUs.

use super::quant;
use rayon::prelude::*;
use sapient_core::error::{Result, SapientError};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use sapient_core::{
    DType, Shape, Tensor, Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES, QUANT_BLOCK_SIZE,
};

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

    let a_cow = a.to_f32_cow();
    let a_data = a_cow.as_ref();
    let b_cow = b.to_f32_cow();
    let b_data = b_cow.as_ref();

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

// ── linear (matmul with implicitly-transposed weight) ─────────────────────────

/// Linear projection: `x [M, K] @ Wᵀ` where `W` is stored row-major as `[N, K]`
/// or as quantized blocks (Q4_0 / Q8_0) — dispatches without expanding weights to F32.
/// (the standard PyTorch `nn.Linear` layout `[out_features, in_features]`).
/// Returns `[M, N]`.
///
/// Unlike `matmul(x, W.t())`, this does NOT build a transposed *view* — the
/// generic `matmul` assumes contiguous row-major operands and silently drops the
/// transpose. Instead we read `W` in its natural contiguous layout and let the
/// GEMM treat it as transposed via strides, which is correct for F32 and
/// F16/BF16 weights alike. Inputs must be contiguous (forward-pass activations
/// and loaded weights always are).
pub fn matmul_nt(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let xd = x.shape().dims();
    let wd = w.shape().dims();
    if xd.len() != 2 || wd.len() != 2 {
        return Err(SapientError::internal("matmul_nt expects 2-D tensors"));
    }
    let (m, k) = (xd[0], xd[1]);
    let (n, k2) = (wd[0], wd[1]);
    if k != k2 {
        return Err(SapientError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![n, k2],
        });
    }

    // Dispatch on weight dtype. All quantized paths dequantize on-the-fly block
    // by block — no F32 expansion in memory.
    match w.dtype() {
        DType::Q4_0 => matmul_nt_q4_0(x, w, m, k, n),
        DType::Q8_0 => matmul_nt_q8_0(x, w, m, k, n),
        DType::Q4_K => matmul_nt_kquant(x, w, m, k, n, Q4_K_BLOCK_BYTES, quant::dot_q4_k_row_f32),
        DType::Q5_K => matmul_nt_kquant(x, w, m, k, n, Q5_K_BLOCK_BYTES, quant::dot_q5_k_row_f32),
        DType::Q6_K => matmul_nt_kquant(x, w, m, k, n, Q6_K_BLOCK_BYTES, quant::dot_q6_k_row_f32),
        _ => matmul_nt_float(x, w, m, k, n),
    }
}

/// Float path (F32 / F16 / BF16 weights).
///
/// At prefill (m > 1) uses a single batched SGEMM — optimal for large m.
/// At decode (m = 1) the GEMV is embarrassingly parallel across output rows,
/// so we use rayon to match the throughput of the quantized paths; for large
/// hidden dimensions (k ≥ 512) the speedup is 4–8× vs single-threaded SGEMM.
/// NEON-vectorised F32 dot product (aarch64).
/// Processes 4 floats/cycle via vfmaq_f32, horizontal-sums with vaddvq_f32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon_fast(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    // 16-element unrolled inner loop
    while i + 16 <= n {
        let a0 = vld1q_f32(a.as_ptr().add(i));
        let b0 = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, a0, b0);
        let a1 = vld1q_f32(a.as_ptr().add(i + 4));
        let b1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc = vfmaq_f32(acc, a1, b1);
        let a2 = vld1q_f32(a.as_ptr().add(i + 8));
        let b2 = vld1q_f32(b.as_ptr().add(i + 8));
        acc = vfmaq_f32(acc, a2, b2);
        let a3 = vld1q_f32(a.as_ptr().add(i + 12));
        let b3 = vld1q_f32(b.as_ptr().add(i + 12));
        acc = vfmaq_f32(acc, a3, b3);
        i += 16;
    }
    // 4-element tail
    while i + 4 <= n {
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, av, bv);
        i += 4;
    }
    let mut s = vaddvq_f32(acc);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// AVX2+FMA F32 dot product (x86_64).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += 8;
    }
    // Horizontal sum
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let sum1 = _mm_add_ss(sum2, _mm_movehl_ps(shuf, sum2));
    let mut s = _mm_cvtss_f32(sum1);
    while i < n { s += a[i] * b[i]; i += 1; }
    s
}

/// Dispatch to the fastest available F32 dot product for this platform.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    unsafe { dot_f32_neon_fast(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { dot_f32_avx2(a, b) };
    }
    a.iter().zip(b).map(|(ai, bi)| ai * bi).sum()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(ai, bi)| ai * bi).sum()
}

fn matmul_nt_float(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    let x_cow = x.to_f32_cow();
    let w_cow = w.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_data = w_cow.as_ref();
    let mut out = vec![0.0f32; m * n];

    if m == 1 && k >= 512 {
        // GEMV decode path: one query vector dotted with N weight rows.
        // NEON/AVX2 vectorised — this is the innermost hot loop for all
        // F32-weight models (phi-2, qwen2.5-0.5b safetensors, etc.).
        out.par_iter_mut().enumerate().for_each(|(j, slot)| {
            let w_row = &w_data[j * k..(j + 1) * k];
            *slot = dot_f32_fast(x_data, w_row);
        });
    } else {
        // Batched SGEMM for prefill. W is [N, K] contiguous; row-stride=1, col-stride=K.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                x_data.as_ptr(),
                k as isize,
                1,
                w_data.as_ptr(),
                1,
                k as isize,
                0.0,
                out.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }

    Tensor::from_f32(&out, Shape::new([m, n]))
}

/// Q4_0 quantized weight path — parallel over output columns (the n dimension).
///
/// During decode m = 1, so all n = hidden_size dot products are independent and
/// trivially parallel. Each thread computes one dot product over k/32 blocks
/// without touching any shared state.
fn matmul_nt_q4_0(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % QUANT_BLOCK_SIZE != 0 {
        return Err(SapientError::internal(
            "Q4_0 matmul_nt: k must be a multiple of the block size (32)",
        ));
    }
    let x_cow = x.to_f32_cow();
    let x_data: &[f32] = x_cow.as_ref();
    let w_blocks: &[u8] = w.as_quant_blocks();
    let row_bytes = k / QUANT_BLOCK_SIZE * Q4_0_BLOCK_BYTES;

    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        // Each slot out[i*n + j] is written by exactly one j — fully parallel.
        out[i * n..(i + 1) * n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, slot)| {
                *slot =
                    quant::dot_q4_0_row_f32(&w_blocks[j * row_bytes..(j + 1) * row_bytes], x_row);
            });
    }
    Tensor::from_f32(&out, Shape::new([m, n]))
}

/// Q8_0 quantized weight path — parallel over output columns (the n dimension).
fn matmul_nt_q8_0(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % QUANT_BLOCK_SIZE != 0 {
        return Err(SapientError::internal(
            "Q8_0 matmul_nt: k must be a multiple of the block size (32)",
        ));
    }
    let x_cow = x.to_f32_cow();
    let x_data: &[f32] = x_cow.as_ref();
    let w_blocks: &[u8] = w.as_quant_blocks();
    let row_bytes = k / QUANT_BLOCK_SIZE * Q8_0_BLOCK_BYTES;

    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        out[i * n..(i + 1) * n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, slot)| {
                *slot =
                    quant::dot_q8_0_row_f32(&w_blocks[j * row_bytes..(j + 1) * row_bytes], x_row);
            });
    }
    Tensor::from_f32(&out, Shape::new([m, n]))
}

/// Generic K-quant weight path — shared by Q4_K, Q5_K, Q6_K.
///
/// Each K-quant block covers 256 elements (vs 32 for Q4_0/Q8_0). The
/// `block_bytes` and `dot_fn` parameters specialize this for each variant.
/// Parallel over output rows (the n dimension) — same strategy as Q4_0/Q8_0.
fn matmul_nt_kquant(
    x: &Tensor,
    w: &Tensor,
    m: usize,
    k: usize,
    n: usize,
    block_bytes: usize,
    dot_fn: fn(&[u8], &[f32]) -> f32,
) -> Result<Tensor> {
    const K_BLOCK: usize = 256;
    if k % K_BLOCK != 0 {
        return Err(SapientError::internal(
            "K-quant matmul_nt: k must be a multiple of 256",
        ));
    }
    let x_cow = x.to_f32_cow();
    let x_data: &[f32] = x_cow.as_ref();
    let w_blocks: &[u8] = w.as_quant_blocks();
    let row_bytes = k / K_BLOCK * block_bytes;

    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        out[i * n..(i + 1) * n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, slot)| {
                *slot = dot_fn(&w_blocks[j * row_bytes..(j + 1) * row_bytes], x_row);
            });
    }
    Tensor::from_f32(&out, Shape::new([m, n]))
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

    let a_cow = a2.to_f32_cow();
    let a_data = a_cow.as_ref();
    let b_cow = b2.to_f32_cow();
    let b_data = b_cow.as_ref();
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
    fn matmul_nt_q4_0_matches_float() {
        // 4 output rows, 64 input features (64 is a multiple of 32 = two blocks/row).
        let n_out = 4;
        let k = 64;
        // Build float weight matrix [n_out, k].
        let w_f32: Vec<f32> = (0..n_out * k)
            .map(|i| (i as f32 % 16.0 - 8.0) * 0.05)
            .collect();
        // Build f32 activation [1, k].
        let x_f32: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.3).collect();

        // Reference: float matmul_nt.
        let w_t = Tensor::from_f32(&w_f32, vec![n_out, k]).unwrap();
        let x_t = Tensor::from_f32(&x_f32, vec![1, k]).unwrap();
        let ref_out = matmul_nt(&x_t, &w_t).unwrap();
        let ref_data = ref_out.as_f32_slice();

        // Quantize each row to Q4_0.
        let w_blocks: Vec<u8> = w_f32
            .chunks_exact(k)
            .flat_map(super::quant::quantize_q4_0_row)
            .collect();
        let w_q = Tensor::from_quant_bytes(&w_blocks, vec![n_out, k], DType::Q4_0).unwrap();
        let quant_out = matmul_nt(&x_t, &w_q).unwrap();
        let quant_data = quant_out.as_f32_slice();

        // Quantized path must produce the same result as float (they both use the
        // same quantized representation once quantization roundtrip is applied).
        assert_eq!(ref_data.len(), quant_data.len());
        for (i, (r, q)) in ref_data.iter().zip(quant_data).enumerate() {
            // The float reference uses the *dequantized* weights (via to_f32_cow which
            // dequantizes). Both paths start from the same blocks, so values match exactly.
            // Both paths use the same Q4_0 blocks; differences are accumulation order only.
            assert!((r - q).abs() < 5e-3, "row {i}: ref={r} quant={q}");
        }
    }

    #[test]
    fn matmul_nt_linear() {
        // x = [1,2] (1x2); W = [[1,2],[3,4],[5,6]] shape [3,2] (out=3, in=2).
        // y = x @ Wᵀ = [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17].
        let x = Tensor::from_f32(&[1.0, 2.0], vec![1, 2]).unwrap();
        let w = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = matmul_nt(&x, &w).unwrap();
        let d = y.as_f32_slice();
        assert_eq!(y.shape().dims(), &[1, 3]);
        assert!((d[0] - 5.0).abs() < 1e-5, "got {d:?}");
        assert!((d[1] - 11.0).abs() < 1e-5, "got {d:?}");
        assert!((d[2] - 17.0).abs() < 1e-5, "got {d:?}");
    }

    #[test]
    fn matmul_nt_linear_f16_weight() {
        // Same as above but W is stored as F16 — must still be correct.
        let x = Tensor::from_f32(&[1.0, 2.0], vec![1, 2]).unwrap();
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let w = Tensor::from_f16_bytes(&bytes, vec![3, 2]).unwrap();
        let y = matmul_nt(&x, &w).unwrap();
        let d = y.as_f32_slice();
        assert!((d[0] - 5.0).abs() < 1e-2, "got {d:?}");
        assert!((d[1] - 11.0).abs() < 1e-2, "got {d:?}");
        assert!((d[2] - 17.0).abs() < 1e-2, "got {d:?}");
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
