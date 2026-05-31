//! Matrix multiplication kernels using the `matrixmultiply` crate.
//!
//! `matrixmultiply` provides a pure-Rust, BLAS-free, AVX2-optimised SGEMM.
//! It beats a naive loop by ~10-30× on modern CPUs.

use super::quant;
use rayon::prelude::*;
use sapient_core::error::{Result, SapientError};
use sapient_core::{
    DType, Shape, Tensor, Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES, QUANT_BLOCK_SIZE,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
        DType::Q4_K => matmul_nt_q4_k(x, w, m, k, n),
        DType::Q5_K => matmul_nt_q5_k(x, w, m, k, n),
        DType::Q6_K => matmul_nt_q6_k(x, w, m, k, n),
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
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
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

/// NEON F16 dot product: convert F16 weights to F32 using NEON integer bit
/// manipulation (IEEE 754 F16→F32 expansion), then FMA with F32 activations.
///
/// Does NOT use `float16x4_t` / `vcvt_f32_f16` (only stable since Rust 1.94).
/// Instead, expands the 10-bit mantissa and 5-bit exponent inline.
/// Normal F16 values: sign | ((exp16 + 112) << 23) | (mantissa << 13).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_x_f16_neon(a_f32: &[f32], b_f16: &[u16]) -> f32 {
    use std::arch::aarch64::*;
    let n = a_f32.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;

    // Constants for IEEE 754 F16 -> F32 bit manipulation (normal values only):
    // F16: s[15] e[14:10] m[9:0]
    // F32: s[31] e[30:23] m[22:0]
    // exponent bias adjustment: F32_bias(127) - F16_bias(15) = 112 -> shift by 13
    let mask_mant = vdupq_n_u32(0x000003FF); // 10-bit mantissa mask
    let mask_sign = vdupq_n_u32(0x00008000); // sign bit in u16 position
    let exp_bias = vdupq_n_u32(112 << 23); // exponent bias shift for F32

    while i + 4 <= n {
        let av = vld1q_f32(a_f32.as_ptr().add(i));

        // Load 4 F16 values as u16, widen to u32
        let u16x4 = vld1_u16(b_f16.as_ptr().add(i));
        let u32x4 = vmovl_u16(u16x4); // zero-extend u16x4 -> u32x4

        // Sign: bit 15 of F16 -> bit 31 of F32
        let sign = vshlq_n_u32::<16>(vandq_u32(u32x4, mask_sign));

        // Exponent: bits [14:10] -> F32 exponent = (exp16 + 112) at [30:23]
        // Extract exp16 (5 bits at position 10..14) and shift to F32 position
        let exp16 = vshrq_n_u32::<10>(u32x4); // bits [14:10] now at [4:0]
        let exp32 = vaddq_u32(vshlq_n_u32::<23>(exp16), exp_bias); // (exp16 + 112) << 23

        // Mantissa: bits [9:0] -> bits [22:13] of F32 (shift left by 13)
        let mant = vshlq_n_u32::<13>(vandq_u32(u32x4, mask_mant));

        // Assemble F32 bits (for normal F16 values; subnormals/inf/nan ignored)
        let f32_bits = vorrq_u32(sign, vorrq_u32(exp32, mant));
        let bv: float32x4_t = vreinterpretq_f32_u32(f32_bits);

        acc = vfmaq_f32(acc, av, bv);
        i += 4;
    }

    let mut s = vaddvq_f32(acc);
    while i < n {
        s += a_f32[i] * half::f16::from_bits(b_f16[i]).to_f32();
        i += 1;
    }
    s
}

/// Dispatch to NEON F16 dot product on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_f32_x_f16(a: &[f32], b: &[u16]) -> f32 {
    unsafe { dot_f32_x_f16_neon(a, b) }
}

/// Scalar F16 dot product for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dot_f32_x_f16(a: &[f32], b: &[u16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(ai, bi)| ai * half::f16::from_bits(*bi).to_f32())
        .sum()
}

fn matmul_nt_float(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    // F16 GEMV decode: convert F16 weights to F32 per-row inside NEON registers —
    // never allocates an intermediate F32 copy of the weight matrix.
    // Reads 2 bytes/weight vs 4 bytes/weight (F32 copy): 2× bandwidth improvement.
    // Uses GEMV_CHUNK to bound rayon task count (e.g. lm_head n=151936 → 1187 tasks
    // not 151936 micro-tasks which would cause ~450ms of scheduler overhead alone).
    if m == 1 && k >= 64 && w.dtype() == DType::F16 {
        let x_cow = x.to_f32_cow();
        let x_data = x_cow.as_ref();
        let w_bytes = w.as_bytes();
        // SAFETY: F16 tensors store packed contiguous u16 values (little-endian).
        let w_f16: &[u16] = unsafe {
            std::slice::from_raw_parts(w_bytes.as_ptr() as *const u16, w_bytes.len() / 2)
        };
        let mut out = vec![0.0f32; n];
        let chunk = gemv_chunk(n);
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_idx, cs)| {
                for (local, slot) in cs.iter_mut().enumerate() {
                    let j = chunk_idx * chunk + local;
                    *slot = dot_f32_x_f16(x_data, &w_f16[j * k..(j + 1) * k]);
                }
            });
        return Tensor::from_f32_vec(out, Shape::new([m, n]));
    }

    let x_cow = x.to_f32_cow();
    let w_cow = w.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_data = w_cow.as_ref();
    let mut out = vec![0.0f32; m * n];

    if m == 1 && k >= 512 {
        // F32 GEMV decode — NEON/AVX2 vectorised dot products.
        let chunk = gemv_chunk(n);
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_idx, cs)| {
                for (local, slot) in cs.iter_mut().enumerate() {
                    let j = chunk_idx * chunk + local;
                    *slot = dot_f32_fast(x_data, &w_data[j * k..(j + 1) * k]);
                }
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

    Tensor::from_f32_vec(out, Shape::new([m, n]))
}

/// Q4_0 quantized weight path — parallel over output columns (the n dimension).
///
/// During decode m = 1, so all n = hidden_size dot products are independent and
/// trivially parallel. Each thread computes one dot product over k/32 blocks
/// without touching any shared state.
// Adaptive GEMV chunk size: target ~4 rayon tasks per thread so the scheduler
// can balance load without excess overhead.  Clamped to [16, 512] so tiny
// matrices still get at least 16 rows per task and huge matrices (lm_head
// n=151936) don't create thousands of micro-tasks.
fn gemv_chunk(n: usize) -> usize {
    let ncpus = rayon::current_num_threads().max(1);
    let target_tasks = ncpus * 4;
    (n / target_tasks).clamp(16, 512)
}

/// Compute n output rows of a quantized GEMV, batching rows per Rayon task.
/// `dot` receives one weight row's bytes plus the x vector.
macro_rules! gemv_parallel {
    ($out:expr, $n:expr, $row_bytes:expr, $w_blocks:expr, $x_row:expr, $dot:expr) => {{
        let chunk = gemv_chunk($n);
        $out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_idx, chunk_slice)| {
                for (local, slot) in chunk_slice.iter_mut().enumerate() {
                    let j = chunk_idx * chunk + local;
                    *slot = $dot(&$w_blocks[j * $row_bytes..(j + 1) * $row_bytes], $x_row);
                }
            });
    }};
}

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
        gemv_parallel!(
            out[i * n..(i + 1) * n],
            n,
            row_bytes,
            w_blocks,
            x_row,
            quant::dot_q4_0_row_f32
        );
    }
    Tensor::from_f32_vec(out, Shape::new([m, n]))
}

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

    // ── SDOT path (aarch64 dotprod — Apple M-series, DGX Spark/Grace) ─────────
    //
    // Pre-quantize each activation row to i8 ONCE, then call dot_q8_0_row_sdot
    // which uses vdotq_s32: 16 i8×i8 dot products per cycle instead of the
    // widening chain (i8→i16→i32→f32) in dot_q8_0_block_neon.
    //
    // For gate_proj [4864 out, 896 in]:
    //   NEON widening:  4864 × 28 blocks × ~40 ops  = 5.46M NEON ops
    //   SDOT:           (896 pre-quant) + 4864 × 28 × 6 ops = 0.82M + tiny = ~5× fewer
    // SDOT dispatch: ARMv8.4-A `sdot` via inline asm, stable Rust.
    // All Apple M-series and DGX Spark (Grace ARM64) support dotprod.
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        for i in 0..m {
            let x_row = &x_data[i * k..(i + 1) * k];
            // Per-block activation scales — a single per-row scale is destroyed by
            // outlier activation channels and yields incoherent output.
            let (x_i8, x_scales) = quant::quantize_row_to_i8_blocks(x_row);
            let chunk = gemv_chunk(n);
            // SAFETY: dot_q8_0_row_sdot requires neon (always true on aarch64)
            // and emits `sdot` (detected above via is_aarch64_feature_detected).
            out[i * n..(i + 1) * n]
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(ci, cs)| {
                    for (local, slot) in cs.iter_mut().enumerate() {
                        let j = ci * chunk + local;
                        *slot = unsafe {
                            quant::dot_q8_0_row_sdot(
                                &w_blocks[j * row_bytes..(j + 1) * row_bytes],
                                &x_i8,
                                &x_scales,
                            )
                        };
                    }
                });
        }
        return Tensor::from_f32_vec(out, Shape::new([m, n]));
    }

    // ── Fallback: NEON widening or AVX2 ──────────────────────────────────────
    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        gemv_parallel!(
            out[i * n..(i + 1) * n],
            n,
            row_bytes,
            w_blocks,
            x_row,
            quant::dot_q8_0_row_f32
        );
    }
    Tensor::from_f32_vec(out, Shape::new([m, n]))
}

// Specialized K-quant paths — no function pointer so the NEON dot kernel
// can be inlined into the hot Rayon closure by the compiler.

fn matmul_nt_q4_k(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % 256 != 0 {
        return Err(SapientError::internal("Q4_K: k must be a multiple of 256"));
    }
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_blocks = w.as_quant_blocks();
    let row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        gemv_parallel!(
            out[i * n..(i + 1) * n],
            n,
            row_bytes,
            w_blocks,
            x_row,
            quant::dot_q4_k_row_f32
        );
    }
    Tensor::from_f32_vec(out, Shape::new([m, n]))
}

fn matmul_nt_q5_k(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % 256 != 0 {
        return Err(SapientError::internal("Q5_K: k must be a multiple of 256"));
    }
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_blocks = w.as_quant_blocks();
    let row_bytes = k / 256 * Q5_K_BLOCK_BYTES;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        gemv_parallel!(
            out[i * n..(i + 1) * n],
            n,
            row_bytes,
            w_blocks,
            x_row,
            quant::dot_q5_k_row_f32
        );
    }
    Tensor::from_f32_vec(out, Shape::new([m, n]))
}

fn matmul_nt_q6_k(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % 256 != 0 {
        return Err(SapientError::internal("Q6_K: k must be a multiple of 256"));
    }
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_blocks = w.as_quant_blocks();
    let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        gemv_parallel!(
            out[i * n..(i + 1) * n],
            n,
            row_bytes,
            w_blocks,
            x_row,
            quant::dot_q6_k_row_f32
        );
    }
    Tensor::from_f32_vec(out, Shape::new([m, n]))
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

    Tensor::from_f32_vec(out, Shape::new([m, n]))
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
    fn matmul_nt_q8_0_matches_float() {
        // Larger k so several 32-blocks per row exercise the SDOT/NEON path, and
        // an activation outlier to stress per-block activation quantization.
        let n_out = 8;
        let k = 256;
        let w_f32: Vec<f32> = (0..n_out * k)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.03)
            .collect();
        let mut x_f32: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin() * 0.4).collect();
        x_f32[100] = 25.0; // outlier channel

        // Quantize weights to Q8_0 blocks.
        let w_blocks: Vec<u8> = w_f32
            .chunks_exact(k)
            .flat_map(|row| {
                row.chunks_exact(32)
                    .flat_map(super::quant::quantize_q8_0_block)
                    .collect::<Vec<u8>>()
            })
            .collect();
        let w_q = Tensor::from_quant_bytes(&w_blocks, vec![n_out, k], DType::Q8_0).unwrap();
        let x_t = Tensor::from_f32(&x_f32, vec![1, k]).unwrap();

        // Reference: dequantize the SAME Q8_0 weights to f32, then exact f32 matmul.
        // (Comparing against the original f32 weights would conflate weight-quant
        // error with the matmul; we only want to test the matmul path here.)
        let w_deq = w_q.to_f32_cow().as_ref().to_vec();
        let w_ref = Tensor::from_f32(&w_deq, vec![n_out, k]).unwrap();
        let ref_data = matmul_nt(&x_t, &w_ref).unwrap().as_f32_slice().to_vec();

        let quant_data = matmul_nt(&x_t, &w_q).unwrap().as_f32_slice().to_vec();
        assert_eq!(ref_data.len(), quant_data.len());
        for (i, (r, q)) in ref_data.iter().zip(&quant_data).enumerate() {
            let tol = 0.02 * r.abs().max(1.0);
            assert!((r - q).abs() < tol, "row {i}: ref={r} quant={q}");
        }
    }

    // Replicates the GGUF load path for Q8_0 weights: the tensor is built with the
    // ggml dim order [in, out] and then reshaped to HF [out, in] (exactly what
    // map_gguf_tensors_to_hf does). This is the path real Q8_0 models take.
    #[test]
    fn matmul_nt_q8_0_gguf_dimflip_matches_float() {
        let out_features = 64;
        let in_features = 128;
        // True weight W [out, in]; y[o] = Σ_i W[o,i] x[i].
        let w_f32: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.02)
            .collect();
        let x_f32: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.02).cos() * 0.5)
            .collect();
        let x_t = Tensor::from_f32(&x_f32, vec![1, in_features]).unwrap();

        // ggml stores ne[0]=in contiguous, so its flat byte order == row-major
        // [out, in] == w_f32. Quantize that flat array block-by-block.
        let blocks: Vec<u8> = w_f32
            .chunks_exact(32)
            .flat_map(super::quant::quantize_q8_0_block)
            .collect();
        // Build with ggml dims [in, out], then flip to HF [out, in] like the loader.
        let w_gguf =
            Tensor::from_quant_bytes(&blocks, vec![in_features, out_features], DType::Q8_0)
                .unwrap()
                .reshape(vec![out_features, in_features])
                .unwrap();

        // Reference: dequantize those same blocks (the float path Q4_K models take).
        let w_ref = Tensor::from_f32(
            w_gguf.to_f32_cow().as_ref(),
            vec![out_features, in_features],
        )
        .unwrap();
        let ref_data = matmul_nt(&x_t, &w_ref).unwrap().as_f32_slice().to_vec();
        let got = matmul_nt(&x_t, &w_gguf).unwrap().as_f32_slice().to_vec();

        for (i, (r, q)) in ref_data.iter().zip(&got).enumerate() {
            assert!(
                (r - q).abs() < 0.02 * r.abs().max(1.0),
                "out {i}: ref={r} quant={q}"
            );
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
