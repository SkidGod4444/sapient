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

    // Thermal governor sample point (Phase 8.4): rate-limited to one sysfs read
    // per 500 ms — every other call is a single atomic compare.
    crate::thermal::tick();

    // Dispatch on weight dtype. All quantized paths dequantize on-the-fly block
    // by block — no F32 expansion in memory.
    match w.dtype() {
        DType::Q4_0 => matmul_nt_q4_0(x, w, m, k, n),
        DType::Q8_0 => matmul_nt_q8_0(x, w, m, k, n),
        DType::Q4_K => matmul_nt_q4_k(x, w, m, k, n),
        DType::Q4_K_R4 => matmul_nt_q4_k_r4(x, w, m, k, n),
        DType::Q5_K => matmul_nt_q5_k(x, w, m, k, n),
        DType::Q6_K => matmul_nt_q6_k(x, w, m, k, n),
        DType::Q6_K_R4 => matmul_nt_q6_k_r4(x, w, m, k, n),
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
        for_each_out_chunk(&mut out, chunk, |chunk_idx, cs| {
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
        for_each_out_chunk(&mut out, chunk, |chunk_idx, cs| {
            for (local, slot) in cs.iter_mut().enumerate() {
                let j = chunk_idx * chunk + local;
                *slot = dot_f32_fast(x_data, &w_data[j * k..(j + 1) * k]);
            }
        });
    } else {
        // Batched SGEMM for prefill — split across X row blocks so it uses
        // every core (one big single-threaded sgemm left most of the machine
        // idle; the SigLIP-896 tower spends its life here). Each block is an
        // independent sgemm over the same K reduction writing a disjoint
        // output slice → bit-identical to the single call.
        let flops = m * k * n;
        let mblock = if m >= 2 && flops >= (1 << 20) {
            m.div_ceil(rayon::current_num_threads().max(1)).max(4)
        } else {
            m
        };
        out.par_chunks_mut(mblock * n)
            .enumerate()
            .for_each(|(bi, out_block)| {
                let m0 = bi * mblock;
                let mc = out_block.len() / n;
                unsafe {
                    matrixmultiply::sgemm(
                        mc,
                        k,
                        n,
                        1.0,
                        x_data[m0 * k..].as_ptr(),
                        k as isize,
                        1,
                        w_data.as_ptr(),
                        1,
                        k as isize,
                        0.0,
                        out_block.as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
            });
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
//
// Thermal-governed (Phase 8.4): while the board runs hot, work splits into
// EXACTLY `effective_threads()` tasks — with fewer tasks than pool threads, the
// surplus rayon workers have nothing to steal and idle, cutting package power so
// the clocks stay up. Two traps found on-device (Pi 5 soak): the ×4 multiplier
// must NOT apply while governed (4 threads × 4 = 16 tasks keeps every core busy
// even at a reduced target), and the 512-row chunk cap must not re-split big
// matrices (lm_head/151936 ÷ 512 = 297 tasks — same effect). Inert path
// (== rayon::current_num_threads) is unchanged.
fn gemv_chunk(n: usize) -> usize {
    // The governed comparison is within RAYON's domain (the governor sheds
    // rayon cores; the spin pool is disabled entirely while governed).
    // Comparing eff against the spin pool's parallelism instead made a
    // plain RAYON_NUM_THREADS=1 run look "governed" and collapse every
    // GEMV to one chunk.
    let rayon_n = rayon::current_num_threads().max(1);
    let eff = crate::thermal::effective_threads();
    if eff < rayon_n {
        // Governed: one task per allowed thread; no upper chunk cap.
        return (n / eff.max(1)).max(16);
    }
    // Task-count parallelism follows whichever pool will run the chunks.
    let ncpus = if crate::spinpool::enabled() {
        crate::spinpool::parallelism()
    } else {
        rayon_n
    };
    {
        // Tasks per core (default 4, for load balancing). Env-tunable to probe
        // rayon fork/join overhead on high-core hosts where fewer, coarser tasks
        // per GEMV scale better (Neoverse: decode is coordination-bound, not
        // bandwidth-bound, so ×4 over-splits). The env path drops the 512 cap so
        // TPC=1 gives exactly one task per core.
        match std::env::var("SAPIENT_GEMV_TPC")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
        {
            Some(tpc) => (n / (ncpus * tpc)).max(16),
            None => (n / (ncpus * 4)).clamp(16, 512),
        }
    }
}

/// Q8_K-format activations for the Q4_K_R4 matmuls (ONE f32 scale per
/// 256-element super-block + per-32 sums; integer-domain sub-scale combine —
/// the pp512 activation-format rung, BENCHMARKS.md). **Default ON — measured
/// a win on every platform** (decode: M4 qwen +6.3%, M4 llama +4.9%, Thor
/// 14-core +22.7%, Pi 5 +3.9%; Thor prefill −12.7% TTFT) with real-model
/// greedy verification passed (llama.cpp-precedented per-256 accuracy
/// class). `SAPIENT_Q8K_ACT=0` reverts to the per-32 W4A8 format.
#[cfg(target_arch = "aarch64")]
fn q8k_activations() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SAPIENT_Q8K_ACT")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Raw output pointer that may cross threads: chunk geometry guarantees the
/// slices built from it are disjoint (same contract as `par_chunks_mut`).
/// Accessed via [`SyncPtr::get`] so closures capture the (Sync) wrapper, not
/// the raw pointer field (edition-2021 closures capture individual fields).
struct SyncPtr(*mut f32);
unsafe impl Sync for SyncPtr {}
impl SyncPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

/// Dispatch a GEMV's disjoint output chunks either on the persistent spin
/// pool (decode hot path — one atomic op-handoff instead of a rayon
/// fork/join per matmul; see `spinpool.rs`) or on rayon (thermal-governed
/// mode / `SAPIENT_SPINPOOL=0`). Chunk geometry is identical on both paths,
/// so per-slot results are bit-identical.
fn for_each_out_chunk<F>(out: &mut [f32], chunk: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if out.is_empty() {
        return;
    }
    // SAPIENT_SPINPOOL_DEBUG=1: periodic dispatch-route census on stderr.
    if std::env::var("SAPIENT_SPINPOOL_DEBUG").is_ok() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SPIN: AtomicU64 = AtomicU64::new(0);
        static RAYON: AtomicU64 = AtomicU64::new(0);
        let (s, r) = if crate::spinpool::enabled() {
            (
                SPIN.fetch_add(1, Ordering::Relaxed) + 1,
                RAYON.load(Ordering::Relaxed),
            )
        } else {
            (
                SPIN.load(Ordering::Relaxed),
                RAYON.fetch_add(1, Ordering::Relaxed) + 1,
            )
        };
        if (s + r) % 2000 == 0 {
            eprintln!(
                "[spinpool-debug] spin={s} rayon={r} chunk={chunk} len={} n_chunks={}",
                out.len(),
                out.len().div_ceil(chunk)
            );
        }
    }
    if crate::spinpool::enabled() {
        let len = out.len();
        let n_chunks = len.div_ceil(chunk);
        let base = SyncPtr(out.as_mut_ptr());
        crate::spinpool::pool().run(n_chunks, &|ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(len);
            // SAFETY: [start, end) ranges are disjoint per chunk index and
            // `out` outlives `run` (it blocks until every chunk completes).
            let cs = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), end - start) };
            f(ci, cs);
        });
    } else {
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, cs)| f(ci, cs));
    }
}

/// Compute n output rows of a quantized GEMV, batching rows per task.
/// `dot` receives one weight row's bytes plus the x vector.
macro_rules! gemv_parallel {
    ($out:expr, $n:expr, $row_bytes:expr, $w_blocks:expr, $x_row:expr, $dot:expr) => {{
        let chunk = gemv_chunk($n);
        for_each_out_chunk(&mut $out, chunk, |chunk_idx, chunk_slice| {
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
        // ── Blocked W8A8 GEMM (m ≥ 8: prefill / vision towers) ───────────────
        // The per-row loop below runs one rayon fork-join and one activation
        // quantization PER ROW — 4096 barriers for a SigLIP-896 tower. Here:
        // all rows quantize once, ONE parallel region over weight-row chunks,
        // and each weight row's bytes stay L1-resident across every activation
        // row (output built row-major-transposed, then flipped). Per-element
        // math is the same kernel with the same scales → bit-identical to the
        // per-row path.
        if m >= 8 {
            let bpr = k / QUANT_BLOCK_SIZE; // activation blocks per row
            let mut x_i8 = vec![0i8; m * k];
            let mut x_scales = vec![0.0f32; m * bpr];
            x_i8.par_chunks_mut(k)
                .zip(x_scales.par_chunks_mut(bpr))
                .enumerate()
                .for_each(|(i, (xi, xs))| {
                    let (qi, qs) = quant::quantize_row_to_i8_blocks(&x_data[i * k..(i + 1) * k]);
                    xi.copy_from_slice(&qi);
                    xs.copy_from_slice(&qs);
                });

            let mut out_t = vec![0.0f32; n * m]; // [n, m]
            let wchunk = gemv_chunk(n);
            out_t
                .par_chunks_mut(wchunk * m)
                .enumerate()
                .for_each(|(ci, oc)| {
                    let j0 = ci * wchunk;
                    for (jl, orow) in oc.chunks_mut(m).enumerate() {
                        let j = j0 + jl;
                        let wrow = &w_blocks[j * row_bytes..(j + 1) * row_bytes];
                        for (i, slot) in orow.iter_mut().enumerate() {
                            // SAFETY: dotprod verified above.
                            *slot = unsafe {
                                quant::dot_q8_0_row_sdot(
                                    wrow,
                                    &x_i8[i * k..(i + 1) * k],
                                    &x_scales[i * bpr..(i + 1) * bpr],
                                )
                            };
                        }
                    }
                });
            // Transpose [n, m] → [m, n] (parallel over output rows).
            out.par_chunks_mut(n).enumerate().for_each(|(i, orow)| {
                for (j, o) in orow.iter_mut().enumerate() {
                    *o = out_t[j * m + i];
                }
            });
            return Tensor::from_f32_vec(out, Shape::new([m, n]));
        }

        for i in 0..m {
            let x_row = &x_data[i * k..(i + 1) * k];
            // Per-block activation scales — a single per-row scale is destroyed by
            // outlier activation channels and yields incoherent output.
            let (x_i8, x_scales) = quant::quantize_row_to_i8_blocks(x_row);
            let chunk = gemv_chunk(n);
            // SAFETY: dot_q8_0_row_sdot requires neon (always true on aarch64)
            // and emits `sdot` (detected above via is_aarch64_feature_detected).
            for_each_out_chunk(&mut out[i * n..(i + 1) * n], chunk, |ci, cs| {
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

// MEASURED NO-GO (2026-07-10): cache-blocking the m ≥ 2 SMMLA paths over
// 128-row activation panels was tried and REVERTED. The activation matrix
// (3–4 MB at 2k tokens) already sits in LLC on every tested machine, so the
// per-group activation re-stream never hit DRAM — while panelling made the
// WEIGHTS stream ceil(m/128)× instead of once (L1-resident group across all
// m). A/B on a ~2.2k-token prompt: Thor +7% TTFT (regression), M4 qwen even,
// M4 llama −6% (confounded by run order). The original loop order is optimal
// for traffic at these shapes; llama.cpp's remaining pp512 edge is register-
// level (deeper int8 tiles amortizing the nibble-unpack), not cache blocking.
// The panelled-vs-per-row bit-identity test below is kept as a prefill gate.

/// Q4_K_R4 (row-interleaved) GEMV: weight rows come in groups of 4 whose
/// super-blocks are block-interleaved into one contiguous stream (see
/// `repack_q4_k_rows4`). The aarch64 SDOT path walks each group front-to-back
/// with `dot_q4_k_4rows_r4_neon` — one prefetch stream per task instead of
/// four. Repacking only happens on aarch64+dotprod CPU engines, but a portable
/// de-interleaving fallback keeps the dtype correct everywhere (tests, x86).
fn matmul_nt_q4_k_r4(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % 256 != 0 || n % 4 != 0 {
        return Err(SapientError::internal(
            "Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4",
        ));
    }
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_blocks = w.as_quant_blocks();
    let row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    #[cfg(target_arch = "aarch64")]
    let group_bytes = 4 * row_bytes;
    let mut out = vec![0.0f32; m * n];

    // ── i8mm SMMLA prefill path (m ≥ 2, ARMv8.6) ────────────────────────────
    // Two activation rows per pass through each weight group: one `smmla`
    // (32 MACs) replaces four `sdot`s, and the weight stream is read once per
    // PAIR of prompt tokens instead of once per token. Output is built
    // group-major (transposed) so rayon tasks own contiguous chunks.
    #[cfg(target_arch = "aarch64")]
    if m >= 2 && std::arch::is_aarch64_feature_detected!("i8mm") {
        let q8k = q8k_activations();
        let quantized: Vec<(Vec<i8>, Vec<f32>, Vec<i32>)> = (0..m)
            .map(|i| {
                let row = &x_data[i * k..(i + 1) * k];
                if q8k {
                    quant::quantize_row_to_q8k(row)
                } else {
                    let (q, s) = quant::quantize_row_to_i8_blocks(row);
                    let sums = quant::i8_block_sums(&q);
                    (q, s, sums)
                }
            })
            .collect();
        let groups = n / 4;
        let mut out_t = vec![0.0f32; n * m]; // [group-rows][m]
        out_t
            .par_chunks_mut(4 * m)
            .enumerate()
            .for_each(|(g, chunk)| {
                let group = &w_blocks[g * group_bytes..(g + 1) * group_bytes];
                let mut xi = 0usize;
                // NOTE: an x4 register tile (weight nibbles unpacked once per
                // TWO activation pairs) was built, bit-identity-gated, and
                // MEASURED here (2026-07-10): M4 −5% (register pressure —
                // 16 activation TRN vectors + 8 weight vectors + accumulators
                // spill past the 32-register budget), Thor exactly neutral.
                // The x2 tile below is the measured optimum; the unpack ALU
                // is already hidden behind smmla latency on OoO cores.
                while xi + 2 <= m {
                    let (x0, s0, b0) = &quantized[xi];
                    let (x1, s1, b1) = &quantized[xi + 1];
                    // SAFETY: i8mm verified above; slice is one R4 group.
                    let v = unsafe {
                        if q8k {
                            quant::dot_q4_k_4rows_r4_x2_q8k_smmla(group, x0, s0, b0, x1, s1, b1)
                        } else {
                            quant::dot_q4_k_4rows_r4_x2_smmla(group, x0, s0, b0, x1, s1, b1)
                        }
                    };
                    for r in 0..4 {
                        chunk[r * m + xi] = v[r][0];
                        chunk[r * m + xi + 1] = v[r][1];
                    }
                    xi += 2;
                }
                if xi < m {
                    let (x0, s0, b0) = &quantized[xi];
                    // SAFETY: FEAT_I8MM implies dotprod-era NEON.
                    let v = unsafe {
                        if q8k {
                            quant::dot_q4_k_4rows_r4_q8k_neon(group, x0, s0, b0)
                        } else {
                            quant::dot_q4_k_4rows_r4_neon(group, x0, s0, b0)
                        }
                    };
                    for r in 0..4 {
                        chunk[r * m + xi] = v[r];
                    }
                }
            });
        debug_assert_eq!(groups * 4, n);
        for g in 0..groups {
            for r in 0..4 {
                for i in 0..m {
                    out[i * n + g * 4 + r] = out_t[(g * 4 + r) * m + i];
                }
            }
        }
        return Tensor::from_f32_vec(out, Shape::new([m, n]));
    }

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        let q8k = q8k_activations();
        for i in 0..m {
            let x_row = &x_data[i * k..(i + 1) * k];
            let (x_i8, x_scales, x_sums) = if q8k {
                quant::quantize_row_to_q8k(x_row)
            } else {
                let (q, s) = quant::quantize_row_to_i8_blocks(x_row);
                let sums = quant::i8_block_sums(&q);
                (q, s, sums)
            };
            let groups = n / 4;
            let gchunk = (gemv_chunk(n) / 4).max(1);
            for_each_out_chunk(&mut out[i * n..(i + 1) * n], gchunk * 4, |ci, cs| {
                let g0 = ci * gchunk;
                for (gl, slots) in cs.chunks_mut(4).enumerate() {
                    let g = g0 + gl;
                    debug_assert!(g < groups);
                    let group = &w_blocks[g * group_bytes..(g + 1) * group_bytes];
                    // SAFETY: dotprod verified above; slice is one R4 group.
                    let v = unsafe {
                        if q8k {
                            quant::dot_q4_k_4rows_r4_q8k_neon(group, &x_i8, &x_scales, &x_sums)
                        } else {
                            quant::dot_q4_k_4rows_r4_neon(group, &x_i8, &x_scales, &x_sums)
                        }
                    };
                    slots.copy_from_slice(&v[..slots.len()]);
                }
            });
        }
        return Tensor::from_f32_vec(out, Shape::new([m, n]));
    }

    // Portable fallback: de-interleave each group's rows and use the scalar
    // W4A8 dot (correctness path — repacking is only enabled where the SDOT
    // path exists, so this only runs in tests / exotic configs).
    for i in 0..m {
        let x_row = &x_data[i * k..(i + 1) * k];
        let (x_i8, x_scales) = quant::quantize_row_to_i8_blocks(x_row);
        let x_sums = quant::i8_block_sums(&x_i8);
        let nb = k / 256;
        let mut row_buf = vec![0u8; row_bytes];
        for g in 0..n / 4 {
            for r in 0..4 {
                for b in 0..nb {
                    let src = (g * 4 * nb + b * 4 + r) * Q4_K_BLOCK_BYTES;
                    row_buf[b * Q4_K_BLOCK_BYTES..(b + 1) * Q4_K_BLOCK_BYTES]
                        .copy_from_slice(&w_blocks[src..src + Q4_K_BLOCK_BYTES]);
                }
                out[i * n + g * 4 + r] =
                    quant::dot_q4_k_row_q8_scalar(&row_buf, &x_i8, &x_scales, &x_sums);
            }
        }
    }
    Tensor::from_f32_vec(out, Shape::new([m, n]))
}

fn matmul_nt_q4_k(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % 256 != 0 {
        return Err(SapientError::internal("Q4_K: k must be a multiple of 256"));
    }
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_blocks = w.as_quant_blocks();
    let row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    let mut out = vec![0.0f32; m * n];

    // ── W4A8 SDOT path (aarch64 dotprod) ─────────────────────────────────────
    // Quantize each activation row to per-32-block i8 ONCE, then do the Q4_K dot
    // with int8 activations + `sdot` (16 MACs/instr) instead of expanding every
    // nibble to f32. This is the ARM hot path for K-quant decode — the dominant
    // cost on a Raspberry Pi 5 / Cortex-A76. Per-block scales (not one per row)
    // keep activation outliers from collapsing the signal (cf. Q8_0 SDOT).
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        let q8k = q8k_activations();
        for i in 0..m {
            let x_row = &x_data[i * k..(i + 1) * k];
            let (x_i8, x_scales, x_sums) = if q8k {
                quant::quantize_row_to_q8k(x_row)
            } else {
                let (q, s) = quant::quantize_row_to_i8_blocks(x_row);
                let sums = quant::i8_block_sums(&q);
                (q, s, sums)
            };
            let chunk = gemv_chunk(n);
            for_each_out_chunk(&mut out[i * n..(i + 1) * n], chunk, |ci, cs| {
                // Multi-row GEMV: 4 weight rows share one pass over the
                // activations (llama.cpp-style) — 4× less x traffic and four
                // independent sdot chains. Remainder rows use the single-row
                // kernel; per-row results are bit-identical either way.
                let start = ci * chunk;
                let mut local = 0usize;
                while local + 4 <= cs.len() {
                    let j = start + local;
                    let r = |o: usize| &w_blocks[(j + o) * row_bytes..(j + o + 1) * row_bytes];
                    let rows = [r(0), r(1), r(2), r(3)];
                    // SAFETY: dotprod verified above; slices are whole Q4_K rows.
                    let v = unsafe {
                        if q8k {
                            quant::dot_q4_k_4rows_q8k_neon(rows, &x_i8, &x_scales, &x_sums)
                        } else {
                            quant::dot_q4_k_4rows_q8_neon(rows, &x_i8, &x_scales, &x_sums)
                        }
                    };
                    cs[local..local + 4].copy_from_slice(&v);
                    local += 4;
                }
                for slot in &mut cs[local..] {
                    let j = start + local;
                    let row = &w_blocks[j * row_bytes..(j + 1) * row_bytes];
                    // SAFETY: as above.
                    *slot = unsafe {
                        if q8k {
                            quant::dot_q4_k_row_q8k_neon(row, &x_i8, &x_scales, &x_sums)
                        } else {
                            quant::dot_q4_k_row_q8_neon(row, &x_i8, &x_scales, &x_sums)
                        }
                    };
                    local += 1;
                }
            });
        }
        return Tensor::from_f32_vec(out, Shape::new([m, n]));
    }

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

/// Q6_K_R4 (row-interleaved) GEMV — same scheme as `matmul_nt_q4_k_r4`: one
/// contiguous packed stream per 4-row group (aarch64 NEON), with a portable
/// de-interleaving scalar fallback for correctness everywhere else.
fn matmul_nt_q6_k_r4(x: &Tensor, w: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    if k % 256 != 0 || n % 4 != 0 {
        return Err(SapientError::internal(
            "Q6_K_R4: k must be a multiple of 256 and rows a multiple of 4",
        ));
    }
    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_blocks = w.as_quant_blocks();
    let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    #[cfg(target_arch = "aarch64")]
    let group_bytes = 4 * row_bytes;
    let mut out = vec![0.0f32; m * n];

    // ── i8mm SMMLA prefill path (m ≥ 2, ARMv8.6) — see matmul_nt_q4_k_r4 ────
    #[cfg(target_arch = "aarch64")]
    if m >= 2 && std::arch::is_aarch64_feature_detected!("i8mm") {
        let quantized: Vec<(Vec<i8>, Vec<f32>)> = (0..m)
            .map(|i| quant::quantize_row_to_i8_blocks(&x_data[i * k..(i + 1) * k]))
            .collect();
        let groups = n / 4;
        let mut out_t = vec![0.0f32; n * m]; // [group-rows][m]
        out_t
            .par_chunks_mut(4 * m)
            .enumerate()
            .for_each(|(g, chunk)| {
                let group = &w_blocks[g * group_bytes..(g + 1) * group_bytes];
                let mut xi = 0usize;
                while xi + 2 <= m {
                    let (x0, s0) = &quantized[xi];
                    let (x1, s1) = &quantized[xi + 1];
                    // SAFETY: i8mm verified above; slice is one R4 group.
                    let v = unsafe { quant::dot_q6_k_4rows_r4_x2_smmla(group, x0, s0, x1, s1) };
                    for r in 0..4 {
                        chunk[r * m + xi] = v[r][0];
                        chunk[r * m + xi + 1] = v[r][1];
                    }
                    xi += 2;
                }
                if xi < m {
                    let (x0, s0) = &quantized[xi];
                    // SAFETY: FEAT_I8MM implies dotprod-era NEON.
                    let v = unsafe { quant::dot_q6_k_4rows_r4_q8_neon(group, x0, s0) };
                    for r in 0..4 {
                        chunk[r * m + xi] = v[r];
                    }
                }
            });
        debug_assert_eq!(groups * 4, n);
        for g in 0..groups {
            for r in 0..4 {
                for i in 0..m {
                    out[i * n + g * 4 + r] = out_t[(g * 4 + r) * m + i];
                }
            }
        }
        return Tensor::from_f32_vec(out, Shape::new([m, n]));
    }

    #[cfg(target_arch = "aarch64")]
    {
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        for i in 0..m {
            let x_row = &x_data[i * k..(i + 1) * k];
            let quantized = dotprod.then(|| quant::quantize_row_to_i8_blocks(x_row));
            let gchunk = (gemv_chunk(n) / 4).max(1);
            for_each_out_chunk(&mut out[i * n..(i + 1) * n], gchunk * 4, |ci, cs| {
                let g0 = ci * gchunk;
                for (gl, slots) in cs.chunks_mut(4).enumerate() {
                    let g = g0 + gl;
                    let group = &w_blocks[g * group_bytes..(g + 1) * group_bytes];
                    // SAFETY: NEON baseline; dotprod verified when used.
                    let v = match &quantized {
                        Some((x_i8, x_scales)) => unsafe {
                            quant::dot_q6_k_4rows_r4_q8_neon(group, x_i8, x_scales)
                        },
                        None => unsafe { quant::dot_q6_k_4rows_r4_neon(group, x_row) },
                    };
                    slots.copy_from_slice(&v[..slots.len()]);
                }
            });
        }
        return Tensor::from_f32_vec(out, Shape::new([m, n]));
    }

    // Portable fallback: de-interleave each group and use the scalar dot.
    #[allow(unreachable_code)]
    {
        for i in 0..m {
            let x_row = &x_data[i * k..(i + 1) * k];
            let nb = k / 256;
            let mut row_buf = vec![0u8; row_bytes];
            for g in 0..n / 4 {
                for r in 0..4 {
                    for b in 0..nb {
                        let src = (g * 4 * nb + b * 4 + r) * Q6_K_BLOCK_BYTES;
                        row_buf[b * Q6_K_BLOCK_BYTES..(b + 1) * Q6_K_BLOCK_BYTES]
                            .copy_from_slice(&w_blocks[src..src + Q6_K_BLOCK_BYTES]);
                    }
                    out[i * n + g * 4 + r] = quant::dot_q6_k_row_f32(&row_buf, x_row);
                }
            }
        }
        Tensor::from_f32_vec(out, Shape::new([m, n]))
    }
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

    // ── W6A8 SDOT path (aarch64 dotprod) ─────────────────────────────────────
    // One `sdot` per 16-element scale group (the −32 folded into the int8
    // quants) instead of the f32 path's widen/convert/FMA chains — the same
    // W·A8 treatment that made Q4_K fast, applied to Q6_K.
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        for i in 0..m {
            let x_row = &x_data[i * k..(i + 1) * k];
            let (x_i8, x_scales) = quant::quantize_row_to_i8_blocks(x_row);
            let chunk = gemv_chunk(n);
            for_each_out_chunk(&mut out[i * n..(i + 1) * n], chunk, |ci, cs| {
                for (local, slot) in cs.iter_mut().enumerate() {
                    let j = ci * chunk + local;
                    // SAFETY: dotprod verified above; slice is one Q6_K row.
                    *slot = unsafe {
                        quant::dot_q6_k_row_q8_neon(
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
    #[cfg(target_arch = "aarch64")]
    fn q8_0_gemm_path_matches_per_row_path() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        // m = 16 triggers the blocked GEMM; compare each row against the m = 1
        // (per-row GEMV) path — same kernel, same scales → bit-identical.
        let (m, k, n) = (16usize, 96usize, 24usize);
        let mut seed = 0x8A8Au64;
        let mut nf = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let xv: Vec<f32> = (0..m * k).map(|_| nf() * 0.5).collect();
        let wv: Vec<f32> = (0..n * k).map(|_| nf() * 0.2).collect();
        // Build raw Q8_0 blocks for W (row-major, per-32 blocks).
        let mut wb: Vec<u8> = Vec::with_capacity(n * k / 32 * 34);
        for row in wv.chunks(k) {
            for blk in row.chunks(32) {
                wb.extend_from_slice(&crate::kernels::quant::quantize_q8_0_block(blk));
            }
        }
        let w_q8 = Tensor::from_quant_bytes(&wb, vec![n, k], sapient_core::DType::Q8_0).unwrap();

        let x_all = Tensor::from_f32(&xv, Shape::new([m, k])).unwrap();
        let full = matmul_nt(&x_all, &w_q8).unwrap().to_f32_vec();
        for i in 0..m {
            let x_row = Tensor::from_f32(&xv[i * k..(i + 1) * k], Shape::new([1, k])).unwrap();
            let want = matmul_nt(&x_row, &w_q8).unwrap().to_f32_vec();
            for j in 0..n {
                assert_eq!(
                    full[i * n + j].to_bits(),
                    want[j].to_bits(),
                    "row {i} col {j}: {} vs {}",
                    full[i * n + j],
                    want[j]
                );
            }
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
    /// Panel-blocked SMMLA prefill must be bit-identical to per-row GEMV:
    /// m = 259 spans two full 128-row panels plus an odd 3-row remainder,
    /// exercising panel boundaries, the mid-panel pair loop, and the odd
    /// final panel. (The R4/x2-SMMLA kernels are bit-identical to the
    /// single-row Q4_K kernel by their own gates; this test pins the LOOP
    /// RESTRUCTURE — chunk bookkeeping, panel offsets, remainders.)
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q4_k_r4_prefill_matches_per_row() {
        let n = 8usize;
        let k = 256usize; // one Q4_K super-block per row
        let m = 259usize; // 2 full panels + odd remainder

        // Deterministic pseudo-random Q4_K blocks with sane f16 scales.
        let mut blocks = vec![0u8; n * Q4_K_BLOCK_BYTES];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = ((i * 131 + 7) % 251) as u8;
        }
        for r in 0..n {
            let base = r * Q4_K_BLOCK_BYTES;
            // d ≈ 0.05, dmin ≈ 0.03 (little-endian f16) — keep values finite.
            blocks[base..base + 2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
            blocks[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes());
        }
        let packed = super::quant::repack_q4_k_rows4(&blocks, n, k);
        let w_r4 = Tensor::from_quant_bytes(&packed, vec![n, k], DType::Q4_K_R4).unwrap();

        let x_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let x_all = Tensor::from_f32(&x_f32, vec![m, k]).unwrap();
        let panelled = matmul_nt(&x_all, &w_r4).unwrap();
        let pd = panelled.as_f32_slice();

        for i in 0..m {
            let x_row = Tensor::from_f32(&x_f32[i * k..(i + 1) * k], vec![1, k]).unwrap();
            let row_out = matmul_nt(&x_row, &w_r4).unwrap();
            let rd = row_out.as_f32_slice();
            for j in 0..n {
                assert_eq!(
                    pd[i * n + j].to_bits(),
                    rd[j].to_bits(),
                    "row {i} col {j}: {} vs {}",
                    pd[i * n + j],
                    rd[j]
                );
            }
        }
    }
}
