//! Quantized weight storage and on-the-fly dequantizing dot-products.
//!
//! The whole point of running huge models on small devices is to **keep weights
//! quantized in memory** and dequantize one block at a time *inside* the
//! dot-product, instead of expanding the whole weight matrix to F32 (which costs
//! 8× the RAM for Q4). This module is the Phase-0 spike proving that mechanic:
//! a Q4_0 matrix-vector product computed straight from the packed blocks matches
//! the F32 reference, while storing only 0.5625 bytes/weight.
//!
//! Block layouts follow the canonical **ggml** conventions (so the same code
//! reads real GGUF files in Phase 1):
//! - `Q4_0`: 32 weights per block, 18 bytes = f16 scale + 16 packed nibble bytes.
//!   Byte `j` holds element `j` (low nibble) and element `j+16` (high nibble).
//! - `Q8_0`: 32 weights per block, 34 bytes = f16 scale + 32 × i8.

use half::f16;

// ── SIMD helpers ─────────────────────────────────────────────────────────────

// aarch64: NEON intrinsics (always available on Apple Silicon and ARM64 Linux)
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// x86_64: AVX2 runtime check wrapper for the hot row-dot path
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Weights per quantized block (both Q4_0 and Q8_0 use 32).
pub const QK: usize = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (packed nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 quants).
pub const Q8_0_BLOCK_BYTES: usize = 34;

// ── Q4_0 ──────────────────────────────────────────────────────────────────────

/// Quantize a length-`QK` slice of f32 into one Q4_0 block (ggml convention).
pub fn quantize_q4_0_block(x: &[f32]) -> [u8; Q4_0_BLOCK_BYTES] {
    debug_assert_eq!(x.len(), QK);
    // Scale from the value with the largest magnitude, preserving its sign
    // (this is how ggml derives `d`, which is why d can be negative).
    let mut amax = 0.0f32;
    let mut vmax = 0.0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            vmax = v;
        }
    }
    let d = vmax / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut out = [0u8; Q4_0_BLOCK_BYTES];
    out[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    for j in 0..QK / 2 {
        let q0 = nibble(x[j] * id);
        let q1 = nibble(x[j + QK / 2] * id);
        out[2 + j] = q0 | (q1 << 4);
    }
    out
}

#[inline]
fn nibble(scaled: f32) -> u8 {
    // ggml: MIN(15, (int)(x*id + 8.5)). Clamp into [0, 15].
    let q = (scaled + 8.5) as i32;
    q.clamp(0, 15) as u8
}

/// Dequantize one Q4_0 block into `out` (length `QK`).
pub fn dequantize_q4_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4_0_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..QK / 2 {
        let byte = block[2 + j];
        let lo = (byte & 0x0f) as i32 - 8;
        let hi = (byte >> 4) as i32 - 8;
        out[j] = lo as f32 * d;
        out[j + QK / 2] = hi as f32 * d;
    }
}

/// Dot product of one Q4_0 block with a length-`QK` f32 activation slice.
///
/// On aarch64 the NEON path vectorises the nibble unpacking and FMA in ~8
/// NEON instructions per block (4× width vs scalar). Falls back to scalar
/// on every other target.
#[inline]
pub fn dot_q4_0_block_f32(block: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(block.len(), Q4_0_BLOCK_BYTES);
    debug_assert_eq!(x.len(), QK);
    #[cfg(target_arch = "aarch64")]
    return unsafe { dot_q4_0_block_neon(block, x) };
    #[cfg(not(target_arch = "aarch64"))]
    dot_q4_0_block_scalar(block, x)
}

#[inline(always)]
#[allow(dead_code)] // used on non-aarch64 targets only
fn dot_q4_0_block_scalar(block: &[u8], x: &[f32]) -> f32 {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let mut acc = 0.0f32;
    for j in 0..QK / 2 {
        let byte = block[2 + j];
        let lo = (byte & 0x0f) as i32 - 8;
        let hi = (byte >> 4) as i32 - 8;
        acc += lo as f32 * x[j] + hi as f32 * x[j + QK / 2];
    }
    acc * d
}

/// NEON Q4_0 block dot product.
///
/// Processes all 16 packed nibble bytes as two 16-element NEON vectors.
/// Lo nibbles → first 16 activations; hi nibbles → second 16 activations.
/// Subtract 8, widen u8→i8→i16→f32, then FMA with activations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q4_0_block_neon(block: &[u8], x: &[f32]) -> f32 {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let packed_ptr = block.as_ptr().add(2); // 16 bytes of packed nibbles

    // Load 16 packed bytes (= 32 nibbles)
    let packed = vld1q_u8(packed_ptr);

    // Extract lo nibbles (elements 0..15) and hi nibbles (elements 16..31)
    let lo_u8 = vandq_u8(packed, vdupq_n_u8(0x0F));
    let hi_u8 = vshrq_n_u8(packed, 4);

    // Subtract 8 (u8 wrap-sub, then reinterpret as signed i8 giving [-8, 7])
    let eight = vdupq_n_u8(8);
    let lo_i8 = vreinterpretq_s8_u8(vsubq_u8(lo_u8, eight));
    let hi_i8 = vreinterpretq_s8_u8(vsubq_u8(hi_u8, eight));

    // Widen i8 → i16 → i32 → f32 in two halves each, then FMA with activations.
    // Each vmovl call produces 8 i16 values; vmovl_high_s8 / vget_low_s8 split
    // the 16-element vector into its low and high 8-element halves.
    macro_rules! to_f32x4 {
        ($i8vec:expr, $half:ident) => {{
            let i16v = $half($i8vec);
            let lo32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16v)));
            let hi32 = vcvtq_f32_s32(vmovl_high_s16(i16v));
            (lo32, hi32)
        }};
    }

    let (lo_f32_0, lo_f32_1) = to_f32x4!(lo_i8, vmovl_s8_low);
    let (lo_f32_2, lo_f32_3) = to_f32x4!(lo_i8, vmovl_s8_high);
    let (hi_f32_0, hi_f32_1) = to_f32x4!(hi_i8, vmovl_s8_low);
    let (hi_f32_2, hi_f32_3) = to_f32x4!(hi_i8, vmovl_s8_high);

    let xp = x.as_ptr();
    let x0 = vld1q_f32(xp);
    let x1 = vld1q_f32(xp.add(4));
    let x2 = vld1q_f32(xp.add(8));
    let x3 = vld1q_f32(xp.add(12));
    let x4 = vld1q_f32(xp.add(16));
    let x5 = vld1q_f32(xp.add(20));
    let x6 = vld1q_f32(xp.add(24));
    let x7 = vld1q_f32(xp.add(28));

    let mut acc = vmulq_f32(lo_f32_0, x0);
    acc = vfmaq_f32(acc, lo_f32_1, x1);
    acc = vfmaq_f32(acc, lo_f32_2, x2);
    acc = vfmaq_f32(acc, lo_f32_3, x3);
    acc = vfmaq_f32(acc, hi_f32_0, x4);
    acc = vfmaq_f32(acc, hi_f32_1, x5);
    acc = vfmaq_f32(acc, hi_f32_2, x6);
    acc = vfmaq_f32(acc, hi_f32_3, x7);

    vaddvq_f32(acc) * scale
}

// Helper: widen the low half of an i8x16 vector to i16x8
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vmovl_s8_low(v: int8x16_t) -> int16x8_t {
    vmovl_s8(vget_low_s8(v))
}

// Helper: widen the high half of an i8x16 vector to i16x8
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vmovl_s8_high(v: int8x16_t) -> int16x8_t {
    vmovl_high_s8(v)
}

/// Dot product of a full Q4_0-quantized weight row.
///
/// Dispatches to the SIMD block kernel (NEON on aarch64, scalar + AVX2-auto
/// on x86) block-by-block. The row loop itself is intentionally kept simple;
/// rayon parallelises across rows at the matmul level.
pub fn dot_q4_0_row_f32(row_blocks: &[u8], x: &[f32]) -> f32 {
    let k = x.len();
    debug_assert_eq!(k % QK, 0);
    let mut acc = 0.0f32;
    for (b, chunk) in row_blocks.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        acc += dot_q4_0_block_f32(chunk, &x[b * QK..b * QK + QK]);
    }
    acc
}

/// Quantize a full f32 weight row (`k % QK == 0`) into packed Q4_0 blocks.
pub fn quantize_q4_0_row(w: &[f32]) -> Vec<u8> {
    debug_assert_eq!(w.len() % QK, 0);
    let mut out = Vec::with_capacity(w.len() / QK * Q4_0_BLOCK_BYTES);
    for chunk in w.chunks_exact(QK) {
        out.extend_from_slice(&quantize_q4_0_block(chunk));
    }
    out
}

// ── Q8_0 ──────────────────────────────────────────────────────────────────────

/// Quantize a length-`QK` (32) slice of f32 into one Q8_0 block (ggml convention).
///
/// Returns the 34-byte block: 2-byte f16 scale followed by 32 i8 quantized values.
/// Used for online quantization of F16/BF16 weight matrices at load time.
pub fn quantize_q8_0_block(x: &[f32]) -> [u8; Q8_0_BLOCK_BYTES] {
    debug_assert_eq!(x.len(), QK);
    let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = max_abs / 127.0;
    let d = half::f16::from_f32(scale);
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let mut out = [0u8; Q8_0_BLOCK_BYTES];
    out[0..2].copy_from_slice(&d.to_le_bytes());
    for (i, &v) in x.iter().enumerate() {
        out[2 + i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8 as u8;
    }
    out
}

/// Dot product of a Q8_0 block with an f32 activation slice.
///
/// On aarch64 widens i8→i16→f32 with NEON vfmaq, processing 8 elements per
/// instruction. On x86 the scalar loop auto-vectorises to SSE/AVX.
#[inline]
pub fn dot_q8_0_block_f32(block: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(block.len(), Q8_0_BLOCK_BYTES);
    debug_assert_eq!(x.len(), QK);
    #[cfg(target_arch = "aarch64")]
    return unsafe { dot_q8_0_block_neon(block, x) };
    #[cfg(not(target_arch = "aarch64"))]
    dot_q8_0_block_scalar(block, x)
}

#[inline(always)]
#[allow(dead_code)] // used on non-aarch64 targets only
fn dot_q8_0_block_scalar(block: &[u8], x: &[f32]) -> f32 {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let mut acc = 0.0f32;
    for j in 0..QK {
        acc += block[2 + j] as i8 as f32 * x[j];
    }
    acc * d
}

/// NEON Q8_0 block dot product: widen four groups of 8 i8 values to f32,
/// then fused-multiply-accumulate with f32 activations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q8_0_block_neon(block: &[u8], x: &[f32]) -> f32 {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let q_ptr = block.as_ptr().add(2) as *const i8;
    let xp = x.as_ptr();
    let mut acc = vdupq_n_f32(0.0);

    macro_rules! fma_group {
        ($qoff:expr, $xoff:expr) => {{
            let q8 = vld1_s8(q_ptr.add($qoff));
            let q16 = vmovl_s8(q8);
            let qlo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            let qhi = vcvtq_f32_s32(vmovl_high_s16(q16));
            acc = vfmaq_f32(acc, qlo, vld1q_f32(xp.add($xoff)));
            acc = vfmaq_f32(acc, qhi, vld1q_f32(xp.add($xoff + 4)));
        }};
    }

    fma_group!(0, 0);
    fma_group!(8, 8);
    fma_group!(16, 16);
    fma_group!(24, 24);

    vaddvq_f32(acc) * scale
}

// ── SDOT (ARMv8.4-A dotprod) hot path ─────────────────────────────────────────
//
// Strategy: quantize the f32 activation vector to i8 ONCE per GEMV row, then
// use `vdotq_s32` (4 × 4-element integer dot products per cycle) for every
// weight block.  Compared to the widening path (i8→i16→i32→f32 per element),
// this needs ~6 NEON ops per Q8_0 block vs ~40, delivering a 4–5× compute
// uplift on Apple Silicon M-series and DGX Spark (Grace ARM64 CPU).
//
// Accuracy: quantizing x to i8 (same resolution as Q8_0 weights) introduces
// ≈ max|x| / (127 × √K) RMS error — indistinguishable from Q8_0 weight noise.

/// Quantize a row of f32 activations to i8 with **per-block** scales — one scale
/// per `QK` (32) element block, matching the Q8_0 weight block layout.
///
/// Returns `(quantized_i8, per_block_scales)` where `scales[b] = max_abs(block b)
/// / 127`. The caller multiplies each weight block's scale by the matching
/// activation block scale to recover f32.
///
/// ## Why per-block, not per-row
/// LLM activations contain a handful of *outlier channels* whose magnitude is
/// 10–100× the rest (a well-documented property of transformer residual streams).
/// A single per-row int8 scale is set by that outlier, so every normal-magnitude
/// value rounds to 0 or ±1 — destroying the signal and producing incoherent
/// "token-salad" output. Scoping the scale to a 32-element block confines the
/// outlier's damage to its own block, which is exactly how llama.cpp quantizes
/// activations for its Q8_0 × Q8_0 kernels. `x.len()` must be a multiple of `QK`.
pub fn quantize_row_to_i8_blocks(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    debug_assert_eq!(x.len() % QK, 0);
    let nblocks = x.len() / QK;
    let mut q = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; nblocks];
    for (b, blk) in x.chunks_exact(QK).enumerate() {
        let max_abs = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (i, &v) in blk.iter().enumerate() {
            q[b * QK + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        scales[b] = scale;
    }
    (q, scales)
}

/// Q8_0 block dot product against a pre-quantized i8 activation slice.
///
/// Uses the ARMv8.4-A `sdot` instruction via inline assembly — stable Rust,
/// no unstable features required. `vdotq_s32` is still behind an unstable
/// feature gate, but inline asm lets us emit the same bytes directly.
///
/// `sdot v0.4s, v1.16b, v2.16b` computes four 4-element i8 dot products
/// into an i32x4 accumulator in one instruction (16 MAC ops per cycle).
/// Two calls cover all 32 elements of a Q8_0 block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q8_0_block_sdot(block: &[u8], x_i8: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(block.len(), Q8_0_BLOCK_BYTES);
    debug_assert_eq!(x_i8.len(), QK);

    let w_ptr = block.as_ptr().add(2) as *const i8;
    let x_ptr = x_i8.as_ptr();

    let w0 = vld1q_s8(w_ptr);
    let x0 = vld1q_s8(x_ptr);
    let w1 = vld1q_s8(w_ptr.add(16));
    let x1 = vld1q_s8(x_ptr.add(16));

    let mut acc = vdupq_n_s32(0i32);
    // sdot v_acc.4s, v_w.16b, v_x.16b — ARM SDOT instruction via inline asm.
    // The :v modifier formats the register as the 128-bit v-register view.
    core::arch::asm!(
        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) acc,
        in(vreg) w0,
        in(vreg) x0,
        options(nomem, nostack),
    );
    core::arch::asm!(
        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) acc,
        in(vreg) w1,
        in(vreg) x1,
        options(nomem, nostack),
    );
    vaddvq_s32(acc)
}

/// Full Q8_0 row dot product with pre-quantized i8 activations.
/// Called by matmul_nt_q8_0 after [`quantize_row_to_i8_blocks`] when dotprod is
/// available. `x_scales` holds one scale per `QK`-element activation block.
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")` before calling.
/// `row_blocks` must be a valid slice of packed Q8_0 blocks; `x_i8` must have
/// length equal to the number of elements covered by those blocks, and
/// `x_scales.len()` must equal the block count.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q8_0_row_sdot(row_blocks: &[u8], x_i8: &[i8], x_scales: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for (bi, block) in row_blocks.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let w_scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dot = dot_q8_0_block_sdot(block, &x_i8[x_off..x_off + QK]);
        acc += w_scale * x_scales[bi] * dot as f32;
        x_off += QK;
    }
    acc
}

/// Scalar fallback for dot_q8_0_row_sdot (non-dotprod aarch64 or other platforms).
/// Uses i32 integer arithmetic — no widening chain, still faster than the f32 path
/// for targets without AVX2, and correct everywhere. `x_scales` holds one scale
/// per `QK`-element activation block (see [`quantize_row_to_i8_blocks`]).
pub fn dot_q8_0_row_i8_scalar(row_blocks: &[u8], x_i8: &[i8], x_scales: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for (bi, block) in row_blocks.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let w_scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let w = &block[2..];
        let dot: i32 = w[..QK]
            .iter()
            .zip(&x_i8[x_off..x_off + QK])
            .map(|(&wi, &xi)| wi as i8 as i32 * xi as i32)
            .sum();
        acc += w_scale * x_scales[bi] * dot as f32;
        x_off += QK;
    }
    acc
}

/// Dot product of a full Q8_0-quantized weight row with an f32 activation vector.
///
/// On x86_64 with AVX2 at runtime, uses the wider FMA path that processes
/// 8 floats per cycle; otherwise falls back to the NEON or scalar block kernel.
pub fn dot_q8_0_row_f32(row_blocks: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { dot_q8_0_row_avx2(row_blocks, x) };
    }
    let k = x.len();
    debug_assert_eq!(k % QK, 0);
    let mut acc = 0.0f32;
    for (b, chunk) in row_blocks.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        acc += dot_q8_0_block_f32(chunk, &x[b * QK..b * QK + QK]);
    }
    acc
}

/// AVX2+FMA path for Q8_0 row dot product on x86_64.
///
/// Processes 8 f32 values per `_mm256_fmadd_ps` instruction.  The i8→f32
/// widening uses `_mm256_cvtepi8_epi32` to convert 8 i8 at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_row_avx2(row_blocks: &[u8], x: &[f32]) -> f32 {
    let k = x.len();
    debug_assert_eq!(k % QK, 0);
    let mut row_acc = _mm256_setzero_ps();

    for (b, block) in row_blocks.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let q_ptr = block.as_ptr().add(2) as *const i32; // load 4 bytes at a time
        let xp = x.as_ptr().add(b * QK);
        let mut block_acc = _mm256_setzero_ps();

        // 4 groups of 8 values each (8 i8 → 8 i32 → 8 f32)
        for g in 0..4usize {
            let q_i32_4 = _mm_loadu_si32(q_ptr.add(2 * g) as *const _); // 4 bytes
            let q_i32_4b = _mm_loadu_si32(q_ptr.add(2 * g + 1) as *const _);
            let q_a = _mm256_cvtepi8_epi32(q_i32_4); // 4 i8 → 4 i32 (low lane)
            let q_b = _mm256_cvtepi8_epi32(q_i32_4b); // next 4
            let xv_a = _mm256_loadu_ps(xp.add(g * 8));
            let xv_b = _mm256_loadu_ps(xp.add(g * 8 + 4));
            let qf_a = _mm256_cvtepi32_ps(q_a);
            let qf_b = _mm256_cvtepi32_ps(q_b);
            block_acc = _mm256_fmadd_ps(qf_a, xv_a, block_acc);
            block_acc = _mm256_fmadd_ps(qf_b, xv_b, block_acc);
        }
        // Horizontal sum of block_acc, multiply by scale, add to row accumulator
        let scale_v = _mm256_set1_ps(scale);
        row_acc = _mm256_fmadd_ps(block_acc, scale_v, row_acc);
    }

    // Horizontal sum of the 8-lane AVX2 accumulator
    let lo = _mm256_castps256_ps128(row_acc);
    let hi = _mm256_extractf128_ps(row_acc, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let sum1 = _mm_add_ss(sum2, _mm_movehl_ps(shuf, sum2));
    _mm_cvtss_f32(sum1)
}

// ── K-quants (Q4_K, Q5_K, Q6_K) ─────────────────────────────────────────────
//
// K-quant blocks use QK_K = 256 elements per block with super-block scaling.
// Weights are kept as packed blocks and dequantized one block at a time inside
// the dot product — no F32 expansion at load time.

pub const QK_K: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;
pub const Q5_K_BLOCK_BYTES: usize = 176;
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Extract 6-bit scale and min for K-quant sub-block `j` (0..7) from the
/// 12-byte `scales` field of a Q4_K or Q5_K block.
#[inline(always)]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dot product of a full Q4_K-quantized weight row with an f32 activation vector.
///
/// Q4_K block layout (144 bytes, 256 weights):
///   [0..1] d (f16) — super-block scale
///   [2..3] dmin (f16) — super-block min scale
///   [4..15] scales (12 bytes) — 8 pairs of 6-bit (scale, min) packed
///   [16..143] qs (128 bytes) — 256 × 4-bit quantized values (lo/hi nibble)
///
/// Dispatches to the NEON path on aarch64, scalar otherwise.
pub fn dot_q4_k_row_f32(row_data: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return unsafe { dot_q4_k_row_f32_neon(row_data, x) };
    #[cfg(not(target_arch = "aarch64"))]
    dot_q4_k_row_f32_scalar(row_data, x)
}

/// Scalar fallback for Q4_K row dot product.
#[allow(dead_code)]
fn dot_q4_k_row_f32_scalar(row_data: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q4_K_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..Q4_K_BLOCK_BYTES];
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;
            for l in 0..32 {
                acc += (d1 * (qs[q_off + l] & 0x0F) as f32 - m1v) * x[x_off + l];
                acc += (d2 * (qs[q_off + l] >> 4) as f32 - m2v) * x[x_off + l + 32];
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

/// NEON Q4_K row dot product (aarch64).
///
/// Processes 8 bytes (16 nibbles) per NEON iteration with FMA for both the
/// lo-nibble (sub-block 1) and hi-nibble (sub-block 2) contributions.
/// Also accumulates the x sums required for the min correction term.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q4_k_row_f32_neon(row_data: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    let mask4 = vdup_n_u8(0x0F);

    for block in row_data.chunks_exact(Q4_K_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..Q4_K_BLOCK_BYTES];

        let mut q_off = 0usize;
        let mut is = 0usize;

        // QK_K / 64 = 4 iterations, each handling 64 weights (32 lo-nibble + 32 hi-nibble)
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;

            // x_lo = activations for lo-nibble sub-block (indices x_off..x_off+32)
            // x_hi = activations for hi-nibble sub-block (indices x_off+32..x_off+64)
            let x_lo = &x[x_off..x_off + 32];
            let x_hi = &x[x_off + 32..x_off + 64];

            // NEON: process 8 bytes of qs at a time -> 8 lo-nibbles, 8 hi-nibbles
            // 4 rounds x 8 bytes = 32 bytes total (covers the 32 lo and 32 hi elements)
            let mut vsum_lo = vdupq_n_f32(0.0); // dot(lo_nibbles, x_lo)
            let mut vsum_hi = vdupq_n_f32(0.0); // dot(hi_nibbles, x_hi)
            let mut vsum_xl = vdupq_n_f32(0.0); // sum(x_lo) for min correction
            let mut vsum_xh = vdupq_n_f32(0.0); // sum(x_hi) for min correction

            for chunk in 0..4usize {
                // Load 8 packed bytes -> 8 lo nibbles + 8 hi nibbles
                let q8 = vld1_u8(qs.as_ptr().add(q_off + chunk * 8));
                let lo8 = vand_u8(q8, mask4);
                let hi8 = vshr_n_u8::<4>(q8);

                // Widen u8x8 -> u16x8 -> two u32x4 -> two f32x4
                let lo16 = vmovl_u8(lo8);
                let lof0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
                let lof1 = vcvtq_f32_u32(vmovl_high_u16(lo16));

                let hi16 = vmovl_u8(hi8);
                let hif0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
                let hif1 = vcvtq_f32_u32(vmovl_high_u16(hi16));

                // Load 8 activation values for each sub-block
                let xl0 = vld1q_f32(x_lo.as_ptr().add(chunk * 8));
                let xl1 = vld1q_f32(x_lo.as_ptr().add(chunk * 8 + 4));
                let xh0 = vld1q_f32(x_hi.as_ptr().add(chunk * 8));
                let xh1 = vld1q_f32(x_hi.as_ptr().add(chunk * 8 + 4));

                vsum_lo = vfmaq_f32(vsum_lo, lof0, xl0);
                vsum_lo = vfmaq_f32(vsum_lo, lof1, xl1);
                vsum_hi = vfmaq_f32(vsum_hi, hif0, xh0);
                vsum_hi = vfmaq_f32(vsum_hi, hif1, xh1);
                vsum_xl = vaddq_f32(vsum_xl, vaddq_f32(xl0, xl1));
                vsum_xh = vaddq_f32(vsum_xh, vaddq_f32(xh0, xh1));
            }

            // acc += d1 * sum(lo * x_lo) - m1v * sum(x_lo)
            // acc += d2 * sum(hi * x_hi) - m2v * sum(x_hi)
            acc += d1 * vaddvq_f32(vsum_lo) - m1v * vaddvq_f32(vsum_xl);
            acc += d2 * vaddvq_f32(vsum_hi) - m2v * vaddvq_f32(vsum_xh);

            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

/// W4A8 Q4_K row dot product: the weights stay 4-bit and the **activation** is
/// pre-quantized to per-32-element int8 blocks (`quantize_row_to_i8_blocks`).
/// Each 32-element Q4_K sub-block is an integer dot of (4-bit nibbles × int8
/// activations) plus a min-correction term, then scaled — this is the W4A8 form
/// llama.cpp uses (`ggml_vec_dot_q4_K_q8_K`), which lets the hot loop use integer
/// MACs (SDOT) instead of converting every nibble to f32. Scalar reference; the
/// NEON SDOT variant must match this bit-for-bit in the integer dot.
///
/// `x_i8`/`x_scales` come from `quantize_row_to_i8_blocks` over the same row that
/// the f32 path would use; `x_scales` has one scale per 32-element block. The math
/// mirrors `dot_q4_k_row_f32`: per sub-block, with d_sub=d·sc, m_sub=dmin·m and
/// x≈x_scale·x_i8,  Σ(d_sub·n−m_sub)·x  =  x_scale·(d_sub·Σ(n·x_i8) − m_sub·Σx_i8).
pub fn dot_q4_k_row_q8_scalar(row_data: &[u8], x_i8: &[i8], x_scales: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q4_K_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..Q4_K_BLOCK_BYTES];
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;

            // lo nibbles → activation block at x_off; hi nibbles → block at x_off+32.
            let blk_lo = x_off / QK;
            let blk_hi = (x_off + 32) / QK;
            let xlo = &x_i8[x_off..x_off + 32];
            let xhi = &x_i8[x_off + 32..x_off + 64];

            let (mut dot_lo, mut dot_hi, mut sum_lo, mut sum_hi) = (0i32, 0i32, 0i32, 0i32);
            for l in 0..32 {
                let nlo = (qs[q_off + l] & 0x0F) as i32;
                let nhi = (qs[q_off + l] >> 4) as i32;
                dot_lo += nlo * xlo[l] as i32;
                dot_hi += nhi * xhi[l] as i32;
                sum_lo += xlo[l] as i32;
                sum_hi += xhi[l] as i32;
            }
            acc += x_scales[blk_lo] * (d1 * dot_lo as f32 - m1v * sum_lo as f32);
            acc += x_scales[blk_hi] * (d2 * dot_hi as f32 - m2v * sum_hi as f32);

            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

/// Accumulate `w·x` (four 4-element i8 dot products) into an i32x4 via the
/// ARMv8.4-A `sdot` instruction (inline asm — stable Rust, same bytes as the
/// unstable `vdotq_s32`). 16 int8 MACs per instruction.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn sdot_s32(
    acc: std::arch::aarch64::int32x4_t,
    w: std::arch::aarch64::int8x16_t,
    x: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut a = acc;
    core::arch::asm!(
        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) a,
        in(vreg) w,
        in(vreg) x,
        options(nomem, nostack),
    );
    a
}

/// NEON W4A8 Q4_K row dot product — the fast decode kernel. Bit-for-bit equal in
/// the integer dot to [`dot_q4_k_row_q8_scalar`] (regression-tested), but uses
/// `sdot` (16 int8 MACs/instr) instead of converting nibbles to f32 + `vfmaq`.
/// This is the ARM equivalent of llama.cpp's `ggml_vec_dot_q4_K_q8_K`.
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`. `x_i8` must cover
/// the row's elements in 32-element blocks; `x_scales` has one scale per block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_row_q8_neon(row_data: &[u8], x_i8: &[i8], x_scales: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q4_K_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..Q4_K_BLOCK_BYTES];
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;

            // 32 packed bytes → 32 lo nibbles (sub-block 2g) + 32 hi nibbles (2g+1).
            let q0 = vld1q_u8(qs.as_ptr().add(q_off));
            let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
            let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));

            let zero = vdupq_n_s32(0);
            let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
            let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));
            // Σ x_i8 per 32-sub-block for the min-correction term.
            let sum_lo = vaddlvq_s8(xlo0) as i32 + vaddlvq_s8(xlo1) as i32;
            let sum_hi = vaddlvq_s8(xhi0) as i32 + vaddlvq_s8(xhi1) as i32;

            let blk_lo = x_off / QK;
            let blk_hi = (x_off + 32) / QK;
            acc += x_scales[blk_lo] * (d1 * dot_lo as f32 - m1v * sum_lo as f32);
            acc += x_scales[blk_hi] * (d2 * dot_hi as f32 - m2v * sum_hi as f32);

            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

/// Dot product of a full Q5_K-quantized weight row with an f32 activation vector.
///
/// Q5_K block layout (176 bytes, 256 weights):
///   [0..1] d (f16), [2..3] dmin (f16), [4..15] scales (12B),
///   [16..47] qh (32B — high bits, one per 32-weight sub-block),
///   [48..175] ql (128B — low 4-bit nibbles)
pub fn dot_q5_k_row_f32(row_data: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return unsafe { dot_q5_k_row_f32_neon(row_data, x) };
    #[cfg(not(target_arch = "aarch64"))]
    dot_q5_k_row_f32_scalar(row_data, x)
}

/// Scalar reference for the Q5_K row dot (oracle for the NEON kernel).
#[allow(dead_code)]
fn dot_q5_k_row_f32_scalar(row_data: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q5_K_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..Q5_K_BLOCK_BYTES];
        let mut ql_off = 0usize;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;
            // The 5th bit is PER-ELEMENT: ggml reads qh[l] (l = 0..32, one of the
            // 32 qh bytes) and selects the active bit-plane with u1/u2 (which shift
            // by 2 each sub-block pair). The previous code used a single qh[is/8]
            // byte for the whole 32-element sub-block — that collapses 32 distinct
            // high bits to one, corrupting Q5_K (Q5_K_M models would hallucinate).
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16.0f32 } else { 0.0 };
                let hi2 = if qh[l] & u2 != 0 { 16.0f32 } else { 0.0 };
                acc += (d1 * ((ql[ql_off + l] & 0x0F) as f32 + hi1) - m1v) * x[x_off + l];
                acc += (d2 * ((ql[ql_off + l] >> 4) as f32 + hi2) - m2v) * x[x_off + l + 32];
            }
            x_off += 64;
            ql_off += 32;
            is += 2;
            if is % 8 == 0 {
                u1 = 1;
                u2 = 2;
            } else {
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
    acc
}

/// NEON Q5_K row dot (aarch64) — vectorises the (now-fixed) scalar reference 16
/// lanes at a time. Q5_K = Q4_K plus a per-element 5th bit from `qh[l]` selected
/// by the bit-plane `u1`/`u2`. Regression-tested bit-close to
/// [`dot_q5_k_row_f32_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q5_k_row_f32_neon(row_data: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mask0f = vdupq_n_u8(0x0F);
    let sixteen = vdupq_n_u8(16);
    let mut acc = vdupq_n_f32(0.0);
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q5_K_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qh = block.as_ptr().add(16); // 32 bytes
        let ql = block.as_ptr().add(48); // 128 bytes
        let mut ql_off = 0usize;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;
            let u1v = vdupq_n_u8(u1);
            let u2v = vdupq_n_u8(u2);

            // Two 16-lane halves cover the 32-element sub-blocks.
            for half in [0usize, 16usize] {
                let qlv = vld1q_u8(ql.add(ql_off + half));
                let qhv = vld1q_u8(qh.add(half));
                // 5th bit → 16 or 0 (per element): (qh & u) ? 16 : 0.
                let hi1 = vandq_u8(vtstq_u8(qhv, u1v), sixteen);
                let hi2 = vandq_u8(vtstq_u8(qhv, u2v), sixteen);
                let val_lo = vaddq_u8(vandq_u8(qlv, mask0f), hi1); // 0..31
                let val_hi = vaddq_u8(vshrq_n_u8::<4>(qlv), hi2);

                // acc += (d·val − m)·x over 16 lanes (4× f32x4).
                macro_rules! accum {
                    ($val:expr, $dd:expr, $mm:expr, $xbase:expr) => {{
                        let v16lo = vmovl_u8(vget_low_u8($val));
                        let v16hi = vmovl_high_u8($val);
                        let vf = [
                            vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16lo))),
                            vcvtq_f32_u32(vmovl_high_u16(v16lo)),
                            vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16hi))),
                            vcvtq_f32_u32(vmovl_high_u16(v16hi)),
                        ];
                        let xb = x.as_ptr().add($xbase);
                        let mneg = vdupq_n_f32($mm);
                        for c in 0..4 {
                            // d·val − m
                            let t = vsubq_f32(vmulq_n_f32(vf[c], $dd), mneg);
                            let xc = vld1q_f32(xb.add(c * 4));
                            acc = vfmaq_f32(acc, t, xc);
                        }
                    }};
                }
                accum!(val_lo, d1, m1v, x_off + half);
                accum!(val_hi, d2, m2v, x_off + 32 + half);
            }

            x_off += 64;
            ql_off += 32;
            is += 2;
            if is % 8 == 0 {
                u1 = 1;
                u2 = 2;
            } else {
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
    vaddvq_f32(acc)
}

/// Dot product of a full Q6_K-quantized weight row with an f32 activation vector.
///
/// Q6_K block layout (210 bytes, 256 weights):
///   [0..127] ql (128B — low 4-bit nibbles)
///   [128..191] qh (64B — upper 2 bits, two per byte)
///   [192..207] scales (16B — i8 per 16-element group)
///   [208..209] d (f16)
pub fn dot_q6_k_row_f32(row_data: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return unsafe { dot_q6_k_row_f32_neon(row_data, x) };
    #[cfg(not(target_arch = "aarch64"))]
    dot_q6_k_row_f32_scalar(row_data, x)
}

/// Scalar reference for the Q6_K row dot (oracle for the NEON kernel).
#[allow(dead_code)]
fn dot_q6_k_row_f32_scalar(row_data: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q6_K_BLOCK_BYTES) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        // Q6_K has 16 i8 scales per super-block (one per 16-element group). Within
        // each 128-element half the 4 sub-groups use scales at offsets +0/+2/+4/+6
        // and the scale splits again at l==16 (`is = l/16`); the base advances by 8
        // per 128-block. (Matches ggml dequantize_row_q6_K — using one scale per
        // 32-group, as the old code did, decodes Q6_K weights incorrectly.)
        let mut sc_base = 0usize;
        for _ in 0..(QK_K / 128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    (((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32) as f32;
                let q2 = (((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4))
                    as i32
                    - 32) as f32;
                let q3 = (((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32)
                    as f32;
                let q4 = (((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32
                    - 32) as f32;
                acc += d * sc[sc_base + is] as i8 as f32 * q1 * x[x_off + l];
                acc += d * sc[sc_base + is + 2] as i8 as f32 * q2 * x[x_off + l + 32];
                acc += d * sc[sc_base + is + 4] as i8 as f32 * q3 * x[x_off + l + 64];
                acc += d * sc[sc_base + is + 6] as i8 as f32 * q4 * x[x_off + l + 96];
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    acc
}

/// NEON Q6_K row dot (aarch64) — vectorises the scalar reference 16 lanes at a
/// time. Q6_K is ~⅓ of a Q4_K_M model (lm_head, half of ffn_down, attn_v) and the
/// scalar path made it the dominant Pi decode cost. Computes the identical f32
/// math (only the reduction order differs); regression-tested against
/// [`dot_q6_k_row_f32_scalar`]. The four 6-bit sub-positions per 128-block and the
/// per-16 scale layout (`sc_base + is + {0,2,4,6}`) match the scalar exactly.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q6_k_row_f32_neon(row_data: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mask0f = vdupq_n_u8(0x0F);
    let mask3 = vdupq_n_u8(0x03);
    let m32 = vdupq_n_f32(32.0);
    let mut acc = vdupq_n_f32(0.0);
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q6_K_BLOCK_BYTES) {
        let ql = block.as_ptr();
        let qh = block.as_ptr().add(128);
        let sc = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_base = 0usize;
        for _ in 0..(QK_K / 128) {
            // Two 16-lane scale groups within this 128-block: L=0 (is=0), L=16 (is=1).
            for &l0 in &[0usize, 16usize] {
                let is = l0 / 16;
                let ql_lo = vld1q_u8(ql.add(ql_off + l0));
                let ql_hi = vld1q_u8(ql.add(ql_off + l0 + 32));
                let qhv = vld1q_u8(qh.add(qh_off + l0));

                // Reconstruct the four 6-bit sub-positions (each 16× u8 in [0,63]).
                let q1 = vorrq_u8(
                    vandq_u8(ql_lo, mask0f),
                    vshlq_n_u8::<4>(vandq_u8(qhv, mask3)),
                );
                let q2 = vorrq_u8(
                    vandq_u8(ql_hi, mask0f),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qhv), mask3)),
                );
                let q3 = vorrq_u8(
                    vshrq_n_u8::<4>(ql_lo),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qhv), mask3)),
                );
                let q4 = vorrq_u8(
                    vshrq_n_u8::<4>(ql_hi),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<6>(qhv), mask3)),
                );

                let s1 = d * (sc[sc_base + is] as i8 as f32);
                let s2 = d * (sc[sc_base + is + 2] as i8 as f32);
                let s3 = d * (sc[sc_base + is + 4] as i8 as f32);
                let s4 = d * (sc[sc_base + is + 6] as i8 as f32);

                // acc += scale · Σ_lane (q − 32) · x, over the 16 lanes (4× f32x4).
                macro_rules! accum {
                    ($q:expr, $scale:expr, $xbase:expr) => {{
                        let q = $q;
                        let q16lo = vmovl_u8(vget_low_u8(q));
                        let q16hi = vmovl_high_u8(q);
                        let qf = [
                            vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16lo))),
                            vcvtq_f32_u32(vmovl_high_u16(q16lo)),
                            vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16hi))),
                            vcvtq_f32_u32(vmovl_high_u16(q16hi)),
                        ];
                        let xb = x.as_ptr().add($xbase);
                        let sv = vdupq_n_f32($scale);
                        for c in 0..4 {
                            let qm = vsubq_f32(qf[c], m32);
                            let xc = vld1q_f32(xb.add(c * 4));
                            acc = vfmaq_f32(acc, vmulq_f32(qm, sv), xc);
                        }
                    }};
                }
                accum!(q1, s1, x_off + l0);
                accum!(q2, s2, x_off + 32 + l0);
                accum!(q3, s3, x_off + 64 + l0);
                accum!(q4, s4, x_off + 96 + l0);
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    vaddvq_f32(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Q6_K must map weight i to scale i/16 (16 scales per 256-weight super-block),
    // matching ggml dequantize_row_q6_K. Build a block where every 6-bit quant
    // decodes to +1 (raw 33 = low-nibble 1 | (hi-bits 2 << 4)) and scales = 0..16,
    // so output[i] = d·scale[i/16]·1. With x = 1 and d = 1 the dot is
    // Σ_i scale[i/16] = 16·(0+1+…+15) = 1920. The old (wrong) per-32-group indexing
    // only used scales 0..8 and would give 896.
    #[test]
    fn q6_k_scale_indexing_matches_ggml() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        for b in block.iter_mut().take(128) {
            *b = 0x11; // every low nibble = 1
        }
        for b in block.iter_mut().take(192).skip(128) {
            *b = 0xAA; // every 2-bit hi field = 0b10 = 2
        }
        for j in 0..16 {
            block[192 + j] = j as i8 as u8; // scales 0..15
        }
        block[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

        let x = vec![1.0f32; QK_K];
        let got = dot_q6_k_row_f32(&block, &x);
        assert!(
            (got - 1920.0).abs() < 1e-3,
            "Q6_K scale indexing wrong: got {got}, expected 1920 (old buggy code gives 896)"
        );
    }

    // Deterministic pseudo-random f32 in roughly [-1, 1] (no rand dependency).
    fn seq(n: usize) -> Vec<f32> {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // Helper: quantize an f32 row into packed Q8_0 weight blocks.
    fn q8_0_weight_row(w: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(w.len() / QK * Q8_0_BLOCK_BYTES);
        for chunk in w.chunks_exact(QK) {
            out.extend_from_slice(&quantize_q8_0_block(chunk));
        }
        out
    }

    // The Q8_0 SDOT path quantizes activations to int8. With a per-block scale it
    // must stay close to the exact f32 path even when the activation row contains
    // an outlier channel — a per-row scale (the old behavior) diverges wildly here
    // and produced garbage LLM output. We assert both: blockwise is accurate, and
    // a single per-row scale is demonstrably bad on the same data.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sdot_q8_0_row_blockwise_survives_activation_outlier() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            eprintln!("dotprod not available — skipping SDOT row test");
            return;
        }
        let k = 256;
        let wf = seq(k);
        let mut xf = seq(k);
        xf[100] = 60.0; // outlier channel, ~60× the rest

        let w_blocks = q8_0_weight_row(&wf);
        // Exact reference: Q8_0 weights dequantized, dotted with f32 activations.
        let reference = dot_q8_0_row_f32(&w_blocks, &xf);

        // Fixed path: per-block activation scales.
        let (x_i8, x_scales) = quantize_row_to_i8_blocks(&xf);
        let blockwise = unsafe { dot_q8_0_row_sdot(&w_blocks, &x_i8, &x_scales) };
        let rel = (blockwise - reference).abs() / reference.abs().max(1e-3);
        assert!(
            rel < 0.05,
            "blockwise SDOT rel err {rel} too high (got {blockwise}, ref {reference})"
        );

        // Old path: a single per-row scale set by the outlier collapses the rest.
        let max_abs = xf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let row_scale = max_abs / 127.0;
        let inv = 1.0 / row_scale;
        let x_row_i8: Vec<i8> = xf
            .iter()
            .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let per_row_scales = vec![row_scale; k / QK];
        let perrow = unsafe { dot_q8_0_row_sdot(&w_blocks, &x_row_i8, &per_row_scales) };
        let perrow_rel = (perrow - reference).abs() / reference.abs().max(1e-3);
        assert!(
            perrow_rel > rel * 2.0,
            "per-row scale should be clearly worse than blockwise (per-row rel \
             {perrow_rel}, blockwise rel {rel})"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sdot_q8_0_block_matches_scalar_integer_dot() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            eprintln!("dotprod not available — skipping SDOT test");
            return;
        }
        // Build a Q8_0 block (2-byte f16 scale + 32 i8 weights) and an i8 activation
        // slice from deterministic pseudo-random data, then compare the integer dot
        // produced by the `sdot` inline-asm kernel against a plain scalar reference.
        let wf = seq(QK);
        let xf = seq(QK);
        let w_i8: Vec<i8> = wf
            .iter()
            .map(|v| (v * 100.0).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let x_i8: Vec<i8> = xf
            .iter()
            .map(|v| (v * 100.0).round().clamp(-127.0, 127.0) as i8)
            .collect();

        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        for (i, &w) in w_i8.iter().enumerate() {
            block[2 + i] = w as u8;
        }

        let reference: i32 = w_i8
            .iter()
            .zip(&x_i8)
            .map(|(&w, &x)| w as i32 * x as i32)
            .sum();
        let got = unsafe { dot_q8_0_block_sdot(&block, &x_i8) };
        assert_eq!(
            got, reference,
            "SDOT integer dot must match scalar reference"
        );
    }

    #[test]
    fn q4_0_on_the_fly_dot_matches_dequantized_reference() {
        let k = 256;
        let w = seq(k);
        let x = seq(k).iter().map(|v| v * 0.5).collect::<Vec<_>>();

        let blocks = quantize_q4_0_row(&w);
        // Storage: 18 bytes per 32 weights = 0.5625 B/weight vs 4 B for F32.
        assert_eq!(blocks.len(), k / QK * Q4_0_BLOCK_BYTES);

        // Reference: dequantize fully, then dot.
        let mut w_hat = vec![0.0f32; k];
        for (b, chunk) in blocks.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
            dequantize_q4_0_block(chunk, &mut w_hat[b * QK..b * QK + QK]);
        }
        let reference: f32 = w_hat.iter().zip(&x).map(|(a, b)| a * b).sum();

        // On-the-fly path (what the real kernel will do): must match the
        // dequantized reference to floating-point tolerance.
        let on_the_fly = dot_q4_0_row_f32(&blocks, &x);
        assert!(
            (on_the_fly - reference).abs() < 1e-3,
            "on-the-fly {on_the_fly} vs reference {reference}"
        );
    }

    #[test]
    fn q4_0_quantization_error_is_bounded() {
        // Dequantized weights should track the originals within Q4 granularity.
        let w = seq(QK * 4);
        let blocks = quantize_q4_0_row(&w);
        let mut w_hat = vec![0.0f32; w.len()];
        for (b, chunk) in blocks.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
            dequantize_q4_0_block(chunk, &mut w_hat[b * QK..b * QK + QK]);
        }
        // Max abs error within a block ≤ ~scale (one quant step). With |w|≤1 and
        // 4-bit range, the step is small; assert a loose but real bound.
        let max_err = w
            .iter()
            .zip(&w_hat)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.2, "max quant error {max_err} too large");
    }

    #[test]
    fn q4_k_w4a8_matches_f32_path() {
        // The W4A8 (int8-activation) Q4_K dot must agree with the proven f32 path
        // within activation-quantization error. A layout/scale bug (the kind that
        // produced Q6_K salad) shows up as a wildly-wrong result, not a few %.
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        // Two 256-weight super-blocks (512 weights → 16 activation blocks of 32).
        let nblocks = 2usize;
        let mut row = vec![0u8; nblocks * Q4_K_BLOCK_BYTES];
        for blk in row.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            blk[0..2].copy_from_slice(&f16::from_f32(0.05).to_le_bytes()); // d
            blk[2..4].copy_from_slice(&f16::from_f32(0.018).to_le_bytes()); // dmin
            for b in blk[4..].iter_mut() {
                *b = (next() & 0xFF) as u8; // scales + packed nibbles
            }
        }
        let x: Vec<f32> = (0..nblocks * QK_K)
            .map(|_| (next() as f32 / u32::MAX as f32) * 4.0 - 2.0)
            .collect();

        let f32_dot = dot_q4_k_row_f32(&row, &x);
        let (xi8, xsc) = quantize_row_to_i8_blocks(&x);
        let q8_dot = dot_q4_k_row_q8_scalar(&row, &xi8, &xsc);

        let rel = (f32_dot - q8_dot).abs() / f32_dot.abs().max(1e-3);
        assert!(
            rel < 0.03,
            "W4A8 mismatch: f32={f32_dot} q8={q8_dot} rel={rel}"
        );

        // The NEON SDOT kernel must match the scalar W4A8 reference exactly (same
        // integer dot; only f32 reduction order differs → tiny tolerance).
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let neon = unsafe { dot_q4_k_row_q8_neon(&row, &xi8, &xsc) };
            let rel_n = (neon - q8_dot).abs() / q8_dot.abs().max(1e-3);
            assert!(
                rel_n < 1e-4,
                "NEON≠scalar W4A8: neon={neon} scalar={q8_dot}"
            );
        }
    }

    #[test]
    fn q6_k_neon_matches_scalar() {
        // The vectorised Q6_K dot must equal the scalar reference (same f32 math,
        // only reduction order differs). A bit-layout/scale bug here = token-salad.
        let mut seed = 0x51ED_C0DEu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let nblocks = 3usize;
        let mut row = vec![0u8; nblocks * Q6_K_BLOCK_BYTES];
        for b in row.iter_mut() {
            *b = (next() & 0xFF) as u8;
        }
        // d (f16) at [208..210] of each block — set a small positive scale.
        for blk in row.chunks_exact_mut(Q6_K_BLOCK_BYTES) {
            blk[208..210].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
        }
        let x: Vec<f32> = (0..nblocks * QK_K)
            .map(|_| (next() as f32 / u32::MAX as f32) * 3.0 - 1.5)
            .collect();
        let scalar = dot_q6_k_row_f32_scalar(&row, &x);
        let got = dot_q6_k_row_f32(&row, &x); // dispatches to NEON on aarch64
        let rel = (got - scalar).abs() / scalar.abs().max(1e-3);
        assert!(
            rel < 1e-4,
            "Q6_K NEON≠scalar: neon={got} scalar={scalar} rel={rel}"
        );
    }

    #[test]
    fn q5_k_neon_matches_scalar() {
        // Q5_K vectorisation must equal the (per-element-qh-fixed) scalar reference.
        let mut seed = 0xA5A5_1234u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let nblocks = 3usize;
        let mut row = vec![0u8; nblocks * Q5_K_BLOCK_BYTES];
        for b in row.iter_mut() {
            *b = (next() & 0xFF) as u8;
        }
        for blk in row.chunks_exact_mut(Q5_K_BLOCK_BYTES) {
            blk[0..2].copy_from_slice(&f16::from_f32(0.05).to_le_bytes());
            blk[2..4].copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
        }
        let x: Vec<f32> = (0..nblocks * QK_K)
            .map(|_| (next() as f32 / u32::MAX as f32) * 3.0 - 1.5)
            .collect();
        let scalar = dot_q5_k_row_f32_scalar(&row, &x);
        let got = dot_q5_k_row_f32(&row, &x);
        let rel = (got - scalar).abs() / scalar.abs().max(1e-3);
        assert!(
            rel < 1e-4,
            "Q5_K NEON≠scalar: neon={got} scalar={scalar} rel={rel}"
        );
    }
}
