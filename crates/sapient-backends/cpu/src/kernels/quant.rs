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

#[cfg(test)]
mod tests {
    use super::*;

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
}
