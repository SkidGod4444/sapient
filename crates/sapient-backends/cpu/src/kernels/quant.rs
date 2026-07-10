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

/// Per-32-block sums of an i8-quantized activation row — Q8_K-style
/// precomputed "bsums" (llama.cpp keeps the same sums inside its `block_q8_K`).
///
/// The Q4_K W4A8 kernels need `Σ x_i8` per activation sub-block for the
/// `dmin·mn` correction term. Computing that reduction inside the row kernel
/// repeats identical work for EVERY weight row (or 4-row group) of the matmul;
/// hoisting it here runs it once per activation row instead. The values are
/// exactly the sums the kernels previously reduced with `vaddlvq_s8`, so
/// kernel results stay bit-for-bit identical.
pub fn i8_block_sums(q: &[i8]) -> Vec<i32> {
    debug_assert_eq!(q.len() % QK, 0);
    q.chunks_exact(QK)
        .map(|b| b.iter().map(|&v| v as i32).sum())
        .collect()
}

/// Quantize an activation row to Q8_K-style blocks: int8 quants with ONE
/// f32 scale per 256-element SUPER-block plus precomputed per-32 sums.
///
/// This is llama.cpp's `block_q8_K` activation format for K-quant matmuls.
/// Versus [`quantize_row_to_i8_blocks`] (per-32 scales, the Q8_0 W8A8
/// format), the per-256 scale lets the K-quant kernels accumulate the
/// weight sub-scales in the INTEGER domain and pay one f32 multiply per
/// super-block instead of one f32 combine per 32-element sub-block — the
/// measured pp512 tail (see BENCHMARKS.md, prefill investigation). The
/// accuracy class changes: an outlier channel now sets the scale for its
/// whole 256-block rather than its 32-block. This is exactly what llama.cpp
/// ships for every K-quant matmul, so quality is llama.cpp-precedented —
/// but any switch of a live path to this format MUST re-verify real-model
/// greedy outputs (repo rule).
///
/// `x.len()` must be a multiple of 256 (every K-quant row is).
pub fn quantize_row_to_q8k(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    let nsuper = x.len() / QK_K;
    let mut q = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; nsuper];
    let mut sums = vec![0i32; x.len() / QK];
    for (b, blk) in x.chunks_exact(QK_K).enumerate() {
        let max_abs = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (i, &v) in blk.iter().enumerate() {
            q[b * QK_K + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        scales[b] = scale;
        for j in 0..QK_K / QK {
            let base = b * QK_K + j * QK;
            sums[b * (QK_K / QK) + j] = q[base..base + QK].iter().map(|&v| v as i32).sum();
        }
    }
    (q, scales, sums)
}

/// Scalar reference: full Q4_K weight row × Q8_K activations. The weight
/// sub-scales (6-bit sc/mn) are accumulated in the INTEGER domain per
/// super-block — `isum = Σ_j sc_j·dot_j`, `imin = Σ_j mn_j·bsum_j` (both
/// fit i32 with headroom: |sc·dot| ≤ 63·61k, ×8 ≈ 31M) — and the f32 tail
/// is exactly two multiplies + one fma per 256 weights:
/// `acc += D_b · (d·isum − dmin·imin)`. Oracle for the NEON/SMMLA variants.
pub fn dot_q4_k_row_q8k_scalar(
    row_data: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for (b, block) in row_data.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..Q4_K_BLOCK_BYTES];
        let mut q_off = 0usize;
        let mut is = 0usize;
        let (mut isum, mut imin) = (0i32, 0i32);
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let xlo = &x_i8[x_off..x_off + 32];
            let xhi = &x_i8[x_off + 32..x_off + 64];
            let (mut dot_lo, mut dot_hi) = (0i32, 0i32);
            for l in 0..32 {
                dot_lo += (qs[q_off + l] & 0x0F) as i32 * xlo[l] as i32;
                dot_hi += (qs[q_off + l] >> 4) as i32 * xhi[l] as i32;
            }
            isum += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
            imin += m1 as i32 * x_sums[x_off / QK] + m2 as i32 * x_sums[(x_off + 32) / QK];
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        acc += x_scales[b] * (d * isum as f32 - dmin * imin as f32);
    }
    acc
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
pub fn dot_q4_k_row_q8_scalar(
    row_data: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> f32 {
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

            let (mut dot_lo, mut dot_hi) = (0i32, 0i32);
            for l in 0..32 {
                let nlo = (qs[q_off + l] & 0x0F) as i32;
                let nhi = (qs[q_off + l] >> 4) as i32;
                dot_lo += nlo * xlo[l] as i32;
                dot_hi += nhi * xhi[l] as i32;
            }
            // Σx per sub-block comes precomputed (once per activation row).
            acc += x_scales[blk_lo] * (d1 * dot_lo as f32 - m1v * x_sums[blk_lo] as f32);
            acc += x_scales[blk_hi] * (d2 * dot_hi as f32 - m2v * x_sums[blk_hi] as f32);

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

/// 2×2 int8 matrix-multiply-accumulate via the ARMv8.6-A `smmla` instruction
/// (inline asm — stable Rust): treats `a` and `b` as row-major 2×8 i8 matrices
/// and accumulates `a · bᵀ` into the four i32 lanes of `acc`
/// (`[a0·b0, a0·b1, a1·b0, a1·b1]`). 32 int8 MACs per instruction — 2× `sdot` —
/// but only useful when BOTH operands carry two distinct rows (weight-row pair ×
/// activation-row pair), i.e. prefill/batch; at m = 1 half the lanes are waste.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
#[inline]
unsafe fn smmla_s32(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut r = acc;
    core::arch::asm!(
        "smmla {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) r,
        in(vreg) a,
        in(vreg) b,
        options(nomem, nostack),
    );
    r
}

/// Four Q4_K rows (R4 interleaved) × TWO int8 activation rows via `smmla` —
/// the CPU prefill kernel (m ≥ 2). Each 16-weight segment-pair costs two `trn`
/// shuffles + one `smmla` per weight-row pair instead of four `sdot`s, and the
/// weight stream is read ONCE for both activation rows. Per-output integer dots
/// and f32 combine order match [`dot_q4_k_row_q8_neon`] exactly, so every lane
/// of the 4×2 result is bit-for-bit the single-row kernel's (regression-tested).
///
/// Returns `[[row0·x0, row0·x1], …, [row3·x0, row3·x1]]`.
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("i8mm")` (implies dotprod-era
/// NEON); `packed` must be a whole Q4_K_R4 row-group; both activation slices must
/// cover the row length with per-32-block scales.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dot_q4_k_4rows_r4_x2_smmla(
    packed: &[u8],
    x0_i8: &[i8],
    x0_scales: &[f32],
    x0_sums: &[i32],
    x1_i8: &[i8],
    x1_scales: &[f32],
    x1_sums: &[i32],
) -> [[f32; 2]; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [[0.0f32; 2]; 4];
    let nb = packed.len() / (4 * Q4_K_BLOCK_BYTES);
    let mut x_off = 0usize;
    for b in 0..nb {
        let gbase = b * 4 * Q4_K_BLOCK_BYTES;
        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..];
            dv[r] = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        }
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            // Activation vectors for BOTH rows, once; per-sub-block sums come
            // precomputed (once per activation row, `i8_block_sums`).
            let x0lo0 = vld1q_s8(x0_i8.as_ptr().add(x_off));
            let x0lo1 = vld1q_s8(x0_i8.as_ptr().add(x_off + 16));
            let x0hi0 = vld1q_s8(x0_i8.as_ptr().add(x_off + 32));
            let x0hi1 = vld1q_s8(x0_i8.as_ptr().add(x_off + 48));
            let x1lo0 = vld1q_s8(x1_i8.as_ptr().add(x_off));
            let x1lo1 = vld1q_s8(x1_i8.as_ptr().add(x_off + 16));
            let x1hi0 = vld1q_s8(x1_i8.as_ptr().add(x_off + 32));
            let x1hi1 = vld1q_s8(x1_i8.as_ptr().add(x_off + 48));
            let sum_lo = [x0_sums[x_off / QK], x1_sums[x_off / QK]];
            let sum_hi = [x0_sums[(x_off + 32) / QK], x1_sums[(x_off + 32) / QK]];
            let xs_lo = [x0_scales[x_off / QK], x1_scales[x_off / QK]];
            let xs_hi = [x0_scales[(x_off + 32) / QK], x1_scales[(x_off + 32) / QK]];
            // Pair the two activation rows per 8-byte k-segment: [x0_seg, x1_seg].
            let xlo_a = vtrn1q_s64_s8(x0lo0, x1lo0);
            let xlo_b = vtrn2q_s64_s8(x0lo0, x1lo0);
            let xlo_c = vtrn1q_s64_s8(x0lo1, x1lo1);
            let xlo_d = vtrn2q_s64_s8(x0lo1, x1lo1);
            let xhi_a = vtrn1q_s64_s8(x0hi0, x1hi0);
            let xhi_b = vtrn2q_s64_s8(x0hi0, x1hi0);
            let xhi_c = vtrn1q_s64_s8(x0hi1, x1hi1);
            let xhi_d = vtrn2q_s64_s8(x0hi1, x1hi1);

            for pair in 0..2usize {
                let (r0, r1) = (pair * 2, pair * 2 + 1);
                let blk0 = &packed[gbase + r0 * Q4_K_BLOCK_BYTES..];
                let blk1 = &packed[gbase + r1 * Q4_K_BLOCK_BYTES..];
                let qs0 = blk0.as_ptr().add(16);
                let qs1 = blk1.as_ptr().add(16);
                let q0a = vld1q_u8(qs0.add(q_off));
                let q0b = vld1q_u8(qs0.add(q_off + 16));
                let q1a = vld1q_u8(qs1.add(q_off));
                let q1b = vld1q_u8(qs1.add(q_off + 16));
                let lo0a = vreinterpretq_s8_u8(vandq_u8(q0a, mask));
                let lo0b = vreinterpretq_s8_u8(vandq_u8(q0b, mask));
                let lo1a = vreinterpretq_s8_u8(vandq_u8(q1a, mask));
                let lo1b = vreinterpretq_s8_u8(vandq_u8(q1b, mask));
                let hi0a = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0a));
                let hi0b = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0b));
                let hi1a = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1a));
                let hi1b = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1b));

                // Weight-row pairs per 8-byte k-segment: [w_r0_seg, w_r1_seg].
                let wlo_a = vtrn1q_s64_s8(lo0a, lo1a);
                let wlo_b = vtrn2q_s64_s8(lo0a, lo1a);
                let wlo_c = vtrn1q_s64_s8(lo0b, lo1b);
                let wlo_d = vtrn2q_s64_s8(lo0b, lo1b);
                let whi_a = vtrn1q_s64_s8(hi0a, hi1a);
                let whi_b = vtrn2q_s64_s8(hi0a, hi1a);
                let whi_c = vtrn1q_s64_s8(hi0b, hi1b);
                let whi_d = vtrn2q_s64_s8(hi0b, hi1b);

                let zero = vdupq_n_s32(0);
                // Lanes: [w_r0·x0, w_r0·x1, w_r1·x0, w_r1·x1].
                let mut dlo = smmla_s32(zero, wlo_a, xlo_a);
                dlo = smmla_s32(dlo, wlo_b, xlo_b);
                dlo = smmla_s32(dlo, wlo_c, xlo_c);
                dlo = smmla_s32(dlo, wlo_d, xlo_d);
                let mut dhi = smmla_s32(zero, whi_a, xhi_a);
                dhi = smmla_s32(dhi, whi_b, xhi_b);
                dhi = smmla_s32(dhi, whi_c, xhi_c);
                dhi = smmla_s32(dhi, whi_d, xhi_d);
                let dlo_arr = [
                    vgetq_lane_s32::<0>(dlo),
                    vgetq_lane_s32::<1>(dlo),
                    vgetq_lane_s32::<2>(dlo),
                    vgetq_lane_s32::<3>(dlo),
                ];
                let dhi_arr = [
                    vgetq_lane_s32::<0>(dhi),
                    vgetq_lane_s32::<1>(dhi),
                    vgetq_lane_s32::<2>(dhi),
                    vgetq_lane_s32::<3>(dhi),
                ];

                for (ri, row) in [r0, r1].into_iter().enumerate() {
                    let blk = &packed[gbase + row * Q4_K_BLOCK_BYTES..];
                    let scales = &blk[4..16];
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d1 = dv[row] * sc1 as f32;
                    let m1v = dminv[row] * m1 as f32;
                    let d2 = dv[row] * sc2 as f32;
                    let m2v = dminv[row] * m2 as f32;
                    for xr in 0..2usize {
                        let dot_lo = dlo_arr[ri * 2 + xr];
                        let dot_hi = dhi_arr[ri * 2 + xr];
                        acc[row][xr] += xs_lo[xr] * (d1 * dot_lo as f32 - m1v * sum_lo[xr] as f32);
                        acc[row][xr] += xs_hi[xr] * (d2 * dot_hi as f32 - m2v * sum_hi[xr] as f32);
                    }
                }
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

/// `TRN1` on the 64-bit halves of two i8 vectors: `[a.lo, b.lo]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vtrn1q_s64_s8(
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;
    vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)))
}

/// `TRN2` on the 64-bit halves of two i8 vectors: `[a.hi, b.hi]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vtrn2q_s64_s8(
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;
    vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)))
}

/// NEON Q4_K × Q8_K row dot — the sdot core of [`dot_q4_k_row_q8_neon`]
/// with the weight sub-scales accumulated in the INTEGER domain and one f32
/// fma per super-block (llama.cpp `ggml_vec_dot_q4_K_q8_K` shape). Integer
/// sums are exact, and the f32 combine order matches
/// [`dot_q4_k_row_q8k_scalar`] — bit-identical to the oracle.
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`. Activations
/// are Q8_K: `x_scales` has ONE f32 per 256-element super-block; `x_sums`
/// one i32 per 32-element sub-block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_row_q8k_neon(
    row_data: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> f32 {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for (b, block) in row_data.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..Q4_K_BLOCK_BYTES];
        let mut q_off = 0usize;
        let mut is = 0usize;
        let (mut isum, mut imin) = (0i32, 0i32);
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

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

            isum += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
            imin += m1 as i32 * x_sums[x_off / QK] + m2 as i32 * x_sums[(x_off + 32) / QK];

            x_off += 64;
            q_off += 32;
            is += 2;
        }
        acc += x_scales[b] * (d * isum as f32 - dmin * imin as f32);
    }
    acc
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
pub unsafe fn dot_q4_k_row_q8_neon(
    row_data: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> f32 {
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

            // Σ x_i8 per 32-sub-block (min-correction term) — precomputed once
            // per activation row (`i8_block_sums`), not re-reduced per weight row.
            let blk_lo = x_off / QK;
            let blk_hi = (x_off + 32) / QK;
            acc += x_scales[blk_lo] * (d1 * dot_lo as f32 - m1v * x_sums[blk_lo] as f32);
            acc += x_scales[blk_hi] * (d2 * dot_hi as f32 - m2v * x_sums[blk_hi] as f32);

            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

/// Four Q4_K rows against ONE i8 activation vector — the multi-row GEMV core
/// (llama.cpp-style): the activation registers and their per-sub-block sums are
/// loaded/computed **once per 64-weight group and reused across all four rows**,
/// cutting activation traffic 4× and giving four independent `sdot` dependency
/// chains for ILP. Per-row arithmetic (values *and* order) is identical to
/// [`dot_q4_k_row_q8_neon`], so each lane of the result is bit-for-bit equal to
/// the single-row kernel (regression-tested).
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`; all four row
/// slices must be complete Q4_K rows of equal length covered by `x_i8`/`x_scales`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_4rows_q8_neon(
    rows: [&[u8]; 4],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; 4];
    let n_blocks = rows[0].len() / Q4_K_BLOCK_BYTES;
    let mut x_off = 0usize;
    for bi in 0..n_blocks {
        let base = bi * Q4_K_BLOCK_BYTES;
        // Per-row super-block headers, hoisted once per block.
        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let b = &rows[r][base..base + Q4_K_BLOCK_BYTES];
            dv[r] = f16::from_le_bytes([b[0], b[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([b[2], b[3]]).to_f32();
        }
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            // ── Shared activation work: loaded ONCE for all 4 rows; the
            // per-sub-block Σx comes precomputed (once per activation row). ──
            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));
            let sum_lo = x_sums[x_off / QK];
            let sum_hi = x_sums[(x_off + 32) / QK];
            let xs_lo = x_scales[x_off / QK];
            let xs_hi = x_scales[(x_off + 32) / QK];

            for r in 0..4 {
                let b = &rows[r][base..base + Q4_K_BLOCK_BYTES];
                let scales = &b[4..16];
                let qs = &b[16..Q4_K_BLOCK_BYTES];
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let d1 = dv[r] * sc1 as f32;
                let m1v = dminv[r] * m1 as f32;
                let d2 = dv[r] * sc2 as f32;
                let m2v = dminv[r] * m2 as f32;

                let q0 = vld1q_u8(qs.as_ptr().add(q_off));
                let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

                let zero = vdupq_n_s32(0);
                let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                acc[r] += xs_lo * (d1 * dot_lo as f32 - m1v * sum_lo as f32);
                acc[r] += xs_hi * (d2 * dot_hi as f32 - m2v * sum_hi as f32);
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        // x_off advanced inside the group loop covers this block's 256 weights.
    }
    acc
}

/// Repack `n` Q4_K rows (row-major raw blocks) into the Q4_K_R4 layout: groups
/// of 4 consecutive rows, super-blocks interleaved block-major within the group
/// — `[r0.b0, r1.b0, r2.b0, r3.b0, r0.b1, …]`. A pure permutation of the
/// 144-byte blocks (same total size); `n` must be a multiple of 4. The 4-row
/// SDOT kernel then reads ONE contiguous stream per row-group instead of four
/// row-strided streams (the prefetcher-thrashing this layout exists to fix).
pub fn repack_q4_k_rows4(blocks: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(n % 4, 0, "Q4_K_R4 repack: rows must be a multiple of 4");
    assert_eq!(k % QK_K, 0);
    let nb = k / QK_K;
    let row_bytes = nb * Q4_K_BLOCK_BYTES;
    assert_eq!(blocks.len(), n * row_bytes);
    let mut out = vec![0u8; blocks.len()];
    for g in 0..n / 4 {
        for b in 0..nb {
            for r in 0..4 {
                let src = ((g * 4 + r) * nb + b) * Q4_K_BLOCK_BYTES;
                let dst = (g * 4 * nb + b * 4 + r) * Q4_K_BLOCK_BYTES;
                out[dst..dst + Q4_K_BLOCK_BYTES]
                    .copy_from_slice(&blocks[src..src + Q4_K_BLOCK_BYTES]);
            }
        }
    }
    out
}

/// Four Q4_K rows in the **Q4_K_R4 interleaved layout** against one i8
/// activation vector. `packed` is one row-group: `4 · (k/256)` blocks, block-major
/// `[r0.b, r1.b, r2.b, r3.b]` — the kernel walks it front to back (one stream).
/// Per-row arithmetic is identical to [`dot_q4_k_row_q8_neon`] (bit-for-bit).
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`; `packed` must be
/// a whole Q4_K_R4 row-group covered by `x_i8`/`x_scales`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_4rows_r4_neon(
    packed: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; 4];
    let nb = packed.len() / (4 * Q4_K_BLOCK_BYTES);
    let mut x_off = 0usize;
    for b in 0..nb {
        let gbase = b * 4 * Q4_K_BLOCK_BYTES;
        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..];
            dv[r] = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        }
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));
            let sum_lo = x_sums[x_off / QK];
            let sum_hi = x_sums[(x_off + 32) / QK];
            let xs_lo = x_scales[x_off / QK];
            let xs_hi = x_scales[(x_off + 32) / QK];

            for r in 0..4 {
                let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..gbase + (r + 1) * Q4_K_BLOCK_BYTES];
                let scales = &blk[4..16];
                let qs = &blk[16..Q4_K_BLOCK_BYTES];
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let d1 = dv[r] * sc1 as f32;
                let m1v = dminv[r] * m1 as f32;
                let d2 = dv[r] * sc2 as f32;
                let m2v = dminv[r] * m2 as f32;

                let q0 = vld1q_u8(qs.as_ptr().add(q_off));
                let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

                let zero = vdupq_n_s32(0);
                let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                acc[r] += xs_lo * (d1 * dot_lo as f32 - m1v * sum_lo as f32);
                acc[r] += xs_hi * (d2 * dot_hi as f32 - m2v * sum_hi as f32);
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

/// Four Q4_K rows (row-major, non-R4) × ONE Q8_K activation row — the plain
/// multi-row GEMV with the integer-domain combine, for un-repacked tensors
/// (mmap weights, MoE experts). Bit-identical per lane to
/// [`dot_q4_k_row_q8k_neon`]; keeps the repack-on/off invariance now that
/// Q8_K is the default activation format on BOTH paths.
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`; all four
/// row slices are whole Q4_K rows covered by the Q8_K activations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_4rows_q8k_neon(
    rows: [&[u8]; 4],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; 4];
    let n_blocks = rows[0].len() / Q4_K_BLOCK_BYTES;
    let mut x_off = 0usize;
    for (bi, &db) in x_scales.iter().enumerate().take(n_blocks) {
        let base = bi * Q4_K_BLOCK_BYTES;
        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let b = &rows[r][base..base + Q4_K_BLOCK_BYTES];
            dv[r] = f16::from_le_bytes([b[0], b[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([b[2], b[3]]).to_f32();
        }
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut isum = [0i32; 4];
        let mut imin = [0i32; 4];
        for _ in 0..(QK_K / 64) {
            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));
            let sum_lo = x_sums[x_off / QK];
            let sum_hi = x_sums[(x_off + 32) / QK];

            for r in 0..4 {
                let b = &rows[r][base..base + Q4_K_BLOCK_BYTES];
                let scales = &b[4..16];
                let qs = &b[16..Q4_K_BLOCK_BYTES];
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);

                let q0 = vld1q_u8(qs.as_ptr().add(q_off));
                let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

                let zero = vdupq_n_s32(0);
                let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                isum[r] += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
                imin[r] += m1 as i32 * sum_lo + m2 as i32 * sum_hi;
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for r in 0..4 {
            acc[r] += db * (dv[r] * isum[r] as f32 - dminv[r] * imin[r] as f32);
        }
    }
    acc
}

/// Four Q4_K rows (R4 interleaved) × ONE Q8_K activation row. The multi-row
/// GEMV core with the integer-domain sub-scale accumulation of
/// [`dot_q4_k_row_q8k_neon`]: per super-block, per-row `isum`/`imin` build in
/// i32 across the four 64-weight groups, then ONE f32 fma per row per
/// super-block. Bit-identical per lane to the single-row Q8_K kernel.
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`; `packed`
/// must be a whole Q4_K_R4 row-group covered by Q8_K activations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_4rows_r4_q8k_neon(
    packed: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; 4];
    let nb = packed.len() / (4 * Q4_K_BLOCK_BYTES);
    let mut x_off = 0usize;
    for (b, &db) in x_scales.iter().enumerate().take(nb) {
        let gbase = b * 4 * Q4_K_BLOCK_BYTES;
        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..];
            dv[r] = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        }
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut isum = [0i32; 4];
        let mut imin = [0i32; 4];
        for _ in 0..(QK_K / 64) {
            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));
            let sum_lo = x_sums[x_off / QK];
            let sum_hi = x_sums[(x_off + 32) / QK];

            for r in 0..4 {
                let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..gbase + (r + 1) * Q4_K_BLOCK_BYTES];
                let scales = &blk[4..16];
                let qs = &blk[16..Q4_K_BLOCK_BYTES];
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);

                let q0 = vld1q_u8(qs.as_ptr().add(q_off));
                let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

                let zero = vdupq_n_s32(0);
                let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                isum[r] += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
                imin[r] += m1 as i32 * sum_lo + m2 as i32 * sum_hi;
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for r in 0..4 {
            acc[r] += db * (dv[r] * isum[r] as f32 - dminv[r] * imin[r] as f32);
        }
    }
    acc
}

/// Four Q4_K rows (R4) × TWO Q8_K activation rows via `smmla` — the prefill
/// kernel with the integer-domain combine. Same trn/smmla core as
/// [`dot_q4_k_4rows_r4_x2_smmla`]; the per-(row, x) `isum`/`imin` build in
/// i32 across the super-block and the f32 tail collapses to one fma per
/// (row, x) per 256 weights. Bit-identical per lane to
/// [`dot_q4_k_row_q8k_neon`].
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("i8mm")`; `packed` must
/// be a whole Q4_K_R4 row-group; both activation rows are Q8_K format.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dot_q4_k_4rows_r4_x2_q8k_smmla(
    packed: &[u8],
    x0_i8: &[i8],
    x0_scales: &[f32],
    x0_sums: &[i32],
    x1_i8: &[i8],
    x1_scales: &[f32],
    x1_sums: &[i32],
) -> [[f32; 2]; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [[0.0f32; 2]; 4];
    let nb = packed.len() / (4 * Q4_K_BLOCK_BYTES);
    let mut x_off = 0usize;
    for (b, (&db0, &db1)) in x0_scales.iter().zip(x1_scales.iter()).enumerate().take(nb) {
        let gbase = b * 4 * Q4_K_BLOCK_BYTES;
        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..];
            dv[r] = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        }
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut isum = [[0i32; 2]; 4];
        let mut imin = [[0i32; 2]; 4];
        for _ in 0..(QK_K / 64) {
            let x0lo0 = vld1q_s8(x0_i8.as_ptr().add(x_off));
            let x0lo1 = vld1q_s8(x0_i8.as_ptr().add(x_off + 16));
            let x0hi0 = vld1q_s8(x0_i8.as_ptr().add(x_off + 32));
            let x0hi1 = vld1q_s8(x0_i8.as_ptr().add(x_off + 48));
            let x1lo0 = vld1q_s8(x1_i8.as_ptr().add(x_off));
            let x1lo1 = vld1q_s8(x1_i8.as_ptr().add(x_off + 16));
            let x1hi0 = vld1q_s8(x1_i8.as_ptr().add(x_off + 32));
            let x1hi1 = vld1q_s8(x1_i8.as_ptr().add(x_off + 48));
            let sum_lo = [x0_sums[x_off / QK], x1_sums[x_off / QK]];
            let sum_hi = [x0_sums[(x_off + 32) / QK], x1_sums[(x_off + 32) / QK]];
            let xlo_a = vtrn1q_s64_s8(x0lo0, x1lo0);
            let xlo_b = vtrn2q_s64_s8(x0lo0, x1lo0);
            let xlo_c = vtrn1q_s64_s8(x0lo1, x1lo1);
            let xlo_d = vtrn2q_s64_s8(x0lo1, x1lo1);
            let xhi_a = vtrn1q_s64_s8(x0hi0, x1hi0);
            let xhi_b = vtrn2q_s64_s8(x0hi0, x1hi0);
            let xhi_c = vtrn1q_s64_s8(x0hi1, x1hi1);
            let xhi_d = vtrn2q_s64_s8(x0hi1, x1hi1);

            for pair in 0..2usize {
                let (r0, r1) = (pair * 2, pair * 2 + 1);
                let blk0 = &packed[gbase + r0 * Q4_K_BLOCK_BYTES..];
                let blk1 = &packed[gbase + r1 * Q4_K_BLOCK_BYTES..];
                let qs0 = blk0.as_ptr().add(16);
                let qs1 = blk1.as_ptr().add(16);
                let q0a = vld1q_u8(qs0.add(q_off));
                let q0b = vld1q_u8(qs0.add(q_off + 16));
                let q1a = vld1q_u8(qs1.add(q_off));
                let q1b = vld1q_u8(qs1.add(q_off + 16));
                let lo0a = vreinterpretq_s8_u8(vandq_u8(q0a, mask));
                let lo0b = vreinterpretq_s8_u8(vandq_u8(q0b, mask));
                let lo1a = vreinterpretq_s8_u8(vandq_u8(q1a, mask));
                let lo1b = vreinterpretq_s8_u8(vandq_u8(q1b, mask));
                let hi0a = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0a));
                let hi0b = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0b));
                let hi1a = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1a));
                let hi1b = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1b));

                let wlo_a = vtrn1q_s64_s8(lo0a, lo1a);
                let wlo_b = vtrn2q_s64_s8(lo0a, lo1a);
                let wlo_c = vtrn1q_s64_s8(lo0b, lo1b);
                let wlo_d = vtrn2q_s64_s8(lo0b, lo1b);
                let whi_a = vtrn1q_s64_s8(hi0a, hi1a);
                let whi_b = vtrn2q_s64_s8(hi0a, hi1a);
                let whi_c = vtrn1q_s64_s8(hi0b, hi1b);
                let whi_d = vtrn2q_s64_s8(hi0b, hi1b);

                let zero = vdupq_n_s32(0);
                let mut dlo = smmla_s32(zero, wlo_a, xlo_a);
                dlo = smmla_s32(dlo, wlo_b, xlo_b);
                dlo = smmla_s32(dlo, wlo_c, xlo_c);
                dlo = smmla_s32(dlo, wlo_d, xlo_d);
                let mut dhi = smmla_s32(zero, whi_a, xhi_a);
                dhi = smmla_s32(dhi, whi_b, xhi_b);
                dhi = smmla_s32(dhi, whi_c, xhi_c);
                dhi = smmla_s32(dhi, whi_d, xhi_d);
                let dlo_arr = [
                    vgetq_lane_s32::<0>(dlo),
                    vgetq_lane_s32::<1>(dlo),
                    vgetq_lane_s32::<2>(dlo),
                    vgetq_lane_s32::<3>(dlo),
                ];
                let dhi_arr = [
                    vgetq_lane_s32::<0>(dhi),
                    vgetq_lane_s32::<1>(dhi),
                    vgetq_lane_s32::<2>(dhi),
                    vgetq_lane_s32::<3>(dhi),
                ];

                for (ri, row) in [r0, r1].into_iter().enumerate() {
                    let blk = &packed[gbase + row * Q4_K_BLOCK_BYTES..];
                    let scales = &blk[4..16];
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    for xr in 0..2usize {
                        let dot_lo = dlo_arr[ri * 2 + xr];
                        let dot_hi = dhi_arr[ri * 2 + xr];
                        isum[row][xr] += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
                        imin[row][xr] += m1 as i32 * sum_lo[xr] + m2 as i32 * sum_hi[xr];
                    }
                }
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        let db = [db0, db1];
        for r in 0..4 {
            for xr in 0..2usize {
                acc[r][xr] += db[xr] * (dv[r] * isum[r][xr] as f32 - dminv[r] * imin[r][xr] as f32);
            }
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

/// Scalar reference for the W6A8 Q6_K row dot: int8 activations (per-32-block
/// scales from `quantize_row_to_i8_blocks`) against 6-bit weights, the −32
/// folded into the integer dot. Oracle for the NEON kernels below.
pub fn dot_q6_k_row_q8_scalar(row_data: &[u8], x_i8: &[i8], x_scales: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    for block in row_data.chunks_exact(Q6_K_BLOCK_BYTES) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_base = 0usize;
        for _ in 0..(QK_K / 128) {
            for &l0 in &[0usize, 16usize] {
                let is = l0 / 16;
                // Four 16-element scale groups at x offsets +0 / +32 / +64 / +96.
                for (sub, sc_off, x_add) in
                    [(0usize, 0usize, 0usize), (1, 2, 32), (2, 4, 64), (3, 6, 96)]
                {
                    let mut dot = 0i32;
                    for l in 0..16usize {
                        let b = l0 + l;
                        let q = match sub {
                            0 => (ql[ql_off + b] & 0x0F) | ((qh[qh_off + b] & 3) << 4),
                            1 => (ql[ql_off + b + 32] & 0x0F) | (((qh[qh_off + b] >> 2) & 3) << 4),
                            2 => (ql[ql_off + b] >> 4) | (((qh[qh_off + b] >> 4) & 3) << 4),
                            _ => (ql[ql_off + b + 32] >> 4) | (((qh[qh_off + b] >> 6) & 3) << 4),
                        };
                        let xi = x_off + x_add + l0 + l;
                        dot += (q as i32 - 32) * x_i8[xi] as i32;
                    }
                    let xs = x_scales[(x_off + x_add + l0) / QK];
                    acc += d * (sc[sc_base + is + sc_off] as i8 as f32) * xs * dot as f32;
                }
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    acc
}

/// NEON W6A8 Q6_K row dot — one `sdot` per 16-element scale group instead of
/// the f32 path's widen/convert/FMA chains. Bit-for-bit equal to
/// [`dot_q6_k_row_q8_scalar`] (integer dots are exact; the f32 combine order
/// matches). The −32 is applied in i8 before the dot (q−32 ∈ [−32,31]).
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q6_k_row_q8_neon(row_data: &[u8], x_i8: &[i8], x_scales: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mask0f = vdupq_n_u8(0x0F);
    let mask3 = vdupq_n_u8(0x03);
    let m32 = vdupq_n_s8(32);
    let mut acc = 0.0f32;
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
            for &l0 in &[0usize, 16usize] {
                let is = l0 / 16;
                let ql_lo = vld1q_u8(ql.add(ql_off + l0));
                let ql_hi = vld1q_u8(ql.add(ql_off + l0 + 32));
                let qhv = vld1q_u8(qh.add(qh_off + l0));

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

                macro_rules! group {
                    ($q:expr, $sc_off:expr, $x_add:expr) => {{
                        let qm = vsubq_s8(vreinterpretq_s8_u8($q), m32);
                        let xi = x_off + $x_add + l0;
                        let xv = vld1q_s8(x_i8.as_ptr().add(xi));
                        let dot = vaddvq_s32(sdot_s32(vdupq_n_s32(0), qm, xv));
                        let xs = x_scales[xi / QK];
                        acc += d * (sc[sc_base + is + $sc_off] as i8 as f32) * xs * dot as f32;
                    }};
                }
                group!(q1, 0, 0);
                group!(q2, 2, 32);
                group!(q3, 4, 64);
                group!(q4, 6, 96);
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    acc
}

/// Repack `n` Q6_K rows into the Q6_K_R4 layout — identical 4-row block-major
/// interleave to [`repack_q4_k_rows4`], over 210-byte super-blocks.
pub fn repack_q6_k_rows4(blocks: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(n % 4, 0, "Q6_K_R4 repack: rows must be a multiple of 4");
    assert_eq!(k % QK_K, 0);
    let nb = k / QK_K;
    let row_bytes = nb * Q6_K_BLOCK_BYTES;
    assert_eq!(blocks.len(), n * row_bytes);
    let mut out = vec![0u8; blocks.len()];
    for g in 0..n / 4 {
        for b in 0..nb {
            for r in 0..4 {
                let src = ((g * 4 + r) * nb + b) * Q6_K_BLOCK_BYTES;
                let dst = (g * 4 * nb + b * 4 + r) * Q6_K_BLOCK_BYTES;
                out[dst..dst + Q6_K_BLOCK_BYTES]
                    .copy_from_slice(&blocks[src..src + Q6_K_BLOCK_BYTES]);
            }
        }
    }
    out
}

/// Four Q6_K rows in the **Q6_K_R4 interleaved layout**, W6A8: one packed
/// weight stream per row-group, int8 activations loaded ONCE per 16-element
/// scale group and reused across all four rows' `sdot`s. Per-row math is
/// bit-for-bit [`dot_q6_k_row_q8_neon`]'s (regression-tested).
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("dotprod")`; `packed` must
/// be a whole Q6_K_R4 row-group covered by `x_i8`/`x_scales`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q6_k_4rows_r4_q8_neon(packed: &[u8], x_i8: &[i8], x_scales: &[f32]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask0f = vdupq_n_u8(0x0F);
    let mask3 = vdupq_n_u8(0x03);
    let m32 = vdupq_n_s8(32);
    let mut acc = [0.0f32; 4];
    let nb = packed.len() / (4 * Q6_K_BLOCK_BYTES);
    let mut x_off = 0usize;
    for b in 0..nb {
        let gbase = b * 4 * Q6_K_BLOCK_BYTES;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_base = 0usize;
        let mut xo = x_off;
        for _ in 0..(QK_K / 128) {
            for &l0 in &[0usize, 16usize] {
                let is = l0 / 16;
                // Shared activation vectors + scales for the four 16-groups.
                let xv1 = vld1q_s8(x_i8.as_ptr().add(xo + l0));
                let xv2 = vld1q_s8(x_i8.as_ptr().add(xo + 32 + l0));
                let xv3 = vld1q_s8(x_i8.as_ptr().add(xo + 64 + l0));
                let xv4 = vld1q_s8(x_i8.as_ptr().add(xo + 96 + l0));
                let xs1 = x_scales[(xo + l0) / QK];
                let xs2 = x_scales[(xo + 32 + l0) / QK];
                let xs3 = x_scales[(xo + 64 + l0) / QK];
                let xs4 = x_scales[(xo + 96 + l0) / QK];

                for r in 0..4 {
                    let block =
                        &packed[gbase + r * Q6_K_BLOCK_BYTES..gbase + (r + 1) * Q6_K_BLOCK_BYTES];
                    let ql = block.as_ptr();
                    let qh = block.as_ptr().add(128);
                    let sc = &block[192..208];
                    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

                    let ql_lo = vld1q_u8(ql.add(ql_off + l0));
                    let ql_hi = vld1q_u8(ql.add(ql_off + l0 + 32));
                    let qhv = vld1q_u8(qh.add(qh_off + l0));

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

                    macro_rules! group {
                        ($q:expr, $sc_off:expr, $xv:expr, $xs:expr) => {{
                            let qm = vsubq_s8(vreinterpretq_s8_u8($q), m32);
                            let dot = vaddvq_s32(sdot_s32(vdupq_n_s32(0), qm, $xv));
                            acc[r] +=
                                d * (sc[sc_base + is + $sc_off] as i8 as f32) * $xs * dot as f32;
                        }};
                    }
                    group!(q1, 0, xv1, xs1);
                    group!(q2, 2, xv2, xs2);
                    group!(q3, 4, xv3, xs3);
                    group!(q4, 6, xv4, xs4);
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        x_off += QK_K;
    }
    acc
}

/// Four Q6_K rows in the **Q6_K_R4 interleaved layout** against one f32
/// activation vector: walks one contiguous packed stream per row-group; per-row
/// math (values and order) is identical to [`dot_q6_k_row_f32_neon`], so each
/// result lane is bit-for-bit the single-row kernel's (regression-tested).
///
/// # Safety
/// NEON (always present on aarch64); `packed` must be a whole Q6_K_R4 row-group
/// covered by `x`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_q6_k_4rows_r4_neon(packed: &[u8], x: &[f32]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask0f = vdupq_n_u8(0x0F);
    let mask3 = vdupq_n_u8(0x03);
    let m32 = vdupq_n_f32(32.0);
    let mut accv = [vdupq_n_f32(0.0); 4];
    let nb = packed.len() / (4 * Q6_K_BLOCK_BYTES);
    let mut x_off = 0usize;
    for b in 0..nb {
        let gbase = b * 4 * Q6_K_BLOCK_BYTES;
        for r in 0..4 {
            let block = &packed[gbase + r * Q6_K_BLOCK_BYTES..gbase + (r + 1) * Q6_K_BLOCK_BYTES];
            let ql = block.as_ptr();
            let qh = block.as_ptr().add(128);
            let sc = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
            let mut acc = accv[r];
            let mut xo = x_off;
            let mut ql_off = 0usize;
            let mut qh_off = 0usize;
            let mut sc_base = 0usize;
            for _ in 0..(QK_K / 128) {
                for &l0 in &[0usize, 16usize] {
                    let is = l0 / 16;
                    let ql_lo = vld1q_u8(ql.add(ql_off + l0));
                    let ql_hi = vld1q_u8(ql.add(ql_off + l0 + 32));
                    let qhv = vld1q_u8(qh.add(qh_off + l0));

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
                    accum!(q1, s1, xo + l0);
                    accum!(q2, s2, xo + 32 + l0);
                    accum!(q3, s3, xo + 64 + l0);
                    accum!(q4, s4, xo + 96 + l0);
                }
                xo += 128;
                ql_off += 64;
                qh_off += 32;
                sc_base += 8;
            }
            accv[r] = acc;
        }
        x_off += QK_K;
    }
    [
        vaddvq_f32(accv[0]),
        vaddvq_f32(accv[1]),
        vaddvq_f32(accv[2]),
        vaddvq_f32(accv[3]),
    ]
}

/// Four Q6_K rows (R4 interleaved) × TWO int8 activation rows via `smmla` —
/// the Q6_K CPU prefill kernel (m ≥ 2), mirroring
/// [`dot_q4_k_4rows_r4_x2_smmla`]. Each 16-element scale group costs two `trn`
/// shuffles + two `smmla`s per weight-row pair (covering both activation rows)
/// instead of four `sdot`s. Per-output integer dots and f32 combine order match
/// [`dot_q6_k_row_q8_neon`] exactly — all 8 result lanes bit-identical
/// (regression-tested).
///
/// Returns `[[row0·x0, row0·x1], …, [row3·x0, row3·x1]]`.
///
/// # Safety
/// Caller must verify `is_aarch64_feature_detected!("i8mm")`; `packed` must be a
/// whole Q6_K_R4 row-group covered by both activation slices.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn dot_q6_k_4rows_r4_x2_smmla(
    packed: &[u8],
    x0_i8: &[i8],
    x0_scales: &[f32],
    x1_i8: &[i8],
    x1_scales: &[f32],
) -> [[f32; 2]; 4] {
    use std::arch::aarch64::*;
    let mask0f = vdupq_n_u8(0x0F);
    let mask3 = vdupq_n_u8(0x03);
    let m32 = vdupq_n_s8(32);
    let mut acc = [[0.0f32; 2]; 4];
    let nb = packed.len() / (4 * Q6_K_BLOCK_BYTES);
    let mut x_off = 0usize;
    for b in 0..nb {
        let gbase = b * 4 * Q6_K_BLOCK_BYTES;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_base = 0usize;
        let mut xo = x_off;
        for _ in 0..(QK_K / 128) {
            for &l0 in &[0usize, 16usize] {
                let is = l0 / 16;
                // Both activation rows' vectors for the four 16-groups, once.
                let x0v = [
                    vld1q_s8(x0_i8.as_ptr().add(xo + l0)),
                    vld1q_s8(x0_i8.as_ptr().add(xo + 32 + l0)),
                    vld1q_s8(x0_i8.as_ptr().add(xo + 64 + l0)),
                    vld1q_s8(x0_i8.as_ptr().add(xo + 96 + l0)),
                ];
                let x1v = [
                    vld1q_s8(x1_i8.as_ptr().add(xo + l0)),
                    vld1q_s8(x1_i8.as_ptr().add(xo + 32 + l0)),
                    vld1q_s8(x1_i8.as_ptr().add(xo + 64 + l0)),
                    vld1q_s8(x1_i8.as_ptr().add(xo + 96 + l0)),
                ];
                let xs0 = [
                    x0_scales[(xo + l0) / QK],
                    x0_scales[(xo + 32 + l0) / QK],
                    x0_scales[(xo + 64 + l0) / QK],
                    x0_scales[(xo + 96 + l0) / QK],
                ];
                let xs1 = [
                    x1_scales[(xo + l0) / QK],
                    x1_scales[(xo + 32 + l0) / QK],
                    x1_scales[(xo + 64 + l0) / QK],
                    x1_scales[(xo + 96 + l0) / QK],
                ];

                for pair in 0..2usize {
                    let (r0, r1) = (pair * 2, pair * 2 + 1);
                    let blk0 = &packed[gbase + r0 * Q6_K_BLOCK_BYTES..];
                    let blk1 = &packed[gbase + r1 * Q6_K_BLOCK_BYTES..];

                    // Reconstruct the four 6-bit groups (q−32, i8) per row.
                    let mut qm = [[vdupq_n_s8(0); 4]; 2];
                    for (ri, blk) in [blk0, blk1].into_iter().enumerate() {
                        let ql = blk.as_ptr();
                        let qh = blk.as_ptr().add(128);
                        let ql_lo = vld1q_u8(ql.add(ql_off + l0));
                        let ql_hi = vld1q_u8(ql.add(ql_off + l0 + 32));
                        let qhv = vld1q_u8(qh.add(qh_off + l0));
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
                        qm[ri] = [
                            vsubq_s8(vreinterpretq_s8_u8(q1), m32),
                            vsubq_s8(vreinterpretq_s8_u8(q2), m32),
                            vsubq_s8(vreinterpretq_s8_u8(q3), m32),
                            vsubq_s8(vreinterpretq_s8_u8(q4), m32),
                        ];
                    }

                    for g in 0..4usize {
                        // [w_r0_seg, w_r1_seg] × [x0_seg, x1_seg] per 8-byte half.
                        let wa = vtrn1q_s64_s8(qm[0][g], qm[1][g]);
                        let wb = vtrn2q_s64_s8(qm[0][g], qm[1][g]);
                        let xa = vtrn1q_s64_s8(x0v[g], x1v[g]);
                        let xb = vtrn2q_s64_s8(x0v[g], x1v[g]);
                        let d2 = smmla_s32(smmla_s32(vdupq_n_s32(0), wa, xa), wb, xb);
                        // Lanes: [r0·x0, r0·x1, r1·x0, r1·x1].
                        let dots = [
                            vgetq_lane_s32::<0>(d2),
                            vgetq_lane_s32::<1>(d2),
                            vgetq_lane_s32::<2>(d2),
                            vgetq_lane_s32::<3>(d2),
                        ];
                        let sc_off = is + 2 * g;
                        for (ri, row) in [r0, r1].into_iter().enumerate() {
                            let blk = &packed[gbase + row * Q6_K_BLOCK_BYTES..];
                            let d = f16::from_le_bytes([blk[208], blk[209]]).to_f32();
                            let scv = blk[192 + sc_base + sc_off] as i8 as f32;
                            acc[row][0] += d * scv * xs0[g] * dots[ri * 2] as f32;
                            acc[row][1] += d * scv * xs1[g] * dots[ri * 2 + 1] as f32;
                        }
                    }
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        x_off += QK_K;
    }
    acc
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

    // ---------------------------------------------------------------------
    // Corruption-magnitude benchmark (differential-verification methodology).
    //
    // Each helper below RECONSTRUCTS a historical silent-correctness bug that
    // SAPIENT shipped and later fixed (Q6_K scale mis-indexing, Q5_K 5th-bit
    // mis-indexing, per-row activation quantization), so we can quantify the
    // relative error each bug injects into a single matmul row vs the verified
    // reference. The reconstructions are self-validated: the Q6_K variant must
    // reproduce the documented magnitude (896 on the canonical block) before its
    // error distribution is trusted. Run with:
    //   cargo test -p sapient-backends-cpu --lib corruption_magnitude -- --nocapture
    // ---------------------------------------------------------------------

    // Deterministic LCG byte stream (no rand dependency).
    fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    fn rand_x(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 3.0 - 1.5
            })
            .collect()
    }

    // A realistic random Q6_K block: random ql/qh/scales, fixed small d.
    fn rand_q6_k_block(seed: u64) -> Vec<u8> {
        let mut blk = lcg_bytes(seed, Q6_K_BLOCK_BYTES);
        blk[208..210].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
        blk
    }

    fn rand_q5_k_block(seed: u64) -> Vec<u8> {
        let mut blk = lcg_bytes(seed, Q5_K_BLOCK_BYTES);
        blk[0..2].copy_from_slice(&f16::from_f32(0.05).to_le_bytes());
        blk[2..4].copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
        blk
    }

    // Buggy Q6_K dot: one scale per 32-element sub-group (the shipped bug —
    // sc[ib..ib+4], ib += 4 per 128-block), which only ever touches scales 0..7.
    fn dot_q6_k_buggy(row_data: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut x_off = 0usize;
        for block in row_data.chunks_exact(Q6_K_BLOCK_BYTES) {
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
            let (mut ql_off, mut qh_off, mut ib) = (0usize, 0usize, 0usize);
            for _ in 0..(QK_K / 128) {
                for l in 0..32 {
                    let q1 = (((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32)
                        as f32;
                    let q2 = (((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4))
                        as i32
                        - 32) as f32;
                    let q3 = (((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32
                        - 32) as f32;
                    let q4 = (((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))
                        as i32
                        - 32) as f32;
                    acc += d * sc[ib] as i8 as f32 * q1 * x[x_off + l];
                    acc += d * sc[ib + 1] as i8 as f32 * q2 * x[x_off + l + 32];
                    acc += d * sc[ib + 2] as i8 as f32 * q3 * x[x_off + l + 64];
                    acc += d * sc[ib + 3] as i8 as f32 * q4 * x[x_off + l + 96];
                }
                x_off += 128;
                ql_off += 64;
                qh_off += 32;
                ib += 4;
            }
        }
        acc
    }

    // Buggy Q5_K dot: read the 5th bit from a single qh[is/8] byte per 32-element
    // sub-block (the shipped bug) instead of the per-element qh[l].
    fn dot_q5_k_buggy(row_data: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut x_off = 0usize;
        for block in row_data.chunks_exact(Q5_K_BLOCK_BYTES) {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales = &block[4..16];
            let qh = &block[16..48];
            let ql = &block[48..Q5_K_BLOCK_BYTES];
            let (mut ql_off, mut is) = (0usize, 0usize);
            let (mut u1, mut u2): (u8, u8) = (1, 2);
            for _ in 0..(QK_K / 64) {
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (d1, m1v) = (d * sc1 as f32, dmin * m1 as f32);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let (d2, m2v) = (d * sc2 as f32, dmin * m2 as f32);
                let qh_byte = qh[is / 8]; // BUG: one byte for all 32 elements
                for l in 0..32 {
                    let hi1 = if qh_byte & u1 != 0 { 16.0f32 } else { 0.0 };
                    let hi2 = if qh_byte & u2 != 0 { 16.0f32 } else { 0.0 };
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

    fn rel_err(got: f32, reference: f32) -> f32 {
        (got - reference).abs() / reference.abs().max(1e-6)
    }

    fn stats(v: &mut [f32]) -> (f32, f32, f32) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let median = v[v.len() / 2];
        let max = *v.last().unwrap();
        (mean, median, max)
    }

    #[test]
    fn corruption_magnitude_report() {
        // --- Q6_K: self-validate the reconstruction reproduces the documented 896.
        let mut canon = vec![0u8; Q6_K_BLOCK_BYTES];
        for b in canon.iter_mut().take(128) {
            *b = 0x11;
        }
        for b in canon.iter_mut().take(192).skip(128) {
            *b = 0xAA;
        }
        for j in 0..16 {
            canon[192 + j] = j as i8 as u8;
        }
        canon[208..210].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let xo = vec![1.0f32; QK_K];
        let buggy_canon = dot_q6_k_buggy(&canon, &xo);
        assert!(
            (buggy_canon - 896.0).abs() < 1e-3,
            "Q6_K bug reconstruction infidelity: got {buggy_canon}, expected documented 896"
        );
        let correct_canon = dot_q6_k_row_f32(&canon, &xo);
        println!("\n=== Corruption-magnitude benchmark (relative error vs verified reference) ===");
        println!(
            "[validate] Q6_K canonical block: correct={correct_canon} buggy={buggy_canon} \
             rel_err={:.4}",
            rel_err(buggy_canon, correct_canon)
        );

        // --- Q6_K: error distribution over 256 random super-blocks.
        let nblk = 256;
        let mut q6: Vec<f32> = (0..nblk)
            .map(|i| {
                let blk = rand_q6_k_block(0xC0DE_0000 + i as u64);
                let x = rand_x(0xBEEF_0000 + i as u64, QK_K);
                rel_err(dot_q6_k_buggy(&blk, &x), dot_q6_k_row_f32(&blk, &x))
            })
            .collect();
        let (m, md, mx) = stats(&mut q6);
        println!("Q6_K scale mis-index   (n={nblk}): mean={m:.3} median={md:.3} max={mx:.3}");

        // --- Q5_K: error distribution over 256 random super-blocks.
        let mut q5: Vec<f32> = (0..nblk)
            .map(|i| {
                let blk = rand_q5_k_block(0x5A5A_0000 + i as u64);
                let x = rand_x(0x1357_0000 + i as u64, QK_K);
                rel_err(dot_q5_k_buggy(&blk, &x), dot_q5_k_row_f32(&blk, &x))
            })
            .collect();
        let (m, md, mx) = stats(&mut q5);
        println!("Q5_K 5th-bit mis-index (n={nblk}): mean={m:.3} median={md:.3} max={mx:.3}");

        // --- Activation quantization: per-row vs per-block as the outlier grows.
        // Uses only the verified public kernels (no reconstruction).
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let k = 4096;
            let wf = rand_x(0xAAAA, k);
            let w_blocks = q8_0_weight_row(&wf);
            println!("Activation quant (Q8_0 W8A8, K={k}):  outlier   per-block   per-row");
            for &mag in &[1.0f32, 5.0, 10.0, 20.0, 40.0, 80.0] {
                let mut xf = rand_x(0xBBBB, k);
                xf[k / 2] = mag; // single outlier channel
                let reference = dot_q8_0_row_f32(&w_blocks, &xf);
                let (x_i8, x_sc) = quantize_row_to_i8_blocks(&xf);
                let block = unsafe { dot_q8_0_row_sdot(&w_blocks, &x_i8, &x_sc) };
                let max_abs = xf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let rs = max_abs / 127.0;
                let inv = 1.0 / rs;
                let x_row: Vec<i8> = xf
                    .iter()
                    .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8)
                    .collect();
                let perrow_sc = vec![rs; k / QK];
                let perrow = unsafe { dot_q8_0_row_sdot(&w_blocks, &x_row, &perrow_sc) };
                println!(
                    "  {:>5.0}x outlier:                {:>10.4} {:>10.4}",
                    mag,
                    rel_err(block, reference),
                    rel_err(perrow, reference)
                );
            }
        }
        println!("===========================================================================\n");
    }

    // Helper: quantize an f32 row into packed Q8_0 weight blocks.
    // Only used by the aarch64 SDOT test below (dead code on other arches).
    #[cfg(target_arch = "aarch64")]
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
        let xsums = i8_block_sums(&xi8);
        let q8_dot = dot_q4_k_row_q8_scalar(&row, &xi8, &xsc, &xsums);

        let rel = (f32_dot - q8_dot).abs() / f32_dot.abs().max(1e-3);
        assert!(
            rel < 0.03,
            "W4A8 mismatch: f32={f32_dot} q8={q8_dot} rel={rel}"
        );

        // The NEON SDOT kernel must match the scalar W4A8 reference exactly (same
        // integer dot; only f32 reduction order differs → tiny tolerance).
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let neon = unsafe { dot_q4_k_row_q8_neon(&row, &xi8, &xsc, &xsums) };
            let rel_n = (neon - q8_dot).abs() / q8_dot.abs().max(1e-3);
            assert!(
                rel_n < 1e-4,
                "NEON≠scalar W4A8: neon={neon} scalar={q8_dot}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q4_k_4rows_matches_single_row() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        // 8 pseudo-random Q4_K rows of k=512 (2 super-blocks each) + a random
        // activation, quantized to per-block i8 like the hot path does.
        let k = 512usize;
        let mut seed = 0x5EEDu64;
        let mut nb = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        let row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
        let mut rows = vec![0u8; 8 * row_bytes];
        for (i, b) in rows.iter_mut().enumerate() {
            *b = match i % Q4_K_BLOCK_BYTES {
                // small positive f16 d/dmin so magnitudes stay sane
                0 | 2 => 0x11,
                1 | 3 => 0x2c,
                _ => nb(),
            };
        }
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let (x_i8, x_scales) = quantize_row_to_i8_blocks(&x);
        let x_sums = i8_block_sums(&x_i8);

        for group in 0..2 {
            let j = group * 4;
            let r = |o: usize| &rows[(j + o) * row_bytes..(j + o + 1) * row_bytes];
            let got = unsafe {
                dot_q4_k_4rows_q8_neon([r(0), r(1), r(2), r(3)], &x_i8, &x_scales, &x_sums)
            };
            for (o, g) in got.iter().enumerate() {
                let want = unsafe { dot_q4_k_row_q8_neon(r(o), &x_i8, &x_scales, &x_sums) };
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "row {} differs: {g} vs {want}",
                    j + o
                );
            }
        }
    }

    #[test]
    fn q4_k_r4_repack_roundtrips_through_dequant() {
        // to_f32_vec on a repacked tensor must equal to_f32_vec on the original
        // (the de-interleave map is the inverse of the repack permutation).
        use sapient_core::{DType, Tensor};
        let (n, k) = (8usize, 512usize);
        let row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
        let mut seed = 0x00D5u64;
        let blocks: Vec<u8> = (0..n * row_bytes)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % Q4_K_BLOCK_BYTES {
                    0 | 2 => 0x11,
                    1 | 3 => 0x2c,
                    _ => (seed >> 33) as u8,
                }
            })
            .collect();
        let orig = Tensor::from_quant_bytes(&blocks, vec![n, k], DType::Q4_K).unwrap();
        let packed = repack_q4_k_rows4(&blocks, n, k);
        let r4 = Tensor::from_quant_bytes(&packed, vec![n, k], DType::Q4_K_R4).unwrap();
        assert_eq!(orig.to_f32_vec(), r4.to_f32_vec());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q4_k_r4_kernel_matches_single_row() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let (n, k) = (4usize, 512usize);
        let row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
        let mut seed = 0x0B0Bu64;
        let blocks: Vec<u8> = (0..n * row_bytes)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % Q4_K_BLOCK_BYTES {
                    0 | 2 => 0x11,
                    1 | 3 => 0x2c,
                    _ => (seed >> 33) as u8,
                }
            })
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 53 % 89) as f32 - 44.0) * 0.02)
            .collect();
        let (x_i8, x_scales) = quantize_row_to_i8_blocks(&x);
        let x_sums = i8_block_sums(&x_i8);
        let packed = repack_q4_k_rows4(&blocks, n, k);
        let got = unsafe { dot_q4_k_4rows_r4_neon(&packed, &x_i8, &x_scales, &x_sums) };
        for (r, g) in got.iter().enumerate() {
            let want = unsafe {
                dot_q4_k_row_q8_neon(
                    &blocks[r * row_bytes..(r + 1) * row_bytes],
                    &x_i8,
                    &x_scales,
                    &x_sums,
                )
            };
            assert_eq!(g.to_bits(), want.to_bits(), "row {r}: {g} vs {want}");
        }
    }

    #[test]
    fn q6_k_r4_repack_roundtrips_through_dequant() {
        use sapient_core::{DType, Tensor};
        let (n, k) = (8usize, 512usize);
        let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
        let mut seed = 0x6B6Bu64;
        let blocks: Vec<u8> = (0..n * row_bytes)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % Q6_K_BLOCK_BYTES {
                    208 => 0x11, // small positive f16 d (low byte)
                    209 => 0x2c, // (high byte)
                    _ => (seed >> 33) as u8,
                }
            })
            .collect();
        let orig = Tensor::from_quant_bytes(&blocks, vec![n, k], DType::Q6_K).unwrap();
        let packed = repack_q6_k_rows4(&blocks, n, k);
        let r4 = Tensor::from_quant_bytes(&packed, vec![n, k], DType::Q6_K_R4).unwrap();
        assert_eq!(orig.to_f32_vec(), r4.to_f32_vec());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q6_k_r4_kernel_matches_single_row() {
        let (n, k) = (4usize, 512usize);
        let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
        let mut seed = 0x6666u64;
        let blocks: Vec<u8> = (0..n * row_bytes)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % Q6_K_BLOCK_BYTES {
                    208 => 0x11,
                    209 => 0x2c,
                    _ => (seed >> 33) as u8,
                }
            })
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 41 % 83) as f32 - 41.0) * 0.02)
            .collect();
        let packed = repack_q6_k_rows4(&blocks, n, k);
        let got = unsafe { dot_q6_k_4rows_r4_neon(&packed, &x) };
        for (r, g) in got.iter().enumerate() {
            let want =
                unsafe { dot_q6_k_row_f32_neon(&blocks[r * row_bytes..(r + 1) * row_bytes], &x) };
            assert_eq!(g.to_bits(), want.to_bits(), "row {r}: {g} vs {want}");
        }
    }

    fn q6_k_test_rows(n: usize, k: usize, seed0: u64) -> Vec<u8> {
        let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
        let mut seed = seed0;
        (0..n * row_bytes)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % Q6_K_BLOCK_BYTES {
                    208 => 0x11,
                    209 => 0x2c,
                    _ => (seed >> 33) as u8,
                }
            })
            .collect()
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q6_k_w6a8_neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let k = 512usize;
        let rows = q6_k_test_rows(2, k, 0x0666);
        let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 29 % 71) as f32 - 35.0) * 0.03)
            .collect();
        let (x_i8, x_scales) = quantize_row_to_i8_blocks(&x);
        for r in 0..2 {
            let row = &rows[r * row_bytes..(r + 1) * row_bytes];
            let want = dot_q6_k_row_q8_scalar(row, &x_i8, &x_scales);
            let got = unsafe { dot_q6_k_row_q8_neon(row, &x_i8, &x_scales) };
            assert_eq!(got.to_bits(), want.to_bits(), "row {r}: {got} vs {want}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q6_k_w6a8_r4_matches_single_row() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let (n, k) = (4usize, 512usize);
        let rows = q6_k_test_rows(n, k, 0x0667);
        let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 31 % 67) as f32 - 33.0) * 0.03)
            .collect();
        let (x_i8, x_scales) = quantize_row_to_i8_blocks(&x);
        let packed = repack_q6_k_rows4(&rows, n, k);
        let got = unsafe { dot_q6_k_4rows_r4_q8_neon(&packed, &x_i8, &x_scales) };
        for (r, g) in got.iter().enumerate() {
            let want = unsafe {
                dot_q6_k_row_q8_neon(&rows[r * row_bytes..(r + 1) * row_bytes], &x_i8, &x_scales)
            };
            assert_eq!(g.to_bits(), want.to_bits(), "row {r}: {g} vs {want}");
        }
    }

    #[test]
    fn q6_k_w6a8_close_to_f32_path() {
        // Activation quantization is per-32-block int8 — same accuracy class as
        // the accepted Q4_K W4A8 path. Bound the relative error vs the exact
        // f32-activation dot.
        let k = 512usize;
        let rows = q6_k_test_rows(1, k, 0x0668);
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 43 % 91) as f32 - 45.0) * 0.02)
            .collect();
        let (x_i8, x_scales) = quantize_row_to_i8_blocks(&x);
        let exact = dot_q6_k_row_f32(&rows, &x);
        let w6a8 = dot_q6_k_row_q8_scalar(&rows, &x_i8, &x_scales);
        let rel = (w6a8 - exact).abs() / exact.abs().max(1e-3);
        assert!(rel < 2e-2, "W6A8 vs f32: {w6a8} vs {exact} (rel {rel})");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q4_k_smmla_x2_matches_single_row() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }
        let (n, k) = (4usize, 512usize);
        let row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
        let mut seed = 0x18AAu64;
        let rows: Vec<u8> = (0..n * row_bytes)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % Q4_K_BLOCK_BYTES {
                    0 | 2 => 0x11,
                    1 | 3 => 0x2c,
                    _ => (seed >> 33) as u8,
                }
            })
            .collect();
        let x0: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let x1: Vec<f32> = (0..k)
            .map(|i| ((i * 59 % 101) as f32 - 50.0) * 0.015)
            .collect();
        let (x0_i8, x0_s) = quantize_row_to_i8_blocks(&x0);
        let (x1_i8, x1_s) = quantize_row_to_i8_blocks(&x1);
        let x0_b = i8_block_sums(&x0_i8);
        let x1_b = i8_block_sums(&x1_i8);
        let packed = repack_q4_k_rows4(&rows, n, k);
        let got = unsafe {
            dot_q4_k_4rows_r4_x2_smmla(&packed, &x0_i8, &x0_s, &x0_b, &x1_i8, &x1_s, &x1_b)
        };
        for r in 0..4 {
            let row = &rows[r * row_bytes..(r + 1) * row_bytes];
            let w0 = unsafe { dot_q4_k_row_q8_neon(row, &x0_i8, &x0_s, &x0_b) };
            let w1 = unsafe { dot_q4_k_row_q8_neon(row, &x1_i8, &x1_s, &x1_b) };
            assert_eq!(
                got[r][0].to_bits(),
                w0.to_bits(),
                "row {r} x0: {} vs {w0}",
                got[r][0]
            );
            assert_eq!(
                got[r][1].to_bits(),
                w1.to_bits(),
                "row {r} x1: {} vs {w1}",
                got[r][1]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q6_k_smmla_x2_matches_single_row() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }
        let (n, k) = (4usize, 512usize);
        let rows = q6_k_test_rows(n, k, 0x68AA);
        let row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
        let x0: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let x1: Vec<f32> = (0..k)
            .map(|i| ((i * 61 % 103) as f32 - 51.0) * 0.015)
            .collect();
        let (x0_i8, x0_s) = quantize_row_to_i8_blocks(&x0);
        let (x1_i8, x1_s) = quantize_row_to_i8_blocks(&x1);
        let packed = repack_q6_k_rows4(&rows, n, k);
        let got = unsafe { dot_q6_k_4rows_r4_x2_smmla(&packed, &x0_i8, &x0_s, &x1_i8, &x1_s) };
        for r in 0..4 {
            let row = &rows[r * row_bytes..(r + 1) * row_bytes];
            let w0 = unsafe { dot_q6_k_row_q8_neon(row, &x0_i8, &x0_s) };
            let w1 = unsafe { dot_q6_k_row_q8_neon(row, &x1_i8, &x1_s) };
            assert_eq!(
                got[r][0].to_bits(),
                w0.to_bits(),
                "row {r} x0: {} vs {w0}",
                got[r][0]
            );
            assert_eq!(
                got[r][1].to_bits(),
                w1.to_bits(),
                "row {r} x1: {} vs {w1}",
                got[r][1]
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
    #[test]
    fn q4_k_q8k_scalar_matches_f32_path() {
        let nblocks = 3usize;
        let mut row = vec![0u8; nblocks * Q4_K_BLOCK_BYTES];
        for (i, b) in row.iter_mut().enumerate() {
            *b = ((i * 197 + 13) % 251) as u8;
        }
        for blk in 0..nblocks {
            let base = blk * Q4_K_BLOCK_BYTES;
            row[base..base + 2].copy_from_slice(&f16::from_f32(0.05).to_le_bytes());
            row[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.03).to_le_bytes());
        }
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let x: Vec<f32> = (0..nblocks * QK_K)
            .map(|_| (next() as f32 / u64::MAX as f32) * 4.0 - 2.0)
            .collect();

        let f32_dot = dot_q4_k_row_f32(&row, &x);
        let (xi8, xsc, xsum) = quantize_row_to_q8k(&x);
        let q8k_dot = dot_q4_k_row_q8k_scalar(&row, &xi8, &xsc, &xsum);
        let rel = (f32_dot - q8k_dot).abs() / f32_dot.abs().max(1e-3);
        assert!(
            rel < 0.03,
            "Q8_K mismatch: f32={f32_dot} q8k={q8k_dot} rel={rel}"
        );

        // The per-256 format must stay in the accuracy class of the accepted
        // per-32 W4A8 path on the same inputs.
        let (pi8, psc) = quantize_row_to_i8_blocks(&x);
        let psum = i8_block_sums(&pi8);
        let w4a8 = dot_q4_k_row_q8_scalar(&row, &pi8, &psc, &psum);
        let rel_vs = (w4a8 - q8k_dot).abs() / w4a8.abs().max(1e-3);
        assert!(
            rel_vs < 0.03,
            "Q8_K vs W4A8 divergence: w4a8={w4a8} q8k={q8k_dot} rel={rel_vs}"
        );

        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let neon = unsafe { dot_q4_k_row_q8k_neon(&row, &xi8, &xsc, &xsum) };
            assert_eq!(
                neon.to_bits(),
                q8k_dot.to_bits(),
                "NEON≠scalar Q8_K: {neon} vs {q8k_dot}"
            );
        }
    }
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q4_k_r4_q8k_kernels_match_single_row() {
        let n = 4usize;
        let k = 512usize;
        let row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
        let mut rows = vec![0u8; n * row_bytes];
        for (i, b) in rows.iter_mut().enumerate() {
            *b = ((i * 149 + 29) % 249) as u8;
        }
        for r in 0..n {
            for blk in 0..k / QK_K {
                let base = r * row_bytes + blk * Q4_K_BLOCK_BYTES;
                rows[base..base + 2].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
                rows[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
            }
        }
        let x0: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let x1: Vec<f32> = (0..k)
            .map(|i| ((i * 59 % 101) as f32 - 50.0) * 0.015)
            .collect();
        let (q0, s0, b0) = quantize_row_to_q8k(&x0);
        let (q1, s1, b1) = quantize_row_to_q8k(&x1);
        let packed = repack_q4_k_rows4(&rows, n, k);

        // 4-row R4 kernel vs single-row Q8_K kernel, exact bits.
        let got4 = unsafe { dot_q4_k_4rows_r4_q8k_neon(&packed, &q0, &s0, &b0) };
        for (r, g) in got4.iter().enumerate() {
            let row = &rows[r * row_bytes..(r + 1) * row_bytes];
            let want = unsafe { dot_q4_k_row_q8k_neon(row, &q0, &s0, &b0) };
            assert_eq!(g.to_bits(), want.to_bits(), "r4 row {r}: {g} vs {want}");
        }

        // SMMLA x2 kernel vs single-row, exact bits over both x rows.
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            let got =
                unsafe { dot_q4_k_4rows_r4_x2_q8k_smmla(&packed, &q0, &s0, &b0, &q1, &s1, &b1) };
            for r in 0..4 {
                let row = &rows[r * row_bytes..(r + 1) * row_bytes];
                let w0 = unsafe { dot_q4_k_row_q8k_neon(row, &q0, &s0, &b0) };
                let w1 = unsafe { dot_q4_k_row_q8k_neon(row, &q1, &s1, &b1) };
                assert_eq!(got[r][0].to_bits(), w0.to_bits(), "smmla row {r} x0");
                assert_eq!(got[r][1].to_bits(), w1.to_bits(), "smmla row {r} x1");
            }
        }
    }
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn q4_k_plain_4rows_q8k_matches_single_row() {
        let n = 4usize;
        let k = 512usize;
        let row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
        let mut rows = vec![0u8; n * row_bytes];
        for (i, b) in rows.iter_mut().enumerate() {
            *b = ((i * 167 + 43) % 247) as u8;
        }
        for r in 0..n {
            for blk in 0..k / QK_K {
                let base = r * row_bytes + blk * Q4_K_BLOCK_BYTES;
                rows[base..base + 2].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
                rows[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 41 % 103) as f32 - 51.0) * 0.02)
            .collect();
        let (q, sc, sm) = quantize_row_to_q8k(&x);
        let r = |o: usize| &rows[o * row_bytes..(o + 1) * row_bytes];
        let got = unsafe { dot_q4_k_4rows_q8k_neon([r(0), r(1), r(2), r(3)], &q, &sc, &sm) };
        for (o, g) in got.iter().enumerate() {
            let want = unsafe { dot_q4_k_row_q8k_neon(r(o), &q, &sc, &sm) };
            assert_eq!(g.to_bits(), want.to_bits(), "row {o}: {g} vs {want}");
        }
    }
}
