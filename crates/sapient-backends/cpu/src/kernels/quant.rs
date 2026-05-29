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

/// Dot product of one Q4_0 block with a length-`QK` f32 activation slice,
/// dequantizing on the fly — never materializing the full weight row.
#[inline]
pub fn dot_q4_0_block_f32(block: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(block.len(), Q4_0_BLOCK_BYTES);
    debug_assert_eq!(x.len(), QK);
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

/// Dot product of a full Q4_0-quantized weight row (`k` elements, `k % QK == 0`,
/// stored as consecutive Q4_0 blocks) with an f32 activation vector.
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
