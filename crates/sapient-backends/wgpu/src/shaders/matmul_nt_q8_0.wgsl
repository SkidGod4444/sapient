// Quantized linear projection: out[M, N] = x[M, K] @ dequant(W)[N, K]^T
//
// W is Q8_0-quantized: each contiguous block of 32 weights shares one f32 scale,
// and each weight is an int8. dequant(w_i) = scale_block * f32(w_i).
//
// Host-side repack (wgpu-friendly, exact):
//   scales : array<f32>  length N * (K/32)        — one scale per (row, block)
//   qw     : array<u32>  length N * (K/32) * 8     — 32 int8 per block → 8 u32 words,
//            int8 values packed little-endian, 4 per u32.
//
// This mirrors the CPU `dot_q8_0` reference (sum of x_i * scale * i8_i) exactly when
// x is f32 and the weights were quantized with the same scale, so GPU output matches
// the host reference bit-for-bit (modulo float add ordering).

@group(0) @binding(0) var<storage, read>       x:      array<f32>;  // [M*K]
@group(0) @binding(1) var<storage, read>       qw:     array<u32>;  // packed int8, 8 u32/block
@group(0) @binding(2) var<storage, read>       scales: array<f32>;  // [N * nblocks]
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;  // [M*N]
@group(0) @binding(4) var<uniform>             dims:   vec4<u32>;   // (M, K, N, _)

// Extract the `idx`-th int8 (0..3) from a u32 word, sign-extended to f32.
fn unpack_i8(word: u32, idx: u32) -> f32 {
    let byte = (word >> (idx * 8u)) & 0xFFu;
    // Shift into the top byte then arithmetic-shift right to sign-extend.
    let val = i32(byte << 24u) >> 24u;
    return f32(val);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    let n = gid.y;
    let big_m = dims.x;
    let k = dims.y;
    let big_n = dims.z;
    if (m >= big_m || n >= big_n) {
        return;
    }

    let nblocks = k / 32u;
    let x_base = m * k;
    var acc = 0.0;

    for (var b = 0u; b < nblocks; b = b + 1u) {
        let scale = scales[n * nblocks + b];
        let w_word_base = (n * nblocks + b) * 8u;
        let x_block_base = x_base + b * 32u;
        for (var w4 = 0u; w4 < 8u; w4 = w4 + 1u) {
            let word = qw[w_word_base + w4];
            let xb = x_block_base + w4 * 4u;
            acc = acc + x[xb + 0u] * (unpack_i8(word, 0u) * scale);
            acc = acc + x[xb + 1u] * (unpack_i8(word, 1u) * scale);
            acc = acc + x[xb + 2u] * (unpack_i8(word, 2u) * scale);
            acc = acc + x[xb + 3u] * (unpack_i8(word, 3u) * scale);
        }
    }
    out[m * big_n + n] = acc;
}
