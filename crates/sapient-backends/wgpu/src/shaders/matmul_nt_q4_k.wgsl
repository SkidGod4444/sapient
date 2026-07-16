// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

// Resident Q4_K linear projection: out[M,N] = x[M,K] @ dequant(W)[N,K]^T (Phase 7.2).
// W stays in raw ggml Q4_K super-blocks on the GPU — 144 bytes per 256 weights,
// uploaded verbatim (144 is word-aligned, so no host repack at all):
//   word 0        : d (f16, low half) | dmin (f16, high half)
//   words 1..3    : scales[12] — 8 pairs of 6-bit (scale, min), ggml get_scale_min_k4
//   words 4..35   : qs[128]    — 256 × 4-bit quants (low nibbles = first 32 of each
//                   64-group, high nibbles = second 32)
// dequant(e) = d·sc(is)·q4(e) − dmin·mn(is), with sub-block is = e/32 (0..7).
// Per sub-block the kernel accumulates Σx·q and Σx separately, then applies the
// affine pair once — same math as the CPU dot_q4_k_row_f32_scalar, so values match
// the CPU dequant exactly (modulo float add order).
// One workgroup per output element; 256 lanes stride over the K/32 sub-blocks.
// f32 accumulation; 2-D-tiled dispatch (idx = wg.x + wg.y*num_workgroups.x).

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       qb:  array<u32>; // 36 words / super-block
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

fn byte_of(word: u32, b: u32) -> u32 { return (word >> (b * 8u)) & 0xFFu; }

// Byte i (0..11) of the 12-byte scales field of the super-block at word `blk`.
fn scale_byte(blk: u32, i: u32) -> u32 { return byte_of(qb[blk + 1u + i / 4u], i % 4u); }

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = wg.x + wg.y * nwg.x;
    if (idx >= p.m * p.n) { return; }
    let rm = idx / p.n;
    let rn = idx % p.n;
    let nblocks = p.k / 256u;          // super-blocks per weight row (k % 256 == 0)
    let row_base = rn * nblocks * 36u; // word offset of this row's first block
    let xb = rm * p.k;
    let tid = lid.x;

    var acc = 0.0;
    var sub = tid;                     // 32-weight sub-block index within the row
    let nsub = p.k / 32u;
    loop {
        if (sub >= nsub) { break; }
        let b = sub / 8u;              // super-block
        let is = sub % 8u;             // sub-block within it (0..7)
        let blk = row_base + b * 36u;
        let dm = unpack2x16float(qb[blk]); // (d, dmin)

        // ggml get_scale_min_k4(is, scales)
        var sc: u32;
        var mn: u32;
        if (is < 4u) {
            sc = scale_byte(blk, is) & 63u;
            mn = scale_byte(blk, is + 4u) & 63u;
        } else {
            sc = (scale_byte(blk, is + 4u) & 0xFu) | ((scale_byte(blk, is - 4u) >> 6u) << 4u);
            mn = (scale_byte(blk, is + 4u) >> 4u) | ((scale_byte(blk, is) >> 6u) << 4u);
        }

        // Sub-block is reads qs bytes (is/2)*32 .. +32, low nibble when is is even,
        // high when odd; its 32 activations are contiguous at e = is*32.
        let qw0 = blk + 4u + (is / 2u) * 8u;
        let shift = (is % 2u) * 4u;
        let xoff = xb + b * 256u + is * 32u;
        // Vectorized dequant: shift+mask drops this sub-block's nibbles into the
        // four byte lanes; unpack4x8unorm reads them as q/255, so the 255 folds
        // into the d·sc scale below. Σx uses a dot against ONES.
        var sum_q = 0.0; // Σ x·(q/255)
        var sum_x = 0.0; // Σ x
        for (var w = 0u; w < 8u; w = w + 1u) {
            let nib = (qb[qw0 + w] >> shift) & 0x0F0F0F0Fu;
            let q4 = unpack4x8unorm(nib);
            let xi = xoff + w * 4u;
            let xv = vec4<f32>(x[xi], x[xi + 1u], x[xi + 2u], x[xi + 3u]);
            sum_q = sum_q + dot(q4, xv);
            sum_x = sum_x + dot(xv, vec4<f32>(1.0));
        }
        acc = acc + dm.x * f32(sc) * 255.0 * sum_q - dm.y * f32(mn) * sum_x;
        sub = sub + WG;
    }
    partial[tid] = acc;
    workgroupBarrier();

    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { out[idx] = partial[0]; }
}
