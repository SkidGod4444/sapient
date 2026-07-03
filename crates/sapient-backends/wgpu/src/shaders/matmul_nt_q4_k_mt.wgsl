// Multi-row Q4_K linear projection (prefill): MT = 8 x-rows per workgroup over
// one Q4_K weight row. Per 32-weight sub-block the 6-bit scale/min pair and the
// 4-bit quants are decoded ONCE and applied to all 8 rows (per-row Σx·q and Σx,
// then the affine pair once per row) — weight traffic and dequant ALU drop ~8×
// per prefill chunk. Same raw-super-block layout and get_scale_min_k4 indexing
// as matmul_nt_q4_k.wgsl (change nothing there without the random-bit reference
// tests). Dispatched for m > 1; decode keeps the single-row kernel.

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       qb:  array<u32>; // 36 words / super-block
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

const WG: u32 = 256u;
const MT: u32 = 8u;
var<workgroup> partial: array<f32, 2048>; // [MT][WG]

fn byte_of(word: u32, b: u32) -> u32 { return (word >> (b * 8u)) & 0xFFu; }
fn scale_byte(blk: u32, i: u32) -> u32 { return byte_of(qb[blk + 1u + i / 4u], i % 4u); }

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = wg.x + wg.y * nwg.x;
    let tiles = (p.m + MT - 1u) / MT;
    if (idx >= p.n * tiles) { return; }
    let rn = idx % p.n;
    let m0 = (idx / p.n) * MT;
    let nblocks = p.k / 256u;
    let row_base = rn * nblocks * 36u;
    let tid = lid.x;

    var xb: array<u32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) {
        xb[t] = min(m0 + t, p.m - 1u) * p.k;
    }
    var acc: array<f32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) { acc[t] = 0.0; }

    var sub = tid; // 32-weight sub-block index within the row
    let nsub = p.k / 32u;
    loop {
        if (sub >= nsub) { break; }
        let b = sub / 8u;
        let is = sub % 8u;
        let blk = row_base + b * 36u;
        let dm = unpack2x16float(qb[blk]); // (d, dmin)

        var sc: u32;
        var mn: u32;
        if (is < 4u) {
            sc = scale_byte(blk, is) & 63u;
            mn = scale_byte(blk, is + 4u) & 63u;
        } else {
            sc = (scale_byte(blk, is + 4u) & 0xFu) | ((scale_byte(blk, is - 4u) >> 6u) << 4u);
            mn = (scale_byte(blk, is + 4u) >> 4u) | ((scale_byte(blk, is) >> 6u) << 4u);
        }

        let qw0 = blk + 4u + (is / 2u) * 8u;
        let shift = (is % 2u) * 4u;
        let eoff = b * 256u + is * 32u; // element offset of this sub-block
        var sum_q: array<f32, 8>;
        var sum_x: array<f32, 8>;
        for (var t = 0u; t < MT; t = t + 1u) { sum_q[t] = 0.0; sum_x[t] = 0.0; }
        for (var w = 0u; w < 8u; w = w + 1u) {
            let word = qb[qw0 + w];
            // Decode the 4 quants ONCE …
            let q0 = f32((byte_of(word, 0u) >> shift) & 0xFu);
            let q1 = f32((byte_of(word, 1u) >> shift) & 0xFu);
            let q2 = f32((byte_of(word, 2u) >> shift) & 0xFu);
            let q3 = f32((byte_of(word, 3u) >> shift) & 0xFu);
            let xi = eoff + w * 4u;
            // … and reuse across all MT rows.
            for (var t = 0u; t < MT; t = t + 1u) {
                let bx = xb[t] + xi;
                let x0 = x[bx];
                let x1 = x[bx + 1u];
                let x2 = x[bx + 2u];
                let x3 = x[bx + 3u];
                sum_q[t] = sum_q[t] + x0 * q0 + x1 * q1 + x2 * q2 + x3 * q3;
                sum_x[t] = sum_x[t] + x0 + x1 + x2 + x3;
            }
        }
        for (var t = 0u; t < MT; t = t + 1u) {
            acc[t] = acc[t] + dm.x * f32(sc) * sum_q[t] - dm.y * f32(mn) * sum_x[t];
        }
        sub = sub + WG;
    }

    for (var t = 0u; t < MT; t = t + 1u) { partial[t * WG + tid] = acc[t]; }
    workgroupBarrier();
    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            for (var t = 0u; t < MT; t = t + 1u) {
                partial[t * WG + tid] = partial[t * WG + tid] + partial[t * WG + tid + s];
            }
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid < MT && m0 + tid < p.m) {
        out[(m0 + tid) * p.n + rn] = partial[tid * WG];
    }
}
