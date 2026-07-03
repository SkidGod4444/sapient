// Multi-row Q6_K linear projection (prefill): MT = 8 x-rows per workgroup over
// one Q6_K weight row. Per 16-weight scale-group the signed int8 scale and the
// 4+2-bit quants are decoded ONCE and applied to all 8 rows — weight traffic
// and dequant ALU drop ~8× per prefill chunk. Same padded-super-block layout
// and +0/+2/+4/+6 per-half scale indexing as matmul_nt_q6_k.wgsl (the
// historical token-salad bug class — change nothing without the random-bit
// reference tests). Dispatched for m > 1; decode keeps the single-row kernel.

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       qb:  array<u32>; // 53 words / super-block
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

const WG: u32 = 256u;
const MT: u32 = 8u;
var<workgroup> partial: array<f32, 2048>; // [MT][WG]

fn byte_of(word: u32, b: u32) -> u32 { return (word >> (b * 8u)) & 0xFFu; }

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
    let row_base = rn * nblocks * 53u;
    let tid = lid.x;

    var xb: array<u32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) {
        xb[t] = min(m0 + t, p.m - 1u) * p.k;
    }
    var acc: array<f32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) { acc[t] = 0.0; }

    var sg = tid; // 16-weight scale-group index within the row
    let nsg = p.k / 16u;
    loop {
        if (sg >= nsg) { break; }
        let b = sg / 16u;
        let s16 = sg % 16u;
        let h = s16 >> 3u;
        let g = (s16 & 7u) >> 1u;
        let l16 = s16 & 1u;
        let blk = row_base + b * 53u;

        let d = unpack2x16float(qb[blk + 52u]).x;
        let sc_raw = byte_of(qb[blk + 48u + (s16 >> 2u)], s16 & 3u);
        let sc = f32(i32(sc_raw << 24u) >> 24u);

        let ql0 = (h * 64u + (g & 1u) * 32u + l16 * 16u) >> 2u;
        let qh0 = (h * 32u + l16 * 16u) >> 2u;
        let shift4 = (g >> 1u) * 4u;
        let qh_shift = g * 2u;
        let eoff = b * 256u + h * 128u + g * 32u + l16 * 16u;

        var sum_q: array<f32, 8>;
        for (var t = 0u; t < MT; t = t + 1u) { sum_q[t] = 0.0; }
        for (var w = 0u; w < 4u; w = w + 1u) {
            let qlw = qb[blk + ql0 + w];
            let qhw = qb[blk + 32u + qh0 + w];
            let xi = eoff + w * 4u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                // Decode each quant ONCE …
                let q = ((byte_of(qlw, j) >> shift4) & 0xFu)
                      | (((byte_of(qhw, j) >> qh_shift) & 3u) << 4u);
                let qv = f32(q) - 32.0;
                // … and reuse across all MT rows.
                for (var t = 0u; t < MT; t = t + 1u) {
                    sum_q[t] = sum_q[t] + x[xb[t] + xi + j] * qv;
                }
            }
        }
        for (var t = 0u; t < MT; t = t + 1u) {
            acc[t] = acc[t] + d * sc * sum_q[t];
        }
        sg = sg + WG;
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
