// Multi-row Q8_0 linear projection (prefill): MT = 8 x-rows per workgroup over
// one Q8_0 weight row — each 4-weight word is dequantized ONCE and FMA-ed into
// 8 accumulators, so both weight traffic and dequant ALU drop ~8× per prefill
// chunk vs the single-row GEMV. Same `upload_q8_0` layout as matmul_nt_q8_0.wgsl
// (qs: 4 int8 per u32 word, scales: one f32 per 32-weight block). Dispatched for
// m > 1; decode keeps the single-row kernel. f32 accumulation.

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       qs:     array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;
@group(0) @binding(4) var<uniform>             p:      P;

const WG: u32 = 256u;
const MT: u32 = 8u;
var<workgroup> partial: array<f32, 2048>; // [MT][WG]

fn unpack_i8(word: u32, lane: u32) -> f32 {
    let byte = (word >> (lane * 8u)) & 0xFFu;
    return f32(i32(byte << 24u) >> 24u);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = wg.x + wg.y * nwg.x;
    let tiles = (p.m + MT - 1u) / MT;
    if (idx >= p.n * tiles) { return; }
    let rn = idx % p.n;
    let m0 = (idx / p.n) * MT;
    let words = p.k / 4u;
    let wb = rn * words;
    let sb = rn * (p.k / 32u);
    let tid = lid.x;

    var xb: array<u32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) {
        xb[t] = min(m0 + t, p.m - 1u) * p.k;
    }
    var acc: array<f32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) { acc[t] = 0.0; }

    var wi = tid;
    loop {
        if (wi >= words) { break; }
        let word = qs[wb + wi];
        let scale = scales[sb + (wi >> 3u)];
        // Dequantize the 4 weights ONCE …
        let w0 = scale * unpack_i8(word, 0u);
        let w1 = scale * unpack_i8(word, 1u);
        let w2 = scale * unpack_i8(word, 2u);
        let w3 = scale * unpack_i8(word, 3u);
        let xi = wi * 4u;
        // … and reuse them across all MT rows.
        for (var t = 0u; t < MT; t = t + 1u) {
            let b = xb[t] + xi;
            acc[t] = acc[t] + w0 * x[b] + w1 * x[b + 1u] + w2 * x[b + 2u] + w3 * x[b + 3u];
        }
        wi = wi + WG;
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
