// Multi-row f32 linear projection (prefill): out[M,N] = x[M,K] @ w[N,K]^T with
// MT = 8 x-rows per workgroup. Each workgroup owns ONE weight row and reads it
// once, FMA-ing into 8 accumulators — weight traffic per prefill chunk drops
// ~8× vs launching the single-row GEMV per output element (which re-reads every
// weight row m times). Dispatched for m > 1; decode (m = 1) keeps matmul_nt.wgsl.
// Tail tiles clamp their x-row index (reads stay in bounds, results discarded)
// and mask the store. f32 accumulation; grid = n × ceil(m/MT), 2-D-tiled.

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       w:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

const WG: u32 = 256u;
const MT: u32 = 8u;
var<workgroup> partial: array<f32, 2048>; // [MT][WG]

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = wg.x + wg.y * nwg.x;
    let tiles = (p.m + MT - 1u) / MT;
    if (idx >= p.n * tiles) { return; }
    let rn = idx % p.n;
    let m0 = (idx / p.n) * MT;
    let wb = rn * p.k;
    let tid = lid.x;

    var xb: array<u32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) {
        xb[t] = min(m0 + t, p.m - 1u) * p.k;
    }
    var acc: array<f32, 8>;
    for (var t = 0u; t < MT; t = t + 1u) { acc[t] = 0.0; }

    var i = tid;
    loop {
        if (i >= p.k) { break; }
        let wv = w[wb + i];
        for (var t = 0u; t < MT; t = t + 1u) {
            acc[t] = acc[t] + wv * x[xb[t] + i];
        }
        i = i + WG;
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
