// Resident linear projection: out[M,N] = x[M,K] @ w[N,K]^T  (w in HF [N,K] layout).
// One workgroup per output element; 256 lanes cooperatively reduce the K dot product
// (GEMV-style — ideal for batch-1 decode). f32 accumulation.
// Index is 2-D-tiled: idx = wg.x + wg.y*num_workgroups.x (handles N>65535, e.g. lm_head).

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       w:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = wg.x + wg.y * nwg.x;
    if (idx >= p.m * p.n) { return; }
    let rm = idx / p.n;
    let rn = idx % p.n;
    let xb = rm * p.k;
    let wb = rn * p.k;
    let tid = lid.x;

    var acc = 0.0;
    var i = tid;
    loop {
        if (i >= p.k) { break; }
        acc = acc + x[xb + i] * w[wb + i];
        i = i + WG;
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
