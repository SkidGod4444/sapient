// Resident quantized linear projection: out[M,N] = x[M,K] @ dequant(W)[N,K]^T.
// W stays Q8_0 on the GPU — int8 weights dequantized inside the kernel, no host-side
// f32 expansion (Phase 7.1). Layout produced by `upload_q8_0` from raw ggml blocks:
//   qs     : array<u32>  [N * K/4]  — int8 weights, 4 per word, little-endian lanes
//   scales : array<f32>  [N * K/32] — one f32 scale per 32-weight block (f16 in ggml,
//            widened once on upload; dequant value scale*i8 is exactly the CPU's)
// One workgroup per output element; 256 lanes cooperatively reduce the K dot product
// (GEMV-style, same shape as matmul_nt.wgsl). f32 accumulation.
// Index is 2-D-tiled: idx = wg.x + wg.y*num_workgroups.x (handles N>65535, e.g. lm_head).

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       qs:     array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;
@group(0) @binding(4) var<uniform>             p:      P;

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
    let words = p.k / 4u;          // u32 words per weight row (k % 32 == 0)
    let xb = rm * p.k;
    let wb = rn * words;
    let sb = rn * (p.k / 32u);
    let tid = lid.x;

    var acc = 0.0;
    var wi = tid;
    loop {
        if (wi >= words) { break; }
        let word = qs[wb + wi];
        let scale = scales[sb + (wi >> 3u)]; // 8 words per 32-weight block
        let xi = xb + wi * 4u;
        // Vectorized dequant: unpack4x8snorm = q/127 per byte lane (Q8_0 quants
        // never reach -128, so the snorm clamp is a no-op) — fold the 127 back
        // into the scale and reduce with one hardware dot per 4 weights.
        let w4 = unpack4x8snorm(word);
        let xv = vec4<f32>(x[xi], x[xi + 1u], x[xi + 2u], x[xi + 3u]);
        acc = acc + (scale * 127.0) * dot(w4, xv);
        wi = wi + WG;
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
