// LayerNorm: out[r,i] = (x[r,i] - mean_r) / sqrt(var_r + eps) * weight[i] + bias[i]
// One workgroup per row; cooperative mean + variance reduction in shared memory.
// f32 accumulation throughout. Unlike RMSNorm this subtracts the row mean and
// adds a per-channel bias — Whisper (and GPT/BERT-family) use this form.

struct Params {
    dim:  u32,
    rows: u32,
    eps:  f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read>       bias:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;
@group(0) @binding(4) var<uniform>             p:      Params;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    if (row >= p.rows) { return; }
    let base = row * p.dim;
    let tid = lid.x;

    // --- Pass 1: mean ---
    var s = 0.0;
    var i = tid;
    loop {
        if (i >= p.dim) { break; }
        s = s + x[base + i];
        i = i + WG;
    }
    partial[tid] = s;
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = partial[0] / f32(p.dim);
    workgroupBarrier();

    // --- Pass 2: variance ---
    var v = 0.0;
    i = tid;
    loop {
        if (i >= p.dim) { break; }
        let d = x[base + i] - mean;
        v = v + d * d;
        i = i + WG;
    }
    partial[tid] = v;
    workgroupBarrier();
    stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(p.dim) + p.eps);

    // --- Normalize, scale, shift ---
    i = tid;
    loop {
        if (i >= p.dim) { break; }
        out[base + i] = (x[base + i] - mean) * inv * weight[i] + bias[i];
        i = i + WG;
    }
}
