// RMSNorm: out[r,i] = x[r,i] / sqrt(mean_i(x[r,i]^2) + eps) * weight[i]
// One workgroup per row; cooperative sum-of-squares reduction in shared memory.
// f32 accumulation (f16 accumulation produces incoherent LLM output).

struct Params {
    dim:  u32,
    rows: u32,
    eps:  f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>             p:      Params;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    if (row >= p.rows) { return; }
    let base = row * p.dim;
    let tid = lid.x;

    // Sum of squares over the row (strided so all 256 lanes stay busy).
    var s = 0.0;
    var i = tid;
    loop {
        if (i >= p.dim) { break; }
        let v = x[base + i];
        s = s + v * v;
        i = i + WG;
    }
    partial[tid] = s;
    workgroupBarrier();

    // Tree reduction.
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let inv = inverseSqrt(partial[0] / f32(p.dim) + p.eps);

    // Normalize and scale.
    i = tid;
    loop {
        if (i >= p.dim) { break; }
        out[base + i] = x[base + i] * inv * weight[i];
        i = i + WG;
    }
}
