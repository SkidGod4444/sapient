// Element-wise kernels, one invocation per element, 2-D-tiled grid.
// `op`: 0 = add (a+b, residual), 1 = SwiGLU (silu(a)*b = gate·sigmoid(gate)·up).

struct P { n: u32, op: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       a:   array<f32>;
@group(0) @binding(1) var<storage, read>       b:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wg.x + wg.y * nwg.x) * 256u + lid.x;
    if (i >= p.n) { return; }
    if (p.op == 1u) {
        let g = a[i];
        let silu = g * (1.0 / (1.0 + exp(-g)));
        out[i] = silu * b[i];
    } else {
        out[i] = a[i] + b[i];
    }
}
