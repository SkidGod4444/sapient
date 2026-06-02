// Element-wise kernels, one invocation per element, 2-D-tiled grid.
// `op`: 0 = add (a+b, residual), 1 = SwiGLU (silu(a)*b = gate·sigmoid(gate)·up),
//       2 = exact erf GELU of `a` (b ignored — Whisper's activation),
//       3 = broadcast bias add `a[i] + b[i % dim]` (b is a per-channel bias of length `dim`).

struct P { n: u32, op: u32, dim: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       a:   array<f32>;
@group(0) @binding(1) var<storage, read>       b:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

// erf via Abramowitz & Stegun 7.1.26 (max error ~1.5e-7) — matches the CPU
// `erf_approx` so the GPU GELU is bit-close to the CPU reference.
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (0.254829592
        + (-0.284496736 + (1.421413741 + (-1.453152027 + 1.061405429 * t) * t) * t) * t) * t
        * exp(-ax * ax);
    return s * y;
}

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
    } else if (p.op == 2u) {
        let v = a[i];
        out[i] = 0.5 * v * (1.0 + erf_approx(v * 0.70710678));
    } else if (p.op == 3u) {
        out[i] = a[i] + b[i % p.dim];
    } else {
        out[i] = a[i] + b[i];
    }
}
