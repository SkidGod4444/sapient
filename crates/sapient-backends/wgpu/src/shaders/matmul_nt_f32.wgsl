// Linear projection: out[M, N] = x[M, K] @ W[N, K]^T
//
// W is stored row-major as [N, K] (PyTorch nn.Linear layout), so output row n is
// the dot product of x's row m with W's row n. One invocation computes one output
// element. This is the correctness-first kernel; a tiled/shared-memory variant
// comes later for throughput.

@group(0) @binding(0) var<storage, read>       x:    array<f32>;  // [M*K]
@group(0) @binding(1) var<storage, read>       w:    array<f32>;  // [N*K]
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;  // [M*N]
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;   // (M, K, N, _)

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    let n = gid.y;
    let big_m = dims.x;
    let k = dims.y;
    let big_n = dims.z;
    if (m >= big_m || n >= big_n) {
        return;
    }

    var acc = 0.0;
    let x_base = m * k;
    let w_base = n * k;
    for (var i = 0u; i < k; i = i + 1u) {
        acc = acc + x[x_base + i] * w[w_base + i];
    }
    out[m * big_n + n] = acc;
}
