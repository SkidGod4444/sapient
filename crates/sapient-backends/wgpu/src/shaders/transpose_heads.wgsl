// Transpose the two leading axes of a [outer, inner, hd] tensor → [inner, outer, hd]
// (head_dim-contiguous blocks move as a unit). One invocation per output element,
// 2-D-tiled grid. Used to convert q/k/v between the matmul layout [seq, n_heads, hd]
// and the attention-kernel layout [n_heads, seq, hd] (and back).

struct P { outer: u32, inner: u32, hd: u32, n: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let oi = (wg.x + wg.y * nwg.x) * 256u + lid.x;
    if (oi >= p.n) { return; }
    // Output layout [inner, outer, hd]: decompose oi → (i, o, c).
    let block = p.outer * p.hd;
    let i = oi / block;
    let rem = oi % block;
    let o = rem / p.hd;
    let c = rem % p.hd;
    // Source layout [outer, inner, hd].
    let src = (o * p.inner + i) * p.hd + c;
    out[oi] = x[src];
}
