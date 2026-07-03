// KV-cache append (f32 cache): write one token's K (or V) — an f32 buffer of
// [n_kv_heads, head_dim] — into position `pos` of a pre-allocated
// [n_kv_heads, max_seq, head_dim] f32 cache. One dispatch replaces the old
// per-kv-head copy_buffer_to_buffer loop. The f16-cache twin is
// kv_append_f16.wgsl (u32-packed halves).

struct P { n_kv: u32, head_dim: u32, max_seq: u32, pos: u32 };

@group(0) @binding(0) var<storage, read>       src: array<f32>;  // [n_kv*head_dim]
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;  // [n_kv*max_seq*head_dim]
@group(0) @binding(2) var<uniform>             p:   P;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wg.x + wg.y * nwg.x) * 256u + lid.x;
    if (i >= p.n_kv * p.head_dim) { return; }
    let hh = i / p.head_dim;
    let c = i % p.head_dim;
    dst[(hh * p.max_seq + p.pos) * p.head_dim + c] = src[i];
}
