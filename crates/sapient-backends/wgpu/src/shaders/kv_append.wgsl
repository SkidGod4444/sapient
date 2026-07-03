// KV-cache append (f32 cache): write `seq` tokens' K (or V) — an f32 buffer of
// [n_kv_heads, seq, head_dim] (heads-major, i.e. post-transpose) — into positions
// pos..pos+seq of a pre-allocated [n_kv_heads, max_seq, head_dim] f32 cache.
// seq = 1 is the decode step; seq > 1 is batched prefill (Phase 7.5). One
// dispatch replaces the old per-kv-head copy_buffer_to_buffer loop. The
// f16-cache twin is kv_append_f16.wgsl (u32-packed halves).

struct P { n_kv: u32, head_dim: u32, max_seq: u32, pos: u32, seq: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       src: array<f32>;  // [n_kv*seq*head_dim]
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;  // [n_kv*max_seq*head_dim]
@group(0) @binding(2) var<uniform>             p:   P;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wg.x + wg.y * nwg.x) * 256u + lid.x;
    if (i >= p.n_kv * p.seq * p.head_dim) { return; }
    let hh = i / (p.seq * p.head_dim);
    let rem = i % (p.seq * p.head_dim);
    let s = rem / p.head_dim;
    let c = rem % p.head_dim;
    dst[(hh * p.max_seq + p.pos + s) * p.head_dim + c] = src[i];
}
