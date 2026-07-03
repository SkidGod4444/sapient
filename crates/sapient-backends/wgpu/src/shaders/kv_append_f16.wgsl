// KV-cache append (f16 cache): write `seq` tokens' f32 K (or V) — layout
// [n_kv_heads, seq, head_dim], heads-major — into positions pos..pos+seq of a
// cache that stores f16 halves packed two-per-u32 word (core-WGSL pack2x16float,
// no SHADER_F16 feature required). One thread per word = two consecutive
// channels of the same (kv-head, position); head_dim must be even (checked
// host-side) so a word never straddles rows and each word has exactly one writer.
// seq = 1 is the decode step; seq > 1 is batched prefill (Phase 7.5).

struct P { n_kv: u32, head_dim: u32, max_seq: u32, pos: u32, seq: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       src: array<f32>;  // [n_kv*seq*head_dim]
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;  // packed f16 pairs
@group(0) @binding(2) var<uniform>             p:   P;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = (wg.x + wg.y * nwg.x) * 256u + lid.x;    // pair index
    if (j >= (p.n_kv * p.seq * p.head_dim) / 2u) { return; }
    let e = j * 2u;
    let hh = e / (p.seq * p.head_dim);
    let rem = e % (p.seq * p.head_dim);
    let s = rem / p.head_dim;
    let c = rem % p.head_dim;
    let d = (hh * p.max_seq + p.pos + s) * p.head_dim + c; // even (head_dim is even)
    dst[d >> 1u] = pack2x16float(vec2<f32>(src[e], src[e + 1u]));
}
