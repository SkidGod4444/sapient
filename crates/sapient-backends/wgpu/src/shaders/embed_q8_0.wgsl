// Embedding gather from a Q8_0-resident table: out[t, i] = dequant(table[ids[t], i]).
// Same `upload_q8_0` layout as matmul_nt_q8_0.wgsl (qs: 4 int8 per u32 word,
// scales: one f32 per 32-weight block). One workgroup per token row; lanes
// dequantize `dim` elements strided — mirrors embed.wgsl.

struct P { n_tokens: u32, dim: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       ids:    array<u32>;   // [n_tokens]
@group(0) @binding(1) var<storage, read>       qs:     array<u32>;   // [vocab * dim/4]
@group(0) @binding(2) var<storage, read>       scales: array<f32>;   // [vocab * dim/32]
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;   // [n_tokens*dim]
@group(0) @binding(4) var<uniform>             p:      P;

fn unpack_i8(word: u32, lane: u32) -> f32 {
    let byte = (word >> (lane * 8u)) & 0xFFu;
    return f32(i32(byte << 24u) >> 24u);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = wg.x + wg.y * nwg.x;
    if (t >= p.n_tokens) { return; }
    let row = ids[t];
    let qbase = row * (p.dim / 4u);
    let sbase = row * (p.dim / 32u);
    let dst = t * p.dim;
    var i = lid.x;
    loop {
        if (i >= p.dim) { break; }
        let word = qs[qbase + (i / 4u)];
        out[dst + i] = scales[sbase + (i / 32u)] * unpack_i8(word, i % 4u);
        i = i + 256u;
    }
}
