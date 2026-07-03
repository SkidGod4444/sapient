// Embedding gather from a Q4_K-resident table: out[t, i] = dequant(table[ids[t], i]).
// Same raw-super-block layout as matmul_nt_q4_k.wgsl (36 words per 256 weights).
// One workgroup per token row; lanes dequantize `dim` elements strided.

struct P { n_tokens: u32, dim: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       ids: array<u32>;
@group(0) @binding(1) var<storage, read>       qb:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

fn byte_of(word: u32, b: u32) -> u32 { return (word >> (b * 8u)) & 0xFFu; }
fn scale_byte(blk: u32, i: u32) -> u32 { return byte_of(qb[blk + 1u + i / 4u], i % 4u); }

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = wg.x + wg.y * nwg.x;
    if (t >= p.n_tokens) { return; }
    let row_base = ids[t] * (p.dim / 256u) * 36u;
    let dst = t * p.dim;
    var i = lid.x;
    loop {
        if (i >= p.dim) { break; }
        let blk = row_base + (i / 256u) * 36u;
        let e = i % 256u;
        let is = e / 32u;
        let dm = unpack2x16float(qb[blk]);
        var sc: u32;
        var mn: u32;
        if (is < 4u) {
            sc = scale_byte(blk, is) & 63u;
            mn = scale_byte(blk, is + 4u) & 63u;
        } else {
            sc = (scale_byte(blk, is + 4u) & 0xFu) | ((scale_byte(blk, is - 4u) >> 6u) << 4u);
            mn = (scale_byte(blk, is + 4u) >> 4u) | ((scale_byte(blk, is) >> 6u) << 4u);
        }
        let bidx = (is / 2u) * 32u + (e % 32u);           // qs byte index
        let q = (byte_of(qb[blk + 4u + bidx / 4u], bidx % 4u) >> ((is % 2u) * 4u)) & 0xFu;
        out[dst + i] = dm.x * f32(sc) * f32(q) - dm.y * f32(mn);
        i = i + 256u;
    }
}
