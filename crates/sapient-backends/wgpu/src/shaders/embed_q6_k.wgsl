// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

// Embedding gather from a Q6_K-resident table: out[t, i] = dequant(table[ids[t], i]).
// Same padded-super-block layout as matmul_nt_q6_k.wgsl (53 words per 256 weights).
// Matters for tied-embedding Q4_K_M models (Llama-3.2 class) whose token table is
// Q6_K — it doubles as the lm_head. One workgroup per token; lanes stride `dim`.

struct P { n_tokens: u32, dim: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       ids: array<u32>;
@group(0) @binding(1) var<storage, read>       qb:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

fn byte_of(word: u32, b: u32) -> u32 { return (word >> (b * 8u)) & 0xFFu; }

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = wg.x + wg.y * nwg.x;
    if (t >= p.n_tokens) { return; }
    let row_base = ids[t] * (p.dim / 256u) * 53u;
    let dst = t * p.dim;
    var i = lid.x;
    loop {
        if (i >= p.dim) { break; }
        let blk = row_base + (i / 256u) * 53u;
        let e = i % 256u;
        let h = e >> 7u;
        let r = e & 127u;
        let g = r >> 5u;
        let l = r & 31u;

        let d = unpack2x16float(qb[blk + 52u]).x;
        let s = h * 8u + 2u * g + (l >> 4u);
        let sc_raw = byte_of(qb[blk + 48u + (s >> 2u)], s & 3u);
        let sc = f32(i32(sc_raw << 24u) >> 24u);

        let ql_b = h * 64u + (g & 1u) * 32u + l;
        let qh_b = h * 32u + l;
        let q = ((byte_of(qb[blk + (ql_b >> 2u)], ql_b & 3u) >> ((g >> 1u) * 4u)) & 0xFu)
              | (((byte_of(qb[blk + 32u + (qh_b >> 2u)], qh_b & 3u) >> (g * 2u)) & 3u) << 4u);
        out[dst + i] = d * sc * (f32(q) - 32.0);
        i = i + 256u;
    }
}
