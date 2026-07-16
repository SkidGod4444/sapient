// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

// Embedding gather: out[t, :] = table[ids[t], :]  (rows of the token-embedding
// matrix). One workgroup per token row; lanes copy `dim` elements strided.

struct P { n_tokens: u32, dim: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       ids:   array<u32>;   // [n_tokens]
@group(0) @binding(1) var<storage, read>       table: array<f32>;   // [vocab*dim]
@group(0) @binding(2) var<storage, read_write> out:   array<f32>;   // [n_tokens*dim]
@group(0) @binding(3) var<uniform>             p:     P;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = wg.x + wg.y * nwg.x;
    if (t >= p.n_tokens) { return; }
    let src = ids[t] * p.dim;
    let dst = t * p.dim;
    var i = lid.x;
    loop {
        if (i >= p.dim) { break; }
        out[dst + i] = table[src + i];
        i = i + 256u;
    }
}
