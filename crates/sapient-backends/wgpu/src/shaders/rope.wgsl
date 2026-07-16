// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

// Rotary position embedding (NEOX / rotate_half split-half convention), in-place.
// Matches the CPU `apply_rope_partial`: data is [rows, head_dim] where a row is one
// (batch, head, seq) slice; rows = batch*n_heads*seq_len laid out so row % seq_len is
// the sequence position index. Only the first `rotary_dim` channels are rotated; the
// half-split is over rotary_dim (half = rotary_dim/2). Channels [rotary_dim..head_dim]
// pass through untouched (no write needed — in-place).
//
//   freq_i = pos / base^(2i / rotary_dim)
//   out[i]      = x0*cos - x1*sin
//   out[i+half] = x1*cos + x0*sin     where x0=data[i], x1=data[i+half]
//
// One invocation per rotated pair; 2-D-tiled grid. f32 math (sin/cos in f32).

struct P { rows: u32, head_dim: u32, rotary_dim: u32, half: u32, seq_len: u32, base: f32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read_write> data: array<f32>;   // [rows*head_dim]
@group(0) @binding(1) var<storage, read>       pos:  array<u32>;   // [seq_len]
@group(0) @binding(2) var<uniform>             p:    P;

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let e = (wg.x + wg.y * nwg.x) * 256u + lid.x;   // global pair index
    let total = p.rows * p.half;
    if (e >= total) { return; }

    let i = e % p.half;          // channel within the rotary half
    let row = e / p.half;        // which (batch,head,seq) row
    let s = row % p.seq_len;     // sequence position index → pos[s]
    let position = f32(pos[s]);

    let theta = position / pow(p.base, 2.0 * f32(i) / f32(p.rotary_dim));
    let cos_f = cos(theta);
    let sin_f = sin(theta);

    let base_idx = row * p.head_dim;
    let x0 = data[base_idx + i];
    let x1 = data[base_idx + i + p.half];
    data[base_idx + i]          = x0 * cos_f - x1 * sin_f;
    data[base_idx + i + p.half] = x1 * cos_f + x0 * sin_f;
}
