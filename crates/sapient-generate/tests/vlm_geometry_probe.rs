// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Numeric orientation probe for the SmolVLM vision path: encode synthetic
//! half-images and check the 8×8 visual-token grid varies along the correct
//! axis (columns for a vertical split, rows for a horizontal one) — verifies
//! patch order / pixel-shuffle / preprocessing geometry WITHOUT trusting the
//! 256M language model's shaky spatial words.

use sapient_generate::VlmPipeline;

fn grid_norms(vlm: &VlmPipeline, pixels: &[f32]) -> Vec<f32> {
    let v = vlm.encode_image_embeddings(pixels).expect("encode");
    let d = v.len() / 64;
    (0..64)
        .map(|t| {
            v[t * d..(t + 1) * d]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads SmolVLM-256M"]
async fn visual_token_grid_matches_image_geometry() {
    let vlm = VlmPipeline::from_pretrained("smolvlm-256m")
        .await
        .expect("load");
    let s = 512usize;

    // Vertical split: black LEFT, white RIGHT → columns 0..4 vs 4..8 differ.
    let mut vbar = vec![0.0f32; 3 * s * s];
    for c in 0..3 {
        for y in 0..s {
            for x in 0..s {
                let val = if x < s / 2 { -0.9 } else { 0.9 };
                vbar[c * s * s + y * s + x] = val;
            }
        }
    }
    fn axis_diffs(g: &[f32]) -> (f32, f32) {
        let col_l: f32 = (0..8)
            .map(|r| (0..4).map(|c| g[r * 8 + c]).sum::<f32>())
            .sum::<f32>()
            / 32.0;
        let col_r: f32 = (0..8)
            .map(|r| (4..8).map(|c| g[r * 8 + c]).sum::<f32>())
            .sum::<f32>()
            / 32.0;
        let row_t: f32 = (0..4)
            .map(|r| (0..8).map(|c| g[r * 8 + c]).sum::<f32>())
            .sum::<f32>()
            / 32.0;
        let row_b: f32 = (4..8)
            .map(|r| (0..8).map(|c| g[r * 8 + c]).sum::<f32>())
            .sum::<f32>()
            / 32.0;
        ((col_l - col_r).abs(), (row_t - row_b).abs())
    }
    let (v_col, v_row) = axis_diffs(&grid_norms(&vlm, &vbar));

    // Horizontal split: dark TOP, light BOTTOM → rows must dominate.
    let mut hbar = vec![0.0f32; 3 * s * s];
    for c in 0..3 {
        for y in 0..s {
            for x in 0..s {
                hbar[c * s * s + y * s + x] = if y < s / 2 { -0.9 } else { 0.9 };
            }
        }
    }
    let (h_col, h_row) = axis_diffs(&grid_norms(&vlm, &hbar));

    println!("vertical split: col_diff {v_col:.3} row_diff {v_row:.3}");
    println!("horizontal split: col_diff {h_col:.3} row_diff {h_row:.3}");
    // Self-normalizing orientation check: each split's signal must dominate on
    // ITS axis (a transpose swaps both; a flip changes neither, but flips are
    // impossible in row-major index math — this pins the transpose class).
    assert!(v_col > v_row, "vertical split should be column-dominant");
    assert!(h_row > h_col, "horizontal split should be row-dominant");
}
