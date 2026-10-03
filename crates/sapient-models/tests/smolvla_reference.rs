// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! SmolVLA against the LeRobot reference, stage by stage.
//!
//! The fixture (`tests/fixtures/smolvla_base.safetensors`) holds the inputs and
//! every intermediate of `lerobot/smolvla_base` run in f32 on CPU — written by
//! `scripts/gen_smolvla_fixture.py`. This test loads the real checkpoint and
//! checks, in order: image embedding → prefix embeddings → per-layer prefix K/V
//! → prefix output → suffix embedding → one velocity → the full 10-step action
//! chunk.
//!
//! Ignored by default: it needs the 0.9 GB checkpoint. Point
//! `SAPIENT_SMOLVLA_DIR` at a directory holding its `model.safetensors`:
//!
//! ```bash
//! SAPIENT_SMOLVLA_DIR=~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/<rev> \
//!   cargo test -p sapient-models --release --test smolvla_reference -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use sapient_core::Tensor;
use sapient_models::forward::{SmolVla, SmolVlaConfig, SmolVlaQuant};

fn fixture() -> HashMap<String, Tensor> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/smolvla_base.safetensors");
    sapient_io::load_safetensors(&p).expect("load SmolVLA fixture")
}

fn fx(f: &HashMap<String, Tensor>, name: &str) -> Vec<f32> {
    f.get(name)
        .unwrap_or_else(|| panic!("fixture tensor missing: {name}"))
        .to_f32_vec()
}

/// The generator's procedural 512×512 image, preprocessed to `[3, S, S]` in
/// `[-1, 1]` (mirrors `test_image` in `scripts/gen_smolvla_fixture.py`).
fn test_image(size: usize) -> Vec<f32> {
    let mut px = vec![0.0f32; 3 * size * size];
    for y in 0..size {
        for x in 0..size {
            let rgb = if (160..352).contains(&y) && (144..368).contains(&x) {
                [217u8, 26, 26]
            } else {
                [
                    (x * 255 / size) as u8,
                    (y * 255 / size) as u8,
                    ((x * 7 + y * 13) % 256) as u8,
                ]
            };
            for (c, v) in rgb.iter().enumerate() {
                px[c * size * size + y * size + x] = *v as f32 / 255.0 * 2.0 - 1.0;
            }
        }
    }
    px
}

/// (max abs error, max abs reference value) over the compared elements.
fn err(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut e = 0.0f32;
    let mut m = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        e = e.max((g - w).abs());
        m = m.max(w.abs());
    }
    (e, m)
}

fn check(stage: &str, got: &[f32], want: &[f32], tol: f32) {
    let (e, m) = err(got, want);
    println!("{stage:28} max_err {e:.3e}  (max |ref| {m:.3}, tol {tol:.0e})");
    assert!(e <= tol, "{stage}: max_err {e} > {tol}");
}

/// Rows of `full` (`[n_full, width]`) where `keep` is set.
fn keep_rows(full: &[f32], width: usize, keep: &[bool]) -> Vec<f32> {
    full.chunks_exact(width)
        .zip(keep)
        .filter(|(_, k)| **k)
        .flat_map(|(r, _)| r.iter().copied())
        .collect()
}

#[test]
#[ignore = "needs the lerobot/smolvla_base checkpoint: set SAPIENT_SMOLVLA_DIR"]
fn smolvla_matches_lerobot_reference() {
    let dir = std::env::var("SAPIENT_SMOLVLA_DIR").expect("set SAPIENT_SMOLVLA_DIR");
    let weights = sapient_io::load_safetensors(&PathBuf::from(dir).join("model.safetensors"))
        .expect("load SmolVLA checkpoint");
    let model = SmolVla::from_weights(SmolVlaConfig::default(), weights).expect("build SmolVLA");
    let c = model.config().clone();
    let f = fixture();

    // ── inputs ──────────────────────────────────────────────────────────────
    let pixels = test_image(model.image_size());
    let lang_mask = fx(&f, "input.lang_mask");
    let lang: Vec<u32> = fx(&f, "input.lang_tokens")
        .iter()
        .zip(&lang_mask)
        .filter(|(_, m)| **m > 0.5)
        .map(|(t, _)| *t as u32)
        .collect();
    let state = fx(&f, "input.state");
    let noise = fx(&f, "input.noise");
    // The reference keeps padded language tokens in the prefix; we drop them.
    let keep: Vec<bool> = fx(&f, "prefix.pad_mask").iter().map(|m| *m > 0.5).collect();
    let n = keep.iter().filter(|k| **k).count();
    println!(
        "prefix: {n} real tokens of {} (reference pads language to 48)",
        keep.len()
    );

    // ── 1. vision ───────────────────────────────────────────────────────────
    let t = std::time::Instant::now();
    let img = model.embed_image(&pixels).unwrap();
    let vision_ms = t.elapsed().as_secs_f64() * 1e3;
    check(
        "image embedding",
        &img,
        &fx(&f, "vision.image_embedding"),
        2e-2,
    );

    // ── 2. prefix embeddings ────────────────────────────────────────────────
    let embs = model.embed_prefix(&[&pixels], &lang, &state).unwrap();
    assert_eq!(embs.len(), n * c.vlm_hidden);
    check(
        "prefix embeddings",
        &embs,
        &keep_rows(&fx(&f, "prefix.embs"), c.vlm_hidden, &keep),
        0.5, // values reach ~±300 after the √hidden scale; stored as f16
    );

    // ── 3. prefix pass: per-layer K/V + output ──────────────────────────────
    let t = std::time::Instant::now();
    let (cache, out) = model.prefix_pass(&embs).unwrap();
    let prefix_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(cache.n, n);
    for layer in [0usize, 1, 15] {
        for (what, ours) in [
            ("keys", &cache.keys[layer]),
            ("values", &cache.values[layer]),
        ] {
            // Fixture layout [kv_heads, n_full, hd]: filter the padded rows per head.
            let full = fx(&f, &format!("prefix.kv.{layer}.{what}"));
            let per_head = full.len() / c.kv_heads;
            let want: Vec<f32> = full
                .chunks_exact(per_head)
                .flat_map(|h| keep_rows(h, c.head_dim, &keep))
                .collect();
            check(&format!("prefix layer {layer} {what}"), ours, &want, 5e-2);
        }
    }
    check(
        "prefix output",
        &out,
        &keep_rows(&fx(&f, "prefix.out"), c.vlm_hidden, &keep),
        5e-2,
    );

    // ── 4. suffix embedding at t = 1 ────────────────────────────────────────
    let suffix = model.embed_suffix(&noise, 1.0).unwrap();
    check(
        "suffix embedding (t=1)",
        &suffix,
        &fx(&f, "suffix.embs_t1"),
        1e-4,
    );

    // ── 5. one velocity ─────────────────────────────────────────────────────
    let t = std::time::Instant::now();
    let v = model.denoise_step(&cache, &noise, 1.0).unwrap();
    let step_ms = t.elapsed().as_secs_f64() * 1e3;
    check("velocity (t=1)", &v, &fx(&f, "denoise.v_t1"), 1e-4);

    // ── 6. full action chunk ────────────────────────────────────────────────
    let t = std::time::Instant::now();
    let actions = model.sample_actions(&cache, &noise).unwrap();
    let sample_ms = t.elapsed().as_secs_f64() * 1e3;
    check(
        "actions (10 Euler steps)",
        &actions,
        &fx(&f, "actions.normalized"),
        1e-4,
    );

    println!(
        "timing (f32, unquantized): vision {vision_ms:.0} ms · prefix {prefix_ms:.0} ms · \
         one denoise step {step_ms:.0} ms · {} steps {sample_ms:.0} ms",
        c.num_steps
    );
}

/// Action error and time of each Q8_0 choice, against the f32 LeRobot reference.
///
/// The yardstick is LeRobot's own default precision: it runs the VLM and the expert in bf16,
/// which moves the action chunk by `BF16_MAX` (max abs, measured with the same
/// inputs) from the f32 reference.
/// Hard inpainting (`sample_actions_inpaint`): an empty frozen prefix must be
/// bit-identical to plain sampling, frozen rows must come back exactly, and a
/// chunk asked to continue another plan's first rows must leave them with a
/// smaller jump than switching plans naively.
#[test]
#[ignore = "needs the lerobot/smolvla_base checkpoint: set SAPIENT_SMOLVLA_DIR"]
fn inpainting_keeps_frozen_rows_and_continues_them() {
    let dir = std::env::var("SAPIENT_SMOLVLA_DIR").expect("set SAPIENT_SMOLVLA_DIR");
    let weights = sapient_io::load_safetensors(&PathBuf::from(dir).join("model.safetensors"))
        .expect("load SmolVLA checkpoint");
    let model = SmolVla::from_weights(SmolVlaConfig::default(), weights).expect("build SmolVLA");
    let c = model.config().clone();
    let f = fixture();
    let pixels = test_image(model.image_size());
    let lang_mask = fx(&f, "input.lang_mask");
    let lang: Vec<u32> = fx(&f, "input.lang_tokens")
        .iter()
        .zip(&lang_mask)
        .filter(|(_, m)| **m > 0.5)
        .map(|(t, _)| *t as u32)
        .collect();
    let state = fx(&f, "input.state");
    let noise = fx(&f, "input.noise");
    let embs = model.embed_prefix(&[&pixels], &lang, &state).unwrap();
    let cache = model.prefix_cache(&embs).unwrap();
    let (w, steps, n) = (c.max_action_dim, c.num_steps, 18);

    let plain = model.sample_actions(&cache, &noise).unwrap();
    let empty = model
        .sample_actions_inpaint(&cache, &noise, steps, &[])
        .unwrap();
    assert_eq!(plain, empty, "an empty frozen prefix must change nothing");

    // Another plan for the same observation: the noise reversed.
    let other_noise: Vec<f32> = noise.iter().rev().copied().collect();
    let other = model.sample_actions(&cache, &other_noise).unwrap();
    let frozen = &other[..n * w];
    let cont = model
        .sample_actions_inpaint(&cache, &noise, steps, frozen)
        .unwrap();
    assert_eq!(
        &cont[..n * w],
        frozen,
        "frozen rows must be returned exactly"
    );

    let jump = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .take(6)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let last = &other[(n - 1) * w..n * w];
    let continued = jump(&cont[n * w..(n + 1) * w], last);
    let naive = jump(&plain[n * w..(n + 1) * w], last);
    println!("jump at the switch: continued {continued:.4}, naive {naive:.4}");
    assert!(
        continued < naive,
        "continuing the queued rows should not jump more than a naive switch: {continued} vs {naive}"
    );
    assert!(cont.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "needs the lerobot/smolvla_base checkpoint: set SAPIENT_SMOLVLA_DIR"]
fn smolvla_quantized_action_error() {
    let dir = PathBuf::from(std::env::var("SAPIENT_SMOLVLA_DIR").expect("set SAPIENT_SMOLVLA_DIR"));
    let f = fixture();
    let lang_mask = fx(&f, "input.lang_mask");
    let lang: Vec<u32> = fx(&f, "input.lang_tokens")
        .iter()
        .zip(&lang_mask)
        .filter(|(_, m)| **m > 0.5)
        .map(|(t, _)| *t as u32)
        .collect();
    let (state, noise) = (fx(&f, "input.state"), fx(&f, "input.noise"));
    let want = fx(&f, "actions.normalized");

    let q = |vision, vlm, expert| SmolVlaQuant {
        vision,
        vlm,
        expert,
        fast_math: false,
        int8_attention: false,
    };
    let configs = [
        ("f32", SmolVlaQuant::NONE),
        ("vision", q(true, false, false)),
        ("vlm", q(false, true, false)),
        ("expert", q(false, false, true)),
        ("vision+vlm", q(true, true, false)),
        ("all", q(true, true, true)),
        ("all+fast", SmolVlaQuant::ALL),
    ];
    for (name, quant) in configs {
        let weights = sapient_io::load_safetensors(&dir.join("model.safetensors")).unwrap();
        let model = SmolVla::from_weights_quant(SmolVlaConfig::default(), weights, quant).unwrap();
        let pixels = test_image(model.image_size());
        let mut best = [f64::MAX; 3];
        let mut actions = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let embs = model.embed_prefix(&[&pixels], &lang, &state).unwrap();
            let t1 = std::time::Instant::now();
            // embed_prefix includes the (tiny) language/state embedding.
            let cache = model.prefix_cache(&embs).unwrap();
            let t2 = std::time::Instant::now();
            actions = model.sample_actions(&cache, &noise).unwrap();
            let t3 = std::time::Instant::now();
            for (b, d) in best.iter_mut().zip([t1 - t0, t2 - t1, t3 - t2]) {
                *b = b.min(d.as_secs_f64() * 1e3);
            }
        }
        let (max, _) = err(&actions, &want);
        let rms = (actions
            .iter()
            .zip(&want)
            .map(|(a, w)| ((a - w) as f64).powi(2))
            .sum::<f64>()
            / want.len() as f64)
            .sqrt();
        println!(
            "{name:11} max_err {max:.3e}  rms {rms:.3e} | vision {:4.0} ms · prefix {:4.0} ms · \
             denoise {:4.0} ms · total {:4.0} ms",
            best[0],
            best[1],
            best[2],
            best.iter().sum::<f64>()
        );
        // Regression guards at 2× the measured values (f32 must stay exact).
        let (max_tol, rms_tol) = if quant == SmolVlaQuant::NONE {
            (1e-4, 1e-5)
        } else {
            (7e-2, 8e-3)
        };
        assert!(
            max <= max_tol && rms <= rms_tol,
            "{name}: action error max {max} rms {rms}"
        );

        // What fewer Euler steps cost (information only — a coarser integral
        // of the same flow, compared with the 10-step f32 reference).
        if quant == SmolVlaQuant::ALL {
            let embs = model.embed_prefix(&[&pixels], &lang, &state).unwrap();
            let cache = model.prefix_cache(&embs).unwrap();
            for steps in [5usize, 3] {
                let t = std::time::Instant::now();
                let a = model.sample_actions_steps(&cache, &noise, steps).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let (max, _) = err(&a, &want);
                let rms = (a
                    .iter()
                    .zip(&want)
                    .map(|(a, w)| ((a - w) as f64).powi(2))
                    .sum::<f64>()
                    / want.len() as f64)
                    .sqrt();
                println!(
                    "all, {steps} steps  max_err {max:.3e}  rms {rms:.3e} | denoise {ms:4.0} ms"
                );
            }
        }
    }
}

/// Quantization error over SEVERAL observations instead of one.
///
/// The f32 engine reproduces LeRobot to 4e-6 (the test above), so it serves as
/// the reference here: for each observation (different image, instruction,
/// state and start noise) the Q8_0 engines' action chunks are compared with the
/// f32 engine's. One observation is a noisy sample of this error — a 2e-7
/// change in the vision tower moved the single-observation RMS by 50%.
#[test]
#[ignore = "needs the lerobot/smolvla_base checkpoint: set SAPIENT_SMOLVLA_DIR"]
fn smolvla_quantized_error_over_observations() {
    let dir = PathBuf::from(std::env::var("SAPIENT_SMOLVLA_DIR").expect("set SAPIENT_SMOLVLA_DIR"));
    let load = |quant| {
        let weights = sapient_io::load_safetensors(&dir.join("model.safetensors")).unwrap();
        SmolVla::from_weights_quant(SmolVlaConfig::default(), weights, quant).unwrap()
    };
    let f = fixture();
    let base_lang: Vec<u32> = fx(&f, "input.lang_tokens")
        .iter()
        .zip(&fx(&f, "input.lang_mask"))
        .filter(|(_, m)| **m > 0.5)
        .map(|(t, _)| *t as u32)
        .collect();

    // Deterministic pseudo-random stream (SplitMix64; "normal" = a sum of 12
    // uniforms — the exact distribution does not matter here).
    let mut seed = 0x5EEDu64;
    let mut uniform = move || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut normal = move || (0..12).map(|_| uniform()).sum::<f32>() - 6.0;

    const N: usize = 8;
    let size = 512usize;
    struct Obs {
        pixels: Vec<f32>,
        lang: Vec<u32>,
        state: Vec<f32>,
        noise: Vec<f32>,
    }
    let observations: Vec<Obs> = (0..N)
        .map(|o| {
            // Image: the fixture frame, dimmed, with a moved, recoloured block.
            let mut pixels = test_image(size);
            let (bx, by) = (40 + 53 * o % 300, 30 + 71 * o % 300);
            let rgb = [
                (o * 37 % 256) as f32,
                (o * 91 % 256) as f32,
                (255 - o * 29 % 256) as f32,
            ];
            for (c, colour) in rgb.iter().enumerate() {
                for y in 0..size {
                    for x in 0..size {
                        let p = &mut pixels[c * size * size + y * size + x];
                        if (by..by + 120).contains(&y) && (bx..bx + 150).contains(&x) {
                            *p = colour / 255.0 * 2.0 - 1.0;
                        } else {
                            *p = (*p * (0.6 + 0.05 * o as f32)).clamp(-1.0, 1.0);
                        }
                    }
                }
            }
            // Instruction: a rotation / truncation of the fixture's tokens
            // (always ending with the newline token).
            let body = &base_lang[..base_lang.len() - 1];
            let mut lang: Vec<u32> = body.iter().cycle().skip(o).take(4 + o).copied().collect();
            lang.push(*base_lang.last().unwrap());
            let mut state = vec![0.0f32; 32];
            for v in state.iter_mut().take(6) {
                *v = normal() * 0.7;
            }
            let noise: Vec<f32> = (0..50 * 32).map(|_| normal()).collect();
            Obs {
                pixels,
                lang,
                state,
                noise,
            }
        })
        .collect();

    let run = |model: &SmolVla, o: &Obs| {
        let embs = model.embed_prefix(&[&o.pixels], &o.lang, &o.state).unwrap();
        let cache = model.prefix_cache(&embs).unwrap();
        model.sample_actions(&cache, &o.noise).unwrap()
    };
    let reference: Vec<Vec<f32>> = {
        let model = load(SmolVlaQuant::NONE);
        observations.iter().map(|o| run(&model, o)).collect()
    };
    let ref_rms = (reference
        .iter()
        .flatten()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        / (N * 1600) as f64)
        .sqrt();
    println!("{N} observations · reference action RMS {ref_rms:.3}");

    // Mirror check for scripts/smolvla_bf16_yardstick.py (same observations).
    println!(
        "first reference action row of observation 0: {:?}",
        reference[0][..6]
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    let q = |vision, vlm, expert, fast_math| SmolVlaQuant {
        vision,
        vlm,
        expert,
        fast_math,
        int8_attention: false,
    };
    for (name, quant) in [
        ("vision", q(true, false, false, false)),
        ("vlm", q(false, true, false, false)),
        ("expert", q(false, false, true, false)),
        ("vision + expert", q(true, false, true, false)),
        ("all Q8_0", q(true, true, true, false)),
        ("all Q8_0 + fast math", q(true, true, true, true)),
        ("fast (+ int8 attention)", SmolVlaQuant::ALL),
        ("fast math only (f32)", q(false, false, false, true)),
    ] {
        let model = load(quant);
        let (mut worst, mut sq, mut per_obs) = (0.0f32, 0.0f64, Vec::new());
        for (o, want) in observations.iter().zip(&reference) {
            let got = run(&model, o);
            let (max, _) = err(&got, want);
            let s: f64 = got
                .iter()
                .zip(want)
                .map(|(a, w)| ((a - w) as f64).powi(2))
                .sum();
            worst = worst.max(max);
            sq += s;
            per_obs.push((s / 1600.0).sqrt());
        }
        let rms = (sq / (N * 1600) as f64).sqrt();
        let per: Vec<String> = per_obs.iter().map(|r| format!("{r:.1e}")).collect();
        println!(
            "{name:22} max_err {worst:.3e}  rms {rms:.3e}  per-observation rms [{}]",
            per.join(" ")
        );
        // Regression guards at 2× the values measured on 2026-10-02.
        let (max_tol, rms_tol) = match name {
            "expert" => (0.13, 9e-3),
            "fast math only (f32)" => (1e-4, 1e-5),
            _ => (0.33, 2.7e-2),
        };
        assert!(
            worst <= max_tol && rms <= rms_tol,
            "{name}: max {worst} rms {rms}"
        );
    }
}
