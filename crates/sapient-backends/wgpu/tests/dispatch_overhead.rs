// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! How much does one kernel dispatch cost inside a batch, apart from the work
//! it does? Run with:
//! `cargo test -p sapient-backends-wgpu --release --test dispatch_overhead -- --ignored --nocapture`
//!
//! A decode token of a 28-layer model records about 560 dispatches. Each case
//! below records `N` dispatches of a trivially small kernel (1 536 floats) in
//! one batch and reports the time per dispatch, so the fixed cost can be read
//! off directly and compared across ways of chaining them.

use sapient_backends_wgpu::WgpuContext;
use std::time::Instant;

const N: usize = 560;
const LEN: usize = 1536;

fn time_batch(ctx: &WgpuContext, name: &str, mut body: impl FnMut()) {
    let mut best = f64::MAX;
    for _ in 0..8 {
        let started = Instant::now();
        body();
        best = best.min(started.elapsed().as_secs_f64() * 1e3);
    }
    let _ = ctx;
    println!(
        "{name:<52} {best:>7.2} ms per batch  {:>6.1} us per dispatch",
        best * 1e3 / N as f64
    );
}

#[test]
#[ignore = "timing probe; needs a GPU adapter"]
fn dispatch_overhead() {
    let Ok(ctx) = WgpuContext::new() else {
        eprintln!("no adapter");
        return;
    };
    println!("adapter: {}", ctx.adapter_label());
    let a = ctx.upload_f32(&vec![0.5f32; LEN], "a");
    let b = ctx.upload_f32(&vec![0.25f32; LEN], "b");

    // Baseline: one dispatch + readback (the cost of a batch that does nothing else).
    time_batch(
        &ctx,
        "1 dispatch + readback (x560 scale is meaningless)",
        || {
            ctx.begin_batch();
            let y = ctx.add(&a, &b);
            ctx.download_f32(&y).unwrap();
        },
    );

    // Chain: each dispatch reads the previous one's output (what a layer does).
    time_batch(&ctx, "560 chained adds, one pipeline", || {
        ctx.begin_batch();
        let mut y = ctx.add(&a, &b);
        for _ in 1..N {
            y = ctx.add(&y, &b);
        }
        ctx.download_f32(&y).unwrap();
    });

    // Independent: every dispatch reads the same two inputs.
    time_batch(&ctx, "560 independent adds, one pipeline", || {
        ctx.begin_batch();
        let mut y = ctx.add(&a, &b);
        for _ in 1..N {
            y = ctx.add(&a, &b);
        }
        ctx.download_f32(&y).unwrap();
    });

    // Chain alternating two pipelines (pipeline state changes between dispatches).
    time_batch(
        &ctx,
        "560 chained, alternating add / swiglu pipelines",
        || {
            ctx.begin_batch();
            let mut y = ctx.add(&a, &b);
            for i in 1..N {
                y = if i % 2 == 0 {
                    ctx.add(&y, &b)
                } else {
                    ctx.swiglu(&y, &b)
                };
            }
            ctx.download_f32(&y).unwrap();
        },
    );

    // The same chain without the shared compute pass (one pass per dispatch).
    // SAFETY-free: the switch is an env var read once, so this case needs its
    // own process: SAPIENT_WGPU_SHARED_PASS=0 cargo test … dispatch_overhead
    if std::env::var("SAPIENT_WGPU_SHARED_PASS").as_deref() == Ok("0") {
        println!("(run with SAPIENT_WGPU_SHARED_PASS=0: one compute pass per dispatch)");
    }
}

/// Cost of each matmul shape of a Qwen2.5-1.5B Q4_K_M decode token (28 layers,
/// hidden 1536, FFN 8960, 2 KV heads × 128, vocab 151 936), measured as 28
/// repeats in one batch. Multiply by the per-token call count to see where a
/// token's GPU time goes.
#[test]
#[ignore = "timing probe; needs a GPU adapter and ~300 MB of GPU memory"]
fn kernel_cost_at_model_sizes() {
    use sapient_backends_wgpu::GpuBuffer;
    const REPS: usize = 28;

    fn time(ctx: &WgpuContext, name: &str, per_token: usize, one: &dyn Fn() -> GpuBuffer) {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let started = Instant::now();
            ctx.begin_batch();
            let mut y = one();
            for _ in 1..REPS {
                y = one();
            }
            ctx.download_f32(&y).unwrap();
            best = best.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let per_call = (best - 1.3) / REPS as f64; // minus the batch floor
        println!(
            "{name:<34} {per_call:>7.3} ms per call  x{per_token:<3} = {:>6.2} ms per token",
            per_call * per_token as f64
        );
    }

    // Deterministic block bytes; the values are irrelevant to the timing but
    // the f16 scale word is kept finite (0x3800 = 0.5).
    fn blocks(numel: usize, block_bytes: usize, scale_at: usize) -> Vec<u8> {
        let mut v = vec![0x55u8; numel / 256 * block_bytes];
        for b in v.chunks_exact_mut(block_bytes) {
            b[scale_at] = 0x00;
            b[scale_at + 1] = 0x38;
        }
        v
    }

    let Ok(ctx) = WgpuContext::new() else {
        eprintln!("no adapter");
        return;
    };
    println!("adapter: {}", ctx.adapter_label());

    for (name, k, n, per_token) in [
        ("Q4_K q / o  1536 -> 1536", 1536usize, 1536usize, 56usize),
        ("Q4_K k / v  1536 -> 256", 1536, 256, 56),
        ("Q4_K gate / up  1536 -> 8960", 1536, 8960, 56),
        ("Q4_K down  8960 -> 1536", 8960, 1536, 28),
    ] {
        let w = ctx.upload_q4_k(&blocks(k * n, 144, 0), k * n, "w").unwrap();
        let x = ctx.upload_f32(&vec![0.01f32; k], "x");
        time(&ctx, name, per_token, &|| {
            ctx.matmul_nt_q4_k(&x, &w, 1, k, n)
        });
    }
    {
        let (k, n) = (1536usize, 151_936usize);
        let w = ctx
            .upload_q6_k(&blocks(k * n, 210, 208), k * n, "lm")
            .unwrap();
        let x = ctx.upload_f32(&vec![0.01f32; k], "x");
        time(&ctx, "Q6_K lm_head  1536 -> 151936", 1, &|| {
            ctx.matmul_nt_q6_k(&x, &w, 1, k, n)
        });
    }
    let x = ctx.upload_f32(&vec![0.5f32; 1536], "x");
    let g = ctx.upload_f32(&vec![1.0f32; 1536], "g");
    time(&ctx, "rms_norm 1536", 57, &|| {
        ctx.rms_norm(&x, &g, 1, 1536, 1e-6)
    });
    let b = ctx.upload_f32(&vec![0.25f32; 1536], "b");
    time(&ctx, "add 1536 (trivial kernel)", 168, &|| ctx.add(&x, &b));
}
