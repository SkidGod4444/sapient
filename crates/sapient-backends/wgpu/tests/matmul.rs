//! Correctness tests for the wgpu GPU kernels, validated against a host reference.
//!
//! These run on whatever GPU wgpu finds — Metal on macOS, Vulkan/DX12 on
//! Linux/Windows. When no GPU is available (headless CI without a software
//! rasterizer) the tests skip rather than fail, so they never block a CPU-only build.

use sapient_backends_wgpu::{quantize_q8_0_rows, WgpuContext};

/// Host reference: out[m,n] = sum_k x[m,k] * w[n,k].
fn cpu_matmul_nt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0;
            for ki in 0..k {
                acc += x[mi * k + ki] * w[ni * k + ki];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

fn ctx_or_skip() -> Option<WgpuContext> {
    match WgpuContext::new() {
        Ok(c) => {
            eprintln!("wgpu test running on: {}", c.adapter_label());
            Some(c)
        }
        Err(e) => {
            eprintln!("SKIP: no GPU available for wgpu tests ({e})");
            None
        }
    }
}

#[test]
fn f32_matmul_small_known() {
    let Some(ctx) = ctx_or_skip() else { return };
    // out[1,2] = x[1,3] @ w[2,3]^T
    let x = [1.0, 2.0, 3.0];
    let w = [
        1.0, 0.0, 0.0, // row 0 → dot = 1
        0.0, 1.0, 1.0, // row 1 → dot = 5
    ];
    let out = ctx.matmul_nt_f32(&x, &w, 1, 3, 2).unwrap();
    assert_eq!(out, vec![1.0, 5.0]);
}

#[test]
fn f32_matmul_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let (m, k, n) = (3, 64, 5);
    // Deterministic pseudo-random fill (no rng dependency).
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
        .collect();

    let gpu = ctx.matmul_nt_f32(&x, &w, m, k, n).unwrap();
    let cpu = cpu_matmul_nt(&x, &w, m, k, n);

    assert_eq!(gpu.len(), cpu.len());
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        assert!((g - c).abs() < 1e-3, "gpu={g} cpu={c}");
    }
}

#[test]
fn q8_0_matmul_matches_dequantized_reference() {
    let Some(ctx) = ctx_or_skip() else { return };
    let (m, k, n) = (2, 128, 4);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i * 9 % 23) as f32 - 11.0) * 0.05)
        .collect();

    // Quantize the weights, then build the host reference from the SAME dequantized
    // weights so the only thing under test is the GPU dequant-and-matmul.
    let (qw, scales) = quantize_q8_0_rows(&w, n, k);
    let w_dequant = dequantize_rows(&qw, &scales, n, k);

    let gpu = ctx.matmul_nt_q8_0(&x, &qw, &scales, m, k, n).unwrap();
    let cpu = cpu_matmul_nt(&x, &w_dequant, m, k, n);

    assert_eq!(gpu.len(), cpu.len());
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        assert!((g - c).abs() < 1e-3, "gpu={g} cpu={c}");
    }
}

/// Reconstruct F32 weights from the Q8_0 GPU buffers (mirror of the shader's dequant).
fn dequantize_rows(qw: &[u32], scales: &[f32], n: usize, k: usize) -> Vec<f32> {
    let nblocks = k / 32;
    let mut out = vec![0.0f32; n * k];
    for row in 0..n {
        for b in 0..nblocks {
            let scale = scales[row * nblocks + b];
            let word_base = (row * nblocks + b) * 8;
            for w4 in 0..8 {
                let word = qw[word_base + w4];
                for lane in 0..4 {
                    let byte = ((word >> (lane * 8)) & 0xFF) as u8;
                    let signed = byte as i8 as f32;
                    out[row * k + b * 32 + w4 * 4 + lane] = signed * scale;
                }
            }
        }
    }
    out
}

#[test]
#[ignore]
fn perf_q8_0_gemv() {
    let Some(ctx) = ctx_or_skip() else { return };
    // Decode-shaped projection: M=1, K=2048, N=2048 (Q8_0).
    let (m, k, n) = (1usize, 2048usize, 2048usize);
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let (qw, scales) = quantize_q8_0_rows(&w, n, k);

    // warm up (shader compile + first dispatch)
    let _ = ctx.matmul_nt_q8_0(&x, &qw, &scales, m, k, n).unwrap();

    let iters = 50;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = ctx.matmul_nt_q8_0(&x, &qw, &scales, m, k, n).unwrap();
    }
    let per = t.elapsed().as_secs_f64() / iters as f64 * 1e3;
    eprintln!(
        "GPU Q8_0 GEMV [1x2048]@[2048x2048]: {per:.3} ms/call ({:.1} GFLOP/s)",
        (2.0 * k as f64 * n as f64) / (per / 1e3) / 1e9
    );
}
