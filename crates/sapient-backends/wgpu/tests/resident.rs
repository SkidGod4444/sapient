//! Resident-compute kernel tests — validate each GPU kernel against a CPU
//! reference. Run on whatever backend wgpu picks (Metal on a Mac, Vulkan/DX12 on
//! Linux/Windows); the same WGSL runs on all of them.

use sapient_backends_wgpu::WgpuContext;

fn ctx() -> Option<WgpuContext> {
    WgpuContext::new().ok()
}

/// CPU RMSNorm reference (f32).
fn cpu_rms_norm(x: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = row[i] * inv * weight[i];
        }
    }
    out
}

#[test]
fn rms_norm_matches_cpu() {
    let Some(ctx) = ctx() else {
        eprintln!("no GPU adapter — skipping wgpu rms_norm test");
        return;
    };
    // Non-power-of-two dim and rows to exercise the strided reduction tail.
    let (rows, dim) = (5usize, 1536usize);
    let mut seed = 0x1234_5678u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..rows * dim).map(|_| next()).collect();
    let weight: Vec<f32> = (0..dim).map(|_| 0.5 + next().abs()).collect();
    let eps = 1e-5f32;

    let xg = ctx.upload_f32(&x, "x");
    let wg = ctx.upload_f32(&weight, "w");
    let outg = ctx.rms_norm(&xg, &wg, rows, dim, eps);
    let got = ctx.download_f32(&outg).expect("download");

    let want = cpu_rms_norm(&x, &weight, rows, dim, eps);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "wgpu rms_norm vs cpu max_err={max_err}");
}
