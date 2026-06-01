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

fn lcg() -> impl FnMut() -> f32 {
    let mut seed = 0x9E37_79B9u64;
    move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
fn matmul_nt_matches_cpu() {
    let Some(ctx) = ctx() else {
        return;
    };
    // Decode-shaped (m=1) and a small batch (m=3) to cover both.
    for (m, k, n) in [(1usize, 896usize, 1536usize), (3, 256, 64)] {
        let mut next = lcg();
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let xg = ctx.upload_f32(&x, "x");
        let wg = ctx.upload_f32(&w, "w");
        let got = ctx.download_f32(&ctx.matmul_nt(&xg, &wg, m, k, n)).unwrap();
        let mut want = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[r * k + i] * w[c * k + i];
                }
                want[r * n + c] = acc;
            }
        }
        let rel = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0f32, f32::max);
        assert!(rel < 1e-4, "matmul {m}x{k}x{n} rel={rel}");
    }
}

#[test]
fn swiglu_and_add_match_cpu() {
    let Some(ctx) = ctx() else {
        return;
    };
    let n = 4096;
    let mut next = lcg();
    let a: Vec<f32> = (0..n).map(|_| next()).collect();
    let b: Vec<f32> = (0..n).map(|_| next()).collect();
    let ag = ctx.upload_f32(&a, "a");
    let bg = ctx.upload_f32(&b, "b");

    let add = ctx.download_f32(&ctx.add(&ag, &bg)).unwrap();
    for i in 0..n {
        assert!((add[i] - (a[i] + b[i])).abs() < 1e-5);
    }
    let sg = ctx.download_f32(&ctx.swiglu(&ag, &bg)).unwrap();
    for i in 0..n {
        let silu = a[i] * (1.0 / (1.0 + (-a[i]).exp()));
        assert!((sg[i] - silu * b[i]).abs() < 1e-4, "swiglu at {i}");
    }
}

#[test]
fn embed_gather_matches_cpu() {
    let Some(ctx) = ctx() else {
        return;
    };
    let (vocab, dim) = (100usize, 320usize);
    let mut next = lcg();
    let table: Vec<f32> = (0..vocab * dim).map(|_| next()).collect();
    let ids: Vec<u32> = vec![7, 0, 99, 42];
    let tg = ctx.upload_f32(&table, "table");
    let ig = ctx.upload_u32(&ids, "ids");
    let got = ctx.download_f32(&ctx.embed(&ig, &tg, ids.len(), dim)).unwrap();
    for (t, &id) in ids.iter().enumerate() {
        for i in 0..dim {
            assert!((got[t * dim + i] - table[id as usize * dim + i]).abs() < 1e-9);
        }
    }
}
