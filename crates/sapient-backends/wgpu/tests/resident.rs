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
    for (m, k, n) in [(1usize, 896usize, 1536usize), (3, 256, 64), (17, 256, 64)] {
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

/// CPU RoPE reference (NEOX rotate_half, partial), data = [n_heads*seq_len, head_dim].
fn cpu_rope(
    data: &mut [f32],
    positions: &[u32],
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
) {
    let half = rotary_dim / 2;
    for h in 0..n_heads {
        for (s, &pos) in positions.iter().enumerate() {
            let base_idx = (h * seq_len + s) * head_dim;
            for i in 0..half {
                let freq = (pos as f32) / base.powf(2.0 * i as f32 / rotary_dim as f32);
                let (sin_f, cos_f) = freq.sin_cos();
                let x0 = data[base_idx + i];
                let x1 = data[base_idx + i + half];
                data[base_idx + i] = x0 * cos_f - x1 * sin_f;
                data[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
            }
        }
    }
}

#[test]
fn rope_matches_cpu() {
    let Some(ctx) = ctx() else {
        return;
    };
    // Phi-style partial rotary (head_dim=80, rotary_dim=32) and full rotary (64/64).
    for (n_heads, seq_len, head_dim, rotary_dim) in
        [(4usize, 7usize, 80usize, 32usize), (3, 5, 64, 64)]
    {
        let mut next = lcg();
        let n = n_heads * seq_len * head_dim;
        let data: Vec<f32> = (0..n).map(|_| next()).collect();
        // Positions starting at an offset (decode appends past the prompt).
        let positions: Vec<u32> = (0..seq_len as u32).map(|s| s + 11).collect();
        let base = 10000.0f32;

        let g = ctx.upload_f32(&data, "rope.x");
        ctx.rope(&g, &positions, n_heads, seq_len, head_dim, rotary_dim, base);
        let got = ctx.download_f32(&g).unwrap();

        let mut want = data.clone();
        cpu_rope(
            &mut want, &positions, n_heads, seq_len, head_dim, rotary_dim, base,
        );

        let max_err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "rope {head_dim}/{rotary_dim} max_err={max_err}"
        );
    }
}

/// CPU naive causal GQA attention reference.
#[allow(clippy::too_many_arguments)]
fn cpu_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let kv_rep = n_heads / n_kv_heads;
    let kv_offset = seq_k - seq_q;
    let mut out = vec![0.0f32; n_heads * seq_q * head_dim];
    for h in 0..n_heads {
        let kvh = h / kv_rep;
        for qi in 0..seq_q {
            let qbase = (h * seq_q + qi) * head_dim;
            let attend = qi + kv_offset + 1;
            let mut scores = vec![0.0f32; attend];
            let mut mx = f32::NEG_INFINITY;
            for (ki, sc) in scores.iter_mut().enumerate() {
                let kbase = (kvh * seq_k + ki) * head_dim;
                let mut d = 0.0f32;
                for c in 0..head_dim {
                    d += q[qbase + c] * k[kbase + c];
                }
                *sc = d * scale;
                mx = mx.max(*sc);
            }
            let mut den = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - mx).exp();
                den += *sc;
            }
            for (ki, &w) in scores.iter().enumerate() {
                let vbase = (kvh * seq_k + ki) * head_dim;
                let pw = w / den;
                for c in 0..head_dim {
                    out[qbase + c] += pw * v[vbase + c];
                }
            }
        }
    }
    out
}

#[test]
fn attention_matches_cpu() {
    let Some(ctx) = ctx() else {
        return;
    };
    // Decode (seq_q=1, GQA 14:2 like Qwen) and prefill (seq_q=seq_k, MHA, head_dim=80).
    let cases = [
        (14usize, 2usize, 1usize, 23usize, 64usize),
        (4, 4, 6, 6, 80),
        (8, 2, 5, 9, 128),
    ];
    for (n_heads, n_kv_heads, seq_q, seq_k, head_dim) in cases {
        let mut next = lcg();
        let q: Vec<f32> = (0..n_heads * seq_q * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..n_kv_heads * seq_k * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..n_kv_heads * seq_k * head_dim).map(|_| next()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let qg = ctx.upload_f32(&q, "q");
        let kg = ctx.upload_f32(&k, "k");
        let vg = ctx.upload_f32(&v, "v");
        let og = ctx.attention(
            &qg, &kg, &vg, 1, n_heads, n_kv_heads, seq_q, seq_k, seq_k, head_dim, scale, true,
        );
        let got = ctx.download_f32(&og).unwrap();

        let want = cpu_attn(
            &q, &k, &v, n_heads, n_kv_heads, seq_q, seq_k, head_dim, scale,
        );
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "attn {n_heads}/{n_kv_heads} q{seq_q} k{seq_k} d{head_dim} max_err={max_err}"
        );
    }
}

#[test]
fn kv_cache_append_then_decode_matches_cpu() {
    // Simulate decode: a pre-allocated [n_kv_heads, max_seq, head_dim] cache that grows
    // one token per step via copy_range, with attention reading the max_seq-strided cache
    // (seq_k = cur_len < kv_stride = max_seq). Compare the last step against CPU.
    let Some(ctx) = ctx() else {
        return;
    };
    let (n_heads, n_kv_heads, head_dim, max_seq) = (4usize, 2usize, 64usize, 16usize);
    let steps = 6usize; // append 6 tokens
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut next = lcg();

    // GPU cache buffers, capacity [n_kv_heads, max_seq, head_dim].
    let kcache = ctx.alloc_f32(n_kv_heads * max_seq * head_dim, "kcache");
    let vcache = ctx.alloc_f32(n_kv_heads * max_seq * head_dim, "vcache");

    // Host mirrors to build the CPU reference (tightly packed [n_kv_heads, steps, head_dim]).
    let mut k_host = vec![0.0f32; n_kv_heads * steps * head_dim];
    let mut v_host = vec![0.0f32; n_kv_heads * steps * head_dim];

    for t in 0..steps {
        // New token's K/V for all kv-heads: [n_kv_heads, head_dim].
        let knew: Vec<f32> = (0..n_kv_heads * head_dim).map(|_| next()).collect();
        let vnew: Vec<f32> = (0..n_kv_heads * head_dim).map(|_| next()).collect();
        let kg = ctx.upload_f32(&knew, "knew");
        let vg = ctx.upload_f32(&vnew, "vnew");
        // Append per head into slot t of the max_seq-strided cache.
        for hh in 0..n_kv_heads {
            let dst = (hh * max_seq + t) * head_dim;
            let src = hh * head_dim;
            ctx.copy_range(&kcache, dst, &kg, src, head_dim);
            ctx.copy_range(&vcache, dst, &vg, src, head_dim);
            for c in 0..head_dim {
                k_host[(hh * steps + t) * head_dim + c] = knew[hh * head_dim + c];
                v_host[(hh * steps + t) * head_dim + c] = vnew[hh * head_dim + c];
            }
        }
    }

    // Decode query at the last position attends to all `steps` cached tokens.
    let q: Vec<f32> = (0..n_heads * head_dim).map(|_| next()).collect();
    let qg = ctx.upload_f32(&q, "q");
    let og = ctx.attention(
        &qg, &kcache, &vcache, 1, n_heads, n_kv_heads, 1, steps, max_seq, head_dim, scale, true,
    );
    let got = ctx.download_f32(&og).unwrap();

    let want = cpu_attn(
        &q, &k_host, &v_host, n_heads, n_kv_heads, 1, steps, head_dim, scale,
    );
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "kv-cache decode max_err={max_err}");
}

#[test]
fn kv_append_then_decode_matches_cpu_f32_and_f16() {
    // Same decode simulation as above, but the cache is filled via the kv_append
    // conversion kernel (which replaced the per-head copy_range loop) — once with
    // an f32 cache and, when the device has SHADER_F16, once with an f16 cache.
    // The f16 reference quantizes K/V to f16 on the host first, so the comparison
    // stays tight: the only difference left is f32 reduction order.
    let Some(ctx) = ctx() else {
        return;
    };
    let (n_heads, n_kv_heads, head_dim, max_seq) = (4usize, 2usize, 64usize, 16usize);
    let steps = 6usize;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // The f16 cache is u32-packed (core WGSL) — both variants run on any adapter.
    for f16 in [false, true] {
        let mut next = lcg();
        let alloc = |label: &str| {
            if f16 {
                ctx.alloc_f16(n_kv_heads * max_seq * head_dim, label)
            } else {
                ctx.alloc_f32(n_kv_heads * max_seq * head_dim, label)
            }
        };
        let kcache = alloc("kcache");
        let vcache = alloc("vcache");

        // Host mirrors hold the values the cache actually stores (f16-rounded
        // when the cache is f16).
        let store = |x: f32| {
            if f16 {
                half::f16::from_f32(x).to_f32()
            } else {
                x
            }
        };
        let mut k_host = vec![0.0f32; n_kv_heads * steps * head_dim];
        let mut v_host = vec![0.0f32; n_kv_heads * steps * head_dim];
        // First 3 positions land in ONE multi-token append (the batched-prefill
        // path, src layout [n_kv, chunk, head_dim]); the rest go one at a time
        // (the decode path, seq = 1).
        let chunk = 3usize;
        {
            let kc: Vec<f32> = (0..n_kv_heads * chunk * head_dim).map(|_| next()).collect();
            let vc: Vec<f32> = (0..n_kv_heads * chunk * head_dim).map(|_| next()).collect();
            let kg = ctx.upload_f32(&kc, "kchunk");
            let vg = ctx.upload_f32(&vc, "vchunk");
            ctx.kv_append(&kg, &kcache, n_kv_heads, chunk, head_dim, max_seq, 0, f16);
            ctx.kv_append(&vg, &vcache, n_kv_heads, chunk, head_dim, max_seq, 0, f16);
            for hh in 0..n_kv_heads {
                for s in 0..chunk {
                    for c in 0..head_dim {
                        let src = (hh * chunk + s) * head_dim + c;
                        k_host[(hh * steps + s) * head_dim + c] = store(kc[src]);
                        v_host[(hh * steps + s) * head_dim + c] = store(vc[src]);
                    }
                }
            }
        }
        for t in chunk..steps {
            let knew: Vec<f32> = (0..n_kv_heads * head_dim).map(|_| next()).collect();
            let vnew: Vec<f32> = (0..n_kv_heads * head_dim).map(|_| next()).collect();
            let kg = ctx.upload_f32(&knew, "knew");
            let vg = ctx.upload_f32(&vnew, "vnew");
            ctx.kv_append(&kg, &kcache, n_kv_heads, 1, head_dim, max_seq, t, f16);
            ctx.kv_append(&vg, &vcache, n_kv_heads, 1, head_dim, max_seq, t, f16);
            for hh in 0..n_kv_heads {
                for c in 0..head_dim {
                    k_host[(hh * steps + t) * head_dim + c] = store(knew[hh * head_dim + c]);
                    v_host[(hh * steps + t) * head_dim + c] = store(vnew[hh * head_dim + c]);
                }
            }
        }

        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| next()).collect();
        let qg = ctx.upload_f32(&q, "q");
        let og = if f16 {
            ctx.attention_f16kv(
                &qg, &kcache, &vcache, 1, n_heads, n_kv_heads, 1, steps, max_seq, head_dim, scale,
                true,
            )
        } else {
            ctx.attention(
                &qg, &kcache, &vcache, 1, n_heads, n_kv_heads, 1, steps, max_seq, head_dim, scale,
                true,
            )
        };
        let got = ctx.download_f32(&og).unwrap();
        let want = cpu_attn(
            &q, &k_host, &v_host, n_heads, n_kv_heads, 1, steps, head_dim, scale,
        );
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "kv_append decode (f16={f16}) max_err={max_err}"
        );
    }
}

#[test]
fn batched_recording_matches_immediate() {
    // A chain of dependent kernels (norm → matmul → swiglu → add) recorded into
    // one begin_batch/flush_batch submission must produce exactly what per-kernel
    // submissions produce — WebGPU executes commands in recording order, and
    // download_f32 flushes the open batch before reading.
    let Some(ctx) = ctx() else {
        return;
    };
    let (dim, n) = (256usize, 128usize);
    let mut next = lcg();
    let x: Vec<f32> = (0..dim).map(|_| next()).collect();
    let w_norm: Vec<f32> = (0..dim).map(|_| 0.5 + next().abs()).collect();
    let w_a: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let w_b: Vec<f32> = (0..n * dim).map(|_| next()).collect();

    let run = |batched: bool| -> Vec<f32> {
        if batched {
            ctx.begin_batch();
        }
        let xg = ctx.upload_f32(&x, "x");
        let ng = ctx.upload_f32(&w_norm, "wn");
        let ag = ctx.upload_f32(&w_a, "wa");
        let bg = ctx.upload_f32(&w_b, "wb");
        let h = ctx.rms_norm(&xg, &ng, 1, dim, 1e-5);
        let ga = ctx.matmul_nt(&h, &ag, 1, dim, n);
        let gb = ctx.matmul_nt(&h, &bg, 1, dim, n);
        let s = ctx.swiglu(&ga, &gb);
        let out = ctx.add(&s, &gb);
        ctx.download_f32(&out).unwrap() // flushes the batch when one is open
    };

    let immediate = run(false);
    let batched = run(true);
    assert_eq!(immediate.len(), batched.len());
    for (i, (a, b)) in immediate.iter().zip(&batched).enumerate() {
        assert!((a - b).abs() < 1e-6, "batched vs immediate diverge at {i}");
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
    let got = ctx
        .download_f32(&ctx.embed(&ig, &tg, ids.len(), dim))
        .unwrap();
    for (t, &id) in ids.iter().enumerate() {
        for i in 0..dim {
            assert!((got[t * dim + i] - table[id as usize * dim + i]).abs() < 1e-9);
        }
    }
}

/// Quantize f32 values into raw ggml Q8_0 blocks (34 bytes per 32 weights:
/// little-endian f16 scale + 32 int8), returning the block bytes **and** the
/// exact dequantized values (`f32(scale_f16) * f32(int8)`) the GPU kernel must
/// reproduce. Mirrors ggml's `quantize_row_q8_0_ref`.
fn quantize_q8_0_blocks(w: &[f32]) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(w.len() % 32, 0);
    let mut blocks = Vec::with_capacity(w.len() / 32 * 34);
    let mut dequant = Vec::with_capacity(w.len());
    for chunk in w.chunks_exact(32) {
        let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let d16 = half::f16::from_f32(d);
        blocks.extend_from_slice(&d16.to_le_bytes());
        let scale = d16.to_f32(); // the shader sees the f16-rounded scale
        for &v in chunk {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            blocks.push(q as u8);
            dequant.push(scale * q as f32);
        }
    }
    (blocks, dequant)
}

#[test]
fn matmul_nt_q8_0_resident_matches_dequant_reference() {
    let Some(ctx) = ctx() else {
        return;
    };
    // Decode-shaped (m=1, lm_head-ish k) and a small batch; k must be %32.
    for (m, k, n) in [(1usize, 896usize, 1536usize), (3, 64, 48), (17, 64, 48)] {
        let mut next = lcg();
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let (blocks, wd) = quantize_q8_0_blocks(&w);

        let xg = ctx.upload_f32(&x, "x");
        let wq = ctx.upload_q8_0(&blocks, n * k, "wq").expect("upload_q8_0");
        let got = ctx
            .download_f32(&ctx.matmul_nt_q8_0(&xg, &wq, m, k, n))
            .unwrap();

        // Reference: f32 matmul over the *dequantized* weights — the GPU kernel
        // computes the identical products, so only reduction order differs.
        let mut want = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[r * k + i] * wd[c * k + i];
                }
                want[r * n + c] = acc;
            }
        }
        let rel = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0f32, f32::max);
        assert!(rel < 1e-4, "q8_0 matmul {m}x{k}x{n} rel={rel}");
    }
}

#[test]
fn embed_q8_0_matches_dequant_reference() {
    let Some(ctx) = ctx() else {
        return;
    };
    let (vocab, dim) = (100usize, 320usize);
    let mut next = lcg();
    let table: Vec<f32> = (0..vocab * dim).map(|_| next()).collect();
    let (blocks, td) = quantize_q8_0_blocks(&table);
    let ids: Vec<u32> = vec![7, 0, 99, 42];

    let tq = ctx
        .upload_q8_0(&blocks, vocab * dim, "table")
        .expect("upload_q8_0");
    let ig = ctx.upload_u32(&ids, "ids");
    let got = ctx
        .download_f32(&ctx.embed_q8_0(&ig, &tq, ids.len(), dim))
        .unwrap();
    for (t, &id) in ids.iter().enumerate() {
        for i in 0..dim {
            let want = td[id as usize * dim + i];
            assert!(
                (got[t * dim + i] - want).abs() < 1e-6,
                "embed_q8_0 token {t} elem {i}"
            );
        }
    }
}

/// Build `nblocks` random-but-valid raw ggml Q4_K super-blocks (144 bytes each:
/// f16 d + f16 dmin + 12 packed 6-bit scale/min bytes + 128 qs bytes) and return
/// them together with their exact dequantized values per the ggml reference
/// (`d·sc·q4 − dmin·mn`, `get_scale_min_k4` scale unpacking). Random bytes
/// exercise every bit path — including the high-bit scale packing for
/// sub-blocks 4..7 that a value-roundtrip quantizer might leave untouched.
fn random_q4_k_blocks(nblocks: usize, next: &mut dyn FnMut() -> f32) -> (Vec<u8>, Vec<f32>) {
    fn get_scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
        if j < 4 {
            (s[j] & 63, s[j + 4] & 63)
        } else {
            (
                (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4),
                (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
            )
        }
    }
    let mut blocks = Vec::with_capacity(nblocks * 144);
    let mut dequant = Vec::with_capacity(nblocks * 256);
    for _ in 0..nblocks {
        // Small positive d/dmin keep dequant magnitudes ~O(0.1) (sc,q ≤ 63·15).
        let d = half::f16::from_f32(1.0e-4 * (1.0 + next().abs()));
        let dmin = half::f16::from_f32(1.0e-4 * (1.0 + next().abs()));
        blocks.extend_from_slice(&d.to_le_bytes());
        blocks.extend_from_slice(&dmin.to_le_bytes());
        let base = blocks.len();
        for _ in 0..12 {
            blocks.push((next().abs() * 255.0) as u8); // scales: fully random bits
        }
        for _ in 0..128 {
            blocks.push((next().abs() * 255.0) as u8); // qs: fully random nibbles
        }
        let scales = blocks[base..base + 12].to_vec();
        let qs = blocks[base + 12..base + 140].to_vec();
        let (df, mf) = (d.to_f32(), dmin.to_f32());
        for is in 0..8 {
            let (sc, mn) = get_scale_min_k4(is, &scales);
            let (d1, m1) = (df * sc as f32, mf * mn as f32);
            for l in 0..32 {
                let byte = qs[(is / 2) * 32 + l];
                let q = if is % 2 == 0 { byte & 0xF } else { byte >> 4 };
                dequant.push(d1 * q as f32 - m1);
            }
        }
    }
    (blocks, dequant)
}

#[test]
fn matmul_nt_q4_k_resident_matches_dequant_reference() {
    let Some(ctx) = ctx() else {
        return;
    };
    // Decode-shaped (m=1) and a small batch; k must be a multiple of 256.
    for (m, k, n) in [(1usize, 1536usize, 512usize), (3, 256, 48), (17, 256, 48)] {
        let mut next = lcg();
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let (blocks, wd) = random_q4_k_blocks(n * k / 256, &mut next);

        let xg = ctx.upload_f32(&x, "x");
        let wq = ctx.upload_q4_k(&blocks, n * k, "wq").expect("upload_q4_k");
        let got = ctx
            .download_f32(&ctx.matmul_nt_q4_k(&xg, &wq, m, k, n))
            .unwrap();

        let mut want = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[r * k + i] * wd[c * k + i];
                }
                want[r * n + c] = acc;
            }
        }
        let rel = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0f32, f32::max);
        assert!(rel < 1e-4, "q4_k matmul {m}x{k}x{n} rel={rel}");
    }
}

#[test]
fn embed_q4_k_matches_dequant_reference() {
    let Some(ctx) = ctx() else {
        return;
    };
    let (vocab, dim) = (50usize, 512usize);
    let mut next = lcg();
    let (blocks, td) = random_q4_k_blocks(vocab * dim / 256, &mut next);
    let ids: Vec<u32> = vec![3, 0, 49, 21];

    let tq = ctx
        .upload_q4_k(&blocks, vocab * dim, "table")
        .expect("upload_q4_k");
    let ig = ctx.upload_u32(&ids, "ids");
    let got = ctx
        .download_f32(&ctx.embed_q4_k(&ig, &tq, ids.len(), dim))
        .unwrap();
    for (t, &id) in ids.iter().enumerate() {
        for i in 0..dim {
            let want = td[id as usize * dim + i];
            assert!(
                (got[t * dim + i] - want).abs() < 1e-6,
                "embed_q4_k token {t} elem {i}: got {} want {want}",
                got[t * dim + i]
            );
        }
    }
}

/// Build `nblocks` random-but-valid raw ggml Q6_K super-blocks (210 bytes each:
/// ql[128] + qh[64] + 16 signed int8 scales + f16 d) and return them with their
/// exact dequantized values per the ggml reference — `d·sc·(q−32)` with the
/// +0/+2/+4/+6 scale-offset indexing per 128-half (the historical token-salad
/// bug class). Random bytes exercise every ql/qh bit path and negative scales.
fn random_q6_k_blocks(nblocks: usize, next: &mut dyn FnMut() -> f32) -> (Vec<u8>, Vec<f32>) {
    let mut blocks = Vec::with_capacity(nblocks * 210);
    let mut dequant = Vec::with_capacity(nblocks * 256);
    for _ in 0..nblocks {
        let base = blocks.len();
        for _ in 0..192 {
            blocks.push((next().abs() * 255.0) as u8); // ql[128] + qh[64]
        }
        for _ in 0..16 {
            blocks.push((next() * 127.0) as i8 as u8); // signed scales
        }
        // Small d keeps dequant magnitudes ~O(0.1): |sc·(q−32)| ≤ 127·32.
        let d = half::f16::from_f32(2.0e-5 * (1.0 + next().abs()));
        blocks.extend_from_slice(&d.to_le_bytes());
        let ql = blocks[base..base + 128].to_vec();
        let qh = blocks[base + 128..base + 192].to_vec();
        let sc = blocks[base + 192..base + 208].to_vec();
        let df = d.to_f32();
        for e in 0..256usize {
            let (h, r) = (e / 128, e % 128);
            let (g, l) = (r / 32, r % 32);
            let q = ((ql[h * 64 + (g % 2) * 32 + l] >> ((g / 2) * 4)) & 0xF)
                | (((qh[h * 32 + l] >> (g * 2)) & 3) << 4);
            let scale = sc[h * 8 + 2 * g + l / 16] as i8 as f32;
            dequant.push(df * scale * (q as i32 - 32) as f32);
        }
    }
    (blocks, dequant)
}

#[test]
fn matmul_nt_q6_k_resident_matches_dequant_reference() {
    let Some(ctx) = ctx() else {
        return;
    };
    for (m, k, n) in [(1usize, 1536usize, 512usize), (3, 256, 48), (17, 256, 48)] {
        let mut next = lcg();
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let (blocks, wd) = random_q6_k_blocks(n * k / 256, &mut next);

        let xg = ctx.upload_f32(&x, "x");
        let wq = ctx.upload_q6_k(&blocks, n * k, "wq").expect("upload_q6_k");
        let got = ctx
            .download_f32(&ctx.matmul_nt_q6_k(&xg, &wq, m, k, n))
            .unwrap();

        let mut want = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[r * k + i] * wd[c * k + i];
                }
                want[r * n + c] = acc;
            }
        }
        let rel = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0f32, f32::max);
        assert!(rel < 1e-4, "q6_k matmul {m}x{k}x{n} rel={rel}");
    }
}

#[test]
fn embed_q6_k_matches_dequant_reference() {
    let Some(ctx) = ctx() else {
        return;
    };
    let (vocab, dim) = (50usize, 512usize);
    let mut next = lcg();
    let (blocks, td) = random_q6_k_blocks(vocab * dim / 256, &mut next);
    let ids: Vec<u32> = vec![11, 0, 49, 30];

    let tq = ctx
        .upload_q6_k(&blocks, vocab * dim, "table")
        .expect("upload_q6_k");
    let ig = ctx.upload_u32(&ids, "ids");
    let got = ctx
        .download_f32(&ctx.embed_q6_k(&ig, &tq, ids.len(), dim))
        .unwrap();
    for (t, &id) in ids.iter().enumerate() {
        for i in 0..dim {
            let want = td[id as usize * dim + i];
            assert!(
                (got[t * dim + i] - want).abs() < 1e-6,
                "embed_q6_k token {t} elem {i}: got {} want {want}",
                got[t * dim + i]
            );
        }
    }
}

#[test]
fn upload_q6_k_rejects_bad_block_count() {
    let Some(ctx) = ctx() else {
        return;
    };
    assert!(ctx.upload_q6_k(&[0u8; 209], 256, "bad").is_err());
    assert!(ctx.upload_q6_k(&[0u8; 210], 200, "bad").is_err());
}

#[test]
fn upload_q4_k_rejects_bad_block_count() {
    let Some(ctx) = ctx() else {
        return;
    };
    // 256 elements need exactly one 144-byte block — hand it 143.
    assert!(ctx.upload_q4_k(&[0u8; 143], 256, "bad").is_err());
    // numel not a multiple of the 256-weight super-block.
    assert!(ctx.upload_q4_k(&[0u8; 144], 200, "bad").is_err());
}

#[test]
fn upload_q8_0_rejects_bad_block_count() {
    let Some(ctx) = ctx() else {
        return;
    };
    // 64 elements need exactly 2 blocks (68 bytes) — hand it 67.
    assert!(ctx.upload_q8_0(&[0u8; 67], 64, "bad").is_err());
    // numel not a multiple of the 32-weight block size.
    assert!(ctx.upload_q8_0(&[0u8; 34], 30, "bad").is_err());
}

/// CPU LayerNorm reference (f32) with weight + bias.
fn cpu_layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean: f32 = row.iter().sum::<f32>() / dim as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = (row[i] - mean) * inv * weight[i] + bias[i];
        }
    }
    out
}

#[test]
fn layer_norm_matches_cpu() {
    let Some(ctx) = ctx() else {
        eprintln!("no GPU adapter — skipping wgpu layer_norm test");
        return;
    };
    // Non-power-of-two dim/rows to exercise the strided reduction tail.
    let (rows, dim) = (7usize, 1536usize);
    let mut next = lcg();
    let x: Vec<f32> = (0..rows * dim).map(|_| next() * 3.0).collect();
    let weight: Vec<f32> = (0..dim).map(|_| 0.5 + next().abs()).collect();
    let bias: Vec<f32> = (0..dim).map(|_| next() * 0.1).collect();
    let eps = 1e-5f32;

    let xg = ctx.upload_f32(&x, "x");
    let wg = ctx.upload_f32(&weight, "w");
    let bg = ctx.upload_f32(&bias, "b");
    let outg = ctx.layer_norm(&xg, &wg, &bg, rows, dim, eps);
    let got = ctx.download_f32(&outg).expect("download");

    let want = cpu_layer_norm(&x, &weight, &bias, rows, dim, eps);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "wgpu layer_norm vs cpu max_err={max_err}");
}

/// CPU exact-erf GELU reference, matching the elementwise A&S erf approximation.
fn cpu_gelu_erf(x: &[f32]) -> Vec<f32> {
    // Constants mirror the WGSL `erf_approx` (A&S 7.1.26) verbatim for a faithful
    // reference; allow the precision lints rather than re-rounding them apart.
    #[allow(clippy::excessive_precision, clippy::unreadable_literal)]
    fn erf_approx(x: f32) -> f32 {
        let s = x.signum();
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let y = 1.0
            - (0.254829592
                + (-0.284496736 + (1.421413741 + (-1.453152027 + 1.061405429 * t) * t) * t) * t)
                * t
                * (-ax * ax).exp();
        s * y
    }
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + erf_approx(v * std::f32::consts::FRAC_1_SQRT_2)))
        .collect()
}

#[test]
fn gelu_erf_matches_cpu() {
    let Some(ctx) = ctx() else {
        eprintln!("no GPU adapter — skipping wgpu gelu_erf test");
        return;
    };
    let n = 4096usize + 37; // exercise the 2-D dispatch tail
    let mut next = lcg();
    let x: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();

    let xg = ctx.upload_f32(&x, "x");
    let outg = ctx.gelu_erf(&xg);
    let got = ctx.download_f32(&outg).expect("download");

    let want = cpu_gelu_erf(&x);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "wgpu gelu_erf vs cpu max_err={max_err}");
}

#[test]
fn add_bias_matches_cpu() {
    let Some(ctx) = ctx() else {
        eprintln!("no GPU adapter — skipping wgpu add_bias test");
        return;
    };
    let (rows, dim) = (13usize, 320usize);
    let mut next = lcg();
    let x: Vec<f32> = (0..rows * dim).map(|_| next()).collect();
    let bias: Vec<f32> = (0..dim).map(|_| next()).collect();

    let xg = ctx.upload_f32(&x, "x");
    let bg = ctx.upload_f32(&bias, "b");
    let got = ctx.download_f32(&ctx.add_bias(&xg, &bg, dim)).unwrap();

    for r in 0..rows {
        for i in 0..dim {
            let want = x[r * dim + i] + bias[i];
            assert!((got[r * dim + i] - want).abs() < 1e-5);
        }
    }
}

#[test]
fn transpose_heads_matches_cpu() {
    let Some(ctx) = ctx() else {
        eprintln!("no GPU adapter — skipping wgpu transpose_heads test");
        return;
    };
    // [outer=seq, inner=heads, hd] → [heads, seq, hd] and back is identity.
    let (seq, heads, hd) = (7usize, 5usize, 16usize);
    let mut next = lcg();
    let x: Vec<f32> = (0..seq * heads * hd).map(|_| next()).collect();

    let xg = ctx.upload_f32(&x, "x");
    let tg = ctx.transpose_heads(&xg, seq, heads, hd); // → [heads, seq, hd]
    let t = ctx.download_f32(&tg).unwrap();

    for s in 0..seq {
        for h in 0..heads {
            for c in 0..hd {
                let src = (s * heads + h) * hd + c;
                let dst = (h * seq + s) * hd + c;
                assert!((t[dst] - x[src]).abs() < 1e-9);
            }
        }
    }
    // Round-trip back to [seq, heads, hd].
    let bg = ctx.transpose_heads(&tg, heads, seq, hd);
    let b = ctx.download_f32(&bg).unwrap();
    for (i, (&a, &c)) in x.iter().zip(&b).enumerate() {
        assert!((a - c).abs() < 1e-9, "roundtrip mismatch at {i}");
    }
}

/// Non-causal (full) attention reference: every query attends to all seq_k keys.
#[allow(clippy::too_many_arguments)]
fn cpu_attn_full(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_heads * seq_q * head_dim];
    for h in 0..n_heads {
        for qi in 0..seq_q {
            let qbase = (h * seq_q + qi) * head_dim;
            let mut scores = vec![0.0f32; seq_k];
            let mut mx = f32::NEG_INFINITY;
            for (ki, sc) in scores.iter_mut().enumerate() {
                let kbase = (h * seq_k + ki) * head_dim;
                let mut d = 0.0f32;
                for c in 0..head_dim {
                    d += q[qbase + c] * k[kbase + c];
                }
                *sc = d * scale;
                mx = mx.max(*sc);
            }
            let mut den = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - mx).exp();
                den += *sc;
            }
            for (ki, &w) in scores.iter().enumerate() {
                let vbase = (h * seq_k + ki) * head_dim;
                let pw = w / den;
                for c in 0..head_dim {
                    out[qbase + c] += pw * v[vbase + c];
                }
            }
        }
    }
    out
}

#[test]
fn non_causal_attention_matches_cpu() {
    let Some(ctx) = ctx() else {
        eprintln!("no GPU adapter — skipping wgpu non-causal attention test");
        return;
    };
    // Cross-attention shape: query seq_q ≠ key seq_k, MHA, full (non-causal).
    let (n_heads, seq_q, seq_k, head_dim) = (4usize, 6usize, 9usize, 64usize);
    let mut next = lcg();
    let q: Vec<f32> = (0..n_heads * seq_q * head_dim).map(|_| next()).collect();
    let k: Vec<f32> = (0..n_heads * seq_k * head_dim).map(|_| next()).collect();
    let v: Vec<f32> = (0..n_heads * seq_k * head_dim).map(|_| next()).collect();
    let scale = 1.0 / (head_dim as f32).sqrt();

    let qg = ctx.upload_f32(&q, "q");
    let kg = ctx.upload_f32(&k, "k");
    let vg = ctx.upload_f32(&v, "v");
    let og = ctx.attention(
        &qg, &kg, &vg, 1, n_heads, n_heads, seq_q, seq_k, seq_k, head_dim, scale, false,
    );
    let got = ctx.download_f32(&og).unwrap();
    let want = cpu_attn_full(&q, &k, &v, n_heads, seq_q, seq_k, head_dim, scale);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "non-causal attention max_err={max_err}");
}
