// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! The ISTFTNet decoder + generator: from the length-regulated text features
//! (`asr`), the F0/N curves, and the decoder (timbre) style it synthesises the
//! 24 kHz waveform. AdaIN+Snake residual blocks, ConvTranspose upsampling, an NSF
//! harmonic source injected into the conv stream, and an iSTFT head.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use sapient_core::{Shape, Tensor};

use super::super::conv::{conv1d, conv_transpose1d, snake};
use super::loader::KokoroConfig;
use super::ops::{ada_in_1d, istft, leaky_relu_inplace, stft_transform};
use super::predictor::adain_res_blk1d;

fn get<'a>(w: &'a HashMap<String, Tensor>, k: &str) -> Result<&'a Tensor> {
    w.get(k)
        .ok_or_else(|| anyhow!("kokoro: missing weight {k}"))
}

/// Concatenate `[1, Ci, L]` tensors along the channel axis → `[1, ΣCi, L]`.
fn cat_channels(parts: &[&Tensor]) -> Result<Tensor> {
    let l = parts[0].shape().dims()[parts[0].shape().dims().len() - 1];
    let mut total_c = 0usize;
    for p in parts {
        let d = p.shape().dims();
        if d[d.len() - 1] != l {
            return Err(anyhow!("cat_channels length mismatch"));
        }
        total_c += d[d.len() - 2];
    }
    let mut out = vec![0.0f32; total_c * l];
    let mut base = 0usize;
    for p in parts {
        let d = p.shape().dims();
        let c = d[d.len() - 2];
        let v = p.to_f32_vec();
        out[base * l..(base + c) * l].copy_from_slice(&v);
        base += c;
    }
    Tensor::from_f32(&out, Shape::new([1, total_c, l])).map_err(|e| anyhow!("{e}"))
}

/// `get_padding(kernel, dilation)` = `dilation*(kernel-1)/2` (HiFi-GAN "same").
fn get_padding(k: usize, d: usize) -> usize {
    d * (k - 1) / 2
}

/// `AdaINResBlock1.forward(x, s)` — `x` is `[1, C, L]`, returns `[1, C, L]`.
/// Three (AdaIN → Snake → dilated conv → AdaIN → Snake → conv) residual sub-blocks.
fn adain_resblock1(
    w: &HashMap<String, Tensor>,
    prefix: &str,
    x: &Tensor,
    s: &[f32],
    kernel: usize,
    dilations: &[usize; 3],
) -> Result<Tensor> {
    let mut x = x.clone();
    let c = x.shape().dims()[1];
    let l = x.shape().dims()[2];
    #[allow(clippy::needless_range_loop)] // j indexes weights by name and dilations
    for j in 0..3 {
        let xt = ada_in_1d(
            &x,
            s,
            get(w, &format!("{prefix}.adain1.{j}.fc.weight"))?,
            Some(get(w, &format!("{prefix}.adain1.{j}.fc.bias"))?),
            1e-5,
        )?;
        let a1 = get(w, &format!("{prefix}.alpha1.{j}"))?.to_f32_vec(); // [1,C,1] → [C]
        let a1 = Tensor::from_f32(&a1, Shape::new([c])).map_err(|e| anyhow!("{e}"))?;
        let xt = snake(&xt, &a1)?;
        let xt = conv1d(
            &xt,
            get(w, &format!("{prefix}.convs1.{j}.weight"))?,
            Some(get(w, &format!("{prefix}.convs1.{j}.bias"))?),
            get_padding(kernel, dilations[j]),
            1,
            dilations[j],
            1,
        )?;
        let xt = ada_in_1d(
            &xt,
            s,
            get(w, &format!("{prefix}.adain2.{j}.fc.weight"))?,
            Some(get(w, &format!("{prefix}.adain2.{j}.fc.bias"))?),
            1e-5,
        )?;
        let a2 = get(w, &format!("{prefix}.alpha2.{j}"))?.to_f32_vec();
        let a2 = Tensor::from_f32(&a2, Shape::new([c])).map_err(|e| anyhow!("{e}"))?;
        let xt = snake(&xt, &a2)?;
        let xt = conv1d(
            &xt,
            get(w, &format!("{prefix}.convs2.{j}.weight"))?,
            Some(get(w, &format!("{prefix}.convs2.{j}.bias"))?),
            get_padding(kernel, 1),
            1,
            1,
            1,
        )?;
        // residual add
        let xv = x.to_f32_vec();
        let tv = xt.to_f32_vec();
        let summed: Vec<f32> = xv.iter().zip(tv.iter()).map(|(a, b)| a + b).collect();
        x = Tensor::from_f32(&summed, Shape::new([1, c, l])).map_err(|e| anyhow!("{e}"))?;
    }
    Ok(x)
}

/// ReflectionPad1d((1, 0)): prepend a reflected sample → length `L+1`.
fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let c = x.shape().dims()[1];
    let l = x.shape().dims()[2];
    let v = x.to_f32_vec();
    let mut out = vec![0.0f32; c * (l + 1)];
    for ci in 0..c {
        out[ci * (l + 1)] = v[ci * l + 1.min(l - 1)]; // reflect: new[0] = x[1]
        out[ci * (l + 1) + 1..ci * (l + 1) + 1 + l].copy_from_slice(&v[ci * l..ci * l + l]);
    }
    Tensor::from_f32(&out, Shape::new([1, c, l + 1])).map_err(|e| anyhow!("{e}"))
}

/// Generator: `x` is `[1, 512, 2T]`, `f0` the `[2T]` pitch curve, `s` the 128-d
/// decoder style. Returns the waveform.
fn generator(
    w: &HashMap<String, Tensor>,
    cfg: &KokoroConfig,
    x: &Tensor,
    s: &[f32],
    f0: &[f32],
    initial_cycles: f64,
) -> Result<Vec<f32>> {
    let g = &cfg.istftnet;
    let n_fft = g.gen_istft_n_fft; // 20
    let hop = g.gen_istft_hop_size; // 5
    let upsample_scale: usize = g.upsample_rates.iter().product::<usize>() * hop; // 300

    // ── NSF harmonic source → STFT → [22, frames] ────────────────────────────
    let l_lin_w = get(w, "decoder.generator.m_source.l_linear.weight")?.to_f32_vec(); // [9]
    let l_lin_b = get(w, "decoder.generator.m_source.l_linear.bias")?.to_f32_vec()[0];
    let har = super::ops::nsf_harmonic_source_from(
        f0,
        &l_lin_w,
        l_lin_b,
        24000.0,
        8,
        0.1,
        10.0,
        upsample_scale,
        initial_cycles,
    );
    let (mag, phase, frames) = stft_transform(&har, n_fft, hop);
    let fbins = n_fft / 2 + 1; // 11
    let mut har_cat = vec![0.0f32; (2 * fbins) * frames];
    har_cat[..fbins * frames].copy_from_slice(&mag);
    har_cat[fbins * frames..].copy_from_slice(&phase);
    let har_cat = Tensor::from_f32(&har_cat, Shape::new([1, 2 * fbins, frames]))
        .map_err(|e| anyhow!("{e}"))?;

    let timing = std::env::var("SAPIENT_KOKORO_TIMING").is_ok();
    let mut t_src_convs = 0u128;
    let mut t_resblocks = 0u128;

    let num_up = g.upsample_rates.len(); // 2
    let mut x = x.clone();
    for i in 0..num_up {
        let _t0 = std::time::Instant::now();
        // x = leaky_relu(x, 0.1)
        let mut xv = x.to_f32_vec();
        leaky_relu_inplace(&mut xv, 0.1);
        let xd = x.shape().dims().to_vec();
        x = Tensor::from_f32(&xv, Shape::new([1, xd[1], xd[2]])).map_err(|e| anyhow!("{e}"))?;

        // The noise-source branch (noise_conv → noise_res) and the upsample branch
        // (ConvTranspose1d → reflection-pad) are independent until they're summed,
        // so run them concurrently — each is a substantial serial cost.
        let (stride_ns, pad_ns) = if i + 1 < num_up {
            let stride_f0: usize = g.upsample_rates[i + 1..].iter().product();
            (stride_f0, stride_f0.div_ceil(2))
        } else {
            (1, 0)
        };
        let nr_k = if i + 1 < num_up { 7 } else { 11 };
        let u = g.upsample_rates[i];
        let k = g.upsample_kernel_sizes[i];
        let (source_res, up_res) = rayon::join(
            || -> Result<Tensor> {
                // x_source = noise_res(noise_convs[i](har)) — noise_convs is a plain
                // Conv1d (kernel from the weight tensor; only stride+pad needed).
                let xs = conv1d(
                    &har_cat,
                    get(w, &format!("decoder.generator.noise_convs.{i}.weight"))?,
                    Some(get(w, &format!("decoder.generator.noise_convs.{i}.bias"))?),
                    pad_ns,
                    stride_ns,
                    1,
                    1,
                )?;
                adain_resblock1(
                    w,
                    &format!("decoder.generator.noise_res.{i}"),
                    &xs,
                    s,
                    nr_k,
                    &[1, 3, 5],
                )
            },
            || -> Result<Tensor> {
                let mut xu = conv_transpose1d(
                    &x,
                    get(w, &format!("decoder.generator.ups.{i}.weight"))?,
                    Some(get(w, &format!("decoder.generator.ups.{i}.bias"))?),
                    u,
                    (k - u) / 2,
                )?;
                if i == num_up - 1 {
                    xu = reflection_pad_left1(&xu)?;
                }
                Ok(xu)
            },
        );
        let x_source = source_res?;
        x = up_res?;

        // x = x + x_source
        let a = x.to_f32_vec();
        let b = x_source.to_f32_vec();
        if a.len() != b.len() {
            return Err(anyhow!(
                "generator stage {i}: x {} vs x_source {}",
                a.len(),
                b.len()
            ));
        }
        let xd = x.shape().dims().to_vec();
        let summed: Vec<f32> = a.iter().zip(b.iter()).map(|(p, q)| p + q).collect();
        x = Tensor::from_f32(&summed, Shape::new([1, xd[1], xd[2]])).map_err(|e| anyhow!("{e}"))?;

        if timing {
            t_src_convs += _t0.elapsed().as_millis();
        }
        let _t1 = std::time::Instant::now();
        // resblocks: the `num_kernels` blocks are independent (same input, summed
        // then averaged) — run them across cores. This is the decoder's hot path.
        let nk = g.resblock_kernel_sizes.len();
        let blocks: Vec<Vec<f32>> = (0..nk)
            .into_par_iter()
            .map(|j| {
                let idx = i * nk + j;
                let d = &g.resblock_dilation_sizes[j];
                adain_resblock1(
                    w,
                    &format!("decoder.generator.resblocks.{idx}"),
                    &x,
                    s,
                    g.resblock_kernel_sizes[j],
                    &[d[0], d[1], d[2]],
                )
                .map(|t| t.to_f32_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut avg = blocks[0].clone();
        for b in &blocks[1..] {
            for (p, q) in avg.iter_mut().zip(b.iter()) {
                *p += *q;
            }
        }
        let n = nk as f32;
        for v in avg.iter_mut() {
            *v /= n;
        }
        let xd = x.shape().dims().to_vec();
        x = Tensor::from_f32(&avg, Shape::new([1, xd[1], xd[2]])).map_err(|e| anyhow!("{e}"))?;
        if timing {
            t_resblocks += _t1.elapsed().as_millis();
        }
    }
    let _tp = std::time::Instant::now();

    // x = leaky_relu(x) [default 0.01]; conv_post; exp/sin; iSTFT
    let mut xv = x.to_f32_vec();
    leaky_relu_inplace(&mut xv, 0.01);
    let xd = x.shape().dims().to_vec();
    let x = Tensor::from_f32(&xv, Shape::new([1, xd[1], xd[2]])).map_err(|e| anyhow!("{e}"))?;
    let post = conv1d(
        &x,
        get(w, "decoder.generator.conv_post.weight")?,
        Some(get(w, "decoder.generator.conv_post.bias")?),
        3,
        1,
        1,
        1,
    )?;
    let pd = post.shape().dims().to_vec(); // [1, 22, L]
    let lf = pd[2];
    let pv = post.to_f32_vec();
    let mut spec = vec![0.0f32; fbins * lf];
    let mut ph = vec![0.0f32; fbins * lf];
    for k in 0..fbins {
        for t in 0..lf {
            spec[k * lf + t] = pv[k * lf + t].exp();
            ph[k * lf + t] = pv[(fbins + k) * lf + t].sin();
        }
    }
    let wav = istft(&spec, &ph, fbins, lf, n_fft, hop);
    if timing {
        use std::sync::atomic::Ordering::Relaxed;
        let im2col = sapient_backends_cpu::kernels::conv2d::IM2COL_NS.swap(0, Relaxed) / 1_000_000;
        let gemm = sapient_backends_cpu::kernels::conv2d::GEMM_NS.swap(0, Relaxed) / 1_000_000;
        let convt = crate::forward::conv::CONVT_NS.swap(0, Relaxed) / 1_000_000;
        let snake_ms = crate::forward::conv::SNAKE_NS.swap(0, Relaxed) / 1_000_000;
        let adain_ms = super::ops::ADAIN_NS.swap(0, Relaxed) / 1_000_000;
        eprintln!(
            "    [ops] im2col {im2col} ms · gemm {gemm} ms · conv_transpose {convt} ms · snake {snake_ms} ms · adain {adain_ms} ms (thread-summed)"
        );
        eprintln!(
            "    [gen] src+convs {t_src_convs} ms · resblocks {t_resblocks} ms · post+istft {} ms",
            _tp.elapsed().as_millis()
        );
    }
    Ok(wav)
}

/// Full ISTFTNet decoder: `asr` `[1, 512, T]`, `f0`/`n` `[2T]`, `s` 128-d style.
pub fn decode(
    w: &HashMap<String, Tensor>,
    cfg: &KokoroConfig,
    asr: &Tensor,
    f0: &[f32],
    n: &[f32],
    s: &[f32],
) -> Result<Vec<f32>> {
    decode_with_phase(w, cfg, asr, f0, n, s, 0.0)
}

/// [`decode`] with an explicit NSF starting phase (accumulated cycles) — the
/// windowed-streaming entry: a mid-utterance window passes the analytic phase
/// the full-utterance cumsum would have reached at its first f0 sample, so the
/// harmonic source is continuous across window joins.
#[allow(clippy::too_many_arguments)]
pub fn decode_with_phase(
    w: &HashMap<String, Tensor>,
    cfg: &KokoroConfig,
    asr: &Tensor,
    f0: &[f32],
    n: &[f32],
    s: &[f32],
    initial_cycles: f64,
) -> Result<Vec<f32>> {
    let t = asr.shape().dims()[2];
    let f0_t = Tensor::from_f32(f0, Shape::new([1, 1, f0.len()])).map_err(|e| anyhow!("{e}"))?;
    let n_t = Tensor::from_f32(n, Shape::new([1, 1, n.len()])).map_err(|e| anyhow!("{e}"))?;
    // F0_conv / N_conv: weight-normed Conv1d(1,1,3,stride2,pad1) → length T
    let f0d = conv1d(
        &f0_t,
        get(w, "decoder.F0_conv.weight")?,
        Some(get(w, "decoder.F0_conv.bias")?),
        1,
        2,
        1,
        1,
    )?;
    let nd = conv1d(
        &n_t,
        get(w, "decoder.N_conv.weight")?,
        Some(get(w, "decoder.N_conv.bias")?),
        1,
        2,
        1,
        1,
    )?;

    let x = cat_channels(&[asr, &f0d, &nd])?; // [1, 514, T]
    let mut x = adain_res_blk1d(w, "decoder.encode", &x, s, false)?; // [1, 1024, T]
    let asr_res = conv1d(
        asr,
        get(w, "decoder.asr_res.0.weight")?,
        Some(get(w, "decoder.asr_res.0.bias")?),
        0,
        1,
        1,
        1,
    )?; // [1,64,T]

    let mut res = true;
    for blk in 0..4 {
        if res {
            x = cat_channels(&[&x, &asr_res, &f0d, &nd])?; // [1, +66, T]
        }
        let upsample = blk == 3;
        x = adain_res_blk1d(w, &format!("decoder.decode.{blk}"), &x, s, upsample)?;
        if upsample {
            res = false;
        }
    }
    let _ = t;
    generator(w, cfg, &x, s, f0, initial_cycles)
}
