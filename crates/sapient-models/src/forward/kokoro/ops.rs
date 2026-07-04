//! New CPU primitives required by Kokoro-82M (StyleTTS2 + ISTFTNet) that the
//! existing SAPIENT kernel stack does not already provide. Everything here is
//! purely additive — the chat / serve / transcribe / Orpheus-speak paths never
//! touch it.
//!
//! Implemented so far:
//! - [`lstm_bidirectional`] — PyTorch `nn.LSTM(bidirectional=True)` for one
//!   layer. Kokoro uses bidirectional LSTMs in the DurationEncoder, the shared
//!   prosody LSTM, and the prosodic `text_encoder`.
//! - [`stft_transform`] / [`istft`] — `torch.stft` / `torch.istft`-equivalent
//!   forward/inverse short-time Fourier transforms (`center=True`, periodic
//!   Hann, one-sided). The ISTFTNet generator head runs `istft` on its
//!   `conv_post` magnitude/phase output; `stft_transform` analyses the NSF
//!   harmonic source. n_fft is tiny (20), so a direct DFT is plenty fast.
//!
//! These run a handful of times per utterance (sequence length = phoneme
//! count, typically < 510), so straightforward correct implementations are fine.

use std::f32::consts::PI;

use anyhow::{anyhow, Result};
use sapient_backends_cpu::kernels::matmul::matmul_nt;
use sapient_core::{Shape, Tensor};

/// Fast linear `y = x·Wᵀ + b` via SAPIENT's SIMD+rayon `matmul_nt`. `x` is a
/// `[L, in]` tensor, `w` a `[out, in]` tensor (PyTorch `nn.Linear` layout). Much
/// faster than [`linear2d`] for the large projections (ALBERT FFN, etc.).
pub fn linear_mm(x: &Tensor, w: &Tensor, b: Option<&[f32]>) -> Result<Vec<f32>> {
    let y = matmul_nt(x, w).map_err(|e| anyhow!("{e}"))?; // [L, out]
    let mut v = y.to_f32_vec();
    if let Some(b) = b {
        let out = w.shape().dims()[0];
        for row in v.chunks_mut(out) {
            for (o, val) in row.iter_mut().enumerate() {
                *val += b[o];
            }
        }
    }
    Ok(v)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Periodic Hann window `w[m] = 0.5 - 0.5·cos(2π·m/N)`, m∈[0,N) — matches
/// `torch.hann_window(N, periodic=True)` (divisor N, not N-1).
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|m| 0.5 - 0.5 * (2.0 * PI * m as f32 / n as f32).cos())
        .collect()
}

/// `torch.stft`-equivalent forward transform (`center=True`, reflect padding,
/// periodic Hann window, `onesided=True`, `win_length == n_fft`). Returns
/// `(magnitude, phase)` each laid out `[F, T]` row-major with `F = n_fft/2 + 1`
/// and `T = 1 + len/hop` frames — mirroring `TorchSTFT.transform` (which returns
/// `abs` and `angle`). Used to analyse the NSF harmonic source.
pub fn stft_transform(signal: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let pad = n_fft / 2;
    let l = signal.len();
    // Reflect-pad `pad` samples each side (no boundary repeat), like np.pad
    // 'reflect' / torch.stft center padding.
    let mut padded = vec![0.0f32; l + 2 * pad];
    for (j, p) in padded.iter_mut().enumerate().take(pad) {
        *p = signal[pad - j]; // left: x[pad], x[pad-1], …, x[1]
    }
    padded[pad..pad + l].copy_from_slice(signal);
    for j in 0..pad {
        padded[l + pad + j] = signal[l - 2 - j]; // right: x[L-2], x[L-3], …
    }

    let win = hann_periodic(n_fft);
    let f = n_fft / 2 + 1;
    let t = 1 + l / hop; // (l_padded - n_fft)/hop + 1 = l/hop + 1
    let mut mag = vec![0.0f32; f * t];
    let mut phase = vec![0.0f32; f * t];

    for ti in 0..t {
        let base = ti * hop;
        for k in 0..f {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for m in 0..n_fft {
                let fw = padded[base + m] * win[m];
                let ang = 2.0 * PI * k as f32 * m as f32 / n_fft as f32;
                re += fw * ang.cos();
                im -= fw * ang.sin();
            }
            mag[k * t + ti] = (re * re + im * im).sqrt();
            phase[k * t + ti] = im.atan2(re);
        }
    }
    (mag, phase, t)
}

/// `torch.istft`-equivalent inverse (`center=True`, periodic Hann, one-sided,
/// `win_length == n_fft`). `mag`/`phase` are `[F, T]` row-major (`F = n_fft/2+1`)
/// — the ISTFTNet generator passes `exp(.)` magnitude and `sin(.)` phase. Returns
/// a waveform of length `hop·(T-1)` (the `n_fft/2`-per-side center trim).
///
/// The accuracy-critical details (matching the shipped `TorchSTFT`, not the
/// approximate ONNX `CustomSTFT`): the one-sided→real inverse uses the **1,2,1**
/// bin weighting (interior bins doubled, DC and Nyquist not), and the overlap-add
/// is normalised by the overlap-add of the **squared** synthesis window.
pub fn istft(mag: &[f32], phase: &[f32], f: usize, t: usize, n_fft: usize, hop: usize) -> Vec<f32> {
    debug_assert_eq!(f, n_fft / 2 + 1);
    let win = hann_periodic(n_fft);
    // Bin weights for the one-sided → real inverse DFT (even n_fft).
    let coef = |k: usize| -> f32 {
        if k == 0 || k == n_fft / 2 {
            1.0
        } else {
            2.0
        }
    };
    let full = (t - 1) * hop + n_fft;
    let mut y = vec![0.0f32; full];
    let mut wsum = vec![0.0f32; full];
    let inv_n = 1.0 / n_fft as f32;

    let mut frame = vec![0.0f32; n_fft];
    for ti in 0..t {
        // irfft of this frame's one-sided spectrum → n_fft real samples.
        for (n, fr) in frame.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..f {
                let re = mag[k * t + ti] * phase[k * t + ti].cos();
                let im = mag[k * t + ti] * phase[k * t + ti].sin();
                let ang = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                acc += coef(k) * (re * ang.cos() - im * ang.sin());
            }
            *fr = acc * inv_n;
        }
        // Windowed overlap-add + squared-window envelope.
        let base = ti * hop;
        for m in 0..n_fft {
            y[base + m] += frame[m] * win[m];
            wsum[base + m] += win[m] * win[m];
        }
    }

    // Normalise by the window² envelope (zero where the envelope vanishes).
    for i in 0..full {
        if wsum[i] > 1e-11 {
            y[i] /= wsum[i];
        } else {
            y[i] = 0.0;
        }
    }

    // center=True trim: drop n_fft/2 from each end → length hop·(T-1).
    let pad = n_fft / 2;
    y[pad..full - pad].to_vec()
}

/// 2-D linear `y = x·Wᵀ + b`: `x` is `[L, in]`, `w` is `[out, in]` (PyTorch
/// `nn.Linear` layout), optional `b` is `[out]`. Returns `[L, out]`.
pub fn linear2d(
    x: &[f32],
    l: usize,
    in_dim: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; l * out_dim];
    for li in 0..l {
        let xrow = &x[li * in_dim..li * in_dim + in_dim];
        for o in 0..out_dim {
            let wrow = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc = b.map(|b| b[o]).unwrap_or(0.0);
            for j in 0..in_dim {
                acc += xrow[j] * wrow[j];
            }
            y[li * out_dim + o] = acc;
        }
    }
    y
}

/// Row-wise LayerNorm over the last dim of `x` (`[L, C]`), in place, with affine
/// `gamma`/`beta` (`[C]`). Matches `nn.LayerNorm((C,))`.
pub fn layer_norm_rows(x: &mut [f32], l: usize, c: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    for li in 0..l {
        let row = &mut x[li * c..li * c + c];
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (ci, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv * gamma[ci] + beta[ci];
        }
    }
}

/// `gelu_new` (tanh approximation), the activation HF ALBERT uses in its FFN:
/// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
#[inline]
pub fn gelu_new(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// Numerically-stable softmax over a slice, in place.
pub fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// `y = W·s + b` for a single style vector `s` — the small `fc` projection that
/// produces per-channel γ/β inside AdaLayerNorm / AdaIN1d. `w` is `[out, in]`.
fn linear_vec(w: &[f32], b: Option<&[f32]>, s: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for (o, yo) in y.iter_mut().enumerate() {
        let mut acc = b.map(|b| b[o]).unwrap_or(0.0);
        let base = o * in_dim;
        for j in 0..in_dim {
            acc += w[base + j] * s[j];
        }
        *yo = acc;
    }
    y
}

/// **AdaLayerNorm** (StyleTTS2): LayerNorm over the channel axis with *no*
/// learned affine, then a style-conditioned affine `(1 + γ)·x̂ + β` where
/// `[γ, β] = fc(s)`. `x` is `[L, C]` (or `[1, L, C]`), `s` is `[style_dim]`,
/// `fc_w` is `[2C, style_dim]`, `fc_b` is `[2C]`. Returns `[L, C]`.
/// Used by the prosody predictor's DurationEncoder.
pub fn ada_layer_norm(
    x: &Tensor,
    s: &[f32],
    fc_w: &Tensor,
    fc_b: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (l, c) = match dims.as_slice() {
        [l, c] => (*l, *c),
        [1, l, c] => (*l, *c),
        _ => return Err(anyhow!("ada_layer_norm expects [L, C] or [1, L, C]")),
    };
    let two_c = fc_w.shape().dims()[0];
    if two_c != 2 * c {
        return Err(anyhow!("ada_layer_norm: fc out {two_c} != 2*C {}", 2 * c));
    }
    let style_dim = fc_w.shape().dims()[1];
    let gb = linear_vec(
        &fc_w.to_f32_vec(),
        fc_b.map(|b| b.to_f32_vec()).as_deref(),
        s,
        two_c,
        style_dim,
    );
    let (gamma, beta) = gb.split_at(c);

    let xv = x.to_f32_vec();
    let mut out = vec![0.0f32; l * c];
    for li in 0..l {
        let row = &xv[li * c..li * c + c];
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ci in 0..c {
            let norm = (row[ci] - mean) * inv;
            out[li * c + ci] = (1.0 + gamma[ci]) * norm + beta[ci];
        }
    }
    Tensor::from_f32(&out, Shape::new([l, c])).map_err(|e| anyhow!("{e}"))
}

/// **AdaIN1d** (StyleTTS2 / ISTFTNet): InstanceNorm1d (per-channel normalisation
/// over time, no learned affine) then a style-conditioned affine
/// `(1 + γ)·x̂ + β`, `[γ, β] = fc(s)`. `x` is `[1, C, L]`, `s` is `[style_dim]`,
/// `fc_w` is `[2C, style_dim]`, `fc_b` is `[2C]`. Returns `[1, C, L]`.
/// Used by the predictor's F0/N blocks and the decoder's ResBlocks.
pub fn ada_in_1d(
    x: &Tensor,
    s: &[f32],
    fc_w: &Tensor,
    fc_b: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (c, l) = match dims.as_slice() {
        [1, c, l] => (*c, *l),
        [c, l] => (*c, *l),
        _ => return Err(anyhow!("ada_in_1d expects [1, C, L] or [C, L]")),
    };
    let two_c = fc_w.shape().dims()[0];
    if two_c != 2 * c {
        return Err(anyhow!("ada_in_1d: fc out {two_c} != 2*C {}", 2 * c));
    }
    let style_dim = fc_w.shape().dims()[1];
    let gb = linear_vec(
        &fc_w.to_f32_vec(),
        fc_b.map(|b| b.to_f32_vec()).as_deref(),
        s,
        two_c,
        style_dim,
    );
    let (gamma, beta) = gb.split_at(c);

    let xv = x.to_f32_vec();
    let mut out = vec![0.0f32; c * l];
    for ci in 0..c {
        let row = &xv[ci * l..ci * l + l];
        let mean = row.iter().sum::<f32>() / l as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for li in 0..l {
            let norm = (row[li] - mean) * inv;
            out[ci * l + li] = (1.0 + gamma[ci]) * norm + beta[ci];
        }
    }
    Tensor::from_f32(&out, Shape::new([1, c, l])).map_err(|e| anyhow!("{e}"))
}

/// **Length regulator**: expand a `[1, C, L]` per-token feature map to
/// `[1, C, T]` (`T = Σ durations`) by repeating token `i`'s column `durations[i]`
/// times — the hard alignment StyleTTS2 builds from predicted durations.
pub fn length_regulate(x: &Tensor, durations: &[usize]) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (c, l) = match dims.as_slice() {
        [1, c, l] => (*c, *l),
        [c, l] => (*c, *l),
        _ => return Err(anyhow!("length_regulate expects [1, C, L] or [C, L]")),
    };
    if durations.len() != l {
        return Err(anyhow!(
            "length_regulate: {} durations for {l} tokens",
            durations.len()
        ));
    }
    let total: usize = durations.iter().sum();
    let xv = x.to_f32_vec();
    let mut out = vec![0.0f32; c * total];
    for ci in 0..c {
        let mut t = 0usize;
        for (li, &d) in durations.iter().enumerate() {
            let v = xv[ci * l + li];
            for _ in 0..d {
                out[ci * total + t] = v;
                t += 1;
            }
        }
    }
    Tensor::from_f32(&out, Shape::new([1, c, total])).map_err(|e| anyhow!("{e}"))
}

/// Nearest-neighbour ×2 upsample over time of `[1, C, L]` → `[1, C, 2L]`
/// (`F.interpolate(scale_factor=2, mode='nearest')`): each sample repeated twice.
pub fn upsample_nearest_x2(x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims().to_vec();
    let (c, l) = match d.as_slice() {
        [1, c, l] => (*c, *l),
        [c, l] => (*c, *l),
        _ => return Err(anyhow!("upsample_nearest_x2 expects [1, C, L]")),
    };
    let xv = x.to_f32_vec();
    let mut out = vec![0.0f32; c * 2 * l];
    for ci in 0..c {
        for li in 0..l {
            let v = xv[ci * l + li];
            out[ci * 2 * l + 2 * li] = v;
            out[ci * 2 * l + 2 * li + 1] = v;
        }
    }
    Tensor::from_f32(&out, Shape::new([1, c, 2 * l])).map_err(|e| anyhow!("{e}"))
}

/// Depthwise transposed conv used as the ×2 "pool" upsampler inside an upsampling
/// `AdainResBlk1d`: PyTorch `ConvTranspose1d(C, C, kernel=3, stride=2, groups=C,
/// padding=1, output_padding=1)` → exact ×2 length. `w` is `[C, 1, 3]`, `b` `[C]`.
pub fn convtr_depthwise_up2(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let d = x.shape().dims().to_vec();
    let (c, l) = match d.as_slice() {
        [1, c, l] => (*c, *l),
        [c, l] => (*c, *l),
        _ => return Err(anyhow!("convtr_depthwise_up2 expects [1, C, L]")),
    };
    let wd = w.shape().dims().to_vec();
    if wd != [c, 1, 3] {
        return Err(anyhow!(
            "convtr_depthwise_up2 weight must be [C,1,3], got {wd:?}"
        ));
    }
    let xv = x.to_f32_vec();
    let wv = w.to_f32_vec();
    let bv = b.map(|b| b.to_f32_vec());
    let full = (l - 1) * 2 + 3; // un-cropped length 2L+1
    let out_len = 2 * l; // full - 2*pad + output_padding = 2L+1 - 2 + 1
    let mut out = vec![0.0f32; c * out_len];
    for ci in 0..c {
        let mut acc = vec![0.0f32; full];
        for li in 0..l {
            let xval = xv[ci * l + li];
            let base = li * 2;
            for kk in 0..3 {
                acc[base + kk] += xval * wv[ci * 3 + kk];
            }
        }
        let bias = bv.as_ref().map(|v| v[ci]).unwrap_or(0.0);
        for oi in 0..out_len {
            out[ci * out_len + oi] = acc[oi + 1] + bias; // crop pad=1 from the left
        }
    }
    Tensor::from_f32(&out, Shape::new([1, c, out_len])).map_err(|e| anyhow!("{e}"))
}

/// LeakyReLU with the given negative slope, in place over a flat buffer.
pub fn leaky_relu_inplace(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// **NSF harmonic source** (SineGen + SourceModuleHnNSF), inference-time and
/// deterministic. From a frame-rate `f0` contour it builds a harmonic sine bank
/// (fundamental + `harmonic_num` harmonics) at per-sample resolution, gates it by
/// the voiced mask (`f0 > voiced_threshold`), and merges the harmonics with the
/// learned `l_linear` (`[1, harmonic_num+1]`) + `tanh` into the single-channel
/// `har_source` the generator STFT-analyses. Returns `[T_frames·upsample_scale]`.
///
/// The training-time randomness (Gaussian noise feeding the merge, and the
/// first-sample random phase) is **omitted** — it is negligible (voiced noise
/// std 0.003 vs sine amp 0.1; rand-phase is first-sample-only and zero for the
/// fundamental) and dropping it makes the source reproducible, exactly the
/// precedent set by omitting SNAC's stochastic NoiseBlock. The reference's
/// frame-rate-cumsum-then-interpolate phase trick is a float-precision
/// optimisation; an f64 full-rate accumulator yields the same signal directly.
#[allow(clippy::too_many_arguments)]
pub fn nsf_harmonic_source(
    f0_frames: &[f32],
    l_linear_w: &[f32],
    l_linear_b: f32,
    sr: f32,
    harmonic_num: usize,
    sine_amp: f32,
    voiced_threshold: f32,
    upsample_scale: usize,
) -> Vec<f32> {
    nsf_harmonic_source_from(
        f0_frames,
        l_linear_w,
        l_linear_b,
        sr,
        harmonic_num,
        sine_amp,
        voiced_threshold,
        upsample_scale,
        0.0,
    )
}

/// [`nsf_harmonic_source`] with an explicit starting phase (in accumulated
/// cycles). A windowed decode starting at f0 index `a` passes
/// `Σ_{i<a} f0[i] · upsample / sr` — the exact value the full-utterance cumsum
/// would have reached — so the sine phase is continuous across window joins.
#[allow(clippy::too_many_arguments)]
pub fn nsf_harmonic_source_from(
    f0_frames: &[f32],
    l_linear_w: &[f32],
    l_linear_b: f32,
    sr: f32,
    harmonic_num: usize,
    sine_amp: f32,
    voiced_threshold: f32,
    upsample_scale: usize,
    initial_cycles: f64,
) -> Vec<f32> {
    let dim = harmonic_num + 1; // fundamental + harmonics (9 for Kokoro)
    let t_up = f0_frames.len() * upsample_scale;
    let mut har = vec![0.0f32; t_up];
    let mut cycles = initial_cycles; // accumulated cycles = Σ f0/sr (inclusive cumsum)
    let two_pi = 2.0 * std::f64::consts::PI;
    for t in 0..t_up {
        let f0 = f0_frames[t / upsample_scale];
        cycles += f0 as f64 / sr as f64; // per-sample phase increment (mod 1 is a no-op here)
        let phase = cycles * two_pi;
        let uv = if f0 > voiced_threshold { 1.0f32 } else { 0.0 };
        let mut acc = l_linear_b;
        for (n, &wn) in l_linear_w.iter().enumerate().take(dim) {
            let s = (((n + 1) as f64 * phase).sin()) as f32 * sine_amp * uv;
            acc += wn * s;
        }
        har[t] = acc.tanh();
    }
    har
}

/// PyTorch `nn.LSTM` parameters for **one direction** of one layer. Gate rows
/// are stacked in PyTorch order `[input, forget, cell, output]` (i, f, g, o), so
/// `weight_ih` / `weight_hh` are `[4*hidden, *]` and the biases are `[4*hidden]`.
pub struct LstmParams<'a> {
    /// `weight_ih_l{n}[_reverse]` — shape `[4*hidden, input]`.
    pub weight_ih: &'a Tensor,
    /// `weight_hh_l{n}[_reverse]` — shape `[4*hidden, hidden]`.
    pub weight_hh: &'a Tensor,
    /// `bias_ih_l{n}[_reverse]` — shape `[4*hidden]` (PyTorch keeps separate
    /// input/hidden biases; their sum is the effective gate bias).
    pub bias_ih: Option<&'a Tensor>,
    /// `bias_hh_l{n}[_reverse]` — shape `[4*hidden]`.
    pub bias_hh: Option<&'a Tensor>,
}

/// Run one LSTM direction over `x` (`[T, input]`, row-major) → `[T, hidden]`.
/// When `reverse`, timesteps are consumed last-to-first but written back at
/// their original index (so the caller can concatenate forward/backward at the
/// same timestep, exactly like PyTorch's bidirectional output layout).
fn lstm_direction(
    x: &[f32],
    t: usize,
    input: usize,
    hidden: usize,
    p: &LstmParams,
    reverse: bool,
) -> Vec<f32> {
    let w_ih = p.weight_ih.to_f32_vec(); // [4H, input]
    let w_hh = p.weight_hh.to_f32_vec(); // [4H, H]
    let b_ih = p.bias_ih.map(|b| b.to_f32_vec());
    let b_hh = p.bias_hh.map(|b| b.to_f32_vec());

    let mut h_prev = vec![0.0f32; hidden];
    let mut c_prev = vec![0.0f32; hidden];
    let mut out = vec![0.0f32; t * hidden];
    let mut z = vec![0.0f32; 4 * hidden];

    for step in 0..t {
        let ti = if reverse { t - 1 - step } else { step };
        let xrow = &x[ti * input..ti * input + input];

        // Pre-activations z = W_ih·x + b_ih + W_hh·h_prev + b_hh  (per gate row).
        for (r, zr) in z.iter_mut().enumerate() {
            let wib = r * input;
            let mut acc = 0.0f32;
            for j in 0..input {
                acc += w_ih[wib + j] * xrow[j];
            }
            let whb = r * hidden;
            for j in 0..hidden {
                acc += w_hh[whb + j] * h_prev[j];
            }
            if let Some(b) = &b_ih {
                acc += b[r];
            }
            if let Some(b) = &b_hh {
                acc += b[r];
            }
            *zr = acc;
        }

        for k in 0..hidden {
            let i = sigmoid(z[k]); //            gate i: rows [0, H)
            let f = sigmoid(z[hidden + k]); //   gate f: rows [H, 2H)
            let g = z[2 * hidden + k].tanh(); // gate g: rows [2H, 3H)
            let o = sigmoid(z[3 * hidden + k]); // gate o: rows [3H, 4H)
            let c = f * c_prev[k] + i * g;
            let h = o * c.tanh();
            c_prev[k] = c;
            h_prev[k] = h;
            out[ti * hidden + k] = h;
        }
    }
    out
}

/// Bidirectional LSTM over `x` (`[T, input]` or `[1, T, input]`). Returns
/// `[T, 2*hidden]` with the forward hidden state in `[0, hidden)` and the
/// backward hidden state in `[hidden, 2*hidden)` at each timestep — matching
/// PyTorch's `output` concatenation for a single bidirectional layer.
pub fn lstm_bidirectional(x: &Tensor, fwd: &LstmParams, bwd: &LstmParams) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (t, input) = match dims.as_slice() {
        [t, input] => (*t, *input),
        [1, t, input] => (*t, *input),
        _ => {
            return Err(anyhow!(
                "lstm_bidirectional expects [T, input] or [1, T, input]"
            ))
        }
    };
    let hidden = fwd.weight_ih.shape().dims()[0] / 4;
    if bwd.weight_ih.shape().dims()[0] / 4 != hidden {
        return Err(anyhow!("lstm_bidirectional: fwd/bwd hidden size mismatch"));
    }
    if fwd.weight_ih.shape().dims()[1] != input {
        return Err(anyhow!(
            "lstm_bidirectional: weight_ih input {} != x input {input}",
            fwd.weight_ih.shape().dims()[1]
        ));
    }

    let xv = x.to_f32_vec();
    let f = lstm_direction(&xv, t, input, hidden, fwd, false);
    let b = lstm_direction(&xv, t, input, hidden, bwd, true);

    let mut out = vec![0.0f32; t * 2 * hidden];
    for ti in 0..t {
        out[ti * 2 * hidden..ti * 2 * hidden + hidden]
            .copy_from_slice(&f[ti * hidden..ti * hidden + hidden]);
        out[ti * 2 * hidden + hidden..ti * 2 * hidden + 2 * hidden]
            .copy_from_slice(&b[ti * hidden..ti * hidden + hidden]);
    }
    Tensor::from_f32(&out, Shape::new([t, 2 * hidden])).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params<'a>(
        wih: &'a Tensor,
        whh: &'a Tensor,
        bih: &'a Tensor,
        bhh: &'a Tensor,
    ) -> LstmParams<'a> {
        LstmParams {
            weight_ih: wih,
            weight_hh: whh,
            bias_ih: Some(bih),
            bias_hh: Some(bhh),
        }
    }

    /// One-step, 1-in/1-hidden LSTM matches a value computed by hand from the
    /// PyTorch recurrence (gate order i,f,g,o; c0=h0=0).
    #[test]
    fn lstm_single_step_matches_hand_computation() {
        // weight_ih rows = [w_i, w_f, w_g, w_o] (each [1]); zero recurrent + bias.
        let wih = Tensor::from_f32(&[0.5, 0.3, 0.7, 0.2], vec![4, 1]).unwrap();
        let whh = Tensor::from_f32(&[0.0, 0.0, 0.0, 0.0], vec![4, 1]).unwrap();
        let zero = Tensor::from_f32(&[0.0, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let p = params(&wih, &whh, &zero, &zero);

        let x = [1.0f32];
        let out = lstm_direction(&x, 1, 1, 1, &p, false);

        let i = 1.0 / (1.0 + (-0.5f32).exp());
        let f = 1.0 / (1.0 + (-0.3f32).exp());
        let g = (0.7f32).tanh();
        let o = 1.0 / (1.0 + (-0.2f32).exp());
        let c = f * 0.0 + i * g;
        let want = o * c.tanh();
        assert!((out[0] - want).abs() < 1e-6, "got {}, want {want}", out[0]);
    }

    /// Reverse direction over x equals forward direction over reversed x, read
    /// back-to-front — the invariant that makes the bidirectional concat valid.
    #[test]
    fn lstm_reverse_equals_forward_on_reversed_input() {
        let (t, input, hidden) = (4usize, 2usize, 3usize);
        let gen = |i: usize| (i as f32 * 0.31 + 0.1).sin();
        let wih = Tensor::from_f32(
            &(0..4 * hidden * input).map(gen).collect::<Vec<_>>(),
            vec![4 * hidden, input],
        )
        .unwrap();
        let whh = Tensor::from_f32(
            &(0..4 * hidden * hidden)
                .map(|i| gen(i + 7))
                .collect::<Vec<_>>(),
            vec![4 * hidden, hidden],
        )
        .unwrap();
        let bih = Tensor::from_f32(
            &(0..4 * hidden).map(|i| gen(i + 50)).collect::<Vec<_>>(),
            vec![4 * hidden],
        )
        .unwrap();
        let bhh = Tensor::from_f32(
            &(0..4 * hidden).map(|i| gen(i + 90)).collect::<Vec<_>>(),
            vec![4 * hidden],
        )
        .unwrap();
        let p = params(&wih, &whh, &bih, &bhh);

        let x: Vec<f32> = (0..t * input).map(|i| gen(i + 3)).collect();
        let rev = lstm_direction(&x, t, input, hidden, &p, true);

        // Reverse the input rows and run forward.
        let mut xr = vec![0.0f32; t * input];
        for ti in 0..t {
            xr[ti * input..ti * input + input]
                .copy_from_slice(&x[(t - 1 - ti) * input..(t - 1 - ti) * input + input]);
        }
        let fwd = lstm_direction(&xr, t, input, hidden, &p, false);

        for ti in 0..t {
            for k in 0..hidden {
                let a = rev[ti * hidden + k];
                let b = fwd[(t - 1 - ti) * hidden + k];
                assert!((a - b).abs() < 1e-6, "ti={ti} k={k}: {a} vs {b}");
            }
        }
    }

    /// NSF source on silence (f0=0 → unvoiced everywhere) collapses to a
    /// constant `tanh(bias)`; on a constant voiced f0 it is exactly periodic at
    /// the fundamental period (sr/f0 samples).
    #[test]
    fn nsf_source_silence_and_periodicity() {
        let w = [0.2f32, 0.1, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]; // [9]
        let b = 0.3f32;
        let (sr, hnum, amp, thr, up) = (24000.0f32, 8usize, 0.1f32, 10.0f32, 300usize);

        // Silence: f0 = 0 → uv = 0 → har = tanh(bias) everywhere.
        let f0_sil = vec![0.0f32; 4];
        let sil = nsf_harmonic_source(&f0_sil, &w, b, sr, hnum, amp, thr, up);
        assert_eq!(sil.len(), 4 * up);
        let want = b.tanh();
        for &v in &sil {
            assert!((v - want).abs() < 1e-6);
        }

        // Split-with-carried-phase equals one pass (windowed-decode continuity).
        let f0_all = [120.0f32, 140.0, 90.0, 200.0];
        let full = nsf_harmonic_source(&f0_all, &w, b, sr, hnum, amp, thr, up);
        let head = nsf_harmonic_source(&f0_all[..2], &w, b, sr, hnum, amp, thr, up);
        let carried: f64 = f0_all[..2]
            .iter()
            .map(|&f| f as f64 * up as f64 / sr as f64)
            .sum();
        let tail = nsf_harmonic_source_from(&f0_all[2..], &w, b, sr, hnum, amp, thr, up, carried);
        let stitched: Vec<f32> = head.iter().chain(tail.iter()).copied().collect();
        assert_eq!(stitched.len(), full.len());
        for (i, (a, c)) in full.iter().zip(&stitched).enumerate() {
            assert!((a - c).abs() < 1e-4, "sample {i}: {a} vs {c}");
        }

        // Constant voiced f0 = 100 Hz → period = sr/f0 = 240 samples; har is
        // exactly periodic (base cycles advance by exactly 1.0 over 240 samples).
        let f0_v = vec![100.0f32; 4];
        let har = nsf_harmonic_source(&f0_v, &w, b, sr, hnum, amp, thr, up);
        let period = 240usize;
        for t in 0..har.len() - period {
            assert!(
                (har[t] - har[t + period]).abs() < 1e-4,
                "t={t}: {} vs {}",
                har[t],
                har[t + period]
            );
        }
        // And it's not a constant (it actually oscillates when voiced).
        let max = har.iter().cloned().fold(f32::MIN, f32::max);
        let min = har.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max - min > 1e-3, "voiced source should oscillate");
    }

    /// AdaLayerNorm with zero γ/β (zero fc weights) reduces to a plain
    /// channel-axis LayerNorm: each row has ~zero mean and unit variance.
    #[test]
    fn ada_layer_norm_zero_style_is_plain_layernorm() {
        let (l, c, style) = (3usize, 4usize, 5usize);
        let gen = |i: usize| (i as f32 * 0.7 + 0.2).sin() * 2.0 + 1.0;
        let x = Tensor::from_f32(&(0..l * c).map(gen).collect::<Vec<_>>(), vec![l, c]).unwrap();
        let fc_w = Tensor::from_f32(&vec![0.0f32; 2 * c * style], vec![2 * c, style]).unwrap();
        let s = vec![0.5f32; style];
        let y = ada_layer_norm(&x, &s, &fc_w, None, 1e-5).unwrap();
        let yv = y.as_f32_slice();
        for li in 0..l {
            let row = &yv[li * c..li * c + c];
            let mean = row.iter().sum::<f32>() / c as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
            assert!(mean.abs() < 1e-4, "row {li} mean {mean}");
            assert!((var - 1.0).abs() < 1e-3, "row {li} var {var}");
        }
    }

    /// AdaIN1d normalises per channel over time; with a known constant γ/β (via a
    /// one-hot style and identity-ish fc) the affine is applied as expected.
    #[test]
    fn ada_in_1d_normalizes_per_channel_then_affine() {
        let (c, l, style) = (2usize, 6usize, 2usize);
        let gen = |i: usize| (i as f32 * 0.33).cos() + 0.5;
        let x = Tensor::from_f32(&(0..c * l).map(gen).collect::<Vec<_>>(), vec![1, c, l]).unwrap();
        // fc(s) = [γ0, γ1, β0, β1]; pick s=[1,0] and weights selecting fixed values.
        // rows: γ0=0.0, γ1=0.0, β0=0.0, β1=0.0 via identity → plain instance norm.
        let fc_w = Tensor::from_f32(&vec![0.0f32; 2 * c * style], vec![2 * c, style]).unwrap();
        let s = vec![1.0f32, 0.0];
        let y = ada_in_1d(&x, &s, &fc_w, None, 1e-5).unwrap();
        let yv = y.as_f32_slice();
        // Each channel row: ~zero mean, unit variance.
        for ci in 0..c {
            let row = &yv[ci * l..ci * l + l];
            let mean = row.iter().sum::<f32>() / l as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l as f32;
            assert!(mean.abs() < 1e-4, "ch {ci} mean {mean}");
            assert!((var - 1.0).abs() < 1e-2, "ch {ci} var {var}");
        }
    }

    /// Length regulator repeats each token column by its duration.
    #[test]
    fn length_regulate_repeats_columns() {
        let (c, l) = (2usize, 3usize);
        // x[c][l]: ch0 = [10,20,30], ch1 = [1,2,3]
        let x = Tensor::from_f32(&[10.0, 20.0, 30.0, 1.0, 2.0, 3.0], vec![1, c, l]).unwrap();
        let dur = [2usize, 0, 3];
        let y = length_regulate(&x, &dur).unwrap();
        assert_eq!(y.shape().dims(), &[1, c, 5]);
        let yv = y.as_f32_slice();
        // ch0: 10,10,30,30,30 ; ch1: 1,1,3,3,3 (token 1 has duration 0 → dropped)
        assert_eq!(&yv[0..5], &[10.0, 10.0, 30.0, 30.0, 30.0]);
        assert_eq!(&yv[5..10], &[1.0, 1.0, 3.0, 3.0, 3.0]);
    }

    /// STFT→iSTFT round-trips a signal back to itself in the interior (away
    /// from the reflect-padded edges). This is the gold-standard check that the
    /// irfft 1,2,1 weighting, the periodic Hann window, and the window² overlap
    /// normalisation are all mutually consistent — at Kokoro's n_fft=20/hop=5.
    #[test]
    fn stft_istft_round_trip_reconstructs_interior() {
        let (n_fft, hop) = (20usize, 5usize);
        let l = 200usize; // multiple of hop
        let x: Vec<f32> = (0..l)
            .map(|i| (i as f32 * 0.13).sin() * 0.6 + (i as f32 * 0.41 + 1.0).cos() * 0.3)
            .collect();

        let (mag, phase, t) = stft_transform(&x, n_fft, hop);
        assert_eq!(t, 1 + l / hop);
        let rec = istft(&mag, &phase, n_fft / 2 + 1, t, n_fft, hop);
        assert_eq!(rec.len(), hop * (t - 1));
        assert_eq!(rec.len(), l);

        // Interior reconstruction is near-exact (edges differ due to reflect pad).
        for i in n_fft..l - n_fft {
            assert!(
                (rec[i] - x[i]).abs() < 1e-3,
                "i={i}: rec {} vs x {}",
                rec[i],
                x[i]
            );
        }
    }

    /// A DC-only spectrum (bin 0 constant across frames, zero phase) reconstructs
    /// to a flat signal in the interior. The per-frame inverse is a constant
    /// v/N, and windowed overlap-add normalised by Σw² yields a constant where
    /// the window envelope is steady — exercising the DC coefficient (weight 1).
    #[test]
    fn istft_dc_bin_is_flat_interior() {
        let (n_fft, hop) = (20usize, 5usize);
        let f = n_fft / 2 + 1;
        let t = 16usize;
        let mut mag = vec![0.0f32; f * t];
        let phase = vec![0.0f32; f * t];
        for m in mag.iter_mut().take(t) {
            *m = 3.0; // bin 0 (DC) = 3 for every frame
        }
        let rec = istft(&mag, &phase, f, t, n_fft, hop);
        // Interior is flat (COLA region); compare every interior sample to the
        // midpoint rather than to a hand closed-form.
        let mid = rec[rec.len() / 2];
        let hi = rec.len() - n_fft;
        for (i, &v) in rec.iter().enumerate().take(hi).skip(n_fft) {
            assert!((v - mid).abs() < 1e-4, "i={i}: {v} vs {mid}");
        }
    }

    /// Bidirectional output is the forward state in the low half and the
    /// backward state in the high half, with the right shape.
    #[test]
    fn lstm_bidirectional_concat_layout() {
        let (t, input, hidden) = (3usize, 2usize, 2usize);
        let gen = |i: usize| (i as f32 * 0.21 + 0.05).cos();
        let mk = |n: usize, off: usize, shape: Vec<usize>| {
            Tensor::from_f32(&(0..n).map(|i| gen(i + off)).collect::<Vec<_>>(), shape).unwrap()
        };
        let wih = mk(4 * hidden * input, 0, vec![4 * hidden, input]);
        let whh = mk(4 * hidden * hidden, 11, vec![4 * hidden, hidden]);
        let bih = mk(4 * hidden, 23, vec![4 * hidden]);
        let bhh = mk(4 * hidden, 41, vec![4 * hidden]);
        let wih2 = mk(4 * hidden * input, 60, vec![4 * hidden, input]);
        let whh2 = mk(4 * hidden * hidden, 71, vec![4 * hidden, hidden]);
        let bih2 = mk(4 * hidden, 83, vec![4 * hidden]);
        let bhh2 = mk(4 * hidden, 91, vec![4 * hidden]);
        let fwd = params(&wih, &whh, &bih, &bhh);
        let bwd = params(&wih2, &whh2, &bih2, &bhh2);

        let xv: Vec<f32> = (0..t * input).map(|i| gen(i + 5)).collect();
        let x = Tensor::from_f32(&xv, vec![t, input]).unwrap();
        let y = lstm_bidirectional(&x, &fwd, &bwd).unwrap();
        assert_eq!(y.shape().dims(), &[t, 2 * hidden]);

        let f = lstm_direction(&xv, t, input, hidden, &fwd, false);
        let b = lstm_direction(&xv, t, input, hidden, &bwd, true);
        let yv = y.as_f32_slice();
        for ti in 0..t {
            for k in 0..hidden {
                assert!((yv[ti * 2 * hidden + k] - f[ti * hidden + k]).abs() < 1e-6);
                assert!((yv[ti * 2 * hidden + hidden + k] - b[ti * hidden + k]).abs() < 1e-6);
            }
        }
    }
}
