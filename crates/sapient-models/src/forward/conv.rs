//! 1-D convolution primitives for audio models.
//!
//! - [`conv1d`] wraps the verified 2-D im2col [`conv2d`] (Whisper's audio stem).
//! - [`conv_transpose1d`] is the transposed (fractionally-strided) conv used by
//!   neural-audio-codec decoders (SNAC/DAC) to upsample codec frames to waveform.
//! - [`snake`] is the periodic Snake activation `x + sin²(αx)/α` (per-channel α)
//!   used throughout those codec decoders.
//!
//! These run a handful of times per generated audio frame, so straightforward
//! correct implementations are fine (no im2col/GEMM needed for the transpose).

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use sapient_backends_cpu::kernels::conv2d::conv2d;
use sapient_core::{Shape, Tensor};

/// Conv1d: `x [1, C_in, L]`, `weight [C_out, C_in/groups, K]`, optional `bias
/// [C_out]` → `[1, C_out, L_out]`, with symmetric padding `pad`, `stride`,
/// `dilation`, and `groups` (groups = C_in for depthwise). Whisper uses
/// `dilation=1, groups=1`; SNAC's codec decoder uses depthwise + dilated convs.
pub fn conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    pad: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let xd = x.shape().dims();
    let wd = weight.shape().dims();
    if xd.len() != 3 || wd.len() != 3 {
        anyhow::bail!("conv1d expects x [1,C_in,L] and weight [C_out,C_in/groups,K]");
    }
    let (n, c_in, l) = (xd[0], xd[1], xd[2]);
    let (c_out, c_in_g, k) = (wd[0], wd[1], wd[2]);

    // Reshape to height-1 images: x → [N, C_in, 1, L], w → [C_out, C_in/g, 1, K].
    let x4 = x.reshape(vec![n, c_in, 1, l]).map_err(|e| anyhow!("{e}"))?;
    let w4 = weight
        .reshape(vec![c_out, c_in_g, 1, k])
        .map_err(|e| anyhow!("{e}"))?;

    // pads = [top, left, bottom, right]; only the width (last) axis is padded.
    let y = conv2d(
        &x4,
        &w4,
        bias,
        [1, k],
        [0, pad, 0, pad],
        [1, stride],
        [1, dilation],
        groups,
    )
    .map_err(|e| anyhow!("{e}"))?;

    let yd = y.shape().dims().to_vec(); // [N, C_out, 1, L_out]
    y.reshape(vec![yd[0], yd[1], yd[3]])
        .map_err(|e| anyhow!("{e}"))
}

/// Transposed 1-D convolution (PyTorch `ConvTranspose1d`, dilation=1,
/// output_padding=0). `x [1, C_in, L]`, `weight [C_in, C_out, K]` (PyTorch
/// transpose layout), optional `bias [C_out]`. Output `[1, C_out, L_out]` with
/// `L_out = (L-1)*stride - 2*pad + K`. Used by SNAC/DAC codec decoders to
/// upsample; implemented as direct scatter-add (cheap at codec frame rates).
/// Accumulated nanoseconds in [`conv_transpose1d`] / [`snake`] (profiling).
pub static CONVT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SNAKE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn conv_transpose1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    pad: usize,
) -> Result<Tensor> {
    let _t = std::time::Instant::now();
    let r = conv_transpose1d_inner(x, weight, bias, stride, pad);
    CONVT_NS.fetch_add(
        _t.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    r
}

fn conv_transpose1d_inner(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    pad: usize,
) -> Result<Tensor> {
    let xd = x.shape().dims();
    let wd = weight.shape().dims();
    if xd.len() != 3 || wd.len() != 3 {
        anyhow::bail!("conv_transpose1d expects x [1,C_in,L] and weight [C_in,C_out,K]");
    }
    let (c_in, l) = (xd[1], xd[2]);
    let (c_in_w, c_out, k) = (wd[0], wd[1], wd[2]);
    if c_in != c_in_w {
        anyhow::bail!("conv_transpose1d C_in mismatch: x {c_in} vs weight {c_in_w}");
    }
    let full = (l - 1) * stride + k; // length before cropping padding
    let l_out = full
        .checked_sub(2 * pad)
        .ok_or_else(|| anyhow!("conv_transpose1d: padding {pad} too large for output"))?;

    let xv = x.to_f32_cow();
    let wv = weight.to_f32_cow();
    let bias_v = bias.map(|b| b.to_f32_vec());

    // Each output channel is independent (it gathers from every input channel),
    // so compute the cropped+biased output row per `co` in parallel — this is the
    // hot upsampling op in the ISTFTNet generator (512→256 ×10, 256→128 ×6). The
    // inner loops are the same scatter, just with `co` as the outer (per-thread)
    // axis so there's no write contention.
    let mut out = vec![0.0f32; c_out * l_out];
    out.par_chunks_mut(l_out)
        .enumerate()
        .for_each(|(co, out_row)| {
            let mut full_row = vec![0.0f32; full];
            for ci in 0..c_in {
                let w_base = (ci * c_out + co) * k;
                let x_off = ci * l;
                for li in 0..l {
                    let x_val = xv[x_off + li];
                    if x_val == 0.0 {
                        continue;
                    }
                    let o = li * stride;
                    for kk in 0..k {
                        full_row[o + kk] += x_val * wv[w_base + kk];
                    }
                }
            }
            let b = bias_v.as_ref().map(|v| v[co]).unwrap_or(0.0);
            for (oi, slot) in out_row.iter_mut().enumerate() {
                *slot = full_row[pad + oi] + b;
            }
        });
    Tensor::from_f32(&out, Shape::new([1, c_out, l_out])).map_err(|e| anyhow!("{e}"))
}

/// Snake activation `x + sin²(α·x) / α` with a per-channel α. `x [1, C, L]`,
/// `alpha [C]`. Used by SNAC/DAC decoder blocks. α is guarded against 0.
pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let _t = std::time::Instant::now();
    let r = snake_inner(x, alpha);
    SNAKE_NS.fetch_add(
        _t.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    r
}

fn snake_inner(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let xd = x.shape().dims().to_vec();
    if xd.len() != 3 {
        anyhow::bail!("snake expects x [1, C, L]");
    }
    let (c, l) = (xd[1], xd[2]);
    let a = alpha.to_f32_vec();
    if a.len() != c {
        anyhow::bail!("snake alpha length {} != channels {c}", a.len());
    }
    let xv = x.to_f32_cow();
    let mut out = vec![0.0f32; c * l];
    // Channels are independent — parallelize rows (bit-identical per element).
    out.par_chunks_mut(l).enumerate().for_each(|(ci, row)| {
        let inv = 1.0 / (a[ci] + 1e-9);
        let src = &xv[ci * l..(ci + 1) * l];
        for (o, &v) in row.iter_mut().zip(src) {
            let s = (a[ci] * v).sin();
            *o = v + inv * s * s;
        }
    });
    Tensor::from_f32(&out, Shape::new([1, c, l])).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// conv1d via conv2d must match a naive direct 1-D convolution.
    #[test]
    fn conv1d_matches_naive() {
        let (c_in, c_out, k, l) = (3usize, 4usize, 3usize, 10usize);
        let pad = 1usize;
        let stride = 1usize;

        let gen = |i: usize| (i as f32 * 0.7 + 0.3).sin();
        let x_data: Vec<f32> = (0..c_in * l).map(gen).collect();
        let w_data: Vec<f32> = (0..c_out * c_in * k).map(|i| gen(i + 50)).collect();
        let b_data: Vec<f32> = (0..c_out).map(|i| gen(i + 200)).collect();

        let x = Tensor::from_f32(&x_data, vec![1, c_in, l]).unwrap();
        let w = Tensor::from_f32(&w_data, vec![c_out, c_in, k]).unwrap();
        let b = Tensor::from_f32(&b_data, vec![c_out]).unwrap();

        let y = conv1d(&x, &w, Some(&b), pad, stride, 1, 1).unwrap();
        let l_out = (l + 2 * pad - k) / stride + 1;
        assert_eq!(y.shape().dims(), &[1, c_out, l_out]);
        let y_data = y.as_f32_slice();

        // Naive reference.
        for co in 0..c_out {
            for ol in 0..l_out {
                let mut acc = b_data[co];
                for ci in 0..c_in {
                    for kk in 0..k {
                        let il = ol as isize * stride as isize + kk as isize - pad as isize;
                        if il >= 0 && (il as usize) < l {
                            acc += x_data[ci * l + il as usize] * w_data[(co * c_in + ci) * k + kk];
                        }
                    }
                }
                let got = y_data[co * l_out + ol];
                assert!(
                    (got - acc).abs() < 1e-4,
                    "co={co} ol={ol}: got {got}, want {acc}"
                );
            }
        }
    }

    /// conv_transpose1d must match a naive scatter-add reference.
    #[test]
    fn conv_transpose1d_matches_naive() {
        let (c_in, c_out, k, l) = (2usize, 3usize, 4usize, 5usize);
        let (stride, pad) = (2usize, 1usize);
        let gen = |i: usize| (i as f32 * 0.37 + 0.2).cos();
        let x: Vec<f32> = (0..c_in * l).map(gen).collect();
        let w: Vec<f32> = (0..c_in * c_out * k).map(|i| gen(i + 17)).collect();
        let b: Vec<f32> = (0..c_out).map(|i| gen(i + 99)).collect();

        let xt = Tensor::from_f32(&x, vec![1, c_in, l]).unwrap();
        let wt = Tensor::from_f32(&w, vec![c_in, c_out, k]).unwrap();
        let bt = Tensor::from_f32(&b, vec![c_out]).unwrap();
        let y = conv_transpose1d(&xt, &wt, Some(&bt), stride, pad).unwrap();

        let full = (l - 1) * stride + k;
        let l_out = full - 2 * pad;
        assert_eq!(y.shape().dims(), &[1, c_out, l_out]);
        let yv = y.as_f32_slice();

        // Naive reference: scatter-add then crop + bias.
        let mut fout = vec![0.0f32; c_out * full];
        for ci in 0..c_in {
            for li in 0..l {
                for co in 0..c_out {
                    for kk in 0..k {
                        fout[co * full + li * stride + kk] +=
                            x[ci * l + li] * w[(ci * c_out + co) * k + kk];
                    }
                }
            }
        }
        for co in 0..c_out {
            for oi in 0..l_out {
                let want = fout[co * full + pad + oi] + b[co];
                assert!((yv[co * l_out + oi] - want).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn snake_matches_formula() {
        let (c, l) = (2usize, 4usize);
        let x =
            Tensor::from_f32(&[0.0, 0.5, 1.0, -1.0, 0.2, 0.4, 0.6, 0.8], vec![1, c, l]).unwrap();
        let alpha = Tensor::from_f32(&[1.0, 2.0], vec![c]).unwrap();
        let y = snake(&x, &alpha).unwrap();
        let yv = y.as_f32_slice();
        let xv = x.as_f32_slice();
        let a = [1.0f32, 2.0];
        for ci in 0..c {
            for li in 0..l {
                let v = xv[ci * l + li];
                let s = (a[ci] * v).sin();
                let want = v + (1.0 / (a[ci] + 1e-9)) * s * s;
                assert!((yv[ci * l + li] - want).abs() < 1e-6);
            }
        }
        // snake(0) == 0 for any alpha.
        assert!(yv[0].abs() < 1e-9);
    }
}
