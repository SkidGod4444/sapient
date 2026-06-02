//! 1-D convolution as a thin wrapper over the 2-D im2col kernel.
//!
//! Whisper's audio stem is two `Conv1d` layers. Rather than add a dedicated 1-D
//! kernel, we treat the length-`L` signal as a height-1 image (`[N, C, 1, L]`)
//! and reuse the verified [`conv2d`] im2col+GEMM path. The convs run once per
//! 30 s chunk (not per token), so the im2col scratch is negligible.

use anyhow::{anyhow, Result};
use sapient_backends_cpu::kernels::conv2d::conv2d;
use sapient_core::Tensor;

/// Conv1d: `x [1, C_in, L]`, `weight [C_out, C_in, K]`, optional `bias [C_out]`
/// → `[1, C_out, L_out]`, with symmetric padding `pad` and stride `stride`.
pub fn conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    pad: usize,
    stride: usize,
) -> Result<Tensor> {
    let xd = x.shape().dims();
    let wd = weight.shape().dims();
    if xd.len() != 3 || wd.len() != 3 {
        anyhow::bail!("conv1d expects x [1,C_in,L] and weight [C_out,C_in,K]");
    }
    let (n, c_in, l) = (xd[0], xd[1], xd[2]);
    let (c_out, c_in_w, k) = (wd[0], wd[1], wd[2]);

    // Reshape to height-1 images: x → [N, C_in, 1, L], w → [C_out, C_in, 1, K].
    let x4 = x.reshape(vec![n, c_in, 1, l]).map_err(|e| anyhow!("{e}"))?;
    let w4 = weight
        .reshape(vec![c_out, c_in_w, 1, k])
        .map_err(|e| anyhow!("{e}"))?;

    // pads = [top, left, bottom, right]; only the width (last) axis is padded.
    let y = conv2d(
        &x4,
        &w4,
        bias,
        [1, k],
        [0, pad, 0, pad],
        [1, stride],
        [1, 1],
        1,
    )
    .map_err(|e| anyhow!("{e}"))?;

    let yd = y.shape().dims().to_vec(); // [N, C_out, 1, L_out]
    y.reshape(vec![yd[0], yd[1], yd[3]])
        .map_err(|e| anyhow!("{e}"))
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

        let y = conv1d(&x, &w, Some(&b), pad, stride).unwrap();
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
}
