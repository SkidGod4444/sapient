//! 2-D convolution via Im2Col + GEMM.
//!
//! This is the standard reference implementation. It trades memory for
//! compute efficiency. Production backends (Metal, Vulkan) use direct
//! convolution or Winograd.

use sapient_core::error::{Result, SapientError};
use sapient_core::{Shape, Tensor};

/// 2-D convolution: (N, C_in, H, W) → (N, C_out, H_out, W_out).
pub fn conv2d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    _kernel_shape: [usize; 2],
    pads: [usize; 4], // [top, left, bottom, right]
    strides: [usize; 2],
    dilations: [usize; 2],
    groups: usize,
) -> Result<Tensor> {
    let xs = x.shape();
    let ws = weight.shape();

    if xs.ndim() != 4 {
        return Err(SapientError::RankMismatch {
            expected: 4,
            got: xs.ndim(),
        });
    }
    if ws.ndim() != 4 {
        return Err(SapientError::RankMismatch {
            expected: 4,
            got: ws.ndim(),
        });
    }

    let (n, c_in, h_in, w_in) = (xs.dims()[0], xs.dims()[1], xs.dims()[2], xs.dims()[3]);
    let (c_out, c_in_g, kh, kw) = (ws.dims()[0], ws.dims()[1], ws.dims()[2], ws.dims()[3]);

    let g = groups;
    if c_in != c_in_g * g {
        return Err(SapientError::InvalidGraph(format!(
            "conv2d: groups={g}, c_in={c_in}, c_in/group={c_in_g}: {c_in_g}*{g}!=c_in"
        )));
    }

    let h_out = (h_in + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
    let w_out = (w_in + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;

    let x_cow = x.to_f32_cow();
    let x_data = x_cow.as_ref();
    let w_cow = weight.to_f32_cow();
    let w_data = w_cow.as_ref();
    let b_cow = bias.map(|t| t.to_f32_cow());
    let b_data = b_cow.as_ref().map(|c| c.as_ref());

    let out_size = n * c_out * h_out * w_out;
    let mut out_data = vec![0.0f32; out_size];

    // im2col matrix dimensions.
    let col_rows = c_in_g * kh * kw;
    let col_cols = h_out * w_out;

    let c_out_g = c_out / g;

    for batch in 0..n {
        for group in 0..g {
            // Build im2col matrix for this (batch, group).
            let mut col = vec![0.0f32; col_rows * col_cols];

            let c_start = group * c_in_g;

            for ci in 0..c_in_g {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let row = (ci * kh + ki) * kw + kj;
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let ih = oh as isize * strides[0] as isize
                                    + ki as isize * dilations[0] as isize
                                    - pads[0] as isize;
                                let iw = ow as isize * strides[1] as isize
                                    + kj as isize * dilations[1] as isize
                                    - pads[1] as isize;

                                let val = if ih >= 0
                                    && ih < h_in as isize
                                    && iw >= 0
                                    && iw < w_in as isize
                                {
                                    let c = c_start + ci;
                                    let flat = batch * (c_in * h_in * w_in)
                                        + c * (h_in * w_in)
                                        + ih as usize * w_in
                                        + iw as usize;
                                    x_data[flat]
                                } else {
                                    0.0 // zero-padding
                                };

                                col[row * col_cols + oh * w_out + ow] = val;
                            }
                        }
                    }
                }
            }

            // GEMM: W_group × col → output.
            // W_group shape: (c_out_g, col_rows)
            // col shape:     (col_rows, col_cols)
            // output:        (c_out_g, col_cols)
            let w_off = group * c_out_g * (c_in_g * kh * kw);
            let m = c_out_g;
            let k = col_rows;
            let n2 = col_cols;

            let mut gemm_out = vec![0.0f32; m * n2];
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n2,
                    1.0,
                    w_data[w_off..].as_ptr(),
                    k as isize,
                    1,
                    col.as_ptr(),
                    n2 as isize,
                    1,
                    0.0,
                    gemm_out.as_mut_ptr(),
                    n2 as isize,
                    1,
                );
            }

            // Copy gemm_out into output tensor.
            let c_out_start = group * c_out_g;
            for co in 0..c_out_g {
                let bias_v = b_data.map(|b| b[c_out_start + co]).unwrap_or(0.0);
                for hw in 0..col_cols {
                    let out_idx =
                        batch * (c_out * h_out * w_out) + (c_out_start + co) * (h_out * w_out) + hw;
                    out_data[out_idx] = gemm_out[co * n2 + hw] + bias_v;
                }
            }
        }
    }

    Tensor::from_f32(&out_data, Shape::new([n, c_out, h_out, w_out]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_core::Tensor;

    #[test]
    fn conv2d_identity_kernel() {
        // 1×1 conv with identity weight.
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]).unwrap();
        let w = Tensor::from_f32(&[1.0], vec![1, 1, 1, 1]).unwrap();
        let y = conv2d(&x, &w, None, [1, 1], [0, 0, 0, 0], [1, 1], [1, 1], 1).unwrap();
        assert_eq!(y.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
