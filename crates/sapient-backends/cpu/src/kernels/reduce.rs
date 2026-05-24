//! Reduction kernels: sum, mean, max, min.

use sapient_core::{Shape, Tensor};
use sapient_core::error::{Result, SapientError};

fn normalise_axes(axes: &[i64], ndim: usize) -> Vec<usize> {
    if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.iter()
            .map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize })
            .collect()
    }
}

/// Generic reduction over one or more axes.
fn reduce<F>(x: &Tensor, axes: &[i64], keep_dims: bool, init: f32, f: F) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    let shape = x.shape();
    let data  = x.as_f32_slice();
    let norm_axes = normalise_axes(axes, shape.ndim());

    // Output shape.
    let out_dims: Vec<usize> = shape
        .dims()
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if norm_axes.contains(&i) {
                if keep_dims { Some(1) } else { None }
            } else {
                Some(d)
            }
        })
        .collect();

    let out_numel = out_dims.iter().product::<usize>().max(1);
    let mut out_data = vec![init; out_numel];

    // Iterate every element and accumulate into its output position.
    let strides = shape.strides();
    for (flat, &val) in data.iter().enumerate() {
        // Compute multi-index.
        let mut rem = flat;
        let mut out_flat = 0usize;
        let mut out_stride = 1usize;
        let mut out_idx = 0usize;

        // We compute the flat output index by skipping reduced dims.
        // Build output index from most-to-least significant.
        let mut multi = vec![0usize; shape.ndim()];
        {
            let mut r = flat;
            for i in (0..shape.ndim()).rev() {
                multi[i] = r % shape.dims()[i];
                r /= shape.dims()[i];
            }
        }

        // Compute flat output index.
        let out_strides = Shape(out_dims.clone()).strides();
        let mut oi = 0;
        for (i, &mi) in multi.iter().enumerate() {
            if !norm_axes.contains(&i) {
                out_flat += mi * if oi < out_strides.len() { out_strides[oi] } else { 1 };
                oi += 1;
            } else if keep_dims {
                // dim = 1, stride may still be 1.
                oi += 1;
            }
        }

        out_data[out_flat] = f(out_data[out_flat], val);
    }

    Tensor::from_f32(&out_data, Shape::new(out_dims))
}

pub fn reduce_sum(x: &Tensor, axes: &[i64], keep_dims: bool) -> Result<Tensor> {
    reduce(x, axes, keep_dims, 0.0, |acc, v| acc + v)
}

pub fn reduce_mean(x: &Tensor, axes: &[i64], keep_dims: bool) -> Result<Tensor> {
    let sum = reduce_sum(x, axes, keep_dims)?;
    let norm_axes = normalise_axes(axes, x.shape().ndim());
    let count: usize = norm_axes.iter().map(|&a| x.shape().dims()[a]).product();
    let d: Vec<f32> = sum.as_f32_slice().iter().map(|&v| v / count as f32).collect();
    Tensor::from_f32(&d, sum.shape().clone())
}

pub fn reduce_max(x: &Tensor, axes: &[i64], keep_dims: bool) -> Result<Tensor> {
    reduce(x, axes, keep_dims, f32::NEG_INFINITY, f32::max)
}

pub fn reduce_min(x: &Tensor, axes: &[i64], keep_dims: bool) -> Result<Tensor> {
    reduce(x, axes, keep_dims, f32::INFINITY, f32::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_core::Tensor;

    #[test]
    fn sum_all() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = reduce_sum(&x, &[], false).unwrap();
        assert!((y.as_f32_slice()[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn mean_axis0() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = reduce_mean(&x, &[0], false).unwrap();
        let d = y.as_f32_slice();
        assert!((d[0] - 2.0).abs() < 1e-5, "d[0]={}", d[0]);
        assert!((d[1] - 3.0).abs() < 1e-5, "d[1]={}", d[1]);
    }
}
