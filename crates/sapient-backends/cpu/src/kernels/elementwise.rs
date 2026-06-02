//! Element-wise CPU kernels — arithmetic, activations, and mathematical ops.
//!
//! All kernels operate on F32 tensors. Binary ops support same-shape operands
//! only (broadcasting handled by the dispatch layer after shape inference).

use sapient_core::error::{Result, SapientError};
use sapient_core::{DType, Tensor};

// ── Helper ────────────────────────────────────────────────────────────────────

/// Apply a unary f32 function element-wise.
fn unary_f32<F: Fn(f32) -> f32>(x: &Tensor, f: F) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        return Err(SapientError::TypeMismatch {
            expected: "f32".into(),
            got: x.dtype().to_string(),
        });
    }
    let data: Vec<f32> = x.to_f32_cow().iter().map(|&v| f(v)).collect();
    Tensor::from_f32(&data, x.shape().clone())
}

/// Apply a binary f32 function element-wise (same shape only).
fn binary_f32<F: Fn(f32, f32) -> f32>(a: &Tensor, b: &Tensor, f: F) -> Result<Tensor> {
    // Handle scalar broadcast (numel == 1).
    let a_cow = a.to_f32_cow();
    let a_data = a_cow.as_ref();
    let b_cow = b.to_f32_cow();
    let b_data = b_cow.as_ref();

    let (out, shape) = if a_data.len() == b_data.len() {
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        (out, a.shape().clone())
    } else if b_data.len() == 1 {
        let scalar = b_data[0];
        let out: Vec<f32> = a_data.iter().map(|&x| f(x, scalar)).collect();
        (out, a.shape().clone())
    } else if a_data.len() == 1 {
        let scalar = a_data[0];
        let out: Vec<f32> = b_data.iter().map(|&y| f(scalar, y)).collect();
        (out, b.shape().clone())
    } else {
        return Err(SapientError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    };

    Tensor::from_f32(&out, shape)
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x + y)
}
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x - y)
}
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x * y)
}
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x / y)
}
pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x.powf(y))
}

pub fn neg(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| -v)
}
pub fn abs(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.abs())
}
pub fn sqrt(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.sqrt())
}
pub fn exp(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.exp())
}
pub fn log(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.ln())
}
pub fn erf(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, erf_approx)
}
pub fn floor(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.floor())
}
pub fn ceil(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.ceil())
}
pub fn round(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.round())
}

// ── Activations ───────────────────────────────────────────────────────────────

pub fn relu(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.max(0.0))
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| 1.0 / (1.0 + (-v).exp()))
}

pub fn tanh_act(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.tanh())
}

/// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEF: f32 = 0.044_715;
    unary_f32(x, |v| {
        let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// Exact (erf-based) GELU: `0.5 * x * (1 + erf(x / √2))`.
///
/// This is the variant used by HuggingFace/OpenAI Whisper (`activation_function
/// = "gelu"`), as distinct from the tanh approximation in [`gelu`]. The two
/// differ by < 1e-3 per element but the error compounds across a Whisper
/// encoder/decoder stack, so the audio path uses this exact form.
pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    unary_f32(x, |v| 0.5 * v * (1.0 + erf_approx(v * INV_SQRT_2)))
}

/// SiLU / Swish: x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v / (1.0 + (-v).exp()))
}

/// Hard Swish: x * relu6(x + 3) / 6
pub fn hard_swish(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v * (v + 3.0).clamp(0.0, 6.0) / 6.0)
}

pub fn leaky_relu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    unary_f32(x, |v| if v >= 0.0 { v } else { alpha * v })
}

pub fn clip(x: &Tensor, min: Option<f32>, max: Option<f32>) -> Result<Tensor> {
    unary_f32(x, |v| {
        let v = min.map_or(v, |lo| v.max(lo));
        max.map_or(v, |hi| v.min(hi))
    })
}

// ── Erf approximation (Abramowitz & Stegun) ───────────────────────────────────

fn erf_approx(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    // Rational approximation — max error ~1.5e-7.
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (0.254_829_59
            + (-0.284_496_74 + (1.421_413_74 + (-1.453_152_03 + 1.061_405_43 * t) * t) * t) * t)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32]) -> Tensor {
        Tensor::from_f32(data, vec![data.len()]).unwrap()
    }

    #[test]
    fn test_add() {
        assert!(
            (add(&t(&[1.0, 2.0]), &t(&[3.0, 4.0]))
                .unwrap()
                .as_f32_slice()[0]
                - 4.0)
                .abs()
                < 1e-6
        );
    }
    #[test]
    fn test_relu() {
        let r = relu(&t(&[-1.0, 0.0, 1.0])).unwrap();
        let d = r.as_f32_slice();
        assert_eq!(d, &[0.0, 0.0, 1.0]);
    }
    #[test]
    fn test_sigmoid() {
        let v = sigmoid(&t(&[0.0])).unwrap().as_f32_slice()[0];
        assert!((v - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_gelu() {
        let v = gelu(&t(&[0.0])).unwrap().as_f32_slice()[0];
        assert!(v.abs() < 1e-5);
    }
    #[test]
    fn test_erf() {
        let v = erf_approx(0.0);
        assert!(v.abs() < 1e-6, "erf(0) should be ~0, got {v}");
    }
    #[test]
    fn test_gelu_erf() {
        // Exact GELU: g(0)=0, g(1)=0.8413447, g(-1)=-0.1586553.
        let out = gelu_erf(&t(&[0.0, 1.0, -1.0])).unwrap();
        let v = out.as_f32_slice();
        assert!(v[0].abs() < 1e-6);
        assert!((v[1] - 0.841_344_7).abs() < 1e-4, "g(1)={}", v[1]);
        assert!((v[2] - (-0.158_655_3)).abs() < 1e-4, "g(-1)={}", v[2]);
    }
    #[test]
    fn test_scalar_broadcast() {
        let a = t(&[1.0, 2.0, 3.0]);
        let b = t(&[2.0]);
        let r = mul(&a, &b).unwrap();
        assert_eq!(r.as_f32_slice(), &[2.0, 4.0, 6.0]);
    }
}
