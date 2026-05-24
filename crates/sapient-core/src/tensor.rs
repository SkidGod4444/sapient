//! `Tensor` — the central multi-dimensional array type in SAPIENT.
//!
//! A `Tensor` owns its shape and dtype metadata, and holds a reference-counted
//! `BufferHandle` for the raw bytes.  Layout is always row-major (C order).

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};

use crate::buffer::{Buffer, BufferHandle, CpuBuffer};
use crate::dtype::DType;
use crate::error::{Result, SapientError};
use crate::shape::Shape;


// ── Tensor ────────────────────────────────────────────────────────────────────

/// A multi-dimensional tensor with reference-counted buffer ownership.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape:   Shape,
    dtype:   DType,
    strides: Vec<usize>,  // row-major by default
    buffer:  BufferHandle,
    // Byte offset into the buffer where element [0,0,...,0] lives.
    offset:  usize,
}

impl Tensor {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a zero-filled tensor on the CPU.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        let numel = shape.numel();
        let strides = shape.strides();
        let buffer = BufferHandle::new(CpuBuffer::zeros(numel, dtype)?);
        Ok(Self { shape, dtype, strides, buffer, offset: 0 })
    }

    /// Create a tensor from a flat `f32` slice (CPU, row-major).
    pub fn from_f32(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        if data.len() != shape.numel() {
            return Err(SapientError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: vec![data.len()],
            });
        }
        let strides = shape.strides();
        let buffer = BufferHandle::new(CpuBuffer::from_f32_slice(data)?);
        Ok(Self { shape, dtype: DType::F32, strides, buffer, offset: 0 })
    }

    /// Create a scalar tensor from a single `f32`.
    pub fn scalar_f32(v: f32) -> Result<Self> {
        Self::from_f32(&[v], Shape::scalar())
    }

    /// Create from a pre-built `BufferHandle` (used by backends).
    pub fn from_buffer(
        shape: impl Into<Shape>,
        dtype: DType,
        buffer: BufferHandle,
        offset: usize,
    ) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        let required = dtype.byte_count(shape.numel());
        if buffer.len() < offset + required {
            return Err(SapientError::BufferSizeMismatch {
                expected: offset + required,
                got: buffer.len(),
            });
        }
        let strides = shape.strides();
        Ok(Self { shape, dtype, strides, buffer, offset })
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn dtype(&self) -> DType  { self.dtype }
    pub fn ndim(&self) -> usize   { self.shape.ndim() }
    pub fn numel(&self) -> usize  { self.shape.numel() }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn buffer(&self) -> &BufferHandle { &self.buffer }
    pub fn offset(&self) -> usize { self.offset }

    /// True if the tensor has a single element.
    pub fn is_scalar(&self) -> bool { self.shape.is_scalar() || self.numel() == 1 }

    /// True if the buffer is row-major contiguous (normal case).
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.strides() && self.offset == 0
    }

    // ── Typed data access (CPU only) ─────────────────────────────────────────

    /// Raw byte view (always works).
    pub fn as_bytes(&self) -> &[u8] {
        let bytes = self.buffer.as_bytes();
        &bytes[self.offset..]
    }

    /// Typed `f32` view — panics if dtype is not F32.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "Tensor dtype is not F32");
        let bytes = self.as_bytes();
        assert_eq!(bytes.len() % 4, 0);
        // SAFETY: alignment ensured by CpuBuffer, dtype checked above.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
    }

    // ── Shape manipulation ───────────────────────────────────────────────────

    /// Returns a new tensor with a different shape but the same buffer.
    /// The total number of elements must be unchanged.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Tensor> {
        let new_shape = self.shape.reshape(new_shape.into().dims().to_vec())?;
        let strides = new_shape.strides();
        Ok(Tensor {
            shape: new_shape,
            dtype: self.dtype,
            strides,
            buffer: self.buffer.clone(),
            offset: self.offset,
        })
    }

    /// Transpose a 2-D tensor (swap axes 0 and 1).
    pub fn t(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(SapientError::internal("t() requires a 2-D tensor"));
        }
        let mut dims  = self.shape.dims().to_vec();
        let mut strides = self.strides.clone();
        dims.swap(0, 1);
        strides.swap(0, 1);
        Ok(Tensor {
            shape: Shape(dims),
            dtype: self.dtype,
            strides,
            buffer: self.buffer.clone(),
            offset: self.offset,
        })
    }

    // ── Metadata convenience ─────────────────────────────────────────────────

    /// Byte count for all elements.
    pub fn byte_size(&self) -> usize {
        self.dtype.byte_count(self.numel())
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={}, dtype={}, device={})",
            self.shape, self.dtype, self.buffer.0.device())
    }
}

// ── Serde support for Tensor ─────────────────────────────────────────────────

/// Serialisable proxy — stores raw f32 data alongside shape/dtype.
#[derive(Serialize, Deserialize)]
struct TensorProxy {
    shape: Shape,
    dtype: DType,
    /// Raw bytes as base64-encoded (for JSON), or raw for binary.
    data: Vec<f32>,
}

impl Serialize for Tensor {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let data: Vec<f32> = if self.dtype == DType::F32 {
            self.as_f32_slice().to_vec()
        } else {
            vec![] // non-f32 tensors: zero data (future work)
        };
        TensorProxy { shape: self.shape.clone(), dtype: self.dtype, data }
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let proxy = TensorProxy::deserialize(deserializer)?;
        if proxy.data.is_empty() {
            Tensor::zeros(proxy.shape, proxy.dtype)
                .map_err(serde::de::Error::custom)
        } else {
            Tensor::from_f32(&proxy.data, proxy.shape)
                .map_err(serde::de::Error::custom)
        }
    }
}

/// A serializable descriptor for a tensor — shape and dtype only (no data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub shape: Shape,
    pub dtype: DType,
}

impl From<&Tensor> for TensorMeta {
    fn from(t: &Tensor) -> Self {
        Self { shape: t.shape.clone(), dtype: t.dtype }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_dtype_shape() {
        let t = Tensor::zeros(vec![2, 3], DType::F32).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn from_f32_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&data, vec![2, 3]).unwrap();
        assert_eq!(t.as_f32_slice(), data.as_slice());
    }

    #[test]
    fn reshape_preserves_data() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&data, vec![2, 3]).unwrap();
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(r.as_f32_slice(), data.as_slice());
    }

    #[test]
    fn reshape_wrong_numel() {
        let t = Tensor::zeros(vec![2, 3], DType::F32).unwrap();
        assert!(t.reshape(vec![5]).is_err());
    }

    #[test]
    fn transpose_2d() {
        let t = Tensor::zeros(vec![3, 4], DType::F32).unwrap();
        let t2 = t.t().unwrap();
        assert_eq!(t2.shape().dims(), &[4, 3]);
    }

    #[test]
    fn byte_size() {
        let t = Tensor::zeros(vec![4, 4], DType::F32).unwrap();
        assert_eq!(t.byte_size(), 64);
    }
}
