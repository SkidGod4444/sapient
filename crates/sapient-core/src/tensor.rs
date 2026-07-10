//! `Tensor` — the central multi-dimensional array type in SAPIENT.
//!
//! A `Tensor` owns its shape and dtype metadata, and holds a reference-counted
//! `BufferHandle` for the raw bytes.  Layout is always row-major (C order).

use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};
use std::sync::Arc;

use crate::buffer::{BufferHandle, CpuBuffer};
use crate::dtype::DType;
use crate::error::{Result, SapientError};
use crate::shape::Shape;

// ── Tensor ────────────────────────────────────────────────────────────────────

/// A multi-dimensional tensor with reference-counted buffer ownership.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    dtype: DType,
    strides: Vec<usize>, // row-major by default
    buffer: BufferHandle,
    // Byte offset into the buffer where element [0,0,...,0] lives.
    offset: usize,
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
        Ok(Self {
            shape,
            dtype,
            strides,
            buffer,
            offset: 0,
        })
    }

    /// Create a tensor from a flat `f32` slice (CPU, row-major).
    /// Take ownership of a `Vec<f32>` without copying.
    /// Use instead of `from_f32` in hot paths to avoid the allocation + memcpy.
    pub fn from_f32_vec(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        if data.len() != shape.numel() {
            return Err(SapientError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: vec![data.len()],
            });
        }
        let strides = shape.strides();
        let buffer = BufferHandle::new(CpuBuffer::from_f32_vec(data)?);
        Ok(Self {
            shape,
            dtype: DType::F32,
            strides,
            buffer,
            offset: 0,
        })
    }

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
        Ok(Self {
            shape,
            dtype: DType::F32,
            strides,
            buffer,
            offset: 0,
        })
    }

    /// Create a tensor from raw BF16 bytes, storing them natively without conversion.
    /// Use `to_f32_vec()` or `to_f32_tensor()` to convert for computation.
    pub fn from_bf16_bytes(data: &[u8], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        let expected_bytes = shape.numel() * 2;
        if data.len() != expected_bytes {
            return Err(SapientError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: vec![data.len() / 2],
            });
        }
        let strides = shape.strides();
        let buffer = BufferHandle::new(CpuBuffer::from_bytes_slice(data)?);
        Ok(Self {
            shape,
            dtype: DType::BF16,
            strides,
            buffer,
            offset: 0,
        })
    }

    /// Create a tensor from raw F16 bytes, storing them natively without conversion.
    pub fn from_f16_bytes(data: &[u8], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        let expected_bytes = shape.numel() * 2;
        if data.len() != expected_bytes {
            return Err(SapientError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: vec![data.len() / 2],
            });
        }
        let strides = shape.strides();
        let buffer = BufferHandle::new(CpuBuffer::from_bytes_slice(data)?);
        Ok(Self {
            shape,
            dtype: DType::F16,
            strides,
            buffer,
            offset: 0,
        })
    }

    /// Create a quantized tensor from raw block bytes (Q4_0 / Q8_0).
    ///
    /// `data` must contain exactly `dtype.byte_count(shape.numel())` bytes, i.e.
    /// the packed ggml block bytes with no expansion.  The shape describes the
    /// *logical* element count; `shape.numel()` must be a multiple of 32.
    pub fn from_quant_bytes(data: &[u8], shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        if !dtype.is_quantized() {
            return Err(SapientError::TypeMismatch {
                expected: "a quantized dtype (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)".into(),
                got: dtype.to_string(),
            });
        }
        let shape = shape.into();
        shape.validate()?;
        let numel = shape.numel();
        let expected_bytes = dtype.byte_count(numel);
        if data.len() != expected_bytes {
            return Err(SapientError::ShapeMismatch {
                expected: vec![expected_bytes],
                got: vec![data.len()],
            });
        }
        let strides = shape.strides();
        let buffer = BufferHandle::new(CpuBuffer::from_bytes_slice(data)?);
        Ok(Self {
            shape,
            dtype,
            strides,
            buffer,
            offset: 0,
        })
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
        Ok(Self {
            shape,
            dtype,
            strides,
            buffer,
            offset,
        })
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn buffer(&self) -> &BufferHandle {
        &self.buffer
    }
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// True if the tensor has a single element.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar() || self.numel() == 1
    }

    /// True if the buffer is row-major contiguous (normal case).
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.strides() && self.offset == 0
    }

    // ── Typed data access (CPU only) ─────────────────────────────────────────

    /// Raw byte view. For non-quantized tensors returns the full buffer slice from
    /// `offset` onwards (preserving the original behavior that stride-based kernels
    /// rely on). For quantized tensors (Q4_0/Q8_0) returns exactly the packed block
    /// bytes for this tensor's logical shape.
    pub fn as_bytes(&self) -> &[u8] {
        let bytes = self.buffer.as_bytes();
        if self.dtype.is_quantized() {
            let end = self.offset + self.dtype.byte_count(self.numel());
            &bytes[self.offset..end]
        } else {
            &bytes[self.offset..]
        }
    }

    /// True when this tensor's bytes are an OS file mapping (see
    /// [`crate::buffer::Buffer::is_mmap`]).
    pub fn is_mmap(&self) -> bool {
        self.buffer.is_mmap()
    }

    /// For quantized tensors (Q4_0, Q8_0): returns the packed block bytes as a
    /// row-major slice where each logical row of `k` elements occupies
    /// `dtype.byte_count(k)` bytes.  Panics if the tensor is not quantized.
    pub fn as_quant_blocks(&self) -> &[u8] {
        assert!(
            self.dtype.is_quantized(),
            "as_quant_blocks() called on non-quantized tensor (dtype = {})",
            self.dtype
        );
        self.as_bytes()
    }

    /// Typed `f32` view — panics if dtype is not F32.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "Tensor dtype is not F32 — call to_f32_vec() instead"
        );
        let bytes = self.as_bytes();
        assert_eq!(bytes.len() % 4, 0);
        // SAFETY: alignment ensured by CpuBuffer, dtype checked above.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
    }

    /// Convert this tensor to a contiguous `Vec<f32>` that matches `shape().numel()`
    /// exactly, even when the tensor is a **non-contiguous view** (e.g. a KV-cache
    /// slice from `slice_axis`).
    ///
    /// For contiguous tensors this is a fast bounded copy.  For non-contiguous
    /// tensors (strides don't match the natural row-major strides, or offset ≠ 0)
    /// it uses stride-based indexing to extract only the logically-reachable elements
    /// in row-major order — the approach that `as_f32_slice` / `to_f32_cow` cannot
    /// do because they return the full backing buffer.
    pub fn to_contiguous_f32_vec(&self) -> Vec<f32> {
        let numel = self.numel();
        if self.is_contiguous() {
            // Fast path: elements are dense starting at offset.
            // Limit to numel to avoid reading past the logical tensor.
            match self.dtype {
                DType::F32 => self.as_f32_slice()[..numel].to_vec(),
                _ => {
                    let v = self.to_f32_vec();
                    v[..numel.min(v.len())].to_vec()
                }
            }
        } else {
            // Slow path: stride-based copy.
            // `raw` gives us the full backing buffer from `self.offset` as f32 units.
            let raw: Vec<f32> = match self.dtype {
                DType::F32 => self.as_f32_slice().to_vec(),
                _ => self.to_f32_vec(),
            };
            let dims = self.shape.dims();
            let strides = &self.strides; // element strides (not byte strides)
            let mut out = vec![0.0f32; numel];
            for (flat, dst) in out.iter_mut().enumerate() {
                // Convert flat (row-major) index to per-dimension indices, then
                // compute the element offset using the tensor's actual strides.
                let mut rem = flat;
                let mut src = 0usize;
                for d in (0..dims.len()).rev() {
                    let idx_d = rem % dims[d];
                    rem /= dims[d];
                    src += idx_d * strides[d];
                }
                *dst = *raw.get(src).unwrap_or(&0.0);
            }
            out
        }
    }

    /// Returns a `Cow<[f32]>`. Borrows if the tensor is already F32, otherwise allocates a new `Vec<f32>`.
    pub fn to_f32_cow(&self) -> std::borrow::Cow<'_, [f32]> {
        if self.dtype == DType::F32 {
            std::borrow::Cow::Borrowed(self.as_f32_slice())
        } else {
            std::borrow::Cow::Owned(self.to_f32_vec())
        }
    }

    /// Convert this tensor to a `Vec<f32>`, handling all dtypes including quantized.
    /// For F32: cheap copy. For F16/BF16: convert. For quantized: dequantize all blocks.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        use crate::dtype::{
            K_QUANT_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES,
            Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QUANT_BLOCK_SIZE,
        };
        match self.dtype {
            DType::F32 => self.as_f32_slice().to_vec(),
            DType::BF16 => {
                let bytes = self.as_bytes();
                bytes
                    .chunks_exact(2)
                    .map(|c| f32::from(half::bf16::from_le_bytes(c.try_into().unwrap())))
                    .collect()
            }
            DType::F16 => {
                let bytes = self.as_bytes();
                bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                    .collect()
            }
            DType::Q4_0 => {
                let numel = self.numel();
                let bytes = self.as_bytes();
                let mut out = vec![0.0f32; numel];
                for (b, block) in bytes.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
                    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..QUANT_BLOCK_SIZE / 2 {
                        let byte = block[2 + j];
                        let lo = (byte & 0x0f) as i32 - 8;
                        let hi = (byte >> 4) as i32 - 8;
                        out[b * QUANT_BLOCK_SIZE + j] = lo as f32 * d;
                        out[b * QUANT_BLOCK_SIZE + j + QUANT_BLOCK_SIZE / 2] = hi as f32 * d;
                    }
                }
                out
            }
            DType::Q8_0 => {
                let numel = self.numel();
                let bytes = self.as_bytes();
                let mut out = vec![0.0f32; numel];
                for (b, block) in bytes.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
                    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..QUANT_BLOCK_SIZE {
                        out[b * QUANT_BLOCK_SIZE + j] = block[2 + j] as i8 as f32 * d;
                    }
                }
                out
            }
            DType::Q4_K => {
                let numel = self.numel();
                let bytes = self.as_bytes();
                let mut out = vec![0.0f32; numel];
                for (b, block) in bytes.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
                    Self::dequant_q4_k_block(block, &mut out[b * K_QUANT_BLOCK_SIZE..]);
                }
                out
            }
            DType::Q4_K_R4 => {
                // Row-interleaved Q4_K (groups of 4 rows, block-major within the
                // group): packed block index p = g·(4·NB) + b·4 + r maps to
                // logical row g·4 + r, super-block b. De-interleave while
                // dequantizing so callers see the normal row-major values.
                let numel = self.numel();
                let dims = self.shape.dims();
                let k = *dims.last().expect("Q4_K_R4 tensor must be 2-D");
                let nb = k / K_QUANT_BLOCK_SIZE; // super-blocks per row
                let bytes = self.as_bytes();
                let mut out = vec![0.0f32; numel];
                for (p, block) in bytes.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
                    let g = p / (4 * nb);
                    let rem = p % (4 * nb);
                    let b = rem / 4;
                    let r = rem % 4;
                    let row = g * 4 + r;
                    let off = (row * nb + b) * K_QUANT_BLOCK_SIZE;
                    Self::dequant_q4_k_block(block, &mut out[off..]);
                }
                out
            }
            DType::Q5_K => {
                let numel = self.numel();
                let bytes = self.as_bytes();
                let mut out = vec![0.0f32; numel];
                let mut out_idx = 0usize;
                for block in bytes.chunks_exact(Q5_K_BLOCK_BYTES) {
                    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let scales = &block[4..16];
                    let qh = &block[16..48];
                    let ql = &block[48..Q5_K_BLOCK_BYTES];
                    let mut ql_off = 0usize;
                    let mut is = 0usize;
                    let mut u1: u8 = 1;
                    let mut u2: u8 = 2;
                    for _ in 0..(K_QUANT_BLOCK_SIZE / 64) {
                        let (sc1, m1) = Self::get_scale_min_k4(is, scales);
                        let d1 = d * sc1 as f32;
                        let m1v = dmin * m1 as f32;
                        let (sc2, m2) = Self::get_scale_min_k4(is + 1, scales);
                        let d2 = d * sc2 as f32;
                        let m2v = dmin * m2 as f32;
                        // The 5th bit is PER-ELEMENT: ggml reads qh[l] and selects
                        // the active bit-plane with u1/u2 (same fix as the CPU
                        // kernel's dot_q5_k_row_f32_scalar — a single qh[is/8]
                        // byte collapses 32 distinct high bits to one and
                        // corrupts every Q5_K tensor this dequantizes).
                        for l in 0..32usize {
                            let hi = if qh[l] & u1 != 0 { 16.0f32 } else { 0.0 };
                            out[out_idx + l] = d1 * ((ql[ql_off + l] & 0x0F) as f32 + hi) - m1v;
                            let hi2 = if qh[l] & u2 != 0 { 16.0f32 } else { 0.0 };
                            out[out_idx + l + 32] = d2 * ((ql[ql_off + l] >> 4) as f32 + hi2) - m2v;
                        }
                        out_idx += 64;
                        ql_off += 32;
                        is += 2;
                        if is % 8 == 0 {
                            u1 = 1;
                            u2 = 2;
                        } else {
                            u1 <<= 2;
                            u2 <<= 2;
                        }
                    }
                }
                out
            }
            DType::Q6_K => {
                let numel = self.numel();
                let bytes = self.as_bytes();
                let mut out = vec![0.0f32; numel];
                for (b, block) in bytes.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
                    Self::dequant_q6_k_block(block, &mut out[b * K_QUANT_BLOCK_SIZE..]);
                }
                out
            }
            DType::Q6_K_R4 => {
                // Row-interleaved Q6_K — identical permutation to Q4_K_R4 (see
                // that arm) over 210-byte blocks.
                let numel = self.numel();
                let dims = self.shape.dims();
                let k = *dims.last().expect("Q6_K_R4 tensor must be 2-D");
                let nb = k / K_QUANT_BLOCK_SIZE;
                let bytes = self.as_bytes();
                let mut out = vec![0.0f32; numel];
                for (p, block) in bytes.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
                    let g = p / (4 * nb);
                    let rem = p % (4 * nb);
                    let b = rem / 4;
                    let r = rem % 4;
                    let off = ((g * 4 + r) * nb + b) * K_QUANT_BLOCK_SIZE;
                    Self::dequant_q6_k_block(block, &mut out[off..]);
                }
                out
            }
            _ => self.as_f32_slice().to_vec(), // fallback for integer dtypes
        }
    }

    /// Extract scale and min for a K-quant sub-block (used in Q4_K/Q5_K dequantization).
    #[inline]
    /// Dequantize one 144-byte Q4_K super-block into `out[..256]` (shared by the
    /// row-major Q4_K and row-interleaved Q4_K_R4 `to_f32_vec` paths).
    fn dequant_q4_k_block(block: &[u8], out: &mut [f32]) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..crate::dtype::Q4_K_BLOCK_BYTES];
        let mut out_idx = 0usize;
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(crate::dtype::K_QUANT_BLOCK_SIZE / 64) {
            let (sc1, m1) = Self::get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let (sc2, m2) = Self::get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;
            for l in 0..32 {
                out[out_idx + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1v;
                out[out_idx + l + 32] = d2 * (qs[q_off + l] >> 4) as f32 - m2v;
            }
            out_idx += 64;
            q_off += 32;
            is += 2;
        }
    }

    /// Dequantize one 210-byte Q6_K super-block into `out[..256]` (shared by the
    /// row-major Q6_K and row-interleaved Q6_K_R4 `to_f32_vec` paths). 16 i8
    /// scales per super-block; within each 128-element half the 4 sub-groups use
    /// scale offsets +0/+2/+4/+6 with a split at l==16 (`is = l/16`), base
    /// advancing by 8 per 128-block — matches ggml dequantize_row_q6_K.
    fn dequant_q6_k_block(block: &[u8], out: &mut [f32]) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
        let mut out_idx = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_base = 0usize;
        for _ in 0..(crate::dtype::K_QUANT_BLOCK_SIZE / 128) {
            for l in 0..32usize {
                let is = l / 16;
                let q1 =
                    (((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32) as f32;
                let q2 = (((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4))
                    as i32
                    - 32) as f32;
                let q3 = (((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32)
                    as f32;
                let q4 = (((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32
                    - 32) as f32;
                out[out_idx + l] = d * sc[sc_base + is] as i8 as f32 * q1;
                out[out_idx + l + 32] = d * sc[sc_base + is + 2] as i8 as f32 * q2;
                out[out_idx + l + 64] = d * sc[sc_base + is + 4] as i8 as f32 * q3;
                out[out_idx + l + 96] = d * sc[sc_base + is + 6] as i8 as f32 * q4;
            }
            out_idx += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }

    fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
        if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            (
                (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )
        }
    }

    /// Returns an F32 tensor, converting BF16/F16 if necessary.
    /// For already-F32 tensors, clones the buffer. For native types, converts.
    pub fn to_f32_tensor(&self) -> Result<Tensor> {
        match self.dtype {
            DType::F32 => Ok(self.clone()),
            _ => Tensor::from_f32(&self.to_f32_vec(), self.shape.clone()),
        }
    }

    /// Mutable typed `f32` view — fails if buffer is shared or not F32.
    /// Mutable byte access for quantized tensors — **in-place** update with zero copy.
    /// Returns an error if the buffer is shared (Arc strong_count > 1).
    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        let offset = self.offset;
        let end = offset + self.dtype.byte_count(self.numel());
        let buf = Arc::get_mut(&mut self.buffer.0)
            .ok_or_else(|| SapientError::internal("Cannot mutate shared tensor buffer"))?;
        let bytes = buf.as_bytes_mut();
        Ok(&mut bytes[offset..end])
    }

    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32]> {
        if self.dtype != DType::F32 {
            return Err(SapientError::internal("Tensor dtype is not F32"));
        }
        let offset = self.offset;
        let buf = Arc::get_mut(&mut self.buffer.0)
            .ok_or_else(|| SapientError::internal("Cannot mutate shared tensor buffer"))?;
        let bytes = buf.as_bytes_mut();
        let bytes = &mut bytes[offset..];
        if bytes.len() % 4 != 0 {
            return Err(SapientError::internal("Buffer length not a multiple of 4"));
        }
        // SAFETY: alignment ensured by CpuBuffer, dtype checked above.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4)
        })
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
        let mut dims = self.shape.dims().to_vec();
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

    /// Return a view of the tensor sliced along the given axis.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> Result<Tensor> {
        let mut dims = self.shape.dims().to_vec();
        if axis >= dims.len() {
            return Err(SapientError::internal("slice axis out of bounds"));
        }
        if start > end || end > dims[axis] {
            return Err(SapientError::internal("slice range out of bounds"));
        }
        dims[axis] = end - start;
        let offset = self.offset + start * self.strides[axis] * self.dtype.element_size();
        Ok(Tensor {
            shape: Shape(dims),
            dtype: self.dtype,
            strides: self.strides.clone(),
            buffer: self.buffer.clone(),
            offset,
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
        write!(
            f,
            "Tensor(shape={}, dtype={}, device={})",
            self.shape,
            self.dtype,
            self.buffer.0.device()
        )
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
        TensorProxy {
            shape: self.shape.clone(),
            dtype: self.dtype,
            data,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let proxy = TensorProxy::deserialize(deserializer)?;
        if proxy.data.is_empty() {
            Tensor::zeros(proxy.shape, proxy.dtype).map_err(serde::de::Error::custom)
        } else {
            Tensor::from_f32(&proxy.data, proxy.shape).map_err(serde::de::Error::custom)
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
        Self {
            shape: t.shape.clone(),
            dtype: t.dtype,
        }
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

    /// Q5_K's 5th bit is per-element: ggml stores it in qh[l] and selects the
    /// bit-plane with u1/u2 (shifting by 2 per 64-element pair). A regression
    /// for reading one qh byte per 32-element sub-block, which collapses 32
    /// distinct high bits to one and corrupts every Q5_K tensor.
    #[test]
    fn q5_k_dequant_high_bits_per_element() {
        let mut block = vec![0u8; crate::dtype::Q5_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d = 1
        block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes()); // dmin = 0
                                                                              // scales: sc=1, m=0 for all 8 sub-blocks (6-bit K-quant packing).
        for i in 0..4 {
            block[4 + i] = 1; // sc for is=0..3
            block[12 + i] = 1; // low nibble → sc for is=4..7
        }
        let qh = &mut block[16..48];
        qh[5] = 0b0000_0001; // bit-plane u1 of group 0 → element 5
        qh[0] = 0b0000_0100; // bit-plane u1 of group 1 → element 64
        let ql = &mut block[48..176];
        ql[3] = 0x02; // element 3 low nibble = 2

        let t = Tensor::from_quant_bytes(&block, vec![256], DType::Q5_K).unwrap();
        let out = t.to_f32_vec();
        assert_eq!(out.len(), 256);
        assert_eq!(out[3], 2.0, "low-nibble value");
        assert_eq!(out[5], 16.0, "per-element high bit (group 0, u1 plane)");
        assert_eq!(out[64], 16.0, "per-element high bit (group 1, u1 plane)");
        let hot = [3usize, 5, 64];
        for (i, v) in out.iter().enumerate() {
            if !hot.contains(&i) {
                assert_eq!(*v, 0.0, "element {i} should be zero");
            }
        }
    }
}
