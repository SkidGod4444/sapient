// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `Buffer` trait and `CpuBuffer` — aligned, heap-allocated byte storage.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Result, SapientError};

// ── Buffer trait ─────────────────────────────────────────────────────────────

/// A raw byte buffer backing a tensor.
///
/// Implementations may reside on CPU heap, GPU device memory, or shared
/// (unified) memory. The trait is object-safe so backends can return
/// `Arc<dyn Buffer>`.
pub trait Buffer: Send + Sync + std::fmt::Debug {
    /// Returns a raw byte slice over all elements.
    fn as_bytes(&self) -> &[u8];

    /// Returns a mutable raw byte slice (may be unavailable for GPU buffers).
    fn as_bytes_mut(&mut self) -> &mut [u8];

    /// Total capacity in bytes.
    fn len(&self) -> usize;

    /// True when the bytes are a read-only file mapping (weights paged by the
    /// OS). Load-time transforms that would materialize tensors in RAM (e.g.
    /// Q4_K_R4 repacking) must skip mmap-backed tensors or they defeat
    /// bigger-than-RAM loading. Defaults to `false` (heap buffers).
    fn is_mmap(&self) -> bool {
        false
    }

    /// True if the buffer has zero capacity.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Alignment used when this buffer was allocated.
    fn alignment(&self) -> usize;

    /// Textual description of where the buffer lives (e.g. "cpu", "metal").
    fn device(&self) -> &str;
}

// ── BufferHandle ─────────────────────────────────────────────────────────────

/// A reference-counted handle to a `Buffer`.
#[derive(Debug, Clone)]
pub struct BufferHandle(pub Arc<dyn Buffer>);

impl BufferHandle {
    pub fn new(buf: impl Buffer + 'static) -> Self {
        Self(Arc::new(buf))
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_mmap(&self) -> bool {
        self.0.is_mmap()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ── CpuBuffer ────────────────────────────────────────────────────────────────

/// A properly-aligned CPU heap buffer.
///
/// Uses Rust's global allocator directly to guarantee alignment, which
/// `Vec<u8>` cannot guarantee beyond its element alignment (1 byte).
pub struct CpuBuffer {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
    layout: Layout,
}

// SAFETY: The raw pointer is owned exclusively by this struct; we implement
// Send + Sync here because the data behind it is plain bytes.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl std::fmt::Debug for CpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuBuffer")
            .field("len", &self.len)
            .field("align", &self.align)
            .finish()
    }
}

impl CpuBuffer {
    /// Allocate a zero-initialised buffer for `numel` elements of `dtype`.
    pub fn zeros(numel: usize, dtype: DType) -> Result<Self> {
        let bytes = dtype.byte_count(numel);
        let align = dtype.alignment().max(64); // cache-line friendly
        Self::with_capacity(bytes, align)
    }

    /// Allocate `bytes` bytes with the given alignment.
    pub fn with_capacity(bytes: usize, align: usize) -> Result<Self> {
        if bytes == 0 {
            // Zero-size allocation: use a dangling-but-aligned pointer.
            let layout = Layout::from_size_align(1, align)
                .map_err(|_| SapientError::AllocationFailed { bytes, align })?;
            let ptr = unsafe { alloc::alloc_zeroed(layout) };
            let ptr = NonNull::new(ptr).ok_or(SapientError::AllocationFailed { bytes, align })?;
            return Ok(Self {
                ptr,
                len: 0,
                align,
                layout,
            });
        }

        let layout = Layout::from_size_align(bytes, align)
            .map_err(|_| SapientError::AllocationFailed { bytes, align })?;

        // SAFETY: layout is well-formed.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or(SapientError::AllocationFailed { bytes, align })?;

        Ok(Self {
            ptr,
            len: bytes,
            align,
            layout,
        })
    }

    /// Wrap existing `f32` data (copies into a new aligned allocation).
    pub fn from_f32_slice(data: &[f32]) -> Result<Self> {
        let bytes = data.len() * 4;
        let buf = Self::with_capacity(bytes, 64)?;
        // SAFETY: sizes are consistent and both regions are valid.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.ptr.as_ptr(), bytes);
        }
        Ok(buf)
    }

    /// Take ownership of a `Vec<f32>` without copying.
    /// Eliminates the allocation + memcopy overhead in `from_f32_slice`.
    /// Used by hot matmul paths that already have a Vec<f32> output buffer.
    pub fn from_f32_vec(data: Vec<f32>) -> Result<Self> {
        if data.is_empty() {
            return Self::with_capacity(0, 4);
        }
        let len = data.len() * 4;
        let layout =
            Layout::array::<f32>(data.len()).map_err(|_| SapientError::AllocationFailed {
                bytes: len,
                align: 4,
            })?;
        let ptr = data.as_ptr() as *mut u8;
        // SAFETY: We transfer ownership from the Vec — `forget` prevents the Vec's
        // drop from freeing the allocation; we free it ourselves in CpuBuffer::drop
        // using the same layout that the Vec's allocator used.
        std::mem::forget(data);
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(SapientError::AllocationFailed {
                bytes: len,
                align: 4,
            })?,
            len,
            align: std::mem::align_of::<f32>(),
            layout,
        })
    }

    /// Wrap existing raw bytes (e.g., native BF16 or F16 from safetensors).
    pub fn from_bytes_slice(data: &[u8]) -> Result<Self> {
        let bytes = data.len();
        if bytes == 0 {
            return Self::with_capacity(0, 16);
        }
        let buf = Self::with_capacity(bytes, 16)?;
        // SAFETY: sizes are consistent and both regions are valid.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr.as_ptr(), bytes);
        }
        Ok(buf)
    }

    /// View as `f32` slice (panics if not properly aligned/sized).
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.len % 4, 0, "buffer length not a multiple of 4");
        // SAFETY: alignment guaranteed, size checked above.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const f32, self.len / 4) }
    }

    /// Mutable view as `f32` slice.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.len % 4, 0, "buffer length not a multiple of 4");
        // SAFETY: alignment guaranteed, exclusive via `&mut`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut f32, self.len / 4) }
    }

    /// Pointer to raw memory (useful for BLAS calls).
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Buffer for CpuBuffer {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: ptr is valid for `len` bytes, exclusively owned.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn alignment(&self) -> usize {
        self.align
    }

    fn device(&self) -> &str {
        "cpu"
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        // SAFETY: layout matches the original allocation.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_read() {
        let buf = CpuBuffer::zeros(4, DType::F32).unwrap();
        assert_eq!(buf.len(), 16);
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn from_f32_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buf = CpuBuffer::from_f32_slice(&data).unwrap();
        assert_eq!(buf.as_f32_slice(), data.as_slice());
    }

    #[test]
    fn alignment_guarantee() {
        let buf = CpuBuffer::with_capacity(32, 64).unwrap();
        assert_eq!(buf.as_ptr() as usize % 64, 0);
    }
}
