//! Quantized GPU-resident weights (Phase 7.1 — in-shader Q8_0 dequant).
//!
//! Raw ggml Q8_0 blocks (34 bytes per 32 weights: one little-endian f16 scale +
//! 32 int8 quants) are repacked once on upload into two GPU storage buffers —
//! the int8 quants as `u32` words and the scales widened to f32 — and stay
//! quantized on-device: `matmul_nt_q8_0` and `embed_q8_0` dequantize inside the
//! shader. No host-side f32 expansion of the weight matrix ever happens, so a
//! Q8_0 tensor costs ~1.125 bytes/weight of VRAM instead of the 4 bytes/weight
//! the f32 upload path pays (a 3.6× reduction).
//!
//! The repack is a byte shuffle, not a numeric change: the shader computes
//! exactly `f32(scale_f16) * f32(int8)` — the same dequant value the CPU
//! kernels use — so GPU output matches the CPU dequant reference to float
//! reduction order.

use wgpu::util::DeviceExt;

use crate::context::{WgpuContext, WgpuError};
use crate::resident::GpuBuffer;

/// Weights per Q8_0 block.
const BLOCK: usize = 32;
/// Bytes per raw ggml Q8_0 block (f16 scale + 32 int8).
const BLOCK_BYTES: usize = 34;
/// Weights per Q4_K super-block.
const K_BLOCK: usize = 256;
/// Bytes per raw ggml Q4_K super-block (d + dmin f16, 12 scale bytes, 128 qs bytes).
const Q4_K_BLOCK_BYTES: usize = 144;

/// A Q8_0-quantized tensor resident in GPU storage buffers: `qs` holds the int8
/// quants packed 4-per-`u32` word (little-endian lanes), `scales` one f32 per
/// 32-weight block. `len` is the logical element count; callers track shape
/// separately (kernels take explicit dims), same convention as [`GpuBuffer`].
pub struct GpuQ8Buffer {
    pub(crate) qs: wgpu::Buffer,
    pub(crate) scales: wgpu::Buffer,
    /// Number of logical (dequantized) elements.
    pub len: usize,
}

impl GpuQ8Buffer {
    /// GPU bytes this tensor occupies (quants + scales) — ~1.125 × element count.
    pub fn byte_size(&self) -> usize {
        self.len + self.len / BLOCK * 4
    }
}

/// A Q4_K-quantized tensor resident in one GPU storage buffer holding the **raw
/// ggml super-blocks verbatim** (144 bytes per 256 weights — word-aligned, so the
/// upload is a plain memcpy with no repack). Kernels decode d/dmin, the packed
/// 6-bit scale/min pairs, and the 4-bit quants in-shader.
pub struct GpuQ4KBuffer {
    pub(crate) qb: wgpu::Buffer,
    /// Number of logical (dequantized) elements.
    pub len: usize,
}

impl GpuQ4KBuffer {
    /// GPU bytes this tensor occupies — 0.5625 × element count.
    pub fn byte_size(&self) -> usize {
        self.len / K_BLOCK * Q4_K_BLOCK_BYTES
    }
}

impl WgpuContext {
    /// Upload raw ggml Q8_0 block bytes (`numel/32` × 34-byte blocks, e.g. from
    /// `Tensor::as_quant_blocks()`) into GPU-resident quantized storage. The
    /// weights are never expanded to f32 on the host — only the per-block f16
    /// scales are widened (4 bytes per 32 weights).
    pub fn upload_q8_0(
        &self,
        blocks: &[u8],
        numel: usize,
        label: &str,
    ) -> Result<GpuQ8Buffer, WgpuError> {
        if numel % BLOCK != 0 || blocks.len() != numel / BLOCK * BLOCK_BYTES {
            return Err(WgpuError::Shape(format!(
                "Q8_0 upload '{label}': {} block bytes for {numel} elements \
                 (expected {} for {} blocks of {BLOCK_BYTES})",
                blocks.len(),
                numel / BLOCK * BLOCK_BYTES,
                numel / BLOCK,
            )));
        }
        let nblocks = numel / BLOCK;
        let mut scales = Vec::with_capacity(nblocks);
        let mut qs = Vec::with_capacity(nblocks * (BLOCK / 4));
        for b in blocks.chunks_exact(BLOCK_BYTES) {
            scales.push(half::f16::from_le_bytes([b[0], b[1]]).to_f32());
            for w in b[2..].chunks_exact(4) {
                qs.push(u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            }
        }
        let qs = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&qs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scales = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&scales),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuQ8Buffer {
            qs,
            scales,
            len: numel,
        })
    }

    /// Resident quantized linear projection `out[m,n] = x[m,k] @ dequant(w)[n,k]^T`
    /// (w in HF `[n,k]` layout, Q8_0-resident). Dequantizes in-shader; GEMV-style
    /// cooperative reduction, f32 accumulation — the quantized twin of `matmul_nt`.
    pub fn matmul_nt_q8_0(
        &self,
        x: &GpuBuffer,
        w: &GpuQ8Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> GpuBuffer {
        debug_assert_eq!(k % BLOCK, 0, "Q8_0 matmul: k must be a multiple of 32");
        debug_assert_eq!(w.len, n * k, "Q8_0 matmul: weight numel mismatch");
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            m: u32,
            k: u32,
            n: u32,
            _pad: u32,
        }
        let out = self.alloc_f32(m * n, "matmul_q8.out");
        let params = self.uniform(
            &P {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                _pad: 0,
            },
            "matmul_q8.params",
        );
        self.dispatch(
            "matmul_nt_q8_0",
            include_str!("shaders/matmul_nt_q8_0.wgsl"),
            &[&x.buf, &w.qs, &w.scales, &out.buf],
            &params,
            (m * n) as u32,
        );
        out
    }

    /// Upload raw ggml Q4_K super-block bytes (`numel/256` × 144-byte blocks, e.g.
    /// from `Tensor::as_quant_blocks()`) into GPU-resident quantized storage. The
    /// bytes upload **verbatim** (144 is a multiple of 4, so the blocks bind
    /// directly as `array<u32>`) — no repack, no dequantization, no f32 copy.
    pub fn upload_q4_k(
        &self,
        blocks: &[u8],
        numel: usize,
        label: &str,
    ) -> Result<GpuQ4KBuffer, WgpuError> {
        if numel % K_BLOCK != 0 || blocks.len() != numel / K_BLOCK * Q4_K_BLOCK_BYTES {
            return Err(WgpuError::Shape(format!(
                "Q4_K upload '{label}': {} block bytes for {numel} elements \
                 (expected {} for {} blocks of {Q4_K_BLOCK_BYTES})",
                blocks.len(),
                numel / K_BLOCK * Q4_K_BLOCK_BYTES,
                numel / K_BLOCK,
            )));
        }
        let qb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: blocks,
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuQ4KBuffer { qb, len: numel })
    }

    /// Resident Q4_K linear projection `out[m,n] = x[m,k] @ dequant(w)[n,k]^T`
    /// (w in HF `[n,k]` layout, raw super-blocks resident). Decodes the 6-bit
    /// scale/min pairs and 4-bit quants in-shader; GEMV-style cooperative
    /// reduction, f32 accumulation.
    pub fn matmul_nt_q4_k(
        &self,
        x: &GpuBuffer,
        w: &GpuQ4KBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> GpuBuffer {
        debug_assert_eq!(k % K_BLOCK, 0, "Q4_K matmul: k must be a multiple of 256");
        debug_assert_eq!(w.len, n * k, "Q4_K matmul: weight numel mismatch");
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            m: u32,
            k: u32,
            n: u32,
            _pad: u32,
        }
        let out = self.alloc_f32(m * n, "matmul_q4k.out");
        let params = self.uniform(
            &P {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                _pad: 0,
            },
            "matmul_q4k.params",
        );
        self.dispatch(
            "matmul_nt_q4_k",
            include_str!("shaders/matmul_nt_q4_k.wgsl"),
            &[&x.buf, &w.qb, &out.buf],
            &params,
            (m * n) as u32,
        );
        out
    }

    /// Embedding gather from a Q4_K-resident table: `out[t,:] = dequant(table[ids[t],:])`.
    /// Returns `[n_tokens, dim]` f32.
    pub fn embed_q4_k(
        &self,
        ids: &GpuBuffer,
        table: &GpuQ4KBuffer,
        n_tokens: usize,
        dim: usize,
    ) -> GpuBuffer {
        debug_assert_eq!(
            dim % K_BLOCK,
            0,
            "Q4_K embed: dim must be a multiple of 256"
        );
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n_tokens: u32,
            dim: u32,
            _b: u32,
            _c: u32,
        }
        let out = self.alloc_f32(n_tokens * dim, "embed_q4k.out");
        let params = self.uniform(
            &P {
                n_tokens: n_tokens as u32,
                dim: dim as u32,
                _b: 0,
                _c: 0,
            },
            "embed_q4k.params",
        );
        self.dispatch(
            "embed_q4_k",
            include_str!("shaders/embed_q4_k.wgsl"),
            &[&ids.buf, &table.qb, &out.buf],
            &params,
            n_tokens as u32,
        );
        out
    }

    /// Embedding gather from a Q8_0-resident table: `out[t,:] = dequant(table[ids[t],:])`.
    /// Returns `[n_tokens, dim]` f32 — the quantized twin of `embed`.
    pub fn embed_q8_0(
        &self,
        ids: &GpuBuffer,
        table: &GpuQ8Buffer,
        n_tokens: usize,
        dim: usize,
    ) -> GpuBuffer {
        debug_assert_eq!(dim % BLOCK, 0, "Q8_0 embed: dim must be a multiple of 32");
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n_tokens: u32,
            dim: u32,
            _b: u32,
            _c: u32,
        }
        let out = self.alloc_f32(n_tokens * dim, "embed_q8.out");
        let params = self.uniform(
            &P {
                n_tokens: n_tokens as u32,
                dim: dim as u32,
                _b: 0,
                _c: 0,
            },
            "embed_q8.params",
        );
        self.dispatch(
            "embed_q8_0",
            include_str!("shaders/embed_q8_0.wgsl"),
            &[&ids.buf, &table.qs, &table.scales, &out.buf],
            &params,
            n_tokens as u32,
        );
        out
    }
}
