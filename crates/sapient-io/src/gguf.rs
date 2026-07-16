// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! GGUF format parser — Phase 4: memory-mapped loading for bigger-than-RAM models.
//!
//! Five guarantees over the original:
//! 1. **Alignment**: `data_start` is rounded to `general.alignment` (default 32).
//! 2. **Nibble order**: Q4_0 uses the ggml split layout (lo nibble → first half,
//!    hi nibble → second half), not the interleaved layout the previous code had.
//! 3. **No F32 expansion for Q4_0/Q8_0**: tensors are stored as packed block bytes
//!    (0.5625/1.0625 B/weight) so RAM ≈ file size.
//! 4. **Metadata parsing**: KV pairs are returned so ModelInfo can be built
//!    directly from the GGUF header instead of requiring a separate config.json.
//! 5. **mmap loading**: `load_tensors_mmap` maps the file into virtual address space;
//!    Q4_0/Q8_0 tensors point directly into the mapped region — zero copy, zero heap.
//!    The OS pages weight blocks in on demand, so peak resident RAM ≈ one active layer.

use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use sapient_core::buffer::{Buffer, BufferHandle};
use sapient_core::error::{Result, SapientError};
use sapient_core::{DType, Shape, Tensor};
use sapient_ir::graph::Graph;

// ── MmapBuffer ────────────────────────────────────────────────────────────────

/// A read-only buffer backed by a memory-mapped file region.
///
/// The operating system pages weight blocks into physical RAM on demand.
/// Only the blocks being actively computed are resident — unused layers are
/// transparently swapped to disk under memory pressure. This lets you run
/// a 4 GB Q4_K_M model on a device with 2 GB RAM.
struct MmapBuffer {
    mmap: Arc<Mmap>,
    offset: usize,
    len: usize,
}

impl fmt::Debug for MmapBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MmapBuffer(offset={}, len={})", self.offset, self.len)
    }
}

// SAFETY: `Mmap` is `Send + Sync`; the buffer is immutable after construction.
unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl Buffer for MmapBuffer {
    fn as_bytes(&self) -> &[u8] {
        &self.mmap[self.offset..self.offset + self.len]
    }

    fn is_mmap(&self) -> bool {
        true
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        panic!("MmapBuffer is read-only — model weights cannot be mutated in-place")
    }

    fn len(&self) -> usize {
        self.len
    }

    fn alignment(&self) -> usize {
        32 // GGUF default alignment
    }

    fn device(&self) -> &str {
        "cpu-mmap"
    }
}

// ── Constants ─────────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
/// Default file-section alignment in bytes (can be overridden by general.alignment KV).
const DEFAULT_ALIGNMENT: u64 = 32;

// ── GgmlType ──────────────────────────────────────────────────────────────────

#[repr(u32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30,
}

const QK_K: usize = 256;

impl GgmlType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K => QK_K,
        }
    }

    fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    /// Map to the corresponding `sapient_core::DType` for types we keep as packed blocks.
    /// All of these are stored zero-copy from the GGUF file and dequantized on-the-fly
    /// inside the matmul kernel — no F32 expansion at load time.
    fn to_sapient_dtype(self) -> Option<DType> {
        match self {
            Self::Q4_0 => Some(DType::Q4_0),
            Self::Q8_0 => Some(DType::Q8_0),
            Self::Q4_K => Some(DType::Q4_K),
            Self::Q5_K => Some(DType::Q5_K),
            Self::Q6_K => Some(DType::Q6_K),
            _ => None,
        }
    }
}

fn tensor_byte_len(kind: GgmlType, numel: usize) -> usize {
    if matches!(kind, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        numel * kind.type_size()
    } else {
        (numel / kind.block_size()) * kind.type_size()
    }
}

// ── GgufValue (metadata KV values) ───────────────────────────────────────────

/// A GGUF metadata value, used to build ModelInfo without a separate config.json.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    ArrayU32(Vec<u32>),
    ArrayStr(Vec<String>),
    ArrayF32(Vec<f32>),
    Other,
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::U64(v) => Some(*v as u32),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            Self::F32(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::U8(v) => Some(*v != 0),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ── Tensor info ───────────────────────────────────────────────────────────────

struct GgufTensorInfo {
    name: String,
    dims: Vec<usize>,
    kind: GgmlType,
    offset: u64,
}

// ── Dequantisation (fallback for non-natively-kept types) ─────────────────────

fn dequantize_q4_0(data: &[u8], numel: usize) -> Vec<f32> {
    // ggml block_q4_0 layout: f16 scale (2 bytes) + 16 nibble bytes (32 values).
    // Byte j encodes element j (low nibble) and element j+16 (high nibble).
    // This is the canonical ggml *split* order, NOT interleaved.
    let mut out = vec![0.0f32; numel];
    let block_size = 32usize;
    let blocks = data.len() / 18;
    for b in 0..blocks {
        let base = b * 18;
        let scale = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        for j in 0..16 {
            let byte = data[base + 2 + j];
            let lo = (byte & 0x0f) as i32 - 8; // element j         (first half)
            let hi = (byte >> 4) as i32 - 8; // element j + 16    (second half)
            let base_out = b * block_size;
            if base_out + j < numel {
                out[base_out + j] = lo as f32 * scale;
            }
            if base_out + j + 16 < numel {
                out[base_out + j + 16] = hi as f32 * scale;
            }
        }
    }
    out
}

/// Quantize F32 → ggml Q8_0 blocks (34 bytes: LE f16 scale + 32 int8). Used to
/// hold a quantized GGUF type SAPIENT can't keep as blocks (e.g. Q5_0 from
/// unsloth "dynamic" quants) at ~1.06 B/weight instead of 4 (F32) — near-lossless
/// since Q8_0's 8 bits ⊇ the source's ≤6. `data.len()` must be a multiple of 32.
fn quantize_to_q8_0(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 32 * 34);
    for block in data.chunks_exact(32) {
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in block {
            out.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    out
}

fn dequantize_q8_0(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let block_size = 32usize;
    let blocks = data.len() / 34;
    for b in 0..blocks {
        let base = b * 34;
        let scale = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        for i in 0..32 {
            let idx = b * block_size + i;
            if idx < numel {
                out[idx] = data[base + 2 + i] as i8 as f32 * scale;
            }
        }
    }
    out
}

fn dequantize_q5_0(data: &[u8], numel: usize) -> Vec<f32> {
    let block_size = 32usize;
    let blocks = numel / block_size;
    let mut out = vec![0.0f32; numel];
    for b in 0..blocks {
        let base = b * 22;
        let scale = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);
        for j in 0..16 {
            let byte = data[base + 6 + j];
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((byte & 0x0F) as u32 | xh_0) as i32 - 16;
            let x1 = ((byte >> 4) as u32 | xh_1) as i32 - 16;
            let idx = b * block_size + j;
            out[idx] = x0 as f32 * scale;
            out[idx + 16] = x1 as f32 * scale;
        }
    }
    out
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequantize_q4_k(data: &[u8], numel: usize) -> Vec<f32> {
    let blocks = numel / QK_K;
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0usize;
    for b in 0..blocks {
        let base = b * 144;
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
        let scales = &data[base + 4..base + 16];
        let qs = &data[base + 16..base + 144];
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for l in 0..32 {
                out[out_idx] = d1 * (qs[q_off + l] & 0xF) as f32 - m1;
                out_idx += 1;
            }
            for l in 0..32 {
                out[out_idx] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
                out_idx += 1;
            }
            q_off += 32;
            is += 2;
        }
    }
    out
}

fn dequantize_q5_k(data: &[u8], numel: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 176;
    let n_blocks = numel / QK_K;
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0usize;
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
        let scales = &data[base + 4..base + 16];
        let qh = &data[base + 16..base + 48];
        let ql = &data[base + 48..base + BLOCK_BYTES];
        let mut ql_off = 0usize;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1v = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2v = dmin * m2 as f32;
            let qh_byte = qh[is / 8];
            for l in 0..32usize {
                let hi = if qh_byte & u1 != 0 { 16.0f32 } else { 0.0 };
                out[out_idx + l] = d1 * ((ql[ql_off + l] & 0x0F) as f32 + hi) - m1v;
                let hi2 = if qh_byte & u2 != 0 { 16.0f32 } else { 0.0 };
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

fn dequantize_q6_k(data: &[u8], numel: usize) -> Vec<f32> {
    const BLOCK_BYTES: usize = 210;
    let n_blocks = numel / QK_K;
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0usize;
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        let sc = &data[base + 192..base + 208];
        let d = f16_to_f32(u16::from_le_bytes([data[base + 208], data[base + 209]]));
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        // 16 i8 scales/super-block; offsets +0/+2/+4/+6 with split at l==16
        // (`is = l/16`), base +8 per 128-block. Matches ggml dequantize_row_q6_K.
        let mut sc_base = 0usize;
        for _ in 0..(QK_K / 128) {
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
    out
}

fn dequantize_to_f32(kind: GgmlType, bytes: &[u8], numel: usize) -> Result<Vec<f32>> {
    let f32_data = match kind {
        GgmlType::F32 => bytes[..numel * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        GgmlType::F16 => bytes[..numel * 2]
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        GgmlType::BF16 => bytes[..numel * 2]
            .chunks_exact(2)
            .map(|c| f32::from(half::bf16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        GgmlType::Q4_0 => dequantize_q4_0(bytes, numel),
        GgmlType::Q5_0 => dequantize_q5_0(bytes, numel),
        GgmlType::Q8_0 => dequantize_q8_0(bytes, numel),
        GgmlType::Q4_K => dequantize_q4_k(bytes, numel),
        GgmlType::Q5_K => dequantize_q5_k(bytes, numel),
        GgmlType::Q6_K => dequantize_q6_k(bytes, numel),
        other => {
            return Err(SapientError::GgufParseError(format!(
                "unsupported GGUF quantization type {other:?}"
            )));
        }
    };
    Ok(f32_data)
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

// ── Core parsing helper ───────────────────────────────────────────────────────

/// Parse a GGUF file header and return:
/// - the KV metadata map
/// - the tensor info list
/// - the `data_start` byte offset (alignment-corrected)
fn parse_header(
    bytes: &[u8],
) -> Result<(
    HashMap<String, GgufValue>,
    Vec<GgufTensorInfo>,
    usize, // data_start
)> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(bytes);

    let magic = read_u32(&mut cursor)?;
    if magic != GGUF_MAGIC {
        return Err(SapientError::GgufParseError("bad GGUF magic".into()));
    }
    let version = read_u32(&mut cursor)?;
    if !(1..=3).contains(&version) {
        return Err(SapientError::GgufParseError(format!(
            "unsupported GGUF version {version} (expected 1–3)"
        )));
    }
    let tensor_count = read_u64(&mut cursor)? as usize;
    let kv_count = read_u64(&mut cursor)? as usize;

    // Parse all KV pairs — needed for general.alignment and model config.
    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let (k, v) = read_kv(&mut cursor)?;
        metadata.insert(k, v);
    }

    // Read tensor infos.
    let mut tensor_infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut cursor)?;
        let n_dims = read_u32(&mut cursor)? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut cursor)? as usize);
        }
        let kind_raw = read_u32(&mut cursor)?;
        let kind = GgmlType::from_u32(kind_raw)
            .ok_or_else(|| SapientError::GgufParseError(format!("unknown ggml type {kind_raw}")))?;
        let offset = read_u64(&mut cursor)?;
        tensor_infos.push(GgufTensorInfo {
            name,
            dims,
            kind,
            offset,
        });
    }

    // Alignment-corrected data start (GGUF spec: the data section begins at
    // the next multiple of `general.alignment` bytes after the header).
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT);
    let raw_pos = cursor.position();
    let data_start = raw_pos.div_ceil(alignment) as usize * alignment as usize;

    Ok((metadata, tensor_infos, data_start))
}

/// Build a `Tensor` from a tensor info entry backed by a memory-mapped region.
/// Q4_0/Q8_0: zero-copy — the tensor points directly into the mmap'd file.
/// K-quants / F16 / BF16: dequantized to F32 (raw bytes come from mmap, no heap copy).
fn make_tensor_mmap(info: &GgufTensorInfo, mmap: &Arc<Mmap>, data_start: usize) -> Result<Tensor> {
    let numel: usize = info.dims.iter().product::<usize>().max(1);
    let byte_len = tensor_byte_len(info.kind, numel);
    let start = data_start + info.offset as usize;

    if start + byte_len > mmap.len() {
        return Err(SapientError::GgufParseError(format!(
            "tensor '{}': data range [{}..{}] exceeds file size {}",
            info.name,
            start,
            start + byte_len,
            mmap.len()
        )));
    }

    let shape = if info.dims.is_empty() {
        Shape::new([1])
    } else {
        Shape::new(info.dims.clone())
    };

    if let Some(dtype) = info.kind.to_sapient_dtype() {
        // Zero-copy: tensor points directly into the mmap'd file region.
        let buf = MmapBuffer {
            mmap: Arc::clone(mmap),
            offset: start,
            len: byte_len,
        };
        let handle = BufferHandle::new(buf);
        Tensor::from_buffer(shape, dtype, handle, 0)
            .map_err(|e| SapientError::GgufParseError(e.to_string()))
    } else {
        // A GGUF type SAPIENT doesn't keep as packed blocks. Two cases:
        //   - quantized (e.g. Q5_0 from unsloth "dynamic" quants) → dequant then
        //     RE-QUANTIZE to Q8_0: near-lossless, ~1.06 B/weight (vs 4 for F32),
        //     matmul-ready. A 63 GB GLM-4.5-Air has 24 Q5_0 ffn_down tensors that
        //     were 70 GB as F32 → ~19 GB as Q8_0.
        //   - F16/BF16/F32 (small norms/biases, or non-32-aligned) → F32.
        let raw = &mmap[start..start + byte_len];
        let f32_data = dequantize_to_f32(info.kind, raw, numel)?;
        if info.kind.block_size() > 1 && numel % 32 == 0 {
            let q8 = quantize_to_q8_0(&f32_data);
            Tensor::from_quant_bytes(&q8, shape, DType::Q8_0)
                .map_err(|e| SapientError::GgufParseError(e.to_string()))
        } else {
            Tensor::from_f32(&f32_data, shape)
                .map_err(|e| SapientError::GgufParseError(e.to_string()))
        }
    }
}

/// Build a `Tensor` from a tensor info entry.  Q4_0 and Q8_0 are kept as packed
/// block bytes (no F32 expansion); all other types are dequantized to F32.
fn make_tensor(info: &GgufTensorInfo, bytes: &[u8], data_start: usize) -> Result<Tensor> {
    let numel: usize = info.dims.iter().product::<usize>().max(1);
    let byte_len = tensor_byte_len(info.kind, numel);
    let start = data_start + info.offset as usize;

    if start + byte_len > bytes.len() {
        return Err(SapientError::GgufParseError(format!(
            "tensor '{}': data range [{}..{}] exceeds file size {}",
            info.name,
            start,
            start + byte_len,
            bytes.len()
        )));
    }

    let raw = &bytes[start..start + byte_len];
    let shape = if info.dims.is_empty() {
        Shape::new([1])
    } else {
        Shape::new(info.dims.clone())
    };

    if let Some(dtype) = info.kind.to_sapient_dtype() {
        // Keep quantized — zero expansion, RAM ≈ file size for these weights.
        Tensor::from_quant_bytes(raw, shape, dtype)
            .map_err(|e| SapientError::GgufParseError(e.to_string()))
    } else {
        // A quantized type SAPIENT doesn't keep as blocks (e.g. Q5_0) → re-quantize
        // to Q8_0 (near-lossless, ~1.06 B/weight); F16/BF16/F32 → F32. Mirrors the
        // mmap path so both loaders keep peak RAM ≈ Q8_0, not F32-expanded.
        let f32_data = dequantize_to_f32(info.kind, raw, numel)?;
        if info.kind.block_size() > 1 && numel % 32 == 0 {
            let q8 = quantize_to_q8_0(&f32_data);
            Tensor::from_quant_bytes(&q8, shape, DType::Q8_0)
                .map_err(|e| SapientError::GgufParseError(e.to_string()))
        } else {
            Tensor::from_f32(&f32_data, shape)
                .map_err(|e| SapientError::GgufParseError(e.to_string()))
        }
    }
}

// ── Public loader ─────────────────────────────────────────────────────────────

pub struct GgufLoader;

impl GgufLoader {
    /// Load a GGUF file into an IR graph (used by the graph-execution path).
    pub fn load(path: &Path) -> Result<Graph> {
        let bytes = std::fs::read(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        let (_, tensor_infos, data_start) = parse_header(&bytes)?;
        let mut graph = Graph::new("gguf_model");
        for info in &tensor_infos {
            let tensor = make_tensor(info, &bytes, data_start)?;
            graph.add_constant(tensor, Some(info.name.clone()));
        }
        Ok(graph)
    }

    /// Parse only the GGUF header KV metadata — no tensor data allocated.
    ///
    /// Use this when you need ModelInfo but haven't decided which loading strategy
    /// (regular heap or mmap) to use yet. Avoids the double-load that occurs when
    /// calling `load_tensors_with_metadata` just to extract metadata and discard weights.
    pub fn parse_metadata_only(path: &Path) -> Result<HashMap<String, GgufValue>> {
        let file = std::fs::File::open(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            SapientError::GgufParseError(format!("mmap failed for header read: {e}"))
        })?;
        let (metadata, _, _) = parse_header(mmap.as_ref())?;
        Ok(metadata)
    }

    /// Load tensors via memory-mapping — zero-copy for Q4_0/Q8_0 weight types.
    ///
    /// The GGUF file is mapped into virtual address space once. Q4_0 and Q8_0
    /// tensors point directly into that region; the OS pages in weight blocks on
    /// demand. Only the blocks being actively computed need to be in physical RAM.
    ///
    /// K-quants (Q4_K, Q5_K, …), F16, and BF16 are still dequantized to F32 on
    /// load — but the raw file bytes come from the mmap, so the raw file is never
    /// heap-allocated. Peak RAM during load = dequantized F32 size, not 2× file.
    pub fn load_tensors_mmap(
        path: &Path,
    ) -> Result<(HashMap<String, GgufValue>, HashMap<String, Tensor>)> {
        let file = std::fs::File::open(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| SapientError::GgufParseError(format!("mmap failed: {e}")))?;
        let mmap = Arc::new(mmap);
        let (metadata, tensor_infos, data_start) = parse_header(mmap.as_ref())?;

        let mut tensors = HashMap::with_capacity(tensor_infos.len());
        for info in &tensor_infos {
            let tensor = make_tensor_mmap(info, &mmap, data_start)?;
            tensors.insert(info.name.clone(), tensor);
        }

        Ok((metadata, tensors))
    }

    /// Load all tensors as `name → Tensor` (for the native forward-pass path).
    ///
    /// Q4_0 and Q8_0 tensors are kept as packed block bytes — RAM ≈ file size.
    /// Other types are dequantized to F32.
    pub fn load_tensors(path: &Path) -> Result<HashMap<String, Tensor>> {
        let (_, tensors) = Self::load_tensors_with_metadata(path)?;
        Ok(tensors)
    }

    /// Load tensors *and* the raw KV metadata from a GGUF file.
    ///
    /// The metadata map is used to build a [`sapient_hub::model_info::ModelInfo`]
    /// without requiring a separate `config.json` file.
    pub fn load_tensors_with_metadata(
        path: &Path,
    ) -> Result<(HashMap<String, GgufValue>, HashMap<String, Tensor>)> {
        let bytes = std::fs::read(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        let (metadata, tensor_infos, data_start) = parse_header(&bytes)?;

        let mut tensors = HashMap::with_capacity(tensor_infos.len());
        for info in &tensor_infos {
            let tensor = make_tensor(info, &bytes, data_start)?;
            tensors.insert(info.name.clone(), tensor);
        }

        Ok((metadata, tensors))
    }

    /// Convenience: load from raw bytes (useful for tests).
    pub fn tensors_from_bytes(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
        let (_, tensor_infos, data_start) = parse_header(bytes)?;
        let mut map = HashMap::with_capacity(tensor_infos.len());
        for info in &tensor_infos {
            let tensor = make_tensor(info, bytes, data_start)?;
            map.insert(info.name.clone(), tensor);
        }
        Ok(map)
    }
}

// ── Low-level readers ─────────────────────────────────────────────────────────

fn read_kv(c: &mut std::io::Cursor<&[u8]>) -> Result<(String, GgufValue)> {
    let key = read_gguf_string(c)?;
    let vtype = read_u32(c)?;
    let value = read_value(c, vtype)?;
    Ok((key, value))
}

fn read_value(c: &mut std::io::Cursor<&[u8]>, vtype: u32) -> Result<GgufValue> {
    Ok(match vtype {
        0 => GgufValue::U8(read_u8(c)?),
        1 => GgufValue::I8(read_u8(c)? as i8),
        2 => GgufValue::U16(read_u16(c)?),
        3 => GgufValue::I16(read_u16(c)? as i16),
        4 => GgufValue::U32(read_u32(c)?),
        5 => GgufValue::I32(read_i32(c)?),
        6 => GgufValue::F32(read_f32(c)?),
        7 => GgufValue::Bool(read_u8(c)? != 0),
        8 => GgufValue::Str(read_gguf_string(c)?),
        9 => {
            let item_type = read_u32(c)?;
            let count = read_u64(c)? as usize;
            match item_type {
                4 => {
                    let v: Vec<u32> = (0..count).map(|_| read_u32(c)).collect::<Result<_>>()?;
                    GgufValue::ArrayU32(v)
                }
                8 => {
                    let v: Vec<String> = (0..count)
                        .map(|_| read_gguf_string(c))
                        .collect::<Result<_>>()?;
                    GgufValue::ArrayStr(v)
                }
                6 => {
                    let v: Vec<f32> = (0..count).map(|_| read_f32(c)).collect::<Result<_>>()?;
                    GgufValue::ArrayF32(v)
                }
                _ => {
                    for _ in 0..count {
                        skip_value(c, item_type)?;
                    }
                    GgufValue::Other
                }
            }
        }
        10 => GgufValue::U64(read_u64(c)?),
        11 => GgufValue::I64(read_i64(c)?),
        12 => GgufValue::F64(read_f64(c)?),
        _ => GgufValue::Other,
    })
}

fn skip_value(c: &mut std::io::Cursor<&[u8]>, vtype: u32) -> Result<()> {
    match vtype {
        0 | 1 | 7 => {
            read_u8(c)?;
        }
        2 | 3 => {
            read_u16(c)?;
        }
        4..=6 => {
            read_u32(c)?;
        }
        8 => {
            read_gguf_string(c)?;
        }
        10..=12 => {
            read_u64(c)?;
        }
        _ => {}
    }
    Ok(())
}

fn read_u8(c: &mut std::io::Cursor<&[u8]>) -> Result<u8> {
    let mut b = [0u8; 1];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(b[0])
}
fn read_u16(c: &mut std::io::Cursor<&[u8]>) -> Result<u16> {
    let mut b = [0u8; 2];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32(c: &mut std::io::Cursor<&[u8]>) -> Result<u32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(c: &mut std::io::Cursor<&[u8]>) -> Result<u64> {
    let mut b = [0u8; 8];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(u64::from_le_bytes(b))
}
fn read_i32(c: &mut std::io::Cursor<&[u8]>) -> Result<i32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(i32::from_le_bytes(b))
}
fn read_i64(c: &mut std::io::Cursor<&[u8]>) -> Result<i64> {
    let mut b = [0u8; 8];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(i64::from_le_bytes(b))
}
fn read_f32(c: &mut std::io::Cursor<&[u8]>) -> Result<f32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(f32::from_le_bytes(b))
}
fn read_f64(c: &mut std::io::Cursor<&[u8]>) -> Result<f64> {
    let mut b = [0u8; 8];
    c.read_exact(&mut b)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(f64::from_le_bytes(b))
}
fn read_gguf_string(c: &mut std::io::Cursor<&[u8]>) -> Result<String> {
    let len = read_u64(c)? as usize;
    let mut buf = vec![0u8; len];
    c.read_exact(&mut buf)
        .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    String::from_utf8(buf).map_err(|e| SapientError::GgufParseError(e.to_string()))
}

#[cfg(test)]
mod q8_quant_tests {
    use super::*;

    #[test]
    fn q8_0_quantize_roundtrips_and_sizes() {
        // Two 32-element blocks; distinct magnitudes per block.
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q = quantize_to_q8_0(&data);
        assert_eq!(q.len(), 64 / 32 * 34, "Q8_0 = 34 bytes / 32 weights");
        let back = dequantize_q8_0(&q, 64);
        assert_eq!(back.len(), 64);
        // Q8_0 error is bounded by ~amax/254 per block (amax ≈ 3.2 → ~0.013).
        for (a, b) in data.iter().zip(&back) {
            assert!((a - b).abs() < 0.03, "roundtrip a={a} b={b}");
        }
    }

    #[test]
    fn q8_0_quantize_handles_all_zeros() {
        let q = quantize_to_q8_0(&[0.0f32; 32]);
        assert_eq!(q.len(), 34);
        assert!(dequantize_q8_0(&q, 32).iter().all(|&v| v == 0.0));
    }
}
