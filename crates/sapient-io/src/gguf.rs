//! GGUF format parser and dequantization.
//!
//! GGUF (GPT-Generated Unified Format) is the binary format used by llama.cpp.
//! Phase 1: parse metadata + load quantized weights, dequantize to F32.
//! Phase 2 (future): native quantized kernel dispatch.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use sapient_core::{DType, Shape, Tensor};
use sapient_core::error::{Result, SapientError};
use sapient_ir::graph::Graph;

// ── GGUF Constants ────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;

// GGMLType enum (quantized types we support).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgmlType {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    BF16 = 30,
}

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
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    fn block_size(self) -> usize {
        match self {
            Self::F32  | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => 32,
            Self::Q4_1 => 32,
            Self::Q5_0 => 32,
            Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q8_1 => 32,
        }
    }

    fn type_size(self) -> usize {
        match self {
            Self::F32  => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,  // sizeof(block_q4_0): 2 (scale) + 16 (nibbles)
            Self::Q4_1 => 20,  // 2+2+16
            Self::Q5_0 => 22,  // 2+4+16
            Self::Q5_1 => 24,  // 2+2+4+16
            Self::Q8_0 => 34,  // 2+32
            Self::Q8_1 => 36,  // 2+2+32
        }
    }
}

// ── Tensor info ───────────────────────────────────────────────────────────────

struct GgufTensorInfo {
    name:   String,
    dims:   Vec<usize>,
    kind:   GgmlType,
    offset: u64,
}

// ── Dequantisation ────────────────────────────────────────────────────────────

fn dequantize_q4_0(data: &[u8], numel: usize) -> Vec<f32> {
    // block_q4_0: f16 scale (2 bytes) + 16 bytes of packed nibbles (32 values)
    let mut out = vec![0.0f32; numel];
    let block_size = 32usize;
    let blocks = data.len() / 18;
    for b in 0..blocks {
        let base = b * 18;
        let scale = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        for i in 0..16 {
            let byte = data[base + 2 + i];
            let lo = (byte & 0x0f) as i8 - 8;
            let hi = ((byte >> 4) & 0x0f) as i8 - 8;
            let idx = b * block_size + i * 2;
            if idx < numel     { out[idx]     = lo as f32 * scale; }
            if idx + 1 < numel { out[idx + 1] = hi as f32 * scale; }
        }
    }
    out
}

fn dequantize_q8_0(data: &[u8], numel: usize) -> Vec<f32> {
    // block_q8_0: f16 scale (2 bytes) + 32 i8 values
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

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

// ── GGUF Reader ───────────────────────────────────────────────────────────────

pub struct GgufLoader;

impl GgufLoader {
    pub fn load(path: &Path) -> Result<Graph> {
        let bytes = fs::read(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Graph> {
        use std::io::Cursor;
        let mut cursor = Cursor::new(bytes);

        // Header.
        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(SapientError::GgufParseError("bad magic".into()));
        }
        let version = read_u32(&mut cursor)?;
        let tensor_count = read_u64(&mut cursor)? as usize;
        let kv_count     = read_u64(&mut cursor)? as usize;

        // Skip key-value metadata.
        for _ in 0..kv_count {
            skip_kv(&mut cursor, bytes)?;
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
            tensor_infos.push(GgufTensorInfo { name, dims, kind, offset });
        }

        // Alignment is at the end of the header section.
        let data_start = cursor.position() as usize;

        // Build graph: every tensor becomes a Constant node.
        let mut graph = Graph::new("gguf_model");

        for info in &tensor_infos {
            let numel: usize = info.dims.iter().product::<usize>().max(1);
            let start = data_start + info.offset as usize;

            let f32_data: Vec<f32> = match info.kind {
                GgmlType::F32 => {
                    bytes[start..start + numel * 4]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                }
                GgmlType::F16 => {
                    bytes[start..start + numel * 2]
                        .chunks_exact(2)
                        .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                        .collect()
                }
                GgmlType::Q4_0 => {
                    let blocks = numel / 32;
                    dequantize_q4_0(&bytes[start..start + blocks * 18], numel)
                }
                GgmlType::Q8_0 => {
                    let blocks = numel / 32;
                    dequantize_q8_0(&bytes[start..start + blocks * 34], numel)
                }
                _ => vec![0.0f32; numel], // unsupported → zero (future work)
            };

            let shape = if info.dims.is_empty() {
                Shape::new([1])
            } else {
                Shape::new(info.dims.clone())
            };

            let tensor = Tensor::from_f32(&f32_data, shape)
                .map_err(|e| SapientError::GgufParseError(e.to_string()))?;

            graph.add_constant(tensor, Some(info.name.clone()));
        }

        Ok(graph)
    }

    /// Load all tensors from a GGUF file as a raw name → Tensor map.
    /// Useful for weight loading when you have a separately-built graph.
    pub fn load_tensors(path: &Path) -> Result<HashMap<String, Tensor>> {
        let bytes = fs::read(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        Self::tensors_from_bytes(&bytes)
    }

    pub fn tensors_from_bytes(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
        use std::io::Cursor;
        let mut cursor = Cursor::new(bytes);
        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(SapientError::GgufParseError("bad magic".into()));
        }
        let _version = read_u32(&mut cursor)?;
        let tensor_count = read_u64(&mut cursor)? as usize;
        let kv_count     = read_u64(&mut cursor)? as usize;
        for _ in 0..kv_count { skip_kv(&mut cursor, bytes)?; }

        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims { dims.push(read_u64(&mut cursor)? as usize); }
            let kind_raw = read_u32(&mut cursor)?;
            let kind = GgmlType::from_u32(kind_raw)
                .ok_or_else(|| SapientError::GgufParseError(format!("unknown ggml type {kind_raw}")))?;
            let offset = read_u64(&mut cursor)?;
            tensor_infos.push(GgufTensorInfo { name, dims, kind, offset });
        }
        let data_start = cursor.position() as usize;

        let mut map = HashMap::with_capacity(tensor_infos.len());
        for info in &tensor_infos {
            let numel: usize = info.dims.iter().product::<usize>().max(1);
            let start = data_start + info.offset as usize;
            let f32_data: Vec<f32> = match info.kind {
                GgmlType::F32 => bytes[start..start + numel * 4].chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                GgmlType::F16 => bytes[start..start + numel * 2].chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
                GgmlType::Q4_0 => dequantize_q4_0(&bytes[start..start + (numel/32) * 18], numel),
                GgmlType::Q8_0 => dequantize_q8_0(&bytes[start..start + (numel/32) * 34], numel),
                _ => vec![0.0f32; numel],
            };
            let shape = if info.dims.is_empty() { Shape::new([1]) } else { Shape::new(info.dims.clone()) };
            let tensor = Tensor::from_f32(&f32_data, shape)
                .map_err(|e| SapientError::GgufParseError(e.to_string()))?;
            map.insert(info.name.clone(), tensor);
        }
        Ok(map)
    }
}

// ── Low-level read helpers ────────────────────────────────────────────────────

fn read_u32(c: &mut std::io::Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    c.read_exact(&mut buf).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(c: &mut std::io::Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    c.read_exact(&mut buf).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string(c: &mut std::io::Cursor<&[u8]>) -> Result<String> {
    let len = read_u64(c)? as usize;
    let mut buf = vec![0u8; len];
    c.read_exact(&mut buf).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    String::from_utf8(buf).map_err(|e| SapientError::GgufParseError(e.to_string()))
}

fn skip_kv(c: &mut std::io::Cursor<&[u8]>, _bytes: &[u8]) -> Result<()> {
    // Skip key.
    let _key = read_gguf_string(c)?;
    // Read value type.
    let vtype = read_u32(c)?;
    // Skip value based on type.
    match vtype {
        0 => { let _ = read_u8(c)?; }   // UINT8
        1 => { let _ = read_i8(c)?; }   // INT8
        2 => { let _ = read_u16(c)?; }  // UINT16
        3 => { let _ = read_i16(c)?; }  // INT16
        4 => { let _ = read_u32(c)?; }  // UINT32
        5 => { let _ = read_i32(c)?; }  // INT32
        6 => { let _ = read_f32(c)?; }  // FLOAT32
        7 => { let _ = read_u8(c)?; }   // BOOL (1 byte)
        8 => { let _ = read_gguf_string(c)?; } // STRING
        9 => {                           // ARRAY
            let item_type = read_u32(c)?;
            let count = read_u64(c)? as usize;
            for _ in 0..count {
                skip_value(c, item_type)?;
            }
        }
        10 => { let _ = read_u64(c)?; } // UINT64
        11 => { let _ = read_i64(c)?; } // INT64
        12 => { let _ = read_f64(c)?; } // FLOAT64
        _ => {}
    }
    Ok(())
}

fn skip_value(c: &mut std::io::Cursor<&[u8]>, vtype: u32) -> Result<()> {
    match vtype {
        0 | 1 | 7 => { read_u8(c)?; }
        2 | 3      => { read_u16(c)?; }
        4 | 5 | 6  => { read_u32(c)?; }
        8          => { read_gguf_string(c)?; }
        10 | 11 | 12 => { read_u64(c)?; }
        _ => {}
    }
    Ok(())
}

fn read_u8(c: &mut std::io::Cursor<&[u8]>) -> Result<u8> {
    let mut b = [0u8; 1];
    c.read_exact(&mut b).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(b[0])
}
fn read_i8(c: &mut std::io::Cursor<&[u8]>) -> Result<i8> { Ok(read_u8(c)? as i8) }
fn read_u16(c: &mut std::io::Cursor<&[u8]>) -> Result<u16> {
    let mut b = [0u8; 2];
    c.read_exact(&mut b).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(u16::from_le_bytes(b))
}
fn read_i16(c: &mut std::io::Cursor<&[u8]>) -> Result<i16> { Ok(read_u16(c)? as i16) }
fn read_i32(c: &mut std::io::Cursor<&[u8]>) -> Result<i32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(i32::from_le_bytes(b))
}
fn read_f32(c: &mut std::io::Cursor<&[u8]>) -> Result<f32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(f32::from_le_bytes(b))
}
fn read_i64(c: &mut std::io::Cursor<&[u8]>) -> Result<i64> {
    let mut b = [0u8; 8];
    c.read_exact(&mut b).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(i64::from_le_bytes(b))
}
fn read_f64(c: &mut std::io::Cursor<&[u8]>) -> Result<f64> {
    let mut b = [0u8; 8];
    c.read_exact(&mut b).map_err(|e| SapientError::GgufParseError(e.to_string()))?;
    Ok(f64::from_le_bytes(b))
}
