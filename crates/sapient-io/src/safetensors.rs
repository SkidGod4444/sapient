//! Safetensors format loader.
//!
//! The Safetensors format stores a JSON header describing tensor shapes/dtypes
//! followed by raw tensor data.  We use `memmap2` for zero-copy reads.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;

use sapient_core::{DType, Shape, Tensor};
use sapient_core::error::{Result, SapientError};
use sapient_ir::graph::Graph;

// ── Safetensors header structs ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct StMeta {
    dtype:        String,
    shape:        Vec<usize>,
    data_offsets: [usize; 2],
}

type StHeader = HashMap<String, StMeta>;

// ── Loader ────────────────────────────────────────────────────────────────────

pub struct SafetensorsLoader;

impl SafetensorsLoader {
    pub fn load(path: &Path) -> Result<HashMap<String, Tensor>> {
        let file = std::fs::File::open(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        // SAFETY: we don't mutate the mmap and hold it for the duration.
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| SapientError::SafetensorsParseError(e.to_string()))?
        };
        Self::from_bytes(&mmap)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
        if bytes.len() < 8 {
            return Err(SapientError::SafetensorsParseError("file too short".into()));
        }

        // First 8 bytes: LE u64 header length.
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;

        if header_end > bytes.len() {
            return Err(SapientError::SafetensorsParseError("header overflows file".into()));
        }

        let header_json = std::str::from_utf8(&bytes[8..header_end])
            .map_err(|e| SapientError::SafetensorsParseError(e.to_string()))?;

        let header: StHeader = serde_json::from_str(header_json)
            .map_err(|e| SapientError::SafetensorsParseError(e.to_string()))?;

        let data_section = &bytes[header_end..];
        let mut tensors = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" { continue; }

            let dtype = match meta.dtype.as_str() {
                "F32"  => DType::F32,
                "F16"  => DType::F16,
                "BF16" => DType::BF16,
                "I32"  => DType::I32,
                "I64"  => DType::I64,
                "U8"   => DType::U8,
                "BOOL" => DType::Bool,
                other  => return Err(SapientError::SafetensorsParseError(
                    format!("unknown dtype '{other}'")
                )),
            };

            let [start, end] = meta.data_offsets;
            if end > data_section.len() {
                return Err(SapientError::SafetensorsParseError(
                    format!("tensor '{name}' data out of bounds")
                ));
            }

            let raw = &data_section[start..end];
            let shape = Shape::new(meta.shape.clone());

            // For F32 we can directly build a tensor; others → convert or store raw.
            let tensor = if dtype == DType::F32 {
                let f32s: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Tensor::from_f32(&f32s, shape)
                    .map_err(|e| SapientError::SafetensorsParseError(e.to_string()))?
            } else {
                // For other dtypes, zero-fill (future: native dtype buffers).
                Tensor::zeros(shape, dtype)
                    .map_err(|e| SapientError::SafetensorsParseError(e.to_string()))?
            };

            tensors.insert(name.clone(), tensor);
        }

        Ok(tensors)
    }

    /// Load tensors and insert them as constants in a new graph.
    pub fn load_as_graph(path: &Path) -> Result<Graph> {
        let tensors = Self::load(path)?;
        let mut graph = Graph::new("safetensors_model");
        for (name, tensor) in tensors {
            graph.add_constant(tensor, Some(name));
        }
        Ok(graph)
    }

    /// Alias for `load` — returns a name → Tensor map.
    pub fn load_tensors(path: &Path) -> Result<HashMap<String, Tensor>> {
        Self::load(path)
    }
}
