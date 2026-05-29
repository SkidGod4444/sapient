//! SAPIENT model format loaders — ONNX, GGUF (quantized LLMs), Safetensors.

pub mod gguf;
pub mod onnx;
pub mod safetensors;

pub use gguf::{GgufLoader, GgufValue};
pub use onnx::OnnxLoader;
pub use safetensors::SafetensorsLoader;

use sapient_core::error::{Result, SapientError};
use sapient_core::Tensor;
use sapient_ir::graph::Graph;
use std::collections::HashMap;
use std::path::Path;

/// Auto-detect format by file extension and load the graph IR.
pub fn load_graph(path: &Path) -> Result<Graph> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "onnx" => OnnxLoader::load(path),
        "gguf" => GgufLoader::load(path),
        _ => Err(SapientError::UnsupportedFormat(ext)),
    }
}

/// Load all weight tensors from a GGUF file (name → Tensor map).
pub fn load_gguf(path: &Path) -> Result<HashMap<String, Tensor>> {
    GgufLoader::load_tensors(path)
}

/// Load all weight tensors from a Safetensors file (name → Tensor map).
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Tensor>> {
    SafetensorsLoader::load_tensors(path)
}
