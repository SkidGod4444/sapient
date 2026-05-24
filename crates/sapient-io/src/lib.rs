//! SAPIENT model format loaders.

pub mod gguf;
pub mod onnx;
pub mod safetensors;

pub use onnx::OnnxLoader;
pub use gguf::GgufLoader;
pub use safetensors::SafetensorsLoader;

use std::path::Path;
use sapient_ir::graph::Graph;
use sapient_core::error::{Result, SapientError};

/// Auto-detect format by file extension and load the graph.
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
