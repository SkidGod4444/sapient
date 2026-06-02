//! Resolved file paths after a Hub download.

use std::path::PathBuf;

/// All local file paths for a downloaded model.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// HuggingFace model ID (e.g. `"meta-llama/Llama-3.2-1B-Instruct"`).
    pub model_id: String,

    /// Local path to `config.json`.
    pub config_path: PathBuf,

    /// Local path to `tokenizer.json` (may not exist for GGUF-only repos).
    pub tokenizer_path: Option<PathBuf>,

    /// Local path to `tokenizer_config.json` (chat template lives here).
    pub tokenizer_config_path: Option<PathBuf>,

    /// Local path to `generation_config.json` (Whisper suppress-token lists,
    /// forced decoder ids). Absent for most repos.
    pub generation_config_path: Option<PathBuf>,

    /// Local paths to weight shards (GGUF, safetensors, or .bin).
    pub weight_paths: Vec<PathBuf>,
}

impl ModelFiles {
    /// Returns the weight format detected from the file extension.
    pub fn format(&self) -> WeightFormat {
        match self.weight_paths.first() {
            None => WeightFormat::Unknown,
            Some(p) => match p.extension().and_then(|e| e.to_str()) {
                Some("gguf") => WeightFormat::Gguf,
                Some("safetensors") => WeightFormat::Safetensors,
                Some("bin") => WeightFormat::PyTorchBin,
                _ => WeightFormat::Unknown,
            },
        }
    }
}

/// Weight file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Gguf,
    Safetensors,
    PyTorchBin,
    Unknown,
}
