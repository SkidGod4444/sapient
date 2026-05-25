//! Real transformer forward passes for text generation.

mod common;
mod llama;
mod phi;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_hub::resolver::WeightFormat;

use crate::gguf_weights::load_gguf_hf_weights;

pub use llama::LlamaForward;
pub use phi::PhiForward;

/// Architecture-specific inference engine with KV-cache support.
pub enum ForwardEngine {
    Llama(LlamaForward),
    Phi(PhiForward),
}

fn weight_format_from_paths(weight_paths: &[PathBuf]) -> WeightFormat {
    match weight_paths.first().and_then(|p| p.extension()).and_then(|e| e.to_str()) {
        Some("gguf") => WeightFormat::Gguf,
        Some("safetensors") => WeightFormat::Safetensors,
        Some("bin") => WeightFormat::PyTorchBin,
        _ => WeightFormat::Unknown,
    }
}

impl ForwardEngine {
    pub fn from_pretrained(info: ModelInfo, weight_paths: &[PathBuf]) -> Result<Self> {
        Self::from_weight_paths(info, weight_paths)
    }

    pub fn from_weight_paths(info: ModelInfo, weight_paths: &[PathBuf]) -> Result<Self> {
        match weight_format_from_paths(weight_paths) {
            WeightFormat::Gguf => {
                let path = weight_paths
                    .first()
                    .context("GGUF model has no weight path")?;
                Self::from_gguf(info, path)
            }
            WeightFormat::Safetensors | WeightFormat::PyTorchBin => match info.arch {
                ArchType::Llama | ArchType::Qwen | ArchType::Gemma | ArchType::Mixtral => {
                    Ok(Self::Llama(LlamaForward::from_files(info, weight_paths)?))
                }
                ArchType::Phi => Ok(Self::Phi(PhiForward::from_files(info, weight_paths)?)),
                other => bail!(
                    "architecture {other:?} does not yet have a native forward engine — \
                     use safetensors weights for Llama, Phi, or Qwen models"
                ),
            },
            WeightFormat::Unknown => bail!("unknown or missing weight file format"),
        }
    }

    pub fn from_gguf(info: ModelInfo, path: &Path) -> Result<Self> {
        let weights = load_gguf_hf_weights(path)?;
        match info.arch {
            ArchType::Llama | ArchType::Qwen | ArchType::Gemma | ArchType::Mixtral => {
                Ok(Self::Llama(LlamaForward::from_weights(info, weights)?))
            }
            ArchType::Phi => bail!(
                "GGUF Phi models are not yet supported — use safetensors weights"
            ),
            other => bail!(
                "architecture {other:?} does not yet support GGUF loading — \
                 try a Llama-family GGUF model or use safetensors weights"
            ),
        }
    }

    pub fn reset_cache(&mut self) {
        match self {
            Self::Llama(f) => f.reset_cache(),
            Self::Phi(f) => f.reset_cache(),
        }
    }

    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        match self {
            Self::Llama(f) => f.forward_logits(input_ids, use_cache),
            Self::Phi(f) => f.forward_logits(input_ids, use_cache),
        }
    }

    pub fn embed(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Llama(f) => f.embed(input_ids),
            Self::Phi(f) => f.embed(input_ids),
        }
    }
}
