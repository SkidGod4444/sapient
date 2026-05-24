//! Real transformer forward passes for text generation.

mod common;
mod llama;
mod phi;

use std::path::PathBuf;

use anyhow::{bail, Result};
use sapient_hub::model_info::{ArchType, ModelInfo};

pub use llama::LlamaForward;
pub use phi::PhiForward;

/// Architecture-specific inference engine with KV-cache support.
pub enum ForwardEngine {
    Llama(LlamaForward),
    Phi(PhiForward),
}

impl ForwardEngine {
    pub fn from_pretrained(info: ModelInfo, weight_paths: &[PathBuf]) -> Result<Self> {
        match info.arch {
            ArchType::Llama | ArchType::Qwen | ArchType::Gemma | ArchType::Mixtral => {
                Ok(Self::Llama(LlamaForward::from_files(info, weight_paths)?))
            }
            ArchType::Phi => Ok(Self::Phi(PhiForward::from_files(info, weight_paths)?)),
            other => bail!(
                "architecture {other:?} does not yet have a native forward engine — \
                 use safetensors weights for Llama, Phi, or Qwen models"
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
