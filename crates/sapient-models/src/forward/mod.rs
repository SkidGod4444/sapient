//! Real transformer forward passes for text generation.

pub mod backend;
pub mod common;
mod llama;
#[cfg(all(target_os = "macos", feature = "mlx"))]
mod mlx_engine;
mod phi;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_hub::resolver::WeightFormat;

use crate::gguf_weights::{load_gguf_hf_weights, load_gguf_hf_weights_mmap};

pub use backend::{mac_gpu_support, total_system_ram_bytes, LlmBackendKind, MacGpuSupport};
pub use llama::LlamaForward;
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub use mlx_engine::MlxForwardEngine;
pub use phi::PhiForward;

/// Architecture-specific inference engine with KV-cache support.
pub enum ForwardEngine {
    Llama(LlamaForward),
    Phi(PhiForward),
    /// Fully MLX-native Llama-family engine: all activations stay as GPU arrays
    /// throughout the forward pass, one eval() per decode step.
    /// Enabled when `--backend metal` (or `auto` on Apple Silicon) for Llama/Qwen/Mistral.
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    MlxLlama(MlxForwardEngine),
}

fn weight_format_from_paths(weight_paths: &[PathBuf]) -> WeightFormat {
    match weight_paths
        .first()
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
    {
        Some("gguf") => WeightFormat::Gguf,
        Some("safetensors") => WeightFormat::Safetensors,
        Some("bin") => WeightFormat::PyTorchBin,
        _ => WeightFormat::Unknown,
    }
}

/// Returns true when the Metal backend is requested/auto-selected on Apple Silicon.
fn use_mlx_engine(backend: LlmBackendKind) -> bool {
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        use backend::MetalLlmBackend;
        matches!(backend, LlmBackendKind::Metal | LlmBackendKind::Auto)
            && MetalLlmBackend::is_available()
    }
    #[cfg(not(all(target_os = "macos", feature = "mlx")))]
    {
        let _ = backend;
        false
    }
}

impl ForwardEngine {
    pub fn from_pretrained(info: ModelInfo, weight_paths: &[PathBuf]) -> Result<Self> {
        Self::from_weight_paths(info, weight_paths)
    }

    pub fn from_weight_paths(info: ModelInfo, weight_paths: &[PathBuf]) -> Result<Self> {
        Self::from_weight_paths_with_backend(info, weight_paths, LlmBackendKind::Auto)
    }

    pub fn from_weight_paths_with_backend(
        info: ModelInfo,
        weight_paths: &[PathBuf],
        backend: LlmBackendKind,
    ) -> Result<Self> {
        match weight_format_from_paths(weight_paths) {
            WeightFormat::Gguf => {
                let path = weight_paths
                    .first()
                    .context("GGUF model has no weight path")?;
                Self::from_gguf_with_backend(info, path, backend)
            }
            WeightFormat::Safetensors | WeightFormat::PyTorchBin => match info.arch {
                ArchType::Llama | ArchType::Qwen | ArchType::Gemma | ArchType::Mixtral => {
                    Ok(Self::Llama(LlamaForward::from_files_with_backend(
                        info,
                        weight_paths,
                        backend,
                    )?))
                }
                ArchType::Phi => Ok(Self::Phi(PhiForward::from_files_with_backend(
                    info,
                    weight_paths,
                    backend,
                )?)),
                other => bail!(
                    "architecture {other:?} does not yet have a native forward engine — \
                     use safetensors weights for Llama, Phi, or Qwen models"
                ),
            },
            WeightFormat::Unknown => bail!("unknown or missing weight file format"),
        }
    }

    pub fn from_gguf(info: ModelInfo, path: &Path) -> Result<Self> {
        Self::from_gguf_with_backend(info, path, LlmBackendKind::Auto)
    }

    pub fn from_gguf_with_backend(
        info: ModelInfo,
        path: &Path,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        let weights = load_gguf_hf_weights(path)?;
        Self::from_gguf_weights(info, weights, backend)
    }

    /// Load via memory-mapping — Q4_0/Q8_0 tensors are zero-copy from disk.
    pub fn from_gguf_mmap_with_backend(
        info: ModelInfo,
        path: &Path,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        let weights = load_gguf_hf_weights_mmap(path)?;
        Self::from_gguf_weights(info, weights, backend)
    }

    fn from_gguf_weights(
        info: ModelInfo,
        mut weights: std::collections::HashMap<String, sapient_core::Tensor>,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        // llama.cpp permutes q_proj/k_proj rows for ggml's NORM-style RoPE (the
        // `llama` architecture: Llama, Mistral, SmolLM, TinyLlama). SAPIENT uses
        // HF/NEOX-style RoPE, so we invert that permutation here — otherwise RoPE
        // scrambles positions across heads and the model emits token-salad.
        // Qwen2/Gemma GGUFs use NEOX RoPE (not permuted) and must be left as-is.
        if matches!(info.arch, ArchType::Llama) {
            crate::gguf_weights::unpermute_llama_gguf_qk(
                &mut weights,
                info.num_attention_heads,
                info.num_key_value_heads,
                info.head_dim,
            )?;
        }
        match info.arch {
            ArchType::Llama | ArchType::Qwen | ArchType::Gemma | ArchType::Mixtral => {
                // Use the fully-native MLX engine when Metal is available and selected.
                if use_mlx_engine(backend) {
                    #[cfg(all(target_os = "macos", feature = "mlx"))]
                    {
                        tracing::info!(
                            "using MlxForwardEngine (lazy-graph, no CPU↔GPU round-trips)"
                        );
                        return Ok(Self::MlxLlama(MlxForwardEngine::from_weights(
                            info, weights,
                        )?));
                    }
                }
                Ok(Self::Llama(LlamaForward::from_weights_with_backend(
                    info, weights, backend,
                )?))
            }
            ArchType::Phi => {
                bail!("GGUF Phi models are not yet supported — use safetensors weights")
            }
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
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => f.reset_cache(),
        }
    }

    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        match self {
            Self::Llama(f) => f.forward_logits(input_ids, use_cache),
            Self::Phi(f) => f.forward_logits(input_ids, use_cache),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => f.forward_logits(input_ids, use_cache),
        }
    }

    pub fn forward_all_logits(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Llama(f) => f.forward_all_logits(input_ids),
            Self::Phi(f) => f.forward_all_logits(input_ids),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => f.forward_all_logits(input_ids),
        }
    }

    pub fn embed(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Llama(f) => f.embed(input_ids),
            Self::Phi(f) => f.embed(input_ids),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(_) => {
                let _ = input_ids;
                bail!("embed() not yet implemented for MlxForwardEngine")
            }
        }
    }

    /// True when layers are split between Metal GPU and CPU (hybrid mode).
    pub fn is_hybrid(&self) -> bool {
        match self {
            Self::Llama(f) => f.is_hybrid(),
            Self::Phi(f) => f.is_hybrid(),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(_) => false,
        }
    }

    /// Human-readable backend label.
    pub fn backend_label(&self) -> String {
        match self {
            Self::Llama(f) => f.backend_label(),
            Self::Phi(f) => f.backend_label(),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(_) => "metal (MLX native graph)".to_string(),
        }
    }
}
