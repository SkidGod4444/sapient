//! Real transformer forward passes for text generation.

pub mod backend;
pub mod common;
mod conv;
mod gemma3;
/// Kokoro-82M TTS (StyleTTS2 + ISTFTNet) — non-autoregressive, real-time `speak`.
pub mod kokoro;
mod llama;
#[cfg(all(target_os = "macos", feature = "mlx"))]
mod mlx_engine;
mod phi;
mod siglip;
/// SNAC codec-decoder (Phase 6d, LM-codec TTS) — drives `sapient speak`.
mod snac;
#[cfg(feature = "wgpu")]
mod wgpu_engine;
mod whisper;
#[cfg(feature = "wgpu")]
mod whisper_wgpu;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_hub::resolver::WeightFormat;

use crate::gguf_weights::{load_gguf_hf_weights, load_gguf_hf_weights_mmap};

pub use backend::{mac_gpu_support, total_system_ram_bytes, LlmBackendKind, MacGpuSupport};
pub use gemma3::Gemma3Forward;
pub use kokoro::{DecoderStreamInputs, KokoroConfig, KokoroModel, KOKORO_SAMPLE_RATE};
pub use llama::LlamaForward;
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub use mlx_engine::MlxForwardEngine;
pub use phi::PhiForward;
pub use siglip::{SiglipConfig, SiglipVision};
pub use snac::{normalize_snac_weights, orpheus_codes_to_snac, SnacDecoder};
#[cfg(feature = "wgpu")]
pub use wgpu_engine::WgpuForwardEngine;
pub use whisper::{AudioEngine, WhisperForward};
#[cfg(feature = "wgpu")]
pub use whisper_wgpu::WhisperWgpuEngine;

/// Architecture-specific inference engine with KV-cache support.
pub enum ForwardEngine {
    Llama(LlamaForward),
    Phi(PhiForward),
    /// Gemma3 text engine (gemma-3-*b, MedGemma text) — QK-norm, sandwich
    /// norms, alternating sliding/global attention. CPU-only v1.
    Gemma3(Gemma3Forward),
    /// Fully MLX-native Llama-family engine: all activations stay as GPU arrays
    /// throughout the forward pass, one eval() per decode step.
    /// Enabled when `--backend metal` (or `auto` on Apple Silicon) for Llama/Qwen/Mistral.
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    MlxLlama(MlxForwardEngine),
    /// Cross-platform GPU engine (wgpu/WGSL — Vulkan/DX12/Metal). Selected by
    /// `--backend wgpu` for Llama-family models; weights are GPU-resident.
    /// Boxed: the engine struct is large relative to the other variants.
    #[cfg(feature = "wgpu")]
    Wgpu(Box<WgpuForwardEngine>),
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

/// Returns true when the cross-platform wgpu GPU backend should be used:
/// explicit `--backend wgpu`, **or** `Auto` on a binary compiled with the `wgpu`
/// feature (the `-gpu` release variant) — so a GPU binary runs on the GPU by
/// default without an explicit flag. MLX (Metal) is preferred first on Apple
/// Silicon, so this is only reached when `use_mlx_engine` is false.
fn use_wgpu_engine(backend: LlmBackendKind) -> bool {
    #[cfg(feature = "wgpu")]
    {
        match backend {
            // An explicit request always routes to wgpu (and the engine build
            // then errors clearly if no adapter exists — an explicit request
            // must not silently degrade).
            LlmBackendKind::Wgpu => true,
            // Auto routes to the GPU only when an adapter actually exists, so
            // a wgpu-featured build on a device with a missing/broken driver
            // (or an emulator without Vulkan) loads on the CPU instead of
            // failing — the same probe-first rule as `whisper_wants_wgpu`.
            // Matters on mobile: the packaged iOS/Android libraries compile
            // this feature in unconditionally.
            LlmBackendKind::Auto => sapient_backends_wgpu::WgpuContext::adapter_available(),
            _ => false,
        }
    }
    #[cfg(not(feature = "wgpu"))]
    {
        // Without the feature, only an *explicit* request reaches build_wgpu (which
        // then errors clearly); Auto falls through to CPU.
        matches!(backend, LlmBackendKind::Wgpu)
    }
}

/// Should the Whisper audio engine run on the wgpu GPU path?
///
/// Explicit `--backend wgpu` is always honored (the engine build then errors
/// clearly if no adapter exists — an explicit request must not silently
/// degrade). `Auto` selects wgpu only when the feature is compiled in, MLX
/// (Metal) does not take precedence on Apple Silicon, and a GPU adapter is
/// actually present — probed up front because `WhisperWgpuEngine::from_weights`
/// consumes the weight map, so there is no falling back to CPU after the fact.
pub fn whisper_wants_wgpu(backend: LlmBackendKind) -> bool {
    if matches!(backend, LlmBackendKind::Wgpu) {
        return true;
    }
    #[cfg(feature = "wgpu")]
    {
        matches!(backend, LlmBackendKind::Auto)
            && !use_mlx_engine(backend)
            && sapient_backends_wgpu::WgpuContext::adapter_available()
    }
    #[cfg(not(feature = "wgpu"))]
    {
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
                ArchType::Llama
                | ArchType::Qwen
                | ArchType::Gemma
                | ArchType::Mixtral
                | ArchType::Glm4Moe => {
                    // MoE (Mixtral etc.) runs only on the CPU engine so far, and is
                    // detected by config, not ArchType (a Mixtral safetensors repo
                    // is ArchType::Mixtral, but a Qwen-MoE one is ArchType::Qwen).
                    if info.is_moe() {
                        let weights = crate::weights::load_hf_weights(weight_paths)?;
                        return Self::build_moe_cpu(info, weights, backend);
                    }
                    // wgpu when explicitly requested or auto-selected on a -gpu build
                    // (MLX/Metal is handled inside LlamaForward's backend dispatch).
                    if use_wgpu_engine(backend) && !use_mlx_engine(backend) {
                        return Self::build_wgpu(info, weight_paths);
                    }
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
                ArchType::Gemma3 => {
                    // CPU-only v1 (safetensors; incl. multimodal checkpoints —
                    // the engine strips the language_model. prefix and skips
                    // vision weights). Explicit GPU backends are not wired yet.
                    let mut weights = std::collections::HashMap::new();
                    for path in weight_paths {
                        weights.extend(sapient_io::safetensors::SafetensorsLoader::load(path)?);
                    }
                    Ok(Self::Gemma3(Gemma3Forward::from_weights(info, weights)?))
                }
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
            ArchType::Llama
            | ArchType::Qwen
            | ArchType::Gemma
            | ArchType::Mixtral
            | ArchType::Glm4Moe => {
                // MoE (detected by config, not ArchType — a Mixtral GGUF reports
                // `general.architecture = "llama"`). Stacked expert tensors are
                // split into per-expert 2-D weights, then the CPU engine runs it.
                if info.is_moe() {
                    crate::gguf_weights::split_moe_gguf_experts(&mut weights, &info)?;
                    return Self::build_moe_cpu(info, weights, backend);
                }
                // Backend precedence (so the compiled binary variant decides):
                // 1. MLX (Metal) — native lazy-graph, preferred on Apple Silicon.
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
                // 2. wgpu (Vulkan/DX12/Metal) — explicit `--backend wgpu`, or Auto on
                //    a `-gpu` (wgpu-feature) build, so the GPU binary uses the GPU.
                if use_wgpu_engine(backend) {
                    return Self::build_wgpu_from_weights(info, weights);
                }
                // 3. CPU.
                Ok(Self::Llama(LlamaForward::from_weights_with_backend(
                    info, weights, backend,
                )?))
            }
            ArchType::Phi => {
                // Phi GGUFs fuse Q/K/V (and, for Phi-3/4, gate+up). Split them into
                // the separate / renamed tensors PhiForward expects before loading.
                let is_phi3 = info.model_type == "phi3";
                crate::gguf_weights::split_phi_gguf_fused(
                    &mut weights,
                    info.num_attention_heads,
                    info.num_key_value_heads,
                    info.head_dim,
                    is_phi3,
                )?;
                Ok(Self::Phi(PhiForward::from_weights_with_backend(
                    info, weights, backend,
                )?))
            }
            other => bail!(
                "architecture {other:?} does not yet support GGUF loading — \
                 try a Llama-family GGUF model or use safetensors weights"
            ),
        }
    }

    /// Build a Mixture-of-Experts model on the CPU `LlamaForward` engine.
    ///
    /// MoE is CPU-only for this first cut: MLX (`MlxForwardEngine`) and wgpu know
    /// nothing about experts, so an explicitly-requested GPU backend errors, and
    /// `Auto` on an accelerated build silently falls back to CPU with a warning.
    fn build_moe_cpu(
        info: ModelInfo,
        weights: std::collections::HashMap<String, sapient_core::Tensor>,
        backend: LlmBackendKind,
    ) -> Result<Self> {
        match backend {
            LlmBackendKind::Metal | LlmBackendKind::Wgpu => bail!(
                "Mixture-of-Experts models are only supported on the CPU backend so far \
                 — re-run with `--backend cpu` (MoE on Metal/wgpu is not implemented yet)"
            ),
            _ => {
                if use_mlx_engine(backend) || use_wgpu_engine(backend) {
                    tracing::warn!(
                        "MoE model detected — using the CPU engine \
                         (GPU-accelerated MoE is not implemented yet)"
                    );
                }
            }
        }
        Ok(Self::Llama(LlamaForward::from_weights_with_backend(
            info,
            weights,
            LlmBackendKind::Cpu,
        )?))
    }

    /// Build the wgpu GPU engine from already-loaded weights (GGUF path).
    fn build_wgpu_from_weights(
        info: ModelInfo,
        weights: std::collections::HashMap<String, sapient_core::Tensor>,
    ) -> Result<Self> {
        #[cfg(feature = "wgpu")]
        {
            Ok(Self::Wgpu(Box::new(
                wgpu_engine::WgpuForwardEngine::from_weights(info, weights)?,
            )))
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (info, weights);
            bail!("wgpu backend not compiled in — rebuild with `--features wgpu`")
        }
    }

    /// Build the wgpu GPU engine from weight files (safetensors path).
    #[cfg_attr(not(feature = "wgpu"), allow(unused_variables))]
    fn build_wgpu(info: ModelInfo, weight_paths: &[PathBuf]) -> Result<Self> {
        #[cfg(feature = "wgpu")]
        {
            let weights = crate::weights::load_hf_weights(weight_paths)?;
            Self::build_wgpu_from_weights(info, weights)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            bail!("wgpu backend not compiled in — rebuild with `--features wgpu`")
        }
    }

    pub fn reset_cache(&mut self) {
        match self {
            Self::Llama(f) => f.reset_cache(),
            Self::Phi(f) => f.reset_cache(),
            Self::Gemma3(f) => f.reset_cache(),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => f.reset_cache(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(f) => f.reset_cache(),
        }
    }

    /// Keep only the first `n` cached positions (prefix reuse) and return the
    /// actual number kept (≤ current cache length). The next `forward_logits`
    /// with `use_cache=true` continues from this position. Engines that can't
    /// truncate (MLX) reset fully and return 0 (correct — just no reuse).
    pub fn truncate_cache(&mut self, n: usize) -> usize {
        match self {
            Self::Llama(f) => f.truncate_cache(n),
            Self::Phi(f) => f.truncate_cache(n),
            Self::Gemma3(f) => {
                // v1: no incremental rollback — reset (no reuse, still correct).
                f.reset_cache();
                0
            }
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => {
                f.reset_cache();
                0
            }
            #[cfg(feature = "wgpu")]
            Self::Wgpu(f) => f.truncate_cache(n),
        }
    }

    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        match self {
            Self::Llama(f) => f.forward_logits(input_ids, use_cache),
            Self::Phi(f) => f.forward_logits(input_ids, use_cache),
            Self::Gemma3(f) => f.forward_logits(input_ids, use_cache),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => f.forward_logits(input_ids, use_cache),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(f) => f.forward_logits(input_ids, use_cache),
        }
    }

    pub fn forward_all_logits(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Llama(f) => f.forward_all_logits(input_ids),
            Self::Phi(f) => f.forward_all_logits(input_ids),
            Self::Gemma3(_) => {
                bail!("speculative decoding is not yet supported on the Gemma3 engine")
            }
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => f.forward_all_logits(input_ids),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(f) => f.forward_all_logits(input_ids),
        }
    }

    /// Logits for ALL positions, **appending** `input_ids` to the KV cache
    /// (positions continue from the current cache length). Speculative decoding
    /// uses this to verify drafts with prompt context, then rolls back rejected
    /// tokens via `truncate_cache`. MLX has no incremental cache rollback, so it
    /// falls back to the non-cached path (correct only for single-shot use).
    pub fn forward_all_logits_cached(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Llama(f) => f.forward_all_logits_cached(input_ids),
            Self::Phi(f) => f.forward_all_logits_cached(input_ids),
            Self::Gemma3(_) => {
                bail!("speculative decoding is not yet supported on the Gemma3 engine")
            }
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(f) => f.forward_all_logits(input_ids),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(f) => f.forward_all_logits_cached(input_ids),
        }
    }

    pub fn embed(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Llama(f) => f.embed(input_ids),
            Self::Phi(f) => f.embed(input_ids),
            Self::Gemma3(_) => {
                let _ = input_ids;
                bail!("embed() not yet implemented for the Gemma3 engine")
            }
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(_) => {
                let _ = input_ids;
                bail!("embed() not yet implemented for MlxForwardEngine")
            }
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => {
                let _ = input_ids;
                bail!("embed() not yet implemented for WgpuForwardEngine")
            }
        }
    }

    /// True when layers are split between Metal GPU and CPU (hybrid mode).
    pub fn is_hybrid(&self) -> bool {
        match self {
            Self::Llama(f) => f.is_hybrid(),
            Self::Phi(f) => f.is_hybrid(),
            Self::Gemma3(_) => false,
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(_) => false,
            #[cfg(feature = "wgpu")]
            Self::Wgpu(f) => f.is_hybrid(),
        }
    }

    /// Human-readable backend label.
    pub fn backend_label(&self) -> String {
        match self {
            Self::Llama(f) => f.backend_label(),
            Self::Phi(f) => f.backend_label(),
            Self::Gemma3(_) => "cpu (Gemma3 engine)".to_string(),
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            Self::MlxLlama(_) => "metal (MLX native graph)".to_string(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(f) => f.backend_label(),
        }
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    /// Phase 10.4: explicit wgpu is always honored; CPU/Metal never route
    /// Whisper to wgpu. `Auto`'s answer is machine-dependent (feature flags,
    /// MLX precedence, adapter probe) — assert only that the probe path is safe.
    #[test]
    fn whisper_wgpu_selection() {
        assert!(whisper_wants_wgpu(LlmBackendKind::Wgpu));
        assert!(!whisper_wants_wgpu(LlmBackendKind::Cpu));
        assert!(!whisper_wants_wgpu(LlmBackendKind::Metal));
        let _ = whisper_wants_wgpu(LlmBackendKind::Auto);
    }
}
