//! Curated registry of models SAPIENT officially supports.
//!
//! SAPIENT does not run arbitrary HuggingFace repos yet — every model listed
//! here has been chosen because its architecture is implemented and verified in
//! the forward engine (`sapient-models`). The registry maps a friendly alias (or
//! the canonical HuggingFace repo id, or any extra alias) to the repo that is
//! actually downloaded.
//!
//! To add a model: confirm its architecture is supported by the forward path,
//! then add a [`SupportedModel`] row below.

use anyhow::{bail, Result};
use std::collections::HashMap;

/// A model SAPIENT knows how to run.
#[derive(Debug, Clone, Copy)]
pub struct SupportedModel {
    /// Canonical short, user-facing name (e.g. `phi-2`).
    pub alias: &'static str,
    /// HuggingFace repository that is downloaded.
    pub repo_id: &'static str,
    /// Architecture family, for display.
    pub family: &'static str,
    /// Approximate parameter count, for display.
    pub params: &'static str,
    /// Whether the repo requires accepting a license / an HF token.
    pub gated: bool,
    /// Additional names that resolve to this model (case-insensitive).
    pub extra_aliases: &'static [&'static str],
}

/// Broad capability bucket for a model — drives `sapient models` grouping and
/// command-level validation (e.g. `speak` rejects a speech-to-text model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    /// Text generation / chat — the default `sapient chat` path.
    Chat,
    /// Speech-to-text (`sapient transcribe`) — Whisper.
    SpeechToText,
    /// Text-to-speech (`sapient speak`) — Orpheus, Kokoro.
    TextToSpeech,
    /// Vision-language (`sapient see`) — SmolVLM.
    Vision,
}

impl ModelCategory {
    /// Short human label for display headers.
    pub fn label(self) -> &'static str {
        match self {
            ModelCategory::Chat => "Text generation (chat)",
            ModelCategory::SpeechToText => "Speech-to-text (transcribe)",
            ModelCategory::TextToSpeech => "Text-to-speech (speak)",
            ModelCategory::Vision => "Vision-language (see)",
        }
    }
}

impl SupportedModel {
    /// Which capability bucket this model belongs to, derived from `family`.
    pub fn category(&self) -> ModelCategory {
        match self.family {
            "Whisper" => ModelCategory::SpeechToText,
            "Orpheus" | "Kokoro" => ModelCategory::TextToSpeech,
            "SmolVLM" => ModelCategory::Vision,
            _ => ModelCategory::Chat,
        }
    }
}

/// The full catalog of supported models.
///
/// Architectures in use:
/// - **Phi** (`microsoft/phi-*`) → `PhiForward` (LayerNorm + biases, partial RoPE, parallel block).
/// - **Llama / Qwen2.5 / SmolLM2 / TinyLlama** → `LlamaForward` (RMSNorm, RoPE, SwiGLU; Qwen adds q/k/v biases).
pub const CATALOG: &[SupportedModel] = &[
    // ── Phi family (Phi forward engine) ──────────────────────────────────────
    SupportedModel {
        alias: "openhorizon/phi-2",
        repo_id: "microsoft/phi-2",
        family: "Phi",
        params: "2.7B",
        gated: false,
        extra_aliases: &["phi-2", "phi2"],
    },
    SupportedModel {
        alias: "openhorizon/phi-1.5",
        repo_id: "microsoft/phi-1_5",
        family: "Phi",
        params: "1.3B",
        gated: false,
        extra_aliases: &["phi-1.5", "phi-1_5", "phi1.5"],
    },
    SupportedModel {
        alias: "openhorizon/phi-1",
        repo_id: "microsoft/phi-1",
        family: "Phi",
        params: "1.3B",
        gated: false,
        extra_aliases: &["phi-1"],
    },
    // ── Qwen2.5 (Llama forward engine, with q/k/v biases) ─────────────────────
    SupportedModel {
        alias: "openhorizon/qwen2.5-0.5b",
        repo_id: "Qwen/Qwen2.5-0.5B-Instruct",
        family: "Qwen2.5",
        params: "0.5B",
        gated: false,
        extra_aliases: &["qwen2.5-0.5b", "qwen2.5-0.5b-instruct", "qwen0.5b"],
    },
    SupportedModel {
        alias: "openhorizon/qwen2.5-1.5b",
        repo_id: "Qwen/Qwen2.5-1.5B-Instruct",
        family: "Qwen2.5",
        params: "1.5B",
        gated: false,
        extra_aliases: &["qwen2.5-1.5b", "qwen2.5-1.5b-instruct", "qwen1.5b"],
    },
    SupportedModel {
        alias: "openhorizon/qwen2.5-3b",
        repo_id: "Qwen/Qwen2.5-3B-Instruct",
        family: "Qwen2.5",
        params: "3B",
        gated: false,
        extra_aliases: &["qwen2.5-3b", "qwen2.5-3b-instruct"],
    },
    // ── SmolLM2 (Llama forward engine) ────────────────────────────────────────
    SupportedModel {
        alias: "openhorizon/smollm2-360m",
        repo_id: "HuggingFaceTB/SmolLM2-360M-Instruct",
        family: "Llama",
        params: "360M",
        gated: false,
        extra_aliases: &["smollm2-360m", "smollm2-360m-instruct"],
    },
    SupportedModel {
        alias: "openhorizon/smollm2-1.7b",
        repo_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family: "Llama",
        params: "1.7B",
        gated: false,
        extra_aliases: &["smollm2-1.7b", "smollm2-1.7b-instruct"],
    },
    // ── TinyLlama (Llama forward engine) ──────────────────────────────────────
    SupportedModel {
        alias: "openhorizon/tinyllama-1.1b",
        repo_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        family: "Llama",
        params: "1.1B",
        gated: false,
        extra_aliases: &["tinyllama-1.1b", "tinyllama"],
    },
    // ── Llama 3.2 (Llama forward engine — Meta ungated these in late 2024) ──────
    SupportedModel {
        alias: "openhorizon/llama-3.2-1b",
        repo_id: "unsloth/Llama-3.2-1B-Instruct",
        family: "Llama",
        params: "1B",
        gated: false,
        extra_aliases: &["llama-3.2-1b", "llama3.2-1b"],
    },
    SupportedModel {
        alias: "openhorizon/llama-3.2-3b",
        repo_id: "meta-llama/Llama-3.2-3B-Instruct",
        family: "Llama",
        params: "3B",
        gated: false,
        extra_aliases: &["llama-3.2-3b", "llama3.2-3b"],
    },
    // ── Mistral (Llama forward engine) ───────────────────────────────────────
    // Note: safetensors = 13.5 GB — prefer openhorizon/mistral-7b-q4 on edge devices.
    SupportedModel {
        alias: "openhorizon/mistral-7b",
        repo_id: "mistralai/Mistral-7B-Instruct-v0.2",
        family: "Mistral",
        params: "7B",
        gated: false,
        extra_aliases: &["mistral-7b", "mistral-7b-instruct"],
    },
    // ── Phi GGUF variants ─────────────────────────────────────────────────────
    SupportedModel {
        alias: "openhorizon/phi-2-q4",
        repo_id: "TheBloke/phi-2-GGUF",
        family: "Phi",
        params: "2.7B Q8_0",
        gated: false,
        extra_aliases: &["phi-2-q4", "phi2-q4", "phi-2-gguf"],
    },
    // ── Quantized GGUF models (Phase 1: huge models on small devices) ─────────
    // These download a single .gguf file; RAM ≈ file size (no F32 expansion).
    SupportedModel {
        alias: "openhorizon/qwen2.5-0.5b-q4",
        repo_id: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        family: "Qwen2.5",
        params: "0.5B Q4_K_M",
        gated: false,
        extra_aliases: &["qwen2.5-0.5b-q4", "qwen0.5b-q4"],
    },
    SupportedModel {
        alias: "openhorizon/qwen2.5-1.5b-q4",
        repo_id: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        family: "Qwen2.5",
        params: "1.5B Q4_K_M",
        gated: false,
        extra_aliases: &["qwen2.5-1.5b-q4", "qwen1.5b-q4"],
    },
    SupportedModel {
        alias: "openhorizon/smollm2-360m-q4",
        repo_id: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        family: "Llama",
        params: "360M Q8_0",
        gated: false,
        extra_aliases: &["smollm2-360m-q4"],
    },
    SupportedModel {
        alias: "openhorizon/smollm2-1.7b-q4",
        repo_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        family: "Llama",
        params: "1.7B Q4_K_M",
        gated: false,
        extra_aliases: &["smollm2-1.7b-q4"],
    },
    SupportedModel {
        alias: "openhorizon/llama-3.2-1b-q4",
        repo_id: "unsloth/Llama-3.2-1B-Instruct-GGUF",
        family: "Llama",
        params: "1B Q4_K_M",
        gated: false,
        extra_aliases: &["llama-3.2-1b-q4", "llama3.2-1b-q4"],
    },
    SupportedModel {
        alias: "openhorizon/llama-3.2-3b-q4",
        repo_id: "unsloth/Llama-3.2-3B-Instruct-GGUF",
        family: "Llama",
        params: "3B Q4_K_M",
        gated: false,
        extra_aliases: &["llama-3.2-3b-q4", "llama3.2-3b-q4"],
    },
    // ── Mistral GGUF (recommended over the 13.5 GB safetensors version) ──────
    SupportedModel {
        alias: "openhorizon/mistral-7b-q4",
        repo_id: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        family: "Mistral",
        params: "7B Q4_K_M",
        gated: false,
        extra_aliases: &["mistral-7b-q4", "mistral7b-q4"],
    },
    // ── New Unsloth edge-device models ────────────────────────────────────────
    SupportedModel {
        alias: "openhorizon/smollm2-135m-q4",
        repo_id: "unsloth/SmolLM2-135M-Instruct-GGUF",
        family: "Llama",
        params: "135M Q4_K_M",
        gated: false,
        extra_aliases: &["smollm2-135m-q4", "smollm-135m"],
    },
    SupportedModel {
        alias: "openhorizon/qwen2.5-coder-0.5b",
        repo_id: "unsloth/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        family: "Qwen2.5",
        params: "0.5B Q4_K_M",
        gated: false,
        extra_aliases: &["qwen2.5-coder-0.5b", "qwen-coder-0.5b"],
    },
    SupportedModel {
        alias: "openhorizon/qwen2.5-coder-1.5b",
        repo_id: "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        family: "Qwen2.5",
        params: "1.5B Q4_K_M",
        gated: false,
        extra_aliases: &["qwen2.5-coder-1.5b", "qwen-coder-1.5b"],
    },
    SupportedModel {
        alias: "openhorizon/llama-3.1-8b-q4",
        repo_id: "unsloth/Llama-3.1-8B-Instruct-GGUF",
        family: "Llama",
        params: "8B Q4_K_M",
        gated: false,
        extra_aliases: &["llama-3.1-8b-q4", "llama3.1-8b-q4", "llama-3.1-8b"],
    },
    SupportedModel {
        alias: "openhorizon/deepseek-r1-8b",
        repo_id: "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        family: "Llama",
        params: "8B Q4_K_M",
        gated: false,
        extra_aliases: &["deepseek-r1-8b", "deepseek-r1-llama-8b"],
    },
    SupportedModel {
        alias: "openhorizon/phi-4-mini",
        repo_id: "unsloth/Phi-4-mini-instruct-GGUF",
        family: "Phi",
        params: "3.8B Q4_K_M",
        gated: false,
        extra_aliases: &["phi-4-mini", "phi4-mini"],
    },
    // ── Whisper (speech-to-text — `sapient transcribe`) ──────────────────────
    SupportedModel {
        alias: "openhorizon/whisper-tiny",
        repo_id: "openai/whisper-tiny",
        family: "Whisper",
        params: "39M",
        gated: false,
        extra_aliases: &["whisper-tiny"],
    },
    SupportedModel {
        alias: "openhorizon/whisper-base",
        repo_id: "openai/whisper-base",
        family: "Whisper",
        params: "74M",
        gated: false,
        extra_aliases: &["whisper-base"],
    },
    SupportedModel {
        alias: "openhorizon/whisper-small",
        repo_id: "openai/whisper-small",
        family: "Whisper",
        params: "244M",
        gated: false,
        extra_aliases: &["whisper-small"],
    },
    // ── Orpheus TTS (text-to-speech — `sapient speak`) ───────────────────────
    // Llama-3.2-3B fine-tune that emits SNAC audio-codec tokens; runs on the
    // Llama forward engine. Ungated Q4_K_M GGUF; vocab 156940.
    SupportedModel {
        alias: "openhorizon/orpheus-3b",
        repo_id: "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
        family: "Orpheus",
        params: "3B",
        gated: false,
        extra_aliases: &["orpheus-3b", "orpheus", "orpheus-tts"],
    },
    // ── Kokoro-82M TTS (text-to-speech — `sapient speak`) ────────────────────
    // Non-autoregressive StyleTTS2 + ISTFTNet; real-time on CPU. Converted
    // safetensors mirror; the default voice for `converse --speak`.
    SupportedModel {
        alias: "openhorizon/kokoro-82m",
        repo_id: "sai1974dev/kokoro-82m-safetensors",
        family: "Kokoro",
        params: "82M",
        gated: false,
        extra_aliases: &["kokoro-82m", "kokoro", "kokoro-tts"],
    },
    // ── Gemma3 (text chat — `Gemma3Forward` engine) ─────────────────────────
    // QK-norm + sandwich norms + alternating sliding/global attention.
    // gemma-3-1b via the ungated unsloth safetensors mirror.
    SupportedModel {
        alias: "openhorizon/gemma-3-1b",
        repo_id: "unsloth/gemma-3-1b-it",
        family: "Gemma3",
        params: "1B",
        gated: false,
        extra_aliases: &["gemma-3-1b", "gemma3-1b", "gemma-3-1b-it"],
    },
    // Multimodal Gemma3 (text chat AND `sapient see`) via the ungated mirror.
    SupportedModel {
        alias: "openhorizon/gemma-3-4b",
        repo_id: "unsloth/gemma-3-4b-it",
        family: "Gemma3",
        params: "4B",
        gated: false,
        extra_aliases: &["gemma-3-4b", "gemma3-4b", "gemma-3-4b-it"],
    },
    // MedGemma: Google's medical Gemma3 (gated — accept the Health AI
    // Developer Foundations terms on HF, then `sapient login`). Multimodal
    // checkpoint: `sapient chat medgemma-4b` for medical Q&A text, and
    // `sapient see scan.png --model medgemma-4b` for medical image analysis
    // (X-ray / dermatology / pathology) via the Gemma3 VLM path.
    SupportedModel {
        alias: "openhorizon/medgemma-4b",
        repo_id: "google/medgemma-4b-it",
        family: "Gemma3",
        params: "4B",
        gated: true,
        extra_aliases: &["medgemma-4b", "medgemma", "medgemma-4b-it"],
    },
    // ── SmolVLM (vision-language — `sapient see`) ───────────────────────────
    // Idefics3: SigLIP tower + pixel-shuffle connector + SmolLM2-135M backbone
    // (safetensors; loaded by `VlmPipeline`, routed via `is_vlm_model`).
    SupportedModel {
        alias: "openhorizon/smolvlm-256m",
        repo_id: "HuggingFaceTB/SmolVLM-256M-Instruct",
        family: "SmolVLM",
        params: "256M",
        gated: false,
        extra_aliases: &["smolvlm-256m", "smolvlm", "smolvlm-256m-instruct"],
    },
];

/// Resolve a VLM alias (or raw repo id) to its Hugging Face repo. Catalog
/// aliases in the `Vision` category resolve normally; anything containing a
/// `/` passes through as a raw repo id (Idefics3-family only — the pipeline
/// validates `model_type`).
pub fn resolve_vlm_repo(model: &str) -> String {
    let lower = model.to_lowercase();
    // Any catalog alias resolves (multimodal Gemma3 models live in the Chat
    // category but also serve `sapient see`); the pipeline validates the
    // config's model_type, so a text-only model errors cleanly.
    for m in CATALOG {
        if m.alias.eq_ignore_ascii_case(model)
            || m.extra_aliases
                .iter()
                .any(|a| a.eq_ignore_ascii_case(&lower))
        {
            return m.repo_id.to_string();
        }
    }
    model.to_string()
}

/// True when `model` names a catalog VLM (routes the CLI to `sapient see`'s
/// pipeline instead of the text `Pipeline`).
pub fn is_vlm_model(model: &str) -> bool {
    let lower = model.to_lowercase();
    CATALOG.iter().any(|m| {
        m.category() == ModelCategory::Vision
            && (m.alias.eq_ignore_ascii_case(model)
                || m.extra_aliases
                    .iter()
                    .any(|a| a.eq_ignore_ascii_case(&lower)))
    })
}

/// All supported models, for display (e.g. `sapient list --available`).
pub fn catalog() -> &'static [SupportedModel] {
    CATALOG
}

/// Look up a model by any of its names (alias, repo id, or extra alias).
pub fn lookup(name: &str) -> Option<&'static SupportedModel> {
    let n = name.trim().to_lowercase();
    CATALOG.iter().find(|m| {
        m.alias.to_lowercase() == n
            || m.repo_id.to_lowercase() == n
            || m.extra_aliases.iter().any(|a| a.to_lowercase() == n)
    })
}

/// Every lowercase name a model answers to (alias, repo id, extra aliases).
fn model_names(m: &SupportedModel) -> Vec<String> {
    let mut names = vec![m.alias.to_lowercase(), m.repo_id.to_lowercase()];
    names.extend(m.extra_aliases.iter().map(|a| a.to_lowercase()));
    names
}

/// Classic Levenshtein edit distance (used for typo tolerance).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Find the model(s) a possibly-mistyped name could refer to. Prefers prefix
/// matches (e.g. `qwen2.5-0.5` → `qwen2.5-0.5b`); falls back to small typos.
/// Returns the unique deduplicated set of candidate models.
fn fuzzy_candidates(input: &str) -> Vec<&'static SupportedModel> {
    let n = input.trim().to_lowercase();
    if n.is_empty() {
        return Vec::new();
    }

    let mut prefix = Vec::new();
    let mut typo = Vec::new();
    // Only allow typo tolerance for inputs long enough to be unambiguous.
    let max_dist = if n.chars().count() >= 5 { 2 } else { 0 };

    for m in CATALOG {
        let names = model_names(m);
        let is_prefix = names
            .iter()
            .any(|name| name.starts_with(&n) || n.starts_with(name.as_str()));
        let is_typo = max_dist > 0 && names.iter().any(|name| edit_distance(name, &n) <= max_dist);
        if is_prefix {
            prefix.push(m);
        } else if is_typo {
            typo.push(m);
        }
    }

    // Prefer prefix matches; only use typo matches if there were no prefix hits.
    let chosen = if !prefix.is_empty() { prefix } else { typo };
    let mut seen = std::collections::HashSet::new();
    chosen
        .into_iter()
        .filter(|m| seen.insert(m.repo_id))
        .collect()
}

/// Resolve a name to the HuggingFace repository id to download.
///
/// Accepts the friendly alias, the canonical repo id, or any registered alias.
/// Errors (with the full supported list) for anything not in the catalog.
pub fn resolve_model_alias(alias: &str) -> Result<String> {
    // 1) Exact match wins.
    if let Some(m) = lookup(alias) {
        return Ok(m.repo_id.to_string());
    }

    // 2) Forgiving match: unambiguous prefix (e.g. `qwen2.5-0.5` → `qwen2.5-0.5b`)
    //    or a close typo. Auto-resolve only when it points to exactly one model.
    let candidates = fuzzy_candidates(alias);
    match candidates.as_slice() {
        [m] => {
            eprintln!(
                "note: '{alias}' isn't an exact model name — using closest match '{}'",
                m.alias
            );
            Ok(m.repo_id.to_string())
        }
        [] => bail!(
            "Model '{}' is not in the SAPIENT registry.\n\nSupported models:\n{}",
            alias,
            supported_list_pretty()
        ),
        many => {
            let names = many
                .iter()
                .map(|m| format!("  {}", m.alias))
                .collect::<Vec<_>>()
                .join("\n");
            bail!(
                "Model '{}' is ambiguous. Did you mean one of:\n{names}",
                alias
            )
        }
    }
}

/// A human-readable, aligned list of supported models for error/help messages.
pub fn supported_list_pretty() -> String {
    let mut lines = Vec::new();
    for m in CATALOG {
        let gated = if m.gated {
            "  (gated — run `sapient login`)"
        } else {
            ""
        };
        lines.push(format!(
            "  {:<16} {:<34} {:<8} {}{}",
            m.alias, m.repo_id, m.params, m.family, gated
        ));
    }
    lines.join("\n")
}

/// Reverse map: HuggingFace repo id (lowercased) → friendly alias.
/// Used to display friendly names in `sapient list`.
pub fn reverse_alias_map() -> HashMap<String, String> {
    CATALOG
        .iter()
        .map(|m| (m.repo_id.to_lowercase(), m.alias.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_alias_and_repo_and_extra() {
        assert_eq!(resolve_model_alias("phi-2").unwrap(), "microsoft/phi-2");
        assert_eq!(
            resolve_model_alias("microsoft/phi-2").unwrap(),
            "microsoft/phi-2"
        );
        assert_eq!(
            resolve_model_alias("openhorizon/phi-2").unwrap(),
            "microsoft/phi-2"
        );
        assert_eq!(
            resolve_model_alias("Qwen/Qwen2.5-0.5B-Instruct").unwrap(),
            "Qwen/Qwen2.5-0.5B-Instruct"
        );
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(resolve_model_alias("PHI-2").unwrap(), "microsoft/phi-2");
    }

    #[test]
    fn unknown_model_errors_with_list() {
        let err = resolve_model_alias("totally/unknown")
            .unwrap_err()
            .to_string();
        assert!(err.contains("not in the SAPIENT registry"));
        assert!(err.contains("phi-2"));
    }

    #[test]
    fn reverse_map_has_friendly_name() {
        let m = reverse_alias_map();
        assert_eq!(
            m.get("microsoft/phi-2").map(String::as_str),
            Some("openhorizon/phi-2")
        );
    }

    #[test]
    fn resolves_unambiguous_prefix_typo() {
        // Now that qwen2.5-0.5b-q4 also exists, 'qwen2.5-0.5' is ambiguous — the
        // resolver should error with the candidate list rather than silently pick one.
        let err = resolve_model_alias("openhorizon/qwen2.5-0.5")
            .unwrap_err()
            .to_string();
        assert!(err.contains("ambiguous"), "expected ambiguous, got: {err}");
        // Full aliases still resolve exactly.
        assert_eq!(
            resolve_model_alias("openhorizon/qwen2.5-0.5b").unwrap(),
            "Qwen/Qwen2.5-0.5B-Instruct"
        );
        assert_eq!(
            resolve_model_alias("openhorizon/qwen2.5-0.5b-q4").unwrap(),
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        );
    }

    #[test]
    fn resolves_close_typo() {
        // One transposed/altered char within edit distance.
        assert_eq!(
            resolve_model_alias("tinyllama-1.1").unwrap(),
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        );
    }

    #[test]
    fn ambiguous_prefix_errors_with_candidates() {
        // `qwen2.5-` is a prefix of three models → must not silently pick one.
        let err = resolve_model_alias("openhorizon/qwen2.5-")
            .unwrap_err()
            .to_string();
        assert!(err.contains("ambiguous"), "got: {err}");
        assert!(err.contains("qwen2.5-0.5b") && err.contains("qwen2.5-1.5b"));
    }
}
