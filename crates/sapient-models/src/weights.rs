// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! HuggingFace safetensors weight loading and key resolution.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use sapient_core::Tensor;
use sapient_io::SafetensorsLoader;

/// Load and merge safetensors shards from disk.
pub fn load_hf_weights(paths: &[PathBuf]) -> Result<HashMap<String, Tensor>> {
    let mut merged = HashMap::new();
    for path in paths {
        let shard = SafetensorsLoader::load(path)
            .with_context(|| format!("failed to load weights from {}", path.display()))?;
        for (k, v) in shard {
            if merged.insert(k.clone(), v).is_some() {
                bail!("duplicate weight key '{k}' in shard {}", path.display());
            }
        }
    }
    Ok(merged)
}

/// Detect the common prefix for transformer weight keys.
pub fn detect_weight_prefix(weights: &HashMap<String, Tensor>) -> String {
    const CANDIDATES: &[&str] = &[
        "model.text_model.",
        "model.language_model.",
        "transformer.",
        "model.",
        "gpt_neox.",
    ];

    for prefix in CANDIDATES {
        let embed_key = format!("{prefix}embed_tokens.weight");
        if weights.contains_key(&embed_key) {
            return prefix.to_string();
        }
    }

    if weights.contains_key("embed_tokens.weight") {
        return String::new();
    }

    // Fall back: find any embed_tokens key.
    weights
        .keys()
        .find(|k| k.ends_with("embed_tokens.weight"))
        .map(|k| {
            k.strip_suffix("embed_tokens.weight")
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_else(|| "model.".to_string())
}

/// Resolve a weight tensor by logical suffix (e.g. `layers.0.self_attn.q_proj`).
pub fn resolve_weight<'a>(
    weights: &'a HashMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Result<&'a Tensor> {
    let key = format!("{prefix}{suffix}.weight");
    weights
        .get(&key)
        .or_else(|| weights.get(suffix))
        .with_context(|| format!("missing weight '{key}'"))
}

/// Resolve an optional bias tensor by logical suffix (e.g. `layers.0.self_attn.q_proj`).
/// Returns `None` when the model has no bias for that layer (e.g. Llama/Mistral).
pub fn resolve_bias<'a>(
    weights: &'a HashMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Option<&'a Tensor> {
    let key = format!("{prefix}{suffix}.bias");
    weights
        .get(&key)
        .or_else(|| weights.get(&format!("{suffix}.bias")))
}

/// Resolve lm_head — may live outside the model prefix.
pub fn resolve_lm_head<'a>(
    weights: &'a HashMap<String, Tensor>,
    prefix: &str,
    tie_word_embeddings: bool,
    embed_key: &str,
) -> Result<&'a Tensor> {
    if tie_word_embeddings {
        return weights
            .get(embed_key)
            .with_context(|| format!("missing tied embedding weight '{embed_key}'"));
    }

    weights
        .get("lm_head.weight")
        .or_else(|| weights.get(&format!("{prefix}lm_head.weight")))
        // Tied embeddings: GGUF metadata has no `tie_word_embeddings` flag, so a
        // model that ties its output projection to the input embedding (SmolLM2,
        // Llama-3.2-1B/3B, Qwen small) simply omits `output.weight`. When no
        // explicit head exists, fall back to the embedding matrix.
        .or_else(|| weights.get(embed_key))
        .with_context(|| format!("missing lm_head.weight (and no '{embed_key}' to tie to)"))
}

pub fn tie_word_embeddings_from_config(raw: &serde_json::Value) -> bool {
    raw.get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .or_else(|| {
            raw.get("text_config")
                .and_then(|tc| tc.get("tie_word_embeddings"))
                .and_then(|v| v.as_bool())
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_text_model_prefix() {
        let mut w = HashMap::new();
        w.insert(
            "model.text_model.embed_tokens.weight".into(),
            Tensor::zeros(vec![1, 1], sapient_core::DType::F32).unwrap(),
        );
        assert_eq!(detect_weight_prefix(&w), "model.text_model.");
    }
}
