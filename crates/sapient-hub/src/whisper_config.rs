// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Whisper model configuration, parsed from a HuggingFace `config.json`.
//!
//! Whisper is an encoder-decoder model with two independent transformer stacks,
//! so it does not fit the single-stack, LLM-centric [`crate::ModelInfo`]. This
//! lightweight struct captures exactly the hyperparameters the forward engine
//! and mel front-end need.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Hyperparameters for an OpenAI Whisper checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Mel filterbank channels (80 for tiny…large-v2, 128 for large-v3).
    pub num_mel_bins: usize,
    /// Model width — shared by encoder and decoder (`d_model`).
    pub d_model: usize,

    /// Encoder transformer depth.
    pub encoder_layers: usize,
    /// Encoder attention heads.
    pub encoder_attention_heads: usize,
    /// Encoder FFN inner size (`encoder_ffn_dim`).
    pub encoder_ffn_dim: usize,

    /// Decoder transformer depth.
    pub decoder_layers: usize,
    /// Decoder attention heads.
    pub decoder_attention_heads: usize,
    /// Decoder FFN inner size (`decoder_ffn_dim`).
    pub decoder_ffn_dim: usize,

    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Maximum decoder positions (learned positional table length, e.g. 448).
    pub max_target_positions: usize,
    /// Maximum encoder positions (audio context frames, e.g. 1500).
    pub max_source_positions: usize,
}

impl WhisperConfig {
    /// Per-head dimension (`d_model / encoder_attention_heads`). Whisper uses the
    /// same head_dim for encoder and decoder.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads.max(1)
    }

    /// Parse from a `config.json` file on disk.
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).context("reading Whisper config.json from disk")?;
        Self::from_json_str(&text)
    }

    /// Parse from a JSON string. Missing fields fall back to Whisper-base values
    /// so a sparse config still loads.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json).context("invalid Whisper config")?;
        let u = |key: &str, default: usize| v[key].as_u64().map(|n| n as usize).unwrap_or(default);
        Ok(Self {
            num_mel_bins: u("num_mel_bins", 80),
            d_model: u("d_model", 512),
            encoder_layers: u("encoder_layers", 6),
            encoder_attention_heads: u("encoder_attention_heads", 8),
            encoder_ffn_dim: u("encoder_ffn_dim", 2048),
            decoder_layers: u("decoder_layers", 6),
            decoder_attention_heads: u("decoder_attention_heads", 8),
            decoder_ffn_dim: u("decoder_ffn_dim", 2048),
            vocab_size: u("vocab_size", 51865),
            max_target_positions: u("max_target_positions", 448),
            max_source_positions: u("max_source_positions", 1500),
        })
    }
}

/// Whisper decoding controls from `generation_config.json` (HF).
///
/// `suppress_tokens` is masked to -inf at *every* decode step (non-speech
/// symbols/markup); `begin_suppress_tokens` (`[220, 50256]` = blank + eot) is
/// additionally masked on the *first* sampled step. Both default empty when the
/// file is absent, so suppression simply no-ops.
#[derive(Debug, Clone, Default)]
pub struct WhisperGenConfig {
    pub suppress_tokens: Vec<u32>,
    pub begin_suppress_tokens: Vec<u32>,
}

impl WhisperGenConfig {
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).context("reading generation_config.json")?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).context("invalid generation_config.json")?;
        // Parse only the two integer arrays; drop negatives (legacy -1 sentinel).
        let ids = |key: &str| -> Vec<u32> {
            v[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|n| n as u32)
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok(Self {
            suppress_tokens: ids("suppress_tokens"),
            begin_suppress_tokens: ids("begin_suppress_tokens"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trimmed openai/whisper-base config.json.
    const BASE: &str = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "model_type": "whisper",
        "num_mel_bins": 80,
        "d_model": 512,
        "encoder_layers": 6,
        "encoder_attention_heads": 8,
        "encoder_ffn_dim": 2048,
        "decoder_layers": 6,
        "decoder_attention_heads": 8,
        "decoder_ffn_dim": 2048,
        "vocab_size": 51865,
        "max_target_positions": 448,
        "max_source_positions": 1500
    }"#;

    #[test]
    fn parse_base_config() {
        let c = WhisperConfig::from_json_str(BASE).unwrap();
        assert_eq!(c.d_model, 512);
        assert_eq!(c.encoder_layers, 6);
        assert_eq!(c.decoder_attention_heads, 8);
        assert_eq!(c.head_dim(), 64); // 512 / 8
        assert_eq!(c.vocab_size, 51865);
        assert_eq!(c.max_source_positions, 1500);
    }

    #[test]
    fn parse_gen_config() {
        let g = WhisperGenConfig::from_json_str(
            r#"{ "suppress_tokens": [1, 2, 7, 50257], "begin_suppress_tokens": [220, 50256] }"#,
        )
        .unwrap();
        assert_eq!(g.suppress_tokens, vec![1, 2, 7, 50257]);
        assert_eq!(g.begin_suppress_tokens, vec![220, 50256]);
        // Missing file / fields → empty (suppression no-ops).
        let empty = WhisperGenConfig::from_json_str("{}").unwrap();
        assert!(empty.suppress_tokens.is_empty() && empty.begin_suppress_tokens.is_empty());
    }
}
