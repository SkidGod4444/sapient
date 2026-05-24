//! Architecture detection from HuggingFace `config.json`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ── ArchType ──────────────────────────────────────────────────────────────────

/// The model architecture family, parsed from `config.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArchType {
    /// Llama 1/2/3, Mistral, Vicuna, CodeLlama, WizardLM, Orca…
    Llama,
    /// Microsoft Phi-1/2/3/3.5
    Phi,
    /// Google Gemma / Gemma 2
    Gemma,
    /// GPT-2, CodeGen, GPT-J
    Gpt2,
    /// BERT, RoBERTa, DistilBERT (encoder-only — for embeddings/classification)
    Bert,
    /// Alibaba Qwen / Qwen2
    Qwen,
    /// Mixtral (MoE)
    Mixtral,
    /// Falcon
    Falcon,
    /// MPT (MosaicML)
    Mpt,
    /// BLOOM / BigScience
    Bloom,
    /// T5 / Flan-T5 (encoder-decoder)
    T5,
    /// Any architecture not yet explicitly recognised — still loadable via GGUF.
    Unknown(String),
}

impl ArchType {
    /// Detect arch from the `architectures` field in `config.json`.
    pub fn from_hf_arch_name(name: &str) -> Self {
        match name {
            n if n.contains("Llama") || n.contains("Mistral") || n.contains("CodeLlama") => {
                Self::Llama
            }
            n if n.contains("Phi") => Self::Phi,
            n if n.contains("Gemma") => Self::Gemma,
            n if n.contains("GPT2") || n.contains("Gpt2") => Self::Gpt2,
            n if n.contains("Bert") || n.contains("Roberta") => Self::Bert,
            n if n.contains("Qwen") => Self::Qwen,
            n if n.contains("Mixtral") => Self::Mixtral,
            n if n.contains("Falcon") => Self::Falcon,
            n if n.contains("MPT") => Self::Mpt,
            n if n.contains("Bloom") => Self::Bloom,
            n if n.contains("T5") => Self::T5,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

// ── ModelInfo ─────────────────────────────────────────────────────────────────

/// Parsed `config.json` — hyperparameters needed to build the model graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub arch: ArchType,
    pub model_type: String,

    // Vocabulary
    pub vocab_size: usize,

    // Architecture dimensions
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Number of KV heads — < num_attention_heads for GQA (Llama2/3, Mistral).
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,

    // Normalization
    pub rms_norm_eps: f64,

    // Activation
    pub hidden_act: String,

    // RoPE
    pub rope_theta: f64,

    // Head dimension (derived)
    pub head_dim: usize,

    // Raw config (for any fields we don't explicitly parse)
    #[serde(skip)]
    pub raw: serde_json::Value,
}

impl ModelInfo {
    /// Parse from a `config.json` file on disk.
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).context("Failed to read config.json")?;
        Self::from_json_str(&text)
    }

    /// Parse from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json).context("Invalid config.json")?;

        let arch_names: Vec<String> = raw["architectures"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let arch = arch_names
            .first()
            .map(|n| ArchType::from_hf_arch_name(n))
            .unwrap_or(ArchType::Unknown("unknown".into()));

        let model_type = raw["model_type"].as_str().unwrap_or("unknown").to_owned();
        let vocab_size = raw["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let hidden_size = raw["hidden_size"].as_u64().unwrap_or(4096) as usize;
        let num_hidden_layers = raw["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let num_attention_heads = raw["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        // GQA: fall back to num_attention_heads if not specified.
        let num_key_value_heads = raw["num_key_value_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = raw["intermediate_size"].as_u64().unwrap_or(11008) as usize;
        let max_position_embeddings =
            raw["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;
        let rms_norm_eps = raw["rms_norm_eps"].as_f64().unwrap_or(1e-5);
        let hidden_act = raw["hidden_act"].as_str().unwrap_or("silu").to_owned();
        let rope_theta = raw["rope_theta"].as_f64().unwrap_or(10000.0);
        let head_dim = hidden_size / num_attention_heads;

        Ok(Self {
            arch,
            model_type,
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            max_position_embeddings,
            rms_norm_eps,
            hidden_act,
            rope_theta,
            head_dim,
            raw: raw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "rope_theta": 10000.0
    }"#;

    const PHI_CONFIG: &str = r#"{
        "architectures": ["PhiForCausalLM"],
        "model_type": "phi",
        "vocab_size": 51200,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "intermediate_size": 8192,
        "max_position_embeddings": 2048,
        "hidden_act": "gelu"
    }"#;

    #[test]
    fn parse_llama_config() {
        let info = ModelInfo::from_json_str(LLAMA_CONFIG).unwrap();
        assert_eq!(info.arch, ArchType::Llama);
        assert_eq!(info.num_key_value_heads, 8); // GQA
        assert_eq!(info.head_dim, 128); // 4096 / 32
    }

    #[test]
    fn parse_phi_config() {
        let info = ModelInfo::from_json_str(PHI_CONFIG).unwrap();
        assert_eq!(info.arch, ArchType::Phi);
        // No KV heads → defaults to n_heads
        assert_eq!(info.num_key_value_heads, 32);
    }
}
