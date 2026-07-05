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
    /// Google Gemma 3 (QK-norm, sandwich norms, sliding/global attention) —
    /// gemma-3-*b and MedGemma text.
    Gemma3,
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
    /// OpenAI Whisper (speech-to-text, encoder-decoder). Parsed into a separate
    /// [`crate::whisper_config::WhisperConfig`], not the LLM-centric `ModelInfo`.
    Whisper,
    /// Any architecture not yet explicitly recognised — still loadable via GGUF.
    Unknown(String),
}

impl ArchType {
    /// Detect arch from the `architectures` field in `config.json`.
    pub fn from_hf_arch_name(name: &str) -> Self {
        match name {
            n if n.contains("Llama")
                || n.contains("Mistral")
                || n.contains("CodeLlama")
                || n.contains("VLlama") =>
            {
                Self::Llama
            }
            n if n.contains("Phi") => Self::Phi,
            // Gemma3 MUST match before the Gemma catch-all ("Gemma3ForCausalLM"
            // contains "Gemma") or it silently routes to the Llama-family
            // engine and emits token salad.
            n if n.contains("Gemma3") => Self::Gemma3,
            n if n.contains("Gemma") => Self::Gemma,
            n if n.contains("GPT2") || n.contains("Gpt2") => Self::Gpt2,
            n if n.contains("Bert") || n.contains("Roberta") => Self::Bert,
            n if n.contains("Qwen") => Self::Qwen,
            n if n.contains("Mixtral") => Self::Mixtral,
            n if n.contains("Falcon") => Self::Falcon,
            n if n.contains("MPT") => Self::Mpt,
            n if n.contains("Bloom") => Self::Bloom,
            n if n.contains("T5") => Self::T5,
            n if n.contains("Whisper") => Self::Whisper,
            n if n.contains("Idefics") || n.contains("SmolVLM") => Self::Llama,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

// ── MoE ───────────────────────────────────────────────────────────────────────

/// Router scoring function for the MoE gate.
///
/// Mixtral/Qwen-MoE score experts with a **softmax** over the router logits;
/// DeepSeek-V3 / GLM-MoE use a **sigmoid** gate (with group-limited routing and a
/// scaling factor that this first cut does not yet implement). Kept as a parsed
/// field so those models fail loudly rather than routing silently-wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MoeScoring {
    Softmax,
    Sigmoid,
}

/// Mixture-of-Experts configuration — present (`Some`) only for MoE models
/// (Mixtral, Qwen-MoE, DeepSeek/GLM-MoE), `None` for dense models. MoE is
/// detected by the presence of this config, **not** by [`ArchType`]: a Mixtral
/// GGUF reports `general.architecture = "llama"`, so it arrives as
/// `ArchType::Llama` with `num_experts > 0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Total routed experts per MoE layer (`num_local_experts` / `n_routed_experts`).
    pub num_experts: usize,
    /// Experts activated per token (`num_experts_per_tok`).
    pub top_k: usize,
    /// Per-expert FFN intermediate size (`moe_intermediate_size`). For Mixtral
    /// this equals the dense `intermediate_size`.
    pub expert_intermediate_size: usize,
    /// Always-on shared experts (`n_shared_experts`, DeepSeek/Qwen-MoE). 0 for
    /// Mixtral. Not yet implemented in the forward engine.
    pub num_shared_experts: usize,
    /// Leading dense layers before MoE begins (`first_k_dense_replace`). 0 for
    /// Mixtral (all layers are MoE).
    pub first_k_dense: usize,
    /// Renormalise the top-k routing weights to sum to 1 (`norm_topk_prob`).
    /// Mixtral renormalises; getting this wrong degrades quality silently.
    pub norm_topk_prob: bool,
    /// Router scoring function (softmax vs sigmoid).
    pub scoring_func: MoeScoring,
}

/// Parse MoE hyperparameters from a `config.json` value. Returns `None` for
/// dense models (no `num_local_experts` / `n_routed_experts`, or it is 0).
fn parse_moe_config(cfg: &serde_json::Value, intermediate_size: usize) -> Option<MoeConfig> {
    let num_experts = cfg["num_local_experts"]
        .as_u64()
        .or_else(|| cfg["n_routed_experts"].as_u64())
        .unwrap_or(0) as usize;
    if num_experts == 0 {
        return None;
    }
    let top_k = cfg["num_experts_per_tok"].as_u64().unwrap_or(2) as usize;
    let expert_intermediate_size = cfg["moe_intermediate_size"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(intermediate_size);
    let num_shared_experts = cfg["n_shared_experts"].as_u64().unwrap_or(0) as usize;
    let first_k_dense = cfg["first_k_dense_replace"].as_u64().unwrap_or(0) as usize;
    let norm_topk_prob = cfg["norm_topk_prob"].as_bool().unwrap_or(true);
    let scoring_func = match cfg["scoring_func"].as_str() {
        Some("sigmoid") => MoeScoring::Sigmoid,
        _ => MoeScoring::Softmax,
    };
    Some(MoeConfig {
        num_experts,
        top_k,
        expert_intermediate_size,
        num_shared_experts,
        first_k_dense,
        norm_topk_prob,
        scoring_func,
    })
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

    // Fraction of head_dim that RoPE is applied to (Phi uses 0.4; most models 1.0).
    pub partial_rotary_factor: f64,

    // Head dimension (derived)
    pub head_dim: usize,

    /// Mixture-of-Experts config — `Some` only for MoE models (Mixtral,
    /// Qwen-MoE, DeepSeek/GLM-MoE). Drives MoE detection independently of `arch`.
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    // Raw config (for any fields we don't explicitly parse)
    #[serde(skip)]
    pub raw: serde_json::Value,
}

impl ModelInfo {
    /// True when this is a Mixture-of-Experts model (has a router + experts).
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }
}

impl ModelInfo {
    /// Build a `ModelInfo` from GGUF KV metadata.
    ///
    /// GGUF stores model config under `{arch}.block_count`, `{arch}.embedding_length`,
    /// etc.  `general.architecture` (e.g. "llama", "phi2", "qwen2") selects the prefix.
    /// Fields absent from the metadata fall back to safe defaults.
    pub fn from_gguf_metadata(
        kv: &std::collections::HashMap<String, sapient_io::GgufValue>,
    ) -> Result<Self> {
        let arch_name = kv
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");
        let p = arch_name;

        let model_type = arch_name.to_string();
        let arch = ArchType::from_model_type(arch_name);

        let g = |key: &str| kv.get(key);
        let u32v = |key: &str| g(key).and_then(|v| v.as_u32());
        let f64v = |key: &str| g(key).and_then(|v| v.as_f64());

        let vocab_size = u32v(&format!("{p}.vocab_size"))
            .or_else(|| {
                // Fall back to tokenizer token count if arch-specific key missing.
                kv.get("tokenizer.ggml.tokens").and_then(|v| match v {
                    sapient_io::GgufValue::ArrayStr(s) => Some(s.len() as u32),
                    sapient_io::GgufValue::ArrayU32(u) => Some(u.len() as u32),
                    _ => None,
                })
            })
            .unwrap_or(32000) as usize;
        let hidden_size = u32v(&format!("{p}.embedding_length")).unwrap_or(4096) as usize;
        let num_hidden_layers = u32v(&format!("{p}.block_count")).unwrap_or(32) as usize;
        let num_attention_heads = u32v(&format!("{p}.attention.head_count")).unwrap_or(32) as usize;
        let num_key_value_heads = u32v(&format!("{p}.attention.head_count_kv"))
            .unwrap_or(num_attention_heads as u32) as usize;
        let intermediate_size = u32v(&format!("{p}.feed_forward_length")).unwrap_or(11008) as usize;
        let max_position_embeddings = u32v(&format!("{p}.context_length")).unwrap_or(4096) as usize;
        let rms_norm_eps = f64v(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .or_else(|| f64v(&format!("{p}.attention.layer_norm_epsilon")))
            .unwrap_or(1e-5);
        let rope_theta = f64v(&format!("{p}.rope.freq_base")).unwrap_or(10000.0);
        let head_dim = hidden_size / num_attention_heads.max(1);
        let partial_rotary_factor = u32v(&format!("{p}.rope.dimension_count"))
            .map(|d| d as f64 / head_dim as f64)
            .unwrap_or(1.0);
        // GGUF omits a hidden_act field; infer from architecture name.
        let hidden_act = match arch_name {
            "phi2" | "gpt2" => "gelu",
            _ => "silu",
        }
        .to_owned();

        // MoE metadata (llama.cpp folds Mixtral into the `llama` arch, so this is
        // keyed off `{arch}.expert_count`, not the architecture name).
        let num_experts = u32v(&format!("{p}.expert_count")).unwrap_or(0) as usize;
        let moe = if num_experts > 0 {
            let top_k = u32v(&format!("{p}.expert_used_count")).unwrap_or(2) as usize;
            let expert_intermediate_size = u32v(&format!("{p}.expert_feed_forward_length"))
                .unwrap_or(intermediate_size as u32)
                as usize;
            let num_shared_experts =
                u32v(&format!("{p}.expert_shared_count")).unwrap_or(0) as usize;
            let first_k_dense =
                u32v(&format!("{p}.leading_dense_block_count")).unwrap_or(0) as usize;
            // llama.cpp's `expert_weights_norm` (bool); Mixtral GGUFs omit it but
            // always renormalise, so default true.
            let norm_topk_prob = g(&format!("{p}.expert_weights_norm"))
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            // `expert_gating_func`: 1 = softmax (Mixtral/Qwen), 2 = sigmoid (DeepSeek/GLM).
            let scoring_func = match u32v(&format!("{p}.expert_gating_func")) {
                Some(2) => MoeScoring::Sigmoid,
                _ => MoeScoring::Softmax,
            };
            Some(MoeConfig {
                num_experts,
                top_k,
                expert_intermediate_size,
                num_shared_experts,
                first_k_dense,
                norm_topk_prob,
                scoring_func,
            })
        } else {
            None
        };

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
            partial_rotary_factor,
            head_dim,
            moe,
            raw: serde_json::Value::Null,
        })
    }

    /// Parse from a `config.json` file on disk.
    pub fn from_config_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).context("Failed to read config.json")?;
        Self::from_json_str(&text)
    }

    /// Parse from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(json).context("Invalid config.json")?;
        let cfg = effective_config(&raw);

        let arch = detect_arch(&raw, &cfg);

        let model_type = cfg["model_type"]
            .as_str()
            .or_else(|| raw["model_type"].as_str())
            .unwrap_or("unknown")
            .to_owned();
        let vocab_size = cfg["vocab_size"]
            .as_u64()
            .or_else(|| raw["vocab_size"].as_u64())
            .unwrap_or(32000) as usize;
        let hidden_size = cfg["hidden_size"].as_u64().unwrap_or(4096) as usize;
        let num_hidden_layers = cfg["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let num_attention_heads = cfg["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        // GQA: fall back to num_attention_heads if not specified.
        let num_key_value_heads = cfg["num_key_value_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = cfg["intermediate_size"].as_u64().unwrap_or(11008) as usize;
        let max_position_embeddings =
            cfg["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;
        // Models that use LayerNorm (e.g. Phi, GPT-2) name this `layer_norm_eps` /
        // `layer_norm_epsilon` rather than `rms_norm_eps`.
        let rms_norm_eps = cfg["rms_norm_eps"]
            .as_f64()
            .or_else(|| cfg["layer_norm_eps"].as_f64())
            .or_else(|| cfg["layer_norm_epsilon"].as_f64())
            .unwrap_or(1e-5);
        let hidden_act = cfg["hidden_act"].as_str().unwrap_or("silu").to_owned();
        let rope_theta = cfg["rope_theta"].as_f64().unwrap_or(10000.0);
        let partial_rotary_factor = cfg["partial_rotary_factor"].as_f64().unwrap_or(1.0);
        let head_dim = cfg["head_dim"]
            .as_u64()
            .map(|d| d as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));
        let moe = parse_moe_config(&cfg, intermediate_size);

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
            partial_rotary_factor,
            head_dim,
            moe,
            raw: raw.clone(),
        })
    }
}

/// For VLMs and other composite models, hyperparameters live in `text_config`.
fn effective_config(raw: &serde_json::Value) -> serde_json::Value {
    let Some(text_config) = raw.get("text_config") else {
        return raw.clone();
    };
    if !text_config.is_object() {
        return raw.clone();
    }

    let mut merged = text_config.clone();
    let Some(obj) = merged.as_object_mut() else {
        return raw.clone();
    };

    for key in [
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
    ] {
        if !obj.contains_key(key) {
            if let Some(v) = raw.get(key) {
                obj.insert(key.to_string(), v.clone());
            }
        }
    }

    merged
}

fn detect_arch(raw: &serde_json::Value, cfg: &serde_json::Value) -> ArchType {
    for source in [cfg, raw] {
        if let Some(name) = architecture_names(source).first() {
            let arch = ArchType::from_hf_arch_name(name);
            if !matches!(arch, ArchType::Unknown(_)) {
                return arch;
            }
        }
        if let Some(model_type) = source["model_type"].as_str() {
            let arch = ArchType::from_model_type(model_type);
            if !matches!(arch, ArchType::Unknown(_)) {
                return arch;
            }
        }
    }

    architecture_names(raw)
        .first()
        .map(|n| ArchType::from_hf_arch_name(n))
        .unwrap_or(ArchType::Unknown("unknown".into()))
}

fn architecture_names(config: &serde_json::Value) -> Vec<String> {
    config["architectures"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

impl ArchType {
    /// Detect arch from the `model_type` field in `config.json`.
    pub fn from_model_type(model_type: &str) -> Self {
        match model_type {
            "llama" | "mistral" => Self::Llama,
            // GGUF arch strings: "phi2" (Phi-1/1.5/2), "phi3" (Phi-3/3.5/4-mini).
            "phi" | "phi2" | "phi3" => Self::Phi,
            "gemma" | "gemma2" => Self::Gemma,
            "gemma3" | "gemma3_text" => Self::Gemma3,
            "gpt2" => Self::Gpt2,
            "bert" | "roberta" => Self::Bert,
            "qwen2" | "qwen3" => Self::Qwen,
            "mixtral" => Self::Mixtral,
            "falcon" => Self::Falcon,
            "mpt" => Self::Mpt,
            "bloom" => Self::Bloom,
            "t5" => Self::T5,
            "whisper" => Self::Whisper,
            "idefics3" => Self::Llama,
            other => Self::Unknown(other.to_owned()),
        }
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

    const SMOLVLM_CONFIG: &str = r#"{
        "architectures": ["Idefics3ForConditionalGeneration"],
        "model_type": "idefics3",
        "vocab_size": 49280,
        "text_config": {
            "architectures": ["VLlama3ForCausalLM"],
            "model_type": "llama",
            "vocab_size": 49280,
            "hidden_size": 960,
            "num_hidden_layers": 32,
            "num_attention_heads": 15,
            "num_key_value_heads": 5,
            "intermediate_size": 2560,
            "max_position_embeddings": 8192,
            "head_dim": 64,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 100000.0
        },
        "vision_config": { "hidden_size": 768 }
    }"#;

    #[test]
    fn parse_smolvlm_text_config() {
        let info = ModelInfo::from_json_str(SMOLVLM_CONFIG).unwrap();
        assert_eq!(info.arch, ArchType::Llama);
        assert_eq!(info.hidden_size, 960);
        assert_eq!(info.num_hidden_layers, 32);
        assert_eq!(info.num_key_value_heads, 5);
        assert_eq!(info.head_dim, 64);
        assert_eq!(info.vocab_size, 49280);
    }

    const MIXTRAL_CONFIG: &str = r#"{
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "rope_theta": 1000000.0,
        "num_local_experts": 8,
        "num_experts_per_tok": 2
    }"#;

    #[test]
    fn parse_mixtral_moe_config() {
        let info = ModelInfo::from_json_str(MIXTRAL_CONFIG).unwrap();
        // Detected as Mixtral for safetensors, but MoE detection is via `moe`.
        assert_eq!(info.arch, ArchType::Mixtral);
        let moe = info.moe.as_ref().expect("Mixtral must parse a MoE config");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
        // Mixtral has no separate moe_intermediate_size → falls back to dense.
        assert_eq!(moe.expert_intermediate_size, 14336);
        assert_eq!(moe.num_shared_experts, 0);
        assert_eq!(moe.first_k_dense, 0);
        assert!(moe.norm_topk_prob);
        assert_eq!(moe.scoring_func, MoeScoring::Softmax);
    }

    // DeepSeek/GLM-style fine-grained MoE (shared expert, leading dense layers,
    // sigmoid gate, distinct expert intermediate) — the extension-point config.
    const DEEPSEEK_MOE_CONFIG: &str = r#"{
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "vocab_size": 129280,
        "hidden_size": 6144,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "intermediate_size": 12288,
        "max_position_embeddings": 163840,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "first_k_dense_replace": 3,
        "norm_topk_prob": true,
        "scoring_func": "sigmoid"
    }"#;

    #[test]
    fn parse_deepseek_style_moe_config() {
        let info = ModelInfo::from_json_str(DEEPSEEK_MOE_CONFIG).unwrap();
        let moe = info.moe.as_ref().expect("must parse a MoE config");
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.top_k, 8);
        assert_eq!(moe.expert_intermediate_size, 2048); // distinct from dense 12288
        assert_eq!(moe.num_shared_experts, 1);
        assert_eq!(moe.first_k_dense, 3);
        assert!(moe.norm_topk_prob);
        assert_eq!(moe.scoring_func, MoeScoring::Sigmoid);
    }

    #[test]
    fn dense_model_has_no_moe_config() {
        let info = ModelInfo::from_json_str(LLAMA_CONFIG).unwrap();
        assert!(info.moe.is_none());
        assert!(!info.is_moe());
    }
}
