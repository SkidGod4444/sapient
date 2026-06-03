//! Kokoro-82M config parsing + weight loading.
//!
//! Weights come from the offline `.pth → safetensors` conversion
//! (`scripts/convert_kokoro_to_safetensors.py`): `model.safetensors` (all module
//! weights, key names preserved) + `voices.safetensors` (each voice `[510, 256]`)
//! plus `config.json`. At load we fold every PyTorch `weight_norm` pair
//! (`.weight_g` / `.weight_v`) into a plain `.weight` so the conv/linear code
//! never renormalizes at runtime — exactly the SNAC precedent.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use sapient_core::Tensor;
use serde::Deserialize;

use super::super::snac::weight_norm_fold;

/// ISTFTNet generator/decoder hyperparameters (the `istftnet` config block).
#[derive(Debug, Clone, Deserialize)]
pub struct IstftnetConfig {
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub gen_istft_n_fft: usize,
    pub gen_istft_hop_size: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_initial_channel: usize,
}

/// PLBERT (ALBERT) hyperparameters (the `plbert` config block).
#[derive(Debug, Clone, Deserialize)]
pub struct PlbertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[allow(dead_code)]
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
}

/// Kokoro-82M `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct KokoroConfig {
    pub hidden_dim: usize,
    pub style_dim: usize,
    pub n_layer: usize,
    pub n_token: usize,
    pub max_dur: usize,
    pub text_encoder_kernel_size: usize,
    pub n_mels: usize,
    pub plbert: PlbertConfig,
    pub istftnet: IstftnetConfig,
    /// IPA-char → token-id map (used by the G2P front-end to build input ids).
    pub vocab: HashMap<String, u32>,
}

impl KokoroConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).with_context(|| format!("read {path:?}"))?;
        serde_json::from_str(&s).with_context(|| format!("parse {path:?}"))
    }
}

/// Fold all `weight_norm` pairs in `raw` into plain `*.weight` tensors. Keys
/// without a `_g`/`_v` partner pass through unchanged; `*.weight_g` is consumed.
pub fn fold_weight_norm(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
    for (k, v) in &raw {
        if let Some(stem) = k.strip_suffix(".weight_v") {
            let gkey = format!("{stem}.weight_g");
            let g = raw
                .get(&gkey)
                .ok_or_else(|| anyhow!("weight_norm: {k} has no matching {gkey}"))?;
            let folded = weight_norm_fold(v, g)
                .with_context(|| format!("folding weight_norm for {stem}"))?;
            out.insert(format!("{stem}.weight"), folded);
        } else if k.ends_with(".weight_g") {
            // consumed alongside its _v partner above
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    Ok(out)
}

/// Loaded Kokoro model assets: folded weights, config, and the voice packs
/// (each `[510, 256]`, indexed by phoneme count).
pub struct KokoroAssets {
    pub config: KokoroConfig,
    pub weights: HashMap<String, Tensor>,
    pub voices: HashMap<String, Tensor>,
}

/// Load from a directory containing `config.json`, `model.safetensors`, and
/// (optionally) `voices.safetensors`.
pub fn load_from_dir(dir: &Path) -> Result<KokoroAssets> {
    let config = KokoroConfig::from_file(&dir.join("config.json"))?;
    let raw = sapient_io::load_safetensors(&dir.join("model.safetensors"))
        .with_context(|| format!("load model.safetensors in {dir:?}"))?;
    let weights = fold_weight_norm(raw)?;
    let voices = match dir.join("voices.safetensors") {
        p if p.exists() => sapient_io::load_safetensors(&p)?,
        _ => HashMap::new(),
    };
    Ok(KokoroAssets {
        config,
        weights,
        voices,
    })
}
