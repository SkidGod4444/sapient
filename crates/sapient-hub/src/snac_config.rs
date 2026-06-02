//! SNAC neural-audio-codec configuration (decoder side).
//!
//! Parsed from a SNAC `config.json` (e.g. `hubertsiuzdak/snac_24khz`). Only the
//! fields the **decoder** needs are kept — the encoder is never run (the LM emits
//! codec tokens directly). The 24 kHz speech model has `attn_window_size = null`
//! (no attention in the decode path), so the decoder is fully convolutional.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Decoder-relevant hyperparameters of a SNAC codec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnacConfig {
    /// Output waveform sample rate (24000 for `snac_24khz`).
    pub sampling_rate: u32,
    /// Decoder base channel width (1024).
    pub decoder_dim: usize,
    /// Per-stage upsample factors, coarse→fine (e.g. `[8, 8, 4, 2]`).
    pub decoder_rates: Vec<usize>,
    /// Continuous latent dimension fed into the decoder (codebook_dim if unset).
    pub latent_dim: Option<usize>,
    /// RVQ codebook entries per level (4096).
    pub codebook_size: usize,
    /// RVQ code vector dimension (8).
    pub codebook_dim: usize,
    /// Temporal stride of each RVQ level (multi-scale; e.g. `[4, 2, 1]`).
    pub vq_strides: Vec<usize>,
    /// Whether decoder blocks include the learned-noise injection.
    pub noise: bool,
    /// Whether the upsample convs are depthwise.
    pub depthwise: bool,
    /// Local-attention window (null/None for the 24 kHz speech model → no attn).
    pub attn_window_size: Option<usize>,
}

impl SnacConfig {
    /// The `snac_24khz` defaults (used when a field is absent).
    pub fn snac_24khz() -> Self {
        Self {
            sampling_rate: 24_000,
            decoder_dim: 1024,
            decoder_rates: vec![8, 8, 4, 2],
            latent_dim: None,
            codebook_size: 4096,
            codebook_dim: 8,
            vq_strides: vec![4, 2, 1],
            noise: true,
            depthwise: true,
            attn_window_size: None,
        }
    }

    pub fn from_config_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).context("reading SNAC config.json")?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json).context("invalid SNAC config")?;
        let d = Self::snac_24khz();
        let u = |k: &str, def: usize| v[k].as_u64().map(|n| n as usize).unwrap_or(def);
        let usize_vec = |k: &str, def: &[usize]| -> Vec<usize> {
            v[k].as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|n| n as usize)
                        .collect()
                })
                .filter(|vv: &Vec<usize>| !vv.is_empty())
                .unwrap_or_else(|| def.to_vec())
        };
        Ok(Self {
            sampling_rate: v["sampling_rate"]
                .as_u64()
                .map(|n| n as u32)
                .unwrap_or(d.sampling_rate),
            decoder_dim: u("decoder_dim", d.decoder_dim),
            decoder_rates: usize_vec("decoder_rates", &d.decoder_rates),
            latent_dim: v["latent_dim"].as_u64().map(|n| n as usize),
            codebook_size: u("codebook_size", d.codebook_size),
            codebook_dim: u("codebook_dim", d.codebook_dim),
            vq_strides: usize_vec("vq_strides", &d.vq_strides),
            noise: v["noise"].as_bool().unwrap_or(d.noise),
            depthwise: v["depthwise"].as_bool().unwrap_or(d.depthwise),
            attn_window_size: v["attn_window_size"].as_u64().map(|n| n as usize),
        })
    }

    /// Number of RVQ codebooks / code levels.
    pub fn n_codebooks(&self) -> usize {
        self.vq_strides.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_snac_24khz() {
        // Trimmed hubertsiuzdak/snac_24khz config.json.
        let c = SnacConfig::from_json_str(
            r#"{
                "sampling_rate": 24000,
                "decoder_dim": 1024,
                "decoder_rates": [8, 8, 4, 2],
                "codebook_size": 4096,
                "codebook_dim": 8,
                "vq_strides": [4, 2, 1],
                "noise": true,
                "depthwise": true,
                "attn_window_size": null
            }"#,
        )
        .unwrap();
        assert_eq!(c.sampling_rate, 24_000);
        assert_eq!(c.decoder_rates, vec![8, 8, 4, 2]);
        assert_eq!(c.n_codebooks(), 3);
        assert!(c.attn_window_size.is_none()); // no attention in the 24 kHz decoder
        assert!(c.noise && c.depthwise);
    }

    #[test]
    fn defaults_fill_missing_fields() {
        let c = SnacConfig::from_json_str("{}").unwrap();
        assert_eq!(c, SnacConfig::snac_24khz());
    }
}
