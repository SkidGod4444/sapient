// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `VlaPipeline` — vision-language-action inference (SmolVLA).
//!
//! Camera image(s) + a language instruction + the robot state → a chunk of
//! future actions. Wraps [`sapient_models::forward::SmolVla`] with everything
//! around the network:
//!
//! * **images** — resized without distortion to the tower's square input
//!   (bilinear, padded on the left/top with black) and mapped to `[-1, 1]`,
//!   LeRobot's `resize_with_pad`;
//! * **language** — `"{task}\n"` tokenized with the SmolVLM2 tokenizer, no
//!   special tokens, truncated to `tokenizer_max_length`;
//! * **state / actions** — mean/std normalization when the checkpoint ships
//!   statistics under the plain LeRobot names (`observation.state.*`,
//!   `action.*`); identity otherwise. `lerobot/smolvla_base` is a pretraining
//!   checkpoint meant to be fine-tuned and has no such statistics — its
//!   actions are in the model's normalized space.
//! * **noise** — the flow-matching start sample, from a seeded generator so a
//!   run is reproducible.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use sapient_core::Tensor;
use sapient_hub::HubClient;
pub use sapient_models::forward::SmolVlaQuant;
use sapient_models::forward::{SmolVla, SmolVlaConfig};
use sapient_tokenizers::{SapientTokenizer, TokenizerOptions};

/// The default SmolVLA checkpoint.
pub const SMOLVLA_REPO: &str = "lerobot/smolvla_base";

/// LeRobot processor pipeline descriptions. Each names the safetensors file its
/// (un)normalizer step stores statistics in — the step index in that name
/// varies between checkpoints (`…step_0_unnormalizer…` vs `…step_1_…`).
const PROCESSOR_CONFIGS: [&str; 2] = ["policy_preprocessor.json", "policy_postprocessor.json"];
/// Fallback names when a checkpoint has no processor configs.
const DEFAULT_STATS_FILES: [&str; 2] = [
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
];

/// The `state_file` of every `normalizer_processor` / `unnormalizer_processor`
/// step in a LeRobot processor config.
fn stats_files_in(processor_json: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(processor_json) else {
        return Vec::new();
    };
    v["steps"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|s| {
            s["registry_name"]
                .as_str()
                .is_some_and(|n| n.ends_with("normalizer_processor"))
        })
        .filter_map(|s| s["state_file"].as_str().map(str::to_string))
        .collect()
}
const NORM_EPS: f32 = 1e-8;

/// How much of the policy runs on 8-bit (Q8_0) weights. Error figures are the
/// RMS difference of the action chunk from the f32 result over eight varied
/// observations (`smolvla_quantized_error_over_observations`); LeRobot's own
/// default precision (bf16) scores 4.6e-3 on the same observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VlaPrecision {
    /// Everything 8-bit — fastest and smallest. RMS error ~1.2e-2, about 2.5×
    /// LeRobot's own bf16 deviation.
    #[default]
    Fast,
    /// Only the action expert 8-bit (it runs 10× per chunk, so it is most of
    /// the saving); vision and VLM stay f32. RMS error 4.2e-3 — the same as
    /// LeRobot's own bf16 deviation.
    Balanced,
    /// Everything f32: reproduces the f32 LeRobot reference to ~4e-6.
    Exact,
}

impl VlaPrecision {
    pub fn quant(self) -> SmolVlaQuant {
        match self {
            Self::Fast => SmolVlaQuant::ALL,
            Self::Balanced => SmolVlaQuant {
                vision: false,
                vlm: false,
                expert: true,
                fast_math: true,
                int8_attention: false,
            },
            Self::Exact => SmolVlaQuant::NONE,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Exact => "exact",
        }
    }
}

impl std::str::FromStr for VlaPrecision {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fast" | "q8" => Ok(Self::Fast),
            "balanced" => Ok(Self::Balanced),
            "exact" | "f32" => Ok(Self::Exact),
            other => bail!("unknown precision '{other}' (expected fast, balanced or exact)"),
        }
    }
}

/// Wall-clock split of one [`VlaPipeline::predict`] call.
#[derive(Debug, Clone, Default)]
pub struct VlaTiming {
    /// SigLIP + connector over every camera image.
    pub vision_ms: f64,
    /// VLM prefix pass (builds the K/V cache the action expert reads).
    pub prefix_ms: f64,
    /// All flow-matching steps of the action expert.
    pub denoise_ms: f64,
    pub total_ms: f64,
    /// Prefix length in tokens (image + language + state).
    pub prefix_tokens: usize,
}

/// A predicted action chunk: `steps` consecutive actions of `dim` values each.
#[derive(Debug, Clone)]
pub struct ActionChunk {
    /// Row-major `[steps, dim]`.
    pub actions: Vec<f32>,
    pub steps: usize,
    pub dim: usize,
    /// True when the checkpoint's action statistics were applied (robot
    /// units); false means `actions` are in the model's normalized space.
    pub robot_units: bool,
    pub timing: VlaTiming,
}

impl ActionChunk {
    pub fn row(&self, step: usize) -> &[f32] {
        &self.actions[step * self.dim..(step + 1) * self.dim]
    }
}

struct MeanStd {
    mean: Vec<f32>,
    std: Vec<f32>,
}

fn mean_std(stats: &HashMap<String, Tensor>, key: &str) -> Option<MeanStd> {
    let mean = stats.get(&format!("{key}.mean"))?.to_f32_vec();
    let std = stats.get(&format!("{key}.std"))?.to_f32_vec();
    (mean.len() == std.len()).then_some(MeanStd { mean, std })
}

pub struct VlaPipeline {
    model: SmolVla,
    tokenizer: SapientTokenizer,
    max_lang_tokens: usize,
    state_dim: usize,
    action_dim: usize,
    state_stats: Option<MeanStd>,
    action_stats: Option<MeanStd>,
}

impl VlaPipeline {
    /// Download (or reuse cached) a SmolVLA checkpoint by Hugging Face repo id
    /// and load it. The tokenizer comes from the VLM the policy was built on
    /// (`vlm_model_name` in its config).
    ///
    /// Uses [`VlaPrecision::Fast`] (everything Q8_0); see [`VlaPrecision`] for
    /// the measured accuracy of each choice and
    /// [`from_pretrained_with`](Self::from_pretrained_with) to pick another.
    pub async fn from_pretrained(repo: &str) -> Result<Self> {
        Self::from_pretrained_with(repo, SmolVlaQuant::ALL).await
    }

    /// [`from_pretrained`](Self::from_pretrained) with an explicit precision.
    pub async fn from_pretrained_with(repo: &str, quant: SmolVlaQuant) -> Result<Self> {
        let client = HubClient::new()?;
        let files = client
            .download_files(repo, &["config.json", "model.safetensors"])
            .await
            .with_context(|| format!("downloading {repo}"))?;
        let cfg = read_config(&files[0])?;
        let vlm_repo = cfg["vlm_model_name"]
            .as_str()
            .unwrap_or("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            .to_string();
        let tok = client
            .download_files(&vlm_repo, &["tokenizer.json"])
            .await
            .with_context(|| format!("downloading the tokenizer from {vlm_repo}"))?;
        // Normalization statistics are optional. Find their file names in the
        // processor configs; fall back to the common names.
        let mut names: Vec<String> = Vec::new();
        for cfg_name in PROCESSOR_CONFIGS {
            if let Ok(p) = client.download_files(repo, &[cfg_name]).await {
                if let Ok(text) = std::fs::read_to_string(&p[0]) {
                    names.extend(stats_files_in(&text));
                }
            }
        }
        if names.is_empty() {
            names = DEFAULT_STATS_FILES.iter().map(|s| s.to_string()).collect();
        }
        names.dedup();
        let mut stats: Vec<PathBuf> = Vec::new();
        for name in &names {
            if let Ok(p) = client.download_files(repo, &[name.as_str()]).await {
                stats.extend(p);
            }
        }
        Self::from_files(&files[0], &files[1], &tok[0], &stats, quant)
    }

    /// Load from already-downloaded files. `stats` are the checkpoint's
    /// normalizer / unnormalizer safetensors (may be empty).
    pub fn from_files(
        config: &Path,
        weights: &Path,
        tokenizer: &Path,
        stats: &[PathBuf],
        quant: SmolVlaQuant,
    ) -> Result<Self> {
        let cfg = read_config(config)?;
        let usize_of = |key: &str, default: usize| -> usize {
            cfg[key].as_u64().map(|v| v as usize).unwrap_or(default)
        };
        let d = SmolVlaConfig::default();
        let model_cfg = SmolVlaConfig {
            chunk: usize_of("chunk_size", d.chunk),
            num_steps: usize_of("num_steps", d.num_steps),
            self_attn_every: usize_of("self_attn_every_n_layers", d.self_attn_every),
            min_period: cfg["min_period"].as_f64().unwrap_or(d.min_period),
            max_period: cfg["max_period"].as_f64().unwrap_or(d.max_period),
            ..d
        };
        if cfg["attention_mode"].as_str().unwrap_or("cross_attn") != "cross_attn" {
            bail!("SmolVLA: only attention_mode = cross_attn is supported");
        }
        if cfg["add_image_special_tokens"].as_bool().unwrap_or(false) {
            bail!("SmolVLA: add_image_special_tokens = true is not supported yet");
        }

        let tensors = sapient_io::load_safetensors(weights)
            .map_err(|e| anyhow!("loading {weights:?}: {e}"))?;
        let model = SmolVla::from_weights_quant(model_cfg, tensors, quant)?;
        let tokenizer = SapientTokenizer::from_file(tokenizer, TokenizerOptions::default())?;

        let feature_dim = |section: &str, key: &str, default: usize| -> usize {
            cfg[section][key]["shape"][0]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let mc = model.config();
        let state_dim = feature_dim("input_features", "observation.state", mc.max_state_dim)
            .min(mc.max_state_dim);
        let action_dim =
            feature_dim("output_features", "action", mc.max_action_dim).min(mc.max_action_dim);

        let mut all = HashMap::new();
        for p in stats {
            if let Ok(t) = sapient_io::load_safetensors(p) {
                all.extend(t);
            }
        }
        Ok(Self {
            max_lang_tokens: usize_of("tokenizer_max_length", 48),
            state_stats: mean_std(&all, "observation.state"),
            action_stats: mean_std(&all, "action"),
            model,
            tokenizer,
            state_dim,
            action_dim,
        })
    }

    /// Robot-state values the policy expects.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Values per action.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Actions per predicted chunk.
    pub fn chunk_len(&self) -> usize {
        self.model.config().chunk
    }

    /// Whether the checkpoint ships action statistics (actions come out in
    /// robot units) or not (normalized space).
    pub fn has_action_stats(&self) -> bool {
        self.action_stats.is_some()
    }

    /// Load + preprocess a camera image file.
    pub fn preprocess_image(&self, path: &Path) -> Result<Vec<f32>> {
        let img = image::open(path).with_context(|| format!("opening image {path:?}"))?;
        Ok(self.preprocess_rgb(&img.to_rgb8()))
    }

    /// Preprocess an RGB frame (e.g. straight from a camera).
    pub fn preprocess_rgb(&self, rgb: &image::RgbImage) -> Vec<f32> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        resize_with_pad(rgb.as_raw(), w, h, self.model.image_size())
    }

    /// `"{task}\n"` → token ids (no special tokens, truncated).
    pub fn tokenize(&self, task: &str) -> Result<Vec<u32>> {
        let text = if task.ends_with('\n') {
            task.to_string()
        } else {
            format!("{task}\n")
        };
        let mut ids = self.tokenizer.encode_ids(&text, false)?;
        ids.truncate(self.max_lang_tokens);
        Ok(ids)
    }

    /// [`preprocess_image`](Self::preprocess_image) from encoded bytes
    /// (PNG/JPEG/…) — the server decodes frames in memory.
    pub fn preprocess_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let img = image::load_from_memory(bytes).context("decoding image bytes")?;
        Ok(self.preprocess_rgb(&img.to_rgb8()))
    }

    /// Flow-matching steps the checkpoint was configured with.
    pub fn default_steps(&self) -> usize {
        self.model.config().num_steps
    }

    /// The start noise [`predict`](Self::predict) draws for `seed`
    /// (`[chunk, max_action_dim]`).
    pub fn noise_for_seed(&self, seed: u64) -> Vec<f32> {
        let mc = self.model.config();
        gaussian_noise(mc.chunk * mc.max_action_dim, seed)
    }

    /// Predict one action chunk; the start noise is drawn from `seed`.
    pub fn predict(
        &self,
        images: &[Vec<f32>],
        task: &str,
        state: &[f32],
        seed: u64,
    ) -> Result<ActionChunk> {
        let mc = self.model.config();
        let noise = gaussian_noise(mc.chunk * mc.max_action_dim, seed);
        self.predict_with_noise(images, task, state, &noise)
    }

    /// [`predict`](Self::predict) with an explicit number of flow-matching
    /// steps (`None` = the checkpoint's own). Fewer steps are proportionally
    /// faster in the denoise stage and give a coarser chunk.
    pub fn predict_steps(
        &self,
        images: &[Vec<f32>],
        task: &str,
        state: &[f32],
        seed: u64,
        steps: Option<usize>,
    ) -> Result<ActionChunk> {
        let mc = self.model.config();
        let noise = gaussian_noise(mc.chunk * mc.max_action_dim, seed);
        self.run(
            images,
            task,
            state,
            &noise,
            steps.unwrap_or(mc.num_steps),
            &[],
        )
    }

    /// [`predict_steps`](Self::predict_steps) with explicit start noise
    /// `[chunk, max_action_dim]` instead of a seed — for comparing against a
    /// reference implementation given the same noise.
    pub fn predict_noise_steps(
        &self,
        images: &[Vec<f32>],
        task: &str,
        state: &[f32],
        noise: &[f32],
        steps: Option<usize>,
    ) -> Result<ActionChunk> {
        let mc = self.model.config();
        if noise.len() != mc.chunk * mc.max_action_dim {
            bail!(
                "noise must be {} × {} values, got {}",
                mc.chunk,
                mc.max_action_dim,
                noise.len()
            );
        }
        self.run(
            images,
            task,
            state,
            noise,
            steps.unwrap_or(mc.num_steps),
            &[],
        )
    }

    /// [`predict_noise_steps`](Self::predict_noise_steps) with the first rows
    /// of the chunk frozen to `queued` — rows of [`action_dim`](Self::action_dim)
    /// values in the same units this pipeline returns (robot units when the
    /// checkpoint has action statistics). Pass the actions still queued for
    /// execution: the new chunk then continues them instead of starting an
    /// unrelated plan (see `SmolVla::sample_actions_inpaint`). The returned
    /// chunk's first rows equal `queued`.
    pub fn predict_continuing(
        &self,
        images: &[Vec<f32>],
        task: &str,
        state: &[f32],
        noise: &[f32],
        steps: Option<usize>,
        queued: &[f32],
    ) -> Result<ActionChunk> {
        let mc = self.model.config();
        if noise.len() != mc.chunk * mc.max_action_dim {
            bail!(
                "noise must be {} × {} values, got {}",
                mc.chunk,
                mc.max_action_dim,
                noise.len()
            );
        }
        let dim = self.action_dim;
        if queued.len() % dim != 0 || queued.len() > mc.chunk * dim {
            bail!(
                "queued actions must be rows of {dim} values, at most {} rows",
                mc.chunk
            );
        }
        // The inverse of the un-normalization in `run`, then zero-padded to
        // the model's action width (padding is zero in training too).
        let mut frozen = vec![0f32; queued.len() / dim * mc.max_action_dim];
        for (row, out) in queued
            .chunks_exact(dim)
            .zip(frozen.chunks_exact_mut(mc.max_action_dim))
        {
            out[..dim].copy_from_slice(row);
            if let Some(s) = &self.action_stats {
                for ((v, m), sd) in out.iter_mut().zip(&s.mean).zip(&s.std) {
                    *v = (*v - m) / (sd + NORM_EPS);
                }
            }
        }
        self.run(
            images,
            task,
            state,
            noise,
            steps.unwrap_or(mc.num_steps),
            &frozen,
        )
    }

    /// [`predict`](Self::predict) with explicit start noise
    /// `[chunk, max_action_dim]` (validation against a reference run).
    pub fn predict_with_noise(
        &self,
        images: &[Vec<f32>],
        task: &str,
        state: &[f32],
        noise: &[f32],
    ) -> Result<ActionChunk> {
        self.run(
            images,
            task,
            state,
            noise,
            self.model.config().num_steps,
            &[],
        )
    }

    fn run(
        &self,
        images: &[Vec<f32>],
        task: &str,
        state: &[f32],
        noise: &[f32],
        steps: usize,
        frozen: &[f32],
    ) -> Result<ActionChunk> {
        if images.is_empty() {
            bail!("SmolVLA needs at least one camera image");
        }
        let mc = self.model.config();
        let mut state = state.to_vec();
        if let Some(s) = &self.state_stats {
            for ((v, m), sd) in state.iter_mut().zip(&s.mean).zip(&s.std) {
                *v = (*v - m) / (sd + NORM_EPS);
            }
        }
        let lang = self.tokenize(task)?;

        let t0 = Instant::now();
        let scale = (mc.vlm_hidden as f32).sqrt();
        let mut image_embs = Vec::new();
        for px in images {
            image_embs.extend(self.model.embed_image(px)?.into_iter().map(|v| v * scale));
        }
        let t1 = Instant::now();
        let mut embs = image_embs;
        embs.extend(self.model.embed_language_and_state(&lang, &state)?);
        let cache = self.model.prefix_cache(&embs)?;
        let t2 = Instant::now();
        let x = self
            .model
            .sample_actions_inpaint(&cache, noise, steps, frozen)?;
        let t3 = Instant::now();

        let dim = self.action_dim;
        let mut actions = Vec::with_capacity(mc.chunk * dim);
        for row in x.chunks_exact(mc.max_action_dim) {
            actions.extend_from_slice(&row[..dim]);
        }
        if let Some(s) = &self.action_stats {
            for row in actions.chunks_exact_mut(dim) {
                for ((v, m), sd) in row.iter_mut().zip(&s.mean).zip(&s.std) {
                    *v = *v * (sd + NORM_EPS) + m;
                }
            }
        }
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        Ok(ActionChunk {
            actions,
            steps: mc.chunk,
            dim,
            robot_units: self.action_stats.is_some(),
            timing: VlaTiming {
                vision_ms: ms(t0, t1),
                prefix_ms: ms(t1, t2),
                denoise_ms: ms(t2, t3),
                total_ms: ms(t0, t3),
                prefix_tokens: cache.n,
            },
        })
    }
}

fn read_config(path: &Path) -> Result<serde_json::Value> {
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?,
    )?;
    match cfg["type"].as_str() {
        Some("smolvla") => Ok(cfg),
        other => bail!("not a SmolVLA policy config (type = {other:?})"),
    }
}

/// LeRobot's `resize_with_pad` + SigLIP range: scale an interleaved RGB8 image
/// to fit `size × size` without distortion (bilinear, `align_corners = false`,
/// no antialiasing — `torch.nn.functional.interpolate`), pad the **left and
/// top** with black, map `[0, 1]` → `[-1, 1]`. Output is channel-major
/// `[3, size, size]`.
fn resize_with_pad(rgb: &[u8], w: usize, h: usize, size: usize) -> Vec<f32> {
    let ratio = (w as f64 / size as f64).max(h as f64 / size as f64);
    let (rw, rh) = if w == size && h == size {
        (size, size)
    } else {
        (
            ((w as f64 / ratio) as usize).clamp(1, size),
            ((h as f64 / ratio) as usize).clamp(1, size),
        )
    };
    let (pad_x, pad_y) = (size - rw, size - rh);
    // Padding is 0 in [0, 1] space → −1 after the range map.
    let mut out = vec![-1.0f32; 3 * size * size];
    let (sx, sy) = (w as f32 / rw as f32, h as f32 / rh as f32);
    let at = |x: usize, y: usize, c: usize| rgb[(y * w + x) * 3 + c] as f32 / 255.0;
    for oy in 0..rh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy as usize).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..rw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx as usize).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let top = at(x0, y0, c) * (1.0 - wx) + at(x1, y0, c) * wx;
                let bot = at(x0, y1, c) * (1.0 - wx) + at(x1, y1, c) * wx;
                let v = top * (1.0 - wy) + bot * wy;
                out[c * size * size + (oy + pad_y) * size + ox + pad_x] = v * 2.0 - 1.0;
            }
        }
    }
    out
}

/// `n` standard-normal samples from a seeded generator (SplitMix64 +
/// Box–Muller). Deterministic across platforms; not the same stream as
/// PyTorch's, so a seed here does not reproduce a LeRobot run — pass explicit
/// noise for that.
fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    // Uniform in (0, 1]: never 0, so ln is finite.
    let mut unit = || ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (unit(), unit());
        let r = (-2.0 * u1.ln()).sqrt();
        let a = 2.0 * std::f64::consts::PI * u2;
        out.push((r * a.cos()) as f32);
        out.push((r * a.sin()) as f32);
    }
    out.truncate(n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_file_names_come_from_processor_configs() {
        let json = r#"{"name":"policy_postprocessor","steps":[
            {"registry_name":"unnormalizer_processor","config":{},
             "state_file":"policy_postprocessor_step_1_unnormalizer_processor.safetensors"},
            {"registry_name":"device_processor","config":{}}]}"#;
        assert_eq!(
            stats_files_in(json),
            vec!["policy_postprocessor_step_1_unnormalizer_processor.safetensors"]
        );
        assert!(stats_files_in("not json").is_empty());
    }

    #[test]
    fn resize_same_size_is_a_range_map() {
        let rgb = [0u8, 255, 51, 255, 0, 102, 10, 20, 30, 40, 50, 60];
        let out = resize_with_pad(&rgb, 2, 2, 2);
        // channel-major: R plane first
        assert_eq!(out[0], -1.0);
        assert_eq!(out[1], 1.0);
        assert!((out[4] - 1.0).abs() < 1e-6); // G of pixel 0
        assert!((out[8] - (51.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-6); // B of pixel 0
    }

    #[test]
    fn wide_image_is_padded_on_top() {
        // 4×2 white image into 4×4: rows 0-1 are padding (−1), rows 2-3 white.
        let rgb = vec![255u8; 4 * 2 * 3];
        let out = resize_with_pad(&rgb, 4, 2, 4);
        for c in 0..3 {
            for y in 0..4 {
                for x in 0..4 {
                    let want = if y < 2 { -1.0 } else { 1.0 };
                    assert_eq!(out[c * 16 + y * 4 + x], want, "c{c} y{y} x{x}");
                }
            }
        }
    }

    #[test]
    fn tall_image_is_padded_on_the_left() {
        let rgb = vec![255u8; 2 * 4 * 3];
        let out = resize_with_pad(&rgb, 2, 4, 4);
        for y in 0..4 {
            assert_eq!(&out[y * 4..y * 4 + 4], &[-1.0, -1.0, 1.0, 1.0]);
        }
    }

    /// 2 → 4 bilinear upscale with `align_corners = false`: source coordinates
    /// −0.25 (clamped to 0), 0.25, 0.75, 1.25 (clamped) → 0, ¼, ¾, 1 of the way.
    #[test]
    fn bilinear_upscale_matches_torch_convention() {
        let rgb = [0u8, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255];
        let out = resize_with_pad(&rgb, 2, 2, 4);
        let row: Vec<f32> = out[..4].iter().map(|v| (v + 1.0) / 2.0).collect();
        for (got, want) in row.iter().zip([0.0, 0.25, 0.75, 1.0]) {
            assert!((got - want).abs() < 1e-6, "{row:?}");
        }
    }

    /// Red plane of two non-square cases, values from LeRobot's
    /// `resize_with_pad(img, 4, 4, pad_value=0) * 2 - 1` on the same pixels
    /// (`u8[i] = (53·i + 17) mod 256`, interleaved RGB).
    #[test]
    fn resize_matches_lerobot_reference() {
        let cases: [(usize, usize, [f32; 16]); 2] = [
            (
                5,
                3,
                [
                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.657843, 0.148039, 0.138235,
                    0.191177, -0.340196, 0.465686, -0.485294, 0.320588,
                ],
            ),
            (
                3,
                7,
                [
                    -1.0, -1.0, -1.0, 0.277451, -1.0, -1.0, -1.0, -0.202941, -1.0, -1.0, -1.0,
                    -0.683333, -1.0, -1.0, -1.0, 0.091177,
                ],
            ),
        ];
        for (w, h, want) in cases {
            let rgb: Vec<u8> = (0..w * h * 3)
                .map(|i| ((i * 53 + 17) % 256) as u8)
                .collect();
            let out = resize_with_pad(&rgb, w, h, 4);
            for (i, (g, r)) in out[..16].iter().zip(want).enumerate() {
                assert!((g - r).abs() < 2e-6, "{w}x{h} element {i}: {g} vs {r}");
            }
        }
    }

    #[test]
    fn noise_is_deterministic_and_standard_normal() {
        let a = gaussian_noise(20_000, 7);
        assert_eq!(a, gaussian_noise(20_000, 7));
        assert_ne!(a[..8], gaussian_noise(8, 8)[..]);
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / a.len() as f32;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "{mean} {var}"
        );
    }
}
