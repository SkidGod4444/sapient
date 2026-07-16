// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `VlmPipeline` — vision-language inference (SmolVLM / Idefics3 family).
//!
//! Architecture: a SigLIP vision tower + pixel-shuffle connector
//! ([`SiglipVision`]) produces visual token embeddings; the text backbone is a
//! stock Llama-family model (SmolVLM-256M's is SmolLM2-135M) that runs on the
//! existing [`LlamaForward`] engine. The only new text-side mechanism is
//! **embedding splicing**: the prompt contains `<image>` placeholder tokens
//! whose embedding rows are overwritten with the vision embeddings before one
//! cache-building prefill (`forward_logits_embeds`); decode steps are the
//! normal token-id path against the same KV cache.
//!
//! Prompt protocol (Idefics3 processor, single unsplit image — the v1 scope):
//! `<|im_start|>User:<fake_token_around_image><global-img><image>×64<fake_token_around_image>{question}<end_of_utterance>\nAssistant:`
//! Images are squash-resized to the tower's square input (512×512) and
//! normalized with mean/std 0.5 (SigLIP convention).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use sapient_core::Tensor;
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_hub::HubClient;
use sapient_models::forward::{Gemma3Forward, LlamaForward, SiglipConfig, SiglipVision};
use sapient_tokenizers::{SapientTokenizer, TokenizerOptions};

/// The text backbone behind a VLM — Llama-family (SmolVLM/Idefics3) or
/// Gemma3 (gemma-3-4b multimodal, MedGemma).
enum VlmTextEngine {
    Llama(Box<LlamaForward>),
    Gemma3(Box<Gemma3Forward>),
}

impl VlmTextEngine {
    /// Input embeddings `[1, seq, hidden]` — Gemma3's are pre-scaled ×√hidden
    /// (its convention; splicing over rows matches transformers either way).
    fn token_embeddings(&self, ids: &[u32]) -> Result<Tensor> {
        match self {
            Self::Llama(e) => e.token_embeddings(ids),
            Self::Gemma3(e) => e.token_embeddings_scaled(ids),
        }
    }

    fn forward_logits_embeds(&mut self, embeds: Tensor, use_cache: bool) -> Result<Vec<f32>> {
        match self {
            Self::Llama(e) => e.forward_logits_embeds(embeds, use_cache),
            Self::Gemma3(e) => e.forward_logits_embeds(embeds, use_cache),
        }
    }

    fn forward_logits(&mut self, ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        match self {
            Self::Llama(e) => e.forward_logits(ids, use_cache),
            Self::Gemma3(e) => e.forward_logits(ids, use_cache),
        }
    }

    fn reset_cache(&mut self) {
        match self {
            Self::Llama(e) => e.reset_cache(),
            Self::Gemma3(e) => e.reset_cache(),
        }
    }
}

/// Which multimodal protocol the checkpoint speaks.
enum VlmFlavor {
    /// Idefics3/SmolVLM: pixel-shuffle connector, `<fake_token_around_image>`.
    Idefics3,
    /// Gemma3 multimodal (incl. MedGemma): avg-pool 4×4 + zero-centered
    /// RMSNorm + raw-matmul projector, `<start_of_image>` + 256 soft tokens.
    Gemma3 {
        /// `multi_modal_projector.*` weights (norm + projection matrix).
        projector: std::collections::HashMap<String, Tensor>,
        mm_tokens: usize,
    },
}

/// A loaded vision-language model: vision tower + text engine + tokenizer.
pub struct VlmPipeline {
    vision: SiglipVision,
    engine: VlmTextEngine,
    flavor: VlmFlavor,
    tokenizer: SapientTokenizer,
    image_token_id: u32,
    eos_ids: Vec<u32>,
}

impl VlmPipeline {
    /// Download (or reuse cached) and load an Idefics3-family repo.
    pub async fn from_pretrained(model: &str) -> Result<Self> {
        let repo = sapient_hub::registry::resolve_vlm_repo(model);
        let client = HubClient::new()?;
        let base = client
            .download_files(&repo, &["config.json", "tokenizer.json"])
            .await
            .with_context(|| format!("downloading {repo}"))?;
        // Single-file or sharded checkpoint.
        let weights: Vec<std::path::PathBuf> =
            match client.download_files(&repo, &["model.safetensors"]).await {
                Ok(w) => w,
                Err(_) => {
                    let idx = client
                        .download_files(&repo, &["model.safetensors.index.json"])
                        .await
                        .with_context(|| format!("{repo}: no model.safetensors or index"))?;
                    let index: serde_json::Value =
                        serde_json::from_str(&std::fs::read_to_string(&idx[0])?)?;
                    let mut shards: Vec<String> = index["weight_map"]
                        .as_object()
                        .map(|m| {
                            m.values()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    shards.sort();
                    shards.dedup();
                    let refs: Vec<&str> = shards.iter().map(String::as_str).collect();
                    client.download_files(&repo, &refs).await?
                }
            };
        Self::from_files(&base[0], &base[1], &weights)
    }

    /// Load from already-downloaded files.
    pub fn from_files(
        config: &Path,
        tokenizer: &Path,
        weights: &[std::path::PathBuf],
    ) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(config)?).context("config.json")?;
        match cfg["model_type"].as_str() {
            Some("idefics3") => Self::from_files_idefics3(&cfg, tokenizer, weights),
            Some("gemma3") => Self::from_files_gemma3(&cfg, tokenizer, weights),
            other => anyhow::bail!(
                "VLM path supports Idefics3/SmolVLM and Gemma3 multimodal; got model_type {other:?}"
            ),
        }
    }

    fn from_files_idefics3(
        cfg: &serde_json::Value,
        tokenizer: &Path,
        weights: &[std::path::PathBuf],
    ) -> Result<Self> {
        let image_token_id = cfg["image_token_id"]
            .as_u64()
            .ok_or_else(|| anyhow!("config missing image_token_id"))?
            as u32;
        let scale_factor = cfg["scale_factor"].as_u64().unwrap_or(4) as usize;

        let v = &cfg["vision_config"];
        let vision_cfg = SiglipConfig {
            hidden: v["hidden_size"].as_u64().unwrap_or(768) as usize,
            layers: v["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
            heads: v["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            intermediate: v["intermediate_size"].as_u64().unwrap_or(3072) as usize,
            image_size: v["image_size"].as_u64().unwrap_or(512) as usize,
            patch: v["patch_size"].as_u64().unwrap_or(16) as usize,
            scale_factor,
            text_hidden: cfg["text_config"]["hidden_size"].as_u64().unwrap_or(576) as usize,
        };

        let t = &cfg["text_config"];
        let hidden = t["hidden_size"].as_u64().unwrap_or(576) as usize;
        let heads = t["num_attention_heads"].as_u64().unwrap_or(9) as usize;
        let info = ModelInfo {
            arch: ArchType::Llama,
            model_type: "llama".into(),
            vocab_size: t["vocab_size"].as_u64().unwrap_or(49280) as usize,
            hidden_size: hidden,
            num_hidden_layers: t["num_hidden_layers"].as_u64().unwrap_or(30) as usize,
            num_attention_heads: heads,
            num_key_value_heads: t["num_key_value_heads"].as_u64().unwrap_or(3) as usize,
            intermediate_size: t["intermediate_size"].as_u64().unwrap_or(1536) as usize,
            max_position_embeddings: t["max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
            rms_norm_eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            hidden_act: "silu".into(),
            rope_theta: t["rope_theta"].as_f64().unwrap_or(100_000.0),
            partial_rotary_factor: 1.0,
            head_dim: hidden / heads,
            moe: None,
            raw: serde_json::Value::Null,
        };

        // ── split the checkpoint: vision+connector vs text backbone ─────────
        let mut all: HashMap<String, Tensor> = HashMap::new();
        for w in weights {
            all.extend(
                sapient_io::safetensors::SafetensorsLoader::load(w)
                    .with_context(|| format!("loading {w:?}"))?,
            );
        }
        let mut vision_w: HashMap<String, Tensor> = HashMap::new();
        let mut text_w: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in all {
            if name.contains("vision_model") || name.contains("connector") {
                let tensor = maybe_quantize_vision(&name, tensor);
                vision_w.insert(name, tensor);
            } else if let Some(rest) = name.strip_prefix("model.text_model.") {
                // → the names LlamaForward expects ("model.layers.N...", etc.)
                text_w.insert(format!("model.{rest}"), tensor);
            } else {
                text_w.insert(name, tensor); // lm_head.weight
            }
        }

        let vision = SiglipVision::new(vision_cfg, vision_w)?;
        let engine = LlamaForward::from_weights(info, text_w)?;

        let tokenizer = SapientTokenizer::from_file(
            tokenizer,
            TokenizerOptions {
                add_bos: false,
                ..Default::default()
            },
        )?;
        let mut eos_ids = Vec::new();
        for tok in ["<end_of_utterance>", "<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = tokenizer.token_id(tok) {
                eos_ids.push(id);
            }
        }

        Ok(Self {
            vision,
            engine: VlmTextEngine::Llama(Box::new(engine)),
            flavor: VlmFlavor::Idefics3,
            tokenizer,
            image_token_id,
            eos_ids,
        })
    }

    fn from_files_gemma3(
        cfg: &serde_json::Value,
        tokenizer: &Path,
        weights: &[std::path::PathBuf],
    ) -> Result<Self> {
        let image_token_id = cfg["image_token_index"]
            .as_u64()
            .ok_or_else(|| anyhow!("gemma3 config missing image_token_index"))?
            as u32;
        let mm_tokens = cfg["mm_tokens_per_image"].as_u64().unwrap_or(256) as usize;

        let v = &cfg["vision_config"];
        let vision_cfg = SiglipConfig {
            hidden: v["hidden_size"].as_u64().unwrap_or(1152) as usize,
            layers: v["num_hidden_layers"].as_u64().unwrap_or(27) as usize,
            heads: v["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            intermediate: v["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            image_size: v["image_size"].as_u64().unwrap_or(896) as usize,
            patch: v["patch_size"].as_u64().unwrap_or(14) as usize,
            scale_factor: 1, // Gemma3 pools instead of pixel-shuffling
            text_hidden: cfg["text_config"]["hidden_size"].as_u64().unwrap_or(2560) as usize,
        };

        let mut all: HashMap<String, Tensor> = HashMap::new();
        for w in weights {
            all.extend(
                sapient_io::safetensors::SafetensorsLoader::load(w)
                    .with_context(|| format!("loading {w:?}"))?,
            );
        }
        let mut vision_w: HashMap<String, Tensor> = HashMap::new();
        let mut projector: HashMap<String, Tensor> = HashMap::new();
        let mut text_w: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in all {
            if name.contains("vision_tower") {
                let tensor = maybe_quantize_vision(&name, tensor);
                vision_w.insert(name, tensor);
            } else if name.contains("multi_modal_projector") {
                projector.insert(name, tensor);
            } else {
                text_w.insert(name, tensor); // language_model.* — engine strips
            }
        }

        // ModelInfo from the composite config (nested text_config handled by
        // the standard parser path — build it via the raw JSON).
        let info = ModelInfo::from_json_str(&cfg.to_string())?;
        let engine = Gemma3Forward::from_weights(info, text_w)?;
        let vision = SiglipVision::with_prefix(vision_cfg, vision_w, "vision_tower.vision_model")?;

        let tokenizer = SapientTokenizer::from_file(
            tokenizer,
            TokenizerOptions {
                add_bos: false,
                ..Default::default()
            },
        )?;
        let mut eos_ids = Vec::new();
        for tok in ["<end_of_turn>", "<eos>"] {
            if let Some(id) = tokenizer.token_id(tok) {
                eos_ids.push(id);
            }
        }

        Ok(Self {
            vision,
            engine: VlmTextEngine::Gemma3(Box::new(engine)),
            flavor: VlmFlavor::Gemma3 {
                projector,
                mm_tokens,
            },
            tokenizer,
            image_token_id,
            eos_ids,
        })
    }

    /// Load + preprocess an image file: squash-resize to the tower's square
    /// input, RGB, `(x/255 − 0.5)/0.5`, channel-major `[3, S, S]`.
    pub fn preprocess_image(&self, path: &Path) -> Result<Vec<f32>> {
        let img = image::open(path).with_context(|| format!("opening image {path:?}"))?;
        Ok(normalize_image(img, self.vision.config().image_size as u32))
    }

    /// [`preprocess_image`](Self::preprocess_image) from encoded bytes
    /// (PNG/JPEG/…) instead of a file — the server's data-URI path (Phase 12.3)
    /// decodes images in memory and never touches disk.
    pub fn preprocess_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let img = image::load_from_memory(bytes).context("decoding image bytes")?;
        Ok(normalize_image(img, self.vision.config().image_size as u32))
    }

    /// Encode preprocessed pixels to visual-token embeddings (probes/tools).
    pub fn encode_image_embeddings(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        self.vision.encode(pixels)
    }

    /// One vision-language turn: describe/answer `question` about the image.
    /// Greedy decode, up to `max_new` tokens. Returns the reply text.
    pub fn answer(&mut self, image: &Path, question: &str, max_new: usize) -> Result<String> {
        Ok(self.answer_with_stats(image, question, max_new)?.0)
    }

    /// [`answer`](Self::answer) + per-stage timing (the numbers `sapient see`
    /// prints): vision tower+connector, prefill, decode.
    pub fn answer_with_stats(
        &mut self,
        image: &Path,
        question: &str,
        max_new: usize,
    ) -> Result<(String, VlmStats)> {
        let t_vision = std::time::Instant::now();
        let pixels = self.preprocess_image(image)?;
        self.answer_pixels_with_stats(pixels, t_vision, question, max_new)
    }

    /// [`answer_with_stats`](Self::answer_with_stats) from encoded image bytes
    /// (the server's data-URI path).
    pub fn answer_bytes_with_stats(
        &mut self,
        image_bytes: &[u8],
        question: &str,
        max_new: usize,
    ) -> Result<(String, VlmStats)> {
        let t_vision = std::time::Instant::now();
        let pixels = self.preprocess_image_bytes(image_bytes)?;
        self.answer_pixels_with_stats(pixels, t_vision, question, max_new)
    }

    /// Shared turn body. `t_vision` was started before preprocessing so
    /// `vision_ms` keeps meaning preprocess + tower + connector.
    fn answer_pixels_with_stats(
        &mut self,
        pixels: Vec<f32>,
        t_vision: std::time::Instant,
        question: &str,
        max_new: usize,
    ) -> Result<(String, VlmStats)> {
        // 1. Vision: pixels → visual token embeddings in text space.
        let text_hidden = self.vision.config().text_hidden;
        let (vis, n_vis, prompt) = match &self.flavor {
            VlmFlavor::Idefics3 => {
                let vis = self.vision.encode(&pixels)?;
                let n_vis = self.vision.config().n_visual_tokens();
                let img_seq = format!(
                    "<fake_token_around_image><global-img>{}<fake_token_around_image>",
                    "<image>".repeat(n_vis)
                );
                let prompt =
                    format!("<|im_start|>User:{img_seq}{question}<end_of_utterance>\nAssistant:");
                (vis, n_vis, prompt)
            }
            VlmFlavor::Gemma3 {
                projector,
                mm_tokens,
            } => {
                let feats = self.vision.encode_features(&pixels)?;
                let vis = gemma3_project(
                    &feats,
                    self.vision.config(),
                    projector,
                    *mm_tokens,
                    text_hidden,
                )?;
                // Processor protocol: "\n\n<start_of_image>" + soft tokens +
                // "<end_of_image>\n\n" inside the user turn.
                let img_seq = format!(
                    "\n\n<start_of_image>{}<end_of_image>\n\n",
                    "<image_soft_token>".repeat(*mm_tokens)
                );
                let prompt = format!(
                    "<bos><start_of_turn>user\n{img_seq}{question}<end_of_turn>\n<start_of_turn>model\n"
                );
                (vis, *mm_tokens, prompt)
            }
        };

        let vision_ms = t_vision.elapsed().as_millis();
        let ids = self.tokenizer.encode_ids(&prompt, false)?;

        // 2. Splice: overwrite the image-token rows with the vision embeddings.
        let embeds = self.engine.token_embeddings(&ids)?;
        let mut ev = embeds.to_f32_vec();
        let img_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.image_token_id)
            .map(|(i, _)| i)
            .collect();
        if img_positions.len() != n_vis {
            anyhow::bail!(
                "expected {n_vis} image tokens in the prompt, found {}",
                img_positions.len()
            );
        }
        for (vi, &pos) in img_positions.iter().enumerate() {
            ev[pos * text_hidden..(pos + 1) * text_hidden]
                .copy_from_slice(&vis[vi * text_hidden..(vi + 1) * text_hidden]);
        }
        let embeds = Tensor::from_f32(&ev, sapient_core::Shape::new([1, ids.len(), text_hidden]))
            .map_err(|e| anyhow!("{e}"))?;

        // 3. Prefill (builds the KV cache), then greedy decode on token ids.
        let prompt_tokens = ids.len();
        let t_prefill = std::time::Instant::now();
        let mut logits = self.engine.forward_logits_embeds(embeds, true)?;
        let prefill_ms = t_prefill.elapsed().as_millis();
        let t_decode = std::time::Instant::now();
        let mut out_ids: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let next = argmax(&logits);
            if self.eos_ids.contains(&next) {
                break;
            }
            out_ids.push(next);
            logits = self.engine.forward_logits(&[next], true)?;
        }
        let decode_ms = t_decode.elapsed().as_millis();
        self.engine.reset_cache();
        let text = self.tokenizer.decode(&out_ids, true)?.trim().to_string();
        Ok((
            text,
            VlmStats {
                vision_ms,
                prompt_tokens,
                prefill_ms,
                gen_tokens: out_ids.len(),
                decode_ms,
            },
        ))
    }
}

/// Squash-resize to the tower's square input, RGB, `(x/255 − 0.5)/0.5`,
/// channel-major `[3, S, S]` (SigLIP convention). Shared by the file and
/// in-memory-bytes preprocessing entry points so both are bit-identical.
fn normalize_image(img: image::DynamicImage, s: u32) -> Vec<f32> {
    let img = img.resize_exact(s, s, image::imageops::FilterType::Lanczos3);
    let rgb = img.to_rgb8();
    let s = s as usize;
    let mut out = vec![0.0f32; 3 * s * s];
    for (x, y, p) in rgb.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        for c in 0..3 {
            out[c * s * s + y * s + x] = (p.0[c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    out
}

/// Online-quantize an eligible vision-tower linear to Q8_0 (the exact rule the
/// text engines use — conv/embed/norm/bias weights pass through untouched).
/// Near-lossless, and it routes the tower's matmuls onto the parallel W8A8
/// kernels instead of f32 GEMM.
fn maybe_quantize_vision(name: &str, t: Tensor) -> Tensor {
    // Q8_0 needs in-features % 32 == 0 (Gemma3's vision MLP is 4304-wide —
    // 4304 % 32 = 16 — so its fc2 stays f32 on the parallel SGEMM path).
    let k_ok = t.shape().dims().last().is_some_and(|d| d % 32 == 0);
    if k_ok && sapient_models::forward::common::should_quantize_online(name, &t) {
        sapient_models::forward::common::quantize_tensor_to_q8_0(t)
    } else {
        t
    }
}

/// Per-stage timing for one [`VlmPipeline::answer_with_stats`] turn.
#[derive(Debug, Clone, Copy)]
pub struct VlmStats {
    /// Image preprocessing + vision tower + connector.
    pub vision_ms: u128,
    /// Prompt length in tokens (text + image tokens).
    pub prompt_tokens: usize,
    /// Prefill (cache-building forward over the spliced embeddings).
    pub prefill_ms: u128,
    /// Generated token count.
    pub gen_tokens: usize,
    /// Total decode wall time.
    pub decode_ms: u128,
}

impl VlmStats {
    pub fn decode_tps(&self) -> f32 {
        if self.decode_ms == 0 {
            return 0.0;
        }
        self.gen_tokens as f32 / (self.decode_ms as f32 / 1000.0)
    }
}

/// Gemma3 multimodal connector: tower features `[side², hidden]` → 4×4 avg
/// pool over the patch grid → zero-centered RMSNorm (`mm_soft_emb_norm`,
/// Gemma convention `x/rms·(1+w)`) → raw matmul with
/// `mm_input_projection_weight [hidden, text_hidden]` → `[mm_tokens, text_hidden]`.
fn gemma3_project(
    feats: &[f32],
    vcfg: &SiglipConfig,
    projector: &std::collections::HashMap<String, Tensor>,
    mm_tokens: usize,
    text_hidden: usize,
) -> Result<Vec<f32>> {
    let side = vcfg.n_patches_side(); // 64
    let c = vcfg.hidden; // 1152
    let out_side = (mm_tokens as f64).sqrt() as usize; // 16
    if out_side * out_side != mm_tokens {
        anyhow::bail!("mm_tokens_per_image {mm_tokens} is not a square");
    }
    let k = side / out_side; // 4
                             // Average-pool the patch grid.
    let mut pooled = vec![0.0f32; mm_tokens * c];
    for oy in 0..out_side {
        for ox in 0..out_side {
            let dst = (oy * out_side + ox) * c;
            for dy in 0..k {
                for dx in 0..k {
                    let src = ((oy * k + dy) * side + (ox * k + dx)) * c;
                    for ci in 0..c {
                        pooled[dst + ci] += feats[src + ci];
                    }
                }
            }
            let inv = 1.0 / (k * k) as f32;
            for ci in 0..c {
                pooled[dst + ci] *= inv;
            }
        }
    }
    // Zero-centered RMSNorm.
    let norm_w = projector
        .get("multi_modal_projector.mm_soft_emb_norm.weight")
        .ok_or_else(|| anyhow!("projector norm weight missing"))?
        .to_f32_vec();
    let eps = 1e-6f32;
    for row in pooled.chunks_mut(c) {
        let ms = row.iter().map(|x| x * x).sum::<f32>() / c as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (x, w) in row.iter_mut().zip(&norm_w) {
            *x = *x * inv * (1.0 + w);
        }
    }
    // Raw matmul with the projection Parameter [c, text_hidden].
    let proj = projector
        .get("multi_modal_projector.mm_input_projection_weight")
        .ok_or_else(|| anyhow!("projector matrix missing"))?;
    let pd = proj.shape().dims().to_vec();
    if pd != [c, text_hidden] {
        anyhow::bail!("projector shape {pd:?}, expected [{c}, {text_hidden}]");
    }
    let pw = proj.to_f32_vec();
    let mut out = vec![0.0f32; mm_tokens * text_hidden];
    for t in 0..mm_tokens {
        let row = &pooled[t * c..(t + 1) * c];
        let dst = &mut out[t * text_hidden..(t + 1) * text_hidden];
        for (ci, &x) in row.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let wrow = &pw[ci * text_hidden..(ci + 1) * text_hidden];
            for (d, w) in dst.iter_mut().zip(wrow) {
                *d += x * w;
            }
        }
    }
    Ok(out)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-memory (data-URI) path must produce bit-identical pixels to the
    /// file path — both funnel through `normalize_image`.
    #[test]
    fn preprocess_bytes_matches_file_path() {
        // 3×2 PNG with distinct corner colors, encoded to bytes in memory.
        let mut img = image::RgbImage::new(3, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(2, 1, image::Rgb([0, 0, 255]));
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut png = Vec::new();
        dyn_img
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();

        let from_bytes = normalize_image(image::load_from_memory(&png).unwrap(), 8);
        let dir = std::env::temp_dir().join("sapient-vlm-preprocess-test.png");
        std::fs::write(&dir, &png).unwrap();
        let from_file = normalize_image(image::open(&dir).unwrap(), 8);
        let _ = std::fs::remove_file(&dir);

        assert_eq!(from_bytes.len(), 3 * 8 * 8);
        assert_eq!(from_bytes, from_file);
        // Normalization maps [0,255] → [-1,1].
        assert!(from_bytes.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}
