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
use sapient_models::forward::{LlamaForward, SiglipConfig, SiglipVision};
use sapient_tokenizers::{SapientTokenizer, TokenizerOptions};

/// A loaded vision-language model: vision tower + text engine + tokenizer.
pub struct VlmPipeline {
    vision: SiglipVision,
    engine: LlamaForward,
    tokenizer: SapientTokenizer,
    image_token_id: u32,
    eos_ids: Vec<u32>,
}

impl VlmPipeline {
    /// Download (or reuse cached) and load an Idefics3-family repo.
    pub async fn from_pretrained(model: &str) -> Result<Self> {
        let repo = sapient_hub::registry::resolve_vlm_repo(model);
        let client = HubClient::new()?;
        let files = client
            .download_files(
                &repo,
                &["config.json", "tokenizer.json", "model.safetensors"],
            )
            .await
            .with_context(|| format!("downloading {repo}"))?;
        Self::from_files(&files[0], &files[1], &files[2])
    }

    /// Load from already-downloaded files.
    pub fn from_files(config: &Path, tokenizer: &Path, weights: &Path) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(config)?).context("config.json")?;
        if cfg["model_type"].as_str() != Some("idefics3") {
            anyhow::bail!(
                "VLM path supports the Idefics3/SmolVLM family; got model_type {:?}",
                cfg["model_type"]
            );
        }
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
            raw: serde_json::Value::Null,
        };

        // ── split the checkpoint: vision+connector vs text backbone ─────────
        let all = sapient_io::safetensors::SafetensorsLoader::load(weights)
            .with_context(|| format!("loading {weights:?}"))?;
        let mut vision_w: HashMap<String, Tensor> = HashMap::new();
        let mut text_w: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in all {
            if name.contains("vision_model") || name.contains("connector") {
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
            engine,
            tokenizer,
            image_token_id,
            eos_ids,
        })
    }

    /// Load + preprocess an image file: squash-resize to the tower's square
    /// input, RGB, `(x/255 − 0.5)/0.5`, channel-major `[3, S, S]`.
    pub fn preprocess_image(&self, path: &Path) -> Result<Vec<f32>> {
        let s = self.vision.config().image_size as u32;
        let img = image::open(path).with_context(|| format!("opening image {path:?}"))?;
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
        Ok(out)
    }

    /// Encode preprocessed pixels to visual-token embeddings (probes/tools).
    pub fn encode_image_embeddings(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        self.vision.encode(pixels)
    }

    /// One vision-language turn: describe/answer `question` about the image.
    /// Greedy decode, up to `max_new` tokens. Returns the reply text.
    pub fn answer(&mut self, image: &Path, question: &str, max_new: usize) -> Result<String> {
        // 1. Vision: pixels → [n_vis, text_hidden] embeddings.
        let pixels = self.preprocess_image(image)?;
        let vis = self.vision.encode(&pixels)?;
        let n_vis = self.vision.config().n_visual_tokens();
        let text_hidden = self.vision.config().text_hidden;

        // 2. Prompt with the processor's image expansion (single global image).
        let img_seq = format!(
            "<fake_token_around_image><global-img>{}<fake_token_around_image>",
            "<image>".repeat(n_vis)
        );
        let prompt = format!("<|im_start|>User:{img_seq}{question}<end_of_utterance>\nAssistant:");
        let ids = self.tokenizer.encode_ids(&prompt, false)?;

        // 3. Splice: overwrite the <image> rows with the vision embeddings.
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
                "expected {n_vis} <image> tokens in the prompt, found {}",
                img_positions.len()
            );
        }
        for (vi, &pos) in img_positions.iter().enumerate() {
            ev[pos * text_hidden..(pos + 1) * text_hidden]
                .copy_from_slice(&vis[vi * text_hidden..(vi + 1) * text_hidden]);
        }
        let embeds = Tensor::from_f32(&ev, sapient_core::Shape::new([1, ids.len(), text_hidden]))
            .map_err(|e| anyhow!("{e}"))?;

        // 4. Prefill (builds the KV cache), then greedy decode on token ids.
        let mut logits = self.engine.forward_logits_embeds(embeds, true)?;
        let mut out_ids: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let next = argmax(&logits);
            if self.eos_ids.contains(&next) {
                break;
            }
            out_ids.push(next);
            logits = self.engine.forward_logits(&[next], true)?;
        }
        self.engine.reset_cache();
        Ok(self.tokenizer.decode(&out_ids, true)?.trim().to_string())
    }
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
