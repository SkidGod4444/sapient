//! Whisper speech-to-text forward engine (encoder + decoder).
//!
//! A pure-Rust port of OpenAI Whisper, reusing SAPIENT's existing transformer
//! kernels: linear/layernorm/add via [`LlmBackendDispatch`], attention via the
//! CPU flash kernel with **explicit masks** (non-causal for the encoder and
//! cross-attention, causal for decoder self-attention — the kernel treats
//! `mask = None` as causal, so a full-attention path must pass an all-zeros
//! mask), exact erf GELU, and conv1d via the im2col conv2d wrapper.
//!
//! Layout conventions match HuggingFace `WhisperForConditionalGeneration`:
//! - encoder: `conv1`→`conv2`→ +`embed_positions` → N pre-LN blocks → `layer_norm`;
//! - decoder: `embed_tokens` + `embed_positions` → N pre-LN blocks
//!   (causal self-attn → cross-attn → MLP) → `layer_norm` → tied `proj_out`.
//!
//! The encoder runs once per 30 s chunk; its per-layer cross-attention K/V are
//! projected once in [`WhisperForward::set_audio_context`] and reused for every
//! decoded token. Decoder self-attention keeps a growing per-layer KV cache.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use sapient_backends_cpu::kernels::attention::{causal_mask, scaled_dot_product_attention};
use sapient_backends_cpu::kernels::elementwise::gelu_erf;
use sapient_core::{Shape, Tensor};
use sapient_hub::whisper_config::WhisperConfig;

use super::backend::{LlmBackend, LlmBackendDispatch, LlmBackendKind};
use super::common::{
    embed_tokens, merge_heads, quantize_tensor_to_q8_0, should_quantize_online, split_heads,
};
use super::conv::conv1d;

const LN_EPS: f32 = 1e-5;

/// Whisper encoder+decoder engine.
pub struct WhisperForward {
    cfg: WhisperConfig,
    /// Weight key prefix: `"model."` for HF safetensors, `""` if absent.
    prefix: String,
    weights: HashMap<String, Tensor>,
    backend: LlmBackendDispatch,

    head_dim: usize,

    /// Cached encoder output `[1, n_audio_ctx, d]` (set by `encode`).
    audio_ctx: Option<Tensor>,
    /// Per-decoder-layer cross-attention K/V `[1, n_head, n_audio_ctx, head_dim]`.
    cross_k: Vec<Option<Tensor>>,
    cross_v: Vec<Option<Tensor>>,
    /// Per-decoder-layer self-attention KV cache `[1, n_head, seq, head_dim]`.
    self_k: Vec<Option<Tensor>>,
    self_v: Vec<Option<Tensor>>,
    /// Number of tokens currently in the decoder self-attention cache.
    decoder_len: usize,
}

impl WhisperForward {
    /// Build from a HuggingFace weight map (CPU backend).
    pub fn from_weights(cfg: WhisperConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        Self::from_weights_with_backend(cfg, weights, LlmBackendKind::Cpu)
    }

    /// Build from a HuggingFace weight map with a chosen backend.
    ///
    /// F16/BF16 linear projections are online-quantized to Q8_0 (matching the
    /// LLM path); conv/embed/positional/norm tensors keep their original dtype.
    /// Attention always runs on the CPU flash kernel (it needs explicit masks),
    /// so on a GPU backend only linear/layernorm/add are accelerated.
    pub fn from_weights_with_backend(
        cfg: WhisperConfig,
        weights: HashMap<String, Tensor>,
        kind: LlmBackendKind,
    ) -> Result<Self> {
        let prefix = if weights.contains_key("model.encoder.conv1.weight") {
            "model.".to_string()
        } else {
            String::new()
        };

        // Online-quantize F16/BF16 2-D linear weights to Q8_0 (skips conv/embed/
        // norm/bias by name + rank, exactly like the LLM loader).
        let weights: HashMap<String, Tensor> = weights
            .into_iter()
            .map(|(name, t)| {
                if should_quantize_online(&name, &t) {
                    (name, quantize_tensor_to_q8_0(t))
                } else {
                    (name, t)
                }
            })
            .collect();

        let backend = LlmBackendDispatch::from_kind_with_head_dim(kind, cfg.head_dim())?;
        let n_dec = cfg.decoder_layers;
        let head_dim = cfg.head_dim();

        Ok(Self {
            cfg,
            prefix,
            weights,
            backend,
            head_dim,
            audio_ctx: None,
            cross_k: vec![None; n_dec],
            cross_v: vec![None; n_dec],
            self_k: vec![None; n_dec],
            self_v: vec![None; n_dec],
            decoder_len: 0,
        })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    // ── weight access ────────────────────────────────────────────────────────

    fn get(&self, key: &str) -> Result<&Tensor> {
        let full = format!("{}{key}", self.prefix);
        self.weights
            .get(&full)
            .ok_or_else(|| anyhow!("missing Whisper weight `{full}`"))
    }

    fn opt(&self, key: &str) -> Option<&Tensor> {
        self.weights.get(&format!("{}{key}", self.prefix))
    }

    // ── encoder ──────────────────────────────────────────────────────────────

    /// Run the audio encoder. `mel` is `[1, n_mels, n_frames]` (3000 frames).
    /// Returns and caches the audio context `[1, n_audio_ctx, d]`.
    pub fn encode(&mut self, mel: &Tensor) -> Result<Tensor> {
        let n_head = self.cfg.encoder_attention_heads;

        // Conv stem: [1, n_mels, 3000] → [1, d, 3000] → [1, d, 1500].
        let mel = mel.to_f32_tensor().map_err(|e| anyhow!("{e}"))?;
        let c1w = self.get("encoder.conv1.weight")?.clone();
        let c1b = self.opt("encoder.conv1.bias").cloned();
        let x = conv1d(&mel, &c1w, c1b.as_ref(), 1, 1, 1, 1)?;
        let x = gelu_erf(&x).map_err(|e| anyhow!("{e}"))?;
        let c2w = self.get("encoder.conv2.weight")?.clone();
        let c2b = self.opt("encoder.conv2.bias").cloned();
        let x = conv1d(&x, &c2w, c2b.as_ref(), 1, 2, 1, 1)?;
        let x = gelu_erf(&x).map_err(|e| anyhow!("{e}"))?;

        // [1, d, T] → [1, T, d], then add positional embedding.
        let x = transpose_12(&x)?;
        let seq = x.shape().dims()[1];
        let pos = self.get("encoder.embed_positions.weight")?.clone();
        let mut h = add_positions(&x, &pos, 0)?;

        // Pre-LN encoder blocks (non-causal self-attention).
        let zero_mask = zeros_mask(seq, seq)?;
        for li in 0..self.cfg.encoder_layers {
            let p = format!("encoder.layers.{li}");
            // Self-attention sublayer.
            let residual = h.clone();
            let normed = self.layer_norm(&h, &p, "self_attn_layer_norm")?;
            let q = self.proj_split(&normed, &p, "self_attn.q_proj", n_head, true)?;
            let k = self.proj_split(&normed, &p, "self_attn.k_proj", n_head, false)?;
            let v = self.proj_split(&normed, &p, "self_attn.v_proj", n_head, true)?;
            let attn = self.sdpa_merge(&q, &k, &v, n_head, &zero_mask)?;
            let attn = self.linear(&attn, &p, "self_attn.out_proj", true)?;
            h = self.backend.add(&residual, &attn)?;
            // MLP sublayer.
            h = self.mlp(&h, &p)?;
        }

        let h = self.layer_norm(&h, "encoder", "layer_norm")?;
        self.audio_ctx = Some(h.clone());
        Ok(h)
    }

    /// Project the encoder output through each decoder layer's cross-attention
    /// K/V once and cache it. Call after [`encode`] and before decoding.
    pub fn set_audio_context(&mut self, audio_ctx: &Tensor) -> Result<()> {
        let n_head = self.cfg.decoder_attention_heads;
        for li in 0..self.cfg.decoder_layers {
            let p = format!("decoder.layers.{li}");
            let k = self.proj_split(audio_ctx, &p, "encoder_attn.k_proj", n_head, false)?;
            let v = self.proj_split(audio_ctx, &p, "encoder_attn.v_proj", n_head, true)?;
            self.cross_k[li] = Some(k);
            self.cross_v[li] = Some(v);
        }
        self.audio_ctx = Some(audio_ctx.clone());
        Ok(())
    }

    // ── decoder ──────────────────────────────────────────────────────────────

    /// Clear decoder state (self-attention KV cache + position counter). The
    /// cached encoder context / cross-attention K/V are preserved.
    pub fn reset_decoder(&mut self) {
        for s in self.self_k.iter_mut() {
            *s = None;
        }
        for s in self.self_v.iter_mut() {
            *s = None;
        }
        self.decoder_len = 0;
    }

    /// Run the decoder over `token_ids` (the forced prompt on the first call,
    /// then one token per step), appending to the self-attention cache. Returns
    /// the next-token logits for the **last** position.
    pub fn decode_step(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            anyhow::bail!("decode_step called with no tokens");
        }
        if self.cross_k.iter().any(|c| c.is_none()) {
            anyhow::bail!("decode_step called before set_audio_context");
        }
        let n_head = self.cfg.decoder_attention_heads;
        let start = self.decoder_len;
        let seq = token_ids.len();

        // Token + learned positional embedding.
        let embed = self.get("decoder.embed_tokens.weight")?;
        let x = embed_tokens(embed, token_ids)?;
        let pos = self.get("decoder.embed_positions.weight")?.clone();
        let mut h = add_positions(&x, &pos, start)?;

        let total = start + seq;
        let self_mask = causal_mask(seq, total); // explicit causal (offset = total-seq)
        let cross_seq = self
            .audio_ctx
            .as_ref()
            .map(|a| a.shape().dims()[1])
            .unwrap_or(self.cfg.max_source_positions);
        let cross_mask = zeros_mask(seq, cross_seq)?;

        for li in 0..self.cfg.decoder_layers {
            let p = format!("decoder.layers.{li}");

            // 1) Causal self-attention with growing KV cache.
            let residual = h.clone();
            let normed = self.layer_norm(&h, &p, "self_attn_layer_norm")?;
            let q = self.proj_split(&normed, &p, "self_attn.q_proj", n_head, true)?;
            let k_new = self.proj_split(&normed, &p, "self_attn.k_proj", n_head, false)?;
            let v_new = self.proj_split(&normed, &p, "self_attn.v_proj", n_head, true)?;
            let k_all = append_kv(self.self_k[li].as_ref(), &k_new)?;
            let v_all = append_kv(self.self_v[li].as_ref(), &v_new)?;
            let attn = self.sdpa_merge(&q, &k_all, &v_all, n_head, &self_mask)?;
            self.self_k[li] = Some(k_all);
            self.self_v[li] = Some(v_all);
            let attn = self.linear(&attn, &p, "self_attn.out_proj", true)?;
            h = self.backend.add(&residual, &attn)?;

            // 2) Cross-attention to the cached encoder K/V.
            let residual = h.clone();
            let normed = self.layer_norm(&h, &p, "encoder_attn_layer_norm")?;
            let q = self.proj_split(&normed, &p, "encoder_attn.q_proj", n_head, true)?;
            let ck = self.cross_k[li].as_ref().unwrap().clone();
            let cv = self.cross_v[li].as_ref().unwrap().clone();
            let attn = self.sdpa_merge(&q, &ck, &cv, n_head, &cross_mask)?;
            let attn = self.linear(&attn, &p, "encoder_attn.out_proj", true)?;
            h = self.backend.add(&residual, &attn)?;

            // 3) MLP.
            h = self.mlp(&h, &p)?;
        }

        let h = self.layer_norm(&h, "decoder", "layer_norm")?;
        self.decoder_len = total;

        // Tied output projection (proj_out is tied to decoder.embed_tokens).
        let proj = match self.opt("proj_out.weight") {
            Some(w) => w.clone(),
            None => self.get("decoder.embed_tokens.weight")?.clone(),
        };
        self.backend.logits_from_hidden(&h, &proj)
    }

    // ── sublayer helpers ───────────────────────────────────────────────────────

    fn mlp(&self, h: &Tensor, layer_prefix: &str) -> Result<Tensor> {
        let residual = h.clone();
        let normed = self.layer_norm(h, layer_prefix, "final_layer_norm")?;
        let up = self.linear(&normed, layer_prefix, "fc1", true)?;
        let up = gelu_erf(&up).map_err(|e| anyhow!("{e}"))?;
        let down = self.linear(&up, layer_prefix, "fc2", true)?;
        self.backend.add(&residual, &down)
    }

    /// `layer_norm` with weight+bias for `{layer_prefix}.{name}`.
    fn layer_norm(&self, x: &Tensor, layer_prefix: &str, name: &str) -> Result<Tensor> {
        let w = self.get(&format!("{layer_prefix}.{name}.weight"))?.clone();
        let b = self.opt(&format!("{layer_prefix}.{name}.bias")).cloned();
        self.backend.layer_norm(x, &w, b.as_ref(), LN_EPS)
    }

    /// Linear `{layer_prefix}.{name}` (`.weight` + optional `.bias`), 3-D in/out.
    fn linear(
        &self,
        x: &Tensor,
        layer_prefix: &str,
        name: &str,
        with_bias: bool,
    ) -> Result<Tensor> {
        let w = self.get(&format!("{layer_prefix}.{name}.weight"))?.clone();
        let b = if with_bias {
            self.opt(&format!("{layer_prefix}.{name}.bias")).cloned()
        } else {
            None
        };
        self.backend.linear_3d_bias(x, &w, b.as_ref())
    }

    /// Linear projection then split into heads → `[1, n_head, seq, head_dim]`.
    fn proj_split(
        &self,
        x: &Tensor,
        layer_prefix: &str,
        name: &str,
        n_head: usize,
        with_bias: bool,
    ) -> Result<Tensor> {
        let y = self.linear(x, layer_prefix, name, with_bias)?;
        split_heads(&y, n_head, self.head_dim)
    }

    /// Scaled-dot-product attention (default scale `1/√head_dim`, matching
    /// Whisper) over already-split q/k/v with an explicit additive mask, then
    /// merge heads back to `[1, seq_q, d]`.
    fn sdpa_merge(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let attn = scaled_dot_product_attention(q, k, v, Some(mask), None, n_head)
            .map_err(|e| anyhow!("{e}"))?;
        merge_heads(&attn)
    }
}

// ── free helpers ───────────────────────────────────────────────────────────────

/// Transpose `[1, d, t]` → `[1, t, d]`.
fn transpose_12(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (d, t) = (dims[1], dims[2]);
    let src = x.as_f32_slice();
    let mut out = vec![0.0f32; d * t];
    for di in 0..d {
        for ti in 0..t {
            out[ti * d + di] = src[di * t + ti];
        }
    }
    Tensor::from_f32(&out, Shape::new([1, t, d])).map_err(|e| anyhow!("{e}"))
}

/// Add positional rows `pos[start..start+seq]` to `x [1, seq, d]`.
fn add_positions(x: &Tensor, pos: &Tensor, start: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (seq, d) = (dims[1], dims[2]);
    let pos_f = pos.to_f32_cow();
    let pos_dims = pos.shape().dims();
    let p_d = pos_dims[pos_dims.len() - 1];
    if p_d != d {
        anyhow::bail!("positional dim {p_d} != hidden {d}");
    }
    if start + seq > pos_dims[pos_dims.len() - 2] {
        anyhow::bail!("position {} out of range", start + seq);
    }
    let mut out = x.as_f32_slice().to_vec();
    for s in 0..seq {
        let pr = (start + s) * d;
        let xr = s * d;
        for i in 0..d {
            out[xr + i] += pos_f[pr + i];
        }
    }
    Tensor::from_f32(&out, Shape::new([1, seq, d])).map_err(|e| anyhow!("{e}"))
}

/// All-zeros additive mask `[seq_q, seq_k]` → full (non-causal) attention.
fn zeros_mask(seq_q: usize, seq_k: usize) -> Result<Tensor> {
    Tensor::from_f32(&vec![0.0f32; seq_q * seq_k], vec![seq_q, seq_k]).map_err(|e| anyhow!("{e}"))
}

/// Concatenate a new KV slice `[1, h, s_new, hd]` after the cached
/// `[1, h, s_old, hd]` along the sequence axis (axis 2), respecting the
/// per-head layout. Returns `[1, h, s_old+s_new, hd]`.
fn append_kv(prev: Option<&Tensor>, new: &Tensor) -> Result<Tensor> {
    let nd = new.shape().dims().to_vec();
    let (h, s_new, hd) = (nd[1], nd[2], nd[3]);
    let new_data = new.to_contiguous_f32_vec();
    let Some(prev) = prev else {
        return Ok(new.clone());
    };
    let pd = prev.shape().dims().to_vec();
    let s_old = pd[2];
    let prev_data = prev.to_contiguous_f32_vec();
    let total = s_old + s_new;
    let mut out = vec![0.0f32; h * total * hd];
    for hi in 0..h {
        let dst = hi * total * hd;
        let psrc = hi * s_old * hd;
        let nsrc = hi * s_new * hd;
        out[dst..dst + s_old * hd].copy_from_slice(&prev_data[psrc..psrc + s_old * hd]);
        out[dst + s_old * hd..dst + total * hd].copy_from_slice(&new_data[nsrc..nsrc + s_new * hd]);
    }
    Tensor::from_f32(&out, Shape::new([1, h, total, hd])).map_err(|e| anyhow!("{e}"))
}

/// Audio inference engine (parallel to [`super::ForwardEngine`] for text).
pub enum AudioEngine {
    /// CPU (and Metal-via-`LlmBackendDispatch`) Whisper.
    Whisper(Box<WhisperForward>),
    /// Cross-platform GPU Whisper (wgpu/WGSL — Vulkan/DX12/Metal).
    #[cfg(feature = "wgpu")]
    WhisperWgpu(Box<super::whisper_wgpu::WhisperWgpuEngine>),
}

impl AudioEngine {
    /// Encode mel `[1, n_mels, n_frames]` → audio context `[1, n_audio_ctx, d]`,
    /// also projecting and caching the per-layer cross-attention K/V.
    pub fn encode(&mut self, mel: &Tensor) -> Result<Tensor> {
        match self {
            Self::Whisper(w) => {
                let ctx = w.encode(mel)?;
                w.set_audio_context(&ctx)?;
                Ok(ctx)
            }
            // The wgpu engine caches cross-attention K/V inside `encode`.
            #[cfg(feature = "wgpu")]
            Self::WhisperWgpu(w) => w.encode(mel),
        }
    }

    /// Next-token logits for the last position of `token_ids`.
    pub fn decode_step(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Whisper(w) => w.decode_step(token_ids),
            #[cfg(feature = "wgpu")]
            Self::WhisperWgpu(w) => w.decode_step(token_ids),
        }
    }

    /// Clear decoder self-attention state for a new utterance/chunk.
    pub fn reset_decoder(&mut self) {
        match self {
            Self::Whisper(w) => w.reset_decoder(),
            #[cfg(feature = "wgpu")]
            Self::WhisperWgpu(w) => w.reset_decoder(),
        }
    }

    pub fn config(&self) -> &WhisperConfig {
        match self {
            Self::Whisper(w) => w.config(),
            #[cfg(feature = "wgpu")]
            Self::WhisperWgpu(w) => w.config(),
        }
    }
}
