// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! SmolVLA — a vision-language-action policy (LeRobot `lerobot/smolvla_base`).
//!
//! One inference turns camera image(s) + a language instruction + the robot
//! state into a **chunk of future actions** (`chunk × max_action_dim`, 50 × 32):
//!
//! 1. **Prefix** — SigLIP + connector image tokens (×√hidden), language token
//!    embeddings (×√hidden) and one state token (`state_proj`) run through the
//!    first 16 layers of SmolVLM2-500M's text model. Each layer's post-RoPE
//!    keys and raw values are kept: that K/V cache is all the action expert
//!    ever sees of the observation.
//! 2. **Action expert** — a 16-layer, 720-wide transformer over the 50 noisy
//!    action tokens. **Even layers** are self-attention over `[prefix K/V ;
//!    action K/V]` (causal inside the chunk); **odd layers** are
//!    cross-attention: the expert's `k_proj`/`v_proj` re-project the VLM
//!    layer's cached K/V and the action tokens attend to the prefix only.
//! 3. **Flow matching** — `num_steps` (10) forward-Euler steps from noise
//!    (t = 1) to actions (t = 0): `x ← x − v(x, t) / num_steps`.
//!
//! Things that differ from the stock VLM path and are easy to get wrong:
//! * RoPE base is **10 000** here, not SmolVLM2's configured 100 000 (the
//!   reference hard-codes it for both the VLM layers and the expert).
//! * The prefix is **not causal**: image and language tokens attend to each
//!   other bidirectionally, none of them attends to the state token, and the
//!   state token attends to everything.
//! * Padded language tokens are never attended to and do not advance the
//!   position counter, so this engine simply **drops** them (same result for
//!   every real token, shorter prefix).
//! * Cross-attention layers rebase the action positions to `0..chunk`;
//!   self-attention layers continue after the prefix (`n_prefix..`).
//!
//! CPU, f32. Validated stage by stage against the LeRobot reference
//! (`scripts/gen_smolvla_fixture.py`, `tests/smolvla_reference.rs`).

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use sapient_backends_cpu::kernels::matmul::{matmul_nt, sgemm_serial};
use sapient_core::{DType, Shape, Tensor};

use super::common::{embed_tokens, quantize_tensor_to_q8_0};
use super::siglip::{SiglipConfig, SiglipVision};

/// Stage counters (ns) printed by `sample_actions` under `SAPIENT_VLA_TIMING`:
/// time inside linears and inside attention; the rest is norms, RoPE, copies.
static T_LINEAR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static T_ATTN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn lap(counter: &std::sync::atomic::AtomicU64, since: std::time::Instant) {
    counter.fetch_add(
        since.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

const VLM_PREFIX: &str = "model.vlm_with_expert.vlm.";
const EXPERT_PREFIX: &str = "model.vlm_with_expert.lm_expert.";
const TEXT: &str = "model.text_model";

/// SmolVLA dimensions. Defaults are `lerobot/smolvla_base`; the widths are
/// re-read from the checkpoint's tensor shapes in [`SmolVla::from_weights`].
#[derive(Debug, Clone)]
pub struct SmolVlaConfig {
    pub vlm_hidden: usize,
    pub expert_hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Actions per chunk (`chunk_size`).
    pub chunk: usize,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    /// Flow-matching Euler steps.
    pub num_steps: usize,
    /// Every n-th expert layer (0, n, 2n, …) is self-attention; the rest are
    /// cross-attention over the VLM K/V.
    pub self_attn_every: usize,
    pub min_period: f64,
    pub max_period: f64,
    pub rms_eps: f32,
    pub rope_base: f32,
}

impl Default for SmolVlaConfig {
    fn default() -> Self {
        Self {
            vlm_hidden: 960,
            expert_hidden: 720,
            layers: 16,
            heads: 15,
            kv_heads: 5,
            head_dim: 64,
            chunk: 50,
            max_state_dim: 32,
            max_action_dim: 32,
            num_steps: 10,
            self_attn_every: 2,
            min_period: 4e-3,
            max_period: 4.0,
            rms_eps: 1e-5,
            rope_base: 10_000.0,
        }
    }
}

/// The observation as the action expert sees it: per VLM layer, the post-RoPE
/// keys and the values of the `n` prefix tokens, both `[kv_heads, n, head_dim]`.
pub struct PrefixCache {
    pub n: usize,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    /// For each cross-attention expert layer: the VLM K/V already re-projected
    /// by that layer's `k_proj` / `v_proj`, `[kv_heads, n, head_dim]`. They
    /// depend only on the prefix, so they are computed once per observation
    /// instead of once per denoising step. `None` for self-attention layers.
    cross: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

/// A loaded SmolVLA policy.
pub struct SmolVla {
    cfg: SmolVlaConfig,
    vision: SiglipVision,
    w: HashMap<String, Tensor>,
}

/// Which parts of the policy store their linear weights as Q8_0 (8-bit blocks,
/// ~1.06 bytes/weight) instead of f32. Only matrices whose input width is a
/// multiple of 32 can be Q8_0, so the expert's 720-wide inputs (`q_proj`,
/// self-attention `k_proj`/`v_proj`, `gate_proj`, `up_proj`) always stay f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmolVlaQuant {
    /// SigLIP tower + connector.
    pub vision: bool,
    /// The 16 SmolVLM2 text layers of the prefix pass.
    pub vlm: bool,
    /// The action expert (run `num_steps` times per chunk).
    pub expert: bool,
    /// Vectorized polynomial `exp` in the vision tower's softmax and GELU
    /// (`SiglipVision::with_fast_math`) instead of libm calls.
    pub fast_math: bool,
    /// int8 attention in the vision tower (`SiglipVision::with_int8_attention`).
    pub int8_attention: bool,
}

impl SmolVlaQuant {
    /// Everything f32 — reproduces the f32 reference.
    pub const NONE: Self = Self {
        vision: false,
        vlm: false,
        expert: false,
        fast_math: false,
        int8_attention: false,
    };
    pub const ALL: Self = Self {
        vision: true,
        vlm: true,
        expert: true,
        fast_math: true,
        int8_attention: true,
    };
}

/// A linear weight as the engine stores it: Q8_0 when asked for, else f32.
///
/// Q8_0 rows are whole 32-element blocks, so a matrix whose input width is not
/// a multiple of 32 (the expert's 720) gets **zero columns appended** up to the
/// next multiple (736) before quantizing; [`SmolVla::linear`] pads the
/// activations with zeros to match. The product is unchanged — the extra terms
/// are `0 · 0`.
fn prepare(name: &str, t: Tensor, quantize: bool) -> Result<Tensor> {
    let dims = t.shape().dims().to_vec();
    if !quantize || dims.len() != 2 || ["norm", "bias", "embed"].iter().any(|s| name.contains(s)) {
        return to_f32(t);
    }
    let (rows, k) = (dims[0], dims[1]);
    let padded_k = k.div_ceil(32) * 32;
    if padded_k == k {
        return Ok(quantize_tensor_to_q8_0(t));
    }
    let src = t.to_f32_vec();
    let mut padded = vec![0.0f32; rows * padded_k];
    for (dst, row) in padded.chunks_exact_mut(padded_k).zip(src.chunks_exact(k)) {
        dst[..k].copy_from_slice(row);
    }
    let t =
        Tensor::from_f32_vec(padded, Shape::new([rows, padded_k])).map_err(|e| anyhow!("{e}"))?;
    Ok(quantize_tensor_to_q8_0(t))
}

/// Convert a float tensor to F32 (exact for F16/BF16 sources).
fn to_f32(t: Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        return Ok(t);
    }
    let dims = t.shape().dims().to_vec();
    Tensor::from_f32_vec(t.to_f32_vec(), Shape::new(dims)).map_err(|e| anyhow!("{e}"))
}

impl SmolVla {
    /// Build from the checkpoint's tensors (keys as in `model.safetensors`).
    ///
    /// Everything except the token-embedding table is widened to F32 so the
    /// engine reproduces the f32 reference exactly; quantizing the linears is
    /// a later, separately measured step. The unused `lm_head` is dropped.
    pub fn from_weights(cfg: SmolVlaConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        Self::from_weights_quant(cfg, weights, SmolVlaQuant::NONE)
    }

    /// [`from_weights`](Self::from_weights) with some parts stored as Q8_0.
    /// Smaller and faster, no longer bit-close to the f32 reference — the
    /// action error of each choice is measured in `tests/smolvla_reference.rs`.
    pub fn from_weights_quant(
        mut cfg: SmolVlaConfig,
        weights: HashMap<String, Tensor>,
        quant: SmolVlaQuant,
    ) -> Result<Self> {
        // Convert / quantize in parallel (quantizing ~400M weights serially
        // dominated load time); `true` routes a tensor to the vision tower.
        let prepared: Vec<(bool, String, Tensor)> = weights
            .into_par_iter()
            .filter_map(|(name, t)| -> Option<Result<(bool, String, Tensor)>> {
                if let Some(rest) = name.strip_prefix(VLM_PREFIX) {
                    if rest.starts_with("lm_head") {
                        return None;
                    }
                    let vision = rest.starts_with("model.vision_model")
                        || rest.starts_with("model.connector");
                    let t = if vision {
                        // Widen to f32 once here: the float matmul would
                        // otherwise convert a BF16 weight on every call.
                        if quant.vision && t.shape().dims().len() == 2 {
                            prepare(rest, t, true)
                        } else {
                            to_f32(t)
                        }
                    } else if rest.ends_with("embed_tokens.weight") {
                        Ok(t)
                    } else {
                        prepare(rest, t, quant.vlm)
                    };
                    Some(t.map(|t| (vision, rest.to_string(), t)))
                } else if let Some(rest) = name.strip_prefix(EXPERT_PREFIX) {
                    Some(
                        prepare(rest, t, quant.expert)
                            .map(|t| (false, format!("expert.{rest}"), t)),
                    )
                } else {
                    name.strip_prefix("model.")
                        .map(|rest| to_f32(t).map(|t| (false, format!("head.{rest}"), t)))
                }
            })
            .collect::<Result<_>>()?;
        let mut vision_w = HashMap::new();
        let mut w = HashMap::new();
        for (vision, name, t) in prepared {
            if vision { &mut vision_w } else { &mut w }.insert(name, t);
        }

        let dims = |name: &str| -> Result<Vec<usize>> {
            w.get(name)
                .map(|t: &Tensor| t.shape().dims().to_vec())
                .ok_or_else(|| anyhow!("SmolVLA weight missing: {name}"))
        };
        cfg.vlm_hidden = dims(&format!("{TEXT}.embed_tokens.weight"))?[1];
        cfg.expert_hidden = dims("expert.norm.weight")?[0];
        let q = dims("expert.layers.0.self_attn.q_proj.weight")?[0];
        let kv = dims(&format!("{TEXT}.layers.0.self_attn.k_proj.weight"))?[0];
        if q != cfg.heads * cfg.head_dim || kv != cfg.kv_heads * cfg.head_dim {
            bail!(
                "SmolVLA attention shape mismatch: q_proj out {q}, k_proj out {kv}, expected \
                 {} heads / {} KV heads of dim {}",
                cfg.heads,
                cfg.kv_heads,
                cfg.head_dim
            );
        }
        cfg.max_state_dim = dims("head.state_proj.weight")?[1];
        cfg.max_action_dim = dims("head.action_in_proj.weight")?[1];
        cfg.layers = (0..)
            .take_while(|i| w.contains_key(&format!("{TEXT}.layers.{i}.input_layernorm.weight")))
            .count();
        if !w.contains_key(&format!(
            "expert.layers.{}.input_layernorm.weight",
            cfg.layers - 1
        )) {
            bail!("SmolVLA: expert must have one layer per VLM layer");
        }

        let pos = vision_w
            .get("model.vision_model.embeddings.position_embedding.weight")
            .ok_or_else(|| anyhow!("SmolVLA: vision position embedding missing"))?
            .shape()
            .dims()
            .to_vec();
        let patch_w = vision_w
            .get("model.vision_model.embeddings.patch_embedding.weight")
            .ok_or_else(|| anyhow!("SmolVLA: vision patch embedding missing"))?
            .shape()
            .dims()
            .to_vec();
        let side = (pos[0] as f64).sqrt() as usize;
        let proj_in = vision_w
            .get("model.connector.modality_projection.proj.weight")
            .ok_or_else(|| anyhow!("SmolVLA: connector projection missing"))?
            .shape()
            .dims()[1];
        let vcfg = SiglipConfig {
            hidden: pos[1],
            layers: (0..)
                .take_while(|i| {
                    vision_w.contains_key(&format!(
                        "model.vision_model.encoder.layers.{i}.layer_norm1.weight"
                    ))
                })
                .count(),
            heads: 12,
            intermediate: vision_w
                .get("model.vision_model.encoder.layers.0.mlp.fc1.weight")
                .ok_or_else(|| anyhow!("SmolVLA: vision MLP missing"))?
                .shape()
                .dims()[0],
            image_size: side * patch_w[2],
            patch: patch_w[2],
            scale_factor: ((proj_in / pos[1]) as f64).sqrt() as usize,
            text_hidden: cfg.vlm_hidden,
        };
        // `SAPIENT_VLA_FAST_MATH=0|1` overrides for A/B timing.
        let fast_math = match std::env::var("SAPIENT_VLA_FAST_MATH").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => quant.fast_math,
        };
        let int8_attention = match std::env::var("SAPIENT_VLA_INT8_ATTN").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => quant.int8_attention,
        };
        let vision = SiglipVision::new(vcfg, vision_w)?
            .with_fast_math(fast_math)
            .with_int8_attention(int8_attention);
        Ok(Self { cfg, vision, w })
    }

    pub fn config(&self) -> &SmolVlaConfig {
        &self.cfg
    }

    /// Side length of the square image the vision tower expects.
    pub fn image_size(&self) -> usize {
        self.vision.config().image_size
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| anyhow!("SmolVLA weight missing: {name}"))
    }

    /// `y = x·Wᵀ (+ b)` over `rows` row vectors; `name` without `.weight`.
    fn linear(&self, x: &[f32], rows: usize, name: &str) -> Result<Vec<f32>> {
        let started = std::time::Instant::now();
        let w = self.get(&format!("{name}.weight"))?;
        let in_dim = w.shape().dims()[1];
        let xt = if x.len() == rows * in_dim {
            Tensor::from_f32(x, Shape::new([rows, in_dim]))
        } else if rows > 0
            && x.len() % rows == 0
            && x.len() / rows < in_dim
            && in_dim - x.len() / rows < 32
            && in_dim % 32 == 0
        {
            // Zero-padded Q8_0 weight (see `prepare`): pad the activations too.
            let k = x.len() / rows;
            let mut padded = vec![0.0f32; rows * in_dim];
            for (dst, row) in padded.chunks_exact_mut(in_dim).zip(x.chunks_exact(k)) {
                dst[..k].copy_from_slice(row);
            }
            Tensor::from_f32_vec(padded, Shape::new([rows, in_dim]))
        } else {
            bail!("{name}: input {} != {rows} x {in_dim}", x.len());
        }
        .map_err(|e| anyhow!("{e}"))?;
        let mut y = matmul_nt(&xt, w)
            .map_err(|e| anyhow!("{name}: {e}"))?
            .to_f32_vec();
        if let Some(b) = self.w.get(&format!("{name}.bias")) {
            let b = b.as_f32_slice();
            for row in y.chunks_exact_mut(b.len()) {
                for (v, bi) in row.iter_mut().zip(b) {
                    *v += bi;
                }
            }
        }
        lap(&T_LINEAR, started);
        Ok(y)
    }

    /// RMSNorm over rows of width `weight.len()`.
    fn rms_norm(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let w = self.get(name)?.as_f32_slice();
        let mut out = vec![0.0f32; x.len()];
        for (src, dst) in x.chunks_exact(w.len()).zip(out.chunks_exact_mut(w.len())) {
            let ms = src.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
            let inv = 1.0 / (ms + self.cfg.rms_eps).sqrt();
            for ((d, s), wi) in dst.iter_mut().zip(src).zip(w) {
                *d = s * inv * wi;
            }
        }
        Ok(out)
    }

    /// SwiGLU MLP: `down(silu(gate(x)) · up(x))`.
    fn mlp(&self, x: &[f32], rows: usize, layer: &str) -> Result<Vec<f32>> {
        let mut g = self.linear(x, rows, &format!("{layer}.mlp.gate_proj"))?;
        let u = self.linear(x, rows, &format!("{layer}.mlp.up_proj"))?;
        for (gi, ui) in g.iter_mut().zip(&u) {
            *gi = *gi / (1.0 + (-*gi).exp()) * ui;
        }
        self.linear(&g, rows, &format!("{layer}.mlp.down_proj"))
    }

    // ── prefix ──────────────────────────────────────────────────────────────

    /// Preprocessed pixels `[3, S, S]` in `[-1, 1]` → `[n_img_tokens · vlm_hidden]`
    /// (SigLIP + connector, before the √hidden scale).
    pub fn embed_image(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        self.vision.encode(pixels)
    }

    /// Build the prefix embeddings `[n, vlm_hidden]`: every image's tokens,
    /// then the language tokens, then the state token. `lang_tokens` must hold
    /// only real tokens (no padding); `state` is padded to `max_state_dim`.
    pub fn embed_prefix(
        &self,
        images: &[&[f32]],
        lang_tokens: &[u32],
        state: &[f32],
    ) -> Result<Vec<f32>> {
        let scale = (self.cfg.vlm_hidden as f32).sqrt();
        let mut embs = Vec::new();
        for pixels in images {
            embs.extend(self.embed_image(pixels)?.into_iter().map(|v| v * scale));
        }
        embs.extend(self.embed_language_and_state(lang_tokens, state)?);
        Ok(embs)
    }

    /// The non-image tail of the prefix: language token embeddings (×√hidden)
    /// followed by the single state token.
    pub fn embed_language_and_state(&self, lang_tokens: &[u32], state: &[f32]) -> Result<Vec<f32>> {
        let scale = (self.cfg.vlm_hidden as f32).sqrt();
        let table = self.get(&format!("{TEXT}.embed_tokens.weight"))?;
        let lang = embed_tokens(table, lang_tokens)?;
        let mut embs: Vec<f32> = lang.as_f32_slice().iter().map(|v| v * scale).collect();

        if state.len() > self.cfg.max_state_dim {
            bail!(
                "state has {} dims, the policy takes at most {}",
                state.len(),
                self.cfg.max_state_dim
            );
        }
        let mut padded = vec![0.0f32; self.cfg.max_state_dim];
        padded[..state.len()].copy_from_slice(state);
        embs.extend(self.linear(&padded, 1, "head.state_proj")?);
        Ok(embs)
    }

    /// Run the prefix through the VLM layers, keeping each layer's K/V.
    ///
    /// The last token must be the state token (it is the only one the others
    /// may not attend to). Also returns the final-norm hidden states — unused
    /// by action sampling, kept for validation.
    pub fn prefix_pass(&self, embs: &[f32]) -> Result<(PrefixCache, Vec<f32>)> {
        self.prefix_inner(embs, true)
    }

    /// The K/V cache only — what action sampling needs. Skips the last layer's
    /// attention and MLP and the final norm: nothing reads their output (the
    /// cache values are identical to [`prefix_pass`](Self::prefix_pass)).
    pub fn prefix_cache(&self, embs: &[f32]) -> Result<PrefixCache> {
        Ok(self.prefix_inner(embs, false)?.0)
    }

    fn prefix_inner(&self, embs: &[f32], want_output: bool) -> Result<(PrefixCache, Vec<f32>)> {
        let c = &self.cfg;
        let h = c.vlm_hidden;
        let n = embs.len() / h;
        if n < 2 || embs.len() != n * h {
            bail!("prefix must be [n ≥ 2, {h}]");
        }
        // Image + language attend among themselves; the state token (last)
        // attends to everything and is attended to only by itself.
        let allow: Vec<bool> = (0..n * n)
            .map(|ij| ij % n < n - 1 || ij / n == n - 1)
            .collect();
        let positions: Vec<usize> = (0..n).collect();

        let mut x = embs.to_vec();
        let mut cache = PrefixCache {
            n,
            keys: Vec::with_capacity(c.layers),
            values: Vec::with_capacity(c.layers),
            cross: Vec::with_capacity(c.layers),
        };
        for l in 0..c.layers {
            let p = format!("{TEXT}.layers.{l}");
            let normed = self.rms_norm(&x, &format!("{p}.input_layernorm.weight"))?;
            let mut q = self.linear(&normed, n, &format!("{p}.self_attn.q_proj"))?;
            let mut k = self.linear(&normed, n, &format!("{p}.self_attn.k_proj"))?;
            let v = self.linear(&normed, n, &format!("{p}.self_attn.v_proj"))?;
            rope(&mut q, c.heads, c.head_dim, &positions, c.rope_base);
            rope(&mut k, c.kv_heads, c.head_dim, &positions, c.rope_base);
            let k = heads_major(&k, n, c.kv_heads, c.head_dim);
            let v = heads_major(&v, n, c.kv_heads, c.head_dim);
            cache.cross.push(self.project_cross(l, &k, &v, n)?);
            if l + 1 == c.layers && !want_output {
                cache.keys.push(k);
                cache.values.push(v);
                return Ok((cache, Vec::new()));
            }
            let att = attention(&q, &k, &v, n, n, c, &allow);
            cache.keys.push(k);
            cache.values.push(v);

            let o = self.linear(&att, n, &format!("{p}.self_attn.o_proj"))?;
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            let normed = self.rms_norm(&x, &format!("{p}.post_attention_layernorm.weight"))?;
            let m = self.mlp(&normed, n, &p)?;
            for (xi, mi) in x.iter_mut().zip(&m) {
                *xi += mi;
            }
        }
        let out = self.rms_norm(&x, &format!("{TEXT}.norm.weight"))?;
        Ok((cache, out))
    }

    fn is_self_attn(&self, layer: usize) -> bool {
        self.cfg.self_attn_every > 0 && layer % self.cfg.self_attn_every == 0
    }

    /// Cross-attention layer `l`: re-project the VLM layer's K/V
    /// (`[kv_heads, n, hd]` → rows of `kv_heads·hd` → expert `k_proj` / `v_proj`
    /// → heads-major again).
    fn project_cross(
        &self,
        l: usize,
        k: &[f32],
        v: &[f32],
        n: usize,
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        if self.is_self_attn(l) {
            return Ok(None);
        }
        let c = &self.cfg;
        let p = format!("expert.layers.{l}.self_attn");
        let k_rows = seq_major(k, n, c.kv_heads, c.head_dim);
        let v_rows = seq_major(v, n, c.kv_heads, c.head_dim);
        let k = self.linear(&k_rows, n, &format!("{p}.k_proj"))?;
        let v = self.linear(&v_rows, n, &format!("{p}.v_proj"))?;
        Ok(Some((
            heads_major(&k, n, c.kv_heads, c.head_dim),
            heads_major(&v, n, c.kv_heads, c.head_dim),
        )))
    }

    // ── action expert ───────────────────────────────────────────────────────

    /// Embed the noisy action chunk `[chunk, max_action_dim]` at flow time `t`
    /// → `[chunk, expert_hidden]`.
    pub fn embed_suffix(&self, x_t: &[f32], t: f32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let e = c.expert_hidden;
        let action = self.linear(x_t, c.chunk, "head.action_in_proj")?;
        let time = sinusoidal_time_embedding(t, e, c.min_period, c.max_period);
        let mut cat = Vec::with_capacity(c.chunk * 2 * e);
        for row in action.chunks_exact(e) {
            cat.extend_from_slice(row);
            cat.extend_from_slice(&time);
        }
        let mut hid = self.linear(&cat, c.chunk, "head.action_time_mlp_in")?;
        for v in hid.iter_mut() {
            *v /= 1.0 + (-*v).exp(); // silu
        }
        self.linear(&hid, c.chunk, "head.action_time_mlp_out")
    }

    /// One evaluation of the velocity field: `v(x_t, t)` as `[chunk, max_action_dim]`.
    pub fn denoise_step(&self, cache: &PrefixCache, x_t: &[f32], t: f32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (s, n, hd) = (c.chunk, cache.n, c.head_dim);
        let mut x = self.embed_suffix(x_t, t)?;

        // Self-attention: all prefix tokens + causal inside the chunk.
        let allow_self: Vec<bool> = (0..s * (n + s))
            .map(|ij| {
                let (i, j) = (ij / (n + s), ij % (n + s));
                j < n || j - n <= i
            })
            .collect();
        let allow_cross = vec![true; s * n];
        let pos_self: Vec<usize> = (n..n + s).collect();
        let pos_cross: Vec<usize> = (0..s).collect();

        for l in 0..c.layers {
            let p = format!("expert.layers.{l}");
            let normed = self.rms_norm(&x, &format!("{p}.input_layernorm.weight"))?;
            let mut q = self.linear(&normed, s, &format!("{p}.self_attn.q_proj"))?;
            let att = if self.is_self_attn(l) {
                let mut k = self.linear(&normed, s, &format!("{p}.self_attn.k_proj"))?;
                let v = self.linear(&normed, s, &format!("{p}.self_attn.v_proj"))?;
                rope(&mut q, c.heads, hd, &pos_self, c.rope_base);
                rope(&mut k, c.kv_heads, hd, &pos_self, c.rope_base);
                let k = heads_major(&k, s, c.kv_heads, hd);
                let v = heads_major(&v, s, c.kv_heads, hd);
                // [prefix ; suffix] per KV head.
                let join = |pre: &[f32], suf: &[f32]| -> Vec<f32> {
                    let mut out = Vec::with_capacity(c.kv_heads * (n + s) * hd);
                    for h in 0..c.kv_heads {
                        out.extend_from_slice(&pre[h * n * hd..(h + 1) * n * hd]);
                        out.extend_from_slice(&suf[h * s * hd..(h + 1) * s * hd]);
                    }
                    out
                };
                let k = join(&cache.keys[l], &k);
                let v = join(&cache.values[l], &v);
                attention(&q, &k, &v, s, n + s, c, &allow_self)
            } else {
                let (k, v) = cache.cross[l]
                    .as_ref()
                    .ok_or_else(|| anyhow!("prefix cache has no cross K/V for layer {l}"))?;
                rope(&mut q, c.heads, hd, &pos_cross, c.rope_base);
                attention(&q, k, v, s, n, c, &allow_cross)
            };
            let o = self.linear(&att, s, &format!("{p}.self_attn.o_proj"))?;
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            let normed = self.rms_norm(&x, &format!("{p}.post_attention_layernorm.weight"))?;
            let m = self.mlp(&normed, s, &p)?;
            for (xi, mi) in x.iter_mut().zip(&m) {
                *xi += mi;
            }
        }
        let out = self.rms_norm(&x, "expert.norm.weight")?;
        self.linear(&out, s, "head.action_out_proj")
    }

    /// Integrate the flow from `noise` (t = 1) to actions (t = 0) in
    /// `num_steps` Euler steps. Returns `[chunk, max_action_dim]` in the
    /// policy's normalized action space.
    pub fn sample_actions(&self, cache: &PrefixCache, noise: &[f32]) -> Result<Vec<f32>> {
        self.sample_actions_steps(cache, noise, self.cfg.num_steps)
    }

    /// [`sample_actions`](Self::sample_actions) with an explicit number of Euler
    /// steps. Fewer steps cost proportionally less and give a coarser
    /// integration of the same flow (a different, less accurate chunk).
    pub fn sample_actions_steps(
        &self,
        cache: &PrefixCache,
        noise: &[f32],
        num_steps: usize,
    ) -> Result<Vec<f32>> {
        self.sample_actions_inpaint(cache, noise, num_steps, &[])
    }

    /// [`sample_actions_steps`](Self::sample_actions_steps) with the first rows
    /// of the chunk fixed to `frozen` (`[n, max_action_dim]`, in the model's
    /// normalized action space, `n ≤ chunk`): hard inpainting. On the flow
    /// path `x_t = t·noise + (1 − t)·actions`, the frozen rows are overwritten
    /// with their exact `x_t` before every step, so the remaining rows are
    /// denoised next to actions that are already decided, and the result's
    /// first `n` rows equal `frozen`.
    ///
    /// This is for asynchronous execution: `frozen` holds the queued actions
    /// the robot will execute while this chunk is computed, so the new chunk
    /// continues them instead of starting an unrelated plan (the gradient-free
    /// "hard masking" form of real-time chunking, Black et al. 2025). An empty
    /// `frozen` is exactly `sample_actions_steps`.
    pub fn sample_actions_inpaint(
        &self,
        cache: &PrefixCache,
        noise: &[f32],
        num_steps: usize,
        frozen: &[f32],
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if frozen.len() % c.max_action_dim != 0 || frozen.len() > c.chunk * c.max_action_dim {
            bail!(
                "frozen actions must be [n, {}] with n ≤ {}",
                c.max_action_dim,
                c.chunk
            );
        }
        if num_steps == 0 {
            bail!("num_steps must be at least 1");
        }
        if noise.len() != c.chunk * c.max_action_dim {
            bail!("noise must be [{}, {}]", c.chunk, c.max_action_dim);
        }
        let dt = -1.0f64 / num_steps as f64;
        let mut x = noise.to_vec();
        let timing = std::env::var_os("SAPIENT_VLA_TIMING").is_some();
        let started = std::time::Instant::now();
        let relaxed = std::sync::atomic::Ordering::Relaxed;
        let (l0, a0) = (T_LINEAR.load(relaxed), T_ATTN.load(relaxed));
        for step in 0..num_steps {
            let t = (1.0 + step as f64 * dt) as f32;
            for ((xi, ni), ai) in x.iter_mut().zip(noise).zip(frozen) {
                *xi = t * ni + (1.0 - t) * ai;
            }
            let v = self.denoise_step(cache, &x, t)?;
            for (xi, vi) in x.iter_mut().zip(&v) {
                *xi += dt as f32 * vi;
            }
        }
        x[..frozen.len()].copy_from_slice(frozen);
        if timing {
            let ms = |ns: u64| ns as f64 / 1e6;
            let (lin, att) = (T_LINEAR.load(relaxed) - l0, T_ATTN.load(relaxed) - a0);
            let total = started.elapsed().as_nanos() as u64;
            eprintln!(
                "[vla] {} denoise steps · linear {:.0} · attention {:.0} · other {:.0} ms",
                num_steps,
                ms(lin),
                ms(att),
                ms(total.saturating_sub(lin + att))
            );
        }
        Ok(x)
    }
}

/// `[seq, heads·hd]` (seq-major, as a linear produces it) → `[heads, seq, hd]`.
fn heads_major(x: &[f32], seq: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for s in 0..seq {
        for h in 0..heads {
            let src = (s * heads + h) * hd;
            let dst = (h * seq + s) * hd;
            out[dst..dst + hd].copy_from_slice(&x[src..src + hd]);
        }
    }
    out
}

/// Inverse of [`heads_major`].
fn seq_major(x: &[f32], seq: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for s in 0..seq {
        for h in 0..heads {
            let src = (h * seq + s) * hd;
            let dst = (s * heads + h) * hd;
            out[dst..dst + hd].copy_from_slice(&x[src..src + hd]);
        }
    }
    out
}

/// In-place rotate-half RoPE on seq-major `[seq, heads·hd]` data.
fn rope(x: &mut [f32], heads: usize, hd: usize, positions: &[usize], base: f32) {
    let half = hd / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / hd as f32))
        .collect();
    for (row, &pos) in x.chunks_exact_mut(heads * hd).zip(positions) {
        for head in row.chunks_exact_mut(hd) {
            for i in 0..half {
                let (sin, cos) = (pos as f32 * inv[i]).sin_cos();
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * cos - b * sin;
                head[i + half] = b * cos + a * sin;
            }
        }
    }
}

/// Masked grouped-query attention.
///
/// `q` is seq-major `[sq, heads·hd]`; `k`/`v` are heads-major
/// `[kv_heads, sk, hd]`; `allow[i·sk + j]` says whether query `i` may attend to
/// key `j`. Returns seq-major `[sq, heads·hd]`. A fully masked row yields a
/// uniform distribution, like the reference's `finfo.min` fill.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sq: usize,
    sk: usize,
    c: &SmolVlaConfig,
    allow: &[bool],
) -> Vec<f32> {
    let started = std::time::Instant::now();
    let (heads, hd) = (c.heads, c.head_dim);
    let rep = heads / c.kv_heads;
    let scale = (hd as f32).powf(-0.5);
    let qh = heads_major(q, sq, heads, hd);
    let mut out_h = vec![0.0f32; heads * sq * hd];
    out_h
        .par_chunks_mut(sq * hd)
        .enumerate()
        .for_each(|(h, out)| {
            let kvh = h / rep;
            let q_h = &qh[h * sq * hd..(h + 1) * sq * hd];
            let k_h = &k[kvh * sk * hd..(kvh + 1) * sk * hd];
            let v_h = &v[kvh * sk * hd..(kvh + 1) * sk * hd];
            let mut scores = vec![0.0f32; sq * sk];
            sgemm_serial(sq, hd, sk, q_h, k_h, 1, hd, &mut scores);
            for (i, row) in scores.chunks_exact_mut(sk).enumerate() {
                let mut mx = f32::MIN;
                for (j, s) in row.iter_mut().enumerate() {
                    *s = if allow[i * sk + j] {
                        *s * scale
                    } else {
                        f32::MIN
                    };
                    mx = mx.max(*s);
                }
                let mut sum = 0.0f32;
                for s in row.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in row.iter_mut() {
                    *s *= inv;
                }
            }
            sgemm_serial(sq, sk, hd, &scores, v_h, hd, 1, out);
        });
    let out = seq_major(&out_h, sq, heads, hd);
    lap(&T_ATTN, started);
    out
}

/// Sine-cosine embedding of the flow time `t` (openpi's
/// `create_sinusoidal_pos_embedding`): periods log-spaced between
/// `min_period` and `max_period`, computed in f64, laid out `[sin…, cos…]`.
fn sinusoidal_time_embedding(t: f32, dim: usize, min_period: f64, max_period: f64) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let frac = if half > 1 {
            i as f64 / (half - 1) as f64
        } else {
            0.0
        };
        let period = min_period * (max_period / min_period).powf(frac);
        let arg = t as f64 * (1.0 / period * 2.0 * std::f64::consts::PI);
        out[i] = arg.sin() as f32;
        out[half + i] = arg.cos() as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_major_round_trips() {
        let (seq, heads, hd) = (3, 4, 2);
        let x: Vec<f32> = (0..seq * heads * hd).map(|i| i as f32).collect();
        let hm = heads_major(&x, seq, heads, hd);
        // token 1, head 2 → heads-major slot (2, 1)
        assert_eq!(hm[(2 * seq + 1) * hd], x[(heads + 2) * hd]);
        assert_eq!(seq_major(&hm, seq, heads, hd), x);
    }

    #[test]
    fn rope_matches_the_shared_kernel() {
        let (seq, heads, hd) = (5usize, 3usize, 8usize);
        let x: Vec<f32> = (0..seq * heads * hd)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let positions = [0usize, 1, 2, 7, 40];
        let mut ours = x.clone();
        rope(&mut ours, heads, hd, &positions, 10_000.0);

        let hm = heads_major(&x, seq, heads, hd);
        let t = Tensor::from_f32(&hm, Shape::new([1, heads, seq, hd])).unwrap();
        let r = sapient_backends_cpu::kernels::rope::apply_rope(&t, &positions, 10_000.0)
            .unwrap()
            .to_f32_vec();
        let r = seq_major(&r, seq, heads, hd);
        for (a, b) in ours.iter().zip(&r) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn time_embedding_endpoints() {
        let e = sinusoidal_time_embedding(1.0, 8, 4e-3, 4.0);
        // Slowest period = 4.0 → angle π/2 at t = 1.
        assert!((e[3] - 1.0).abs() < 1e-6 && e[7].abs() < 1e-6);
        // Fastest period = 4e-3 → 250 full turns.
        assert!(e[0].abs() < 1e-4 && (e[4] - 1.0).abs() < 1e-6);
    }

    /// One query, two keys, identity-like values: masking a key must move all
    /// the weight to the other one, and a GQA group must share its K/V head.
    #[test]
    fn attention_respects_mask_and_gqa() {
        let c = SmolVlaConfig {
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            ..Default::default()
        };
        let q = vec![1.0, 0.0, 0.0, 1.0]; // [sq=1, heads=2, hd=2]
        let k = vec![1.0, 0.0, 0.0, 1.0]; // [kv=1, sk=2, hd=2]
        let v = vec![10.0, 0.0, 0.0, 20.0];
        let masked = attention(&q, &k, &v, 1, 2, &c, &[true, false]);
        assert_eq!(masked, vec![10.0, 0.0, 10.0, 0.0]);

        let open = attention(&q, &k, &v, 1, 2, &c, &[true, true]);
        let s = (2.0f32).powf(-0.5);
        let p = s.exp() / (s.exp() + 1.0); // head 0 favours key 0
        assert!((open[0] - 10.0 * p).abs() < 1e-5);
        assert!((open[3] - 20.0 * p).abs() < 1e-5); // head 1 favours key 1
    }
}
