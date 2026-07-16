// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Gemma3 forward engine (CPU) — gemma-3-*b-it text models and the text half
//! of MedGemma.
//!
//! Gemma3 differs from Llama in five load-bearing ways, each handled here:
//!
//! 1. **Zero-centered RMSNorm**: every norm computes `x/rms · (1 + w)`. We fold
//!    the `+1` into the stored weights ONCE at load (`fold_gemma_norms`), after
//!    which every site is a standard `rms_norm` dispatch call — including the
//!    per-head **QK-norms** (RMSNorm over `head_dim`, applied after the q/k
//!    projections and before RoPE).
//! 2. **Sandwich norms**: `x += post_attention_layernorm(attn(input_layernorm(x)))`
//!    and `x += post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))` —
//!    four norms per layer, normalizing the branch OUTPUT before the residual.
//! 3. **Alternating attention**: layers where `(idx+1) % sliding_window_pattern != 0`
//!    use a **sliding window** (causal, last `sliding_window` positions); every
//!    pattern-th layer is global. Local layers RoPE with `rope_local_base_freq`
//!    (10k), global layers with `rope_theta` (1M) and, when `rope_scaling`
//!    (type linear) is present (4B+), positions divided by its factor.
//! 4. **Explicit `head_dim`** (256) that is NOT `hidden/heads`, and attention
//!    scale `query_pre_attn_scalar^-0.5` rather than `head_dim^-0.5`.
//! 5. **√hidden embedding scale** and a **tied lm_head**; the 262k-vocab
//!    embedding (~30% of a 1B model) is quantized to Q8_0 at load so both the
//!    row gather and the logits GEMV stay fast and memory-sane.
//!
//! Activation is `gelu_pytorch_tanh` (the `gelu` kernel) with a GeGLU MLP.
//! v1 keeps a full-length f32 KV cache and enforces the window via the mask —
//! bounded-cache locals are a later memory optimization.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use sapient_backends_cpu::kernels::attention::{causal_mask, scaled_dot_product_attention};
use sapient_backends_cpu::kernels::elementwise::gelu;
use sapient_backends_cpu::kernels::rope::apply_rope_partial_scaled;
use sapient_core::{Shape, Tensor};
use sapient_hub::model_info::ModelInfo;

use super::backend::{LlmBackend, LlmBackendDispatch, LlmBackendKind};
use super::common::{
    embed_tokens, kv_cache_ctx, merge_heads, quantize_tensor_to_q8_0, should_quantize_online,
    split_heads,
};

/// Gemma3 extras parsed from the raw config (with 1B-class defaults).
#[derive(Debug, Clone)]
struct Gemma3Ext {
    sliding_window: usize,
    pattern: usize,
    rope_local: f32,
    /// Linear rope_scaling factor for GLOBAL layers (1.0 = none; 8.0 on 4B+).
    global_pos_scale: f32,
    attn_scale: f32,
}

impl Gemma3Ext {
    fn parse(info: &ModelInfo) -> Self {
        // Composite (multimodal) configs nest the text fields.
        let raw = info
            .raw
            .get("text_config")
            .filter(|t| t.is_object())
            .unwrap_or(&info.raw);
        let q_scalar = raw["query_pre_attn_scalar"]
            .as_f64()
            .unwrap_or(info.head_dim as f64);
        let global_pos_scale = raw["rope_scaling"]["factor"].as_f64().unwrap_or(1.0) as f32;
        Self {
            sliding_window: raw["sliding_window"].as_u64().unwrap_or(512) as usize,
            pattern: raw["sliding_window_pattern"].as_u64().unwrap_or(6) as usize,
            rope_local: raw["rope_local_base_freq"].as_f64().unwrap_or(10_000.0) as f32,
            global_pos_scale,
            attn_scale: (q_scalar as f32).powf(-0.5),
        }
    }
}

pub struct Gemma3Forward {
    info: ModelInfo,
    ext: Gemma3Ext,
    weights: HashMap<String, Tensor>,
    backend: LlmBackendDispatch,
    k_cache: Vec<Option<Tensor>>,
    v_cache: Vec<Option<Tensor>>,
    seq_len: usize,
    max_seq: usize,
    embed_key: String,
}

impl Gemma3Forward {
    pub fn from_weights(info: ModelInfo, weights: HashMap<String, Tensor>) -> Result<Self> {
        let ext = Gemma3Ext::parse(&info);
        let backend = LlmBackendDispatch::from_kind(LlmBackendKind::Cpu)
            .map_err(|e| anyhow!("gemma3 backend: {e}"))?;

        // Strip a possible multimodal prefix so text keys are uniform
        // ("model.layers.N…"): MedGemma checkpoints use "language_model.model.*".
        let mut weights: HashMap<String, Tensor> = weights
            .into_iter()
            .filter_map(|(k, v)| {
                if k.contains("vision_tower") || k.contains("multi_modal_projector") {
                    None // text-only v1: the tower is loaded by the VLM path later
                } else if let Some(rest) = k.strip_prefix("language_model.") {
                    Some((rest.to_string(), v))
                } else {
                    Some((k, v))
                }
            })
            .collect();

        let embed_key = "model.embed_tokens.weight".to_string();
        if !weights.contains_key(&embed_key) {
            bail!("gemma3: missing {embed_key}");
        }

        // (1) Fold the Gemma `(1 + w)` into every norm weight, once.
        fold_gemma_norms(&mut weights)?;

        // (5) Quantize the huge embedding (row gather + tied logits GEMV both
        // handle Q8_0), and online-quantize eligible 2-D linears like the
        // Llama engine does, so both engines see the same weight treatment.
        let weights: HashMap<String, Tensor> = weights
            .into_iter()
            .map(|(k, v)| {
                if k == embed_key {
                    let numel_ok = v.shape().dims().last().is_some_and(|d| d % 32 == 0);
                    if numel_ok {
                        return (k, quantize_tensor_to_q8_0(v));
                    }
                    (k, v)
                } else if should_quantize_online(&k, &v) {
                    (k, quantize_tensor_to_q8_0(v))
                } else {
                    (k, v)
                }
            })
            .collect();

        let max_seq = kv_cache_ctx(info.max_position_embeddings);
        let layers = info.num_hidden_layers;
        Ok(Self {
            info,
            ext,
            weights,
            backend,
            k_cache: vec![None; layers],
            v_cache: vec![None; layers],
            seq_len: 0,
            max_seq,
            embed_key,
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn reset_cache(&mut self) {
        for k in &mut self.k_cache {
            *k = None;
        }
        for v in &mut self.v_cache {
            *v = None;
        }
        self.seq_len = 0;
    }

    pub fn truncate_cache(&mut self, _n: usize) {
        // v1: no incremental rollback — reset (same fallback the MLX engine uses).
        self.reset_cache();
    }

    pub fn cache_len(&self) -> usize {
        self.seq_len
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("gemma3 weight missing: {name}"))
    }

    fn rms(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.get(name)?;
        self.backend
            .rms_norm(x, w, self.info.rms_norm_eps as f32)
            .map_err(|e| anyhow!("{e}"))
    }

    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.get(name)?;
        self.backend
            .linear_3d_bias(x, w, None)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Input embeddings for `input_ids`, already ×√hidden (Gemma convention) —
    /// `[1, seq, hidden]`. The VLM path splices visual features over the
    /// `<image_soft_token>` rows of this (matching transformers, which scales
    /// ALL rows then overwrites the image positions).
    pub fn token_embeddings_scaled(&self, input_ids: &[u32]) -> Result<Tensor> {
        let embed = self.get(&self.embed_key)?;
        let x = embed_tokens(embed, input_ids)?;
        let scale = (self.info.hidden_size as f64).sqrt() as f32;
        let mut xv = x.to_f32_vec();
        for v in xv.iter_mut() {
            *v *= scale;
        }
        Tensor::from_f32(&xv, Shape::new([1, input_ids.len(), self.info.hidden_size]))
            .map_err(|e| anyhow!("{e}"))
    }

    /// Last-position logits; appends to the KV cache when `use_cache`.
    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        let x = self.token_embeddings_scaled(input_ids)?;
        self.forward_logits_embeds(x, use_cache)
    }

    /// [`forward_logits`](Self::forward_logits) from pre-built (already-scaled)
    /// embeddings — the multimodal prefill entry.
    pub fn forward_logits_embeds(&mut self, embeds: Tensor, use_cache: bool) -> Result<Vec<f32>> {
        if !use_cache {
            self.reset_cache();
        }
        let seq = embeds.shape().dims()[1];
        let pos0 = self.seq_len;
        if pos0 + seq > self.max_seq {
            // Sliding conversations: same pragmatic reset the caches use.
            self.reset_cache();
        }
        let pos0 = self.seq_len;
        let positions: Vec<usize> = (pos0..pos0 + seq).collect();
        let hd = self.info.head_dim;
        let n_heads = self.info.num_attention_heads;
        let n_kv = self.info.num_key_value_heads;

        let mut x = embeds;
        let xv = x.to_f32_vec();

        let trace = std::env::var("SAPIENT_G3_TRACE").is_ok();
        if trace {
            let m = xv.iter().map(|v| v.abs()).sum::<f32>() / xv.len() as f32;
            eprintln!("[g3] embed×√h mean|x| {m:.4}");
        }
        for l in 0..self.info.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let is_global = (l + 1) % self.ext.pattern == 0;
            let theta = if is_global {
                self.info.rope_theta as f32
            } else {
                self.ext.rope_local
            };
            let pos_scale = if is_global {
                self.ext.global_pos_scale
            } else {
                1.0
            };

            // ── attention branch ────────────────────────────────────────────
            let h = self.rms(&x, &format!("{p}.input_layernorm.weight"))?;
            let q = self.linear(&h, &format!("{p}.self_attn.q_proj.weight"))?;
            let k = self.linear(&h, &format!("{p}.self_attn.k_proj.weight"))?;
            let v = self.linear(&h, &format!("{p}.self_attn.v_proj.weight"))?;
            let q = split_heads(&q, n_heads, hd)?;
            let k = split_heads(&k, n_kv, hd)?;
            let v = split_heads(&v, n_kv, hd)?;
            // QK-norm (RMS over head_dim; the (1+w) is already folded).
            let q = self.rms(&q, &format!("{p}.self_attn.q_norm.weight"))?;
            let k = self.rms(&k, &format!("{p}.self_attn.k_norm.weight"))?;
            let q = apply_rope_partial_scaled(&q, &positions, theta, hd, pos_scale)
                .map_err(|e| anyhow!("{e}"))?;
            let k = apply_rope_partial_scaled(&k, &positions, theta, hd, pos_scale)
                .map_err(|e| anyhow!("{e}"))?;

            let k_all = append_kv(self.k_cache[l].as_ref(), &k)?;
            let v_all = append_kv(self.v_cache[l].as_ref(), &v)?;
            if use_cache {
                self.k_cache[l] = Some(k_all.clone());
                self.v_cache[l] = Some(v_all.clone());
            }
            let total = k_all.shape().dims()[2];
            let mask = if is_global {
                causal_mask(seq, total)
            } else {
                sliding_causal_mask(seq, total, self.ext.sliding_window)
            };
            let attn = scaled_dot_product_attention(
                &q,
                &k_all,
                &v_all,
                Some(&mask),
                Some(self.ext.attn_scale),
                n_kv,
            )
            .map_err(|e| anyhow!("{e}"))?;
            let attn = merge_heads(&attn)?;
            let attn = self.linear(&attn, &format!("{p}.self_attn.o_proj.weight"))?;
            let attn = self.rms(&attn, &format!("{p}.post_attention_layernorm.weight"))?;
            x = self.backend.add(&x, &attn).map_err(|e| anyhow!("{e}"))?;

            // ── MLP branch (GeGLU) ──────────────────────────────────────────
            let h2 = self.rms(&x, &format!("{p}.pre_feedforward_layernorm.weight"))?;
            let gate = self.linear(&h2, &format!("{p}.mlp.gate_proj.weight"))?;
            let up = self.linear(&h2, &format!("{p}.mlp.up_proj.weight"))?;
            let act = {
                let g = gelu(&gate).map_err(|e| anyhow!("{e}"))?;
                let gv = g.to_f32_vec();
                let uv = up.to_f32_vec();
                let prod: Vec<f32> = gv.iter().zip(&uv).map(|(a, b)| a * b).collect();
                let d = up.shape().dims().to_vec();
                Tensor::from_f32(&prod, Shape::new([d[0], d[1], d[2]]))
                    .map_err(|e| anyhow!("{e}"))?
            };
            let down = self.linear(&act, &format!("{p}.mlp.down_proj.weight"))?;
            let down = self.rms(&down, &format!("{p}.post_feedforward_layernorm.weight"))?;
            x = self.backend.add(&x, &down).map_err(|e| anyhow!("{e}"))?;
            if trace {
                let v = x.to_f32_vec();
                let m = v.iter().map(|a| a.abs()).sum::<f32>() / v.len() as f32;
                let mx = v.iter().fold(0.0f32, |acc, a| acc.max(a.abs()));
                eprintln!(
                    "[g3] layer {l:2} ({}) mean|x| {m:9.4} max {mx:9.2}",
                    if is_global { "glob" } else { "slid" }
                );
            }
        }

        if use_cache {
            self.seq_len = pos0 + seq;
        }
        let x = self.rms(&x, "model.norm.weight")?;
        // Tied head: logits against the (Q8_0) embedding matrix.
        let lm_head = self.get(&self.embed_key)?.clone();
        self.backend
            .logits_from_hidden(&x, &lm_head)
            .map_err(|e| anyhow!("{e}"))
    }
}

/// Fold Gemma's zero-centered norm convention (`1 + w`) into the stored
/// weights so every downstream site is a plain RMSNorm.
fn fold_gemma_norms(weights: &mut HashMap<String, Tensor>) -> Result<()> {
    const NORM_SUFFIXES: [&str; 7] = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".pre_feedforward_layernorm.weight",
        ".post_feedforward_layernorm.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        "model.norm.weight",
    ];
    let keys: Vec<String> = weights
        .keys()
        .filter(|k| NORM_SUFFIXES.iter().any(|s| k.ends_with(s)))
        .cloned()
        .collect();
    for k in keys {
        let t = weights.remove(&k).unwrap();
        let mut v = t.to_f32_vec();
        for x in v.iter_mut() {
            *x += 1.0;
        }
        let dims = t.shape().dims().to_vec();
        let folded = Tensor::from_f32(&v, Shape::new(dims)).map_err(|e| anyhow!("{e}"))?;
        weights.insert(k, folded);
    }
    Ok(())
}

/// Additive causal mask limited to the trailing `window` positions:
/// query at absolute position `i` attends to `j ∈ [i+1-window, i]`.
fn sliding_causal_mask(seq_q: usize, seq_k: usize, window: usize) -> Tensor {
    let mut data = vec![0.0f32; seq_q * seq_k];
    let offset = seq_k.saturating_sub(seq_q);
    for qi in 0..seq_q {
        let abs = offset + qi;
        for ki in 0..seq_k {
            let visible = ki <= abs && ki + window > abs;
            if !visible {
                data[qi * seq_k + ki] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_f32(&data, vec![seq_q, seq_k]).expect("mask tensor")
}

/// Concatenate a new KV slice `[1, h, s_new, hd]` after `[1, h, s_old, hd]`.
fn append_kv(prev: Option<&Tensor>, new: &Tensor) -> Result<Tensor> {
    let Some(prev) = prev else {
        return Ok(new.clone());
    };
    let pd = prev.shape().dims().to_vec();
    let nd = new.shape().dims().to_vec();
    let (h, s_old, hd) = (pd[1], pd[2], pd[3]);
    let s_new = nd[2];
    let pv = prev.to_f32_vec();
    let nv = new.to_f32_vec();
    let mut out = vec![0.0f32; h * (s_old + s_new) * hd];
    for hi in 0..h {
        let dst = hi * (s_old + s_new) * hd;
        out[dst..dst + s_old * hd].copy_from_slice(&pv[hi * s_old * hd..(hi + 1) * s_old * hd]);
        out[dst + s_old * hd..dst + (s_old + s_new) * hd]
            .copy_from_slice(&nv[hi * s_new * hd..(hi + 1) * s_new * hd]);
    }
    Tensor::from_f32(&out, Shape::new([1, h, s_old + s_new, hd])).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_mask_windows_correctly() {
        // seq_q 1 at absolute position 5 (seq_k 6), window 3 → attends 3,4,5.
        let m = sliding_causal_mask(1, 6, 3);
        let v = m.to_f32_vec();
        assert_eq!(
            v.iter().map(|x| x.is_finite()).collect::<Vec<_>>(),
            [false, false, false, true, true, true]
        );
        // Prefill: seq_q == seq_k == 4, window 2.
        let m = sliding_causal_mask(4, 4, 2);
        let v = m.to_f32_vec();
        let vis: Vec<bool> = v.iter().map(|x| x.is_finite()).collect();
        assert_eq!(
            vis,
            [
                true, false, false, false, // q0: only j0
                true, true, false, false, // q1: j0..1
                false, true, true, false, // q2: j1..2
                false, false, true, true, // q3: j2..3
            ]
        );
    }

    #[test]
    fn gemma_norm_fold_adds_one() {
        let mut w = HashMap::new();
        w.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::from_f32(&[0.0, -0.5, 0.25], vec![3]).unwrap(),
        );
        w.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::from_f32(&[9.0], vec![1, 1]).unwrap(),
        );
        fold_gemma_norms(&mut w).unwrap();
        assert_eq!(
            w["model.layers.0.input_layernorm.weight"].to_f32_vec(),
            vec![1.0, 0.5, 1.25]
        );
        // Non-norm weights untouched.
        assert_eq!(
            w["model.layers.0.self_attn.q_proj.weight"].to_f32_vec(),
            vec![9.0]
        );
    }
}
