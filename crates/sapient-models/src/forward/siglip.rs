//! SigLIP vision tower + Idefics3 connector — the vision half of SmolVLM.
//!
//! `SiglipVision::encode` turns preprocessed pixels `[3, S, S]` into visual
//! token embeddings `[n_vis, text_hidden]` ready to splice into the text
//! model's embedding sequence at the `<image>` token positions:
//!
//! 1. patch embedding: `conv2d` stride=patch (existing kernel) + learned
//!    position embeddings — `[n_patches, vision_hidden]`
//! 2. N pre-LN transformer blocks (LayerNorm+bias, full non-causal attention
//!    via an explicit all-zeros mask — same invariant as the Whisper encoder:
//!    `mask=None` means CAUSAL in the CPU kernel — and a `gelu_pytorch_tanh`
//!    MLP, which is the existing `gelu` kernel, NOT `gelu_erf`)
//! 3. `post_layernorm`
//! 4. Idefics3 pixel shuffle (scale s): `[h·w, c]` → `[h/s · w/s, c·s²]` with
//!    the exact transformers ordering `out[hj][wj] = concat_{dh, dw} in[hj·s+dh][wj·s+dw]`
//!    (dh outer, dw inner, channels innermost) — getting this ordering wrong
//!    is this model's token-salad class of bug
//! 5. modality projection: linear `c·s² → text_hidden` (no bias)
//!
//! CPU-only and f32 (the tower is ~93 M params; one 512² image is 1024
//! patches × 12 layers — comfortably fast on the parallel conv/GEMM kernels).

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use sapient_backends_cpu::kernels::attention::scaled_dot_product_attention;
use sapient_backends_cpu::kernels::elementwise::gelu;
use sapient_core::{Shape, Tensor};

use super::backend::{LlmBackend, LlmBackendDispatch, LlmBackendKind};
use super::common::{merge_heads, split_heads};

const LN_EPS: f32 = 1e-6;

/// Vision-tower dimensions (SmolVLM-256M: 768/12L/12H, image 512, patch 16,
/// scale 4, text hidden 576).
#[derive(Debug, Clone)]
pub struct SiglipConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub image_size: usize,
    pub patch: usize,
    pub scale_factor: usize,
    pub text_hidden: usize,
}

impl SiglipConfig {
    pub fn n_patches_side(&self) -> usize {
        self.image_size / self.patch
    }
    pub fn n_patches(&self) -> usize {
        self.n_patches_side() * self.n_patches_side()
    }
    /// Visual tokens after pixel shuffle.
    pub fn n_visual_tokens(&self) -> usize {
        self.n_patches() / (self.scale_factor * self.scale_factor)
    }
}

/// The loaded tower: weight map (keys as in the HF checkpoint, e.g.
/// `model.vision_model.encoder.layers.0.self_attn.q_proj.weight`) + CPU dispatch.
pub struct SiglipVision {
    cfg: SiglipConfig,
    weights: HashMap<String, Tensor>,
    backend: LlmBackendDispatch,
    head_dim: usize,
}

impl SiglipVision {
    pub fn new(cfg: SiglipConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        let head_dim = cfg.hidden / cfg.heads;
        let backend = LlmBackendDispatch::from_kind(LlmBackendKind::Cpu)
            .map_err(|e| anyhow!("vision backend: {e}"))?;
        Ok(Self {
            cfg,
            weights,
            backend,
            head_dim,
        })
    }

    pub fn config(&self) -> &SiglipConfig {
        &self.cfg
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("vision weight missing: {name}"))
    }

    fn opt(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?.clone();
        let b = self.opt(&format!("{prefix}.bias")).cloned();
        self.backend
            .layer_norm(x, &w, b.as_ref(), LN_EPS)
            .map_err(|e| anyhow!("{e}"))
    }

    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?.clone();
        let b = self.opt(&format!("{prefix}.bias")).cloned();
        self.backend
            .linear_3d_bias(x, &w, b.as_ref())
            .map_err(|e| anyhow!("{e}"))
    }

    /// Preprocessed pixels `[3, S, S]` (normalized f32) → visual token
    /// embeddings `[n_visual_tokens · text_hidden]` (row-major).
    pub fn encode(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        let s = self.cfg.image_size;
        let c = self.cfg.hidden;
        let side = self.cfg.n_patches_side();
        let n_patch = self.cfg.n_patches();
        if pixels.len() != 3 * s * s {
            anyhow::bail!("expected {}x{s}x{s} pixels, got {}", 3, pixels.len());
        }

        // ── 1. patch embedding: conv2d stride=patch → [1, c, side, side] ────
        let x = Tensor::from_f32(pixels, Shape::new([1, 3, s, s])).map_err(|e| anyhow!("{e}"))?;
        let pw = self.get("model.vision_model.embeddings.patch_embedding.weight")?;
        let pb = self.opt("model.vision_model.embeddings.patch_embedding.bias");
        let patches = sapient_backends_cpu::kernels::conv2d::conv2d(
            &x,
            pw,
            pb,
            [self.cfg.patch, self.cfg.patch],
            [0, 0, 0, 0],
            [self.cfg.patch, self.cfg.patch],
            [1, 1],
            1,
        )
        .map_err(|e| anyhow!("{e}"))?;
        // [1, c, side, side] → [n_patch, c] (patch-major, channels contiguous).
        let pv = patches.to_f32_vec();
        let mut h = vec![0.0f32; n_patch * c];
        for ci in 0..c {
            for p in 0..n_patch {
                h[p * c + ci] = pv[ci * n_patch + p];
            }
        }
        // + learned position embeddings [n_patch, c].
        let pos = self
            .get("model.vision_model.embeddings.position_embedding.weight")?
            .to_f32_vec();
        if pos.len() != h.len() {
            anyhow::bail!("position embedding {} != patches {}", pos.len(), h.len());
        }
        for (a, b) in h.iter_mut().zip(&pos) {
            *a += b;
        }
        let mut x =
            Tensor::from_f32(&h, Shape::new([1, n_patch, c])).map_err(|e| anyhow!("{e}"))?;

        // ── 2. transformer blocks (pre-LN) ───────────────────────────────────
        let zero_mask = Tensor::from_f32(&vec![0.0f32; n_patch * n_patch], vec![n_patch, n_patch])
            .map_err(|e| anyhow!("{e}"))?;
        for l in 0..self.cfg.layers {
            let p = format!("model.vision_model.encoder.layers.{l}");
            // attn
            let normed = self.layer_norm(&x, &format!("{p}.layer_norm1"))?;
            let q = split_heads(
                &self.linear(&normed, &format!("{p}.self_attn.q_proj"))?,
                self.cfg.heads,
                self.head_dim,
            )?;
            let k = split_heads(
                &self.linear(&normed, &format!("{p}.self_attn.k_proj"))?,
                self.cfg.heads,
                self.head_dim,
            )?;
            let v = split_heads(
                &self.linear(&normed, &format!("{p}.self_attn.v_proj"))?,
                self.cfg.heads,
                self.head_dim,
            )?;
            let attn =
                scaled_dot_product_attention(&q, &k, &v, Some(&zero_mask), None, self.cfg.heads)
                    .map_err(|e| anyhow!("{e}"))?;
            let attn = merge_heads(&attn)?;
            let attn = self.linear(&attn, &format!("{p}.self_attn.out_proj"))?;
            x = self.backend.add(&x, &attn).map_err(|e| anyhow!("{e}"))?;
            // mlp
            let normed = self.layer_norm(&x, &format!("{p}.layer_norm2"))?;
            let up = self.linear(&normed, &format!("{p}.mlp.fc1"))?;
            let up = gelu(&up).map_err(|e| anyhow!("{e}"))?; // gelu_pytorch_tanh
            let down = self.linear(&up, &format!("{p}.mlp.fc2"))?;
            x = self.backend.add(&x, &down).map_err(|e| anyhow!("{e}"))?;
        }
        let x = self.layer_norm(&x, "model.vision_model.post_layernorm")?;

        // ── 3. pixel shuffle: [side, side, c] → [side/s², c·s²] ──────────────
        let sf = self.cfg.scale_factor;
        let out_side = side / sf;
        let cs2 = c * sf * sf;
        let xv = x.to_f32_vec();
        let mut shuffled = vec![0.0f32; out_side * out_side * cs2];
        for hj in 0..out_side {
            for wj in 0..out_side {
                let dst = (hj * out_side + wj) * cs2;
                for dh in 0..sf {
                    for dw in 0..sf {
                        let src_patch = (hj * sf + dh) * side + (wj * sf + dw);
                        let d = dst + (dh * sf + dw) * c;
                        shuffled[d..d + c].copy_from_slice(&xv[src_patch * c..(src_patch + 1) * c]);
                    }
                }
            }
        }

        // ── 4. modality projection: [n_vis, c·s²] → [n_vis, text_hidden] ────
        let n_vis = out_side * out_side;
        let shuffled =
            Tensor::from_f32(&shuffled, Shape::new([1, n_vis, cs2])).map_err(|e| anyhow!("{e}"))?;
        let proj = self.linear(&shuffled, "model.connector.modality_projection.proj")?;
        Ok(proj.to_f32_vec())
    }
}

#[cfg(test)]
mod tests {
    /// The pixel-shuffle ordering must be exactly transformers'
    /// `Idefics3Connector.pixel_shuffle`: out[hj][wj] = concat over dh (outer),
    /// dw (inner) of in[hj·s+dh][wj·s+dw], channels innermost.
    #[test]
    fn pixel_shuffle_ordering_matches_reference() {
        // 4×4 grid, c=1, s=2 → 2×2 output with 4 channels each.
        let side = 4usize;
        let c = 1usize;
        let sf = 2usize;
        let xv: Vec<f32> = (0..side * side).map(|i| i as f32).collect();
        let out_side = side / sf;
        let cs2 = c * sf * sf;
        let mut shuffled = vec![0.0f32; out_side * out_side * cs2];
        for hj in 0..out_side {
            for wj in 0..out_side {
                let dst = (hj * out_side + wj) * cs2;
                for dh in 0..sf {
                    for dw in 0..sf {
                        let src_patch = (hj * sf + dh) * side + (wj * sf + dw);
                        let d = dst + (dh * sf + dw) * c;
                        shuffled[d..d + c].copy_from_slice(&xv[src_patch * c..(src_patch + 1) * c]);
                    }
                }
            }
        }
        // Reference (worked by hand from the transformers view/permute chain):
        // out[0][0] = [in(0,0), in(0,1), in(1,0), in(1,1)] = [0, 1, 4, 5]
        assert_eq!(&shuffled[..4], &[0.0, 1.0, 4.0, 5.0]);
        // out[1][1] = [in(2,2), in(2,3), in(3,2), in(3,3)] = [10, 11, 14, 15]
        assert_eq!(&shuffled[12..16], &[10.0, 11.0, 14.0, 15.0]);
    }
}
