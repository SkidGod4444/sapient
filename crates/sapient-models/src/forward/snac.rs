// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! SNAC neural-audio-codec **decoder** building blocks (Phase 6d, LM-codec TTS).
//!
//! The LM (Orpheus/OuteTTS, a Llama-3.2 run by `LlamaForward`) emits SNAC codec
//! token ids; this module turns those tokens back into a waveform. SNAC's 24 kHz
//! decoder is fully convolutional (no attention/LSTM/ISTFT): RVQ codebook lookup
//! → conv stack with `conv_transpose1d` upsampling + `snake` activations
//! (see [`super::conv`]) → tanh → 24 kHz samples.
//!
//! Weights load from safetensors: either the ungated `mlx-community/snac_24khz`
//! mirror (`model.safetensors`, adapted by [`normalize_snac_weights`]) or the
//! folded output of `scripts/convert_snac_to_safetensors.py`. The decode path is
//! validated bit-close to a torch reference by the `snac_coherence` test.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use sapient_backends_cpu::kernels::elementwise;
use sapient_core::{Shape, Tensor};
use sapient_hub::snac_config::SnacConfig;

use super::conv::{conv1d, conv_transpose1d, snake};

/// Fold PyTorch `weight_norm` parameters into a plain weight tensor at load time
/// (so the decoder convs need no runtime renormalization).
///
/// `weight_norm` factorizes `w = g · v / ‖v‖`, with the norm taken over every
/// dimension except the output-channel axis (dim 0, the default). `v` has the
/// full weight shape `[out, …]`; `g` has one magnitude per output channel
/// (shape `[out, 1, …]` or `[out]`). Returns `w` with `v`'s shape, where each
/// output row has L2 norm exactly `g[o]`.
pub fn weight_norm_fold(v: &Tensor, g: &Tensor) -> Result<Tensor> {
    let dims = v.shape().dims().to_vec();
    let out = *dims
        .first()
        .ok_or_else(|| anyhow!("weight_norm_fold: empty weight"))?;
    let total = dims.iter().product::<usize>();
    if out == 0 || total % out != 0 {
        anyhow::bail!("weight_norm_fold: bad shape {dims:?}");
    }
    let per = total / out; // elements per output channel

    let vv = v.to_f32_vec();
    let gv = g.to_f32_vec();
    if gv.len() != out {
        anyhow::bail!(
            "weight_norm_fold: g has {} magnitudes, expected {out}",
            gv.len()
        );
    }

    let mut w = vec![0.0f32; total];
    for o in 0..out {
        let row = &vv[o * per..(o + 1) * per];
        let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let scale = if norm > 0.0 { gv[o] / norm } else { 0.0 };
        for (i, &x) in row.iter().enumerate() {
            w[o * per + i] = x * scale;
        }
    }
    Tensor::from_f32(&w, v.shape().clone()).map_err(|e| anyhow!("{e}"))
}

/// Swap the last two axes of a 3-D tensor (`[d0, d1, d2]` → `[d0, d2, d1]`).
fn transpose_last_two(t: &Tensor) -> Result<Tensor> {
    let d = t.shape().dims().to_vec();
    if d.len() != 3 {
        anyhow::bail!("transpose_last_two expects a 3-D tensor, got {d:?}");
    }
    let (d0, d1, d2) = (d[0], d[1], d[2]);
    let v = t.to_f32_vec();
    let mut out = vec![0.0f32; v.len()];
    for a in 0..d0 {
        for b in 0..d1 {
            for c in 0..d2 {
                out[(a * d2 + c) * d1 + b] = v[(a * d1 + b) * d2 + c];
            }
        }
    }
    Tensor::from_f32(&out, Shape::new([d0, d2, d1])).map_err(|e| anyhow!("{e}"))
}

/// Normalize raw SNAC codec weights into the exact layout [`SnacDecoder`] reads.
///
/// The released codec is distributed two ways, and this accepts both:
/// - **`mlx-community/snac_24khz`** (`model.safetensors`) stores un-folded
///   `weight_norm` params (`*.weight_g` + `*.weight_v`), conv kernels in MLX
///   channel-last layout (`[d0, K, d1]`), and a `…layers.N…` key prefix on
///   every `nn.Sequential` index. This function folds the weight_norm, swaps
///   each conv kernel's last two axes to PyTorch layout (`[d0, d1, K]`), strips
///   the `.layers.` prefixes, and drops the encoder-only `in_proj.*` params.
/// - **`convert_snac_to_safetensors.py`** output — already folded torch-layout
///   weights — passes through unchanged (no `weight_v` keys present).
pub fn normalize_snac_weights(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    // Already-folded torch weights have no weight_norm params → nothing to do.
    if !raw.keys().any(|k| k.ends_with(".weight_v")) {
        return Ok(raw);
    }
    // MLX wraps each Sequential index as `…layers.N…`; the decoder addresses
    // them as `…N…`. Stripping every `.layers.` segment maps both the outer
    // `decoder.model.layers.2` and the nested `block.layers.3` form.
    let rename = |k: &str| k.replace(".layers.", ".");

    let mut out: HashMap<String, Tensor> = HashMap::new();
    for (k, t) in &raw {
        if k.contains("in_proj") {
            continue; // encoder projection — unused on the decode path
        }
        if k.ends_with(".weight_g") {
            continue; // consumed together with its weight_v sibling
        }
        if let Some(stem) = k.strip_suffix(".weight_v") {
            let g = raw
                .get(&format!("{stem}.weight_g"))
                .ok_or_else(|| anyhow!("SNAC `{k}` has no matching weight_g"))?;
            let folded = weight_norm_fold(t, g)?;
            // weight_norm's per-output-channel norm is axis-0 in both layouts,
            // so fold first (order-independent within a row), then transpose
            // the MLX channel-last kernel to PyTorch kernel-last.
            out.insert(
                rename(&format!("{stem}.weight")),
                transpose_last_two(&folded)?,
            );
        } else {
            // bias / alpha / codebook.weight — copied verbatim (no transpose).
            out.insert(rename(k), t.clone());
        }
    }
    Ok(out)
}

/// SNAC codec decoder: codec token ids → waveform. Fully convolutional
/// (no attention/LSTM/ISTFT). The learned noise block is **omitted** (it injects
/// `randn` excitation, so it is stochastic); the deterministic path is what we
/// validate against the reference and ship first.
///
/// Weights are the folded safetensors produced by
/// `scripts/convert_snac_to_safetensors.py` (`decoder.*` + `quantizer.*`).
pub struct SnacDecoder {
    cfg: SnacConfig,
    w: HashMap<String, Tensor>,
}

impl SnacDecoder {
    pub fn from_weights(cfg: SnacConfig, weights: HashMap<String, Tensor>) -> Self {
        Self { cfg, w: weights }
    }

    pub fn config(&self) -> &SnacConfig {
        &self.cfg
    }

    fn get(&self, key: &str) -> Result<&Tensor> {
        self.w
            .get(key)
            .ok_or_else(|| anyhow!("missing SNAC weight `{key}`"))
    }

    /// Decode RVQ codes (one `Vec<u32>` per codebook level, coarse→fine) into a
    /// mono waveform at `cfg.sampling_rate`.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        if codes.len() != self.cfg.n_codebooks() {
            anyhow::bail!(
                "expected {} code levels, got {}",
                self.cfg.n_codebooks(),
                codes.len()
            );
        }
        // ── Quantizer.from_codes: embed → out_proj (1×1) → repeat_interleave → sum.
        // The continuous latent width = the decoder conv-in's channel count. Some
        // SNAC config.json variants (e.g. the `mlx-community` mirror) omit
        // `latent_dim`, so derive it from the depthwise conv-in weight `[latent,
        // 1, 7]` rather than mis-defaulting to `codebook_dim`.
        let latent = match self.cfg.latent_dim {
            Some(d) => d,
            None => self.get("decoder.model.0.weight")?.shape().dims()[0],
        };
        let base_t = codes[0].len() * self.cfg.vq_strides[0];
        let mut z = vec![0.0f32; latent * base_t]; // [latent, base_t]

        for (i, level) in codes.iter().enumerate() {
            let stride = self.cfg.vq_strides[i];
            let t_i = level.len();
            if t_i * stride != base_t {
                anyhow::bail!("level {i} length {t_i}*{stride} != base {base_t}");
            }
            let cb = self.get(&format!("quantizer.quantizers.{i}.codebook.weight"))?; // [size, dim]
            let cb_v = cb.to_f32_vec();
            let cd = self.cfg.codebook_dim;
            // Gather → [dim, t_i] (channels-major) for conv1d.
            let mut emb = vec![0.0f32; cd * t_i];
            for (t, &code) in level.iter().enumerate() {
                let row = code as usize * cd;
                if row + cd > cb_v.len() {
                    anyhow::bail!("code {code} out of codebook range (level {i})");
                }
                for c in 0..cd {
                    emb[c * t_i + t] = cb_v[row + c];
                }
            }
            let emb_t =
                Tensor::from_f32(&emb, Shape::new([1, cd, t_i])).map_err(|e| anyhow!("{e}"))?;
            let ow = self
                .get(&format!("quantizer.quantizers.{i}.out_proj.weight"))?
                .clone();
            let ob = self
                .get(&format!("quantizer.quantizers.{i}.out_proj.bias"))?
                .clone();
            let zqi = conv1d(&emb_t, &ow, Some(&ob), 0, 1, 1, 1)?; // [1, latent, t_i]
            let zqi = repeat_interleave_time(&zqi, stride)?; // [1, latent, base_t]
            let zqi_v = zqi.as_f32_slice();
            for (acc, &v) in z.iter_mut().zip(zqi_v) {
                *acc += v;
            }
        }

        // ── Decoder.
        let mut x =
            Tensor::from_f32(&z, Shape::new([1, latent, base_t])).map_err(|e| anyhow!("{e}"))?;
        // model.0: depthwise conv (groups = latent), k=7, pad=3.
        x = conv1d(
            &x,
            &self.get("decoder.model.0.weight")?.clone(),
            self.opt_bias("decoder.model.0.bias").as_ref(),
            3,
            1,
            1,
            latent,
        )?;
        // model.1: 1×1 conv latent → decoder_dim.
        x = conv1d(
            &x,
            &self.get("decoder.model.1.weight")?.clone(),
            self.opt_bias("decoder.model.1.bias").as_ref(),
            0,
            1,
            1,
            1,
        )?;

        // model.2..: one DecoderBlock per upsample rate.
        for (bi, &stride) in self.cfg.decoder_rates.iter().enumerate() {
            let p = format!("decoder.model.{}", 2 + bi);
            x = self.snake_at(&x, &format!("{p}.block.0.alpha"))?;
            // ConvTranspose1d upsample (k = 2*stride, pad = ceil(stride/2)).
            x = conv_transpose1d(
                &x,
                &self.get(&format!("{p}.block.1.weight"))?.clone(),
                self.opt_bias(&format!("{p}.block.1.bias")).as_ref(),
                stride,
                stride.div_ceil(2),
            )?;
            // block.2 = NoiseBlock — omitted (deterministic path).
            // block.3/4/5 = ResidualUnit with dilation 1/3/9.
            for (ri, dil) in [(3usize, 1usize), (4, 3), (5, 9)] {
                x = self.residual_unit(&x, &format!("{p}.block.{ri}"), dil)?;
            }
        }

        // Final Snake → conv (→ 1 channel) → tanh.
        let last = format!("decoder.model.{}", 2 + self.cfg.decoder_rates.len());
        x = self.snake_at(&x, &format!("{last}.alpha"))?;
        x = conv1d(
            &x,
            &self
                .get(&format!(
                    "decoder.model.{}.weight",
                    3 + self.cfg.decoder_rates.len()
                ))?
                .clone(),
            self.opt_bias(&format!(
                "decoder.model.{}.bias",
                3 + self.cfg.decoder_rates.len()
            ))
            .as_ref(),
            3,
            1,
            1,
            1,
        )?;
        let x = elementwise::tanh_act(&x).map_err(|e| anyhow!("{e}"))?;
        Ok(x.to_f32_vec())
    }

    fn opt_bias(&self, key: &str) -> Option<Tensor> {
        self.w.get(key).cloned()
    }

    fn snake_at(&self, x: &Tensor, alpha_key: &str) -> Result<Tensor> {
        let alpha = self.get(alpha_key)?.clone();
        snake(x, &alpha)
    }

    /// SNAC ResidualUnit: `x + conv1x1(snake(dwconv_dilated(snake(x))))`.
    /// The depthwise dilated conv uses same-padding (`3*dil`), so lengths match
    /// and the residual add needs no trimming.
    fn residual_unit(&self, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
        let dim = x.shape().dims()[1];
        let y = self.snake_at(x, &format!("{prefix}.block.0.alpha"))?;
        let y = conv1d(
            &y,
            &self.get(&format!("{prefix}.block.1.weight"))?.clone(),
            self.opt_bias(&format!("{prefix}.block.1.bias")).as_ref(),
            3 * dilation,
            1,
            dilation,
            dim, // depthwise
        )?;
        let y = self.snake_at(&y, &format!("{prefix}.block.2.alpha"))?;
        let y = conv1d(
            &y,
            &self.get(&format!("{prefix}.block.3.weight"))?.clone(),
            self.opt_bias(&format!("{prefix}.block.3.bias")).as_ref(),
            0,
            1,
            1,
            1,
        )?;
        elementwise::add(x, &y).map_err(|e| anyhow!("{e}"))
    }
}

/// Codebook size per RVQ level for the Orpheus token layout (SNAC 24 kHz).
const ORPHEUS_CODEBOOK: u32 = 4096;

/// Convert an Orpheus audio-token stream into the 3 SNAC code levels.
///
/// Orpheus emits **7 codes per frame**, each offset by its position
/// (`code_k = token − audio_base − k·4096`), interleaving SNAC's hierarchy:
/// positions `0`→level0 (1/frame), `{1,4}`→level1 (2/frame), `{2,3,5,6}`→level2
/// (4/frame) — i.e. SNAC `vq_strides = [4, 2, 1]`. `audio_codes` are the raw
/// token ids with `audio_base` already subtracted (values in `0..7·4096`).
/// Trailing partial frames are dropped. Returns `[level0, level1, level2]`.
pub fn orpheus_codes_to_snac(audio_codes: &[u32]) -> Result<[Vec<u32>; 3]> {
    let frames = audio_codes.len() / 7;
    let (mut l0, mut l1, mut l2) = (Vec::new(), Vec::new(), Vec::new());
    let sub = |v: u32, k: u32| -> Result<u32> {
        v.checked_sub(k * ORPHEUS_CODEBOOK)
            .filter(|&c| c < ORPHEUS_CODEBOOK)
            .ok_or_else(|| anyhow!("Orpheus code {v} out of range for position {k}"))
    };
    for i in 0..frames {
        let f = &audio_codes[7 * i..7 * i + 7];
        l0.push(sub(f[0], 0)?);
        l1.push(sub(f[1], 1)?);
        l2.push(sub(f[2], 2)?);
        l2.push(sub(f[3], 3)?);
        l1.push(sub(f[4], 4)?);
        l2.push(sub(f[5], 5)?);
        l2.push(sub(f[6], 6)?);
    }
    Ok([l0, l1, l2])
}

/// Repeat each time step `stride` times along the last axis of `[1, C, T]`.
fn repeat_interleave_time(x: &Tensor, stride: usize) -> Result<Tensor> {
    let d = x.shape().dims().to_vec();
    let (c, t) = (d[1], d[2]);
    let xv = x.as_f32_slice();
    let mut out = vec![0.0f32; c * t * stride];
    for ch in 0..c {
        for ti in 0..t {
            let v = xv[ch * t + ti];
            let base = ch * t * stride + ti * stride;
            for r in 0..stride {
                out[base + r] = v;
            }
        }
    }
    Tensor::from_f32(&out, Shape::new([1, c, t * stride])).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_gives_rows_with_norm_g() {
        // v: [out=3, in=2, k=4]; g: [out, 1, 1].
        let (out, inn, k) = (3usize, 2usize, 4usize);
        let per = inn * k;
        let v: Vec<f32> = (0..out * per)
            .map(|i| (i as f32 * 0.31 + 0.5).sin())
            .collect();
        let g: Vec<f32> = vec![0.5, 1.0, 2.0];
        let vt = Tensor::from_f32(&v, vec![out, inn, k]).unwrap();
        let gt = Tensor::from_f32(&g, vec![out, 1, 1]).unwrap();

        let w = weight_norm_fold(&vt, &gt).unwrap();
        assert_eq!(w.shape().dims(), &[out, inn, k]);
        let wv = w.as_f32_slice();
        for o in 0..out {
            let norm = wv[o * per..(o + 1) * per]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt();
            // ‖w[o]‖ must equal g[o] (the weight_norm invariant).
            assert!(
                (norm - g[o]).abs() < 1e-5,
                "row {o}: norm {norm} != g {}",
                g[o]
            );
        }
    }

    #[test]
    fn zero_direction_row_folds_to_zero() {
        let vt = Tensor::from_f32(&[0.0, 0.0, 1.0, 1.0], vec![2, 1, 2]).unwrap();
        let gt = Tensor::from_f32(&[3.0, 3.0], vec![2, 1, 1]).unwrap();
        let w = weight_norm_fold(&vt, &gt).unwrap();
        let wv = w.as_f32_slice();
        assert_eq!(&wv[0..2], &[0.0, 0.0]); // zero v → zero w (no div-by-zero)
    }

    #[test]
    fn orpheus_deframe_splits_7_codes_into_3_levels() {
        let cb = ORPHEUS_CODEBOOK;
        // One frame with per-position offsets applied (as Orpheus emits them).
        let frame = [
            10,
            cb + 11,
            2 * cb + 12,
            3 * cb + 13,
            4 * cb + 14,
            5 * cb + 15,
            6 * cb + 16,
        ];
        let [l0, l1, l2] = orpheus_codes_to_snac(&frame).unwrap();
        assert_eq!(l0, vec![10]); // level0: 1/frame
        assert_eq!(l1, vec![11, 14]); // level1: positions 1,4
        assert_eq!(l2, vec![12, 13, 15, 16]); // level2: positions 2,3,5,6
                                              // Two frames → lengths [2,4,8] (matches vq_strides [4,2,1] base ratios).
        let two: Vec<u32> = frame.iter().chain(frame.iter()).copied().collect();
        let [a, b, c] = orpheus_codes_to_snac(&two).unwrap();
        assert_eq!((a.len(), b.len(), c.len()), (2, 4, 8));
    }

    #[test]
    fn transpose_last_two_swaps_kernel_axis() {
        // [d0=2, d1=3, d2=2] → [2, 2, 3]; element (a,b,c) → (a,c,b).
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::from_f32(&v, vec![2, 3, 2]).unwrap();
        let r = transpose_last_two(&t).unwrap();
        assert_eq!(r.shape().dims(), &[2, 2, 3]);
        let rv = r.as_f32_slice();
        for a in 0..2 {
            for b in 0..3 {
                for c in 0..2 {
                    assert_eq!(rv[(a * 2 + c) * 3 + b], v[(a * 3 + b) * 2 + c]);
                }
            }
        }
    }

    #[test]
    fn normalize_folds_renames_and_transposes_mlx_export() {
        // Mimic one MLX-export conv: `decoder.model.layers.1.weight_{g,v}` in
        // channel-last layout [out, K, in] = [2, 1, 3] (a 1×1 conv, in=3→out=2).
        let mut raw: HashMap<String, Tensor> = HashMap::new();
        let v = Tensor::from_f32(&[1.0, 2.0, 2.0, 0.0, 3.0, 4.0], vec![2, 1, 3]).unwrap();
        let g = Tensor::from_f32(&[3.0, 5.0], vec![2, 1, 1]).unwrap();
        raw.insert("decoder.model.layers.1.weight_v".into(), v);
        raw.insert("decoder.model.layers.1.weight_g".into(), g);
        raw.insert(
            "decoder.model.layers.1.bias".into(),
            Tensor::from_f32(&[0.5, -0.5], vec![2]).unwrap(),
        );
        // Encoder-only key must be dropped.
        raw.insert(
            "quantizer.quantizers.0.in_proj.weight_v".into(),
            Tensor::from_f32(&[0.0], vec![1, 1, 1]).unwrap(),
        );
        raw.insert(
            "quantizer.quantizers.0.in_proj.weight_g".into(),
            Tensor::from_f32(&[1.0], vec![1, 1, 1]).unwrap(),
        );

        let out = normalize_snac_weights(raw).unwrap();
        // Renamed (no `.layers.`) + folded weight present; bias renamed too.
        let w = out.get("decoder.model.1.weight").expect("folded weight");
        assert_eq!(w.shape().dims(), &[2, 3, 1]); // transposed to [out, in, K]
        assert!(out.contains_key("decoder.model.1.bias"));
        assert!(!out.keys().any(|k| k.contains("in_proj")));
        assert!(!out
            .keys()
            .any(|k| k.contains(".layers.") || k.ends_with(".weight_v")));
        // Row norms equal g (weight_norm invariant, preserved through transpose).
        let wv = w.as_f32_slice();
        for (o, &gg) in [3.0f32, 5.0].iter().enumerate() {
            let n = (wv[o * 3] * wv[o * 3]
                + wv[o * 3 + 1] * wv[o * 3 + 1]
                + wv[o * 3 + 2] * wv[o * 3 + 2])
                .sqrt();
            assert!((n - gg).abs() < 1e-5, "row {o} norm {n} != {gg}");
        }
    }

    #[test]
    fn normalize_passes_through_already_folded() {
        let mut raw: HashMap<String, Tensor> = HashMap::new();
        raw.insert(
            "decoder.model.1.weight".into(),
            Tensor::from_f32(&[1.0, 2.0], vec![1, 2, 1]).unwrap(),
        );
        let out = normalize_snac_weights(raw.clone()).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out.contains_key("decoder.model.1.weight"));
    }

    #[test]
    fn orpheus_deframe_rejects_out_of_range() {
        // A code at position 0 that's ≥ 4096 (would belong to a later position).
        assert!(orpheus_codes_to_snac(&[5000, 0, 0, 0, 0, 0, 0]).is_err());
    }
}
