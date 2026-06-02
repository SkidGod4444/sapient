//! SNAC neural-audio-codec **decoder** building blocks (Phase 6d, LM-codec TTS).
//!
//! The LM (Orpheus/OuteTTS, a Llama-3.2 run by `LlamaForward`) emits SNAC codec
//! token ids; this module turns those tokens back into a waveform. SNAC's 24 kHz
//! decoder is fully convolutional (no attention/LSTM/ISTFT): RVQ codebook lookup
//! → conv stack with `conv_transpose1d` upsampling + `snake` activations
//! (see [`super::conv`]) → tanh → 24 kHz samples.
//!
//! This file currently provides the load-time helpers that are independently
//! verifiable without the model weights; the full decoder forward is wired once
//! the SNAC weights are converted to safetensors (the released checkpoint is a
//! `.pth` pickle SAPIENT can't load directly).

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
        let latent = self.cfg.latent_dim.unwrap_or(self.cfg.codebook_dim);
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
}
