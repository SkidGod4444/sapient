// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! The ProsodyPredictor: from the ALBERT features + the prosody style vector it
//! predicts per-token durations (→ the hard alignment that length-regulates the
//! sequence) and the per-frame F0 (pitch) and N (energy) curves the decoder
//! consumes. Built from bidirectional LSTMs, AdaLayerNorm, and AdaIN+LeakyReLU
//! residual blocks (`AdainResBlk1d`) — all in [`super::ops`].

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use sapient_core::{Shape, Tensor};

use super::super::conv::conv1d;
use super::loader::KokoroConfig;
use super::ops::{
    ada_in_1d, ada_layer_norm, convtr_depthwise_up2, leaky_relu_inplace, length_regulate, linear2d,
    lstm_bidirectional, upsample_nearest_x2, LstmParams,
};

fn get<'a>(w: &'a HashMap<String, Tensor>, k: &str) -> Result<&'a Tensor> {
    w.get(k)
        .ok_or_else(|| anyhow!("kokoro: missing weight {k}"))
}

/// Transpose a row-major `[r, c]` buffer to `[c, r]`.
fn transpose(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = x[i * c + j];
        }
    }
    out
}

/// Build forward+backward LSTM params for a PyTorch `nn.LSTM` at `prefix`.
struct Lstm<'a> {
    fwd: LstmParams<'a>,
    bwd: LstmParams<'a>,
}
fn lstm_at<'a>(w: &'a HashMap<String, Tensor>, prefix: &str) -> Result<Lstm<'a>> {
    let g = |suf: &str| get(w, &format!("{prefix}.{suf}"));
    Ok(Lstm {
        fwd: LstmParams {
            weight_ih: g("weight_ih_l0")?,
            weight_hh: g("weight_hh_l0")?,
            bias_ih: Some(g("bias_ih_l0")?),
            bias_hh: Some(g("bias_hh_l0")?),
        },
        bwd: LstmParams {
            weight_ih: g("weight_ih_l0_reverse")?,
            weight_hh: g("weight_hh_l0_reverse")?,
            bias_ih: Some(g("bias_ih_l0_reverse")?),
            bias_hh: Some(g("bias_hh_l0_reverse")?),
        },
    })
}

/// One `AdainResBlk1d.forward(x, s)` — `x` is `[1, dim_in, L]`, returns
/// `[1, dim_out, L']` (`L' = 2L` when `upsample`). LeakyReLU(0.2) activations.
/// Shared with the decoder (`encode`/`decode` blocks).
pub(super) fn adain_res_blk1d(
    w: &HashMap<String, Tensor>,
    prefix: &str,
    x: &Tensor,
    s: &[f32],
    upsample: bool,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (dim_in, _l) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let conv1_w = get(w, &format!("{prefix}.conv1.weight"))?;
    let dim_out = conv1_w.shape().dims()[0];
    let learned_sc = dim_in != dim_out;
    let eps = 1e-5f32;

    // ── residual path ────────────────────────────────────────────────────────
    let xt = ada_in_1d(
        x,
        s,
        get(w, &format!("{prefix}.norm1.fc.weight"))?,
        Some(get(w, &format!("{prefix}.norm1.fc.bias"))?),
        eps,
    )?;
    let mut xt_v = xt.to_f32_vec();
    leaky_relu_inplace(&mut xt_v, 0.2);
    let cur_l = xt.shape().dims()[xt.shape().dims().len() - 1];
    let mut xt =
        Tensor::from_f32(&xt_v, Shape::new([1, dim_in, cur_l])).map_err(|e| anyhow!("{e}"))?;
    if upsample {
        xt = convtr_depthwise_up2(
            &xt,
            get(w, &format!("{prefix}.pool.weight"))?,
            Some(get(w, &format!("{prefix}.pool.bias"))?),
        )?;
    }
    let xt = conv1d(
        &xt,
        conv1_w,
        Some(get(w, &format!("{prefix}.conv1.bias"))?),
        1,
        1,
        1,
        1,
    )?;
    let xt = ada_in_1d(
        &xt,
        s,
        get(w, &format!("{prefix}.norm2.fc.weight"))?,
        Some(get(w, &format!("{prefix}.norm2.fc.bias"))?),
        eps,
    )?;
    let mut xt_v = xt.to_f32_vec();
    leaky_relu_inplace(&mut xt_v, 0.2);
    let l2 = xt.shape().dims()[xt.shape().dims().len() - 1];
    let xt = Tensor::from_f32(&xt_v, Shape::new([1, dim_out, l2])).map_err(|e| anyhow!("{e}"))?;
    let residual = conv1d(
        &xt,
        get(w, &format!("{prefix}.conv2.weight"))?,
        Some(get(w, &format!("{prefix}.conv2.bias"))?),
        1,
        1,
        1,
        1,
    )?;

    // ── shortcut path ────────────────────────────────────────────────────────
    let mut sc = if upsample {
        upsample_nearest_x2(x)?
    } else {
        x.clone()
    };
    if learned_sc {
        // conv1x1, bias=False
        sc = conv1d(
            &sc,
            get(w, &format!("{prefix}.conv1x1.weight"))?,
            None,
            0,
            1,
            1,
            1,
        )?;
    }

    // out = (residual + shortcut) / sqrt(2)
    let rv = residual.to_f32_vec();
    let sv = sc.to_f32_vec();
    let inv = 1.0 / (2.0f32).sqrt();
    let out: Vec<f32> = rv
        .iter()
        .zip(sv.iter())
        .map(|(a, b)| (a + b) * inv)
        .collect();
    Tensor::from_f32(&out, residual.shape().clone()).map_err(|e| anyhow!("{e}"))
}

/// Output of the prosody prediction stage.
pub struct Prosody {
    pub pred_dur: Vec<usize>,
    /// length-regulated DurationEncoder features `[640, T]` (the `en` tensor).
    pub en: Tensor,
}

/// DurationEncoder + duration projection + alignment. `d_en` is `[hidden, L]`
/// (the transposed `bert_encoder` output), `style` the 128-d prosody style.
pub fn predict_prosody(
    w: &HashMap<String, Tensor>,
    cfg: &KokoroConfig,
    d_en: &[f32],
    l: usize,
    style: &[f32],
    speed: f32,
) -> Result<Prosody> {
    let h = cfg.hidden_dim; // 512
    let sty = cfg.style_dim; // 128
    let feat = h + sty; // 640

    // ── DurationEncoder: x starts as [640, L] = cat(d_en[512,L], style broadcast)
    let mut x = vec![0.0f32; feat * l]; // [C, L] — always [feat, L] at each loop top
    for c in 0..h {
        for t in 0..l {
            x[c * l + t] = d_en[c * l + t];
        }
    }
    for c in 0..sty {
        for t in 0..l {
            x[(h + c) * l + t] = style[c];
        }
    }

    for layer in 0..cfg.n_layer {
        // LSTM block (lstms.{2*layer}): [640,L] → [L,640] → BiLSTM → [L,512] → [512,L]
        let xt = transpose(&x, feat, l); // [L, 640]
        let xt = Tensor::from_f32(&xt, Shape::new([l, feat])).map_err(|e| anyhow!("{e}"))?;
        let lp = lstm_at(w, &format!("predictor.text_encoder.lstms.{}", 2 * layer))?;
        let y = lstm_bidirectional(&xt, &lp.fwd, &lp.bwd)?; // [L, 512]
        let yv = y.to_f32_vec();
        x = transpose(&yv, l, h); // [512, L]

        // AdaLayerNorm block (lstms.{2*layer+1}): on [512,L] → cat style → [640,L]
        let xt = transpose(&x, h, l); // [L, 512]
        let xt = Tensor::from_f32(&xt, Shape::new([l, h])).map_err(|e| anyhow!("{e}"))?;
        let normed = ada_layer_norm(
            &xt,
            style,
            get(
                w,
                &format!("predictor.text_encoder.lstms.{}.fc.weight", 2 * layer + 1),
            )?,
            Some(get(
                w,
                &format!("predictor.text_encoder.lstms.{}.fc.bias", 2 * layer + 1),
            )?),
            1e-5,
        )?; // [L, 512]
        let nv = normed.to_f32_vec();
        let nt = transpose(&nv, l, h); // [512, L]
        let mut nx = vec![0.0f32; feat * l];
        nx[..h * l].copy_from_slice(&nt);
        for c in 0..sty {
            for t in 0..l {
                nx[(h + c) * l + t] = style[c];
            }
        }
        x = nx;
    }
    // d = x.transpose(-1,-2): [640, L] → d [L, 640]
    let d = transpose(&x, feat, l); // [L, 640]

    // ── duration: predictor.lstm(d) → [L,512]; duration_proj → [L,50]
    let dt = Tensor::from_f32(&d, Shape::new([l, feat])).map_err(|e| anyhow!("{e}"))?;
    let lp = lstm_at(w, "predictor.lstm")?;
    let dur_h = lstm_bidirectional(&dt, &lp.fwd, &lp.bwd)?; // [L, 512]
    let dur_hv = dur_h.to_f32_vec();
    let dp_w = get(w, "predictor.duration_proj.linear_layer.weight")?.to_f32_vec();
    let dp_b = get(w, "predictor.duration_proj.linear_layer.bias")?.to_f32_vec();
    let dproj = linear2d(&dur_hv, l, h, &dp_w, Some(&dp_b), cfg.max_dur); // [L, 50]
                                                                          // duration = sigmoid(.).sum(-1)/speed; pred_dur = round.clamp(min=1)
    let mut pred_dur = vec![0usize; l];
    for t in 0..l {
        let mut s = 0.0f32;
        for k in 0..cfg.max_dur {
            s += 1.0 / (1.0 + (-dproj[t * cfg.max_dur + k]).exp());
        }
        let dur = (s / speed).round().max(1.0);
        pred_dur[t] = dur as usize;
    }

    // en = d.transpose(-1,-2) @ aln = length_regulate(d_transposed [640,L], pred_dur)
    let d_tr = transpose(&d, l, feat); // [640, L]
    let d_tr = Tensor::from_f32(&d_tr, Shape::new([1, feat, l])).map_err(|e| anyhow!("{e}"))?;
    let en = length_regulate(&d_tr, &pred_dur)?; // [1, 640, T]

    Ok(Prosody { pred_dur, en })
}

/// `ProsodyPredictor.F0Ntrain(en, s)` → (F0_pred, N_pred), each `[2T]`.
pub fn f0_n_train(
    w: &HashMap<String, Tensor>,
    cfg: &KokoroConfig,
    en: &Tensor,
    style: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let feat = cfg.hidden_dim + cfg.style_dim; // 640
    let t = en.shape().dims()[en.shape().dims().len() - 1];
    let h = cfg.hidden_dim; // 512

    // shared LSTM: en [640,T] → [T,640] → BiLSTM → [T,512] → x [512,T]
    let en_v = en.to_f32_vec();
    let en_t = transpose(&en_v, feat, t); // [T, 640]
    let en_t = Tensor::from_f32(&en_t, Shape::new([t, feat])).map_err(|e| anyhow!("{e}"))?;
    let lp = lstm_at(w, "predictor.shared")?;
    let shared = lstm_bidirectional(&en_t, &lp.fwd, &lp.bwd)?; // [T, 512]
    let shared_v = shared.to_f32_vec();
    let x0 = transpose(&shared_v, t, h); // [512, T]
    let x0 = Tensor::from_f32(&x0, Shape::new([1, h, t])).map_err(|e| anyhow!("{e}"))?;

    let run = |branch: &str| -> Result<Vec<f32>> {
        // blocks: {branch}.0 (512→512), .1 (512→256, upsample), .2 (256→256)
        let mut x = adain_res_blk1d(w, &format!("predictor.{branch}.0"), &x0, style, false)?;
        x = adain_res_blk1d(w, &format!("predictor.{branch}.1"), &x, style, true)?;
        x = adain_res_blk1d(w, &format!("predictor.{branch}.2"), &x, style, false)?;
        // proj: Conv1d(256,1,1) plain → [1,1,2T] → [2T]
        let proj = conv1d(
            &x,
            get(w, &format!("predictor.{branch}_proj.weight"))?,
            Some(get(w, &format!("predictor.{branch}_proj.bias"))?),
            0,
            1,
            1,
            1,
        )?;
        Ok(proj.to_f32_vec())
    };
    let f0 = run("F0")?;
    let n = run("N")?;
    Ok((f0, n))
}
