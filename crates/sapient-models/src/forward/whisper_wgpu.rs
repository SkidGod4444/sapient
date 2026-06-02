//! Cross-platform GPU Whisper engine (wgpu/WGSL — Vulkan/DX12/Metal).
//!
//! Mirrors the CPU [`super::whisper::WhisperForward`] but runs the transformer
//! body on the GPU: weights upload once to storage buffers; the encoder and
//! decoder blocks (LayerNorm+bias, q/k/v/out and fc projections, attention,
//! exact-erf GELU, residual adds) execute on-device. The mel front-end and the
//! conv stem stay on the CPU (they run once per 30 s chunk and are cheap), and
//! only the final logits are read back per decode step.
//!
//! Weights are uploaded as **f32** (`to_f32_vec`) — the wgpu matmul reads f32,
//! and this avoids the lossy round-trip of online-Q8_0 quantization.
//!
//! Attention reuses the resident `attention` kernel with the new `causal` flag:
//! `false` for the non-causal encoder self-attn and decoder cross-attn,
//! `true` for the causal decoder self-attn (offset by the cached prefix).

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use sapient_backends_cpu::kernels::elementwise::gelu_erf as cpu_gelu_erf;
use sapient_backends_wgpu::{GpuBuffer, WgpuContext};
use sapient_core::Tensor;
use sapient_hub::whisper_config::WhisperConfig;

use super::conv::conv1d;

const LN_EPS: f32 = 1e-5;

struct EncLayer {
    ln1_w: GpuBuffer,
    ln1_b: GpuBuffer,
    wq: GpuBuffer,
    bq: GpuBuffer,
    wk: GpuBuffer,
    wv: GpuBuffer,
    bv: GpuBuffer,
    wo: GpuBuffer,
    bo: GpuBuffer,
    ln2_w: GpuBuffer,
    ln2_b: GpuBuffer,
    fc1_w: GpuBuffer,
    fc1_b: GpuBuffer,
    fc2_w: GpuBuffer,
    fc2_b: GpuBuffer,
}

struct DecLayer {
    // self-attention
    sln_w: GpuBuffer,
    sln_b: GpuBuffer,
    sq: GpuBuffer,
    sbq: GpuBuffer,
    sk: GpuBuffer,
    sv: GpuBuffer,
    sbv: GpuBuffer,
    so: GpuBuffer,
    sbo: GpuBuffer,
    // cross-attention
    cln_w: GpuBuffer,
    cln_b: GpuBuffer,
    cq: GpuBuffer,
    cbq: GpuBuffer,
    ck: GpuBuffer,
    cv: GpuBuffer,
    cbv: GpuBuffer,
    co: GpuBuffer,
    cbo: GpuBuffer,
    // MLP
    fln_w: GpuBuffer,
    fln_b: GpuBuffer,
    fc1_w: GpuBuffer,
    fc1_b: GpuBuffer,
    fc2_w: GpuBuffer,
    fc2_b: GpuBuffer,
    // caches (GPU-resident)
    self_k: GpuBuffer, // [n_head, max_target, head_dim]
    self_v: GpuBuffer,
    cross_k: Option<GpuBuffer>, // [n_head, n_audio_ctx, head_dim] (set per chunk)
    cross_v: Option<GpuBuffer>,
}

/// GPU-resident Whisper engine.
pub struct WhisperWgpuEngine {
    ctx: WgpuContext,
    cfg: WhisperConfig,
    head_dim: usize,
    // CPU conv stem weights (front-end stays on CPU).
    conv1_w: Tensor,
    conv1_b: Option<Tensor>,
    conv2_w: Tensor,
    conv2_b: Option<Tensor>,
    // encoder
    enc_pos: GpuBuffer, // [n_audio_ctx, d]
    enc_layers: Vec<EncLayer>,
    enc_ln_w: GpuBuffer,
    enc_ln_b: GpuBuffer,
    // decoder
    tok_embed: GpuBuffer, // [vocab, d]
    dec_pos: GpuBuffer,   // [max_target, d]
    dec_layers: Vec<DecLayer>,
    dec_ln_w: GpuBuffer,
    dec_ln_b: GpuBuffer,
    proj_out: GpuBuffer, // [vocab, d] (tied to tok_embed)
    // decoder self-attention cache length
    decoder_len: usize,
    audio_ctx_len: usize,
}

fn upload(ctx: &WgpuContext, t: &Tensor, label: &str) -> GpuBuffer {
    ctx.upload_f32(&t.to_f32_vec(), label)
}

impl WhisperWgpuEngine {
    /// Build from a raw (un-quantized) HF Whisper weight map.
    pub fn from_weights(cfg: WhisperConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        let ctx = WgpuContext::new().context("no wgpu GPU adapter available")?;
        let prefix = if weights.contains_key("model.encoder.conv1.weight") {
            "model.".to_string()
        } else {
            String::new()
        };
        let get = |key: &str| -> Result<&Tensor> {
            let full = format!("{prefix}{key}");
            weights
                .get(&full)
                .ok_or_else(|| anyhow!("missing Whisper weight `{full}`"))
        };
        let opt = |key: &str| weights.get(&format!("{prefix}{key}"));
        let g = |key: &str| -> Result<GpuBuffer> { Ok(upload(&ctx, get(key)?, key)) };

        let head_dim = cfg.head_dim();
        let max_target = cfg.max_target_positions;
        let n_head = cfg.encoder_attention_heads;

        // Encoder layers.
        let mut enc_layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            let p = format!("encoder.layers.{i}");
            enc_layers.push(EncLayer {
                ln1_w: g(&format!("{p}.self_attn_layer_norm.weight"))?,
                ln1_b: g(&format!("{p}.self_attn_layer_norm.bias"))?,
                wq: g(&format!("{p}.self_attn.q_proj.weight"))?,
                bq: g(&format!("{p}.self_attn.q_proj.bias"))?,
                wk: g(&format!("{p}.self_attn.k_proj.weight"))?,
                wv: g(&format!("{p}.self_attn.v_proj.weight"))?,
                bv: g(&format!("{p}.self_attn.v_proj.bias"))?,
                wo: g(&format!("{p}.self_attn.out_proj.weight"))?,
                bo: g(&format!("{p}.self_attn.out_proj.bias"))?,
                ln2_w: g(&format!("{p}.final_layer_norm.weight"))?,
                ln2_b: g(&format!("{p}.final_layer_norm.bias"))?,
                fc1_w: g(&format!("{p}.fc1.weight"))?,
                fc1_b: g(&format!("{p}.fc1.bias"))?,
                fc2_w: g(&format!("{p}.fc2.weight"))?,
                fc2_b: g(&format!("{p}.fc2.bias"))?,
            });
        }

        // Decoder layers.
        let mut dec_layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let p = format!("decoder.layers.{i}");
            dec_layers.push(DecLayer {
                sln_w: g(&format!("{p}.self_attn_layer_norm.weight"))?,
                sln_b: g(&format!("{p}.self_attn_layer_norm.bias"))?,
                sq: g(&format!("{p}.self_attn.q_proj.weight"))?,
                sbq: g(&format!("{p}.self_attn.q_proj.bias"))?,
                sk: g(&format!("{p}.self_attn.k_proj.weight"))?,
                sv: g(&format!("{p}.self_attn.v_proj.weight"))?,
                sbv: g(&format!("{p}.self_attn.v_proj.bias"))?,
                so: g(&format!("{p}.self_attn.out_proj.weight"))?,
                sbo: g(&format!("{p}.self_attn.out_proj.bias"))?,
                cln_w: g(&format!("{p}.encoder_attn_layer_norm.weight"))?,
                cln_b: g(&format!("{p}.encoder_attn_layer_norm.bias"))?,
                cq: g(&format!("{p}.encoder_attn.q_proj.weight"))?,
                cbq: g(&format!("{p}.encoder_attn.q_proj.bias"))?,
                ck: g(&format!("{p}.encoder_attn.k_proj.weight"))?,
                cv: g(&format!("{p}.encoder_attn.v_proj.weight"))?,
                cbv: g(&format!("{p}.encoder_attn.v_proj.bias"))?,
                co: g(&format!("{p}.encoder_attn.out_proj.weight"))?,
                cbo: g(&format!("{p}.encoder_attn.out_proj.bias"))?,
                fln_w: g(&format!("{p}.final_layer_norm.weight"))?,
                fln_b: g(&format!("{p}.final_layer_norm.bias"))?,
                fc1_w: g(&format!("{p}.fc1.weight"))?,
                fc1_b: g(&format!("{p}.fc1.bias"))?,
                fc2_w: g(&format!("{p}.fc2.weight"))?,
                fc2_b: g(&format!("{p}.fc2.bias"))?,
                self_k: ctx.alloc_f32(n_head * max_target * head_dim, "self_k"),
                self_v: ctx.alloc_f32(n_head * max_target * head_dim, "self_v"),
                cross_k: None,
                cross_v: None,
            });
        }

        let tok_embed = g("decoder.embed_tokens.weight")?;
        let proj_out = match opt("proj_out.weight") {
            Some(t) => upload(&ctx, t, "proj_out"),
            None => g("decoder.embed_tokens.weight")?, // tied
        };

        // Hoist all remaining GPU uploads + CPU conv tensors into locals so the
        // `g` closure's borrow of `ctx` ends before `ctx` is moved into Self.
        let enc_pos = g("encoder.embed_positions.weight")?;
        let enc_ln_w = g("encoder.layer_norm.weight")?;
        let enc_ln_b = g("encoder.layer_norm.bias")?;
        let dec_pos = g("decoder.embed_positions.weight")?;
        let dec_ln_w = g("decoder.layer_norm.weight")?;
        let dec_ln_b = g("decoder.layer_norm.bias")?;
        let conv1_w = get("encoder.conv1.weight")?.clone();
        let conv1_b = opt("encoder.conv1.bias").cloned();
        let conv2_w = get("encoder.conv2.weight")?.clone();
        let conv2_b = opt("encoder.conv2.bias").cloned();

        Ok(Self {
            ctx,
            cfg,
            head_dim,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            enc_pos,
            enc_layers,
            enc_ln_w,
            enc_ln_b,
            tok_embed,
            dec_pos,
            dec_layers,
            dec_ln_w,
            dec_ln_b,
            proj_out,
            decoder_len: 0,
            audio_ctx_len: 0,
        })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    pub fn backend_label(&self) -> String {
        format!("wgpu ({})", self.ctx.adapter_label())
    }

    pub fn reset_decoder(&mut self) {
        self.decoder_len = 0;
    }

    /// CPU conv stem: mel `[1, n_mels, 3000]` → `[seq, d]` flat vec + seq len.
    fn conv_stem(&self, mel: &Tensor) -> Result<(Vec<f32>, usize)> {
        let mel = mel.to_f32_tensor().map_err(|e| anyhow!("{e}"))?;
        let x = conv1d(&mel, &self.conv1_w, self.conv1_b.as_ref(), 1, 1, 1, 1)?;
        let x = cpu_gelu_erf(&x).map_err(|e| anyhow!("{e}"))?;
        let x = conv1d(&x, &self.conv2_w, self.conv2_b.as_ref(), 1, 2, 1, 1)?;
        let x = cpu_gelu_erf(&x).map_err(|e| anyhow!("{e}"))?;
        // [1, d, T] → [T, d].
        let d = self.cfg.d_model;
        let t = x.shape().dims()[2];
        let src = x.as_f32_slice();
        let mut out = vec![0.0f32; t * d];
        for di in 0..d {
            for ti in 0..t {
                out[ti * d + di] = src[di * t + ti];
            }
        }
        Ok((out, t))
    }

    /// Run the encoder and cache each decoder layer's cross-attention K/V.
    /// Returns the audio context `[seq, d]` (downloaded to a CPU tensor).
    pub fn encode(&mut self, mel: &Tensor) -> Result<Tensor> {
        let d = self.cfg.d_model;
        let n_head = self.cfg.encoder_attention_heads;
        let head_dim = self.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let ffn = self.cfg.encoder_ffn_dim;

        let (stem, seq) = self.conv_stem(mel)?;
        self.audio_ctx_len = seq;
        let ctx = &self.ctx;

        // Upload conv output and add the (sinusoidal) positional embedding.
        let x0 = ctx.upload_f32(&stem, "enc.x");
        let mut x = ctx.add(&x0, &self.enc_pos);

        for layer in &self.enc_layers {
            // Self-attention (non-causal MHA).
            let normed = ctx.layer_norm(&x, &layer.ln1_w, &layer.ln1_b, seq, d, LN_EPS);
            let q = ctx.add_bias(&ctx.matmul_nt(&normed, &layer.wq, seq, d, d), &layer.bq, d);
            let k = ctx.matmul_nt(&normed, &layer.wk, seq, d, d); // no bias
            let v = ctx.add_bias(&ctx.matmul_nt(&normed, &layer.wv, seq, d, d), &layer.bv, d);
            let qh = ctx.transpose_heads(&q, seq, n_head, head_dim);
            let kh = ctx.transpose_heads(&k, seq, n_head, head_dim);
            let vh = ctx.transpose_heads(&v, seq, n_head, head_dim);
            let attn = ctx.attention(
                &qh, &kh, &vh, 1, n_head, n_head, seq, seq, seq, head_dim, scale, false,
            );
            let merged = ctx.transpose_heads(&attn, n_head, seq, head_dim);
            let o = ctx.add_bias(&ctx.matmul_nt(&merged, &layer.wo, seq, d, d), &layer.bo, d);
            x = ctx.add(&x, &o);

            // MLP.
            let normed = ctx.layer_norm(&x, &layer.ln2_w, &layer.ln2_b, seq, d, LN_EPS);
            let up = ctx.add_bias(
                &ctx.matmul_nt(&normed, &layer.fc1_w, seq, d, ffn),
                &layer.fc1_b,
                ffn,
            );
            let act = ctx.gelu_erf(&up);
            let down = ctx.add_bias(
                &ctx.matmul_nt(&act, &layer.fc2_w, seq, ffn, d),
                &layer.fc2_b,
                d,
            );
            x = ctx.add(&x, &down);
        }

        let x = ctx.layer_norm(&x, &self.enc_ln_w, &self.enc_ln_b, seq, d, LN_EPS);

        // Project & cache cross-attention K/V per decoder layer (head layout).
        for li in 0..self.cfg.decoder_layers {
            let layer = &self.dec_layers[li];
            let k = ctx.matmul_nt(&x, &layer.ck, seq, d, d); // no bias
            let v = ctx.add_bias(&ctx.matmul_nt(&x, &layer.cv, seq, d, d), &layer.cbv, d);
            let kh = ctx.transpose_heads(&k, seq, n_head, head_dim);
            let vh = ctx.transpose_heads(&v, seq, n_head, head_dim);
            self.dec_layers[li].cross_k = Some(kh);
            self.dec_layers[li].cross_v = Some(vh);
        }

        let data = ctx.download_f32(&x)?;
        Tensor::from_f32(&data, vec![1, seq, d]).map_err(|e| anyhow!("{e}"))
    }

    /// Decode `token_ids` (forced prompt, then one token at a time), append to
    /// the self-attention cache, return next-token logits for the last position.
    pub fn decode_step(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            anyhow::bail!("decode_step called with no tokens");
        }
        if self.dec_layers.iter().any(|l| l.cross_k.is_none()) {
            anyhow::bail!("decode_step called before encode/set_audio_context");
        }
        let d = self.cfg.d_model;
        let n_head = self.cfg.decoder_attention_heads;
        let head_dim = self.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let ffn = self.cfg.decoder_ffn_dim;
        let max_target = self.cfg.max_target_positions;
        let vocab = self.cfg.vocab_size;
        let seq = token_ids.len();
        let start = self.decoder_len;
        let total = start + seq;
        let kv_seq = self.audio_ctx_len;
        if total > max_target {
            anyhow::bail!("decoder length {total} exceeds max_target_positions {max_target}");
        }
        let ctx = &self.ctx;

        // Token embedding + positional slice [start..start+seq].
        let ids = ctx.upload_u32(token_ids, "dec.ids");
        let emb = ctx.embed(&ids, &self.tok_embed, seq, d);
        let pos_slice = ctx.alloc_f32(seq * d, "dec.pos");
        ctx.copy_range(&pos_slice, 0, &self.dec_pos, start * d, seq * d);
        let mut x = ctx.add(&emb, &pos_slice);

        for layer in &self.dec_layers {
            // 1) Causal self-attention with growing cache.
            let normed = ctx.layer_norm(&x, &layer.sln_w, &layer.sln_b, seq, d, LN_EPS);
            let q = ctx.add_bias(&ctx.matmul_nt(&normed, &layer.sq, seq, d, d), &layer.sbq, d);
            let k = ctx.matmul_nt(&normed, &layer.sk, seq, d, d); // no bias
            let v = ctx.add_bias(&ctx.matmul_nt(&normed, &layer.sv, seq, d, d), &layer.sbv, d);
            let qh = ctx.transpose_heads(&q, seq, n_head, head_dim);
            let kh = ctx.transpose_heads(&k, seq, n_head, head_dim);
            let vh = ctx.transpose_heads(&v, seq, n_head, head_dim);
            // Append new K/V into each head's cache slot at `start`.
            for h in 0..n_head {
                let dst = (h * max_target + start) * head_dim;
                ctx.copy_range(&layer.self_k, dst, &kh, h * seq * head_dim, seq * head_dim);
                ctx.copy_range(&layer.self_v, dst, &vh, h * seq * head_dim, seq * head_dim);
            }
            let attn = ctx.attention(
                &qh,
                &layer.self_k,
                &layer.self_v,
                1,
                n_head,
                n_head,
                seq,
                total,
                max_target,
                head_dim,
                scale,
                true, // causal
            );
            let merged = ctx.transpose_heads(&attn, n_head, seq, head_dim);
            let o = ctx.add_bias(&ctx.matmul_nt(&merged, &layer.so, seq, d, d), &layer.sbo, d);
            x = ctx.add(&x, &o);

            // 2) Cross-attention to the cached encoder K/V (non-causal).
            let normed = ctx.layer_norm(&x, &layer.cln_w, &layer.cln_b, seq, d, LN_EPS);
            let q = ctx.add_bias(&ctx.matmul_nt(&normed, &layer.cq, seq, d, d), &layer.cbq, d);
            let qh = ctx.transpose_heads(&q, seq, n_head, head_dim);
            let ck = layer.cross_k.as_ref().unwrap();
            let cv = layer.cross_v.as_ref().unwrap();
            let attn = ctx.attention(
                &qh, ck, cv, 1, n_head, n_head, seq, kv_seq, kv_seq, head_dim, scale, false,
            );
            let merged = ctx.transpose_heads(&attn, n_head, seq, head_dim);
            let o = ctx.add_bias(&ctx.matmul_nt(&merged, &layer.co, seq, d, d), &layer.cbo, d);
            x = ctx.add(&x, &o);

            // 3) MLP.
            let normed = ctx.layer_norm(&x, &layer.fln_w, &layer.fln_b, seq, d, LN_EPS);
            let up = ctx.add_bias(
                &ctx.matmul_nt(&normed, &layer.fc1_w, seq, d, ffn),
                &layer.fc1_b,
                ffn,
            );
            let act = ctx.gelu_erf(&up);
            let down = ctx.add_bias(
                &ctx.matmul_nt(&act, &layer.fc2_w, seq, ffn, d),
                &layer.fc2_b,
                d,
            );
            x = ctx.add(&x, &down);
        }

        let x = ctx.layer_norm(&x, &self.dec_ln_w, &self.dec_ln_b, seq, d, LN_EPS);
        self.decoder_len = total;

        // Logits for the LAST position only.
        let last = ctx.alloc_f32(d, "dec.last");
        ctx.copy_range(&last, 0, &x, (seq - 1) * d, d);
        let logits = ctx.matmul_nt(&last, &self.proj_out, 1, d, vocab);
        Ok(ctx.download_f32(&logits)?)
    }
}
