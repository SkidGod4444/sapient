//! Cross-platform GPU forward engine via wgpu/WGSL (Vulkan on Linux, DX12 on
//! Windows, Metal on macOS — Intel/AMD/Nvidia/Apple). Weights are uploaded to GPU
//! storage buffers once at load; every decode step runs entirely on-device (embed →
//! per-layer RMSNorm/QKV/RoPE/flash-attention/SwiGLU → final norm → lm_head) and only
//! the final logits are read back. The KV cache lives in GPU buffers and grows by a
//! per-token `copy_range` append — no CPU↔GPU round-trips in the hot loop.
//!
//! Scope (first cut): Llama-family architectures (Llama/Qwen/Mistral/SmolLM) — RMSNorm,
//! full RoPE, GQA, SwiGLU MLP, optional q/k/v projection biases (Qwen2). Weights are
//! dequantized to f32 on upload (P5 will add in-shader Q4_K/Q8_0 unpacking and an f16
//! / quantized KV cache). Tokens are processed one at a time (`seq_q = 1`), so prefill
//! is a sequential append — correct and simple; batched prefill is a later optimization.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use sapient_backends_wgpu::{GpuBuffer, WgpuContext};
use sapient_core::Tensor;
use sapient_hub::model_info::{ArchType, ModelInfo};

use crate::weights::{detect_weight_prefix, resolve_bias, resolve_lm_head, resolve_weight};

use super::common::kv_cache_ctx;

/// f32 KV cache is 4× a Q8_0 cache, so cap the wgpu first cut more conservatively than
/// the CPU engine's 8192 to avoid OOMing modest GPUs at load (P5 adds a quantized cache).
const WGPU_MAX_CTX: usize = 4096;

struct LayerWeights {
    input_ln: GpuBuffer,
    wq: GpuBuffer,
    bq: Option<GpuBuffer>,
    wk: GpuBuffer,
    bk: Option<GpuBuffer>,
    wv: GpuBuffer,
    bv: Option<GpuBuffer>,
    wo: GpuBuffer,
    bo: Option<GpuBuffer>,
    post_ln: GpuBuffer,
    wgate: GpuBuffer,
    wup: GpuBuffer,
    wdown: GpuBuffer,
    /// KV cache, layout `[n_kv_heads, max_seq, head_dim]`.
    kcache: GpuBuffer,
    vcache: GpuBuffer,
}

pub struct WgpuForwardEngine {
    ctx: WgpuContext,
    info: ModelInfo,
    embed: GpuBuffer,
    final_norm: GpuBuffer,
    lm_head: GpuBuffer,
    layers: Vec<LayerWeights>,
    max_seq: usize,
    cur_len: usize,
    rotary_dim: usize,
}

fn up(ctx: &WgpuContext, t: &Tensor, label: &str) -> GpuBuffer {
    ctx.upload_f32(&t.to_f32_vec(), label)
}

impl WgpuForwardEngine {
    /// Build from an HF-named weight map (the same map the CPU `LlamaForward` consumes).
    /// Only Llama-family architectures are supported here.
    pub fn from_weights(info: ModelInfo, weights: HashMap<String, Tensor>) -> Result<Self> {
        if !matches!(
            info.arch,
            ArchType::Llama | ArchType::Qwen | ArchType::Mixtral
        ) {
            bail!(
                "WgpuForwardEngine supports Llama-family models only (got {:?})",
                info.arch
            );
        }
        let ctx = WgpuContext::new().context("no wgpu GPU adapter available")?;
        let prefix = detect_weight_prefix(&weights);
        let embed_key = format!("{prefix}embed_tokens.weight");

        let hidden = info.hidden_size;
        let n_heads = info.num_attention_heads;
        let n_kv = info.num_key_value_heads;
        let head_dim = info.head_dim;
        let inter = info.intermediate_size;

        let mut rotary_dim = ((info.partial_rotary_factor * head_dim as f64) as usize).max(2);
        if rotary_dim % 2 != 0 {
            rotary_dim -= 1;
        }

        let max_seq = kv_cache_ctx(info.max_position_embeddings).min(WGPU_MAX_CTX);

        let embed_t = weights
            .get(&embed_key)
            .with_context(|| format!("missing embedding weight '{embed_key}'"))?;
        let embed = up(&ctx, embed_t, "embed");

        let final_norm_t = resolve_weight(&weights, &prefix, "norm")?;
        let final_norm = up(&ctx, final_norm_t, "final_norm");

        let lm_head_t = resolve_lm_head(&weights, &prefix, false, &embed_key)?;
        let lm_head = up(&ctx, lm_head_t, "lm_head");

        let mut layers = Vec::with_capacity(info.num_hidden_layers);
        for i in 0..info.num_hidden_layers {
            let pfx = format!("layers.{i}");
            let g = |suffix: &str| -> Result<GpuBuffer> {
                Ok(up(
                    &ctx,
                    resolve_weight(&weights, &prefix, &format!("{pfx}.{suffix}"))?,
                    suffix,
                ))
            };
            let gb = |suffix: &str| -> Option<GpuBuffer> {
                resolve_bias(&weights, &prefix, &format!("{pfx}.{suffix}"))
                    .map(|t| up(&ctx, t, "bias"))
            };
            layers.push(LayerWeights {
                input_ln: g("input_layernorm")?,
                wq: g("self_attn.q_proj")?,
                bq: gb("self_attn.q_proj"),
                wk: g("self_attn.k_proj")?,
                bk: gb("self_attn.k_proj"),
                wv: g("self_attn.v_proj")?,
                bv: gb("self_attn.v_proj"),
                wo: g("self_attn.o_proj")?,
                bo: gb("self_attn.o_proj"),
                post_ln: g("post_attention_layernorm")?,
                wgate: g("mlp.gate_proj")?,
                wup: g("mlp.up_proj")?,
                wdown: g("mlp.down_proj")?,
                kcache: ctx.alloc_f32(n_kv * max_seq * head_dim, "kcache"),
                vcache: ctx.alloc_f32(n_kv * max_seq * head_dim, "vcache"),
            });
        }

        tracing::info!(
            "WgpuForwardEngine ready: {} layers, {hidden} hidden, {n_heads}/{n_kv} heads, \
             head_dim {head_dim}, inter {inter}, ctx {max_seq} ({})",
            info.num_hidden_layers,
            ctx.adapter_label()
        );

        Ok(Self {
            ctx,
            info,
            embed,
            final_norm,
            lm_head,
            layers,
            max_seq,
            cur_len: 0,
            rotary_dim,
        })
    }

    pub fn reset_cache(&mut self) {
        self.cur_len = 0;
    }

    pub fn truncate_cache(&mut self, n: usize) -> usize {
        self.cur_len = n.min(self.cur_len);
        self.cur_len
    }

    pub fn backend_label(&self) -> String {
        format!("wgpu ({})", self.ctx.adapter_label())
    }

    pub fn is_hybrid(&self) -> bool {
        false
    }

    /// Run one token through all layers at the current cache position, append its K/V,
    /// and return the post-final-norm hidden state `[hidden]`. Advances `cur_len`.
    fn forward_token(&mut self, tok: u32) -> Result<GpuBuffer> {
        let hidden = self.info.hidden_size;
        let n_heads = self.info.num_attention_heads;
        let n_kv = self.info.num_key_value_heads;
        let head_dim = self.info.head_dim;
        let inter = self.info.intermediate_size;
        let eps = self.info.rms_norm_eps as f32;
        let theta = self.info.rope_theta as f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos = self.cur_len;
        if pos >= self.max_seq {
            bail!("wgpu KV cache full ({} positions)", self.max_seq);
        }
        let positions = [pos as u32];
        let ctx = &self.ctx;

        let ids = ctx.upload_u32(&[tok], "tok");
        let mut x = ctx.embed(&ids, &self.embed, 1, hidden);

        for layer in &self.layers {
            // ── Attention ───────────────────────────────────────────────────────
            let h = ctx.rms_norm(&x, &layer.input_ln, 1, hidden, eps);
            let mut q = ctx.matmul_nt(&h, &layer.wq, 1, hidden, n_heads * head_dim);
            if let Some(b) = &layer.bq {
                q = ctx.add(&q, b);
            }
            let mut k = ctx.matmul_nt(&h, &layer.wk, 1, hidden, n_kv * head_dim);
            if let Some(b) = &layer.bk {
                k = ctx.add(&k, b);
            }
            let mut v = ctx.matmul_nt(&h, &layer.wv, 1, hidden, n_kv * head_dim);
            if let Some(b) = &layer.bv {
                v = ctx.add(&v, b);
            }
            ctx.rope(&q, &positions, n_heads, 1, head_dim, self.rotary_dim, theta);
            ctx.rope(&k, &positions, n_kv, 1, head_dim, self.rotary_dim, theta);

            // Append this token's K/V into each kv-head's slot in the cache.
            for hh in 0..n_kv {
                let dst = (hh * self.max_seq + pos) * head_dim;
                ctx.copy_range(&layer.kcache, dst, &k, hh * head_dim, head_dim);
                ctx.copy_range(&layer.vcache, dst, &v, hh * head_dim, head_dim);
            }

            let attn = ctx.attention(
                &q,
                &layer.kcache,
                &layer.vcache,
                1,
                n_heads,
                n_kv,
                1,
                pos + 1,
                self.max_seq,
                head_dim,
                scale,
            );
            let mut o = ctx.matmul_nt(&attn, &layer.wo, 1, n_heads * head_dim, hidden);
            if let Some(b) = &layer.bo {
                o = ctx.add(&o, b);
            }
            x = ctx.add(&x, &o);

            // ── MLP (SwiGLU) ────────────────────────────────────────────────────
            let h2 = ctx.rms_norm(&x, &layer.post_ln, 1, hidden, eps);
            let gate = ctx.matmul_nt(&h2, &layer.wgate, 1, hidden, inter);
            let upp = ctx.matmul_nt(&h2, &layer.wup, 1, hidden, inter);
            let act = ctx.swiglu(&gate, &upp);
            let down = ctx.matmul_nt(&act, &layer.wdown, 1, inter, hidden);
            x = ctx.add(&x, &down);
        }

        self.cur_len += 1;
        Ok(ctx.rms_norm(&x, &self.final_norm, 1, hidden, eps))
    }

    /// Logits for the final token. Appends `input_ids` to the KV cache when
    /// `use_cache`; otherwise resets first.
    pub fn forward_logits(&mut self, input_ids: &[u32], use_cache: bool) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            bail!("forward_logits: empty input");
        }
        if !use_cache {
            self.cur_len = 0;
        }
        let hidden = self.info.hidden_size;
        let vocab = self.info.vocab_size;
        let mut last = None;
        for &tok in input_ids {
            last = Some(self.forward_token(tok)?);
        }
        let h = last.expect("non-empty");
        let logits = self.ctx.matmul_nt(&h, &self.lm_head, 1, hidden, vocab);
        Ok(self.ctx.download_f32(&logits)?)
    }

    /// Logits for every position (resets the cache first).
    pub fn forward_all_logits(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.cur_len = 0;
        self.forward_all_logits_cached(input_ids)
    }

    /// Logits for every position, appending to the current cache (no reset).
    pub fn forward_all_logits_cached(&mut self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let hidden = self.info.hidden_size;
        let vocab = self.info.vocab_size;
        let mut out = Vec::with_capacity(input_ids.len());
        for &tok in input_ids {
            let h = self.forward_token(tok)?;
            let logits = self.ctx.matmul_nt(&h, &self.lm_head, 1, hidden, vocab);
            out.push(self.ctx.download_f32(&logits)?);
        }
        Ok(out)
    }
}
