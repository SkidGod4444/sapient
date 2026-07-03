//! Cross-platform GPU forward engine via wgpu/WGSL (Vulkan on Linux, DX12 on
//! Windows, Metal on macOS — Intel/AMD/Nvidia/Apple). Weights are uploaded to GPU
//! storage buffers once at load; every decode step runs entirely on-device (embed →
//! per-layer RMSNorm/QKV/RoPE/flash-attention/SwiGLU → final norm → lm_head) and only
//! the final logits are read back. The KV cache lives in GPU buffers (f16-packed by
//! default — Phase 7.3) and grows by a per-token `kv_append` dispatch — no CPU↔GPU
//! round-trips in the hot loop.
//!
//! Scope: Llama-family architectures (Llama/Qwen/Mistral/SmolLM) — RMSNorm, full RoPE,
//! GQA, SwiGLU MLP, optional q/k/v projection biases (Qwen2). **Q8_0, Q4_K, and Q6_K
//! weights stay quantized on the GPU** (Phase 7): raw ggml blocks upload without f32
//! expansion (Q8_0 as packed int8 + f32 scales, Q4_K super-blocks verbatim, Q6_K
//! blocks padded 210→212 bytes) and are dequantized inside the matmul/embed shaders —
//! 1.125 / 0.5625 / 0.83 bytes per weight of VRAM instead of 4 (F16/BF16 linears are
//! online-quantized to Q8_0 first, mirroring the CPU engine, so both engines see
//! identical weight values). A Q4_K_M GGUF therefore loads fully quantized. Other
//! dtypes (F32, Q4_0/Q5_K) dequantize to f32 on upload. Tokens are processed
//! one at a time (`seq_q = 1`), so prefill is a sequential append — correct and simple;
//! batched prefill is a later optimization (Phase 7.5).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use sapient_backends_wgpu::{GpuBuffer, GpuQ4KBuffer, GpuQ6KBuffer, GpuQ8Buffer, WgpuContext};
use sapient_core::{DType, Tensor};
use sapient_hub::model_info::{ArchType, ModelInfo};

use crate::weights::{detect_weight_prefix, resolve_bias, resolve_lm_head, resolve_weight};

use super::common::{kv_cache_ctx, quantize_tensor_to_q8_0, should_quantize_online};

/// An f32 KV cache is 4× a Q8_0 cache, so the f32 fallback stays capped more
/// conservatively than the CPU engine's 8192 to avoid OOMing modest GPUs at load.
/// With the f16 cache (the default for even head_dim — Phase 7.3) the cap is
/// lifted to the standard `kv_cache_ctx` (8192 / `SAPIENT_CTX`): double the
/// context for the same bytes as f32@4096.
const WGPU_MAX_CTX_F32: usize = 4096;

/// A weight matrix resident on the GPU — dense f32, or kept quantized and
/// dequantized in-shader: Q8_0 (packed int8 + per-block scales), Q4_K (raw
/// ggml super-blocks verbatim), or Q6_K (super-blocks padded to word size).
enum GpuWeight {
    F32(GpuBuffer),
    Q8(GpuQ8Buffer),
    Q4K(GpuQ4KBuffer),
    Q6K(GpuQ6KBuffer),
}

impl GpuWeight {
    /// GPU bytes this weight occupies.
    fn byte_size(&self) -> usize {
        match self {
            GpuWeight::F32(b) => b.len * 4,
            GpuWeight::Q8(q) => q.byte_size(),
            GpuWeight::Q4K(q) => q.byte_size(),
            GpuWeight::Q6K(q) => q.byte_size(),
        }
    }

    fn is_quantized(&self) -> bool {
        !matches!(self, GpuWeight::F32(_))
    }
}

struct LayerWeights {
    input_ln: GpuBuffer,
    wq: GpuWeight,
    bq: Option<GpuBuffer>,
    wk: GpuWeight,
    bk: Option<GpuBuffer>,
    wv: GpuWeight,
    bv: Option<GpuBuffer>,
    wo: GpuWeight,
    bo: Option<GpuBuffer>,
    post_ln: GpuBuffer,
    wgate: GpuWeight,
    wup: GpuWeight,
    wdown: GpuWeight,
    /// KV cache, layout `[n_kv_heads, max_seq, head_dim]`.
    kcache: GpuBuffer,
    vcache: GpuBuffer,
}

pub struct WgpuForwardEngine {
    ctx: WgpuContext,
    info: ModelInfo,
    embed: GpuWeight,
    final_norm: GpuBuffer,
    /// `None` when the output projection is tied to the embedding (the GGUF omits
    /// `output.weight`) — the logits matmul then reuses the embed buffer, so tied
    /// models don't pay for the same matrix twice in VRAM.
    lm_head: Option<GpuWeight>,
    layers: Vec<LayerWeights>,
    max_seq: usize,
    cur_len: usize,
    rotary_dim: usize,
    /// KV cache stored as f16 packed two-per-u32 word (half the bytes, double the
    /// context cap; core WGSL — no device feature). Auto-on whenever head_dim is
    /// even; attention still accumulates in f32.
    kv_f16: bool,
}

fn up(ctx: &WgpuContext, t: &Tensor, label: &str) -> GpuBuffer {
    ctx.upload_f32(&t.to_f32_vec(), label)
}

/// Upload a weight matrix, keeping it quantized on the GPU whenever possible:
/// Q8_0, Q4_K, and Q6_K tensors upload their raw ggml blocks directly (no f32
/// expansion — Q4_K verbatim, Q6_K padded to word size, Q8_0 with a byte
/// repack); F16/BF16 linears are online-quantized to Q8_0 first — the same
/// `should_quantize_online` rule (and therefore the same weight values) as the
/// CPU engine. Everything else (F32, Q4_0/Q5_K, rows not a multiple of the
/// block size) dequantizes to f32 on upload.
fn upload_weight(ctx: &WgpuContext, name: &str, t: &Tensor, label: &str) -> Result<GpuWeight> {
    let k = t.shape().dims().last().copied().unwrap_or(0);
    if t.dtype() == DType::Q8_0 && k % 32 == 0 {
        return Ok(GpuWeight::Q8(ctx.upload_q8_0(
            t.as_quant_blocks(),
            t.numel(),
            label,
        )?));
    }
    if t.dtype() == DType::Q4_K && k % 256 == 0 {
        return Ok(GpuWeight::Q4K(ctx.upload_q4_k(
            t.as_quant_blocks(),
            t.numel(),
            label,
        )?));
    }
    if t.dtype() == DType::Q6_K && k % 256 == 0 {
        return Ok(GpuWeight::Q6K(ctx.upload_q6_k(
            t.as_quant_blocks(),
            t.numel(),
            label,
        )?));
    }
    if should_quantize_online(name, t) {
        let q = quantize_tensor_to_q8_0(t.clone());
        if q.dtype() == DType::Q8_0 {
            return Ok(GpuWeight::Q8(ctx.upload_q8_0(
                q.as_quant_blocks(),
                q.numel(),
                label,
            )?));
        }
    }
    Ok(GpuWeight::F32(up(ctx, t, label)))
}

/// Linear projection dispatching on the resident weight form.
fn mm(ctx: &WgpuContext, x: &GpuBuffer, w: &GpuWeight, m: usize, k: usize, n: usize) -> GpuBuffer {
    match w {
        GpuWeight::F32(b) => ctx.matmul_nt(x, b, m, k, n),
        GpuWeight::Q8(q) => ctx.matmul_nt_q8_0(x, q, m, k, n),
        GpuWeight::Q4K(q) => ctx.matmul_nt_q4_k(x, q, m, k, n),
        GpuWeight::Q6K(q) => ctx.matmul_nt_q6_k(x, q, m, k, n),
    }
}

impl WgpuForwardEngine {
    /// Build from an HF-named weight map (the same map the CPU `LlamaForward` consumes).
    /// Only Llama-family architectures are supported here. The KV cache is f16
    /// (u32-packed halves) whenever head_dim is even, f32 otherwise.
    pub fn from_weights(info: ModelInfo, weights: HashMap<String, Tensor>) -> Result<Self> {
        Self::from_weights_with_kv(info, weights, None)
    }

    /// [`Self::from_weights`] with an explicit KV-cache dtype: `None` = auto (f16
    /// whenever head_dim is even — the packed-u32 representation needs no device
    /// feature), `Some(true)` = f16 (errors on an odd head_dim), `Some(false)` =
    /// force f32 (the coherence tests use this for bit-tight comparison against
    /// the CPU's f32 cache).
    pub fn from_weights_with_kv(
        info: ModelInfo,
        weights: HashMap<String, Tensor>,
        kv_f16: Option<bool>,
    ) -> Result<Self> {
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
        // The f16 cache packs two halves per u32 word (core WGSL — works on every
        // adapter); it only needs an even head_dim so words never straddle heads.
        let kv_f16 = match kv_f16 {
            Some(true) if info.head_dim % 2 != 0 => {
                bail!(
                    "f16 KV cache requires an even head_dim (got {})",
                    info.head_dim
                )
            }
            Some(v) => v,
            None => info.head_dim % 2 == 0,
        };
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

        // f16 halves the per-position bytes, so it keeps the full default context;
        // the f32 fallback stays capped to bound the footprint on modest GPUs.
        let max_seq = if kv_f16 {
            kv_cache_ctx(info.max_position_embeddings)
        } else {
            kv_cache_ctx(info.max_position_embeddings).min(WGPU_MAX_CTX_F32)
        };

        let embed_t = weights
            .get(&embed_key)
            .with_context(|| format!("missing embedding weight '{embed_key}'"))?;
        // A Q8_0 embed table stays quantized on-GPU (dequantized per-row in the
        // gather shader); F16/BF16 tables upload f32, matching the CPU engine.
        let embed = upload_weight(&ctx, "embed_tokens", embed_t, "embed")?;

        let final_norm_t = resolve_weight(&weights, &prefix, "norm")?;
        let final_norm = up(&ctx, final_norm_t, "final_norm");

        let lm_head_t = resolve_lm_head(&weights, &prefix, false, &embed_key)?;
        let lm_head = if std::ptr::eq(lm_head_t, embed_t) {
            None // tied output projection — logits reuse the embed buffer
        } else {
            Some(upload_weight(&ctx, "lm_head", lm_head_t, "lm_head")?)
        };

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
            let gw = |suffix: &str| -> Result<GpuWeight> {
                upload_weight(
                    &ctx,
                    suffix,
                    resolve_weight(&weights, &prefix, &format!("{pfx}.{suffix}"))?,
                    suffix,
                )
            };
            let gb = |suffix: &str| -> Option<GpuBuffer> {
                resolve_bias(&weights, &prefix, &format!("{pfx}.{suffix}"))
                    .map(|t| up(&ctx, t, "bias"))
            };
            layers.push(LayerWeights {
                input_ln: g("input_layernorm")?,
                wq: gw("self_attn.q_proj")?,
                bq: gb("self_attn.q_proj"),
                wk: gw("self_attn.k_proj")?,
                bk: gb("self_attn.k_proj"),
                wv: gw("self_attn.v_proj")?,
                bv: gb("self_attn.v_proj"),
                wo: gw("self_attn.o_proj")?,
                bo: gb("self_attn.o_proj"),
                post_ln: g("post_attention_layernorm")?,
                wgate: gw("mlp.gate_proj")?,
                wup: gw("mlp.up_proj")?,
                wdown: gw("mlp.down_proj")?,
                kcache: if kv_f16 {
                    ctx.alloc_f16(n_kv * max_seq * head_dim, "kcache")
                } else {
                    ctx.alloc_f32(n_kv * max_seq * head_dim, "kcache")
                },
                vcache: if kv_f16 {
                    ctx.alloc_f16(n_kv * max_seq * head_dim, "vcache")
                } else {
                    ctx.alloc_f32(n_kv * max_seq * head_dim, "vcache")
                },
            });
        }

        let (mut weight_bytes, mut quant_matrices, mut total_matrices) = (0usize, 0usize, 0usize);
        let mut tally = |w: &GpuWeight| {
            weight_bytes += w.byte_size();
            total_matrices += 1;
            if w.is_quantized() {
                quant_matrices += 1;
            }
        };
        tally(&embed);
        if let Some(lm) = &lm_head {
            tally(lm);
        }
        for l in &layers {
            for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.wgate, &l.wup, &l.wdown] {
                tally(w);
            }
        }
        tracing::info!(
            "WgpuForwardEngine ready: {} layers, {hidden} hidden, {n_heads}/{n_kv} heads, \
             head_dim {head_dim}, inter {inter}, ctx {max_seq} (KV {}), weights {} MiB \
             resident ({quant_matrices}/{total_matrices} matrices quantized{}) ({})",
            info.num_hidden_layers,
            if kv_f16 { "f16" } else { "f32" },
            weight_bytes >> 20,
            if lm_head.is_none() {
                ", lm_head tied to embed"
            } else {
                ""
            },
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
            kv_f16,
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
        let mut x = match &self.embed {
            GpuWeight::F32(t) => ctx.embed(&ids, t, 1, hidden),
            GpuWeight::Q8(t) => ctx.embed_q8_0(&ids, t, 1, hidden),
            GpuWeight::Q4K(t) => ctx.embed_q4_k(&ids, t, 1, hidden),
            GpuWeight::Q6K(t) => ctx.embed_q6_k(&ids, t, 1, hidden),
        };

        for layer in &self.layers {
            // ── Attention ───────────────────────────────────────────────────────
            let h = ctx.rms_norm(&x, &layer.input_ln, 1, hidden, eps);
            let mut q = mm(ctx, &h, &layer.wq, 1, hidden, n_heads * head_dim);
            if let Some(b) = &layer.bq {
                q = ctx.add(&q, b);
            }
            let mut k = mm(ctx, &h, &layer.wk, 1, hidden, n_kv * head_dim);
            if let Some(b) = &layer.bk {
                k = ctx.add(&k, b);
            }
            let mut v = mm(ctx, &h, &layer.wv, 1, hidden, n_kv * head_dim);
            if let Some(b) = &layer.bv {
                v = ctx.add(&v, b);
            }
            ctx.rope(&q, &positions, n_heads, 1, head_dim, self.rotary_dim, theta);
            ctx.rope(&k, &positions, n_kv, 1, head_dim, self.rotary_dim, theta);

            // Append this token's K/V into the cache (one dispatch per tensor,
            // converting to the cache dtype in-shader).
            ctx.kv_append(
                &k,
                &layer.kcache,
                n_kv,
                head_dim,
                self.max_seq,
                pos,
                self.kv_f16,
            );
            ctx.kv_append(
                &v,
                &layer.vcache,
                n_kv,
                head_dim,
                self.max_seq,
                pos,
                self.kv_f16,
            );

            let attn = if self.kv_f16 {
                ctx.attention_f16kv(
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
                    true, // causal (decoder LLM)
                )
            } else {
                ctx.attention(
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
                    true,
                )
            };
            let mut o = mm(ctx, &attn, &layer.wo, 1, n_heads * head_dim, hidden);
            if let Some(b) = &layer.bo {
                o = ctx.add(&o, b);
            }
            x = ctx.add(&x, &o);

            // ── MLP (SwiGLU) ────────────────────────────────────────────────────
            let h2 = ctx.rms_norm(&x, &layer.post_ln, 1, hidden, eps);
            let gate = mm(ctx, &h2, &layer.wgate, 1, hidden, inter);
            let upp = mm(ctx, &h2, &layer.wup, 1, hidden, inter);
            let act = ctx.swiglu(&gate, &upp);
            let down = mm(ctx, &act, &layer.wdown, 1, inter, hidden);
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
        let lm = self.lm_head.as_ref().unwrap_or(&self.embed);
        let logits = mm(&self.ctx, &h, lm, 1, hidden, vocab);
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
            let lm = self.lm_head.as_ref().unwrap_or(&self.embed);
            let logits = mm(&self.ctx, &h, lm, 1, hidden, vocab);
            out.push(self.ctx.download_f32(&logits)?);
        }
        Ok(out)
    }
}
