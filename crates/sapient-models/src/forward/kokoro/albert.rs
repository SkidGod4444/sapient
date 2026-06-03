//! The PLBERT text encoder — HuggingFace ALBERT with cross-layer parameter
//! sharing (one `albert_layer` group applied `num_hidden_layers` times) and a
//! factorized embedding (`embedding_size=128` → `hidden_size=768`). Kokoro runs
//! it over the padded phoneme id sequence and feeds `last_hidden_state` to
//! `bert_encoder`. Inference is single-sequence and unmasked (no padding), so
//! attention is full (non-causal).

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use sapient_core::Tensor;

use sapient_core::Shape;

use super::loader::PlbertConfig;
use super::ops::{gelu_new, layer_norm_rows, linear2d, linear_mm, softmax_inplace};

fn get<'a>(w: &'a HashMap<String, Tensor>, k: &str) -> Result<&'a Tensor> {
    w.get(k)
        .ok_or_else(|| anyhow!("kokoro: missing weight {k}"))
}
fn vget(w: &HashMap<String, Tensor>, k: &str) -> Result<Vec<f32>> {
    Ok(get(w, k)?.to_f32_vec())
}

/// Run the ALBERT encoder over `input_ids` (length `L`). Returns the last hidden
/// state laid out `[L, hidden_size]` row-major.
pub fn albert_encode(
    w: &HashMap<String, Tensor>,
    input_ids: &[u32],
    cfg: &PlbertConfig,
) -> Result<Vec<f32>> {
    let l = input_ids.len();
    let emb_size = get(w, "bert.embeddings.word_embeddings.weight")?
        .shape()
        .dims()[1]; // 128
    let h = cfg.hidden_size; // 768
    let n_heads = cfg.num_attention_heads; // 12
    let head_dim = h / n_heads; // 64
    let eps = 1e-12f32; // HF ALBERT LayerNorm eps

    // ── embeddings: word + position + token_type(0), then LayerNorm ──────────
    let word = vget(w, "bert.embeddings.word_embeddings.weight")?;
    let pos = vget(w, "bert.embeddings.position_embeddings.weight")?;
    let tok_type = vget(w, "bert.embeddings.token_type_embeddings.weight")?; // [2, emb]
    let mut emb = vec![0.0f32; l * emb_size];
    for (i, &id) in input_ids.iter().enumerate() {
        let wbase = id as usize * emb_size;
        let pbase = i * emb_size;
        for e in 0..emb_size {
            emb[i * emb_size + e] = word[wbase + e] + pos[pbase + e] + tok_type[e];
        }
    }
    let ln_w = vget(w, "bert.embeddings.LayerNorm.weight")?;
    let ln_b = vget(w, "bert.embeddings.LayerNorm.bias")?;
    layer_norm_rows(&mut emb, l, emb_size, &ln_w, &ln_b, eps);

    // ── factorized embedding → hidden ────────────────────────────────────────
    let map_w = vget(w, "bert.encoder.embedding_hidden_mapping_in.weight")?;
    let map_b = vget(w, "bert.encoder.embedding_hidden_mapping_in.bias")?;
    let mut x = linear2d(&emb, l, emb_size, &map_w, Some(&map_b), h); // [L, 768]

    // Shared layer weights (one group, one layer — applied num_hidden_layers×).
    // Keep the matmul weights as Tensors so `linear_mm` (SIMD+rayon) handles the
    // big projections; only biases/norms need f32 vecs.
    let p = "bert.encoder.albert_layer_groups.0.albert_layers.0";
    let (qw, qb) = (
        get(w, &format!("{p}.attention.query.weight"))?,
        vget(w, &format!("{p}.attention.query.bias"))?,
    );
    let (kw, kb) = (
        get(w, &format!("{p}.attention.key.weight"))?,
        vget(w, &format!("{p}.attention.key.bias"))?,
    );
    let (vw, vb) = (
        get(w, &format!("{p}.attention.value.weight"))?,
        vget(w, &format!("{p}.attention.value.bias"))?,
    );
    let (dw, db) = (
        get(w, &format!("{p}.attention.dense.weight"))?,
        vget(w, &format!("{p}.attention.dense.bias"))?,
    );
    let attn_ln_w = vget(w, &format!("{p}.attention.LayerNorm.weight"))?;
    let attn_ln_b = vget(w, &format!("{p}.attention.LayerNorm.bias"))?;
    let (fw, fb) = (
        get(w, &format!("{p}.ffn.weight"))?,
        vget(w, &format!("{p}.ffn.bias"))?,
    );
    let (ow, ob) = (
        get(w, &format!("{p}.ffn_output.weight"))?,
        vget(w, &format!("{p}.ffn_output.bias"))?,
    );
    let full_ln_w = vget(w, &format!("{p}.full_layer_layer_norm.weight"))?;
    let full_ln_b = vget(w, &format!("{p}.full_layer_layer_norm.bias"))?;
    let inter = fb.len(); // 2048
    let scale = 1.0 / (head_dim as f32).sqrt();

    for _layer in 0..cfg.num_hidden_layers {
        let x_t = Tensor::from_f32(&x, Shape::new([l, h])).map_err(|e| anyhow!("{e}"))?;
        // self-attention projections (SIMD+rayon matmul)
        let q = linear_mm(&x_t, qw, Some(&qb))?;
        let k = linear_mm(&x_t, kw, Some(&kb))?;
        let v = linear_mm(&x_t, vw, Some(&vb))?;
        let mut ctx = vec![0.0f32; l * h];
        let mut scores = vec![0.0f32; l];
        for head in 0..n_heads {
            let off = head * head_dim;
            for i in 0..l {
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q[i * h + off + d] * k[j * h + off + d];
                    }
                    *sc = acc * scale;
                }
                softmax_inplace(&mut scores);
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, &sc) in scores.iter().enumerate() {
                        acc += sc * v[j * h + off + d];
                    }
                    ctx[i * h + off + d] = acc;
                }
            }
        }
        // dense projection + residual + post-LN
        let ctx_t = Tensor::from_f32(&ctx, Shape::new([l, h])).map_err(|e| anyhow!("{e}"))?;
        let mut attn = linear_mm(&ctx_t, dw, Some(&db))?;
        for (a, xv) in attn.iter_mut().zip(x.iter()) {
            *a += *xv;
        }
        layer_norm_rows(&mut attn, l, h, &attn_ln_w, &attn_ln_b, eps);

        // FFN: ffn → gelu_new → ffn_output, then residual + full-layer LN
        let attn_t = Tensor::from_f32(&attn, Shape::new([l, h])).map_err(|e| anyhow!("{e}"))?;
        let mut ff = linear_mm(&attn_t, fw, Some(&fb))?;
        for v in ff.iter_mut() {
            *v = gelu_new(*v);
        }
        let ff_t = Tensor::from_f32(&ff, Shape::new([l, inter])).map_err(|e| anyhow!("{e}"))?;
        let mut out = linear_mm(&ff_t, ow, Some(&ob))?;
        for (o, a) in out.iter_mut().zip(attn.iter()) {
            *o += *a;
        }
        layer_norm_rows(&mut out, l, h, &full_ln_w, &full_ln_b, eps);
        x = out;
    }
    Ok(x)
}
