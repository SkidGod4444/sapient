//! Correctness gate for the Mixtral-class sparse-MoE FFN in `LlamaForward`.
//!
//! There is no second MoE engine to diff against, so this uses two independent
//! oracles:
//!
//! 1. **Identical-experts equivalence** (`moe_with_identical_experts_matches_dense`):
//!    a MoE whose experts are ALL the same matrix `D`, with `norm_topk_prob`, must
//!    produce byte-close logits to the dense model with FFN `D` — because
//!    `Σ wᵢ·D(h) = (Σ wᵢ)·D(h) = D(h)` when the top-k weights renormalise to 1.
//!    This runs the REAL attention path in both models, so it proves the MoE block
//!    integrates without disturbing attention, the KV cache, or the residual — and
//!    that gather/scatter map each token to its own row.
//!
//! 2. **Hand-computed reference with distinct experts** (`moe_matches_reference`):
//!    the router, top-k selection, per-expert weighting and scatter are validated
//!    against a from-scratch reference forward pass written in this file. Attention
//!    is neutralised (`v_proj = 0` ⇒ attention output 0 ⇒ the residual leaves `x`
//!    unchanged), so the reference need not re-implement RoPE/GQA — yet the experts
//!    are all DIFFERENT, so choosing the wrong expert or mis-weighting it is caught.
//!
//! Pure f32 throughout (F32 weights aren't online-quantized), so the only spread is
//! floating-point reduction order.

use std::collections::HashMap;

use sapient_core::{Shape, Tensor};
use sapient_hub::model_info::{ArchType, ModelInfo, MoeConfig, MoeScoring};
use sapient_models::forward::LlamaForward;

// ── tiny dims ────────────────────────────────────────────────────────────────
const HIDDEN: usize = 32;
const N_HEADS: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 8;
const INTER: usize = 48; // dense intermediate
const EXPERT_INTER: usize = 24; // per-expert intermediate (distinct from dense)
const LAYERS: usize = 2;
const VOCAB: usize = 40;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const EPS: f32 = 1e-5;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // ~U(-1,1)
    }
}

fn mat(rows: usize, cols: usize, scale: f32, n: &mut dyn FnMut() -> f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|_| n() * scale).collect();
    Tensor::from_f32_vec(data, Shape::new([rows, cols])).unwrap()
}

fn norm_w(dim: usize, n: &mut dyn FnMut() -> f32) -> Tensor {
    let data: Vec<f32> = (0..dim).map(|_| 1.0 + n() * 0.05).collect();
    Tensor::from_f32_vec(data, Shape::new([dim])).unwrap()
}

fn base_info(moe: Option<MoeConfig>) -> ModelInfo {
    ModelInfo {
        arch: ArchType::Mixtral,
        model_type: "mixtral".into(),
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: LAYERS,
        num_attention_heads: N_HEADS,
        num_key_value_heads: N_KV,
        intermediate_size: INTER,
        max_position_embeddings: 512,
        rms_norm_eps: EPS as f64,
        hidden_act: "silu".into(),
        rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        head_dim: HEAD_DIM,
        moe,
        raw: serde_json::Value::Null,
    }
}

/// Common non-FFN weights (embed, attention, norms, head). `zero_v` zeroes the
/// value projection so attention contributes nothing (used by the hand-oracle test).
fn common_weights(next: &mut dyn FnMut() -> f32, zero_v: bool) -> HashMap<String, Tensor> {
    let mut w = HashMap::new();
    w.insert(
        "model.embed_tokens.weight".into(),
        mat(VOCAB, HIDDEN, 0.2, next),
    );
    w.insert("model.norm.weight".into(), norm_w(HIDDEN, next));
    w.insert("lm_head.weight".into(), mat(VOCAB, HIDDEN, 0.2, next));

    let qd = N_HEADS * HEAD_DIM;
    let kvd = N_KV * HEAD_DIM;
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        w.insert(format!("{p}.input_layernorm.weight"), norm_w(HIDDEN, next));
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            norm_w(HIDDEN, next),
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            mat(qd, HIDDEN, 0.1, next),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            mat(kvd, HIDDEN, 0.1, next),
        );
        let v = if zero_v {
            Tensor::from_f32_vec(vec![0.0; kvd * HIDDEN], Shape::new([kvd, HIDDEN])).unwrap()
        } else {
            mat(kvd, HIDDEN, 0.1, next)
        };
        w.insert(format!("{p}.self_attn.v_proj.weight"), v);
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            mat(HIDDEN, qd, 0.1, next),
        );
    }
    w
}

fn moe_config(first_k_dense: usize, norm_topk_prob: bool) -> MoeConfig {
    MoeConfig {
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        expert_intermediate_size: EXPERT_INTER,
        num_shared_experts: 0,
        first_k_dense,
        norm_topk_prob,
        scoring_func: MoeScoring::Softmax,
    }
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// ── Test 1: identical experts == dense (real attention path) ──────────────────
#[test]
fn moe_with_identical_experts_matches_dense() {
    // Dense model.
    let mut next = lcg(0x5EED01);
    let mut dense_w = common_weights(&mut next, false);
    // One dense FFN triple per layer, reused as the single shared expert below.
    let mut ffns: Vec<(Tensor, Tensor, Tensor)> = Vec::new();
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        let gate = mat(EXPERT_INTER, HIDDEN, 0.1, &mut next);
        let up = mat(EXPERT_INTER, HIDDEN, 0.1, &mut next);
        let down = mat(HIDDEN, EXPERT_INTER, 0.1, &mut next);
        dense_w.insert(format!("{p}.mlp.gate_proj.weight"), gate.clone());
        dense_w.insert(format!("{p}.mlp.up_proj.weight"), up.clone());
        dense_w.insert(format!("{p}.mlp.down_proj.weight"), down.clone());
        ffns.push((gate, up, down));
    }
    // Dense info uses EXPERT_INTER as its intermediate so the FFN matrices match.
    let mut dense_info = base_info(None);
    dense_info.intermediate_size = EXPERT_INTER;
    let mut dense_engine = LlamaForward::from_weights(dense_info, dense_w).unwrap();

    // MoE model: every expert == the dense FFN, all layers MoE, renorm on.
    let mut moe_w = common_weights(&mut lcg(0x5EED01), false); // same seed → same attn/embed
    for (i, (gate, up, down)) in ffns.iter().enumerate() {
        let p = format!("model.layers.{i}.block_sparse_moe");
        // Router weights are arbitrary — identical experts make routing irrelevant.
        moe_w.insert(
            format!("{p}.gate.weight"),
            mat(NUM_EXPERTS, HIDDEN, 0.1, &mut lcg(i as u64 + 7)),
        );
        for e in 0..NUM_EXPERTS {
            moe_w.insert(format!("{p}.experts.{e}.w1.weight"), gate.clone());
            moe_w.insert(format!("{p}.experts.{e}.w3.weight"), up.clone());
            moe_w.insert(format!("{p}.experts.{e}.w2.weight"), down.clone());
        }
    }
    let moe_info = base_info(Some(moe_config(0, true)));
    let mut moe_engine = LlamaForward::from_weights(moe_info, moe_w).unwrap();

    let ids = [3u32, 1, 4, 1, 5, 9];
    let dense_logits = dense_engine.forward_logits(&ids, false).unwrap();
    let moe_logits = moe_engine.forward_logits(&ids, false).unwrap();
    let err = max_abs_err(&dense_logits, &moe_logits);
    assert!(
        err < 1e-4,
        "MoE with identical experts must equal the dense model (max_err={err})"
    );
}

/// Build a Q4_K tensor from random valid block bytes (small positive f16 d/dmin so
/// dequant magnitudes stay ~O(0.1)). Both models decode identical bytes, so no
/// quantizer is needed — the gate is that they agree on what the bytes mean.
fn q4_k_random(rows: usize, cols: usize, next: &mut dyn FnMut() -> f32) -> Tensor {
    let numel = rows * cols;
    assert_eq!(numel % 256, 0);
    let mut blocks = Vec::with_capacity(numel / 256 * 144);
    for _ in 0..numel / 256 {
        let d = half::f16::from_f32(1.0e-4 * (1.0 + next().abs()));
        let dmin = half::f16::from_f32(1.0e-4 * (1.0 + next().abs()));
        blocks.extend_from_slice(&d.to_le_bytes());
        blocks.extend_from_slice(&dmin.to_le_bytes());
        for _ in 0..140 {
            blocks.push((next().abs() * 255.0) as u8); // 12 scale bytes + 128 qs bytes
        }
    }
    sapient_core::Tensor::from_quant_bytes(&blocks, vec![rows, cols], sapient_core::DType::Q4_K)
        .unwrap()
}

// ── Test 1b: identical Q4_K experts == dense, exercising the aarch64 R4 repack ──
// On aarch64+dotprod the load path repacks these Q4_K experts to Q4_K_R4 and the
// MoE matmul reads them through the multi-row SDOT/SMMLA kernels — a path no other
// test covers. MoE-R4 ≡ dense-R4 here, and `cpu_repack.rs` proves dense-R4 ≡
// dense-non-R4, so this transitively validates the R4 expert path. On x86 the
// experts stay plain Q4_K (repack is aarch64-gated) — still a valid Q4_K gate.
#[test]
fn moe_q4k_identical_experts_matches_dense() {
    // Q4_K needs the contracted dim to be a multiple of 256.
    const H: usize = 256;
    const I: usize = 256;
    const NE: usize = 4;
    let heads = 4;
    let kv = 2;
    let hd = 64;
    let vocab = 40;

    let mk_info = |moe: Option<MoeConfig>| ModelInfo {
        arch: ArchType::Mixtral,
        model_type: "mixtral".into(),
        vocab_size: vocab,
        hidden_size: H,
        num_hidden_layers: 1,
        num_attention_heads: heads,
        num_key_value_heads: kv,
        intermediate_size: I,
        max_position_embeddings: 256,
        rms_norm_eps: EPS as f64,
        hidden_act: "silu".into(),
        rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        head_dim: hd,
        moe,
        raw: serde_json::Value::Null,
    };

    // Attention/embed/norm weights (F32) — built identically for both models.
    let common = |next: &mut dyn FnMut() -> f32| {
        let mut w = HashMap::new();
        w.insert("model.embed_tokens.weight".into(), mat(vocab, H, 0.2, next));
        w.insert("model.norm.weight".into(), norm_w(H, next));
        w.insert("lm_head.weight".into(), mat(vocab, H, 0.2, next));
        let qd = heads * hd;
        let kvd = kv * hd;
        w.insert(
            "model.layers.0.input_layernorm.weight".into(),
            norm_w(H, next),
        );
        w.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            norm_w(H, next),
        );
        w.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            mat(qd, H, 0.1, next),
        );
        w.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            mat(kvd, H, 0.1, next),
        );
        w.insert(
            "model.layers.0.self_attn.v_proj.weight".into(),
            mat(kvd, H, 0.1, next),
        );
        w.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            mat(H, qd, 0.1, next),
        );
        w
    };

    // The shared Q4_K FFN triple (w1=gate, w3=up, w2=down), reused as every expert.
    let mut wsrc = lcg(0x04A5C0);
    let w1 = q4_k_random(I, H, &mut wsrc);
    let w3 = q4_k_random(I, H, &mut wsrc);
    let w2 = q4_k_random(H, I, &mut wsrc);

    let mut dense_w = common(&mut lcg(0xD00D));
    dense_w.insert("model.layers.0.mlp.gate_proj.weight".into(), w1.clone());
    dense_w.insert("model.layers.0.mlp.up_proj.weight".into(), w3.clone());
    dense_w.insert("model.layers.0.mlp.down_proj.weight".into(), w2.clone());
    let mut dense = LlamaForward::from_weights(mk_info(None), dense_w).unwrap();

    let mut moe_w = common(&mut lcg(0xD00D)); // same seed → identical attention/embed
    moe_w.insert(
        "model.layers.0.block_sparse_moe.gate.weight".into(),
        mat(NE, H, 0.1, &mut lcg(0x9)),
    );
    for e in 0..NE {
        let p = format!("model.layers.0.block_sparse_moe.experts.{e}");
        moe_w.insert(format!("{p}.w1.weight"), w1.clone());
        moe_w.insert(format!("{p}.w3.weight"), w3.clone());
        moe_w.insert(format!("{p}.w2.weight"), w2.clone());
    }
    let moe_cfg = MoeConfig {
        num_experts: NE,
        top_k: 2,
        expert_intermediate_size: I,
        num_shared_experts: 0,
        first_k_dense: 0,
        norm_topk_prob: true,
        scoring_func: MoeScoring::Softmax,
    };
    let mut moe = LlamaForward::from_weights(mk_info(Some(moe_cfg)), moe_w).unwrap();

    let ids = [3u32, 1, 4, 1, 5];
    let err = max_abs_err(
        &dense.forward_logits(&ids, false).unwrap(),
        &moe.forward_logits(&ids, false).unwrap(),
    );
    assert!(
        err < 1e-3,
        "Q4_K identical-experts MoE must equal dense (max_err={err})"
    );
}

// ── Test 2: distinct experts vs a from-scratch reference (attention neutralised) ─
#[test]
fn moe_matches_reference() {
    let mut next = lcg(0xABCDEF);
    // first_k_dense = 1 → layer 0 dense, layer 1 MoE (exercises both branches + the mix).
    let first_k_dense = 1;
    let mut w = common_weights(&mut next, true); // zero_v → attention output is 0

    // Layer 0: dense FFN. Layer 1: MoE (router + NUM_EXPERTS distinct experts).
    let dense_gate = mat(INTER, HIDDEN, 0.15, &mut next);
    let dense_up = mat(INTER, HIDDEN, 0.15, &mut next);
    let dense_down = mat(HIDDEN, INTER, 0.15, &mut next);
    w.insert(
        "model.layers.0.mlp.gate_proj.weight".into(),
        dense_gate.clone(),
    );
    w.insert("model.layers.0.mlp.up_proj.weight".into(), dense_up.clone());
    w.insert(
        "model.layers.0.mlp.down_proj.weight".into(),
        dense_down.clone(),
    );

    let router = mat(NUM_EXPERTS, HIDDEN, 0.3, &mut next);
    let mut experts: Vec<(Tensor, Tensor, Tensor)> = Vec::new();
    for _ in 0..NUM_EXPERTS {
        experts.push((
            mat(EXPERT_INTER, HIDDEN, 0.2, &mut next),
            mat(EXPERT_INTER, HIDDEN, 0.2, &mut next),
            mat(HIDDEN, EXPERT_INTER, 0.2, &mut next),
        ));
    }
    w.insert(
        "model.layers.1.block_sparse_moe.gate.weight".into(),
        router.clone(),
    );
    for (e, (w1, w3, w2)) in experts.iter().enumerate() {
        let p = format!("model.layers.1.block_sparse_moe.experts.{e}");
        w.insert(format!("{p}.w1.weight"), w1.clone());
        w.insert(format!("{p}.w3.weight"), w3.clone());
        w.insert(format!("{p}.w2.weight"), w2.clone());
    }

    // Read the weights the reference needs BEFORE moving `w` into the engine.
    let embed = tensor_rows(w.get("model.embed_tokens.weight").unwrap(), HIDDEN);
    let final_norm = w.get("model.norm.weight").unwrap().to_f32_vec();
    let lm_head = tensor_rows(w.get("lm_head.weight").unwrap(), HIDDEN);
    let post_norm: Vec<Vec<f32>> = (0..LAYERS)
        .map(|i| {
            w.get(&format!("model.layers.{i}.post_attention_layernorm.weight"))
                .unwrap()
                .to_f32_vec()
        })
        .collect();

    let info = base_info(Some(moe_config(first_k_dense, true)));
    let mut engine = LlamaForward::from_weights(info, w).unwrap();

    let ids = [2u32, 7, 1, 5];
    let engine_logits = engine.forward_logits(&ids, false).unwrap();

    // ── reference forward (attention == 0, so x is unchanged by attention) ──
    let mut x: Vec<Vec<f32>> = ids.iter().map(|&id| embed[id as usize].clone()).collect();
    for (layer, pnorm) in post_norm.iter().enumerate() {
        for xt in x.iter_mut() {
            let h = rms_norm(xt, pnorm);
            let ffn = if layer < first_k_dense {
                dense_swiglu(&h, &dense_gate, &dense_up, &dense_down)
            } else {
                moe_block(&h, &router, &experts, TOP_K)
            };
            for (o, f) in xt.iter_mut().zip(ffn) {
                *o += f;
            }
        }
    }
    // forward_logits returns only the last token's logits.
    let last = rms_norm(x.last().unwrap(), &final_norm);
    let ref_logits: Vec<f32> = lm_head.iter().map(|row| dot(&last, row)).collect();

    let err = max_abs_err(&engine_logits, &ref_logits);
    assert!(
        err < 1e-4,
        "MoE engine must match the hand-computed reference (max_err={err})"
    );
}

// ── reference helpers (plain, straightforward — the independent oracle) ────────

/// Split a `[rows, cols]` tensor into `rows` f32 vectors of length `cols`.
fn tensor_rows(t: &Tensor, cols: usize) -> Vec<Vec<f32>> {
    t.to_f32_vec()
        .chunks_exact(cols)
        .map(|c| c.to_vec())
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y = x · Wᵀ` where W is `[out, in]` row-major (SAPIENT/HF linear layout).
fn linear(x: &[f32], w: &Tensor, out: usize, in_dim: usize) -> Vec<f32> {
    let wd = w.to_f32_vec();
    (0..out)
        .map(|o| dot(x, &wd[o * in_dim..(o + 1) * in_dim]))
        .collect()
}

fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + EPS).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * r * w).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn swiglu(h: &[f32], w1: &Tensor, w3: &Tensor, w2: &Tensor, inter: usize) -> Vec<f32> {
    let g = linear(h, w1, inter, HIDDEN);
    let u = linear(h, w3, inter, HIDDEN);
    let m: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
    linear(&m, w2, HIDDEN, inter)
}

fn dense_swiglu(h: &[f32], w1: &Tensor, w3: &Tensor, w2: &Tensor) -> Vec<f32> {
    swiglu(h, w1, w3, w2, INTER)
}

fn moe_block(
    h: &[f32],
    router: &Tensor,
    experts: &[(Tensor, Tensor, Tensor)],
    top_k: usize,
) -> Vec<f32> {
    let logits = linear(h, router, NUM_EXPERTS, HIDDEN);
    // softmax over all experts
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let scores: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    // top-k by score (ties → lower index), then renormalise.
    let mut idx: Vec<usize> = (0..NUM_EXPERTS).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    idx.truncate(top_k);
    let wsum: f32 = idx.iter().map(|&i| scores[i]).sum();
    let mut out = vec![0f32; HIDDEN];
    for &e in &idx {
        let w = scores[e] / wsum;
        let (w1, w3, w2) = &experts[e];
        let d = swiglu(h, w1, w3, w2, EXPERT_INTER);
        for (o, v) in out.iter_mut().zip(d) {
            *o += w * v;
        }
    }
    out
}
