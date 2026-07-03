//! End-to-end coherence: the wgpu GPU forward engine must produce the same logits as
//! the proven CPU `LlamaForward` for an identical (synthetic) Llama-family model. This
//! is the gating test for the cross-platform GPU path — if a kernel (RoPE axis, GQA
//! mapping, KV-cache layout, reduction) were subtly wrong, the per-token logits would
//! diverge and the model would emit token-salad (exactly the class of bug this guards).
//!
//! Pure f32 throughout: F32 weights aren't online-quantized by the CPU path, and a
//! head_dim not divisible by 32 keeps the CPU KV cache in F32 too — so the only
//! differences are floating-point reduction order (tiny).
#![cfg(feature = "wgpu")]

use std::collections::HashMap;

use sapient_core::{DType, Shape, Tensor};
use sapient_hub::model_info::{ArchType, ModelInfo};
use sapient_models::forward::{LlamaForward, WgpuForwardEngine};

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // ~U(-1,1)
    }
}

fn tiny_llama() -> (ModelInfo, HashMap<String, Tensor>) {
    let hidden = 64usize;
    let n_heads = 4usize;
    let n_kv = 2usize;
    let head_dim = 16usize; // not a multiple of 32 → CPU keeps an F32 KV cache
    let inter = 128usize;
    let layers = 2usize;
    let vocab = 48usize;

    let info = ModelInfo {
        arch: ArchType::Llama,
        model_type: "llama".into(),
        vocab_size: vocab,
        hidden_size: hidden,
        num_hidden_layers: layers,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv,
        intermediate_size: inter,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-5,
        hidden_act: "silu".into(),
        rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        head_dim,
        raw: serde_json::Value::Null,
    };

    let mut next = lcg(0xC0FFEE);
    let mut w = HashMap::new();
    let mat = |rows: usize, cols: usize, scale: f32, n: &mut dyn FnMut() -> f32| {
        let data: Vec<f32> = (0..rows * cols).map(|_| n() * scale).collect();
        Tensor::from_f32_vec(data, Shape::new([rows, cols])).unwrap()
    };
    let norm = |dim: usize, n: &mut dyn FnMut() -> f32| {
        let data: Vec<f32> = (0..dim).map(|_| 1.0 + n() * 0.05).collect();
        Tensor::from_f32_vec(data, Shape::new([dim])).unwrap()
    };

    w.insert(
        "model.embed_tokens.weight".into(),
        mat(vocab, hidden, 0.1, &mut next),
    );
    w.insert("model.norm.weight".into(), norm(hidden, &mut next));
    w.insert("lm_head.weight".into(), mat(vocab, hidden, 0.1, &mut next));

    let qd = n_heads * head_dim;
    let kvd = n_kv * head_dim;
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        w.insert(
            format!("{p}.input_layernorm.weight"),
            norm(hidden, &mut next),
        );
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            norm(hidden, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            mat(qd, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            mat(kvd, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            mat(kvd, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            mat(hidden, qd, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            mat(inter, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            mat(inter, hidden, 0.1, &mut next),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            mat(hidden, inter, 0.1, &mut next),
        );
    }
    (info, w)
}

/// Quantize an f32 matrix into a raw-ggml-block Q8_0 tensor (34-byte blocks:
/// little-endian f16 scale + 32 int8) — the storage GGUF weights arrive in.
fn q8_0_tensor(data: &[f32], shape: [usize; 2]) -> Tensor {
    assert_eq!(data.len() % 32, 0);
    let mut blocks = Vec::with_capacity(data.len() / 32 * 34);
    for chunk in data.chunks_exact(32) {
        let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        blocks.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in chunk {
            blocks.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    Tensor::from_quant_bytes(&blocks, shape.to_vec(), DType::Q8_0).unwrap()
}

/// The tiny Llama with every 2-D weight stored as raw Q8_0 blocks (as a Q8_0 GGUF
/// ships them) and **no explicit lm_head** — the output projection ties to the
/// Q8_0 embedding, exercising the tied-buffer + Q8_0 embed-gather paths.
fn tiny_llama_q8_0() -> (ModelInfo, HashMap<String, Tensor>) {
    let (info, weights) = tiny_llama();
    let quantized = weights
        .into_iter()
        .filter(|(name, _)| name != "lm_head.weight") // tied: fall back to embed
        .map(|(name, t)| {
            if t.shape().dims().len() == 2 {
                let dims = t.shape().dims();
                let q = q8_0_tensor(t.as_f32_slice(), [dims[0], dims[1]]);
                (name, q)
            } else {
                (name, t) // norms stay f32
            }
        })
        .collect();
    (info, quantized)
}

/// Build a raw-ggml-block Q4_K tensor from **random valid block bytes** (small
/// positive f16 d/dmin so dequant magnitudes stay ~O(0.1); fully random scale and
/// qs bytes so every bit path is exercised, including the high-bit packing of
/// sub-blocks 4..7). Both engines decode the identical bytes, so no "quantizer"
/// is needed — the coherence gate is that they agree on what the bytes mean.
fn q4_k_random_tensor(shape: [usize; 2], next: &mut dyn FnMut() -> f32) -> Tensor {
    let numel = shape[0] * shape[1];
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
    Tensor::from_quant_bytes(&blocks, shape.to_vec(), DType::Q4_K).unwrap()
}

/// Build a raw-ggml-block Q6_K tensor from random valid block bytes (210 bytes per
/// 256 weights: ql[128] + qh[64] + 16 signed int8 scales + f16 d) — same
/// random-bit rationale as [`q4_k_random_tensor`]. Q6_K's +0/+2/+4/+6 per-half
/// scale indexing is the historical token-salad bug, so both engines decoding
/// identical bytes IS the regression net.
fn q6_k_random_tensor(shape: [usize; 2], next: &mut dyn FnMut() -> f32) -> Tensor {
    let numel = shape[0] * shape[1];
    assert_eq!(numel % 256, 0);
    let mut blocks = Vec::with_capacity(numel / 256 * 210);
    for _ in 0..numel / 256 {
        for _ in 0..192 {
            blocks.push((next().abs() * 255.0) as u8); // ql + qh
        }
        for _ in 0..16 {
            blocks.push((next() * 127.0) as i8 as u8); // signed scales
        }
        let d = half::f16::from_f32(2.0e-5 * (1.0 + next().abs()));
        blocks.extend_from_slice(&d.to_le_bytes());
    }
    Tensor::from_quant_bytes(&blocks, shape.to_vec(), DType::Q6_K).unwrap()
}

/// A tiny Llama with **mixed quantization**, as a real Q4_K_M GGUF ships: Q4_K for
/// most matrices whose row length is a multiple of 256 (q/k, gate/up/down, embed),
/// **Q6_K for v_proj and lm_head** (exactly where Q4_K_M files use it), Q8_0 where
/// the row length isn't a multiple of 256 (o_proj, k = 96), f32 norms.
/// head_dim = 24 keeps the CPU KV cache in f32 (not a multiple of 32).
fn tiny_llama_q4_k() -> (ModelInfo, HashMap<String, Tensor>) {
    let hidden = 256usize;
    let n_heads = 4usize;
    let n_kv = 2usize;
    let head_dim = 24usize;
    let inter = 512usize;
    let layers = 2usize;
    let vocab = 48usize;

    let info = ModelInfo {
        arch: ArchType::Llama,
        model_type: "llama".into(),
        vocab_size: vocab,
        hidden_size: hidden,
        num_hidden_layers: layers,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv,
        intermediate_size: inter,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-5,
        hidden_act: "silu".into(),
        rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        head_dim,
        raw: serde_json::Value::Null,
    };

    let mut next = lcg(0xBEEF);
    let mut w = HashMap::new();
    let norm = |dim: usize, n: &mut dyn FnMut() -> f32| {
        let data: Vec<f32> = (0..dim).map(|_| 1.0 + n() * 0.05).collect();
        Tensor::from_f32_vec(data, Shape::new([dim])).unwrap()
    };

    let qd = n_heads * head_dim; // 96 — not a multiple of 256, so o_proj can't be Q4_K
    let kvd = n_kv * head_dim; // 48
    w.insert(
        "model.embed_tokens.weight".into(),
        q4_k_random_tensor([vocab, hidden], &mut next),
    );
    w.insert("model.norm.weight".into(), norm(hidden, &mut next));
    w.insert(
        "lm_head.weight".into(),
        q6_k_random_tensor([vocab, hidden], &mut next),
    );
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        w.insert(
            format!("{p}.input_layernorm.weight"),
            norm(hidden, &mut next),
        );
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            norm(hidden, &mut next),
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            q4_k_random_tensor([qd, hidden], &mut next),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            q4_k_random_tensor([kvd, hidden], &mut next),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            q6_k_random_tensor([kvd, hidden], &mut next),
        );
        // o_proj rows are qd=96 wide → Q8_0 (mixed-quant, like real Q4_K_M files).
        let o: Vec<f32> = (0..hidden * qd).map(|_| next() * 0.1).collect();
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            q8_0_tensor(&o, [hidden, qd]),
        );
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            q4_k_random_tensor([inter, hidden], &mut next),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            q4_k_random_tensor([inter, hidden], &mut next),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            q4_k_random_tensor([hidden, inter], &mut next),
        );
    }
    (info, w)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
fn wgpu_logits_match_cpu_llama() {
    let (info, weights) = tiny_llama();

    let mut cpu =
        LlamaForward::from_weights(info.clone(), weights.clone()).expect("build CPU LlamaForward");
    let mut gpu = match WgpuForwardEngine::from_weights_with_kv(info, weights, Some(false)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no wgpu GPU adapter ({e}) — skipping coherence test");
            return;
        }
    };

    let tokens: Vec<u32> = vec![1, 5, 9, 3, 7, 2, 11];

    // Full-prompt last-token logits.
    let cpu_logits = cpu.forward_logits(&tokens, false).unwrap();
    let gpu_logits = gpu.forward_logits(&tokens, false).unwrap();
    assert_eq!(cpu_logits.len(), gpu_logits.len());

    let max_err = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 5e-3,
        "wgpu vs cpu prompt logits max_err={max_err} (argmax cpu={} gpu={})",
        argmax(&cpu_logits),
        argmax(&gpu_logits)
    );
    assert_eq!(
        argmax(&cpu_logits),
        argmax(&gpu_logits),
        "greedy next-token must match"
    );

    // Incremental decode (use_cache=true) must match a fresh CPU run of prompt+token.
    let next_tok = argmax(&gpu_logits) as u32;
    let gpu_step = gpu.forward_logits(&[next_tok], true).unwrap();
    let mut full = tokens.clone();
    full.push(next_tok);
    let cpu_step = cpu.forward_logits(&full, false).unwrap();
    let step_err = cpu_step
        .iter()
        .zip(&gpu_step)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        step_err < 5e-3,
        "wgpu incremental-decode logits max_err={step_err}"
    );
    assert_eq!(
        argmax(&cpu_step),
        argmax(&gpu_step),
        "decode greedy must match"
    );
}

/// Phase 7.1 gate: the GPU-resident Q8_0 path (raw ggml blocks uploaded without f32
/// expansion, dequantized in-shader) must agree with the CPU engine on the same
/// quantized weights. Both engines dequantize identical blocks, so weight rounding
/// cancels; the only expected divergence is the CPU's aarch64 W8A8 SDOT path, which
/// quantizes *activations* per 32-block while the GPU keeps activations f32. Greedy
/// agreement is the hard gate, with a bounded logit error on top.
#[test]
fn wgpu_q8_0_logits_match_cpu_llama() {
    let (info, weights) = tiny_llama_q8_0();

    let mut cpu = LlamaForward::from_weights(info.clone(), weights.clone())
        .expect("build CPU LlamaForward (Q8_0)");
    let mut gpu = match WgpuForwardEngine::from_weights_with_kv(info, weights, Some(false)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no wgpu GPU adapter ({e}) — skipping Q8_0 coherence test");
            return;
        }
    };

    let tokens: Vec<u32> = vec![1, 5, 9, 3, 7, 2, 11];

    let cpu_logits = cpu.forward_logits(&tokens, false).unwrap();
    let gpu_logits = gpu.forward_logits(&tokens, false).unwrap();
    assert_eq!(cpu_logits.len(), gpu_logits.len());

    let max_err = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        argmax(&cpu_logits),
        argmax(&gpu_logits),
        "Q8_0 greedy next-token must match (max_err={max_err})"
    );
    assert!(max_err < 0.1, "Q8_0 wgpu vs cpu logits max_err={max_err}");

    // Incremental decode through the GPU KV cache must stay coherent too.
    let next_tok = argmax(&gpu_logits) as u32;
    let gpu_step = gpu.forward_logits(&[next_tok], true).unwrap();
    let mut full = tokens.clone();
    full.push(next_tok);
    let cpu_step = cpu.forward_logits(&full, false).unwrap();
    let step_err = cpu_step
        .iter()
        .zip(&gpu_step)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        argmax(&cpu_step),
        argmax(&gpu_step),
        "Q8_0 decode greedy must match (max_err={step_err})"
    );
    assert!(step_err < 0.1, "Q8_0 decode logits max_err={step_err}");
}

/// K-quant gate (Phase 7.2 + Q6_K): the GPU-resident Q4_K path (raw 144-byte
/// super-blocks uploaded verbatim) and Q6_K path (210→212-byte padded blocks) must
/// produce the same logits as the CPU engine running the **same weights dequantized
/// to f32 on the host** (`to_f32_vec` — the reference decoder). Both sides then
/// compute with identical weight values and f32 activations, so the tolerance is as
/// tight as the all-f32 test — this pins the in-shader scale-indexing math exactly
/// (Q6_K's +0/+2/+4/+6 per-half indexing is the bug class that made it emit
/// token-salad). Comparing against the CPU *quantized* path instead would add its
/// aarch64 W4A8/W8A8 activation-quantization noise (~0.1 logit) and mask real
/// dequant bugs of the same magnitude.
#[test]
fn wgpu_k_quant_logits_match_cpu_llama() {
    let (info, weights) = tiny_llama_q4_k();

    // Host-dequantized twin: every quantized tensor expanded to f32 by the proven
    // CPU decoder; norms pass through.
    let dequant: HashMap<String, Tensor> = weights
        .iter()
        .map(|(name, t)| {
            let f = if t.dtype().is_quantized() {
                let dims = t.shape().dims().to_vec();
                Tensor::from_f32_vec(t.to_f32_vec(), Shape::new([dims[0], dims[1]])).unwrap()
            } else {
                t.clone()
            };
            (name.clone(), f)
        })
        .collect();

    let mut cpu = LlamaForward::from_weights(info.clone(), dequant)
        .expect("build CPU LlamaForward (dequantized f32 reference)");
    let mut gpu = match WgpuForwardEngine::from_weights_with_kv(info, weights, Some(false)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no wgpu GPU adapter ({e}) — skipping Q4_K coherence test");
            return;
        }
    };

    let tokens: Vec<u32> = vec![2, 40, 17, 5, 33, 8, 21];

    let cpu_logits = cpu.forward_logits(&tokens, false).unwrap();
    let gpu_logits = gpu.forward_logits(&tokens, false).unwrap();
    assert_eq!(cpu_logits.len(), gpu_logits.len());

    let max_err = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 5e-3,
        "K-quant wgpu vs dequant-f32 cpu logits max_err={max_err} (argmax cpu={} gpu={})",
        argmax(&cpu_logits),
        argmax(&gpu_logits)
    );
    assert_eq!(
        argmax(&cpu_logits),
        argmax(&gpu_logits),
        "K-quant greedy next-token must match"
    );

    let next_tok = argmax(&gpu_logits) as u32;
    let gpu_step = gpu.forward_logits(&[next_tok], true).unwrap();
    let mut full = tokens.clone();
    full.push(next_tok);
    let cpu_step = cpu.forward_logits(&full, false).unwrap();
    let step_err = cpu_step
        .iter()
        .zip(&gpu_step)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        step_err < 5e-3,
        "K-quant incremental-decode logits max_err={step_err}"
    );
    assert_eq!(
        argmax(&cpu_step),
        argmax(&gpu_step),
        "K-quant decode greedy must match"
    );
}

/// Phase 7.3 gate: the f16 KV cache (halves packed two-per-u32 word, decoded with
/// core-WGSL unpack2x16float — the auto default for even head_dim) must track the
/// forced-f32-cache engine on the same weights. K/V values round to f16 on write
/// (~5e-4 relative), so the bound is looser than the f32 gates but far below
/// anything a layout/packing bug would produce; greedy agreement is the hard gate.
/// Uses the Q8_0 tiny model (head_dim 16 — even) so quantized matmuls and the f16
/// cache are exercised together, decoding several tokens to grow the cache.
#[test]
fn wgpu_f16_kv_cache_matches_f32_kv_cache() {
    let (info, weights) = tiny_llama_q8_0();

    let mut gpu32 =
        match WgpuForwardEngine::from_weights_with_kv(info.clone(), weights.clone(), Some(false)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no wgpu GPU adapter ({e}) — skipping f16-KV coherence test");
                return;
            }
        };
    let mut gpu16 = WgpuForwardEngine::from_weights_with_kv(info, weights, Some(true))
        .expect("f16 KV engine (head_dim is even)");

    let tokens: Vec<u32> = vec![1, 5, 9, 3, 7, 2, 11];
    let mut l32 = gpu32.forward_logits(&tokens, false).unwrap();
    let mut l16 = gpu16.forward_logits(&tokens, false).unwrap();

    // Greedy-decode a few steps through the growing caches.
    for _ in 0..4 {
        let max_err = l32
            .iter()
            .zip(&l16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 2e-2, "f16 vs f32 KV logits max_err={max_err}");
        assert_eq!(
            argmax(&l32),
            argmax(&l16),
            "greedy must match (err={max_err})"
        );
        let tok = argmax(&l32) as u32;
        l32 = gpu32.forward_logits(&[tok], true).unwrap();
        l16 = gpu16.forward_logits(&[tok], true).unwrap();
    }
}

/// Phase 7.5 gate: chunked prefill (`forward_logits` on a long prompt → 128-token
/// `forward_chunk` batches with transposes + multi-token kv_append) must produce
/// the same logits as feeding the same prompt one token at a time through the
/// decode path. 300 tokens exercises two full chunks + a partial one, chunk
/// boundaries, and pos0 > 0 RoPE/attention offsets. Same engine config on both
/// sides (auto f16 KV), so the only difference is batched vs sequential math.
#[test]
fn wgpu_chunked_prefill_matches_per_token() {
    let (info, weights) = tiny_llama_q8_0();

    let mut chunked = match WgpuForwardEngine::from_weights(info.clone(), weights.clone()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no wgpu GPU adapter ({e}) — skipping chunked-prefill test");
            return;
        }
    };
    let mut per_token = WgpuForwardEngine::from_weights(info, weights).expect("engine");

    // 300 pseudo-random tokens in-vocab (48).
    let mut s = 0xFEED_u64;
    let prompt: Vec<u32> = (0..300)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % 48) as u32
        })
        .collect();

    let l_chunked = chunked.forward_logits(&prompt, false).unwrap();
    let mut l_seq = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        l_seq = per_token.forward_logits(&[tok], i != 0).unwrap();
    }

    let max_err = l_chunked
        .iter()
        .zip(&l_seq)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 5e-3,
        "chunked vs per-token prefill logits max_err={max_err}"
    );
    assert_eq!(
        argmax(&l_chunked),
        argmax(&l_seq),
        "greedy after prefill must match"
    );
}
