//! Map llama.cpp GGUF tensor names to HuggingFace layout for native forward passes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use sapient_core::{DType, Tensor};
use sapient_hub::model_info::ModelInfo;

const HF_PREFIX: &str = "model.";

/// Expand a GGUF path to all shards of its split set (`-NNNNN-of-MMMMM.gguf`),
/// discovered as sibling files in the same directory; a single-file GGUF returns
/// just itself. So the caller only needs shard 1 (which carries the metadata).
fn gguf_shard_paths(path: &Path) -> Vec<std::path::PathBuf> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    match sapient_hub::gguf_split_shards(name) {
        Some(shard_names) => {
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            shard_names.into_iter().map(|n| dir.join(n)).collect()
        }
        None => vec![path.to_path_buf()],
    }
}

/// Load a GGUF file (or all shards of a split set) and remap tensor names to
/// HuggingFace `model.*` keys.
pub fn load_gguf_hf_weights(path: &Path) -> Result<HashMap<String, Tensor>> {
    let mut raw = HashMap::new();
    for shard in gguf_shard_paths(path) {
        let t = sapient_io::GgufLoader::load_tensors(&shard)
            .with_context(|| format!("failed to load GGUF {}", shard.display()))?;
        raw.extend(t);
    }
    map_gguf_tensors_to_hf(raw)
}

/// Load a GGUF file (or split set) via memory-mapping and remap to HF layout.
/// Q4_0/Q8_0 tensors point directly into the mmap'd file — zero heap copy.
pub fn load_gguf_hf_weights_mmap(path: &Path) -> Result<HashMap<String, Tensor>> {
    let mut raw = HashMap::new();
    for shard in gguf_shard_paths(path) {
        let (_, t) = sapient_io::GgufLoader::load_tensors_mmap(&shard)
            .with_context(|| format!("failed to mmap GGUF {}", shard.display()))?;
        raw.extend(t);
    }
    map_gguf_tensors_to_hf(raw)
}

/// Convert a GGUF name → HF weight key map into the layout expected by `LlamaForward`.
///
/// GGUF stores tensor dims in ggml convention: 2-D weight matrices are `[n_cols, n_rows]`
/// i.e. the shape is the transpose of the HF `[out_features, in_features]` convention.
/// We swap the dims so the shape matches what the forward pass expects.
pub fn map_gguf_tensors_to_hf(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut mapped = HashMap::with_capacity(raw.len());

    for (name, tensor) in raw {
        match map_gguf_tensor_name(&name) {
            Some(hf_key) => {
                // Bias tensors (1-D) keep their shape. For 2-D weight matrices, GGUF
                // dim order is [in, out] but HF expects [out, in] — flip the shape.
                let tensor = if hf_key.ends_with(".weight") && tensor.shape().ndim() == 2 {
                    let dims = tensor.shape().dims().to_vec();
                    let new_shape = sapient_core::Shape::new(vec![dims[1], dims[0]]);
                    tensor
                        .reshape(new_shape)
                        .map_err(|e| anyhow::anyhow!("reshape failed for '{name}': {e}"))?
                } else {
                    tensor
                };
                if mapped.insert(hf_key.clone(), tensor).is_some() {
                    bail!("duplicate mapped weight key '{hf_key}' from GGUF tensor '{name}'");
                }
            }
            None => {
                // Unknown tensor names (e.g. MoE expert weights, RoPE freq caches) are
                // silently skipped — they aren't part of the HF forward pass we support.
                tracing::debug!(tensor = %name, "skipping unmapped GGUF tensor");
            }
        }
    }

    // Require minimum Llama weights.
    if !mapped.contains_key(&format!("{HF_PREFIX}embed_tokens.weight")) {
        bail!("GGUF file missing token embedding weights (token_embd.weight)");
    }

    Ok(mapped)
}

/// Un-permute the q_proj/k_proj rows of llama-family GGUF weights into HF layout.
///
/// llama.cpp's HF→GGUF converter permutes the rows of `attn_q`/`attn_k` so that
/// ggml's NORM-style RoPE (`rope_type = NORM`, used for the `llama` architecture:
/// Llama, Mistral, SmolLM, TinyLlama, …) produces the right result. SAPIENT's
/// RoPE is HF/NEOX-style (`rotate_half`), so the permuted weights make RoPE
/// scramble positions across each head → incoherent "token-salad" output. We
/// invert the permutation at load time so q/k match SAPIENT's RoPE.
///
/// Architectures that already use NEOX RoPE in ggml (Qwen2, Gemma) are NOT
/// permuted by the converter and must be left untouched — hence this is gated on
/// the `llama` architecture by the caller.
///
/// The forward permutation maps HF row `h·D + a·(D/2) + b` (head `h`, half `a∈{0,1}`,
/// `b∈0..D/2`) to GGUF row `h·D + b·2 + a`; we apply the inverse. Works on any
/// dtype because each output row is a contiguous `byte_count(in)` byte chunk.
pub fn unpermute_qk_rows(t: &Tensor, n_head: usize, head_dim: usize) -> Result<Tensor> {
    let dims = t.shape().dims().to_vec();
    if dims.len() != 2 || head_dim < 2 || head_dim % 2 != 0 || n_head * head_dim != dims[0] {
        // Shape doesn't match the expected [n_head*head_dim, in] — leave as-is.
        return Ok(t.clone());
    }
    let (out, in_dim) = (dims[0], dims[1]);
    let half = head_dim / 2;
    let row_bytes = t.dtype().byte_count(in_dim);
    let src = t.as_bytes();
    if src.len() < out * row_bytes {
        bail!(
            "unpermute_qk_rows: buffer too small ({} < {})",
            src.len(),
            out * row_bytes
        );
    }
    let mut dst = vec![0u8; out * row_bytes];
    for h in 0..n_head {
        for a in 0..2 {
            for b in 0..half {
                let hf_row = h * head_dim + a * half + b;
                let gguf_row = h * head_dim + b * 2 + a;
                dst[hf_row * row_bytes..hf_row * row_bytes + row_bytes]
                    .copy_from_slice(&src[gguf_row * row_bytes..gguf_row * row_bytes + row_bytes]);
            }
        }
    }

    if t.dtype().is_quantized() {
        Tensor::from_quant_bytes(&dst, dims, t.dtype()).map_err(|e| anyhow::anyhow!("{e}"))
    } else if t.dtype() == sapient_core::DType::F32 {
        let f: Vec<f32> = dst
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Tensor::from_f32(&f, dims).map_err(|e| anyhow::anyhow!("{e}"))
    } else {
        bail!("unpermute_qk_rows: unsupported dtype {:?}", t.dtype())
    }
}

/// Slice `nrows` rows starting at `start_row` from a 1-D or 2-D tensor, preserving
/// dtype via per-row byte copies (quantized rows align to block boundaries because
/// `in` is a multiple of the block size). Used to split fused Phi tensors.
fn slice_rows(t: &Tensor, start_row: usize, nrows: usize) -> Result<Tensor> {
    let dims = t.shape().dims().to_vec();
    let (total_rows, in_dim, two_d) = match dims.len() {
        2 => (dims[0], dims[1], true),
        1 => (dims[0], 1, false),
        _ => bail!("slice_rows: expected 1-D or 2-D tensor, got {dims:?}"),
    };
    if start_row + nrows > total_rows {
        bail!("slice_rows out of range: {start_row}+{nrows} > {total_rows}");
    }
    let row_bytes = t.dtype().byte_count(in_dim);
    let src = t.as_bytes();
    let (off, end) = (start_row * row_bytes, (start_row + nrows) * row_bytes);
    if end > src.len() {
        bail!("slice_rows: buffer too small ({} < {end})", src.len());
    }
    let bytes = &src[off..end];
    let new_dims = if two_d {
        vec![nrows, in_dim]
    } else {
        vec![nrows]
    };
    match t.dtype() {
        DType::F32 => {
            let f: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_f32(&f, new_dims).map_err(|e| anyhow::anyhow!("{e}"))
        }
        DType::F16 => Tensor::from_f16_bytes(bytes, new_dims).map_err(|e| anyhow::anyhow!("{e}")),
        d if d.is_quantized() => {
            Tensor::from_quant_bytes(bytes, new_dims, d).map_err(|e| anyhow::anyhow!("{e}"))
        }
        other => bail!("slice_rows: unsupported dtype {other:?}"),
    }
}

/// Split the **stacked** 3-D MoE expert tensors of a GGUF into per-expert 2-D
/// weights, so the CPU MoE forward path is format-agnostic (identical keys to the
/// HF safetensors layout).
///
/// llama.cpp packs a layer's experts into one tensor per projection:
/// `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` (mapped here to
/// `block_sparse_moe.experts_stacked.{w1,w3,w2}`), loaded in ggml `ne` order
/// `[d0, d1, n_expert]` with the expert axis **outermost** (slowest-varying). So
/// expert `e`'s matrix is the contiguous byte range `[e·eb, (e+1)·eb)` and its HF
/// `[out, in]` shape is `[d1, d0]` — the same axis swap the 2-D loader applies.
/// `d0·d1` is always a whole number of quant blocks (hidden/ff are multiples of
/// 256), so the byte split never straddles a block boundary. This runs before the
/// engine's online-quant / repack passes, which then see ordinary 2-D experts.
pub fn split_moe_gguf_experts(
    weights: &mut HashMap<String, Tensor>,
    info: &ModelInfo,
) -> Result<()> {
    let Some(moe) = &info.moe else {
        return Ok(());
    };
    let n_expert = moe.num_experts;
    for layer in 0..info.num_hidden_layers {
        for w in ["w1", "w3", "w2"] {
            let key =
                format!("{HF_PREFIX}layers.{layer}.block_sparse_moe.experts_stacked.{w}.weight");
            let Some(t) = weights.remove(&key) else {
                continue;
            };
            let dims = t.shape().dims().to_vec();
            if dims.len() != 3 {
                bail!("stacked MoE tensor '{key}' expected 3-D [d0,d1,n_expert], got {dims:?}");
            }
            let (d0, d1, ne) = (dims[0], dims[1], dims[2]);
            if ne != n_expert {
                bail!("stacked MoE tensor '{key}' has {ne} experts, config declares {n_expert}");
            }
            let dtype = t.dtype();
            let expert_bytes = dtype.byte_count(d0 * d1);
            if expert_bytes * ne > t.as_bytes().len() {
                bail!(
                    "stacked MoE tensor '{key}' buffer too small: {} < {}",
                    t.as_bytes().len(),
                    expert_bytes * ne
                );
            }
            // ZERO-COPY: each per-expert tensor SHARES the stacked buffer (mmap or
            // heap) at a byte offset — no expert data is copied. All MoE models
            // mmap, so this keeps peak RSS ≈ file size instead of ~2× (a 63 GB
            // GLM-4.5-Air measured 119 GB with the byte-copy → near-OOM thrashing
            // on the 122 GB box, decode 0.34 tok/s). The per-expert views stay
            // mmap'd, so the aarch64 Q4_K→R4 repack (heap-only) skips them — fine,
            // since R4 doesn't help m=1 decode anyway.
            for e in 0..ne {
                // Per-expert HF 2-D shape is [out, in] = [d1, d0] (reverse of ggml ne).
                let expert = Tensor::from_buffer(
                    sapient_core::Shape::new(vec![d1, d0]),
                    dtype,
                    t.buffer().clone(),
                    t.offset() + e * expert_bytes,
                )
                .map_err(|err| anyhow::anyhow!("{err}"))?;
                weights.insert(
                    format!("{HF_PREFIX}layers.{layer}.block_sparse_moe.experts.{e}.{w}.weight"),
                    expert,
                );
            }
        }
    }
    Ok(())
}

/// Prepare Phi-family GGUF weights for `PhiForward`. Phi GGUFs fuse Q/K/V into one
/// `attn_qkv` tensor (mapped to `self_attn.qkv_proj`) which we split into separate
/// q/k/v by GQA dims; Phi-3/4 additionally fuse gate+up into `ffn_up` (mapped to
/// `mlp.up_proj`) which `PhiForward` expects as `mlp.gate_up_proj`. Phi-1/1.5/2 use
/// `mlp.fc1`/`fc2` (GELU). Splits weights and (for Phi-2) biases.
pub fn split_phi_gguf_fused(
    weights: &mut HashMap<String, Tensor>,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    is_phi3: bool,
) -> Result<()> {
    let q_rows = n_heads * head_dim;
    let k_rows = n_kv * head_dim;
    let v_rows = n_kv * head_dim;
    let total = q_rows + k_rows + v_rows;

    let qkv_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("self_attn.qkv_proj."))
        .cloned()
        .collect();
    for key in qkv_keys {
        let (base, suffix) = key.split_once("qkv_proj.").unwrap();
        let (base, suffix) = (base.to_string(), suffix.to_string()); // "weight" | "bias"
        let t = weights.remove(&key).unwrap();
        let actual = t.shape().dims()[0];
        if actual != total {
            bail!(
                "Phi qkv split for '{key}': expected {total} rows \
                 (q {q_rows} + k {k_rows} + v {v_rows}) but tensor has {actual} — \
                 check head_count/head_count_kv/head_dim"
            );
        }
        weights.insert(format!("{base}q_proj.{suffix}"), slice_rows(&t, 0, q_rows)?);
        weights.insert(
            format!("{base}k_proj.{suffix}"),
            slice_rows(&t, q_rows, k_rows)?,
        );
        weights.insert(
            format!("{base}v_proj.{suffix}"),
            slice_rows(&t, q_rows + k_rows, v_rows)?,
        );
    }

    // Rename the generically-mapped FFN tensors to what PhiForward expects.
    let rename = |w: &mut HashMap<String, Tensor>, from: &str, to: &str| {
        let keys: Vec<String> = w.keys().filter(|k| k.contains(from)).cloned().collect();
        for k in keys {
            let nk = k.replace(from, to);
            let t = w.remove(&k).unwrap();
            w.insert(nk, t);
        }
    };
    if is_phi3 {
        // ggml's phi3 `ffn_up` is the FUSED gate_up (generically mapped to up_proj).
        rename(weights, "mlp.up_proj.", "mlp.gate_up_proj.");
    } else {
        rename(weights, "mlp.up_proj.", "mlp.fc1.");
        rename(weights, "mlp.down_proj.", "mlp.fc2.");
    }
    Ok(())
}

/// Apply [`unpermute_qk_rows`] to every `self_attn.q_proj`/`k_proj` weight in a
/// freshly-loaded llama-family GGUF weight map. `n_head`/`n_kv_head` come from the
/// model config; `head_dim` is `hidden / n_head`.
pub fn unpermute_llama_gguf_qk(
    weights: &mut HashMap<String, Tensor>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<()> {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        let nh = if key.ends_with("self_attn.q_proj.weight") {
            Some(n_head)
        } else if key.ends_with("self_attn.k_proj.weight") {
            Some(n_kv_head)
        } else {
            None
        };
        if let Some(nh) = nh {
            let t = weights.get(&key).unwrap();
            let fixed = unpermute_qk_rows(t, nh, head_dim)?;
            weights.insert(key, fixed);
        }
    }
    Ok(())
}

/// Map a single GGUF tensor name to a HuggingFace weight key.
pub fn map_gguf_tensor_name(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" | "tok_embeddings.weight" => {
            Some(format!("{HF_PREFIX}embed_tokens.weight"))
        }
        "output_norm.weight" | "norm.weight" => Some(format!("{HF_PREFIX}norm.weight")),
        "output.weight" | "lm_head.weight" => Some("lm_head.weight".into()),
        key if key.starts_with("model.") => Some(key.to_string()),
        // GLM4-MoE / DeepSeek router correction bias `blk.{i}.exp_probs_b.bias`.
        // Handled here (not `map_blk_tensor`, which would append a stray `.bias`)
        // so the key exactly matches the `e_score_correction_bias` lookup in moe_ffn.
        key if key.starts_with("blk.") && key.contains(".exp_probs_b") => {
            let layer = key.strip_prefix("blk.")?.split('.').next()?;
            if !layer.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            Some(format!(
                "{HF_PREFIX}layers.{layer}.block_sparse_moe.gate.e_score_correction_bias"
            ))
        }
        key if key.starts_with("blk.") => map_blk_tensor(key),
        _ => None,
    }
}

/// Parse an older-style MoE per-expert projection suffix (`ffn_gate.3`,
/// `ffn_up.0`, `ffn_down.7`) into `(canonical_proj, expert_index)` — e.g.
/// `("w1", "3")`. Returns `None` for dense projections (`ffn_gate`, no index) and
/// everything else, so it never shadows the dense arms.
fn expert_proj_2d(suffix: &str) -> Option<(&'static str, &str)> {
    let (proj, e) = suffix.rsplit_once('.')?;
    if e.is_empty() || !e.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let w = match proj {
        "ffn_gate" => "w1",
        "ffn_up" => "w3",
        "ffn_down" => "w2",
        _ => return None,
    };
    Some((w, e))
}

fn map_blk_tensor(key: &str) -> Option<String> {
    // blk.{layer}.{component}.(weight|bias)
    let rest = key.strip_prefix("blk.")?;
    let (layer, component) = rest.split_once('.')?;
    if !layer.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    // Determine the tensor kind (.weight or .bias) and the component name.
    let (suffix, kind) = if let Some(s) = component.strip_suffix(".weight") {
        (s, "weight")
    } else {
        (component.strip_suffix(".bias")?, "bias")
    };

    let hf_suffix = match suffix {
        "attn_norm" => format!("layers.{layer}.input_layernorm"),
        "attn_q" => format!("layers.{layer}.self_attn.q_proj"),
        "attn_k" => format!("layers.{layer}.self_attn.k_proj"),
        "attn_v" => format!("layers.{layer}.self_attn.v_proj"),
        // Phi (and some others) fuse Q/K/V into one tensor; preserve it so the Phi
        // GGUF loader can split it into q/k/v by GQA dims (`split_phi_gguf_fused`).
        "attn_qkv" => format!("layers.{layer}.self_attn.qkv_proj"),
        "attn_output" => format!("layers.{layer}.self_attn.o_proj"),
        "ffn_norm" => format!("layers.{layer}.post_attention_layernorm"),
        // GLM4-MoE names the post-attention norm `post_attention_norm`.
        "post_attention_norm" => format!("layers.{layer}.post_attention_layernorm"),
        "ffn_gate" => format!("layers.{layer}.mlp.gate_proj"),
        "ffn_up" => format!("layers.{layer}.mlp.up_proj"),
        "ffn_down" => format!("layers.{layer}.mlp.down_proj"),
        // GLM4-MoE / DeepSeek shared expert (`*_shexp`, always-on FFN).
        "ffn_gate_shexp" => format!("layers.{layer}.block_sparse_moe.shared_expert.w1"),
        "ffn_up_shexp" => format!("layers.{layer}.block_sparse_moe.shared_expert.w3"),
        "ffn_down_shexp" => format!("layers.{layer}.block_sparse_moe.shared_expert.w2"),
        // MoE router (Mixtral folded into the `llama` arch).
        "ffn_gate_inp" => format!("layers.{layer}.block_sparse_moe.gate"),
        // MoE experts, NEWER stacked format: `*_exps` is all experts in one 3-D
        // blob `[.., .., n_expert]` — `split_moe_gguf_experts` slices them per-expert.
        "ffn_gate_exps" => format!("layers.{layer}.block_sparse_moe.experts_stacked.w1"),
        "ffn_up_exps" => format!("layers.{layer}.block_sparse_moe.experts_stacked.w3"),
        "ffn_down_exps" => format!("layers.{layer}.block_sparse_moe.experts_stacked.w2"),
        // MoE experts, OLDER per-expert format (TheBloke Mixtral): each expert is
        // its own 2-D tensor `ffn_{gate,up,down}.{e}` — maps straight to the
        // canonical per-expert key (no split needed).
        s if expert_proj_2d(s).is_some() => {
            let (w, e) = expert_proj_2d(s).unwrap();
            format!("layers.{layer}.block_sparse_moe.experts.{e}.{w}")
        }
        _ => return None,
    };

    Some(format!("{HF_PREFIX}{hf_suffix}.{kind}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // llama.cpp's forward HF→GGUF permutation (the thing we must invert):
    // GGUF[h·D + b·2 + a] = HF[h·D + a·(D/2) + b].
    fn llama_cpp_permute(hf_rows: &[Vec<f32>], n_head: usize, head_dim: usize) -> Vec<Vec<f32>> {
        let half = head_dim / 2;
        let mut gguf = vec![Vec::new(); hf_rows.len()];
        for h in 0..n_head {
            for a in 0..2 {
                for b in 0..half {
                    let hf_row = h * head_dim + a * half + b;
                    let gguf_row = h * head_dim + b * 2 + a;
                    gguf[gguf_row] = hf_rows[hf_row].clone();
                }
            }
        }
        gguf
    }

    #[test]
    fn unpermute_qk_inverts_llama_cpp_permutation() {
        let (n_head, head_dim, hidden) = (3usize, 4usize, 5usize);
        let out = n_head * head_dim;
        // Distinct HF rows so any mis-mapping is detectable.
        let hf_rows: Vec<Vec<f32>> = (0..out)
            .map(|r| (0..hidden).map(|c| (r * 100 + c) as f32).collect())
            .collect();
        // ggml stores the permuted weights; build that flat F32 buffer.
        let gguf_rows = llama_cpp_permute(&hf_rows, n_head, head_dim);
        let flat: Vec<f32> = gguf_rows.iter().flatten().copied().collect();
        let gguf = sapient_core::Tensor::from_f32(&flat, vec![out, hidden]).unwrap();

        // Un-permuting must recover the original HF layout exactly.
        let recovered = unpermute_qk_rows(&gguf, n_head, head_dim).unwrap();
        let r = recovered.to_f32_cow();
        let expected: Vec<f32> = hf_rows.iter().flatten().copied().collect();
        assert_eq!(r.as_ref(), expected.as_slice());
    }

    #[test]
    fn unpermute_qk_leaves_mismatched_shapes_untouched() {
        // 1-D / shape that isn't [n_head*head_dim, in] must be returned unchanged.
        let t = sapient_core::Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let out = unpermute_qk_rows(&t, 3, 4).unwrap(); // 3*4 != 4
        assert_eq!(out.to_f32_cow().as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn maps_llama_gguf_names() {
        assert_eq!(
            map_gguf_tensor_name("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("blk.0.attn_q.weight").as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("output.weight").as_deref(),
            Some("lm_head.weight")
        );
    }

    #[test]
    fn maps_mixtral_moe_gguf_names() {
        // Router.
        assert_eq!(
            map_gguf_tensor_name("blk.0.ffn_gate_inp.weight").as_deref(),
            Some("model.layers.0.block_sparse_moe.gate.weight")
        );
        // NEWER stacked format (Qwen3-A3B / modern Mixtral requants): `*_exps`.
        assert_eq!(
            map_gguf_tensor_name("blk.3.ffn_gate_exps.weight").as_deref(),
            Some("model.layers.3.block_sparse_moe.experts_stacked.w1.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("blk.3.ffn_up_exps.weight").as_deref(),
            Some("model.layers.3.block_sparse_moe.experts_stacked.w3.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("blk.3.ffn_down_exps.weight").as_deref(),
            Some("model.layers.3.block_sparse_moe.experts_stacked.w2.weight")
        );
        // OLDER per-expert format (TheBloke Mixtral): `ffn_{gate,up,down}.{e}`.
        assert_eq!(
            map_gguf_tensor_name("blk.0.ffn_gate.5.weight").as_deref(),
            Some("model.layers.0.block_sparse_moe.experts.5.w1.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("blk.0.ffn_up.5.weight").as_deref(),
            Some("model.layers.0.block_sparse_moe.experts.5.w3.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("blk.7.ffn_down.0.weight").as_deref(),
            Some("model.layers.7.block_sparse_moe.experts.0.w2.weight")
        );
        // Dense projections (no expert index) must NOT be mistaken for experts.
        assert_eq!(
            map_gguf_tensor_name("blk.0.ffn_gate.weight").as_deref(),
            Some("model.layers.0.mlp.gate_proj.weight")
        );
    }

    #[test]
    fn maps_glm4moe_gguf_names() {
        // GLM4-MoE FFN norm name + shared expert + correction bias.
        assert_eq!(
            map_gguf_tensor_name("blk.1.post_attention_norm.weight").as_deref(),
            Some("model.layers.1.post_attention_layernorm.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("blk.1.ffn_gate_shexp.weight").as_deref(),
            Some("model.layers.1.block_sparse_moe.shared_expert.w1.weight")
        );
        assert_eq!(
            map_gguf_tensor_name("blk.1.ffn_down_shexp.weight").as_deref(),
            Some("model.layers.1.block_sparse_moe.shared_expert.w2.weight")
        );
        // Correction bias: NO stray `.bias` — must match the moe_ffn lookup exactly.
        assert_eq!(
            map_gguf_tensor_name("blk.1.exp_probs_b.bias").as_deref(),
            Some("model.layers.1.block_sparse_moe.gate.e_score_correction_bias")
        );
        // Stacked experts + router are shared with Mixtral.
        assert_eq!(
            map_gguf_tensor_name("blk.1.ffn_gate_exps.weight").as_deref(),
            Some("model.layers.1.block_sparse_moe.experts_stacked.w1.weight")
        );
    }

    // A tiny MoE ModelInfo with `n_expert` experts and one layer.
    fn tiny_moe_info(n_expert: usize) -> ModelInfo {
        let cfg = format!(
            r#"{{"architectures":["MixtralForCausalLM"],"model_type":"mixtral",
                 "vocab_size":32,"hidden_size":4,"num_hidden_layers":1,
                 "num_attention_heads":2,"num_key_value_heads":2,"intermediate_size":6,
                 "max_position_embeddings":64,"rms_norm_eps":1e-5,"rope_theta":1e6,
                 "num_local_experts":{n_expert},"num_experts_per_tok":2}}"#
        );
        ModelInfo::from_json_str(&cfg).unwrap()
    }

    #[test]
    fn split_moe_experts_slices_stacked_blob_correctly() {
        // Real GGUF experts are quantized (Q4_K/Q6_K/Q8_0); use Q8_0 here so the test
        // exercises the actual byte path (`as_quant_blocks`), and — since the split is
        // now ZERO-COPY (per-expert views sharing the stacked buffer at a byte offset)
        // — proves each view reads exactly its own expert's bytes, not the neighbour's.
        // Stacked ggml layout is `[d0, d1, n_expert]`, expert axis outermost; each
        // per-expert output is `[d1, d0]` = HF `[out, in]`.
        let (d0, d1, ne) = (32usize, 2usize, 2usize); // d0*d1 = 64 = 2 Q8_0 blocks/expert
        let eb = DType::Q8_0.byte_count(d0 * d1); // 68 bytes/expert
                                                  // Distinct bytes per expert so any mis-slice (or unbounded read) is detectable.
        let expert0: Vec<u8> = (0..eb).map(|i| (i % 251) as u8).collect();
        let expert1: Vec<u8> = (0..eb).map(|i| (100 + i % 151) as u8).collect();
        let mut buf = expert0.clone();
        buf.extend_from_slice(&expert1);
        let stacked = Tensor::from_quant_bytes(&buf, vec![d0, d1, ne], DType::Q8_0).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.block_sparse_moe.experts_stacked.w1.weight".to_string(),
            stacked,
        );
        split_moe_gguf_experts(&mut weights, &tiny_moe_info(ne)).unwrap();

        // The stacked key is consumed; per-expert 2-D [d1,d0] tensors appear.
        assert!(!weights.contains_key("model.layers.0.block_sparse_moe.experts_stacked.w1.weight"));
        let e0 = weights
            .get("model.layers.0.block_sparse_moe.experts.0.w1.weight")
            .unwrap();
        let e1 = weights
            .get("model.layers.0.block_sparse_moe.experts.1.w1.weight")
            .unwrap();
        assert_eq!(
            e0.shape().dims(),
            &[d1, d0],
            "per-expert shape is [out, in]"
        );
        assert_eq!(e0.as_quant_blocks(), expert0.as_slice());
        assert_eq!(e1.as_quant_blocks(), expert1.as_slice());
    }

    #[test]
    fn split_moe_experts_rejects_expert_count_mismatch() {
        let (d0, d1, ne) = (4usize, 6usize, 3usize);
        let stacked = Tensor::from_f32_vec(vec![0.0; d0 * d1 * ne], vec![d0, d1, ne]).unwrap();
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.block_sparse_moe.experts_stacked.w1.weight".to_string(),
            stacked,
        );
        // Config declares 2 experts but the blob has 3 → must error, not silently mis-split.
        assert!(split_moe_gguf_experts(&mut weights, &tiny_moe_info(2)).is_err());
    }
}
