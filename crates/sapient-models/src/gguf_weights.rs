//! Map llama.cpp GGUF tensor names to HuggingFace layout for native forward passes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use sapient_core::Tensor;

const HF_PREFIX: &str = "model.";

/// Load a GGUF file and remap tensor names to HuggingFace `model.*` keys.
pub fn load_gguf_hf_weights(path: &Path) -> Result<HashMap<String, Tensor>> {
    let raw = sapient_io::GgufLoader::load_tensors(path)
        .with_context(|| format!("failed to load GGUF {}", path.display()))?;
    map_gguf_tensors_to_hf(raw)
}

/// Load a GGUF file via memory-mapping and remap tensor names to HF layout.
/// Q4_0/Q8_0 tensors point directly into the mmap'd file — zero heap copy.
pub fn load_gguf_hf_weights_mmap(path: &Path) -> Result<HashMap<String, Tensor>> {
    let (_, raw) = sapient_io::GgufLoader::load_tensors_mmap(path)
        .with_context(|| format!("failed to mmap GGUF {}", path.display()))?;
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
        key if key.starts_with("blk.") => map_blk_tensor(key),
        _ => None,
    }
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
    } else if let Some(s) = component.strip_suffix(".bias") {
        (s, "bias")
    } else {
        return None;
    };

    let hf_suffix = match suffix {
        "attn_norm" => format!("layers.{layer}.input_layernorm"),
        "attn_q" => format!("layers.{layer}.self_attn.q_proj"),
        "attn_k" => format!("layers.{layer}.self_attn.k_proj"),
        "attn_v" => format!("layers.{layer}.self_attn.v_proj"),
        "attn_output" => format!("layers.{layer}.self_attn.o_proj"),
        "ffn_norm" => format!("layers.{layer}.post_attention_layernorm"),
        "ffn_gate" => format!("layers.{layer}.mlp.gate_proj"),
        "ffn_up" => format!("layers.{layer}.mlp.up_proj"),
        "ffn_down" => format!("layers.{layer}.mlp.down_proj"),
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
}
