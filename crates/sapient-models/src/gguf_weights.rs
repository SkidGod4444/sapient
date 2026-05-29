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
