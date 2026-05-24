//! Architecture registry — dispatch from ArchType → graph builder.

use anyhow::{bail, Result};
use sapient_ir::graph::Graph;

use sapient_hub::model_info::{ArchType, ModelInfo};

use crate::architectures::{bert, gemma, gpt2, llama, mixtral, phi, qwen};

/// A fully-built model graph ready for execution.
pub struct ModelGraph {
    pub graph: Graph,
    pub info:  ModelInfo,
}

/// Build a SAPIENT `Graph` from a parsed `ModelInfo`.
///
/// The returned graph has named inputs:
/// - `"input_ids"` — (batch, seq_len) i32 token IDs
/// - `"attention_mask"` — (batch, seq_len) i32 (1=attend, 0=mask) [optional]
/// - `"position_ids"` — (batch, seq_len) i32 [optional, for RoPE offset]
///
/// And named outputs:
/// - `"logits"` — (batch, seq_len, vocab_size) f32
pub fn build_graph(info: &ModelInfo) -> Result<ModelGraph> {
    let graph = match &info.arch {
        ArchType::Llama   => llama::build(info)?,
        ArchType::Phi     => phi::build(info)?,
        ArchType::Gemma   => gemma::build(info)?,
        ArchType::Gpt2    => gpt2::build(info)?,
        ArchType::Bert    => bert::build(info)?,
        ArchType::Qwen    => qwen::build(info)?,
        ArchType::Mixtral => mixtral::build(info)?,
        ArchType::Falcon  => {
            // Falcon uses a Llama-like architecture with ALiBi — use Llama builder.
            llama::build(info)?
        }
        ArchType::Unknown(name) => bail!(
            "Unsupported architecture: '{name}'. \
             If this is a GGUF model, load it directly via `Pipeline::from_gguf(path)`."
        ),
        _ => bail!("Architecture {:?} not yet implemented", info.arch),
    };

    Ok(ModelGraph { graph, info: info.clone() })
}
