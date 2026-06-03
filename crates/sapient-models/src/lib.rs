//! `sapient-models` — pre-built LLM architecture graph builders.
//!
//! Each architecture module builds a SAPIENT `Graph` from a `ModelInfo`,
//! matching the exact HuggingFace architecture for that model family.
//!
//! # Supported architectures
//! | HuggingFace class | Module | Models |
//! |---|---|---|
//! | `LlamaForCausalLM` | `llama` | Llama 1/2/3, Mistral, CodeLlama, Vicuna, WizardLM |
//! | `PhiForCausalLM` | `phi` | Phi-1/2/3/3.5 |
//! | `GemmaForCausalLM` | `gemma` | Gemma, Gemma 2 |
//! | `GPT2LMHeadModel` | `gpt2` | GPT-2, CodeGen, GPT-J |
//! | `BertForMaskedLM` | `bert` | BERT, RoBERTa, DistilBERT |
//! | `Qwen2ForCausalLM` | `qwen` | Qwen, Qwen2, Qwen2.5 |
//! | `MixtralForCausalLM` | `mixtral` | Mixtral-8x7B, Mixtral-8x22B |

pub mod architectures;
pub mod forward;
pub mod gguf_weights;
pub mod registry;
pub mod weights;

pub use forward::{
    mac_gpu_support, total_system_ram_bytes, AudioEngine, ForwardEngine, KokoroConfig, KokoroModel,
    LlmBackendKind, MacGpuSupport, WhisperForward, KOKORO_SAMPLE_RATE,
};
pub use registry::{build_graph, ModelGraph};
