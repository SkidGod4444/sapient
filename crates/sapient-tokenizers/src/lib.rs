//! `sapient-tokenizers` — HuggingFace-compatible tokenization.
//!
//! Wraps the official HuggingFace `tokenizers` Rust crate, which supports:
//! - BPE (GPT-2, Llama, Falcon, Phi, Qwen)
//! - WordPiece (BERT, RoBERTa, DistilBERT)
//! - SentencePiece (T5, Gemma, Llama)
//!
//! Also provides Jinja2 chat template rendering for chat models.

pub mod chat;
pub mod tokenizer;
pub mod whisper;

pub use chat::{ChatMessage, ChatRole, ChatTemplate, ToolCall};
pub use tokenizer::{SapientTokenizer, TokenizerOptions};
pub use whisper::WhisperTokenizer;
