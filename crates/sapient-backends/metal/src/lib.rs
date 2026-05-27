//! Apple MLX / Metal GPU backend for SAPIENT.
//!
//! This backend runs operations on Apple Silicon's GPU via the MLX framework.
//! MLX uses unified memory (same RAM for CPU and GPU) — no copy overhead.
//!
//! # Status
//! - [ ] MLX tensor bridge (`mlx-rs` crate)
//! - [ ] MatMul via `mlx::ops::matmul`
//! - [ ] Elementwise ops via `mlx::ops::*`
//! - [ ] Softmax, LayerNorm
//! - [ ] Automatic graph dispatch
//!
//! # Usage (when complete)
//! ```ignore
//! use sapient_backends_metal::MlxBackend;
//! use sapient_runtime::{InferenceSession, Model, ModelConfig, SessionOptions};
//!
//! let model = Model::load("model.onnx".as_ref(), ModelConfig {
//!     backend: "metal".into(),
//!     ..Default::default()
//! }).unwrap();
//!
//! let session = InferenceSession::new_with_backend(
//!     model,
//!     MlxBackend::new(),
//!     SessionOptions::default(),
//! ).unwrap();
//! ```

#[cfg(target_os = "macos")]
pub mod backend;

#[cfg(target_os = "macos")]
pub use backend::MlxBackend;

#[cfg(target_os = "macos")]
pub type MetalBackend = MlxBackend;
