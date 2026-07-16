// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `sapient-hub` — HuggingFace Hub integration.
//!
//! Download any model from the HuggingFace Hub, with:
//! - Token authentication (reads `HF_TOKEN` env var or `~/.cache/huggingface/token`)
//! - XDG-compatible local cache (`~/.cache/sapient/hub/`)
//! - SHA256 integrity checks
//! - Fast parallel downloads (HTTP range chunks + concurrent shards)
//! - Automatic architecture detection from `config.json`
//!
//! # Example
//! ```no_run
//! use sapient_hub::{HubClient, ModelFiles};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = HubClient::new()?;
//!     let files = client.download("microsoft/phi-2").await?;
//!     println!("Config: {}", files.config_path.display());
//!     println!("Weights: {:?}", files.weight_paths);
//!     Ok(())
//! }
//! ```

pub mod cache;
pub mod client;
pub mod download;
pub mod gguf;
pub mod model_info;
pub mod registry;
pub mod resolver;
pub mod snac_config;
pub mod whisper_config;

pub use client::{HubClient, LoadOptions};
pub use gguf::{gguf_split_shards, select_best_gguf, tokenizer_fallback_model};
pub use model_info::{ArchType, ModelInfo};
pub use registry::resolve_model_alias;
pub use resolver::{ModelFiles, WeightFormat};
pub use snac_config::SnacConfig;
pub use whisper_config::{WhisperConfig, WhisperGenConfig};
