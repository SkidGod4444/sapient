//! `sapient-hub` — HuggingFace Hub integration.
//!
//! Download any model from the HuggingFace Hub, with:
//! - Token authentication (reads `HF_TOKEN` env var or `~/.cache/huggingface/token`)
//! - XDG-compatible local cache (`~/.cache/sapient/hub/`)
//! - SHA256 integrity checks
//! - Progress bars via `indicatif`
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
pub mod model_info;
pub mod resolver;

pub use client::{HubClient, LoadOptions};
pub use model_info::{ArchType, ModelInfo};
pub use resolver::ModelFiles;
