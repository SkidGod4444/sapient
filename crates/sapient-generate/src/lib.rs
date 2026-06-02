#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    clippy::derivable_impls
)]

//! `sapient-generate` — LLM text generation pipeline.
//!
//! The main entry point is [`Pipeline`], which provides a dead-simple API
//! for running any HuggingFace LLM:
//!
//! ```no_run
//! use sapient_generate::Pipeline;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let pipeline = Pipeline::from_pretrained("microsoft/phi-2").await?;
//!
//!     // Simple completion
//!     let text = pipeline.generate("The meaning of life is").await?;
//!     println!("{text}");
//!
//!     // Chat (for instruct models)
//!     use sapient_tokenizers::ChatMessage;
//!     let reply = pipeline.chat(&[
//!         ChatMessage::system("You are a helpful assistant."),
//!         ChatMessage::user("Explain quantum computing in simple terms."),
//!     ]).await?;
//!     println!("{reply}");
//!
//!     // Streaming
//!     use futures::StreamExt;
//!     let mut stream = pipeline.generate_stream("Once upon a time").await;
//!     while let Some(token) = stream.next().await {
//!         print!("{token}");
//!     }
//!     Ok(())
//! }
//! ```

pub mod converse;
pub mod device;
pub mod kv_cache;
pub mod pipeline;
pub mod sampler;
pub mod sentence;
pub mod speak;
pub mod speculative;
pub mod transcribe;

pub use converse::{ConversePipeline, NoopTts, Tts, Turn};
pub use device::{
    detect as detect_devices, recommend as recommend_backend, BackendPlan, DeviceProfile,
};
pub use kv_cache::KVCache;
pub use pipeline::{GenerationConfig, LoadOptions, Pipeline};
pub use sampler::{Sampler, SamplingStrategy};
#[cfg(feature = "audio-io")]
pub use sapient_audio::{
    microphone_guidance, open_privacy_settings, request_microphone, MicCapture, MicPermission,
    SpeakerPlayback,
};
pub use sapient_audio::{EnergyVad, VadConfig};
pub use sapient_models::{mac_gpu_support, LlmBackendKind as GenerationBackend, MacGpuSupport};
pub use sentence::SentenceChunker;
pub use speak::{SpeakPipeline, DEFAULT_ORPHEUS_VOICE, ORPHEUS_VOICES};
pub use speculative::SpeculativePipeline;
pub use transcribe::{TranscribeOptions, TranscribePipeline};
