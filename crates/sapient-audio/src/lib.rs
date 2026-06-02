//! SAPIENT audio front-end.
//!
//! Pure-Rust (no FFI) building blocks for on-device speech models:
//!
//! - [`io`] — decode any common container/codec to mono `f32` and resample to a
//!   target rate (16 kHz for Whisper), via `symphonia` + `rubato`.
//! - [`mel`] — Short-Time Fourier Transform + mel filterbank → Whisper-style
//!   log-mel spectrogram, via `realfft`. Numerically aligned with OpenAI Whisper
//!   / `librosa` (slaney mel scale, slaney normalization).
//! - [`config`] — [`MelConfig`], the front-end parameters (n_fft, hop, n_mels, …).
//! - [`vad`] — energy-based voice activity detection / utterance segmentation
//!   (pure Rust, no device deps) for the speech-to-speech cascade.
//!
//! The front-end is deliberately CPU-only: STFT/mel run once per 30 s audio
//! chunk and are sub-100 ms on one core, so there is no reason to push them onto
//! the GPU. The transformer body that consumes the mel tensor is a separate
//! concern (see `sapient-models`).

pub mod config;
pub mod io;
pub mod mel;
pub mod vad;

pub use config::MelConfig;
pub use io::load_audio;
pub use mel::MelFrontend;
pub use vad::{EnergyVad, Vad, VadConfig};
