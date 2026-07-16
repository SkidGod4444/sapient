// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

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

#[cfg(feature = "audio-io")]
pub mod capture;
pub mod config;
pub mod io;
pub mod mel;
#[cfg(feature = "audio-io")]
pub mod permissions;
#[cfg(feature = "audio-io")]
pub mod playback;
pub mod vad;

#[cfg(feature = "audio-io")]
pub use capture::MicCapture;
pub use config::MelConfig;
pub use io::{encode_wav, load_audio, write_wav};
pub use mel::MelFrontend;
#[cfg(feature = "audio-io")]
pub use permissions::{
    microphone_guidance, open_privacy_settings, request_microphone, MicPermission,
};
#[cfg(feature = "audio-io")]
pub use playback::SpeakerPlayback;
pub use vad::{EnergyVad, Vad, VadConfig};
