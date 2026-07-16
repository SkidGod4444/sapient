// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Mel front-end configuration.

/// Parameters for the log-mel spectrogram front-end.
///
/// Defaults match OpenAI Whisper: 16 kHz mono, `n_fft = 400` (25 ms),
/// `hop_length = 160` (10 ms), 30 s chunks → 3000 frames. The only knob that
/// differs across Whisper checkpoints is `n_mels`: 80 for tiny…large-v2,
/// 128 for large-v3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MelConfig {
    /// Target sample rate the audio is resampled to before STFT.
    pub sample_rate: u32,
    /// FFT window size in samples.
    pub n_fft: usize,
    /// Hop between successive frames in samples.
    pub hop_length: usize,
    /// Number of mel filterbank channels (80 or 128).
    pub n_mels: usize,
    /// Chunk length in seconds (Whisper processes fixed 30 s windows).
    pub chunk_length: usize,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self::whisper()
    }
}

impl MelConfig {
    /// 80-mel Whisper front-end (tiny … large-v2, distil-whisper).
    pub const fn whisper() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            chunk_length: 30,
        }
    }

    /// 128-mel Whisper front-end (large-v3 / large-v3-turbo).
    pub const fn whisper_v3() -> Self {
        Self {
            n_mels: 128,
            ..Self::whisper()
        }
    }

    /// Whisper front-end with an explicit mel-channel count.
    pub const fn with_n_mels(n_mels: usize) -> Self {
        Self {
            n_mels,
            ..Self::whisper()
        }
    }

    /// Samples in one chunk (`sample_rate * chunk_length`), e.g. 480 000.
    pub const fn n_samples(&self) -> usize {
        self.sample_rate as usize * self.chunk_length
    }

    /// Frames produced per chunk (`n_samples / hop_length`), e.g. 3000.
    pub const fn n_frames(&self) -> usize {
        self.n_samples() / self.hop_length
    }

    /// Number of non-redundant FFT bins (`n_fft / 2 + 1`), e.g. 201.
    pub const fn n_freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }
}
