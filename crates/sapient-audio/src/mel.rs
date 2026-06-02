//! Log-mel spectrogram front-end (Whisper-compatible).
//!
//! Mirrors OpenAI Whisper's `log_mel_spectrogram` exactly:
//! - `torch.stft(audio, n_fft=400, hop_length=160, window=hann_window(400),
//!   center=True)` — reflect-pad by `n_fft/2`, drop the final frame so a 30 s
//!   chunk yields 3000 frames;
//! - power spectrum `|stft|^2`;
//! - mel filterbank (`librosa` slaney scale + slaney norm) projects 201 FFT bins
//!   to `n_mels`;
//! - `log10`, clamp to `max - 8`, then `(x + 4) / 4`.
//!
//! The filterbank is built analytically (no embedded data file) and is unit
//! tested for the librosa-known anchor points (`hz_to_mel(1000) == 15`).

use std::sync::Arc;

use anyhow::{Context, Result};
use realfft::{RealFftPlanner, RealToComplex};
use sapient_core::Tensor;

use crate::config::MelConfig;

/// Reusable log-mel front-end: holds the Hann window, mel filterbank, and FFT
/// plan so repeated `log_mel` calls allocate only per-chunk scratch.
pub struct MelFrontend {
    cfg: MelConfig,
    window: Vec<f32>,                 // Hann, length n_fft
    filters: Vec<f32>,                // [n_mels, n_freqs] row-major
    fft: Arc<dyn RealToComplex<f32>>, // real → complex, size n_fft
}

impl MelFrontend {
    /// Build a front-end for the given configuration.
    pub fn new(cfg: MelConfig) -> Self {
        let window = hann_window(cfg.n_fft);
        let filters = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        Self {
            cfg,
            window,
            filters,
            fft,
        }
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    pub fn n_mels(&self) -> usize {
        self.cfg.n_mels
    }

    /// Compute the Whisper log-mel spectrogram for one chunk of mono 16 kHz
    /// audio. `audio` is padded with zeros (or trimmed) to exactly
    /// `cfg.n_samples()` before transform. Output shape `[1, n_mels, n_frames]`.
    pub fn log_mel(&self, audio: &[f32]) -> Result<Tensor> {
        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.hop_length;
        let n_freqs = self.cfg.n_freqs();
        let n_frames = self.cfg.n_frames();
        let n_mels = self.cfg.n_mels;
        let pad = n_fft / 2;

        // Pad/trim to a fixed 30 s window, then reflect-pad n_fft/2 each side
        // (torch.stft center=True).
        let mut padded = vec![0.0f32; self.cfg.n_samples() + 2 * pad];
        let n = audio.len().min(self.cfg.n_samples());
        // Copy the (trimmed) signal into the centre.
        padded[pad..pad + n].copy_from_slice(&audio[..n]);
        reflect_pad(&mut padded, pad, self.cfg.n_samples());

        // STFT → power spectrum, immediately projected through the mel filters.
        let mut frame = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut power = vec![0.0f32; n_freqs];
        let mut mel = vec![0.0f32; n_mels * n_frames];

        for t in 0..n_frames {
            let start = t * hop;
            for i in 0..n_fft {
                frame[i] = padded[start + i] * self.window[i];
            }
            self.fft
                .process(&mut frame, &mut spectrum)
                .map_err(|e| anyhow::anyhow!("FFT failed: {e}"))
                .context("stft frame")?;
            for (f, c) in spectrum.iter().enumerate() {
                power[f] = c.norm_sqr();
            }
            // mel[:, t] = filters @ power
            for m in 0..n_mels {
                let row = &self.filters[m * n_freqs..(m + 1) * n_freqs];
                let mut acc = 0.0f32;
                for f in 0..n_freqs {
                    acc += row[f] * power[f];
                }
                mel[m * n_frames + t] = acc;
            }
        }

        // Whisper log compression + normalization.
        let mut max_log = f32::NEG_INFINITY;
        for v in mel.iter_mut() {
            let l = v.max(1e-10).log10();
            *v = l;
            if l > max_log {
                max_log = l;
            }
        }
        let floor = max_log - 8.0;
        for v in mel.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }

        Tensor::from_f32_vec(mel, vec![1, n_mels, n_frames]).context("building mel tensor")
    }
}

/// Periodic Hann window (matches `torch.hann_window(n, periodic=True)`).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::TAU * i as f32 / n as f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

/// Reflect-pad `buf` in place: the centre `[pad, pad+len)` already holds the
/// signal; fill the `pad` samples on each side by reflecting (excluding the edge
/// sample), matching numpy/torch `mode='reflect'`.
fn reflect_pad(buf: &mut [f32], pad: usize, len: usize) {
    if len <= 1 {
        return;
    }
    for i in 0..pad {
        // Left: buf[pad-1-i] = signal[i+1]
        buf[pad - 1 - i] = buf[pad + i + 1];
        // Right: mirror around the last sample.
        buf[pad + len + i] = buf[pad + len - 2 - i];
    }
}

/// librosa slaney mel scale (`htk=False`).
fn hz_to_mel(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = min_log_hz / f_sp; // 15.0
    let logstep = (6.4f32).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = min_log_hz / f_sp; // 15.0
    let logstep = (6.4f32).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * ((m - min_log_mel) * logstep).exp()
    } else {
        f_sp * m
    }
}

/// Build the `[n_mels, n_fft/2+1]` mel filterbank, matching
/// `librosa.filters.mel(sr, n_fft, n_mels, htk=False, norm='slaney')` — the
/// matrix OpenAI Whisper ships in `mel_filters.npz`.
fn mel_filterbank(sr: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let f_max = sr as f32 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 band edges, evenly spaced in mel, mapped back to Hz.
    let mut edges = vec![0.0f32; n_mels + 2];
    for (i, e) in edges.iter_mut().enumerate() {
        let m = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        *e = mel_to_hz(m);
    }

    // rfft bin centre frequencies.
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let lower = edges[m];
        let center = edges[m + 1];
        let upper = edges[m + 2];
        let enorm = 2.0 / (upper - lower); // slaney normalization
        for (k, &f) in fft_freqs.iter().enumerate() {
            let down = (f - lower) / (center - lower);
            let up = (upper - f) / (upper - center);
            let w = down.min(up).max(0.0);
            filters[m * n_freqs + k] = w * enorm;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mel_scale_anchor_points() {
        // librosa-known: the linear/log break is at 1000 Hz ↔ 15 mel.
        assert_relative_eq!(hz_to_mel(1000.0), 15.0, epsilon = 1e-4);
        assert_relative_eq!(mel_to_hz(15.0), 1000.0, epsilon = 1e-2);
        // Round-trip.
        for &hz in &[80.0f32, 440.0, 1000.0, 4000.0, 8000.0] {
            assert_relative_eq!(mel_to_hz(hz_to_mel(hz)), hz, epsilon = 1e-1);
        }
    }

    #[test]
    fn filterbank_shape_and_nonneg() {
        let f = mel_filterbank(16_000, 400, 80);
        assert_eq!(f.len(), 80 * 201);
        assert!(f.iter().all(|&w| w >= 0.0));
        // Every mel channel has at least one non-zero weight.
        for m in 0..80 {
            let row = &f[m * 201..(m + 1) * 201];
            assert!(row.iter().any(|&w| w > 0.0), "empty mel channel {m}");
        }
    }

    #[test]
    fn log_mel_shape_and_range() {
        let cfg = MelConfig::whisper();
        let fe = MelFrontend::new(cfg);
        // 1 s of a 440 Hz tone at 16 kHz.
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let mel = fe.log_mel(&tone).unwrap();
        assert_eq!(mel.shape().dims(), &[1, 80, 3000]);
        // Whisper clamps log_spec to `max_log - 8` then scales by /4, so the
        // dynamic range is bounded to exactly 8 dB → 2.0 (it is NOT bounded to
        // [.., 1.0]; the absolute level depends on signal energy).
        let v = mel.as_f32_slice();
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max > min);
        assert_relative_eq!(max - min, 2.0, epsilon = 1e-3);
    }
}
