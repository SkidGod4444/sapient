// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Audio file decoding and resampling.
//!
//! [`load_audio`] decodes any container/codec `symphonia` understands (WAV,
//! FLAC, OGG/Vorbis, MP3, AAC/M4A, ALAC), downmixes to mono `f32` in `[-1, 1]`,
//! and resamples to `target_sr` (16 kHz for Whisper) with `rubato`'s FFT
//! resampler.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decode `path` to mono `f32` samples resampled to `target_sr`.
pub fn load_audio(path: impl AsRef<Path>, target_sr: u32) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let (samples, src_sr) = decode_to_mono(path)?;
    if src_sr == target_sr {
        Ok(samples)
    } else {
        resample(&samples, src_sr, target_sr)
    }
}

/// Decode `path` to mono `f32` at its native sample rate.
///
/// Returns `(samples, source_sample_rate)`. Multi-channel audio is averaged to
/// mono.
pub fn decode_to_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening audio file {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probing audio format (unsupported or corrupt file?)")?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no decodable audio track found"))?;
    let track_id = track.id;
    let src_sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("audio track has no sample rate"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("no decoder for this codec")?;

    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // Clean end-of-stream: symphonia surfaces EOF as an IoError.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(anyhow!(e).context("reading audio packet")),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Recoverable decode hiccups: skip the packet.
            Err(SymphoniaError::DecodeError(_)) | Err(SymphoniaError::IoError(_)) => continue,
            Err(e) => return Err(anyhow!(e).context("decoding audio packet")),
        };

        let spec = *decoded.spec();
        let n_channels = spec.channels.count().max(1);

        // (Re)allocate the interleaving buffer to match this frame's capacity.
        if sample_buf.is_none() {
            sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
        }
        let sb = sample_buf.as_mut().unwrap();
        sb.copy_interleaved_ref(decoded);
        let interleaved = sb.samples();

        // Downmix interleaved → mono by channel average.
        if n_channels == 1 {
            mono.extend_from_slice(interleaved);
        } else {
            mono.reserve(interleaved.len() / n_channels);
            for frame in interleaved.chunks_exact(n_channels) {
                let sum: f32 = frame.iter().sum();
                mono.push(sum / n_channels as f32);
            }
        }
    }

    if mono.is_empty() {
        return Err(anyhow!(
            "decoded zero audio samples from {}",
            path.display()
        ));
    }
    Ok((mono, src_sr))
}

/// Resample mono `f32` audio from `from` Hz to `to` Hz with a high-quality FFT
/// resampler. A no-op when the rates already match.
pub fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to || input.is_empty() {
        return Ok(input.to_vec());
    }

    const CHUNK: usize = 1024;
    let mut resampler = FftFixedIn::<f32>::new(from as usize, to as usize, CHUNK, 2, 1)
        .context("constructing resampler")?;

    let est = input.len() * to as usize / from as usize + CHUNK;
    let mut out: Vec<f32> = Vec::with_capacity(est);
    let mut inbuf = [vec![0.0f32; CHUNK]];

    let mut pos = 0;
    while pos + CHUNK <= input.len() {
        inbuf[0].copy_from_slice(&input[pos..pos + CHUNK]);
        let res = resampler
            .process(&inbuf, None)
            .context("resampling chunk")?;
        out.extend_from_slice(&res[0]);
        pos += CHUNK;
    }

    // Final partial chunk: zero-pad to CHUNK, keep only the proportional output.
    if pos < input.len() {
        let rem = input.len() - pos;
        for (i, slot) in inbuf[0].iter_mut().enumerate() {
            *slot = if i < rem { input[pos + i] } else { 0.0 };
        }
        let res = resampler
            .process(&inbuf, None)
            .context("resampling final chunk")?;
        let keep = (rem * to as usize / from as usize).min(res[0].len());
        out.extend_from_slice(&res[0][..keep]);
    }

    Ok(out)
}

/// Encode mono `f32` samples (in `[-1, 1]`) as a 16-bit PCM WAV byte buffer at
/// `sample_rate`. Pure Rust — emits the 44-byte canonical header + i16 samples;
/// no `hound`/codec dependency needed for output.
///
/// Separate from [`write_wav`] so TTS can be served straight over HTTP
/// (`POST /v1/audio/speech`) without staging a temp file on disk.
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let n = samples.len();
    let data_bytes = (n * 2) as u32; // 16-bit mono
    let byte_rate = sample_rate * 2;
    let mut buf = Vec::with_capacity(44 + n * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Write mono `f32` samples to a 16-bit PCM WAV file. Used by `sapient speak`.
pub fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    std::fs::write(path.as_ref(), encode_wav(samples, sample_rate))
        .with_context(|| format!("writing WAV {}", path.as_ref().display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_is_noop_when_rates_match() {
        let x: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(resample(&x, 16_000, 16_000).unwrap(), x);
    }

    #[test]
    fn resample_48k_to_16k_thirds_the_length() {
        // 1 s of 440 Hz at 48 kHz → ~1 s at 16 kHz (length within a chunk).
        let n = 48_000;
        let tone: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let out = resample(&tone, 48_000, 16_000).unwrap();
        let expected = n / 3;
        let diff = (out.len() as i64 - expected as i64).unsigned_abs() as usize;
        assert!(
            diff <= 1024,
            "resampled len {} not near {expected}",
            out.len()
        );
        // Output must be finite and bounded (no resampler blow-up).
        assert!(out.iter().all(|v| v.is_finite() && v.abs() < 4.0));
    }

    #[test]
    fn write_wav_roundtrips_through_decode() {
        let sr = 24_000u32;
        let samples: Vec<f32> = (0..2400)
            .map(|i| (std::f32::consts::TAU * 220.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let tmp = std::env::temp_dir().join("sapient_write_wav_test.wav");
        write_wav(&tmp, &samples, sr).unwrap();
        let (back, rate) = decode_to_mono(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(rate, sr);
        assert_eq!(back.len(), samples.len());
        // 16-bit quantization error ≤ ~1/32767.
        let max_err = samples
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "wav roundtrip max_err={max_err}");
    }
}
