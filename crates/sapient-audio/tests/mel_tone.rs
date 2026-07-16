// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! End-to-end DSP sanity: a pure tone must concentrate its energy in the mel
//! channel(s) whose triangular filter covers that frequency. This validates the
//! STFT framing, windowing, power spectrum, and filterbank projection together
//! without needing an external (Python/librosa) reference matrix — the exact
//! numerical match against HF is covered by the ignored end-to-end transcription
//! test in `sapient-generate`.

use sapient_audio::{MelConfig, MelFrontend};

/// Mel channel index whose centre is nearest `hz`, computed independently from
/// the production filterbank via the slaney mel scale.
fn nearest_channel(hz: f32, n_mels: usize) -> usize {
    // slaney hz→mel (htk=false)
    let to_mel = |f: f32| {
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0f32;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let mel_max = to_mel(8000.0);
    let target = to_mel(hz);
    // Channel m centre sits at edge m+1 → mel = mel_max*(m+1)/(n_mels+1).
    let frac = target / mel_max * (n_mels + 1) as f32;
    (frac.round() as usize).saturating_sub(1).min(n_mels - 1)
}

fn tone(freq: f32, secs: usize, sr: u32) -> Vec<f32> {
    let n = secs * sr as usize;
    (0..n)
        .map(|i| (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin() * 0.5)
        .collect()
}

#[test]
fn tone_energy_lands_in_expected_mel_channel() {
    let cfg = MelConfig::whisper();
    let fe = MelFrontend::new(cfg);

    for &freq in &[300.0f32, 1000.0, 3000.0] {
        // Fill the whole 30 s window so the steady-state frame range (below)
        // actually contains the tone rather than zero-padding.
        let audio = tone(freq, cfg.chunk_length, cfg.sample_rate);
        let mel = fe.log_mel(&audio).unwrap();
        let data = mel.as_f32_slice(); // [1, 80, 3000]
        let n_frames = cfg.n_frames();

        // Average each mel channel over a steady-state interior frame range.
        let mut energy = vec![0.0f32; cfg.n_mels];
        let (lo, hi) = (n_frames / 4, n_frames * 3 / 4);
        for (m, e) in energy.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for t in lo..hi {
                acc += data[m * n_frames + t];
            }
            *e = acc / (hi - lo) as f32;
        }

        let argmax = energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let expected = nearest_channel(freq, cfg.n_mels);

        // Allow ±2 channels (FFT bin granularity + triangular overlap).
        let diff = argmax.abs_diff(expected);
        assert!(
            diff <= 2,
            "tone {freq} Hz: peak mel channel {argmax}, expected ~{expected} (diff {diff})"
        );
    }
}
