//! Voice activity detection + utterance segmentation (pure Rust, no device deps).
//!
//! [`EnergyVad`] is a streaming, frame-based segmenter: push fixed-size frames of
//! mono 16 kHz `f32` audio and it returns a finalized utterance (the buffered
//! speech samples) once it sees `silence_hang` of trailing silence. It uses
//! short-time energy (RMS) against an adaptive noise floor plus a zero-crossing
//! sanity bound, with debounce (`enter_frames`) and hangover. No model, no
//! allocation on the hot path beyond the utterance buffer — so it is fully
//! unit-testable without a microphone (feed it a WAV or a synthetic tone burst).
//!
//! A learned VAD (e.g. a WebRTC-GMM port) can later implement the same
//! [`Vad`] trait; `EnergyVad` is the dependency-free default.

/// Per-frame activity decision an implementation exposes (used by tests / future
/// backends). The streaming segmentation in [`EnergyVad::push`] is built on this.
pub trait Vad {
    /// Classify one frame as speech (`true`) or non-speech (`false`).
    fn is_speech(&mut self, frame: &[f32]) -> bool;
}

/// Configuration for [`EnergyVad`]. Frame/threshold timing is in 16 kHz frames.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Samples per frame (320 = 20 ms @ 16 kHz).
    pub frame_samples: usize,
    /// Consecutive speech frames required to *enter* the speech state (debounce).
    pub enter_frames: usize,
    /// Consecutive silence frames that *finalize* an utterance (hangover).
    pub silence_hang_frames: usize,
    /// Energy threshold = `noise_floor * (1 + sensitivity * SCALE)`. Higher
    /// `sensitivity` ⇒ requires louder speech (fewer false triggers). 0..1.
    pub sensitivity: f32,
    /// Drop utterances shorter than this many frames (coughs/clicks).
    pub min_utterance_frames: usize,
    /// Force-finalize an utterance once it reaches this many frames.
    pub max_utterance_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        // 20 ms frames; ~120 ms to engage, ~700 ms silence to end a turn.
        Self {
            frame_samples: 320,
            enter_frames: 6,
            silence_hang_frames: 35,
            sensitivity: 0.5,
            min_utterance_frames: 10,   // 200 ms
            max_utterance_frames: 1500, // 30 s
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silence,
    Speech,
}

/// Streaming energy-based utterance segmenter.
pub struct EnergyVad {
    cfg: VadConfig,
    state: State,
    noise_floor: f32,
    /// Trailing speech-frame run length while in `Silence` (debounce counter).
    run_speech: usize,
    /// Trailing silence-frame run length while in `Speech` (hangover counter).
    run_silence: usize,
    /// Buffered samples for the in-progress utterance (incl. the debounce lead).
    buffer: Vec<f32>,
    /// Recent frames held back during debounce so the utterance keeps its onset.
    lead: Vec<f32>,
    frames_in_utterance: usize,
}

impl EnergyVad {
    pub fn new(cfg: VadConfig) -> Self {
        Self {
            cfg,
            state: State::Silence,
            noise_floor: 1e-4,
            run_speech: 0,
            run_silence: 0,
            buffer: Vec::new(),
            lead: Vec::new(),
            frames_in_utterance: 0,
        }
    }

    fn threshold(&self) -> f32 {
        // Map sensitivity 0..1 → multiplier ~2x..7x over the adaptive noise floor.
        let mult = 1.0 + self.cfg.sensitivity * 6.0 + 1.0;
        (self.noise_floor * mult).max(1e-4)
    }

    /// Push one frame (`cfg.frame_samples` samples). Returns the finalized
    /// utterance samples once a full speech→silence turn completes.
    pub fn push(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let rms = rms(frame);
        let zcr = zero_crossing_rate(frame);
        // Speech = loud enough AND not pure tonal/DC noise (ZCR in a voice band).
        let speech = rms > self.threshold() && (0.01..0.8).contains(&zcr);

        match self.state {
            State::Silence => {
                // Adapt the noise floor only on genuine non-speech frames.
                if !speech {
                    self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms;
                    self.run_speech = 0;
                    self.lead.clear();
                    return None;
                }
                // Candidate speech: hold frames as lead until debounce passes.
                self.run_speech += 1;
                self.lead.extend_from_slice(frame);
                if self.run_speech >= self.cfg.enter_frames {
                    self.state = State::Speech;
                    self.buffer.clear();
                    self.buffer.append(&mut self.lead);
                    self.frames_in_utterance = self.run_speech;
                    self.run_silence = 0;
                }
                None
            }
            State::Speech => {
                self.buffer.extend_from_slice(frame);
                self.frames_in_utterance += 1;
                if speech {
                    self.run_silence = 0;
                } else {
                    self.run_silence += 1;
                }
                let ended = self.run_silence >= self.cfg.silence_hang_frames;
                let too_long = self.frames_in_utterance >= self.cfg.max_utterance_frames;
                if ended || too_long {
                    return self.finalize();
                }
                None
            }
        }
    }

    /// Finalize any in-progress utterance (call at end of stream).
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.state == State::Speech {
            self.finalize()
        } else {
            None
        }
    }

    fn finalize(&mut self) -> Option<Vec<f32>> {
        self.state = State::Silence;
        self.run_speech = 0;
        self.run_silence = 0;
        self.lead.clear();
        let frames = self.frames_in_utterance;
        self.frames_in_utterance = 0;
        let utterance = std::mem::take(&mut self.buffer);
        if frames >= self.cfg.min_utterance_frames {
            Some(utterance)
        } else {
            None // too short — discard
        }
    }
}

impl Vad for EnergyVad {
    fn is_speech(&mut self, frame: &[f32]) -> bool {
        rms(frame) > self.threshold()
    }
}

fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.iter().map(|&v| v * v).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

fn zero_crossing_rate(frame: &[f32]) -> f32 {
    if frame.len() < 2 {
        return 0.0;
    }
    let crossings = frame
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (frame.len() - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(samples: &[f32], n: usize) -> Vec<Vec<f32>> {
        samples.chunks(n).map(|c| c.to_vec()).collect()
    }

    /// A tone burst surrounded by silence segments into exactly one utterance.
    #[test]
    fn segments_one_utterance() {
        let cfg = VadConfig {
            silence_hang_frames: 10,
            min_utterance_frames: 5,
            ..VadConfig::default()
        };
        let mut vad = EnergyVad::new(cfg);
        let fs = cfg.frame_samples;

        // 15 frames silence, 30 frames 300 Hz tone, 20 frames silence.
        let mut sig = vec![0.0f32; 15 * fs];
        let start = sig.len();
        for i in 0..30 * fs {
            sig.push((std::f32::consts::TAU * 300.0 * (start + i) as f32 / 16000.0).sin() * 0.4);
        }
        sig.resize(sig.len() + 20 * fs, 0.0);

        let mut utterances = Vec::new();
        for f in frames(&sig, fs) {
            if let Some(u) = vad.push(&f) {
                utterances.push(u);
            }
        }
        if let Some(u) = vad.flush() {
            utterances.push(u);
        }

        assert_eq!(utterances.len(), 1, "expected exactly one utterance");
        // Roughly the tone duration (± debounce/hangover); generous bounds.
        let secs = utterances[0].len() as f32 / 16000.0;
        assert!((0.4..1.2).contains(&secs), "utterance {secs}s out of range");
    }

    #[test]
    fn pure_silence_yields_nothing() {
        let mut vad = EnergyVad::new(VadConfig::default());
        let fs = VadConfig::default().frame_samples;
        for _ in 0..100 {
            assert!(vad.push(&vec![0.0f32; fs]).is_none());
        }
        assert!(vad.flush().is_none());
    }

    #[test]
    fn rms_and_zcr_basic() {
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        assert!((zero_crossing_rate(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        assert_eq!(zero_crossing_rate(&[1.0, 1.0, 1.0]), 0.0);
    }
}
