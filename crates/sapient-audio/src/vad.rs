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
    /// Discard finalized utterances whose MEAN frame RMS is below this
    /// (0.0 = disabled). Guards against low-energy noise/echo tails that pass
    /// the per-frame threshold long enough to finalize — Whisper hallucinates
    /// famously on such non-speech ("MBC 뉴스…" class artifacts).
    pub min_mean_rms: f32,
    /// Force-finalize an utterance once it reaches this many frames.
    pub max_utterance_frames: usize,
    /// Absolute minimum RMS for a frame to count as speech. Floors the adaptive
    /// threshold so typical room ambient (~0.003–0.01) is **not** mistaken for
    /// speech — the bug that made every turn run to `max_utterance_frames`.
    pub min_rms: f32,
    /// In the speech state, a frame ends the turn when its RMS drops below
    /// `end_ratio × (recent speech peak)` — a relative-drop detector that
    /// finalizes on a real pause even when ambient noise exceeds `min_rms`.
    pub end_ratio: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        // 20 ms frames; ~120 ms to engage, ~600 ms silence to end a turn.
        Self {
            frame_samples: 320,
            enter_frames: 6,
            silence_hang_frames: 30,
            sensitivity: 0.5,
            min_utterance_frames: 10,   // 200 ms
            max_utterance_frames: 1500, // 30 s
            min_rms: 0.01,
            end_ratio: 0.35,
            min_mean_rms: 0.0,
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
    /// Σ frame RMS across the utterance (for the mean-energy gate).
    rms_sum: f32,
    /// Decaying peak RMS of the in-progress utterance (relative-drop reference).
    peak: f32,
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
            rms_sum: 0.0,
            peak: 0.0,
        }
    }

    fn threshold(&self) -> f32 {
        // Map sensitivity 0..1 → multiplier ~2x..7x over the adaptive noise floor,
        // floored at `min_rms` so room ambient never reads as speech.
        let mult = 1.0 + self.cfg.sensitivity * 6.0 + 1.0;
        (self.noise_floor * mult).max(self.cfg.min_rms)
    }

    /// True while inside an utterance (the speech state) — the live loop uses
    /// this to drive incremental STT on the in-progress utterance.
    pub fn in_speech(&self) -> bool {
        matches!(self.state, State::Speech)
    }

    /// Snapshot of the in-progress utterance samples (empty outside speech).
    /// Grows as frames are pushed; a streaming transcriber can re-transcribe
    /// this while the speaker is still talking so the final transcript is
    /// ready the moment the utterance ends.
    pub fn speech_so_far(&self) -> &[f32] {
        if self.in_speech() {
            &self.buffer
        } else {
            &[]
        }
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
                    self.rms_sum = rms * self.run_speech as f32;
                    self.run_silence = 0;
                    self.peak = rms; // seed the relative-drop reference
                }
                None
            }
            State::Speech => {
                self.buffer.extend_from_slice(frame);
                self.frames_in_utterance += 1;
                self.rms_sum += rms;
                // Track a slowly-decaying speech peak. A frame keeps the turn
                // alive only if it's loud both in absolute terms (`threshold`)
                // and relative to that peak — so a real pause (which drops well
                // below the speaker's level) ends the turn even when ambient
                // noise sits above the absolute floor.
                self.peak = rms.max(self.peak * 0.995);
                let end_level = (self.cfg.end_ratio * self.peak).max(self.threshold());
                if rms >= end_level && (0.01..0.8).contains(&zcr) {
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
        self.peak = 0.0;
        self.lead.clear();
        let frames = self.frames_in_utterance;
        self.frames_in_utterance = 0;
        let mean_rms = if frames > 0 {
            self.rms_sum / frames as f32
        } else {
            0.0
        };
        self.rms_sum = 0.0;
        let utterance = std::mem::take(&mut self.buffer);
        let debug = std::env::var("SAPIENT_VAD_DEBUG").is_ok();
        if frames < self.cfg.min_utterance_frames {
            if debug {
                eprintln!(
                    "[vad] DISCARD short: {frames} frames (min {})",
                    self.cfg.min_utterance_frames
                );
            }
            return None; // too short — discard
        }
        if self.cfg.min_mean_rms > 0.0 && mean_rms < self.cfg.min_mean_rms {
            if debug {
                eprintln!(
                    "[vad] DISCARD quiet: mean_rms {mean_rms:.4} < {} over {frames} frames",
                    self.cfg.min_mean_rms
                );
            }
            return None; // low-energy noise/echo tail — Whisper would hallucinate
        }
        if debug {
            eprintln!("[vad] ACCEPT: {frames} frames, mean_rms {mean_rms:.4}");
        }
        Some(utterance)
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

    /// The mean-energy gate discards quiet noise that squeaks past the
    /// per-frame threshold (the Whisper-hallucination class), while real
    /// speech passes.
    #[test]
    fn mean_rms_gate_discards_quiet_noise() {
        let cfg = VadConfig {
            silence_hang_frames: 6,
            min_utterance_frames: 4,
            min_mean_rms: 0.05,
            ..VadConfig::default()
        };
        let fs = cfg.frame_samples;
        // Quiet tone just above the entry threshold (~0.02 rms) → finalizes by
        // duration but fails the mean gate.
        let quiet: Vec<f32> = (0..20 * fs)
            .map(|i| 0.028 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mut vad = EnergyVad::new(cfg);
        let mut got = None;
        for f in quiet.chunks(fs) {
            if let Some(u) = vad.push(f) {
                got = Some(u);
            }
        }
        for _ in 0..10 {
            if let Some(u) = vad.push(&vec![0.0; fs]) {
                got = Some(u);
            }
        }
        assert!(got.is_none(), "quiet noise should be gated out");

        // Loud speech-level tone passes.
        let loud: Vec<f32> = (0..20 * fs)
            .map(|i| 0.25 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mut vad = EnergyVad::new(VadConfig {
            silence_hang_frames: 6,
            min_utterance_frames: 4,
            min_mean_rms: 0.05,
            ..VadConfig::default()
        });
        let mut got = None;
        for f in loud.chunks(fs) {
            if let Some(u) = vad.push(f) {
                got = Some(u);
            }
        }
        for _ in 0..10 {
            if let Some(u) = vad.push(&vec![0.0; fs]) {
                got = Some(u);
            }
        }
        assert!(got.is_some(), "real speech must pass the gate");
    }

    /// The live-STT taps: `in_speech`/`speech_so_far` expose the in-progress
    /// utterance while speaking and go quiet after finalization.
    #[test]
    fn live_taps_track_in_progress_utterance() {
        let cfg = VadConfig {
            silence_hang_frames: 10,
            min_utterance_frames: 5,
            ..VadConfig::default()
        };
        let mut vad = EnergyVad::new(cfg);
        let fs = cfg.frame_samples;
        assert!(!vad.in_speech());
        assert!(vad.speech_so_far().is_empty());

        // 300 Hz tone frames = speech.
        let tone: Vec<f32> = (0..30 * fs)
            .map(|i| 0.25 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mut grew = 0usize;
        for f in tone.chunks(fs) {
            assert!(vad.push(f).is_none());
            if vad.in_speech() {
                let now = vad.speech_so_far().len();
                assert!(now >= grew, "snapshot must grow monotonically");
                grew = now;
            }
        }
        assert!(vad.in_speech());
        assert!(grew >= 20 * fs, "most of the tone should be buffered");

        // Silence until finalization.
        let silence = vec![0.0f32; fs];
        let mut fin = None;
        for _ in 0..20 {
            if let Some(u) = vad.push(&silence) {
                fin = Some(u);
                break;
            }
        }
        assert!(fin.is_some(), "utterance should finalize");
        assert!(!vad.in_speech());
        assert!(vad.speech_so_far().is_empty());
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

    // 300 Hz tone of a given RMS — `amp = rms·√2`.
    fn tone(idx: usize, fs: usize, rms_target: f32) -> Vec<f32> {
        let amp = rms_target * std::f32::consts::SQRT_2;
        (0..fs)
            .map(|k| (std::f32::consts::TAU * 300.0 * (idx * fs + k) as f32 / 16_000.0).sin() * amp)
            .collect()
    }

    #[test]
    fn ambient_below_min_rms_never_triggers() {
        // Steady room hum at rms ~0.004 (< min_rms 0.01) must never be treated as
        // speech — this is the regression that made every turn run to the 30 s cap.
        let cfg = VadConfig::default();
        let mut vad = EnergyVad::new(cfg);
        for i in 0..120 {
            assert!(vad.push(&tone(i, cfg.frame_samples, 0.004)).is_none());
        }
        assert!(vad.flush().is_none());
    }

    #[test]
    fn finalizes_promptly_on_real_pause() {
        // Loud speech, then a realistic ambient pause (rms ~0.005, below min_rms):
        // the turn must finalize at the silence-hang, NOT run to the 30 s cap, and
        // the trailing ambient must not re-trigger a second phantom utterance.
        let cfg = VadConfig {
            silence_hang_frames: 10,
            min_utterance_frames: 5,
            ..VadConfig::default()
        };
        let mut vad = EnergyVad::new(cfg);
        let fs = cfg.frame_samples;
        let mut utts = Vec::new();
        let mut idx = 0;
        for _ in 0..25 {
            if let Some(u) = vad.push(&tone(idx, fs, 0.2)) {
                utts.push(u);
            }
            idx += 1;
        }
        for _ in 0..40 {
            if let Some(u) = vad.push(&tone(idx, fs, 0.005)) {
                utts.push(u);
            }
            idx += 1;
        }
        if let Some(u) = vad.flush() {
            utts.push(u);
        }
        assert_eq!(
            utts.len(),
            1,
            "exactly one utterance, finalized on the pause"
        );
        // Finalized well before the max cap (25 speech + ~10 hang ≈ 35 frames).
        let secs = utts[0].len() as f32 / 16_000.0;
        assert!(secs < 1.0, "utterance {secs}s — should not run to the cap");
    }

    #[test]
    fn rms_and_zcr_basic() {
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        assert!((zero_crossing_rate(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        assert_eq!(zero_crossing_rate(&[1.0, 1.0, 1.0]), 0.0);
    }
}
