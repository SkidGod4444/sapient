//! Live speaker playback via `cpal` (behind the `audio-io` feature).
//!
//! Opens the default output device and plays mono `f32` samples submitted to it,
//! resampling from the synthesizer's rate (e.g. 24 kHz TTS) to the device rate
//! and fanning out to all channels. The realtime callback drains a [`flume`]
//! channel and emits silence on underrun. Used to speak the TTS reply in the
//! speech-to-speech cascade.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SizedSample;

/// A running speaker output. Dropping it stops playback.
pub struct SpeakerPlayback {
    _stream: cpal::Stream,
    tx: flume::Sender<f32>,
    /// Second receiver handle onto the same queue (flume is MPMC): draining it
    /// steals queued samples from the device callback — instant silence.
    rx_drain: flume::Receiver<f32>,
    sample_rate: u32,
    /// Per-20 ms RMS envelope of everything submitted (device rate), plus the
    /// total sample count — lets a barge-in detector ask "how loud is the
    /// speaker RIGHT NOW?" and gate the mic against expected bleed instead of
    /// a fixed threshold (echo-referenced gating; there is no real AEC).
    env: std::sync::Mutex<Vec<f32>>,
    submitted: std::sync::atomic::AtomicUsize,
}

impl SpeakerPlayback {
    /// Open the default output device and start the (initially silent) stream.
    pub fn default_output() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default audio output device"))?;
        let supported = device
            .default_output_config()
            .context("querying default output config")?;
        let sample_rate = supported.sample_rate().0;
        let channels = supported.channels() as usize;
        let fmt = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        let (tx, rx) = flume::unbounded::<f32>();
        let rx_drain = rx.clone();

        let stream = match fmt {
            cpal::SampleFormat::F32 => build_output(&device, &config, channels, rx, |s: f32| s)?,
            cpal::SampleFormat::I16 => build_output(&device, &config, channels, rx, |s: f32| {
                (s.clamp(-1.0, 1.0) * 32767.0) as i16
            })?,
            cpal::SampleFormat::U16 => build_output(&device, &config, channels, rx, |s: f32| {
                ((s.clamp(-1.0, 1.0) * 32767.0) as i32 + 32768) as u16
            })?,
            other => anyhow::bail!("unsupported output sample format {other:?}"),
        };
        stream.play().context("starting output stream")?;
        tracing::info!("speaker playback: {sample_rate} Hz, {channels} ch, {fmt:?}");
        Ok(Self {
            _stream: stream,
            tx,
            rx_drain,
            sample_rate,
            env: std::sync::Mutex::new(Vec::new()),
            submitted: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Seconds of audio still queued (not yet played by the device). Lets a
    /// caller wait for playback to finish draining after the last `submit`.
    pub fn pending_secs(&self) -> f32 {
        self.tx.len() as f32 / self.sample_rate.max(1) as f32
    }

    /// Drop all queued (unplayed) audio immediately — the barge-in primitive.
    /// The device callback keeps running and emits silence until the next
    /// [`submit`](Self::submit).
    pub fn clear(&self) {
        while self.rx_drain.try_recv().is_ok() {}
    }

    /// Queue mono `samples` (at `src_rate`) for playback, resampling to the
    /// device rate. Returns immediately; audio drains on the device thread.
    pub fn submit(&self, samples: &[f32], src_rate: u32) -> Result<()> {
        let out = crate::io::resample(samples, src_rate, self.sample_rate)?;
        // Track the played-signal envelope (per-20 ms RMS at device rate).
        {
            let hop = (self.sample_rate as usize / 50).max(1);
            let mut env = self.env.lock().unwrap();
            for chunk in out.chunks(hop) {
                let r = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
                env.push(r);
            }
        }
        self.submitted
            .fetch_add(out.len(), std::sync::atomic::Ordering::Relaxed);
        for s in out {
            // Unbounded queue — playback is short (a reply); never blocks.
            let _ = self.tx.send(s);
        }
        Ok(())
    }

    /// RMS of the speaker signal at the CURRENT playback position (0.0 when
    /// idle) — the echo reference for barge-in: mic energy is compared against
    /// `α · expected_bleed()` rather than a fixed bar, so a reply that gets
    /// louder mid-sentence no longer out-shouts a threshold calibrated on its
    /// quiet opening.
    pub fn expected_bleed(&self) -> f32 {
        let submitted = self.submitted.load(std::sync::atomic::Ordering::Relaxed);
        let pending = self.tx.len();
        let played = submitted.saturating_sub(pending);
        let hop = (self.sample_rate as usize / 50).max(1);
        let idx = played / hop;
        let env = self.env.lock().unwrap();
        // Small look-around: device/room latency smears the reference by a few
        // hops — take the local max so attacks aren't under-estimated.
        let lo = idx.saturating_sub(3);
        let hi = (idx + 4).min(env.len());
        env[lo..hi].iter().cloned().fold(0.0f32, f32::max)
    }

    /// Drop envelope history — call between turns (after playback has drained
    /// or been cleared) so the reference stays bounded and re-based at zero.
    pub fn reset_reference(&self) {
        self.env.lock().unwrap().clear();
        self.submitted
            .store(self.tx.len(), std::sync::atomic::Ordering::Relaxed);
    }
}

fn build_output<T: SizedSample + 'static>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    rx: flume::Receiver<f32>,
    from_f32: fn(f32) -> T,
) -> Result<cpal::Stream> {
    let ch = channels.max(1);
    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                for frame in data.chunks_mut(ch) {
                    // One mono sample fanned out to every channel; silence on underrun.
                    let v = from_f32(rx.try_recv().unwrap_or(0.0));
                    for slot in frame.iter_mut() {
                        *slot = v;
                    }
                }
            },
            on_stream_error,
            None,
        )
        .context("building output stream")?;
    Ok(stream)
}

fn on_stream_error(e: cpal::StreamError) {
    tracing::warn!("audio output stream error: {e}");
}
