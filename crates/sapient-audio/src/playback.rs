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
    sample_rate: u32,
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
            sample_rate,
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

    /// Queue mono `samples` (at `src_rate`) for playback, resampling to the
    /// device rate. Returns immediately; audio drains on the device thread.
    pub fn submit(&self, samples: &[f32], src_rate: u32) -> Result<()> {
        let out = crate::io::resample(samples, src_rate, self.sample_rate)?;
        for s in out {
            // Unbounded queue — playback is short (a reply); never blocks.
            let _ = self.tx.send(s);
        }
        Ok(())
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
