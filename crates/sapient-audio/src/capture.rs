//! Live microphone capture via `cpal` (behind the `audio-io` feature).
//!
//! Opens the default input device at its native rate/format and streams **mono
//! `f32`** chunks over a [`flume`] channel. Format/channel conversion happens in
//! the realtime callback (downmix to mono, any sample format → f32); the callback
//! never blocks — it `try_send`s and drops on overrun. The consumer resamples to
//! 16 kHz (the STT rate) once it has a full utterance. The `cpal::Stream` is not
//! `Send`, so keep the [`MicCapture`] on the thread that drives the loop.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SizedSample;

/// A running microphone capture. Dropping it stops the stream.
pub struct MicCapture {
    _stream: cpal::Stream,
    rx: flume::Receiver<Vec<f32>>,
    sample_rate: u32,
}

impl MicCapture {
    /// Open the default input device and start capturing.
    pub fn default_input() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default audio input device (is a mic connected?)"))?;
        // Prefer the device's default config, but fall back to enumerating the
        // supported ranges — on ALSA (e.g. a Raspberry Pi) `default_input_config()`
        // often fails with "requested stream type not supported" even though the
        // device advertises usable configs. Pick 16 kHz when it's in range (matches
        // STT), else the range's max.
        let supported = match device.default_input_config() {
            Ok(c) => c,
            Err(default_err) => {
                let mut best: Option<cpal::SupportedStreamConfigRange> = None;
                if let Ok(ranges) = device.supported_input_configs() {
                    for r in ranges {
                        let prefer = r.channels() == 1; // mono preferred
                        if best.is_none()
                            || (prefer && best.as_ref().map(|b| b.channels()) != Some(1))
                        {
                            best = Some(r);
                        }
                    }
                }
                let range = best.ok_or_else(|| {
                    anyhow!("no usable input config (default_input_config: {default_err})")
                })?;
                let want = cpal::SampleRate(16_000);
                let rate = if want >= range.min_sample_rate() && want <= range.max_sample_rate() {
                    want
                } else {
                    range.max_sample_rate()
                };
                range.with_sample_rate(rate)
            }
        };
        let sample_rate = supported.sample_rate().0;
        let channels = supported.channels() as usize;
        let fmt = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        // Bounded so a stalled consumer can't grow memory unbounded; overruns drop.
        let (tx, rx) = flume::bounded::<Vec<f32>>(64);

        let stream = match fmt {
            cpal::SampleFormat::F32 => build_input(&device, &config, channels, tx, |x: f32| x)?,
            cpal::SampleFormat::I16 => {
                build_input(&device, &config, channels, tx, |x: i16| x as f32 / 32768.0)?
            }
            cpal::SampleFormat::U16 => build_input(&device, &config, channels, tx, |x: u16| {
                (x as f32 - 32768.0) / 32768.0
            })?,
            other => anyhow::bail!("unsupported input sample format {other:?}"),
        };
        stream.play().context("starting input stream")?;
        tracing::info!("mic capture: {sample_rate} Hz, {channels} ch, {fmt:?}");
        Ok(Self {
            _stream: stream,
            rx,
            sample_rate,
        })
    }

    /// Native capture sample rate (resample to 16 kHz before STT).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Receiver of mono `f32` chunks at [`sample_rate`](Self::sample_rate).
    pub fn frames(&self) -> flume::Receiver<Vec<f32>> {
        self.rx.clone()
    }
}

/// cpal error callback (function item → `Copy`, usable across the format arms).
fn on_stream_error(e: cpal::StreamError) {
    tracing::warn!("audio input stream error: {e}");
}

fn build_input<T: SizedSample + 'static>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: flume::Sender<Vec<f32>>,
    to_f32: fn(T) -> f32,
) -> Result<cpal::Stream> {
    let ch = channels.max(1);
    let stream = device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let mut mono = Vec::with_capacity(data.len() / ch);
                for frame in data.chunks(ch) {
                    let sum: f32 = frame.iter().map(|&x| to_f32(x)).sum();
                    mono.push(sum / ch as f32);
                }
                // Never block the realtime thread; drop the chunk if the consumer
                // is behind (an overrun just shortens the captured audio slightly).
                let _ = tx.try_send(mono);
            },
            on_stream_error,
            None,
        )
        .context("building input stream")?;
    Ok(stream)
}
