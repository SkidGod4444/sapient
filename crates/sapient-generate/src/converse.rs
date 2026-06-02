//! `ConversePipeline` — the speech-to-speech cascade orchestration core.
//!
//! Ties the shipped pieces into one turn: mic samples → **STT** ([`TranscribePipeline`])
//! → **LLM** ([`Pipeline::chat`], with running history) → **TTS** ([`Tts`]) → audio.
//! This is device-agnostic — it operates on `&[f32]` mono 16 kHz samples in and
//! `Vec<f32>` samples out, so it is fully usable today (voice-in / text-out via the
//! [`NoopTts`] stub) and testable by injecting a decoded WAV. The microphone /
//! speaker layer (cpal) and the `sapient converse` CLI sit on top behind an
//! `audio-io` feature; the real Phase 6d TTS replaces [`NoopTts`] when it lands.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use sapient_tokenizers::ChatMessage;

use crate::transcribe::{TranscribeOptions, TranscribePipeline};
use crate::Pipeline;

/// A text-to-speech backend. Implementors turn a reply string into mono audio
/// samples at [`Tts::sample_rate`]. Phase 6d (Llama→SNAC) will provide the real one.
pub trait Tts: Send + Sync {
    fn synthesize(&self, text: &str) -> Result<Vec<f32>>;
    fn sample_rate(&self) -> u32;

    /// Synthesize `text`, emitting audio in chunks via `on_audio(samples, rate)`
    /// as soon as each is ready — for real-time playback that starts before the
    /// whole clip is generated. The default batches (calls [`synthesize`] and
    /// emits once); a codec TTS overrides this to stream as its LM decodes.
    fn synthesize_streaming(
        &self,
        text: &str,
        on_audio: &mut dyn FnMut(&[f32], u32),
    ) -> Result<()> {
        let audio = self.synthesize(text)?;
        if !audio.is_empty() {
            on_audio(&audio, self.sample_rate());
        }
        Ok(())
    }
}

/// Placeholder TTS that produces no audio — lets the cascade run voice-in /
/// text-out before a real synthesizer is wired in.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopTts;

impl Tts for NoopTts {
    fn synthesize(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(Vec::new())
    }
    fn sample_rate(&self) -> u32 {
        24_000
    }
}

/// The result of one conversational turn.
#[derive(Debug, Clone)]
pub struct Turn {
    /// What the user said (STT output).
    pub transcript: String,
    /// The assistant's text reply (LLM output).
    pub reply: String,
    /// Synthesized reply audio (empty with [`NoopTts`]).
    pub audio: Vec<f32>,
    /// Sample rate of `audio`.
    pub audio_sample_rate: u32,
    /// Number of reply tokens generated (streamed pieces — approximate).
    pub gen_tokens: usize,
    /// Wall-clock time spent generating the reply.
    pub gen_ms: u128,
    /// Wall-clock time spent synthesizing TTS audio (0 with [`NoopTts`]).
    pub tts_ms: u128,
}

/// Orchestrates STT → LLM → TTS across a conversation, holding chat history.
pub struct ConversePipeline {
    stt: TranscribePipeline,
    llm: Pipeline,
    tts: Box<dyn Tts>,
    history: Vec<ChatMessage>,
    stt_opts: TranscribeOptions,
}

impl ConversePipeline {
    /// Build from a loaded STT pipeline, LLM pipeline, and TTS backend.
    pub fn new(stt: TranscribePipeline, mut llm: Pipeline, tts: Box<dyn Tts>) -> Self {
        // Each turn re-sends the whole conversation (system + history) as the
        // prompt, so its token prefix is identical to the previous turn's up to
        // the new user message. Prefix/prompt KV caching reuses that prefix
        // instead of re-prefilling it every turn — the per-turn latency win grows
        // with the conversation length. (Same feature `sapient serve` enables;
        // it's in-process, so converse needs no HTTP round-trip to benefit.)
        llm.enable_prefix_cache();
        Self {
            stt,
            llm,
            tts,
            history: Vec::new(),
            stt_opts: TranscribeOptions::default(),
        }
    }

    /// Seed a system prompt (must be called before the first turn).
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.history.insert(0, ChatMessage::system(system));
        self
    }

    /// Force the STT language (otherwise auto-detected per utterance).
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.stt_opts.language = Some(lang.into());
        self
    }

    /// The conversation history so far (system + user/assistant turns).
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// A display label for the backend the LLM resolved to (e.g.
    /// `"metal (MLX native graph)"` or a CPU label). The TTS uses the same
    /// backend kind, so this reflects the whole compute path's accelerator.
    pub fn backend_label(&self) -> String {
        self.llm.backend_display_label()
    }

    /// Transcribe one utterance (mono 16 kHz) to text — STT only, no history
    /// mutation. Returns the trimmed transcript (possibly empty for silence).
    /// Used by the live CLI to show the transcript before streaming the reply.
    pub fn transcribe_utterance(&mut self, samples: &[f32]) -> Result<String> {
        let opts = self.stt_opts.clone();
        let t = tokio::task::block_in_place(|| self.stt.transcribe_samples(samples, &opts))?;
        Ok(t.trim().to_string())
    }

    /// Given a user `transcript`, stream the assistant's reply token-by-token to
    /// `on_token`, then synthesize TTS **one sentence at a time** — invoking
    /// `on_audio(samples, sample_rate)` for each sentence as soon as it is ready
    /// so playback can begin (and pipeline with synthesis) before the whole
    /// reply is spoken. Appends both messages to history and returns the [`Turn`]
    /// (with timing/token metrics). The live counterpart of
    /// [`run_utterance`](Self::run_utterance).
    ///
    /// With [`NoopTts`] each `synthesize` is empty, so `on_audio` is never called
    /// and `audio` is empty — the text path is unaffected.
    pub async fn respond_streaming<F, G>(
        &mut self,
        transcript: &str,
        mut on_token: F,
        mut on_audio: G,
    ) -> Result<Turn>
    where
        F: FnMut(&str),
        G: FnMut(&[f32], u32),
    {
        use std::time::Instant;

        use futures::StreamExt;

        self.history.push(ChatMessage::user(transcript.to_string()));

        let gen_start = Instant::now();
        let mut stream = self.llm.chat_stream(&self.history).await;
        let mut reply = String::new();
        let mut gen_tokens = 0usize;
        while let Some(tok) = stream.next().await {
            gen_tokens += 1;
            reply.push_str(&tok);
            on_token(&tok);
        }
        let gen_ms = gen_start.elapsed().as_millis();
        self.history.push(ChatMessage::assistant(reply.clone()));

        // Synthesize the **whole reply as one clip**, then play it once. Splitting
        // per sentence made the player run dry between sentences (the codec LM at
        // ~17 tok/s is slower than real-time, so the next sentence isn't ready
        // when the current finishes → a long mid-reply break). Decoding the full
        // reply up front trades a bit more time-to-first-audio for **gap-free
        // playback start-to-finish**. The brevity system prompt keeps `--speak`
        // replies short so that upfront wait stays small. TTS is CPU-bound — keep
        // it off the async reactor. (The streaming `Tts` path is retained for a
        // future real-time-capable small model.)
        let sr = self.tts.sample_rate();
        let t = std::time::Instant::now();
        let audio = tokio::task::block_in_place(|| self.tts.synthesize(&reply))?;
        let tts_ms = t.elapsed().as_millis();
        if !audio.is_empty() {
            on_audio(&audio, sr);
        }

        Ok(Turn {
            transcript: transcript.to_string(),
            reply,
            audio,
            audio_sample_rate: sr,
            gen_tokens,
            gen_ms,
            tts_ms,
        })
    }

    /// Run one turn from mono 16 kHz samples: transcribe → chat → synthesize.
    /// Appends both the user transcript and the assistant reply to history.
    /// Returns `None` when the utterance transcribes to empty (silence/noise).
    pub async fn run_utterance(&mut self, samples: &[f32]) -> Result<Option<Turn>> {
        // STT (CPU-bound, synchronous) off the async reactor.
        let transcript = {
            let opts = self.stt_opts.clone();
            tokio::task::block_in_place(|| self.stt.transcribe_samples(samples, &opts))?
        };
        let transcript = transcript.trim().to_string();
        if transcript.is_empty() {
            return Ok(None);
        }

        self.history.push(ChatMessage::user(transcript.clone()));
        let gen_start = std::time::Instant::now();
        let reply = self.llm.chat(&self.history).await?;
        let gen_ms = gen_start.elapsed().as_millis();
        self.history.push(ChatMessage::assistant(reply.clone()));

        let tts_start = std::time::Instant::now();
        let audio = self.tts.synthesize(&reply)?;
        let tts_ms = tts_start.elapsed().as_millis();
        let audio_sample_rate = self.tts.sample_rate();
        Ok(Some(Turn {
            transcript,
            reply,
            audio,
            audio_sample_rate,
            gen_tokens: 0,
            gen_ms,
            tts_ms,
        }))
    }
}

/// A [`Tts`] backed by a closure — convenient for tests and for plugging in a
/// synthesizer without a dedicated type.
pub struct FnTts<F: Fn(&str) -> Result<Vec<f32>> + Send + Sync> {
    f: F,
    sample_rate: u32,
}

impl<F: Fn(&str) -> Result<Vec<f32>> + Send + Sync> FnTts<F> {
    pub fn new(sample_rate: u32, f: F) -> Self {
        Self { f, sample_rate }
    }
}

impl<F: Fn(&str) -> Result<Vec<f32>> + Send + Sync> Tts for FnTts<F> {
    fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        (self.f)(text)
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Shared-handle alias so a real-time loop can hold the pipeline across tasks.
pub type SharedConverse = Arc<Mutex<ConversePipeline>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_tts_is_silent_24k() {
        let t = NoopTts;
        assert!(t.synthesize("hello").unwrap().is_empty());
        assert_eq!(t.sample_rate(), 24_000);
    }

    #[test]
    fn fn_tts_invokes_closure() {
        let t = FnTts::new(16_000, |s: &str| Ok(vec![s.len() as f32]));
        assert_eq!(t.synthesize("abc").unwrap(), vec![3.0]);
        assert_eq!(t.sample_rate(), 16_000);
    }
}
