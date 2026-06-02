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
    pub fn new(stt: TranscribePipeline, llm: Pipeline, tts: Box<dyn Tts>) -> Self {
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
        let reply = self.llm.chat(&self.history).await?;
        self.history.push(ChatMessage::assistant(reply.clone()));

        let audio = self.tts.synthesize(&reply)?;
        let audio_sample_rate = self.tts.sample_rate();
        Ok(Some(Turn {
            transcript,
            reply,
            audio,
            audio_sample_rate,
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
