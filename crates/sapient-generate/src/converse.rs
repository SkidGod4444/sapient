// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

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
    /// Time from reply-start to the first LLM token (prefill latency).
    pub ttft_ms: u128,
    /// Time from reply-start to the first audio chunk handed to `on_audio`
    /// (`None` when no audio was produced — e.g. [`NoopTts`]). This is the
    /// number the voice loop optimizes: silence between user and assistant.
    pub first_audio_ms: Option<u128>,
}

/// Orchestrates STT → LLM → TTS across a conversation, holding chat history.
pub struct ConversePipeline {
    stt: TranscribePipeline,
    llm: Pipeline,
    // `Arc` (not `Box`) so a background synthesis worker can hold the TTS while the
    // LLM keeps generating on the main task — the streaming overlap in `respond_streaming`.
    tts: Arc<dyn Tts>,
    history: Vec<ChatMessage>,
    stt_opts: TranscribeOptions,
}

impl ConversePipeline {
    /// Build from a loaded STT pipeline, LLM pipeline, and TTS backend.
    pub fn new(stt: TranscribePipeline, mut llm: Pipeline, tts: Arc<dyn Tts>) -> Self {
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

    /// Build a [`LiveStt`] — a background incremental transcriber sharing this
    /// pipeline's STT engine and options. Feed it snapshots of the in-progress
    /// utterance while the user is still speaking; by end-of-utterance the
    /// transcript is (usually) already computed, taking STT off the
    /// perceived-latency critical path entirely.
    pub fn live_stt(&self) -> LiveStt {
        LiveStt::new(self.stt.clone(), self.stt_opts.clone())
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

        use crate::sentence::SentenceChunker;

        self.history.push(ChatMessage::user(transcript.to_string()));
        let sr = self.tts.sample_rate();

        // ── streaming TTS pipeline ───────────────────────────────────────────
        // A background worker synthesizes one sentence at a time while the LLM
        // keeps generating the next on the main task and the speaker plays the
        // previous — three stages overlapped. Time-to-first-audio collapses to
        // "first sentence" (not the whole reply), and LLM generation no longer
        // waits on synthesis. The `SentenceChunker` `min_chars` guard avoids
        // splitting on abbreviations/decimals; short interjections merge forward.
        let tts = Arc::clone(&self.tts);
        let (sent_tx, mut sent_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Result<Vec<f32>>>();
        let worker = tokio::task::spawn_blocking(move || -> u128 {
            let mut synth_ms = 0u128;
            while let Some(sentence) = sent_rx.blocking_recv() {
                let t = Instant::now();
                // Streaming synthesis: chunks are forwarded the moment the
                // synthesizer emits them (Kokoro's decoder-only prefix path
                // emits the first ~0.6 s long before the fragment finishes;
                // batch TTS backends fall back to one emission at the end).
                let atx = audio_tx.clone();
                let r = tts.synthesize_streaming(&sentence, &mut |samples, _rate| {
                    let _ = atx.send(Ok(samples.to_vec()));
                });
                synth_ms += t.elapsed().as_millis();
                if let Err(e) = r {
                    let _ = audio_tx.send(Err(e));
                    break;
                }
            }
            synth_ms
        });

        let total_start = Instant::now();
        let mut ttft_ms = 0u128;
        let mut first_audio_ms: Option<u128> = None;
        let mut reply = String::new();
        let mut gen_tokens = 0usize;
        let gen_ms; // set exactly once, on the stream-end branch below
        let mut all_audio: Vec<f32> = Vec::new();
        // Early-first-clause mode: the speaker starts after the first ~24
        // chars at a clause boundary instead of the first full sentence —
        // time-to-first-audio ≈ TTS RTF × one clause, not × one sentence.
        let mut chunker = SentenceChunker::new(8, 200).with_early_first(24);
        let mut stream = self.llm.chat_stream(&self.history).await;

        // Drive the LLM stream and drain synthesized audio concurrently.
        loop {
            tokio::select! {
                got = audio_rx.recv() => {
                    match got {
                        Some(Ok(audio)) => {
                            if !audio.is_empty() {
                                first_audio_ms
                                    .get_or_insert_with(|| total_start.elapsed().as_millis());
                                on_audio(&audio, sr);
                                all_audio.extend_from_slice(&audio);
                            }
                        }
                        Some(Err(e)) => return Err(e),
                        None => {}
                    }
                }
                tok = stream.next() => {
                    match tok {
                        Some(t) => {
                            if gen_tokens == 0 {
                                ttft_ms = total_start.elapsed().as_millis();
                            }
                            gen_tokens += 1;
                            reply.push_str(&t);
                            on_token(&t);
                            for sentence in chunker.push(&t) {
                                let _ = sent_tx.send(sentence);
                            }
                        }
                        None => {
                            gen_ms = total_start.elapsed().as_millis();
                            break;
                        }
                    }
                }
            }
        }
        drop(stream);

        // LLM done: flush the tail, close the sentence channel (the worker exits
        // once its queue drains), then drain any audio still in flight.
        if let Some(rest) = chunker.flush() {
            let _ = sent_tx.send(rest);
        }
        drop(sent_tx);
        while let Some(got) = audio_rx.recv().await {
            match got {
                Ok(audio) => {
                    if !audio.is_empty() {
                        first_audio_ms.get_or_insert_with(|| total_start.elapsed().as_millis());
                        on_audio(&audio, sr);
                        all_audio.extend_from_slice(&audio);
                    }
                }
                Err(e) => return Err(e),
            }
        }
        let tts_ms = worker.await.unwrap_or(0);

        self.history.push(ChatMessage::assistant(reply.clone()));

        Ok(Turn {
            transcript: transcript.to_string(),
            reply,
            audio: all_audio,
            audio_sample_rate: sr,
            gen_tokens,
            gen_ms,
            tts_ms,
            ttft_ms,
            first_audio_ms,
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
            ttft_ms: 0,
            first_audio_ms: None,
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

/// Background incremental speech-to-text over a growing utterance.
///
/// The live loop feeds monotonically-growing snapshots of the in-progress
/// utterance (mono 16 kHz); a worker thread re-transcribes the newest snapshot
/// whenever it is idle (intermediate snapshots are skipped, so the worker never
/// falls behind). When the utterance finalizes, [`settle`](Self::settle) waits
/// for the in-flight pass and returns the latest transcript plus how many
/// samples it covered — if only trailing silence arrived since, the caller can
/// use it directly and skip the final full transcription.
pub struct LiveStt {
    tx: std::sync::mpsc::Sender<Vec<f32>>,
    state: Arc<(Mutex<LiveSttState>, std::sync::Condvar)>,
}

#[derive(Default)]
struct LiveSttState {
    transcript: String,
    covered_samples: usize,
    /// Snapshots submitted vs completed — settle() waits for equality.
    fed_seq: u64,
    done_seq: u64,
}

impl LiveStt {
    /// Build directly from a transcriber (the [`ConversePipeline::live_stt`]
    /// path is the normal route; this exists for tests/tools that stream STT
    /// without a full converse pipeline).
    pub fn for_transcriber(stt: TranscribePipeline, opts: TranscribeOptions) -> Self {
        Self::new(stt, opts)
    }

    fn new(stt: TranscribePipeline, opts: TranscribeOptions) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let state: Arc<(Mutex<LiveSttState>, std::sync::Condvar)> = Arc::default();
        let wstate = Arc::clone(&state);
        std::thread::Builder::new()
            .name("live-stt".into())
            .spawn(move || {
                while let Ok(mut snapshot) = rx.recv() {
                    // Drain to the newest snapshot; count everything as handled.
                    let mut skipped = 0u64;
                    while let Ok(newer) = rx.try_recv() {
                        snapshot = newer;
                        skipped += 1;
                    }
                    let covered = snapshot.len();
                    let text = stt
                        .transcribe_samples(&snapshot, &opts)
                        .map(|t| t.trim().to_string())
                        .unwrap_or_default();
                    let (lock, cvar) = &*wstate;
                    let mut st = lock.lock().unwrap();
                    // A longer pass may already have landed (shouldn\'t happen
                    // with one worker, but keep the invariant explicit).
                    if covered >= st.covered_samples {
                        st.covered_samples = covered;
                        st.transcript = text;
                    }
                    st.done_seq += 1 + skipped;
                    cvar.notify_all();
                }
            })
            .expect("spawning live-stt worker");
        Self { tx, state }
    }

    /// Feed the utterance-so-far (mono 16 kHz). Non-blocking; the worker picks
    /// up the newest pending snapshot when idle.
    pub fn feed(&self, utterance_so_far: Vec<f32>) {
        if utterance_so_far.is_empty() {
            return;
        }
        let (lock, _) = &*self.state;
        lock.lock().unwrap().fed_seq += 1;
        let _ = self.tx.send(utterance_so_far);
    }

    /// Wait (up to `max_wait`) for all fed snapshots to be transcribed, then
    /// return `(transcript, samples_covered)` — the caller compares
    /// `samples_covered` against the finalized utterance length to decide
    /// whether a final full pass is still needed.
    pub fn settle(&self, max_wait: std::time::Duration) -> (String, usize) {
        let (lock, cvar) = &*self.state;
        let deadline = std::time::Instant::now() + max_wait;
        let mut st = lock.lock().unwrap();
        while st.done_seq < st.fed_seq {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let (next, timeout) = cvar.wait_timeout(st, deadline - now).unwrap();
            st = next;
            if timeout.timed_out() {
                break;
            }
        }
        (st.transcript.clone(), st.covered_samples)
    }

    /// Reset between utterances so a stale transcript can never leak forward.
    pub fn reset(&self) {
        let (lock, _) = &*self.state;
        let mut st = lock.lock().unwrap();
        st.transcript.clear();
        st.covered_samples = 0;
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
