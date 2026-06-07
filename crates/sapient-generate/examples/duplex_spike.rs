//! Non-autoregressive duplex-codec spike harness (Paper 1, §7 "de-risking spike").
//!
//! GOAL: decide GO/NO-GO on a CPU-real-time, full-duplex voice agent built on a
//! NON-autoregressive synthesizer. The flagship paper's thesis is that an
//! autoregressive (AR) neural codec cannot hold the duplex latency budget on a
//! CPU (measured: Orpheus AR RTF 5.5 on Metal, 17.4 on CPU), whereas a single-pass
//! non-AR model can (measured: Kokoro RTF ~0.76 on CPU). The open question this
//! spike answers empirically:
//!
//!   Can the non-AR synthesizer be driven in STREAMING CHUNKS small enough to
//!   start speaking within the duplex latency budget (~200 ms, target ≤2–3×)?
//!
//! WHAT THIS HARNESS MEASURES TODAY (real, on the shipped Kokoro decoder):
//!   1. Per-chunk synthesis latency vs. chunk audio duration (the dominant term
//!      in latency-to-first-audio for a chunked non-AR stream).
//!   2. Steady-state RTF per chunk size (is each chunk faster than it plays?).
//!   3. A GO/NO-GO verdict against a configurable duplex budget.
//!
//! WHAT IS SCAFFOLDED (marked TODO(spike)) — the real codec work the paper gates on:
//!   * A trained STREAMING dual-stream backbone that emits a per-frame prosody
//!     latent (cheap, not acoustic tokens) — see `StreamingDuplexSynth`.
//!   * A streaming ISTFTNet decoder callable on a latent + duration with bounded
//!     right-context, and a rigorous minimum-stable-look-ahead measurement.
//!
//! Run:
//!   SAPIENT_KOKORO_DIR=<dir> cargo run --release -p sapient-generate --example duplex_spike
//!   (or with weights cached, just `cargo run --release ... --example duplex_spike`)

use std::time::Instant;

use anyhow::Result;
use sapient_generate::{KokoroTts, Tts};

/// Duplex latency budget (ms). Moshi reports ~200 ms practical (on a GPU); we
/// allow the spike's target to be a multiple of it before declaring NO-GO.
const MOSHI_BUDGET_MS: f64 = 200.0;
const GO_MULTIPLE: f64 = 3.0; // ≤3× Moshi's budget on commodity CPU = GO.

/// The interface a non-autoregressive streaming duplex synthesizer must satisfy.
/// This is the SCAFFOLD for the trained codec; the harness below measures whether
/// the underlying non-AR vocoder is fast enough for it to be viable at all.
pub trait StreamingDuplexSynth {
    /// Push one frame's prosody/style latent (produced by the duplex backbone)
    /// and receive any audio samples whose value is now STABLE (won't change as
    /// more right-context arrives). Returns the stable samples for this step.
    fn push_latent(&mut self, latent: &[f32]) -> Result<Vec<f32>>;
    /// Flush remaining (held-back look-ahead) audio at end of turn.
    fn flush(&mut self) -> Result<Vec<f32>>;
    fn sample_rate(&self) -> u32;
}

// TODO(spike): implement `StreamingKokoroDecoder: StreamingDuplexSynth` that
// drives the ISTFTNet decoder (crates/sapient-models/.../kokoro/decoder.rs) on a
// rolling latent window with a fixed right-context margin, mirroring the
// stable-prefix logic in speak.rs::emit_stable (STREAM_MARGIN_FRAMES). The
// decoder is convolutional, so its receptive field bounds the look-ahead; the
// rigorous "minimum stable look-ahead" number is the receptive field measured by
// appending right-context and finding where the emitted prefix stops changing.

struct ChunkStat {
    label: &'static str,
    audio_s: f64,
    wall_ms: f64,
    rtf: f64,
}

fn main() -> Result<()> {
    // Kokoro construction is async (Hub cache check); use a small runtime.
    let rt = tokio::runtime::Runtime::new()?;
    let tts = rt.block_on(async { KokoroTts::from_default().await })?;
    let sr = tts.sample_rate() as f64;

    // Representative streaming chunk granularities, from a single word (smallest
    // useful unit) up to a short sentence. The first row is the critical one:
    // the per-chunk fixed cost that bounds latency-to-first-audio.
    let chunks: &[(&str, &str)] = &[
        ("1 word", "Hello."),
        ("3 words", "Hello there friend."),
        ("6 words", "Hello there, how are you today?"),
        ("12 words", "Hello there, how are you today? I hope this finds you well."),
        (
            "~24 words",
            "Hello there, how are you today? I hope this finds you well, and that \
             the cross backend latency study is going to be useful for the paper.",
        ),
    ];

    // Warm up (first call pays one-time setup we don't want in the numbers).
    let _ = tts.synthesize("warm up")?;

    println!("\n=== Non-AR duplex spike: per-chunk synthesis latency (Kokoro-82M) ===");
    println!("budget: Moshi {MOSHI_BUDGET_MS:.0} ms; GO threshold ≤ {:.0} ms ({GO_MULTIPLE:.0}×)\n", MOSHI_BUDGET_MS * GO_MULTIPLE);
    println!(
        "{:<10} {:>6} {:>10} {:>10} {:>8}",
        "chunk", "words", "audio(s)", "wall(ms)", "RTF"
    );

    let runs = 3usize;
    let mut stats = Vec::new();
    for (label, text) in chunks {
        let words = text.split_whitespace().count();
        let mut best_ms = f64::INFINITY;
        let mut audio_s = 0.0;
        for _ in 0..runs {
            let t0 = Instant::now();
            let samples = tts.synthesize(text)?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            best_ms = best_ms.min(ms);
            audio_s = samples.len() as f64 / sr;
        }
        let rtf = (best_ms / 1000.0) / audio_s.max(1e-6);
        println!(
            "{:<10} {:>6} {:>10.3} {:>10.1} {:>8.3}",
            label, words, audio_s, best_ms, rtf
        );
        let _ = words;
        stats.push(ChunkStat { label, audio_s, wall_ms: best_ms, rtf });
    }

    verdict(&stats);
    Ok(())
}

/// GO/NO-GO logic. Latency-to-first-audio for a chunked non-AR stream is
/// approximately the synthesis time of the SMALLEST useful chunk plus a
/// look-ahead margin. If the smallest chunk synthesizes well within the budget
/// AND larger chunks stay at RTF < 1 (sustainable), the approach is viable.
fn verdict(stats: &[ChunkStat]) {
    println!("\n--- GO/NO-GO ---");
    let smallest = &stats[0];
    let sustainable = stats.iter().all(|s| s.rtf < 1.0);
    let budget = MOSHI_BUDGET_MS * GO_MULTIPLE;

    println!(
        "smallest useful chunk ('{}', {:.2}s audio): synth {:.1} ms (budget {:.0} ms)",
        smallest.label, smallest.audio_s, smallest.wall_ms, budget
    );
    println!(
        "all chunks RTF < 1.0 (sustainable streaming): {}",
        if sustainable { "yes" } else { "NO" }
    );

    // TODO(spike): add the measured minimum stable look-ahead (ms) once the
    // streaming decoder exists; the real latency budget is
    //   latency = smallest_chunk_synth + look_ahead.
    let go = smallest.wall_ms < budget && sustainable;
    println!(
        "\nVERDICT (synthesis-cost gate only): {}",
        if go {
            "GO — non-AR synthesis is fast enough; build the streaming decoder + \
             measure look-ahead next."
        } else {
            "NO-GO on synthesis cost alone — even the smallest chunk exceeds the \
             budget; fall back to a tightened cascade."
        }
    );
    println!(
        "NOTE: this gate covers ONLY synthesis cost. The full GO also requires the \
         minimum-stable-look-ahead measurement (TODO(spike)) to fit within \
         {budget:.0} ms total.\n"
    );
}
