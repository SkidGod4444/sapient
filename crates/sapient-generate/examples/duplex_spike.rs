// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Non-autoregressive duplex-codec spike harness (Paper 1, §7 "de-risking spike").
//!
//! GOAL: decide GO/NO-GO on a CPU-real-time, full-duplex voice agent built on a
//! NON-autoregressive synthesizer. The thesis: an autoregressive (AR) neural codec
//! cannot hold the duplex latency budget on a CPU (measured: Orpheus AR RTF 5.5 on
//! Metal, 17.4 on CPU), whereas a single-pass non-AR model can (Kokoro RTF ~0.6 on
//! M4 CPU). The open question:
//!
//!   Can the non-AR synthesizer be driven in STREAMING CHUNKS small enough to
//!   start speaking within the duplex budget (~200 ms; target ≤2–3×)?
//!
//! GATE 1 (synthesis cost, whole-utterance): does naive chunking work? (NO — the
//!   backbone + min-utterance padding impose a high fixed floor.)
//! GATE 2 (decoder-only streaming): run the amortizable backbone ONCE, then run
//!   ONLY the convolutional ISTFTNet decoder per time-slice. Measures the two
//!   quantities that turn the verdict from extrapolated to MEASURED:
//!     (a) decoder-only per-chunk latency for a ~0.3 s chunk;
//!     (b) minimum stable look-ahead = extra right-context frames after which the
//!         chunk's audio stops changing (the decoder's conv receptive field).
//!   Total latency-to-first-audio ≈ (a) + audio-duration of (b).
//!
//! Run:
//!   cargo run --release -p sapient-generate --example duplex_spike
//!   (Kokoro weights auto-load from cache or SAPIENT_KOKORO_DIR.)

use std::time::Instant;

use anyhow::Result;
use sapient_generate::{DecoderStreamInputs, KokoroTts, Tts};

/// Duplex latency budget (ms). Moshi reports ~200 ms practical (on a GPU); we
/// allow the spike target to be a multiple of it before declaring NO-GO.
const MOSHI_BUDGET_MS: f64 = 200.0;
const GO_MULTIPLE: f64 = 3.0; // ≤3× Moshi's budget on commodity CPU = GO.
const TARGET_CHUNK_S: f64 = 0.3; // streamed audio chunk granularity.
const STABLE_TOL: f32 = 1e-3; // strict: |Δsample| below this → bit-stable.
const PERCEPTUAL_TOL: f32 = 1e-2; // ~1% of full-scale → perceptually stable.

/// The interface a non-autoregressive streaming duplex synthesizer must satisfy.
/// Scaffold for the trained codec; `decode_prefix` below is the concrete decoder
/// half, now wired to the real ISTFTNet decoder.
#[allow(dead_code)]
pub trait StreamingDuplexSynth {
    fn push_latent(&mut self, latent: &[f32]) -> Result<Vec<f32>>;
    fn flush(&mut self) -> Result<Vec<f32>>;
    fn sample_rate(&self) -> u32;
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    let tts = rt.block_on(async { KokoroTts::from_default().await })?;
    let sr = tts.sample_rate();

    let _ = tts.synthesize("warm up")?; // pay one-time setup outside the numbers.

    gate1_whole_utterance(&tts, sr)?;
    gate2_decoder_only(&tts, sr)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// GATE 1 — whole-utterance synthesis cost (naive chunking).
// ---------------------------------------------------------------------------
fn gate1_whole_utterance(tts: &KokoroTts, sr: u32) -> Result<()> {
    let chunks: &[(&str, &str)] = &[
        ("1 word", "Hello."),
        ("3 words", "Hello there friend."),
        ("6 words", "Hello there, how are you today?"),
        (
            "12 words",
            "Hello there, how are you today? I hope this finds you well.",
        ),
    ];
    println!("\n=== GATE 1: whole-utterance synthesis (Kokoro-82M non-AR) ===");
    println!(
        "{:<10} {:>9} {:>10} {:>8}",
        "chunk", "audio(s)", "wall(ms)", "RTF"
    );
    for (label, text) in chunks {
        let mut best_ms = f64::INFINITY;
        let mut audio_s = 0.0;
        for _ in 0..3 {
            let t0 = Instant::now();
            let s = tts.synthesize(text)?;
            best_ms = best_ms.min(t0.elapsed().as_secs_f64() * 1000.0);
            audio_s = s.len() as f64 / sr as f64;
        }
        println!(
            "{:<10} {:>9.3} {:>10.1} {:>8.3}",
            label,
            audio_s,
            best_ms,
            (best_ms / 1000.0) / audio_s.max(1e-6)
        );
    }
    println!("(naive whole-utterance chunking has a high fixed floor — see GATE 2.)");
    Ok(())
}

// ---------------------------------------------------------------------------
// GATE 2 — decoder-only streaming: amortize the backbone, decode per chunk.
// ---------------------------------------------------------------------------
fn gate2_decoder_only(tts: &KokoroTts, sr: u32) -> Result<()> {
    // A longer utterance so we have enough decoder frames to chunk + probe.
    let text = "Hello there, how are you today? I hope this finds you well, and that \
                the cross backend latency study turns out to be useful for the paper.";

    println!("\n=== GATE 2: decoder-only streaming (amortized backbone) ===");

    // Backbone runs ONCE (amortized across the whole stream).
    let t0 = Instant::now();
    let inp: DecoderStreamInputs = tts.prepare_stream(text)?;
    let backbone_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let frames = inp.t;

    // Full decode → samples-per-frame, and the look-ahead reference ("infinite"
    // right context).
    let t0 = Instant::now();
    let full = tts.decode_prefix(&inp, frames)?;
    let full_decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let spf = full.len() as f64 / frames as f64; // samples per decoder frame
    let total_audio_s = full.len() as f64 / sr as f64;

    println!(
        "utterance: {frames} decoder frames, {:.2}s audio, {:.1} samples/frame",
        total_audio_s, spf
    );
    println!(
        "backbone (amortized once): {backbone_ms:.1} ms   full decode: {full_decode_ms:.1} ms \
         ({:.0} ms/s audio)",
        full_decode_ms / total_audio_s
    );

    // Chunk size for ~TARGET_CHUNK_S of audio.
    let chunk_frames = ((TARGET_CHUNK_S * sr as f64) / spf).round().max(1.0) as usize;
    let chunk_frames = chunk_frames.min(frames);

    // (a) Decoder-only latency for one ~0.3 s chunk (best of 3).
    let mut chunk_ms = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let _ = tts.decode_prefix(&inp, chunk_frames)?;
        chunk_ms = chunk_ms.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let chunk_audio_ms = chunk_frames as f64 * spf / sr as f64 * 1000.0;

    // (b) Minimum stable look-ahead: grow right-context until the chunk's first
    // `chunk_frames*spf` samples stop changing vs the full-context reference.
    let base = chunk_frames.min(frames);
    let cmp_len = (base as f64 * spf) as usize;
    let ref_prefix = &full[..cmp_len.min(full.len())];
    let mut strict_frames: Option<usize> = None;
    let mut perceptual_frames: Option<usize> = None;
    println!("look-ahead convergence (Δframes → max |sample diff| vs full context):");
    for &delta in &[0usize, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128] {
        if base + delta > frames {
            break;
        }
        let out = tts.decode_prefix(&inp, base + delta)?;
        let n = cmp_len.min(out.len()).min(ref_prefix.len());
        let max_diff = ref_prefix[..n]
            .iter()
            .zip(&out[..n])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("    Δ={delta:<4} max_diff={max_diff:.5}");
        if strict_frames.is_none() && max_diff < STABLE_TOL {
            strict_frames = Some(delta);
        }
        if perceptual_frames.is_none() && max_diff < PERCEPTUAL_TOL {
            perceptual_frames = Some(delta);
        }
    }

    println!(
        "\nchunk = {chunk_frames} frames (~{:.0} ms audio)",
        chunk_audio_ms
    );
    println!("(a) decoder-only latency / chunk: {chunk_ms:.1} ms");
    let la_frames_to_ms = |d: usize| d as f64 * spf / sr as f64 * 1000.0;
    match strict_frames {
        Some(d) => println!(
            "(b) strict look-ahead (<{STABLE_TOL}): {d} frames (~{:.0} ms audio)",
            la_frames_to_ms(d)
        ),
        None => println!(
            "(b) strict look-ahead (<{STABLE_TOL}): did NOT converge in 128 frames — a \
             global/length-dependent component (the iSTFT boundary) prevents clean prefix \
             decoding. A streaming overlap-add iSTFT is required for bit-stable chunks."
        ),
    }
    match perceptual_frames {
        Some(d) => {
            let la_ms = la_frames_to_ms(d);
            println!(
                "    perceptual look-ahead (<{PERCEPTUAL_TOL}): {d} frames (~{la_ms:.0} ms audio)"
            );
            verdict(chunk_ms, la_ms);
        }
        None => println!(
            "\nVERDICT: NO-GO — not even perceptually stable within 128 frames; the decoder \
             needs an explicit streaming/overlap-add redesign before a verdict."
        ),
    }
    Ok(())
}

fn verdict(chunk_ms: f64, lookahead_ms: f64) {
    let budget = MOSHI_BUDGET_MS * GO_MULTIPLE;
    let total = chunk_ms + lookahead_ms;
    println!(
        "\n--- GATE 2 GO/NO-GO ---\n\
         latency-to-first-audio ≈ decoder/chunk {chunk_ms:.0} ms + look-ahead {lookahead_ms:.0} ms \
         = {total:.0} ms   (budget {budget:.0} ms, Moshi {MOSHI_BUDGET_MS:.0} ms)"
    );
    let verdict = if total <= MOSHI_BUDGET_MS {
        "GO (within Moshi's own 200 ms budget)"
    } else if total <= budget {
        "GO (within 3× budget on commodity CPU)"
    } else {
        "NO-GO on this device — optimize the decoder or use a faster edge CPU"
    };
    println!("VERDICT: {verdict}");
    println!(
        "NOTE: per-device; rerun on Pi 5 / WebGPU. Real duplex also shares cores with \
         STT+LLM, which this isolated decoder measurement excludes.\n"
    );
}
