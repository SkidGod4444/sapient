// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! MoE decode/prefill micro-benchmark on real Mixtral-8x7B Q4_K_M.
//!
//! Ignored (needs the model + 32 GB). Loads the model ONCE and emits, apples-to-
//! apples with llama.cpp: prefill tok/s, **pure** decode tok/s (steps after the
//! first token — no prefill in the denominator), peak RSS, and the greedy output
//! (for token-parity against `llama-cli` at temp 0). Match thread count with
//! `RAYON_NUM_THREADS` (and llama.cpp's `-t`); set `MOE_BENCH_DECODE` for the
//! generated-token count.
//!
//! `RAYON_NUM_THREADS=N cargo test -p sapient-generate --test moe_bench --release -- --ignored --nocapture`

use std::time::Instant;

use sapient_generate::{Pipeline, SamplingStrategy};
use sapient_tokenizers::ChatMessage;

/// Peak resident set size (GB) from Linux `/proc/self/status` VmHWM.
fn peak_rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "MoE benchmark — needs Mixtral-8x7B Q4_K_M (~26 GB) + 32 GB RAM"]
async fn mixtral_decode_bench() {
    let decode_n: usize = std::env::var("MOE_BENCH_DECODE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    // Model alias — defaults to Mixtral, override to bench a dense proxy
    // (e.g. `qwen2.5-1.5b-q4`) for the SAPIENT-vs-llama.cpp base-kernel gap.
    let model = std::env::var("MOE_BENCH_MODEL").unwrap_or_else(|_| "mixtral-8x7b-q4".into());

    let p = Pipeline::from_pretrained(&model)
        .await
        .unwrap_or_else(|e| panic!("load {model}: {e}"));

    // Fixed prompt so the run is reproducible and can be replayed verbatim in llama.cpp.
    let prompt = p
        .format_chat_prompt(&[ChatMessage::user(
            "Write a detailed paragraph about the history of the Roman Empire.",
        )])
        .expect("format prompt");
    let prompt_ids = p.tokenizer().encode(&prompt).expect("encode");
    let plen = prompt_ids.len();

    // Greedy (temp 0) for determinism + llama.cpp parity. No stop_ids → a fixed
    // token count for a clean rate (steps after the first are all warm decode).
    let t0 = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut toks: Vec<u32> = Vec::new();
    p.generate_token_ids_streaming(
        &prompt_ids,
        decode_n,
        &[],
        SamplingStrategy::Greedy,
        |tok| {
            if t_first.is_none() {
                t_first = Some(Instant::now());
            }
            toks.push(tok);
            true
        },
    )
    .expect("generate");
    let t_end = Instant::now();

    let t_first = t_first.expect("no tokens generated");
    let prefill_s = (t_first - t0).as_secs_f64(); // prompt prefill + first token
    let decode_s = (t_end - t_first).as_secs_f64(); // pure decode (steps 2..N)
    let decode_toks = toks.len().saturating_sub(1);
    let text = p.tokenizer().decode(&toks, true).unwrap_or_default();
    let threads =
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all-cores(default)".into());

    println!("\n===== BENCH · {model} · SAPIENT =====");
    println!("RAYON_NUM_THREADS = {threads}");
    println!("prompt_tokens = {plen}   generated_tokens = {}", toks.len());
    println!(
        "PREFILL : {prefill_s:.3} s  ->  {:.2} tok/s  (prompt processing)",
        plen as f64 / prefill_s.max(1e-9)
    );
    println!(
        "DECODE  : {decode_s:.3} s  ->  {:.2} tok/s  (pure, first token excluded)",
        decode_toks as f64 / decode_s.max(1e-9)
    );
    println!("PEAK RSS: {:.2} GB", peak_rss_gb());
    println!("--- greedy output ---\n{text}");
    println!("--- prompt token ids (for llama.cpp parity) ---\n{prompt_ids:?}");
}
