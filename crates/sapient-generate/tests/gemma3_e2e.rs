// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Golden gate for the Gemma3 text engine: gemma-3-1b-it greedy must answer
//! the capital question exactly — this catches the whole class of "wired but
//! wrong math" regressions (norm folding, QK-norm, sandwich norms, sliding
//! masks, embedding scale).
//!
//! Ignored by default (downloads ~2 GB):
//! `cargo test -p sapient-generate --test gemma3_e2e --release -- --ignored --nocapture`

use sapient_generate::Pipeline;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads gemma-3-1b-it"]
async fn gemma3_1b_answers_capital_greedy() {
    let p = Pipeline::from_pretrained("gemma-3-1b")
        .await
        .expect("loading gemma-3-1b");
    let reply = p
        .chat(&[sapient_tokenizers::ChatMessage::user(
            "What is the capital of France? Answer in one short sentence.",
        )])
        .await
        .expect("chat");
    println!("reply: {reply:?}");
    assert!(reply.contains("Paris"), "expected Paris in {reply:?}");
}
