//! Golden gate for the Mixtral-class sparse-MoE path: Mixtral-8x7B-Instruct
//! (Q4_K_M GGUF) greedy must answer the capital question correctly. This is the
//! end-to-end check that no synthetic test can be: it exercises the real load
//! pipeline (GGUF → per-expert / stacked split → q/k un-permute → CPU dispatch),
//! Q4_K experts (repacked to Q4_K_R4 on aarch64, hitting the multi-row SDOT/SMMLA
//! kernels through `moe_ffn`), the real router, and Mixtral's attention — the exact
//! combination the unit/coherence tests can only approximate.
//!
//! Heavy: downloads ~26 GB and needs a 32 GB+ device (big Mac / Jetson Thor /
//! workstation). MoE is CPU-only for now, so `Auto` resolves to CPU.
//!
//! `cargo test -p sapient-generate --test moe_e2e --release -- --ignored --nocapture`

use sapient_generate::Pipeline;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads Mixtral-8x7B Q4_K_M (~26 GB); needs a 32 GB+ device"]
async fn mixtral_answers_capital_greedy() {
    let p = Pipeline::from_pretrained("mixtral-8x7b-q4")
        .await
        .expect("loading Mixtral-8x7B Q4_K_M");
    let reply = p
        .chat(&[sapient_tokenizers::ChatMessage::user(
            "What is the capital of France? Answer in one short sentence.",
        )])
        .await
        .expect("chat");
    println!("reply: {reply:?}");
    assert!(reply.contains("Paris"), "expected Paris in {reply:?}");
}
