//! End-to-end Whisper transcription test (ignored — needs network + an audio file).
//!
//! This is the absolute-correctness gate: it downloads the real `whisper-tiny`
//! checkpoint and transcribes a known clip, so a math error anywhere in the mel
//! front-end / encoder / cross-attention / decoder produces a wrong transcript
//! and fails the assertion.
//!
//! It is `#[ignore]` (network + multi-second). Provide a 16 kHz speech WAV via
//! the `SAPIENT_TEST_WAV` env var, e.g. the whisper.cpp JFK sample:
//!
//! ```text
//! curl -sL -o /tmp/jfk.wav \
//!   https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
//! SAPIENT_TEST_WAV=/tmp/jfk.wav \
//!   cargo test -p sapient-generate --test transcribe_e2e -- --ignored --nocapture
//! ```
//!
//! With the JFK clip the transcript must contain "country" (… "ask not what your
//! country can do for you" …). With any other clip it just asserts non-empty.

use sapient_generate::TranscribePipeline;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads whisper-tiny and needs SAPIENT_TEST_WAV"]
async fn transcribes_known_clip() {
    let Ok(wav) = std::env::var("SAPIENT_TEST_WAV") else {
        eprintln!("SAPIENT_TEST_WAV not set — skipping (set it to a 16 kHz speech WAV)");
        return;
    };

    let pipe = TranscribePipeline::from_pretrained("whisper-tiny")
        .await
        .expect("load whisper-tiny");
    let text = pipe
        .transcribe(&wav)
        .await
        .expect("transcribe")
        .to_lowercase();

    eprintln!("transcript: {text}");
    assert!(!text.trim().is_empty(), "empty transcript");

    // The JFK sample is the canonical clip; assert a distinctive word when used.
    if wav.contains("jfk") {
        assert!(
            text.contains("country"),
            "expected the JFK quote (… 'country' …), got: {text}"
        );
    }
}
