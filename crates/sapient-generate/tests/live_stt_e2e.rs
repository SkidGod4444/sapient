//! End-to-end gate for the Phase-10.1 incremental transcriber ([`LiveStt`]):
//! feed a speech clip in growing snapshots (as the live loop does while the
//! user is still talking), `settle`, and require the incremental transcript to
//! cover the whole clip and agree with a full-pass transcription on the key
//! word.
//!
//! Ignored by default (downloads `whisper-tiny`, needs a local WAV):
//! ```sh
//! curl -sL -o /tmp/jfk.wav \
//!   https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
//! SAPIENT_TEST_WAV=/tmp/jfk.wav \
//!   cargo test -p sapient-generate --test live_stt_e2e -- --ignored --nocapture
//! ```

use std::time::Duration;

use sapient_generate::LiveStt;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads whisper-tiny and needs SAPIENT_TEST_WAV"]
async fn incremental_transcript_matches_full_pass() {
    let wav = std::env::var("SAPIENT_TEST_WAV").expect("set SAPIENT_TEST_WAV to a speech clip");
    let stt = sapient_generate::TranscribePipeline::from_pretrained("openhorizon/whisper-tiny")
        .await
        .expect("loading whisper-tiny");
    let samples = sapient_audio::io::load_audio(std::path::Path::new(&wav), 16_000)
        .expect("loading test WAV");

    let live = LiveStt::for_transcriber(stt.clone(), Default::default());

    // Feed growing snapshots every ~0.5 s of audio, like the mic loop does.
    let step = 8_000usize; // 0.5 s @ 16 kHz
    let mut end = step;
    while end < samples.len() {
        live.feed(samples[..end].to_vec());
        end += step;
    }
    live.feed(samples.to_vec());

    let t = std::time::Instant::now();
    let (text, covered) = live.settle(Duration::from_secs(300));
    println!(
        "settled in {:?}: covered {covered}/{} — {text:?}",
        t.elapsed(),
        samples.len()
    );
    assert_eq!(
        covered,
        samples.len(),
        "final snapshot must be the covered one"
    );
    assert!(
        text.to_lowercase().contains("country"),
        "incremental transcript should contain the key word: {text:?}"
    );

    // Full-pass reference agrees.
    let full = stt
        .transcribe_samples(&samples, &Default::default())
        .expect("full pass");
    assert!(
        full.to_lowercase().contains("country"),
        "full pass: {full:?}"
    );

    // After reset, stale state is gone.
    live.reset();
    let (empty, zero) = live.settle(Duration::from_millis(50));
    assert!(empty.is_empty());
    assert_eq!(zero, 0);
}
