//! End-to-end `KokoroTts` test: plain text → misaki G2P → Kokoro-82M → 24 kHz
//! audio. Ignored by default (needs the converted weights). Run with:
//!   SAPIENT_KOKORO_DIR=~/.cache/sapient-kokoro \
//!   cargo test -p sapient-generate --test kokoro_tts_e2e -- --ignored --nocapture

use std::path::PathBuf;

use sapient_generate::converse::Tts;
use sapient_generate::{KokoroTts, KOKORO_REPO};

fn kokoro_dir() -> PathBuf {
    std::env::var("SAPIENT_KOKORO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/sapient-kokoro")
        })
}

fn write_wav(path: &str, samples: &[f32], sr: u32) {
    use std::io::Write;
    let mut data = Vec::new();
    for &s in samples {
        data.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    let mut f = std::fs::File::create(path).unwrap();
    let n = data.len() as u32;
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(36 + n).to_le_bytes()).unwrap();
    f.write_all(b"WAVEfmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&sr.to_le_bytes()).unwrap();
    f.write_all(&(sr * 2).to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&data).unwrap();
}

#[test]
#[ignore = "needs converted Kokoro weights via SAPIENT_KOKORO_DIR"]
fn kokoro_tts_synthesizes_text() {
    let tts = KokoroTts::from_dir(&kokoro_dir()).expect("load KokoroTts");
    let text = "Hello there. How are you today?";

    let phonemes = tts.phonemize(text).expect("g2p");
    println!("text: {text:?}\nphonemes: {phonemes:?}");
    assert!(!phonemes.is_empty(), "G2P produced no phonemes");

    // Warm up, then time the synthesis to report the real-time factor (RTF).
    let _ = tts.synthesize(text).expect("warmup");
    let t0 = std::time::Instant::now();
    let audio = tts.synthesize(text).expect("synthesize");
    let synth_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let sr = tts.sample_rate();
    let secs = audio.len() as f32 / sr as f32;
    let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt();
    let rtf = (synth_ms / 1000.0) / secs;
    println!(
        "audio: {} samples ({secs:.2}s @ {sr} Hz), rms {rms:.4}",
        audio.len()
    );
    println!(
        "SYNTHESIS: {synth_ms:.0} ms for {secs:.2}s audio → RTF {rtf:.3} ({:.1}× real-time)",
        1.0 / rtf
    );

    let _ = std::fs::create_dir_all("/tmp/kokoro_out");
    write_wav("/tmp/kokoro_out/tts_hello_there.wav", &audio, sr);

    assert_eq!(sr, 24_000);
    assert!(secs > 1.0 && secs < 6.0, "unexpected duration {secs}s");
    assert!(rms > 0.01 && rms < 0.5, "unexpected rms {rms}");
}

/// Out-of-the-box path: download the converted mirror from the Hub (no local
/// dir) and synthesize. Needs network; unset `SAPIENT_KOKORO_DIR` to exercise
/// the real download.
#[tokio::test]
#[ignore = "network: downloads the Kokoro safetensors mirror"]
async fn kokoro_tts_from_pretrained_downloads_and_speaks() {
    std::env::remove_var("SAPIENT_KOKORO_DIR");
    let tts = KokoroTts::from_pretrained(KOKORO_REPO)
        .await
        .expect("download + load Kokoro from mirror");
    let audio = tts
        .synthesize("Testing the downloaded model.")
        .expect("synthesize");
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    println!("from_pretrained: {} samples ({secs:.2}s)", audio.len());
    assert!(secs > 0.8, "too short: {secs}s");
    write_wav(
        "/tmp/kokoro_out/tts_from_pretrained.wav",
        &audio,
        tts.sample_rate(),
    );
}
