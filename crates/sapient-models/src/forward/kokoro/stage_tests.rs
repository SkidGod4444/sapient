//! Stage-by-stage coherence tests against a PyTorch Kokoro reference fixture
//! (`tests/fixtures/kokoro_hello.safetensors`, every intermediate dumped by
//! `scripts/`-generated ground truth for phonemes "həlˈoʊ" + voice af_heart).
//!
//! These are `#[ignore]` because they need the converted Kokoro weights (~327 MB,
//! not committed). Point `SAPIENT_KOKORO_DIR` at the output of
//! `scripts/convert_kokoro_to_safetensors.py` (default `~/.cache/sapient-kokoro`)
//! and run e.g. `cargo test -p sapient-models --lib kokoro -- --ignored`.

use std::collections::HashMap;
use std::path::PathBuf;

use sapient_core::Tensor;

use super::albert::albert_encode;
use super::loader::{load_from_dir, KokoroAssets};
use super::model::{text_encode, KokoroModel};
use super::ops::linear2d;
use super::predictor::{f0_n_train, predict_prosody};

fn kokoro_dir() -> PathBuf {
    std::env::var("SAPIENT_KOKORO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/sapient-kokoro")
        })
}

fn fixture() -> HashMap<String, Tensor> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kokoro_hello.safetensors");
    sapient_io::load_safetensors(&p).expect("load kokoro fixture")
}

fn assets() -> KokoroAssets {
    load_from_dir(&kokoro_dir()).expect("load kokoro weights (set SAPIENT_KOKORO_DIR)")
}

fn input_ids(fx: &HashMap<String, Tensor>) -> Vec<u32> {
    // stored as int64; load_safetensors surfaces it — read via to_f32_vec round-trip.
    fx["input_ids"]
        .to_f32_vec()
        .iter()
        .map(|&v| v as u32)
        .collect()
}

/// max abs error between a flat slice and a fixture tensor.
fn max_err(got: &[f32], want: &Tensor) -> f32 {
    let w = want.to_f32_vec();
    assert_eq!(
        got.len(),
        w.len(),
        "length mismatch {} vs {}",
        got.len(),
        w.len()
    );
    got.iter()
        .zip(w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "needs converted Kokoro weights via SAPIENT_KOKORO_DIR"]
fn stage_albert_matches_reference() {
    let fx = fixture();
    let a = assets();
    let ids = input_ids(&fx);
    let out = albert_encode(&a.weights, &ids, &a.config.plbert).unwrap();
    let err = max_err(&out, &fx["bert_dur"]);
    println!("ALBERT bert_dur max_err = {err}");
    assert!(err < 1e-3, "ALBERT bert_dur max_err {err} too high");
}

/// Transpose `[r,c]` → `[c,r]`.
fn transpose(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            o[j * r + i] = x[i * c + j];
        }
    }
    o
}

/// bert_encoder Linear(768→512) on bert_dur, transposed to `[512, L]` (`d_en`).
fn compute_d_en(a: &KokoroAssets, fx: &HashMap<String, Tensor>) -> (Vec<f32>, usize) {
    let bert = fx["bert_dur"].to_f32_vec();
    let l = fx["bert_dur"].shape().dims()[0];
    let h = a.config.hidden_dim;
    let bw = a.weights["bert_encoder.weight"].to_f32_vec();
    let bb = a.weights["bert_encoder.bias"].to_f32_vec();
    let be = linear2d(&bert, l, a.config.plbert.hidden_size, &bw, Some(&bb), h); // [L,512]
    (transpose(&be, l, h), l) // [512, L]
}

#[test]
#[ignore = "needs converted Kokoro weights via SAPIENT_KOKORO_DIR"]
fn stage_bert_encoder_matches_reference() {
    let fx = fixture();
    let a = assets();
    let (d_en, _l) = compute_d_en(&a, &fx);
    let err = max_err(&d_en, &fx["d_en"]);
    println!("bert_encoder d_en max_err = {err}");
    assert!(err < 1e-3, "d_en max_err {err}");
}

#[test]
#[ignore = "needs converted Kokoro weights via SAPIENT_KOKORO_DIR"]
fn stage_prosody_matches_reference() {
    let fx = fixture();
    let a = assets();
    let (d_en, l) = compute_d_en(&a, &fx);
    let ref_s = fx["ref_s"].to_f32_vec(); // [256]
    let style = &ref_s[128..]; // predictor style
    let p = predict_prosody(&a.weights, &a.config, &d_en, l, style, 1.0).unwrap();

    // durations
    let want_dur: Vec<usize> = fx["pred_dur"]
        .to_f32_vec()
        .iter()
        .map(|&v| v as usize)
        .collect();
    println!("pred_dur got {:?} want {:?}", p.pred_dur, want_dur);
    assert_eq!(p.pred_dur, want_dur, "duration mismatch");

    // en (length-regulated features)
    let en_err = max_err(&p.en.to_f32_vec(), &fx["en"]);
    println!("en max_err = {en_err}");
    assert!(en_err < 5e-3, "en max_err {en_err}");

    // F0 / N curves
    let (f0, n) = f0_n_train(&a.weights, &a.config, &p.en, style).unwrap();
    let f0_err = max_err(&f0, &fx["F0_pred"]);
    let n_err = max_err(&n, &fx["N_pred"]);
    println!("F0 max_err = {f0_err}, N max_err = {n_err}");
    assert!(f0_err < 1e-1, "F0 max_err {f0_err}");
    assert!(n_err < 1e-2, "N max_err {n_err}");
}

#[test]
#[ignore = "needs converted Kokoro weights via SAPIENT_KOKORO_DIR"]
fn stage_text_encoder_matches_reference() {
    let fx = fixture();
    let a = assets();
    let ids = input_ids(&fx);
    let t_en = text_encode(&a.weights, &ids, &a.config).unwrap(); // [512, L]
    let err = max_err(&t_en, &fx["t_en"]);
    println!("text_encoder t_en max_err = {err}");
    assert!(err < 1e-3, "t_en max_err {err}");
}

/// Pearson correlation between two equal-length signals.
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut da = 0.0f32;
    let mut db = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        num += (x - ma) * (y - mb);
        da += (x - ma) * (x - ma);
        db += (y - mb) * (y - mb);
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}

#[test]
#[ignore = "needs converted Kokoro weights via SAPIENT_KOKORO_DIR"]
fn stage_full_audio_correlates_with_reference() {
    let fx = fixture();
    let dir = kokoro_dir();
    let model = KokoroModel::from_dir(&dir).expect("load model");
    let ids = input_ids(&fx);
    let ref_s = fx["ref_s"].to_f32_vec();
    let audio = model.synthesize_ids(&ids, &ref_s, 1.0).unwrap();

    let want = fx["audio"].to_f32_vec();
    println!("audio len got {} want {}", audio.len(), want.len());
    assert_eq!(audio.len(), want.len(), "audio length mismatch");

    // The NSF source omits training-time noise + random initial phase (which the
    // reference includes), so the waveform is perceptually identical but not
    // bit-exact — validate by correlation + matched energy.
    // Write both first for an audible/spectral A/B check.
    let _ = std::fs::create_dir_all("/tmp/kokoro_out");
    write_wav("/tmp/kokoro_out/rust_hello.wav", &audio);
    write_wav("/tmp/kokoro_out/ref_hello.wav", &want);

    // Raw-sample correlation is meaningless for speech with differing excitation
    // phase (NSF noise/phase omitted); validate the short-time energy envelope,
    // which captures prosody/timing and is phase-robust.
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let (ra, rw) = (rms(&audio), rms(&want));
    let ea = energy_envelope(&audio, 512, 256);
    let ew = energy_envelope(&want, 512, 256);
    let env_corr = correlation(&ea, &ew);
    println!("energy-envelope corr = {env_corr:.4}, rms got {ra:.4} want {rw:.4}");
    assert!(
        env_corr > 0.95,
        "energy-envelope correlation {env_corr} too low"
    );
    assert!(
        (ra / rw - 1.0).abs() < 0.25,
        "audio energy ratio off: {ra} vs {rw}"
    );
}

/// Frame-RMS energy envelope (window/hop in samples).
fn energy_envelope(x: &[f32], win: usize, hop: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + win <= x.len() {
        let e = (x[i..i + win].iter().map(|v| v * v).sum::<f32>() / win as f32).sqrt();
        out.push(e);
        i += hop;
    }
    out
}

fn write_wav(path: &str, samples: &[f32]) {
    use std::io::Write;
    let sr = 24000u32;
    let mut data = Vec::new();
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        data.extend_from_slice(&v.to_le_bytes());
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
