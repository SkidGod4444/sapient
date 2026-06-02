//! SNAC decoder coherence test (ignored — needs the converted codec weights).
//!
//! Validates the pure-Rust SNAC decoder (`forward::snac::SnacDecoder`) against a
//! torch reference. The fixture (`fixtures/snac_decode.json`) holds RVQ codes and
//! the reference waveform produced by the Python `snac` package with the noise
//! block disabled (the decoder's only stochastic part) — matching the
//! deterministic Rust path. The folded weights are produced once by
//! `scripts/convert_snac_to_safetensors.py`; point `SAPIENT_SNAC_DIR` at the
//! output directory (containing `snac.safetensors` + `config.json`):
//!
//! ```text
//! python scripts/convert_snac_to_safetensors.py --out /tmp/snac_24khz
//! SAPIENT_SNAC_DIR=/tmp/snac_24khz \
//!   cargo test -p sapient-models --test snac_coherence -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use sapient_hub::snac_config::SnacConfig;
use sapient_models::forward::{normalize_snac_weights, SnacDecoder};
use sapient_models::weights;

#[test]
#[ignore = "needs converted SNAC weights at $SAPIENT_SNAC_DIR"]
fn snac_decode_matches_reference() {
    let Ok(dir) = std::env::var("SAPIENT_SNAC_DIR") else {
        eprintln!("SAPIENT_SNAC_DIR not set — skipping (run convert_snac_to_safetensors.py first)");
        return;
    };
    let dir = PathBuf::from(dir);
    let cfg = SnacConfig::from_config_file(&dir.join("config.json")).expect("snac config");
    // Accept either the converted torch weights (`snac.safetensors`) or the
    // ungated `mlx-community/snac_24khz` mirror (`model.safetensors`);
    // `normalize_snac_weights` adapts the layout of the latter.
    let st = ["model.safetensors", "snac.safetensors"]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.exists())
        .expect("a SNAC safetensors file in $SAPIENT_SNAC_DIR");
    let raw = weights::load_hf_weights(&[st]).expect("snac weights");
    let w = normalize_snac_weights(raw).expect("normalize snac weights");
    let dec = SnacDecoder::from_weights(cfg, w);

    let fx: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/snac_decode.json")).expect("fixture json");
    let codes: Vec<Vec<u32>> = fx["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lvl| {
            lvl.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        })
        .collect();
    let reference: Vec<f32> = fx["reference"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let out = dec.decode(&codes).expect("decode");
    assert_eq!(out.len(), reference.len(), "waveform length mismatch");

    let max_err = out
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("SNAC decode max_err vs torch reference: {max_err:.2e}");
    assert!(
        max_err < 2e-2,
        "SNAC decode diverged from reference: max_err={max_err}"
    );
}
