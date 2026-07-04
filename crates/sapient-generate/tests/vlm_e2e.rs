//! Golden end-to-end gate for the SmolVLM path (Roadmap 12.1: "golden-output
//! test with a fixed image"): a synthetic solid-red image must be identified
//! as red, greedily, end to end (SigLIP tower → pixel shuffle → connector →
//! embedding splice → SmolLM2 decode).
//!
//! Ignored by default (downloads SmolVLM-256M-Instruct, ~500 MB):
//! `cargo test -p sapient-generate --test vlm_e2e --release -- --ignored --nocapture`

use sapient_generate::VlmPipeline;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads SmolVLM-256M"]
async fn solid_red_image_is_identified() {
    let dir = std::env::temp_dir().join("sapient-vlm-e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("red.png");
    let img = image::RgbImage::from_pixel(512, 512, image::Rgb([216, 32, 32]));
    img.save(&path).expect("writing fixture image");

    let mut vlm = VlmPipeline::from_pretrained("smolvlm-256m")
        .await
        .expect("loading smolvlm-256m");
    let answer = vlm
        .answer(
            &path,
            "What is the dominant color of this image? Answer with one word.",
            16,
        )
        .expect("answering");
    println!("answer: {answer:?}");
    assert!(
        answer.to_lowercase().contains("red"),
        "expected 'red' in {answer:?}"
    );
}
