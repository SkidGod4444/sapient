//! Measured negative result, kept as a regression record: windowed (rolling)
//! decoding of the Kokoro ISTFTNet decoder is NOT viable without a
//! streaming-norm redesign. Window bookkeeping + analytic NSF phase carry are
//! byte-exact (halo = whole utterance → max diff 0.000000), but AdaIN's
//! InstanceNorm statistics are GLOBAL over the time axis, so mid-utterance
//! windows diverge from the full decode by 0.45 / 0.34 / 0.20 max-diff at
//! halo 16 / 32 / 64 (measured 2026-07-04 on the real model) — a slow decay
//! that is statistics, not receptive field. See docs/DUPLEX_SPIKE.md.

use sapient_generate::KokoroTts;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads the Kokoro weights mirror"]
async fn window_bookkeeping_vs_locality() {
    let tts = KokoroTts::from_default().await.expect("loading kokoro");
    let text = "The quick brown fox jumps over the lazy dog, while morning light \
                spreads across the quiet valley, and the birds begin to sing.";
    let inp = tts.prepare_stream(text).expect("backbone");
    let t = inp.t;
    println!("t = {t} frames");
    let full = tts.decode_prefix(&inp, t).expect("full decode");
    println!(
        "full len = {} ({} per frame avg)",
        full.len(),
        full.len() as f64 / t as f64
    );

    let (a, b) = (t / 3, t / 3 + 64.min(t / 3));
    // 1. Bookkeeping: halo covers everything → window IS the full decode, sliced.
    let w_all = tts.model().decode_window(&inp, a, b, t).expect("halo=t");
    let spf = full.len() as f64 / t as f64;
    let lo = (a as f64 * spf) as usize;
    let want = &full[lo..lo + w_all.len().min(full.len() - lo)];
    let max_bk = w_all
        .iter()
        .zip(want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "bookkeeping (halo=t) max diff vs full slice: {max_bk:.6} (len {} vs {})",
        w_all.len(),
        want.len()
    );

    // 2. Locality: spike-margin halo.
    let w16 = tts.model().decode_window(&inp, a, b, 16).expect("halo=16");
    let max_loc = w16
        .iter()
        .zip(want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "locality  (halo=16) max diff vs full slice: {max_loc:.6} (len {})",
        w16.len()
    );

    // 3. Larger halo for the trend.
    for halo in [32usize, 64] {
        let wx = tts.model().decode_window(&inp, a, b, halo).expect("halo");
        let d = wx
            .iter()
            .zip(want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("locality  (halo={halo}) max diff: {d:.6}");
    }
}
