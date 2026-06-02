//! SNAC neural-audio-codec **decoder** building blocks (Phase 6d, LM-codec TTS).
//!
//! The LM (Orpheus/OuteTTS, a Llama-3.2 run by `LlamaForward`) emits SNAC codec
//! token ids; this module turns those tokens back into a waveform. SNAC's 24 kHz
//! decoder is fully convolutional (no attention/LSTM/ISTFT): RVQ codebook lookup
//! → conv stack with `conv_transpose1d` upsampling + `snake` activations
//! (see [`super::conv`]) → tanh → 24 kHz samples.
//!
//! This file currently provides the load-time helpers that are independently
//! verifiable without the model weights; the full decoder forward is wired once
//! the SNAC weights are converted to safetensors (the released checkpoint is a
//! `.pth` pickle SAPIENT can't load directly).

use anyhow::{anyhow, Result};
use sapient_core::Tensor;

/// Fold PyTorch `weight_norm` parameters into a plain weight tensor at load time
/// (so the decoder convs need no runtime renormalization).
///
/// `weight_norm` factorizes `w = g · v / ‖v‖`, with the norm taken over every
/// dimension except the output-channel axis (dim 0, the default). `v` has the
/// full weight shape `[out, …]`; `g` has one magnitude per output channel
/// (shape `[out, 1, …]` or `[out]`). Returns `w` with `v`'s shape, where each
/// output row has L2 norm exactly `g[o]`.
pub fn weight_norm_fold(v: &Tensor, g: &Tensor) -> Result<Tensor> {
    let dims = v.shape().dims().to_vec();
    let out = *dims
        .first()
        .ok_or_else(|| anyhow!("weight_norm_fold: empty weight"))?;
    let total = dims.iter().product::<usize>();
    if out == 0 || total % out != 0 {
        anyhow::bail!("weight_norm_fold: bad shape {dims:?}");
    }
    let per = total / out; // elements per output channel

    let vv = v.to_f32_vec();
    let gv = g.to_f32_vec();
    if gv.len() != out {
        anyhow::bail!(
            "weight_norm_fold: g has {} magnitudes, expected {out}",
            gv.len()
        );
    }

    let mut w = vec![0.0f32; total];
    for o in 0..out {
        let row = &vv[o * per..(o + 1) * per];
        let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let scale = if norm > 0.0 { gv[o] / norm } else { 0.0 };
        for (i, &x) in row.iter().enumerate() {
            w[o * per + i] = x * scale;
        }
    }
    Tensor::from_f32(&w, v.shape().clone()).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_gives_rows_with_norm_g() {
        // v: [out=3, in=2, k=4]; g: [out, 1, 1].
        let (out, inn, k) = (3usize, 2usize, 4usize);
        let per = inn * k;
        let v: Vec<f32> = (0..out * per)
            .map(|i| (i as f32 * 0.31 + 0.5).sin())
            .collect();
        let g: Vec<f32> = vec![0.5, 1.0, 2.0];
        let vt = Tensor::from_f32(&v, vec![out, inn, k]).unwrap();
        let gt = Tensor::from_f32(&g, vec![out, 1, 1]).unwrap();

        let w = weight_norm_fold(&vt, &gt).unwrap();
        assert_eq!(w.shape().dims(), &[out, inn, k]);
        let wv = w.as_f32_slice();
        for o in 0..out {
            let norm = wv[o * per..(o + 1) * per]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt();
            // ‖w[o]‖ must equal g[o] (the weight_norm invariant).
            assert!(
                (norm - g[o]).abs() < 1e-5,
                "row {o}: norm {norm} != g {}",
                g[o]
            );
        }
    }

    #[test]
    fn zero_direction_row_folds_to_zero() {
        let vt = Tensor::from_f32(&[0.0, 0.0, 1.0, 1.0], vec![2, 1, 2]).unwrap();
        let gt = Tensor::from_f32(&[3.0, 3.0], vec![2, 1, 1]).unwrap();
        let w = weight_norm_fold(&vt, &gt).unwrap();
        let wv = w.as_f32_slice();
        assert_eq!(&wv[0..2], &[0.0, 0.0]); // zero v → zero w (no div-by-zero)
    }
}
