//! Shared tensor ops for transformer forward passes.

use anyhow::Result;
use sapient_backends_cpu::kernels::{self, attention, layernorm, matmul, rope};
use sapient_core::error::SapientError;
use sapient_core::{Shape, Tensor};

fn map_err<T>(result: std::result::Result<T, SapientError>) -> Result<T> {
    result.map_err(|e| anyhow::anyhow!("{e}"))
}

/// Gather token embeddings: weight `[vocab, hidden]`, ids `[seq]` → `[1, seq, hidden]`.
pub fn embed_tokens(weight: &Tensor, input_ids: &[u32]) -> Result<Tensor> {
    let hidden = weight.shape().dims()[1];
    let seq_len = input_ids.len();
    let w = weight.as_f32_slice();
    let mut out = vec![0.0f32; seq_len * hidden];

    for (i, &id) in input_ids.iter().enumerate() {
        let row = id as usize * hidden;
        if row + hidden > w.len() {
            anyhow::bail!("token id {id} out of vocab range");
        }
        out[i * hidden..(i + 1) * hidden].copy_from_slice(&w[row..row + hidden]);
    }

    Tensor::from_f32(&out, Shape::new([1, seq_len, hidden])).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Linear on 3-D activations: `[1, seq, in] @ W^T` where W is `[out, in]`.
pub fn linear_3d(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        anyhow::bail!("linear_3d expects [batch, seq, hidden]");
    }
    let (batch, seq, in_dim) = (dims[0], dims[1], dims[2]);
    let w_dims = weight.shape().dims();
    if w_dims.len() != 2 {
        anyhow::bail!("linear weight must be 2-D");
    }
    let out_dim = w_dims[0];
    if w_dims[1] != in_dim {
        anyhow::bail!("linear weight in_dim mismatch: {} vs {in_dim}", w_dims[1]);
    }

    let x2d = map_err(x.reshape(vec![batch * seq, in_dim]))?;
    let wt = weight.t().map_err(|e| anyhow::anyhow!("{e}"))?;
    let y2d = map_err(matmul::matmul(&x2d, &wt))?;
    map_err(y2d.reshape(vec![batch, seq, out_dim]))
}

/// Reshape `[1, seq, n_heads * head_dim]` → `[1, n_heads, seq, head_dim]`.
pub fn split_heads(x: &Tensor, n_heads: usize, head_dim: usize) -> Result<Tensor> {
    let seq = x.shape().dims()[1];
    permute(
        &map_err(x.reshape(vec![1, seq, n_heads, head_dim]))?,
        &[0, 2, 1, 3],
    )
}

/// Merge heads back: `[1, n_heads, seq, head_dim]` → `[1, seq, n_heads * head_dim]`.
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims();
    let (n_heads, seq, head_dim) = (d[1], d[2], d[3]);
    permute(x, &[0, 2, 1, 3])?
        .reshape(vec![1, seq, n_heads * head_dim])
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn permute(x: &Tensor, order: &[usize]) -> Result<Tensor> {
    let dims = x.shape().dims();
    if order.len() != dims.len() {
        anyhow::bail!("permute rank mismatch");
    }
    let new_dims: Vec<usize> = order.iter().map(|&i| dims[i]).collect();
    let src = x.as_f32_slice();
    let mut out = vec![0.0f32; src.len()];

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        dims: &[usize],
        order: &[usize],
        src: &[f32],
        out: &mut [f32],
        src_strides: &[usize],
        dst_strides: &[usize],
        idx: &mut [usize],
        depth: usize,
    ) {
        if depth == dims.len() {
            let src_off: usize = idx
                .iter()
                .zip(src_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            let dst_off: usize = order
                .iter()
                .enumerate()
                .map(|(dst_ax, &src_ax)| idx[src_ax] * dst_strides[dst_ax])
                .sum();
            out[dst_off] = src[src_off];
            return;
        }
        for i in 0..dims[depth] {
            idx[depth] = i;
            recurse(
                dims,
                order,
                src,
                out,
                src_strides,
                dst_strides,
                idx,
                depth + 1,
            );
        }
    }

    let src_strides = strides_for(dims);
    let dst_strides = strides_for(&new_dims);
    let mut idx = vec![0usize; dims.len()];
    recurse(
        dims,
        order,
        src,
        &mut out,
        &src_strides,
        &dst_strides,
        &mut idx,
        0,
    );
    Tensor::from_f32(&out, Shape::new(new_dims)).map_err(|e| anyhow::anyhow!("{e}"))
}

fn strides_for(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Concatenate two 4-D tensors along the sequence axis (dim 2).
pub fn concat_seq(k1: &Tensor, k2: &Tensor) -> Result<Tensor> {
    let d1 = k1.shape().dims();
    let d2 = k2.shape().dims();
    if d1.len() != 4 || d2.len() != 4 {
        anyhow::bail!("concat_seq expects 4-D tensors");
    }
    if d1[0] != d2[0] || d1[1] != d2[1] || d1[3] != d2[3] {
        anyhow::bail!("concat_seq shape mismatch");
    }
    let new_seq = d1[2] + d2[2];
    let mut out = vec![0.0f32; d1[0] * d1[1] * new_seq * d1[3]];
    let a = k1.as_f32_slice();
    let b = k2.as_f32_slice();
    let (b_sz, h, _, hd) = (d1[0], d1[1], d1[2], d1[3]);
    let s1 = d1[2];
    let s2 = d2[2];

    for bi in 0..b_sz {
        for hi in 0..h {
            let dst_base = ((bi * h + hi) * new_seq) * hd;
            let a_base = ((bi * h + hi) * s1) * hd;
            let b_base = ((bi * h + hi) * s2) * hd;
            out[dst_base..dst_base + s1 * hd].copy_from_slice(&a[a_base..a_base + s1 * hd]);
            out[dst_base + s1 * hd..dst_base + (s1 + s2) * hd]
                .copy_from_slice(&b[b_base..b_base + s2 * hd]);
        }
    }

    Tensor::from_f32(&out, Shape::new([b_sz, h, new_seq, hd])).map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn apply_rope_positions(x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
    map_err(rope::apply_rope(x, positions, base))
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    map_err(layernorm::rms_norm(x, Some(weight), eps))
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    map_err(layernorm::layer_norm(x, Some(weight), bias, -1, eps))
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::silu(x))
}

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::gelu(x))
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::add(a, b))
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    map_err(kernels::elementwise::mul(a, b))
}

pub fn gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_kv_heads: usize,
    causal: bool,
) -> Result<Tensor> {
    let mask = if causal {
        let sq = q.shape().dims()[2];
        let sk = k.shape().dims()[2];
        Some(attention::causal_mask(sq, sk))
    } else {
        None
    };
    map_err(attention::scaled_dot_product_attention(
        q,
        k,
        v,
        mask.as_ref(),
        None,
        n_kv_heads,
    ))
}

pub fn logits_from_hidden(hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>> {
    // hidden: [1, seq, hidden], take last position
    let dims = hidden.shape().dims();
    let hidden_size = dims[2];
    let seq = dims[1];
    let h = hidden.as_f32_slice();
    let last = &h[(seq - 1) * hidden_size..seq * hidden_size];
    let h_last =
        Tensor::from_f32(last, Shape::new([1, hidden_size])).map_err(|e| anyhow::anyhow!("{e}"))?;
    let wt = lm_head.t().map_err(|e| anyhow::anyhow!("{e}"))?;
    let logits = map_err(matmul::matmul(&h_last, &wt))?;
    Ok(logits.as_f32_slice().to_vec())
}

pub fn mean_pool_hidden(hidden: &Tensor) -> Result<Vec<f32>> {
    let dims = hidden.shape().dims();
    let (seq, hidden_size) = (dims[1], dims[2]);
    let h = hidden.as_f32_slice();
    let mut out = vec![0.0f32; hidden_size];
    for t in 0..seq {
        for i in 0..hidden_size {
            out[i] += h[t * hidden_size + i];
        }
    }
    let n = seq as f32;
    for v in &mut out {
        *v /= n;
    }
    Ok(out)
}
