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
    // Embedding tables are commonly stored in F16/BF16; convert on the fly.
    let w_cow = weight.to_f32_cow();
    let w = w_cow.as_ref();
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
    // weight is [out, in] (PyTorch nn.Linear layout); matmul_nt computes x @ weightᵀ
    // directly, honouring the layout and any F16/BF16 weight dtype.
    let y2d = map_err(matmul::matmul_nt(&x2d, weight))?;
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

/// Update the pre-allocated KV cache in place and return a view of length `seq_len + new_seq`.
pub fn update_kv_cache(
    cache: &mut Tensor,
    current_seq_len: usize,
    new_k: &Tensor,
) -> Result<Tensor> {
    let cd = cache.shape().dims();
    let nd = new_k.shape().dims();

    if cd.len() != 4 || nd.len() != 4 {
        anyhow::bail!("update_kv_cache expects 4-D tensors");
    }
    if cd[0] != nd[0] || cd[1] != nd[1] || cd[3] != nd[3] {
        anyhow::bail!("update_kv_cache shape mismatch");
    }

    let max_seq = cd[2];
    let new_seq = nd[2];

    if new_seq > max_seq {
        anyhow::bail!("new tokens {} exceeds max cache size {}", new_seq, max_seq);
    }

    let mut total_seq = current_seq_len + new_seq;
    let shift = total_seq.saturating_sub(max_seq);

    let (b_sz, h, hd) = (cd[0], cd[1], cd[3]);
    let new_k_slice = new_k.as_f32_slice();
    let cache_strides = cache.strides().to_vec();

    {
        let cache_slice = cache.as_f32_slice_mut()?;

        // If we need to shift, move existing elements left
        if shift > 0 {
            let keep_seq = current_seq_len - shift;
            for bi in 0..b_sz {
                for hi in 0..h {
                    let cache_base = bi * cache_strides[0] + hi * cache_strides[1];
                    for si in 0..keep_seq {
                        let src_idx = cache_base + (si + shift) * cache_strides[2];
                        let dst_idx = cache_base + si * cache_strides[2];
                        cache_slice.copy_within(src_idx..src_idx + hd, dst_idx);
                    }
                }
            }
        }

        // Now append the new tokens
        let insert_pos = if shift > 0 {
            current_seq_len - shift
        } else {
            current_seq_len
        };
        for bi in 0..b_sz {
            for hi in 0..h {
                let cache_base =
                    bi * cache_strides[0] + hi * cache_strides[1] + insert_pos * cache_strides[2];
                let new_base = ((bi * h + hi) * new_seq) * hd; // new_k is assumed contiguous from split_heads

                for si in 0..new_seq {
                    let c_idx = cache_base + si * cache_strides[2];
                    let n_idx = new_base + si * hd;

                    // Copy head_dim elements
                    cache_slice[c_idx..c_idx + hd].copy_from_slice(&new_k_slice[n_idx..n_idx + hd]);
                }
            }
        }
    }

    if shift > 0 {
        total_seq = max_seq;
    }

    // Return a sliced view of the cache from 0 to total_seq
    cache
        .slice_axis(2, 0, total_seq)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn apply_rope_positions(x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
    map_err(rope::apply_rope(x, positions, base))
}

/// RoPE applied to only the first `rotary_dim` channels (Phi partial rotary).
pub fn apply_rope_partial(
    x: &Tensor,
    positions: &[usize],
    base: f32,
    rotary_dim: usize,
) -> Result<Tensor> {
    map_err(rope::apply_rope_partial(x, positions, base, rotary_dim))
}

/// Add a per-feature bias `[n]` broadcast over the last dimension of `y`
/// (shape `[.., n]`). `y` must be F32; `bias` may be F16/BF16.
pub fn add_bias_last_dim(y: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = y.shape().dims().to_vec();
    let n = *dims.last().ok_or_else(|| anyhow::anyhow!("empty tensor"))?;
    let bias_cow = bias.to_f32_cow();
    let b = bias_cow.as_ref();
    if b.len() != n {
        anyhow::bail!("bias length {} does not match last dim {n}", b.len());
    }
    let mut data = y.as_f32_slice().to_vec();
    for (i, v) in data.iter_mut().enumerate() {
        *v += b[i % n];
    }
    map_err(Tensor::from_f32(&data, Shape::new(dims)))
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
    // lm_head is [vocab, hidden]; matmul_nt computes h_last @ lm_headᵀ directly.
    let logits = map_err(matmul::matmul_nt(&h_last, lm_head))?;
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
