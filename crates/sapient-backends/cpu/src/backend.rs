//! `ExecutionBackend` trait and `CpuBackend` implementation.

use std::collections::HashMap;

use tracing::{debug, instrument};

use sapient_core::buffer::{BufferHandle, CpuBuffer};
use sapient_core::error::{Result, SapientError};
use sapient_core::{DType, Tensor};
use sapient_ir::graph::Graph;
use sapient_ir::node::{Node, NodeId};
use sapient_ir::op::OpType;

use crate::kernels;
use crate::pool::PoolAllocator;

// ── ExecutionBackend trait ────────────────────────────────────────────────────

/// The unified backend interface every hardware target must implement.
///
/// Backends may be selected at runtime via `Box<dyn ExecutionBackend>` or at
/// compile time via generics.
pub trait ExecutionBackend: Send + Sync {
    /// Short name for logging / CLI display.
    fn name(&self) -> &str;

    /// Allocate an uninitialised buffer for the given shape and dtype.
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<BufferHandle>;

    /// Execute the graph, returning output tensors in the order of
    /// `graph.outputs`.
    fn execute(&self, graph: &Graph, inputs: HashMap<String, Tensor>) -> Result<Vec<Tensor>>;

    /// Whether this backend can execute the given op natively.
    fn supports_op(&self, op: &OpType) -> bool;

    /// True if this backend is available on the current system.
    fn is_available() -> bool
    where
        Self: Sized,
    {
        true
    }
}

// ── CpuBackend ────────────────────────────────────────────────────────────────

/// Pure-Rust CPU execution backend.
pub struct CpuBackend {
    pool: PoolAllocator,
}

impl std::fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuBackend").finish()
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new(256 * 1024 * 1024) // 256 MiB pool
    }
}

impl CpuBackend {
    /// Create a new CPU backend with the given pool capacity (bytes).
    pub fn new(pool_bytes: usize) -> Self {
        Self {
            pool: PoolAllocator::new(pool_bytes),
        }
    }
}

impl ExecutionBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<BufferHandle> {
        let numel: usize = shape.iter().product();
        // Try the pool first; fall back to a fresh allocation.
        if let Some(handle) = self.pool.acquire(numel, dtype) {
            return Ok(handle);
        }
        let buf = CpuBuffer::zeros(numel, dtype)?;
        Ok(BufferHandle::new(buf))
    }

    #[instrument(skip_all, fields(graph = %graph.name))]
    fn execute(&self, graph: &Graph, inputs: HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        // Topological execution order.
        let order = graph.topological_order()?;

        // Value map: (NodeId, output_index) → Tensor.
        let mut values: HashMap<(NodeId, usize), Tensor> = HashMap::new();

        // Seed inputs.
        for id in &graph.inputs {
            if let Some(Node::Input { name, .. }) = graph.get(*id) {
                if let Some(t) = inputs.get(name) {
                    values.insert((*id, 0), t.clone());
                }
            }
        }

        for id in &order {
            match graph.get(*id) {
                Some(Node::Constant { value, .. }) => {
                    values.insert((*id, 0), value.clone());
                }
                Some(Node::Input { .. }) => {
                    // Already seeded above.
                }
                Some(Node::Operator {
                    op,
                    inputs: inp_ids,
                    num_outputs,
                    ..
                }) => {
                    let op = op.clone();
                    let inp_ids = inp_ids.clone();
                    let _num_outputs = *num_outputs;

                    // Collect input tensors.
                    let input_tensors: Vec<Tensor> = inp_ids
                        .iter()
                        .map(|&inp| {
                            values.get(&(inp, 0)).cloned().ok_or_else(|| {
                                SapientError::internal(format!("missing value for node {inp}"))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Dispatch to kernel.
                    let outputs = self.dispatch(&op, &input_tensors)?;

                    for (i, t) in outputs.into_iter().enumerate() {
                        values.insert((*id, i), t);
                    }
                }
                Some(Node::Output { source, .. }) => {
                    // Alias output to its source.
                    if let Some(t) = values.get(&(*source, 0)).cloned() {
                        values.insert((*id, 0), t);
                    }
                }
                None => {}
            }
        }

        // Collect graph outputs in order.
        let out_tensors: Vec<Tensor> = graph
            .outputs
            .iter()
            .map(|&oid| {
                values
                    .get(&(oid, 0))
                    .cloned()
                    .ok_or_else(|| SapientError::internal(format!("output {oid} not computed")))
            })
            .collect::<Result<Vec<_>>>()?;

        debug!(
            outputs = out_tensors.len(),
            "CpuBackend: execution complete"
        );
        Ok(out_tensors)
    }

    fn supports_op(&self, op: &OpType) -> bool {
        matches!(
            op,
            OpType::MatMul | OpType::Gemm { .. }
            | OpType::Add | OpType::Sub | OpType::Mul | OpType::Div | OpType::Pow
            | OpType::Neg | OpType::Abs | OpType::Sqrt | OpType::Exp | OpType::Log
            | OpType::Relu | OpType::Sigmoid | OpType::Tanh | OpType::Gelu
            | OpType::LeakyRelu { .. } | OpType::Silu | OpType::HardSwish
            | OpType::Softmax { .. } | OpType::LogSoftmax { .. }
            | OpType::LayerNorm { .. } | OpType::RmsNorm { .. }
            | OpType::Conv2d { .. }
            | OpType::Reshape | OpType::Transpose { .. } | OpType::Flatten { .. }
            | OpType::Concat { .. }
            | OpType::ReduceSum { .. } | OpType::ReduceMean { .. }
            | OpType::ReduceMax { .. } | OpType::ReduceMin { .. }
            | OpType::Identity | OpType::Clip { .. }
            | OpType::Erf | OpType::Floor | OpType::Ceil | OpType::Round
            // LLM ops
            | OpType::Embedding { .. }
            | OpType::MultiHeadAttention { .. }
            | OpType::GroupedQueryAttention { .. }
            | OpType::RotaryEmbedding { .. }
            | OpType::CausalMask
            | OpType::KVCacheConcat
            | OpType::RepeatKV { .. }
        )
    }
}

impl CpuBackend {
    /// Dispatch an op to its kernel.
    fn dispatch(&self, op: &OpType, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let out = match op {
            // ── Linear algebra ────────────────────────────────────────────
            OpType::MatMul => {
                let a = inputs
                    .get(0)
                    .ok_or_else(|| SapientError::internal("MatMul: missing a"))?;
                let b = inputs
                    .get(1)
                    .ok_or_else(|| SapientError::internal("MatMul: missing b"))?;
                vec![kernels::matmul::matmul(a, b)?]
            }
            OpType::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
            } => {
                let a = inputs
                    .get(0)
                    .ok_or_else(|| SapientError::internal("Gemm: missing a"))?;
                let b = inputs
                    .get(1)
                    .ok_or_else(|| SapientError::internal("Gemm: missing b"))?;
                let c = inputs.get(2);
                vec![kernels::matmul::gemm(
                    a,
                    b,
                    c,
                    alpha.0 as f32,
                    beta.0 as f32,
                    *trans_a,
                    *trans_b,
                )?]
            }

            // ── Element-wise ──────────────────────────────────────────────
            OpType::Add => vec![kernels::elementwise::add(
                inputs.get(0).unwrap(),
                inputs.get(1).unwrap(),
            )?],
            OpType::Sub => vec![kernels::elementwise::sub(
                inputs.get(0).unwrap(),
                inputs.get(1).unwrap(),
            )?],
            OpType::Mul => vec![kernels::elementwise::mul(
                inputs.get(0).unwrap(),
                inputs.get(1).unwrap(),
            )?],
            OpType::Div => vec![kernels::elementwise::div(
                inputs.get(0).unwrap(),
                inputs.get(1).unwrap(),
            )?],
            OpType::Pow => vec![kernels::elementwise::pow(
                inputs.get(0).unwrap(),
                inputs.get(1).unwrap(),
            )?],
            OpType::Neg => vec![kernels::elementwise::neg(inputs.get(0).unwrap())?],
            OpType::Abs => vec![kernels::elementwise::abs(inputs.get(0).unwrap())?],
            OpType::Sqrt => vec![kernels::elementwise::sqrt(inputs.get(0).unwrap())?],
            OpType::Exp => vec![kernels::elementwise::exp(inputs.get(0).unwrap())?],
            OpType::Log => vec![kernels::elementwise::log(inputs.get(0).unwrap())?],
            OpType::Erf => vec![kernels::elementwise::erf(inputs.get(0).unwrap())?],
            OpType::Floor => vec![kernels::elementwise::floor(inputs.get(0).unwrap())?],
            OpType::Ceil => vec![kernels::elementwise::ceil(inputs.get(0).unwrap())?],
            OpType::Round => vec![kernels::elementwise::round(inputs.get(0).unwrap())?],

            // ── Activations ───────────────────────────────────────────────
            OpType::Relu => vec![kernels::elementwise::relu(inputs.get(0).unwrap())?],
            OpType::Sigmoid => vec![kernels::elementwise::sigmoid(inputs.get(0).unwrap())?],
            OpType::Tanh => vec![kernels::elementwise::tanh_act(inputs.get(0).unwrap())?],
            OpType::Gelu => vec![kernels::elementwise::gelu(inputs.get(0).unwrap())?],
            OpType::Silu => vec![kernels::elementwise::silu(inputs.get(0).unwrap())?],
            OpType::HardSwish => vec![kernels::elementwise::hard_swish(inputs.get(0).unwrap())?],
            OpType::LeakyRelu { alpha } => {
                vec![kernels::elementwise::leaky_relu(
                    inputs.get(0).unwrap(),
                    alpha.0 as f32,
                )?]
            }
            OpType::Clip { min, max } => {
                vec![kernels::elementwise::clip(
                    inputs.get(0).unwrap(),
                    min.map(|v| v.0 as f32),
                    max.map(|v| v.0 as f32),
                )?]
            }

            // ── Normalisation ──────────────────────────────────────────────
            OpType::Softmax { axis } => {
                vec![kernels::softmax::softmax(inputs.get(0).unwrap(), *axis)?]
            }
            OpType::LogSoftmax { axis } => {
                vec![kernels::softmax::log_softmax(
                    inputs.get(0).unwrap(),
                    *axis,
                )?]
            }
            OpType::LayerNorm { axis, epsilon } => {
                let weight = inputs.get(1);
                let bias = inputs.get(2);
                vec![kernels::layernorm::layer_norm(
                    inputs.get(0).unwrap(),
                    weight,
                    bias,
                    *axis,
                    epsilon.0 as f32,
                )?]
            }
            OpType::RmsNorm { epsilon } => {
                let weight = inputs.get(1);
                vec![kernels::layernorm::rms_norm(
                    inputs.get(0).unwrap(),
                    weight,
                    epsilon.0 as f32,
                )?]
            }

            // ── Convolution ────────────────────────────────────────────────
            OpType::Conv2d {
                kernel_shape,
                pads,
                strides,
                dilations,
                groups,
            } => {
                let x = inputs.get(0).unwrap();
                let w = inputs.get(1).unwrap();
                let b = inputs.get(2);
                vec![kernels::conv2d::conv2d(
                    x,
                    w,
                    b,
                    *kernel_shape,
                    *pads,
                    *strides,
                    *dilations,
                    *groups,
                )?]
            }

            // ── Shape ops ─────────────────────────────────────────────────
            OpType::Reshape => {
                let x = inputs.get(0).unwrap();
                // The new shape comes from the second input (if present) or is
                // determined at shape-inference time.
                // For now, identity (shape already baked in by runtime).
                vec![x.clone()]
            }
            OpType::Identity => vec![inputs.get(0).unwrap().clone()],

            // ── Reduce ────────────────────────────────────────────────────
            OpType::ReduceSum { axes, keep_dims } => {
                vec![kernels::reduce::reduce_sum(
                    inputs.get(0).unwrap(),
                    axes,
                    *keep_dims,
                )?]
            }
            OpType::ReduceMean { axes, keep_dims } => {
                vec![kernels::reduce::reduce_mean(
                    inputs.get(0).unwrap(),
                    axes,
                    *keep_dims,
                )?]
            }
            OpType::ReduceMax { axes, keep_dims } => {
                vec![kernels::reduce::reduce_max(
                    inputs.get(0).unwrap(),
                    axes,
                    *keep_dims,
                )?]
            }

            // ── LLM ops ───────────────────────────────────────────────────

            // Embedding lookup: weight `[vocab, hidden]` at inputs[0], token ids at inputs[1].
            OpType::Embedding { .. } => {
                let weight = inputs
                    .get(0)
                    .ok_or_else(|| SapientError::internal("Embedding: missing weight"))?;
                let ids_t = inputs
                    .get(1)
                    .ok_or_else(|| SapientError::internal("Embedding: missing input_ids"))?;
                let dims = ids_t.shape().dims();
                let seq_len: usize = dims.iter().product();
                let hidden = weight.shape().dims()[1];
                let w = weight.as_f32_slice();
                let ids: Vec<u32> = if ids_t.dtype() == DType::F32 {
                    ids_t.as_f32_slice().iter().map(|&v| v as u32).collect()
                } else {
                    ids_t
                        .as_bytes()
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                };
                let mut out = vec![0.0f32; seq_len * hidden];
                for (i, &id) in ids.iter().enumerate() {
                    let row = id as usize * hidden;
                    out[i * hidden..(i + 1) * hidden]
                        .copy_from_slice(&w[row..row + hidden]);
                }
                let batch = if dims.len() >= 2 { dims[0] } else { 1 };
                let seq = if dims.len() >= 2 { dims[1] } else { seq_len };
                vec![Tensor::from_f32(&out, vec![batch, seq, hidden])
                    .map_err(|e| SapientError::internal(e.to_string()))?]
            }

            // Grouped-Query Attention — calls the attention kernel.
            OpType::GroupedQueryAttention {
                n_heads: _,
                n_kv_heads,
                head_dim: _,
                causal,
            } => {
                let q = inputs
                    .get(0)
                    .ok_or_else(|| SapientError::internal("GQA: missing Q"))?;
                let k = inputs
                    .get(1)
                    .ok_or_else(|| SapientError::internal("GQA: missing K"))?;
                let v = inputs
                    .get(2)
                    .ok_or_else(|| SapientError::internal("GQA: missing V"))?;
                let mask = if *causal {
                    let seq_q = q.shape().dims().get(2).copied().unwrap_or(1);
                    let seq_k = k.shape().dims().get(2).copied().unwrap_or(1);
                    Some(kernels::attention::causal_mask(seq_q, seq_k))
                } else {
                    None
                };
                vec![kernels::attention::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    mask.as_ref(),
                    None,
                    *n_kv_heads,
                )?]
            }

            // Multi-head attention (non-GQA, n_kv_heads = n_heads).
            OpType::MultiHeadAttention {
                num_heads,
                head_dim: _,
                causal,
                scale,
            } => {
                let q = inputs
                    .get(0)
                    .ok_or_else(|| SapientError::internal("MHA: missing Q"))?;
                let k = inputs
                    .get(1)
                    .ok_or_else(|| SapientError::internal("MHA: missing K"))?;
                let v = inputs
                    .get(2)
                    .ok_or_else(|| SapientError::internal("MHA: missing V"))?;
                let mask = if *causal {
                    let sq = q.shape().dims().get(2).copied().unwrap_or(1);
                    let sk = k.shape().dims().get(2).copied().unwrap_or(1);
                    Some(kernels::attention::causal_mask(sq, sk))
                } else {
                    None
                };
                vec![kernels::attention::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    mask.as_ref(),
                    scale.map(|s| s.0 as f32),
                    *num_heads,
                )?]
            }

            // RoPE — apply rotary embeddings to Q or K.
            OpType::RotaryEmbedding { base, dim: _ } => {
                let x = inputs
                    .get(0)
                    .ok_or_else(|| SapientError::internal("RoPE: missing input"))?;
                let seq_len = x.shape().dims().get(2).copied().unwrap_or(1);
                let positions: Vec<usize> = (0..seq_len).collect();
                vec![kernels::rope::apply_rope(x, &positions, base.0 as f32)?]
            }

            // Causal mask generation.
            OpType::CausalMask => {
                let seq = inputs
                    .get(0)
                    .map(|t| t.shape().dims().get(1).copied().unwrap_or(1))
                    .unwrap_or(1);
                vec![kernels::attention::causal_mask(seq, seq)]
            }

            // KV cache concat — identity for now (cache is managed by Pipeline).
            OpType::KVCacheConcat | OpType::RepeatKV { .. } => {
                vec![inputs.get(0).unwrap().clone()]
            }

            // MoE gate / dispatch — identity (scheduler handles expert routing).
            OpType::MoEGate { .. } | OpType::ScaledDotProductAttention { .. } => {
                vec![inputs.get(0).unwrap().clone()]
            }

            // ALiBi — return a zero tensor (will be added to attention logits).
            OpType::ALiBi { .. } => {
                vec![Tensor::zeros(vec![1], DType::F32).unwrap()]
            }

            // ── Fallback ──────────────────────────────────────────────────
            other => {
                return Err(SapientError::unsupported_op("cpu", &other.to_string()));
            }
        };
        Ok(out)
    }
}
