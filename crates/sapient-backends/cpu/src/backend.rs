//! `ExecutionBackend` trait and `CpuBackend` implementation.

use std::collections::HashMap;

use tracing::{debug, instrument};

use sapient_core::{DType, Shape, Tensor};
use sapient_core::buffer::{BufferHandle, CpuBuffer};
use sapient_core::error::{Result, SapientError};
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
    fn execute(
        &self,
        graph: &Graph,
        inputs: HashMap<String, Tensor>,
    ) -> Result<Vec<Tensor>>;

    /// Whether this backend can execute the given op natively.
    fn supports_op(&self, op: &OpType) -> bool;

    /// True if this backend is available on the current system.
    fn is_available() -> bool where Self: Sized {
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
    fn name(&self) -> &str { "cpu" }

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
    fn execute(
        &self,
        graph: &Graph,
        inputs: HashMap<String, Tensor>,
    ) -> Result<Vec<Tensor>> {
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
                Some(Node::Operator { op, inputs: inp_ids, num_outputs, .. }) => {
                    let op = op.clone();
                    let inp_ids = inp_ids.clone();
                    let num_outputs = *num_outputs;

                    // Collect input tensors.
                    let input_tensors: Vec<Tensor> = inp_ids
                        .iter()
                        .map(|&inp| {
                            values.get(&(inp, 0))
                                .cloned()
                                .ok_or_else(|| {
                                    SapientError::internal(format!(
                                        "missing value for node {inp}"
                                    ))
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
                values.get(&(oid, 0))
                    .cloned()
                    .ok_or_else(|| SapientError::internal(format!("output {oid} not computed")))
            })
            .collect::<Result<Vec<_>>>()?;

        debug!(outputs = out_tensors.len(), "CpuBackend: execution complete");
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
        )
    }
}

impl CpuBackend {
    /// Dispatch an op to its kernel.
    fn dispatch(&self, op: &OpType, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let out = match op {
            // ── Linear algebra ────────────────────────────────────────────
            OpType::MatMul => {
                let a = inputs.get(0).ok_or_else(|| SapientError::internal("MatMul: missing a"))?;
                let b = inputs.get(1).ok_or_else(|| SapientError::internal("MatMul: missing b"))?;
                vec![kernels::matmul::matmul(a, b)?]
            }
            OpType::Gemm { alpha, beta, trans_a, trans_b } => {
                let a = inputs.get(0).ok_or_else(|| SapientError::internal("Gemm: missing a"))?;
                let b = inputs.get(1).ok_or_else(|| SapientError::internal("Gemm: missing b"))?;
                let c = inputs.get(2);
                vec![kernels::matmul::gemm(a, b, c, alpha.0 as f32, beta.0 as f32, *trans_a, *trans_b)?]
            }

            // ── Element-wise ──────────────────────────────────────────────
            OpType::Add  => vec![kernels::elementwise::add(inputs.get(0).unwrap(), inputs.get(1).unwrap())?],
            OpType::Sub  => vec![kernels::elementwise::sub(inputs.get(0).unwrap(), inputs.get(1).unwrap())?],
            OpType::Mul  => vec![kernels::elementwise::mul(inputs.get(0).unwrap(), inputs.get(1).unwrap())?],
            OpType::Div  => vec![kernels::elementwise::div(inputs.get(0).unwrap(), inputs.get(1).unwrap())?],
            OpType::Pow  => vec![kernels::elementwise::pow(inputs.get(0).unwrap(), inputs.get(1).unwrap())?],
            OpType::Neg  => vec![kernels::elementwise::neg(inputs.get(0).unwrap())?],
            OpType::Abs  => vec![kernels::elementwise::abs(inputs.get(0).unwrap())?],
            OpType::Sqrt => vec![kernels::elementwise::sqrt(inputs.get(0).unwrap())?],
            OpType::Exp  => vec![kernels::elementwise::exp(inputs.get(0).unwrap())?],
            OpType::Log  => vec![kernels::elementwise::log(inputs.get(0).unwrap())?],
            OpType::Erf  => vec![kernels::elementwise::erf(inputs.get(0).unwrap())?],
            OpType::Floor => vec![kernels::elementwise::floor(inputs.get(0).unwrap())?],
            OpType::Ceil  => vec![kernels::elementwise::ceil(inputs.get(0).unwrap())?],
            OpType::Round => vec![kernels::elementwise::round(inputs.get(0).unwrap())?],

            // ── Activations ───────────────────────────────────────────────
            OpType::Relu    => vec![kernels::elementwise::relu(inputs.get(0).unwrap())?],
            OpType::Sigmoid => vec![kernels::elementwise::sigmoid(inputs.get(0).unwrap())?],
            OpType::Tanh    => vec![kernels::elementwise::tanh_act(inputs.get(0).unwrap())?],
            OpType::Gelu    => vec![kernels::elementwise::gelu(inputs.get(0).unwrap())?],
            OpType::Silu    => vec![kernels::elementwise::silu(inputs.get(0).unwrap())?],
            OpType::HardSwish => vec![kernels::elementwise::hard_swish(inputs.get(0).unwrap())?],
            OpType::LeakyRelu { alpha } => {
                vec![kernels::elementwise::leaky_relu(inputs.get(0).unwrap(), alpha.0 as f32)?]
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
                vec![kernels::softmax::log_softmax(inputs.get(0).unwrap(), *axis)?]
            }
            OpType::LayerNorm { axis, epsilon } => {
                let weight = inputs.get(1);
                let bias   = inputs.get(2);
                vec![kernels::layernorm::layer_norm(
                    inputs.get(0).unwrap(), weight, bias, *axis, epsilon.0 as f32,
                )?]
            }
            OpType::RmsNorm { epsilon } => {
                let weight = inputs.get(1);
                vec![kernels::layernorm::rms_norm(
                    inputs.get(0).unwrap(), weight, epsilon.0 as f32,
                )?]
            }

            // ── Convolution ────────────────────────────────────────────────
            OpType::Conv2d { kernel_shape, pads, strides, dilations, groups } => {
                let x = inputs.get(0).unwrap();
                let w = inputs.get(1).unwrap();
                let b = inputs.get(2);
                vec![kernels::conv2d::conv2d(
                    x, w, b, *kernel_shape, *pads, *strides, *dilations, *groups,
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
                vec![kernels::reduce::reduce_sum(inputs.get(0).unwrap(), axes, *keep_dims)?]
            }
            OpType::ReduceMean { axes, keep_dims } => {
                vec![kernels::reduce::reduce_mean(inputs.get(0).unwrap(), axes, *keep_dims)?]
            }
            OpType::ReduceMax { axes, keep_dims } => {
                vec![kernels::reduce::reduce_max(inputs.get(0).unwrap(), axes, *keep_dims)?]
            }

            // ── Fallback ──────────────────────────────────────────────────
            other => {
                return Err(SapientError::unsupported_op("cpu", &other.to_string()));
            }
        };
        Ok(out)
    }
}
