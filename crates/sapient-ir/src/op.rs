// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Operator type enum — every compute operation in the SAPIENT IR.

use serde::{Deserialize, Serialize};

/// All supported operator types in the SAPIENT IR.
///
/// When adding a new op, also update:
/// - `shape_inference::ShapeRegistry::infer_op`
/// - CPU backend `kernels/`
/// - ONNX importer mapping
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OpType {
    // ── Linear algebra ────────────────────────────────────────────────────────
    /// Matrix multiply: (M, K) × (K, N) → (M, N), supports batched.
    MatMul,
    /// Dot product (alias for 2-D MatMul).
    Gemm {
        alpha: ordered_float::OrderedFloat<f64>,
        beta: ordered_float::OrderedFloat<f64>,
        trans_a: bool,
        trans_b: bool,
    },

    // ── Element-wise arithmetic ───────────────────────────────────────────────
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,

    // ── Activations ───────────────────────────────────────────────────────────
    Relu,
    Sigmoid,
    Tanh,
    /// Gaussian Error Linear Unit.
    Gelu,
    /// Leaky ReLU with configurable alpha.
    LeakyRelu {
        alpha: ordered_float::OrderedFloat<f64>,
    },
    Silu,
    HardSwish,

    // ── Normalisation ─────────────────────────────────────────────────────────
    Softmax {
        axis: i64,
    },
    LogSoftmax {
        axis: i64,
    },
    LayerNorm {
        axis: i64,
        epsilon: ordered_float::OrderedFloat<f64>,
    },
    BatchNorm {
        epsilon: ordered_float::OrderedFloat<f64>,
        momentum: ordered_float::OrderedFloat<f64>,
    },
    RmsNorm {
        epsilon: ordered_float::OrderedFloat<f64>,
    },

    // ── Convolution / pooling ─────────────────────────────────────────────────
    Conv2d {
        kernel_shape: [usize; 2],
        pads: [usize; 4],
        strides: [usize; 2],
        dilations: [usize; 2],
        groups: usize,
    },
    MaxPool {
        kernel_shape: [usize; 2],
        pads: [usize; 4],
        strides: [usize; 2],
    },
    AvgPool {
        kernel_shape: [usize; 2],
        pads: [usize; 4],
        strides: [usize; 2],
    },
    GlobalAvgPool,

    // ── Shape / layout ────────────────────────────────────────────────────────
    Reshape,
    Transpose {
        perm: Vec<usize>,
    },
    Flatten {
        axis: i64,
    },
    Squeeze {
        axes: Vec<i64>,
    },
    Unsqueeze {
        axes: Vec<i64>,
    },
    Expand,
    Concat {
        axis: i64,
    },
    Split {
        axis: i64,
        num_outputs: usize,
    },
    Slice,
    Gather {
        axis: i64,
    },
    ScatterElements {
        axis: i64,
    },
    Pad {
        mode: PadMode,
    },
    Tile,

    // ── Reduce ops ────────────────────────────────────────────────────────────
    ReduceSum {
        axes: Vec<i64>,
        keep_dims: bool,
    },
    ReduceMean {
        axes: Vec<i64>,
        keep_dims: bool,
    },
    ReduceMax {
        axes: Vec<i64>,
        keep_dims: bool,
    },
    ReduceMin {
        axes: Vec<i64>,
        keep_dims: bool,
    },
    ArgMax {
        axis: i64,
        keep_dims: bool,
    },
    ArgMin {
        axis: i64,
        keep_dims: bool,
    },

    // ── Comparison ────────────────────────────────────────────────────────────
    Equal,
    Greater,
    Less,
    Not,
    And,
    Or,
    Where,

    // ── Type conversion ───────────────────────────────────────────────────────
    Cast {
        to: sapient_core::DType,
    },

    // ── Clip ─────────────────────────────────────────────────────────────────
    Clip {
        min: Option<ordered_float::OrderedFloat<f64>>,
        max: Option<ordered_float::OrderedFloat<f64>>,
    },

    // ── LLM / Transformer-specific ────────────────────────────────────────────
    /// Token embedding lookup: (vocab_size, dim) weight × token_ids → (seq, dim).
    Embedding {
        vocab_size: usize,
        dim: usize,
    },

    /// Multi-head self-attention (standard, causal or bidirectional).
    MultiHeadAttention {
        num_heads: usize,
        head_dim: usize,
        causal: bool,
        /// Softmax scale override (default: 1/√head_dim).
        scale: Option<ordered_float::OrderedFloat<f64>>,
    },

    /// Grouped-Query Attention — used by Llama2/3, Mistral, Gemma.
    /// `n_kv_heads` < `n_heads`; KV heads are repeated to match Q heads.
    GroupedQueryAttention {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        causal: bool,
    },

    /// Rotary Position Embedding (RoPE) — applied to Q and K tensors.
    RotaryEmbedding {
        /// RoPE base frequency (default 10000.0 for Llama).
        base: ordered_float::OrderedFloat<f64>,
        /// Rotary dimension (usually head_dim).
        dim: usize,
    },

    /// ALiBi positional bias — added to attention logits (MPT, BLOOM).
    ALiBi {
        n_heads: usize,
    },

    /// Generate a causal (lower-triangular) attention mask of shape (seq, seq).
    CausalMask,

    /// KV-cache read/write: concatenate new K or V with the rolling cache.
    KVCacheConcat,

    /// Scaled dot-product attention (kernel-fused, no explicit QKV split).
    ScaledDotProductAttention {
        causal: bool,
    },

    /// Mixture-of-Experts gate + dispatch (Mixtral, Qwen-MoE).
    MoEGate {
        num_experts: usize,
        top_k: usize,
    },

    /// Repeat/expand KV heads to match Q heads (part of GQA expansion).
    RepeatKV {
        n_rep: usize,
    },

    // ── Misc ─────────────────────────────────────────────────────────────────
    Identity,
    Constant, // leaf node — value stored in Node::Constant
    Dropout {
        ratio: ordered_float::OrderedFloat<f64>,
    },
    Erf,
    Floor,
    Ceil,
    Round,
    Sign,
    IsNaN,
    NonZero,
    Size,
    /// Returns the shape of a tensor as a 1-D int64 tensor.
    ShapeOp,
    Einsum {
        equation: String,
    },
}

/// Padding mode for the `Pad` operator.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PadMode {
    Constant,
    Reflect,
    Edge,
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            OpType::MatMul => "MatMul",
            OpType::Gemm { .. } => "Gemm",
            OpType::Add => "Add",
            OpType::Sub => "Sub",
            OpType::Mul => "Mul",
            OpType::Div => "Div",
            OpType::Pow => "Pow",
            OpType::Neg => "Neg",
            OpType::Abs => "Abs",
            OpType::Sqrt => "Sqrt",
            OpType::Exp => "Exp",
            OpType::Log => "Log",
            OpType::Relu => "Relu",
            OpType::Sigmoid => "Sigmoid",
            OpType::Tanh => "Tanh",
            OpType::Gelu => "Gelu",
            OpType::LeakyRelu { .. } => "LeakyRelu",
            OpType::Silu => "Silu",
            OpType::HardSwish => "HardSwish",
            OpType::Softmax { .. } => "Softmax",
            OpType::LogSoftmax { .. } => "LogSoftmax",
            OpType::LayerNorm { .. } => "LayerNorm",
            OpType::BatchNorm { .. } => "BatchNorm",
            OpType::RmsNorm { .. } => "RmsNorm",
            OpType::Conv2d { .. } => "Conv2d",
            OpType::MaxPool { .. } => "MaxPool",
            OpType::AvgPool { .. } => "AvgPool",
            OpType::GlobalAvgPool => "GlobalAvgPool",
            OpType::Reshape => "Reshape",
            OpType::Transpose { .. } => "Transpose",
            OpType::Flatten { .. } => "Flatten",
            OpType::Squeeze { .. } => "Squeeze",
            OpType::Unsqueeze { .. } => "Unsqueeze",
            OpType::Expand => "Expand",
            OpType::Concat { .. } => "Concat",
            OpType::Split { .. } => "Split",
            OpType::Slice => "Slice",
            OpType::Gather { .. } => "Gather",
            OpType::ScatterElements { .. } => "ScatterElements",
            OpType::Pad { .. } => "Pad",
            OpType::Tile => "Tile",
            OpType::ReduceSum { .. } => "ReduceSum",
            OpType::ReduceMean { .. } => "ReduceMean",
            OpType::ReduceMax { .. } => "ReduceMax",
            OpType::ReduceMin { .. } => "ReduceMin",
            OpType::ArgMax { .. } => "ArgMax",
            OpType::ArgMin { .. } => "ArgMin",
            OpType::Equal => "Equal",
            OpType::Greater => "Greater",
            OpType::Less => "Less",
            OpType::Not => "Not",
            OpType::And => "And",
            OpType::Or => "Or",
            OpType::Where => "Where",
            OpType::Cast { .. } => "Cast",
            OpType::MultiHeadAttention { .. } => "MultiHeadAttention",
            OpType::GroupedQueryAttention { .. } => "GroupedQueryAttention",
            OpType::RotaryEmbedding { .. } => "RotaryEmbedding",
            OpType::ALiBi { .. } => "ALiBi",
            OpType::CausalMask => "CausalMask",
            OpType::KVCacheConcat => "KVCacheConcat",
            OpType::ScaledDotProductAttention { .. } => "ScaledDotProductAttention",
            OpType::MoEGate { .. } => "MoEGate",
            OpType::RepeatKV { .. } => "RepeatKV",
            OpType::Embedding { .. } => "Embedding",
            OpType::Clip { .. } => "Clip",
            OpType::Identity => "Identity",
            OpType::Constant => "Constant",
            OpType::Dropout { .. } => "Dropout",
            OpType::Erf => "Erf",
            OpType::Floor => "Floor",
            OpType::Ceil => "Ceil",
            OpType::Round => "Round",
            OpType::Sign => "Sign",
            OpType::IsNaN => "IsNaN",
            OpType::NonZero => "NonZero",
            OpType::Size => "Size",
            OpType::ShapeOp => "Shape",
            OpType::Einsum { .. } => "Einsum",
        };
        f.write_str(name)
    }
}
