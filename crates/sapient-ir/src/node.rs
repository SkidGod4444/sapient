//! Graph nodes — operators, constants, inputs, and outputs.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use sapient_core::{DType, Shape, Tensor};

use crate::op::OpType;

// ── NodeId ────────────────────────────────────────────────────────────────────

/// Opaque identifier for a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

// ── AttributeValue ────────────────────────────────────────────────────────────

/// An op attribute — scalars, ints, lists, shapes stored here instead of
/// embedding in OpType variants for more flexibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttrValue {
    Float(f64),
    Int(i64),
    String(String),
    Floats(Vec<f64>),
    Ints(Vec<i64>),
    Shape(Shape),
}

// ── Node ─────────────────────────────────────────────────────────────────────

/// A node in the compute graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    /// A compute operator consuming inputs and producing outputs.
    Operator {
        id: NodeId,
        op: OpType,
        /// Names of input tensors/nodes in order.
        inputs: Vec<NodeId>,
        /// Number of output tensors this op produces.
        num_outputs: usize,
        /// Additional key-value attributes (supplements OpType fields).
        attrs: HashMap<String, AttrValue>,
        /// Human-readable name (from ONNX node name or auto-generated).
        name: Option<String>,
        /// Inferred output shapes (one per output).
        output_shapes: Vec<Option<Shape>>,
        /// Inferred output dtypes (one per output).
        output_dtypes: Vec<Option<DType>>,
    },

    /// A constant tensor (model weights, biases, etc.).
    Constant {
        id: NodeId,
        value: Tensor,
        name: Option<String>,
    },

    /// A graph input placeholder.
    Input {
        id: NodeId,
        name: String,
        shape: Option<Shape>,
        dtype: Option<DType>,
    },

    /// A graph output marker.
    Output {
        id: NodeId,
        name: String,
        source: NodeId,
    },
}

impl Node {
    /// The unique identifier of this node.
    pub fn id(&self) -> NodeId {
        match self {
            Node::Operator { id, .. } => *id,
            Node::Constant { id, .. } => *id,
            Node::Input { id, .. } => *id,
            Node::Output { id, .. } => *id,
        }
    }

    /// Human-readable name (falls back to id).
    pub fn name(&self) -> String {
        match self {
            Node::Operator { name, id, .. } => {
                name.clone().unwrap_or_else(|| format!("op_{}", id.0))
            }
            Node::Constant { name, id, .. } => {
                name.clone().unwrap_or_else(|| format!("const_{}", id.0))
            }
            Node::Input { name, .. } => name.clone(),
            Node::Output { name, .. } => name.clone(),
        }
    }

    /// Node kind as a static string.
    pub fn kind(&self) -> &'static str {
        match self {
            Node::Operator { .. } => "operator",
            Node::Constant { .. } => "constant",
            Node::Input { .. } => "input",
            Node::Output { .. } => "output",
        }
    }

    /// Input node IDs consumed by this node (empty for Input/Constant/Output).
    pub fn input_ids(&self) -> &[NodeId] {
        match self {
            Node::Operator { inputs, .. } => inputs,
            Node::Output { .. } => std::slice::from_ref(match self {
                Node::Output { source, .. } => source,
                _ => unreachable!(),
            }),
            _ => &[],
        }
    }
}
