//! Constant folding pass — evaluate sub-graphs whose inputs are all constants
//! at graph-build time and replace them with a single `Constant` node.

use std::collections::HashSet;

use tracing::debug;

use crate::graph::Graph;
use crate::node::{Node, NodeId};
use crate::op::OpType;
use crate::passes::Pass;
use sapient_core::error::Result;

#[derive(Debug)]
pub struct ConstantFoldingPass;

impl Pass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant-folding"
    }

    fn run(&self, graph: &mut Graph) -> Result<()> {
        let order = graph.topological_order()?;
        let mut const_nodes: HashSet<NodeId> = HashSet::new();

        // Collect all constant / input nodes first.
        for id in &order {
            if let Some(Node::Constant { .. }) = graph.get(*id) {
                const_nodes.insert(*id);
            }
        }

        let mut folded = 0usize;

        for id in &order {
            if let Some(Node::Operator { op, inputs, .. }) = graph.get(*id) {
                // Skip if not all inputs are constants.
                if inputs.iter().all(|inp| const_nodes.contains(inp)) {
                    // Attempt fold for simple element-wise ops.
                    if let Some(result) = try_fold(op, inputs, graph) {
                        // Replace the operator node with a constant node.
                        let name = Some(format!("folded_{}", id.0));
                        let id_copy = *id;
                        if let Some(node) = graph.get_mut(id_copy) {
                            *node = Node::Constant {
                                id: id_copy,
                                value: result,
                                name,
                            };
                            const_nodes.insert(id_copy);
                            folded += 1;
                        }
                    }
                }
            }
        }

        if folded > 0 {
            debug!(folded, "constant-folding: folded {} node(s)", folded);
        }
        Ok(())
    }
}

/// Attempt to fold a simple op given all-constant inputs.
fn try_fold(op: &OpType, inputs: &[NodeId], graph: &Graph) -> Option<sapient_core::Tensor> {
    use sapient_core::{DType, Tensor};

    let get_f32 = |id: NodeId| -> Option<Vec<f32>> {
        match graph.get(id)? {
            Node::Constant { value, .. } if value.dtype() == DType::F32 => {
                Some(value.as_f32_slice().to_vec())
            }
            _ => None,
        }
    };

    let get_shape = |id: NodeId| -> Option<sapient_core::Shape> {
        match graph.get(id)? {
            Node::Constant { value, .. } => Some(value.shape().clone()),
            _ => None,
        }
    };

    match op {
        OpType::Add if inputs.len() == 2 => {
            let a = get_f32(inputs[0])?;
            let b = get_f32(inputs[1])?;
            if a.len() != b.len() {
                return None;
            }
            let out: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
            let shape = get_shape(inputs[0])?;
            Tensor::from_f32(&out, shape).ok()
        }
        OpType::Mul if inputs.len() == 2 => {
            let a = get_f32(inputs[0])?;
            let b = get_f32(inputs[1])?;
            if a.len() != b.len() {
                return None;
            }
            let out: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
            let shape = get_shape(inputs[0])?;
            Tensor::from_f32(&out, shape).ok()
        }
        OpType::Relu if inputs.len() == 1 => {
            let a = get_f32(inputs[0])?;
            let out: Vec<f32> = a.iter().map(|&x| x.max(0.0)).collect();
            let shape = get_shape(inputs[0])?;
            Tensor::from_f32(&out, shape).ok()
        }
        _ => None,
    }
}
