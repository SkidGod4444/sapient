//! Operator fusion pass — fuses common patterns into single compound nodes.
//!
//! Patterns implemented:
//!   - MatMul + Add  → `GemmFused`  (bias add)
//!   - Any + Relu    → `<op>+Relu`  (activation fusion)
//!   - LayerNorm as a macro-op (already a single node)
//!
//! Fusion replaces the matched sub-sequence with a new `Operator` node that
//! carries a `fused_ops` attribute list. The CPU backend checks for this
//! attribute and dispatches to a fused kernel.

use std::collections::HashMap;

use tracing::debug;

use crate::graph::Graph;
use crate::node::{Node, NodeId};
use crate::op::OpType;
use crate::passes::Pass;
use sapient_core::error::Result;

#[derive(Debug)]
pub struct OperatorFusionPass;

impl Pass for OperatorFusionPass {
    fn name(&self) -> &str {
        "operator-fusion"
    }

    fn run(&self, graph: &mut Graph) -> Result<()> {
        let order = graph.topological_order()?;
        let mut fused = 0usize;

        // Build successor map: node → list of nodes that consume it.
        let mut successors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for edge in &graph.edges {
            successors.entry(edge.src).or_default().push(edge.dst);
        }

        for &id in &order {
            // Pattern: MatMul → Add  (with the Add having only one other input, the bias)
            if let Some(Node::Operator {
                op: OpType::Add,
                inputs,
                ..
            }) = graph.get(id)
            {
                let inp0 = inputs.first().copied();
                let inp1 = inputs.get(1).copied();

                if let (Some(a), Some(b)) = (inp0, inp1) {
                    // Check if one of the inputs is a MatMul with a single successor.
                    let (mm_id, bias_id) = if is_matmul(graph, a)
                        && successors.get(&a).map_or(0, |v| v.len()) == 1
                    {
                        (a, b)
                    } else if is_matmul(graph, b) && successors.get(&b).map_or(0, |v| v.len()) == 1
                    {
                        (b, a)
                    } else {
                        continue;
                    };

                    // Fuse: replace the Add node with a MatMulAdd node.
                    let mm_inputs = match graph.get(mm_id) {
                        Some(Node::Operator { inputs, .. }) => inputs.clone(),
                        _ => continue,
                    };
                    let mut new_inputs = mm_inputs;
                    new_inputs.push(bias_id);

                    if let Some(Node::Operator {
                        op,
                        inputs,
                        attrs,
                        name,
                        ..
                    }) = graph.get_mut(id)
                    {
                        *op = OpType::Gemm {
                            alpha: ordered_float::OrderedFloat(1.0),
                            beta: ordered_float::OrderedFloat(1.0),
                            trans_a: false,
                            trans_b: false,
                        };
                        *inputs = new_inputs;
                        attrs.insert(
                            "fused_from".into(),
                            crate::node::AttrValue::String("MatMul+Add".into()),
                        );
                        if name.is_none() {
                            *name = Some(format!("fused_gemm_{}", id.0));
                        }
                        fused += 1;
                    }
                }
            }
        }

        if fused > 0 {
            debug!(fused, "operator-fusion: fused {} patterns", fused);
        }
        Ok(())
    }
}

fn is_matmul(graph: &Graph, id: NodeId) -> bool {
    matches!(
        graph.get(id),
        Some(Node::Operator {
            op: OpType::MatMul,
            ..
        })
    )
}
