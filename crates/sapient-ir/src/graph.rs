//! `Graph` — the central compute graph data structure in the SAPIENT IR.
//!
//! The graph stores nodes in a `Vec<Node>` indexed by `NodeId`.  Edges are
//! implicit: each `Operator` node lists its input `NodeId`s.
//!
//! Invariants maintained by the graph:
//!   - No two nodes share the same `NodeId`.
//!   - The graph is a DAG (enforced by `validate()`).
//!   - All referenced input `NodeId`s exist.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use sapient_core::error::{Result, SapientError};

use crate::node::{Node, NodeId};
use crate::op::OpType;

// ── Edge ─────────────────────────────────────────────────────────────────────

/// Represents a directed data-flow edge `src → dst` with an output index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub src: NodeId,
    /// Which output of `src` (0-indexed).
    pub src_output: usize,
    pub dst: NodeId,
    /// Which input slot of `dst` (0-indexed).
    pub dst_input: usize,
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// An acyclic compute graph (DAG).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub name: String,
    nodes: Vec<Node>,
    id_to_idx: HashMap<NodeId, usize>,
    next_id: u32,
    /// Graph-level input node IDs.
    pub inputs: Vec<NodeId>,
    /// Graph-level output node IDs.
    pub outputs: Vec<NodeId>,
    /// Explicit edge list (redundant but useful for serialisation).
    pub edges: Vec<Edge>,
}

impl Graph {
    // ── Construction ─────────────────────────────────────────────────────────

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            next_id: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Allocate the next `NodeId`.
    pub fn next_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Insert a node into the graph and return its `NodeId`.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = node.id();
        let idx = self.nodes.len();
        self.id_to_idx.insert(id, idx);
        self.nodes.push(node);
        id
    }

    /// Convenience: add an Input node.
    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        shape: Option<sapient_core::Shape>,
        dtype: Option<sapient_core::DType>,
    ) -> NodeId {
        let id = self.next_id();
        let node = Node::Input {
            id,
            name: name.into(),
            shape,
            dtype,
        };
        self.add_node(node);
        self.inputs.push(id);
        id
    }

    /// Convenience: add a Constant node.
    pub fn add_constant(&mut self, value: sapient_core::Tensor, name: Option<String>) -> NodeId {
        let id = self.next_id();
        let node = Node::Constant { id, value, name };
        self.add_node(node)
    }

    /// Convenience: add an Operator node.
    pub fn add_op(
        &mut self,
        op: OpType,
        inputs: Vec<NodeId>,
        num_outputs: usize,
        name: Option<String>,
    ) -> NodeId {
        let id = self.next_id();
        // Record edges.
        for (slot, &src) in inputs.iter().enumerate() {
            self.edges.push(Edge {
                src,
                src_output: 0,
                dst: id,
                dst_input: slot,
            });
        }
        let node = Node::Operator {
            id,
            op,
            inputs,
            num_outputs,
            attrs: Default::default(),
            name,
            output_shapes: vec![None; num_outputs],
            output_dtypes: vec![None; num_outputs],
        };
        self.add_node(node)
    }

    /// Mark a node as a graph output.
    pub fn mark_output(&mut self, source: NodeId, name: impl Into<String>) -> NodeId {
        let id = self.next_id();
        let node = Node::Output {
            id,
            name: name.into(),
            source,
        };
        self.add_node(node);
        self.outputs.push(id);
        id
    }

    // ── Lookup ───────────────────────────────────────────────────────────────

    /// Get a reference to a node by ID.
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.id_to_idx.get(&id).map(|&i| &self.nodes[i])
    }

    /// Get a mutable reference to a node by ID.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.id_to_idx.get(&id).map(|&i| &mut self.nodes[i])
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ── Traversal ────────────────────────────────────────────────────────────

    /// Return nodes in topological (execution) order.
    ///
    /// Uses Kahn's algorithm — also validates the DAG property.
    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in &self.nodes {
            let id = node.id();
            in_degree.entry(id).or_insert(0);
            adj.entry(id).or_default();
        }

        for edge in &self.edges {
            *in_degree.entry(edge.dst).or_insert(0) += 1;
            adj.entry(edge.src).or_default().push(edge.dst);
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = queue.pop_front() {
            order.push(id);
            if let Some(succs) = adj.get(&id) {
                for &s in succs {
                    let deg = in_degree.get_mut(&s).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(s);
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(SapientError::CyclicGraph);
        }

        debug!(graph = %self.name, nodes = order.len(), "topological sort complete");
        Ok(order)
    }

    // ── Validation ───────────────────────────────────────────────────────────

    /// Full graph validation: DAG check + referential integrity.
    pub fn validate(&self) -> Result<()> {
        // Check topological order (implicitly checks for cycles).
        self.topological_order()?;

        // All referenced input IDs must exist.
        for node in &self.nodes {
            for &dep in node.input_ids() {
                if !self.id_to_idx.contains_key(&dep) {
                    return Err(SapientError::NodeNotFound(format!("{dep}")));
                }
            }
        }

        // At least one input and one output.
        if self.inputs.is_empty() {
            warn!(graph = %self.name, "graph has no inputs");
        }
        if self.outputs.is_empty() {
            warn!(graph = %self.name, "graph has no outputs");
        }

        Ok(())
    }

    // ── Serialisation ────────────────────────────────────────────────────────

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| SapientError::internal(e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| SapientError::internal(e.to_string()))
    }

    /// DOT (Graphviz) format for visualisation.
    pub fn to_dot(&self) -> String {
        let mut s = format!(
            "digraph {} {{\n  rankdir=TB;\n",
            self.name.replace(' ', "_")
        );
        for node in &self.nodes {
            let label = match node {
                Node::Operator { op, name, .. } => {
                    format!("{} [{}]", op, name.as_deref().unwrap_or(""))
                }
                Node::Constant { name, value, .. } => {
                    format!(
                        "Const:{}",
                        name.as_deref().unwrap_or(&value.shape().to_string())
                    )
                }
                Node::Input { name, .. } => format!("Input:{name}"),
                Node::Output { name, .. } => format!("Output:{name}"),
            };
            s.push_str(&format!("  {} [label=\"{}\"];\n", node.id().0, label));
        }
        for edge in &self.edges {
            s.push_str(&format!("  {} -> {};\n", edge.src.0, edge.dst.0));
        }
        s.push_str("}\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OpType;
    use sapient_core::{DType, Tensor};

    fn simple_mlp() -> Graph {
        let mut g = Graph::new("mlp");
        let x = g.add_input("x", None, Some(DType::F32));
        let w = g.add_constant(
            Tensor::zeros(vec![4, 4], DType::F32).unwrap(),
            Some("W".into()),
        );
        let mm = g.add_op(OpType::MatMul, vec![x, w], 1, Some("mm".into()));
        let r = g.add_op(OpType::Relu, vec![mm], 1, Some("relu".into()));
        g.mark_output(r, "output");
        g
    }

    #[test]
    fn topo_order() {
        let g = simple_mlp();
        let order = g.topological_order().unwrap();
        assert_eq!(order.len(), g.node_count());
    }

    #[test]
    fn validate_ok() {
        let g = simple_mlp();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn cycle_detected() {
        let mut g = Graph::new("cyclic");
        let a = g.next_id();
        let b = g.next_id();
        g.nodes.push(Node::Operator {
            id: a,
            op: OpType::Identity,
            inputs: vec![b],
            num_outputs: 1,
            attrs: Default::default(),
            name: None,
            output_shapes: vec![None],
            output_dtypes: vec![None],
        });
        g.nodes.push(Node::Operator {
            id: b,
            op: OpType::Identity,
            inputs: vec![a],
            num_outputs: 1,
            attrs: Default::default(),
            name: None,
            output_shapes: vec![None],
            output_dtypes: vec![None],
        });
        g.id_to_idx.insert(a, 0);
        g.id_to_idx.insert(b, 1);
        g.edges.push(Edge {
            src: a,
            src_output: 0,
            dst: b,
            dst_input: 0,
        });
        g.edges.push(Edge {
            src: b,
            src_output: 0,
            dst: a,
            dst_input: 0,
        });
        assert!(matches!(
            g.topological_order(),
            Err(SapientError::CyclicGraph)
        ));
    }

    #[test]
    fn dot_output() {
        let g = simple_mlp();
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("MatMul"));
        assert!(dot.contains("Relu"));
    }

    #[test]
    fn json_roundtrip() {
        let g = simple_mlp();
        let json = g.to_json().unwrap();
        let g2 = Graph::from_json(&json).unwrap();
        assert_eq!(g.node_count(), g2.node_count());
    }
}
