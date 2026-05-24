//! Shape inference — propagates tensor shapes through the compute graph.

use std::collections::HashMap;

use sapient_core::error::{Result, SapientError};
use sapient_core::{DType, Shape};

use crate::graph::Graph;
use crate::node::{Node, NodeId};
use crate::op::OpType;

// ── ShapeRegistry ─────────────────────────────────────────────────────────────

/// Propagates concrete shapes and dtypes through a validated DAG.
///
/// After calling `infer()`, every operator node in the graph has its
/// `output_shapes` and `output_dtypes` fields populated.
pub struct ShapeRegistry<'g> {
    graph: &'g mut Graph,
    /// Resolved shapes: (node_id, output_index) → Shape
    shapes: HashMap<(NodeId, usize), Shape>,
    /// Resolved dtypes: (node_id, output_index) → DType
    dtypes: HashMap<(NodeId, usize), DType>,
}

impl<'g> ShapeRegistry<'g> {
    pub fn new(graph: &'g mut Graph) -> Self {
        Self {
            graph,
            shapes: HashMap::new(),
            dtypes: HashMap::new(),
        }
    }

    /// Run shape inference in topological order.
    pub fn infer(&mut self) -> Result<()> {
        let order = self.graph.topological_order()?;
        // Collect all node data first to avoid borrow conflicts.
        let order_clone = order.clone();
        for id in order_clone {
            self.infer_node(id)?;
        }
        Ok(())
    }

    fn infer_node(&mut self, id: NodeId) -> Result<()> {
        // Retrieve node — we need to be careful about borrow rules.
        let node = match self.graph.get(id) {
            Some(n) => n.clone(),
            None => return Ok(()),
        };

        match &node {
            Node::Input { shape, dtype, .. } => {
                if let Some(s) = shape {
                    self.shapes.insert((id, 0), s.clone());
                }
                if let Some(d) = dtype {
                    self.dtypes.insert((id, 0), *d);
                }
            }

            Node::Constant { value, .. } => {
                self.shapes.insert((id, 0), value.shape().clone());
                self.dtypes.insert((id, 0), value.dtype());
            }

            Node::Operator {
                op,
                inputs,
                num_outputs,
                ..
            } => {
                // Collect input shapes/dtypes.
                let in_shapes: Vec<Option<Shape>> = inputs
                    .iter()
                    .map(|&inp| self.shapes.get(&(inp, 0)).cloned())
                    .collect();
                let in_dtypes: Vec<Option<DType>> = inputs
                    .iter()
                    .map(|&inp| self.dtypes.get(&(inp, 0)).copied())
                    .collect();

                let (out_shapes, out_dtypes) =
                    self.infer_op(op, &in_shapes, &in_dtypes, *num_outputs)?;

                for (i, s) in out_shapes.iter().enumerate() {
                    if let Some(s) = s {
                        self.shapes.insert((id, i), s.clone());
                    }
                }
                for (i, d) in out_dtypes.iter().enumerate() {
                    if let Some(d) = d {
                        self.dtypes.insert((id, i), *d);
                    }
                }

                // Write back inferred shapes/dtypes into the graph node.
                if let Some(Node::Operator {
                    output_shapes,
                    output_dtypes,
                    ..
                }) = self.graph.get_mut(id)
                {
                    *output_shapes = out_shapes;
                    *output_dtypes = out_dtypes;
                }
            }

            Node::Output { source, .. } => {
                if let Some(s) = self.shapes.get(&(*source, 0)).cloned() {
                    self.shapes.insert((id, 0), s);
                }
                if let Some(d) = self.dtypes.get(&(*source, 0)).copied() {
                    self.dtypes.insert((id, 0), d);
                }
            }
        }

        Ok(())
    }

    /// Get the inferred shape for a node's output.
    pub fn shape_of(&self, id: NodeId, output: usize) -> Option<&Shape> {
        self.shapes.get(&(id, output))
    }

    /// Get the inferred dtype for a node's output.
    pub fn dtype_of(&self, id: NodeId, output: usize) -> Option<DType> {
        self.dtypes.get(&(id, output)).copied()
    }

    // ── Per-op shape rules ────────────────────────────────────────────────────

    #[allow(clippy::type_complexity)]
    fn infer_op(
        &self,
        op: &OpType,
        in_shapes: &[Option<Shape>],
        in_dtypes: &[Option<DType>],
        num_outputs: usize,
    ) -> Result<(Vec<Option<Shape>>, Vec<Option<DType>>)> {
        let _dtype = in_dtypes.first().copied().flatten();

        let shapes = match op {
            OpType::Identity
            | OpType::Relu
            | OpType::Sigmoid
            | OpType::Tanh
            | OpType::Gelu
            | OpType::Silu
            | OpType::HardSwish
            | OpType::Neg
            | OpType::Abs
            | OpType::Sqrt
            | OpType::Exp
            | OpType::Log
            | OpType::Floor
            | OpType::Ceil
            | OpType::Round
            | OpType::Sign
            | OpType::Erf => {
                // Unary — same shape as input.
                vec![in_shapes.first().cloned().flatten()]
            }

            OpType::Add
            | OpType::Sub
            | OpType::Mul
            | OpType::Div
            | OpType::Pow
            | OpType::And
            | OpType::Or => {
                // Binary — broadcast.
                let s0 = in_shapes.first().cloned().flatten();
                let s1 = in_shapes.get(1).cloned().flatten();
                match (s0, s1) {
                    (Some(a), Some(b)) => vec![Some(a.broadcast_with(&b)?)],
                    (Some(a), None) | (None, Some(a)) => vec![Some(a)],
                    _ => vec![None],
                }
            }

            OpType::MatMul => {
                let s0 = in_shapes.first().cloned().flatten();
                let s1 = in_shapes.get(1).cloned().flatten();
                match (s0, s1) {
                    (Some(a), Some(b)) => {
                        let out = infer_matmul_shape(&a, &b).ok();
                        vec![out]
                    }
                    _ => vec![None],
                }
            }

            OpType::Gemm {
                trans_a, trans_b, ..
            } => {
                let s0 = in_shapes.first().cloned().flatten();
                let s1 = in_shapes.get(1).cloned().flatten();
                match (s0, s1) {
                    (Some(a), Some(b)) => {
                        let a_dims = a.dims().to_vec();
                        let b_dims = b.dims().to_vec();
                        if a_dims.len() == 2 && b_dims.len() == 2 {
                            let m = if *trans_a { a_dims[1] } else { a_dims[0] };
                            let n = if *trans_b { b_dims[0] } else { b_dims[1] };
                            vec![Some(Shape::new([m, n]))]
                        } else {
                            vec![None]
                        }
                    }
                    _ => vec![None],
                }
            }

            OpType::Softmax { .. }
            | OpType::LogSoftmax { .. }
            | OpType::LayerNorm { .. }
            | OpType::BatchNorm { .. }
            | OpType::RmsNorm { .. }
            | OpType::Dropout { .. }
            | OpType::LeakyRelu { .. }
            | OpType::Clip { .. } => {
                // Same shape as first input.
                vec![in_shapes.first().cloned().flatten()]
            }

            OpType::Reshape => {
                // Shape determined at runtime by second input.
                vec![None]
            }

            OpType::Transpose { perm } => {
                let s = in_shapes.first().cloned().flatten();
                match s {
                    Some(s) if s.ndim() == perm.len() => {
                        let new_dims: Vec<usize> = perm.iter().map(|&p| s.dims()[p]).collect();
                        vec![Some(Shape::new(new_dims))]
                    }
                    other => vec![other],
                }
            }

            OpType::Flatten { axis } => {
                let s = in_shapes.first().cloned().flatten();
                match s {
                    Some(s) => {
                        let ax = normalise_axis(*axis, s.ndim() as i64) as usize;
                        let outer: usize = s.dims()[..ax].iter().product();
                        let inner: usize = s.dims()[ax..].iter().product();
                        let outer = if outer == 0 { 1 } else { outer };
                        vec![Some(Shape::new([outer, inner]))]
                    }
                    None => vec![None],
                }
            }

            OpType::Concat { axis } => {
                let ax = *axis;
                let valid: Vec<Shape> = in_shapes.iter().filter_map(|s| s.clone()).collect();
                if valid.is_empty() {
                    vec![None]
                } else {
                    let rank = valid[0].ndim() as i64;
                    let ax = normalise_axis(ax, rank) as usize;
                    let mut dims = valid[0].dims().to_vec();
                    for s in valid.iter().skip(1) {
                        dims[ax] += s.dims()[ax];
                    }
                    vec![Some(Shape(dims))]
                }
            }

            OpType::ReduceSum { axes, keep_dims }
            | OpType::ReduceMean { axes, keep_dims }
            | OpType::ReduceMax { axes, keep_dims }
            | OpType::ReduceMin { axes, keep_dims } => {
                let s = in_shapes.first().cloned().flatten();
                match s {
                    Some(s) => vec![Some(reduce_shape(&s, axes, *keep_dims))],
                    None => vec![None],
                }
            }

            OpType::ArgMax { axis, keep_dims } | OpType::ArgMin { axis, keep_dims } => {
                let s = in_shapes.first().cloned().flatten();
                match s {
                    Some(s) => {
                        let mut dims = s.dims().to_vec();
                        let ax = normalise_axis(*axis, s.ndim() as i64) as usize;
                        if *keep_dims {
                            dims[ax] = 1;
                        } else {
                            dims.remove(ax);
                        }
                        vec![Some(Shape(dims))]
                    }
                    None => vec![None],
                }
            }

            OpType::Cast { to } => {
                let shape = in_shapes.first().cloned().flatten();
                return Ok((vec![shape], vec![Some(*to)]));
            }

            OpType::Conv2d {
                kernel_shape,
                pads,
                strides,
                ..
            } => {
                let s = in_shapes.first().cloned().flatten();
                let w = in_shapes.get(1).cloned().flatten();
                match (s, w) {
                    (Some(s), Some(w)) if s.ndim() == 4 && w.ndim() == 4 => {
                        let (n, _, h, ww) = (s.dims()[0], s.dims()[1], s.dims()[2], s.dims()[3]);
                        let c_out = w.dims()[0];
                        let kh = kernel_shape[0];
                        let kw = kernel_shape[1];
                        let ph = pads[0] + pads[2];
                        let pw = pads[1] + pads[3];
                        let out_h = (h + ph - kh) / strides[0] + 1;
                        let out_w = (ww + pw - kw) / strides[1] + 1;
                        vec![Some(Shape::new([n, c_out, out_h, out_w]))]
                    }
                    _ => vec![None],
                }
            }

            _ => vec![None; num_outputs],
        };

        let dtypes = in_dtypes
            .first()
            .copied()
            .flatten()
            .map(|d| vec![Some(d); num_outputs])
            .unwrap_or_else(|| vec![None; num_outputs]);

        Ok((shapes, dtypes))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn infer_matmul_shape(a: &Shape, b: &Shape) -> Result<Shape> {
    match (a.ndim(), b.ndim()) {
        (2, 2) => {
            let (m, k1) = (a.dims()[0], a.dims()[1]);
            let (k2, n) = (b.dims()[0], b.dims()[1]);
            if k1 != k2 {
                return Err(SapientError::ShapeMismatch {
                    expected: vec![m, k1],
                    got: vec![k2, n],
                });
            }
            Ok(Shape::new([m, n]))
        }
        (a_r, b_r) if a_r >= 2 && b_r >= 2 => {
            // Batched: broadcast batch dims.
            let a_batch = &a.dims()[..a_r - 2];
            let b_batch = &b.dims()[..b_r - 2];
            let a_bs = Shape::from(a_batch);
            let b_bs = Shape::from(b_batch);
            let batch = a_bs.broadcast_with(&b_bs)?;
            let m = a.dims()[a_r - 2];
            let n = b.dims()[b_r - 1];
            let mut dims = batch.0;
            dims.push(m);
            dims.push(n);
            Ok(Shape(dims))
        }
        _ => Err(SapientError::RankMismatch {
            expected: 2,
            got: a.ndim(),
        }),
    }
}

fn reduce_shape(s: &Shape, axes: &[i64], keep_dims: bool) -> Shape {
    let mut reduced: std::collections::HashSet<usize> = axes
        .iter()
        .map(|&a| normalise_axis(a, s.ndim() as i64) as usize)
        .collect();
    if axes.is_empty() {
        // Reduce all dims.
        reduced = (0..s.ndim()).collect();
    }
    let dims: Vec<usize> = s
        .dims()
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if reduced.contains(&i) {
                if keep_dims {
                    Some(1)
                } else {
                    None
                }
            } else {
                Some(d)
            }
        })
        .collect();
    Shape(dims)
}

fn normalise_axis(axis: i64, rank: i64) -> i64 {
    if axis < 0 {
        rank + axis
    } else {
        axis
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::Graph, op::OpType};
    use sapient_core::{DType, Tensor};

    fn make_mlp_graph() -> Graph {
        let mut g = Graph::new("mlp");
        let x = g.add_input("x", Some(Shape::new([2, 4])), Some(DType::F32));
        let w = g.add_constant(
            Tensor::zeros(vec![4, 8], DType::F32).unwrap(),
            Some("W".into()),
        );
        let mm = g.add_op(OpType::MatMul, vec![x, w], 1, None);
        let r = g.add_op(OpType::Relu, vec![mm], 1, None);
        g.mark_output(r, "out");
        g
    }

    #[test]
    fn infer_mlp() {
        let mut g = make_mlp_graph();
        let mut reg = ShapeRegistry::new(&mut g);
        reg.infer().unwrap();

        // MatMul output: [2, 8]
        // Relu: same shape
        // We can't easily get the node IDs back here; just check no error.
    }
}
