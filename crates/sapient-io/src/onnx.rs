#![allow(dead_code)]

//! Minimal ONNX ModelProto parser → SAPIENT IR converter.
//!
//! We implement our own protobuf decoder for ONNX rather than pulling in full
//! code-generation, keeping compile times low.  This supports Opset 17+.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use sapient_core::error::{Result, SapientError};
use sapient_core::{DType, Shape, Tensor};
use sapient_ir::graph::Graph;
use sapient_ir::node::NodeId;
use sapient_ir::op::OpType;

// ── Protobuf wire-type helpers ────────────────────────────────────────────────

mod proto {
    /// Decode a varint from a byte slice, returning (value, bytes_consumed).
    pub fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
        let mut val: u64 = 0;
        let mut shift = 0u32;
        for (i, &b) in buf.iter().enumerate() {
            val |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some((val, i + 1));
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
        None
    }

    pub fn read_tag(buf: &[u8]) -> Option<(u32, u8, usize)> {
        let (v, n) = read_varint(buf)?;
        Some(((v >> 3) as u32, (v & 7) as u8, n))
    }

    pub fn read_len_delimited(buf: &[u8]) -> Option<(&[u8], usize)> {
        let (len, n) = read_varint(buf)?;
        let end = n + len as usize;
        if end > buf.len() {
            return None;
        }
        Some((&buf[n..end], end))
    }

    pub fn read_f32(buf: &[u8]) -> Option<(f32, usize)> {
        if buf.len() < 4 {
            return None;
        }
        let v = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        Some((v, 4))
    }

    pub fn read_i32(buf: &[u8]) -> Option<(i32, usize)> {
        if buf.len() < 4 {
            return None;
        }
        let v = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        Some((v, 4))
    }

    pub fn read_i64(buf: &[u8]) -> Option<(i64, usize)> {
        if buf.len() < 8 {
            return None;
        }
        let v = i64::from_le_bytes(buf[..8].try_into().unwrap());
        Some((v, 8))
    }
}

// ── ONNX field numbers (from proto spec) ─────────────────────────────────────

// ModelProto
const MODEL_GRAPH: u32 = 7;

// GraphProto
const GRAPH_NODE: u32 = 1;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;
const GRAPH_INIT: u32 = 5; // initializer (weights)

// NodeProto
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const NODE_OP_TYPE: u32 = 4;
const NODE_NAME: u32 = 3;
const NODE_ATTR: u32 = 5;

// TensorProto
const TENSOR_DIMS: u32 = 1;
const TENSOR_DTYPE: u32 = 2;
const TENSOR_NAME: u32 = 8;
const TENSOR_FLOAT_DATA: u32 = 4;
const TENSOR_RAW_DATA: u32 = 9;
const TENSOR_INT32_DATA: u32 = 6;
const TENSOR_INT64_DATA: u32 = 7;
const TENSOR_DOUBLE_DATA: u32 = 10;

// ValueInfoProto
const VI_NAME: u32 = 1;
const VI_TYPE: u32 = 2;

// ── Parsed intermediate structures ───────────────────────────────────────────

#[derive(Debug, Default)]
struct OnnxNode {
    inputs: Vec<String>,
    outputs: Vec<String>,
    op_type: String,
    name: String,
}

#[derive(Debug, Default)]
struct OnnxTensor {
    name: String,
    dims: Vec<i64>,
    dtype: i32,
    data: Vec<f32>,
}

#[derive(Debug, Default)]
struct OnnxGraph {
    nodes: Vec<OnnxNode>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializers: Vec<OnnxTensor>,
}

// ── Parser ────────────────────────────────────────────────────────────────────

fn parse_string(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf).into_owned()
}

fn parse_tensor(buf: &[u8]) -> OnnxTensor {
    let mut t = OnnxTensor::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, wire, n) = match proto::read_tag(&buf[pos..]) {
            Some(v) => v,
            None => break,
        };
        pos += n;
        match (field, wire) {
            (TENSOR_DIMS, 0) => {
                if let Some((v, n)) = proto::read_varint(&buf[pos..]) {
                    t.dims.push(v as i64);
                    pos += n;
                }
            }
            (TENSOR_DTYPE, 0) => {
                if let Some((v, n)) = proto::read_varint(&buf[pos..]) {
                    t.dtype = v as i32;
                    pos += n;
                }
            }
            (TENSOR_NAME, 2) => {
                if let Some((slice, n)) = proto::read_len_delimited(&buf[pos..]) {
                    t.name = parse_string(slice);
                    pos += n;
                }
            }
            (TENSOR_FLOAT_DATA, 2) => {
                if let Some((slice, n)) = proto::read_len_delimited(&buf[pos..]) {
                    t.data = slice
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    pos += n;
                }
            }
            (TENSOR_RAW_DATA, 2) => {
                if let Some((slice, n)) = proto::read_len_delimited(&buf[pos..]) {
                    // For F32 raw data.
                    if t.dtype == 1 {
                        t.data = slice
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                    }
                    pos += n;
                }
            }
            (TENSOR_DOUBLE_DATA, 2) => {
                if let Some((slice, n)) = proto::read_len_delimited(&buf[pos..]) {
                    t.data = slice
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect();
                    pos += n;
                }
            }
            // Skip unknown fields.
            (_, 0) => {
                if let Some((_, n)) = proto::read_varint(&buf[pos..]) {
                    pos += n;
                }
            }
            (_, 2) => {
                if let Some((_, n)) = proto::read_len_delimited(&buf[pos..]) {
                    pos += n;
                }
            }
            (_, 5) => {
                pos += 4;
            }
            (_, 1) => {
                pos += 8;
            }
            _ => break,
        }
    }
    t
}

fn parse_node(buf: &[u8]) -> OnnxNode {
    let mut n = OnnxNode::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, wire, consumed) = match proto::read_tag(&buf[pos..]) {
            Some(v) => v,
            None => break,
        };
        pos += consumed;
        match (field, wire) {
            (NODE_INPUT, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    n.inputs.push(parse_string(slice));
                    pos += c;
                }
            }
            (NODE_OUTPUT, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    n.outputs.push(parse_string(slice));
                    pos += c;
                }
            }
            (NODE_OP_TYPE, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    n.op_type = parse_string(slice);
                    pos += c;
                }
            }
            (NODE_NAME, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    n.name = parse_string(slice);
                    pos += c;
                }
            }
            // Skip attributes for now (handled at dispatch time via defaults).
            (NODE_ATTR, 2) => {
                if let Some((_, c)) = proto::read_len_delimited(&buf[pos..]) {
                    pos += c;
                }
            }
            (_, 0) => {
                if let Some((_, c)) = proto::read_varint(&buf[pos..]) {
                    pos += c;
                }
            }
            (_, 2) => {
                if let Some((_, c)) = proto::read_len_delimited(&buf[pos..]) {
                    pos += c;
                }
            }
            (_, 5) => {
                pos += 4;
            }
            (_, 1) => {
                pos += 8;
            }
            _ => break,
        }
    }
    n
}

fn parse_vi_name(buf: &[u8]) -> String {
    let mut pos = 0;
    while pos < buf.len() {
        let (field, wire, n) = match proto::read_tag(&buf[pos..]) {
            Some(v) => v,
            None => break,
        };
        pos += n;
        if field == VI_NAME && wire == 2 {
            if let Some((slice, _)) = proto::read_len_delimited(&buf[pos..]) {
                return parse_string(slice);
            }
        }
        // Skip this field.
        match wire {
            0 => {
                if let Some((_, n)) = proto::read_varint(&buf[pos..]) {
                    pos += n;
                }
            }
            2 => {
                if let Some((_, n)) = proto::read_len_delimited(&buf[pos..]) {
                    pos += n;
                }
            }
            5 => {
                pos += 4;
            }
            1 => {
                pos += 8;
            }
            _ => break,
        }
    }
    String::new()
}

fn parse_graph(buf: &[u8]) -> OnnxGraph {
    let mut g = OnnxGraph::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, wire, n) = match proto::read_tag(&buf[pos..]) {
            Some(v) => v,
            None => break,
        };
        pos += n;
        match (field, wire) {
            (GRAPH_NODE, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    g.nodes.push(parse_node(slice));
                    pos += c;
                }
            }
            (GRAPH_INPUT, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    let name = parse_vi_name(slice);
                    if !name.is_empty() {
                        g.inputs.push(name);
                    }
                    pos += c;
                }
            }
            (GRAPH_OUTPUT, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    let name = parse_vi_name(slice);
                    if !name.is_empty() {
                        g.outputs.push(name);
                    }
                    pos += c;
                }
            }
            (GRAPH_INIT, 2) => {
                if let Some((slice, c)) = proto::read_len_delimited(&buf[pos..]) {
                    g.initializers.push(parse_tensor(slice));
                    pos += c;
                }
            }
            (_, 0) => {
                if let Some((_, c)) = proto::read_varint(&buf[pos..]) {
                    pos += c;
                }
            }
            (_, 2) => {
                if let Some((_, c)) = proto::read_len_delimited(&buf[pos..]) {
                    pos += c;
                }
            }
            (_, 5) => {
                pos += 4;
            }
            (_, 1) => {
                pos += 8;
            }
            _ => break,
        }
    }
    g
}

fn parse_model(buf: &[u8]) -> OnnxGraph {
    let mut pos = 0;
    while pos < buf.len() {
        let (field, wire, n) = match proto::read_tag(&buf[pos..]) {
            Some(v) => v,
            None => break,
        };
        pos += n;
        if field == MODEL_GRAPH && wire == 2 {
            if let Some((slice, _)) = proto::read_len_delimited(&buf[pos..]) {
                return parse_graph(slice);
            }
        }
        match wire {
            0 => {
                if let Some((_, n)) = proto::read_varint(&buf[pos..]) {
                    pos += n;
                }
            }
            2 => {
                if let Some((_, n)) = proto::read_len_delimited(&buf[pos..]) {
                    pos += n;
                }
            }
            5 => {
                pos += 4;
            }
            1 => {
                pos += 8;
            }
            _ => break,
        }
    }
    OnnxGraph::default()
}

// ── Op name → OpType mapping ─────────────────────────────────────────────────

fn onnx_op_to_sapient(op: &str) -> OpType {
    match op {
        "MatMul" => OpType::MatMul,
        "Gemm" => OpType::Gemm {
            alpha: ordered_float::OrderedFloat(1.0),
            beta: ordered_float::OrderedFloat(1.0),
            trans_a: false,
            trans_b: false,
        },
        "Add" => OpType::Add,
        "Sub" => OpType::Sub,
        "Mul" => OpType::Mul,
        "Div" => OpType::Div,
        "Relu" => OpType::Relu,
        "Sigmoid" => OpType::Sigmoid,
        "Tanh" => OpType::Tanh,
        "Gelu" => OpType::Gelu,
        "Sqrt" => OpType::Sqrt,
        "Exp" => OpType::Exp,
        "Log" => OpType::Log,
        "Neg" => OpType::Neg,
        "Abs" => OpType::Abs,
        "Flatten" => OpType::Flatten { axis: 1 },
        "Reshape" => OpType::Reshape,
        "Transpose" => OpType::Transpose { perm: vec![] },
        "Softmax" => OpType::Softmax { axis: -1 },
        "LogSoftmax" => OpType::LogSoftmax { axis: -1 },
        "LayerNormalization" => OpType::LayerNorm {
            axis: -1,
            epsilon: ordered_float::OrderedFloat(1e-5),
        },
        "ReduceSum" => OpType::ReduceSum {
            axes: vec![],
            keep_dims: false,
        },
        "ReduceMean" => OpType::ReduceMean {
            axes: vec![],
            keep_dims: false,
        },
        "ReduceMax" => OpType::ReduceMax {
            axes: vec![],
            keep_dims: false,
        },
        "Conv" => OpType::Conv2d {
            kernel_shape: [3, 3],
            pads: [0, 0, 0, 0],
            strides: [1, 1],
            dilations: [1, 1],
            groups: 1,
        },
        "MaxPool" => OpType::MaxPool {
            kernel_shape: [2, 2],
            pads: [0, 0, 0, 0],
            strides: [2, 2],
        },
        "Erf" => OpType::Erf,
        "Identity" => OpType::Identity,
        "Concat" => OpType::Concat { axis: 0 },
        "Clip" => OpType::Clip {
            min: None,
            max: None,
        },
        _ => OpType::Identity, // unknown op → passthrough
    }
}

// ── OnnxLoader ────────────────────────────────────────────────────────────────

pub struct OnnxLoader;

impl OnnxLoader {
    /// Load an ONNX model file and convert it to a SAPIENT `Graph`.
    pub fn load(path: &Path) -> Result<Graph> {
        let bytes = fs::read(path)
            .map_err(|e| SapientError::ModelNotFound(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    /// Convert raw ONNX protobuf bytes to a SAPIENT `Graph`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Graph> {
        let onnx_graph = parse_model(bytes);

        let mut graph = Graph::new("onnx_model");

        // name → NodeId map for wiring edges.
        let mut name_to_id: HashMap<String, NodeId> = HashMap::new();

        // Add initializers as constants.
        for init in &onnx_graph.initializers {
            let dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
            let shape = if dims.is_empty() {
                Shape::new([1])
            } else {
                Shape::new(dims)
            };
            let tensor = if init.data.is_empty() {
                Tensor::zeros(shape, DType::F32)
                    .map_err(|e| SapientError::OnnxParseError(e.to_string()))?
            } else {
                Tensor::from_f32(&init.data, shape)
                    .map_err(|e| SapientError::OnnxParseError(e.to_string()))?
            };
            let id = graph.add_constant(tensor, Some(init.name.clone()));
            name_to_id.insert(init.name.clone(), id);
        }

        // Add graph inputs.
        for input_name in &onnx_graph.inputs {
            // Skip if already an initializer.
            if name_to_id.contains_key(input_name) {
                continue;
            }
            let id = graph.add_input(input_name, None, Some(DType::F32));
            name_to_id.insert(input_name.clone(), id);
        }

        // Add operator nodes.
        for node in &onnx_graph.nodes {
            let op = onnx_op_to_sapient(&node.op_type);
            let input_ids: Vec<NodeId> = node
                .inputs
                .iter()
                .filter_map(|n| name_to_id.get(n).copied())
                .collect();
            let num_outputs = node.outputs.len().max(1);
            let node_name = if node.name.is_empty() {
                None
            } else {
                Some(node.name.clone())
            };
            let id = graph.add_op(op, input_ids, num_outputs, node_name);
            for (i, out_name) in node.outputs.iter().enumerate() {
                if i == 0 {
                    name_to_id.insert(out_name.clone(), id);
                }
                // Multi-output nodes: only first output is tracked in this minimal impl.
            }
        }

        // Mark outputs.
        for out_name in &onnx_graph.outputs {
            if let Some(&src_id) = name_to_id.get(out_name) {
                graph.mark_output(src_id, out_name);
            }
        }

        graph
            .validate()
            .map_err(|e| SapientError::OnnxParseError(e.to_string()))?;

        tracing::info!(nodes = graph.node_count(), "ONNX model loaded successfully");
        Ok(graph)
    }
}
