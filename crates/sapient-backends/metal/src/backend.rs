//! `MlxBackend` — Apple Silicon GPU execution via MLX.
//!
//! MLX unified memory model: tensors live in shared CPU/GPU memory.
//! There is zero copy overhead between CPU preprocessing and GPU matmul.
//!
//! This is a scaffold. Wire in `mlx-rs` ops as they become stable.

use std::collections::HashMap;

use sapient_core::{DType, Tensor};
use sapient_core::error::{Result, SapientError};
use sapient_ir::graph::Graph;
use sapient_ir::node::{Node, NodeId};
use sapient_ir::op::OpType;

// ── MlxBackend ────────────────────────────────────────────────────────────────

/// Apple Silicon GPU backend using the MLX framework.
///
/// Falls back to the CPU backend for unsupported ops (progressive offload).
#[derive(Debug)]
pub struct MlxBackend {
    /// Device index (0 = default GPU, −1 = CPU fallback).
    device_id: i32,
}

impl MlxBackend {
    /// Create a backend on the default Metal GPU device.
    pub fn new() -> Self {
        tracing::info!("MlxBackend: initializing on Apple Silicon GPU");
        Self { device_id: 0 }
    }

    /// Force CPU-only execution (useful for debugging correctness).
    pub fn cpu_fallback() -> Self {
        Self { device_id: -1 }
    }
}

impl Default for MlxBackend {
    fn default() -> Self { Self::new() }
}

// ── ExecutionBackend impl ─────────────────────────────────────────────────────
//
// When mlx-rs is added as a dependency, replace the todo!() stubs with:
//   mlx::ops::matmul(&a, &b)     → MatMul
//   mlx::ops::add(&a, &b)        → Add
//   mlx::ops::relu(&x)           → Relu
//   mlx::ops::softmax(&x, axis)  → Softmax
//   mlx::eval(&[&result])        → force evaluation (lazy eval flush)

impl sapient_backends_cpu::backend::ExecutionBackend for MlxBackend {
    fn name(&self) -> &str { "metal-mlx" }

    fn execute(
        &self,
        graph: &Graph,
        inputs: HashMap<String, Tensor>,
    ) -> Result<Vec<Tensor>> {
        // For now: delegate to CPU backend until mlx-rs ops are wired.
        // TODO: replace with MLX dispatch per op.
        tracing::warn!(
            "MlxBackend: MLX not yet wired — falling back to CPU for all ops"
        );
        let cpu = sapient_backends_cpu::backend::CpuBackend::default();
        cpu.execute(graph, inputs)
    }
}
