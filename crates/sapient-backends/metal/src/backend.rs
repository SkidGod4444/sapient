// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `MlxBackend` — Apple Silicon GPU execution via MLX.
//!
//! MLX unified memory model: tensors live in shared CPU/GPU memory.
//! There is zero copy overhead between CPU preprocessing and GPU matmul.
//!
//! This is a scaffold. Wire in `mlx-rs` ops as they become stable.

use std::collections::HashMap;
use std::process::Command;

use sapient_backends_cpu::backend::{CpuBackend, ExecutionBackend};
use sapient_core::buffer::BufferHandle;
use sapient_core::error::Result;
use sapient_core::{DType, Tensor};
use sapient_ir::graph::Graph;
use sapient_ir::op::OpType;

// ── MlxBackend ────────────────────────────────────────────────────────────────

/// Apple Silicon GPU backend using the MLX framework.
///
/// Falls back to the CPU backend for unsupported ops (progressive offload).
#[derive(Debug)]
pub struct MlxBackend {
    /// Device index (0 = default GPU, −1 = CPU fallback).
    device_id: i32,
    cpu: CpuBackend,
}

impl MlxBackend {
    /// Create a backend on the default Metal GPU device.
    pub fn new() -> Self {
        tracing::info!("MlxBackend: initializing on Apple Silicon GPU");
        Self {
            device_id: 0,
            cpu: CpuBackend::default(),
        }
    }

    /// Force CPU-only execution (useful for debugging correctness).
    pub fn cpu_fallback() -> Self {
        Self {
            device_id: -1,
            cpu: CpuBackend::default(),
        }
    }

    /// Human-readable Metal device name when it can be discovered cheaply.
    pub fn device_name() -> Option<String> {
        let output = Command::new("system_profiler")
            .args(["SPDisplaysDataType"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.lines().map(str::trim).find_map(|line| {
            line.strip_prefix("Chipset Model:")
                .or_else(|| line.strip_prefix("GPU:"))
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        })
    }
}

impl Default for MlxBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ── ExecutionBackend impl ─────────────────────────────────────────────────────
//
// When mlx-rs is added as a dependency, replace the todo!() stubs with:
//   mlx::ops::matmul(&a, &b)     → MatMul
//   mlx::ops::add(&a, &b)        → Add
//   mlx::ops::relu(&x)           → Relu
//   mlx::ops::softmax(&x, axis)  → Softmax
//   mlx::eval(&[&result])        → force evaluation (lazy eval flush)

impl ExecutionBackend for MlxBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<BufferHandle> {
        self.cpu.allocate(shape, dtype)
    }

    fn execute(&self, graph: &Graph, inputs: HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        // Direct Metal kernels are wired incrementally. Until an op has a
        // native kernel, delegate through the CPU reference path so backend
        // selection is explicit without compromising correctness.
        tracing::warn!(
            device_id = self.device_id,
            "Metal graph kernels are not yet complete — falling back to CPU for this graph"
        );
        self.cpu.execute(graph, inputs)
    }

    fn supports_op(&self, op: &OpType) -> bool {
        self.cpu.supports_op(op)
    }

    fn is_available() -> bool
    where
        Self: Sized,
    {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }
}
