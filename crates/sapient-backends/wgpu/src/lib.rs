// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Cross-platform GPU backend for SAPIENT, built on [`wgpu`].
//!
//! This crate provides GPU-accelerated inference kernels that run on **any** modern
//! GPU through a portable compute API:
//!
//! | Platform | API used by wgpu |
//! |----------|------------------|
//! | Linux / Windows — Intel, AMD, Nvidia | Vulkan (or DX12 on Windows) |
//! | macOS — Apple Silicon / AMD          | Metal |
//!
//! The **same WGSL compute shaders** run on every backend, so a kernel validated on
//! one machine (e.g. an M-series Mac via Metal) runs unchanged on an Intel Arc or
//! AMD Radeon card via Vulkan.
//!
//! # Layout
//!
//! [`WgpuContext`] acquires the device (adapter-max limits, `SHADER_F16` when
//! available, a compute-pipeline cache). On top of it sits the GPU-resident
//! compute layer the `WgpuForwardEngine` drives: [`GpuBuffer`] (f32 tensors in
//! storage buffers), [`GpuQ8Buffer`] and [`GpuQ4KBuffer`] (Q8_0 / Q4_K tensors kept
//! quantized on-device, dequantized inside the shader — no host-side f32 expansion),
//! plus the kernels `rms_norm`, `layer_norm`, `matmul_nt`, `matmul_nt_q8_0`,
//! `matmul_nt_q4_k`, `rope`, `attention` (causal + non-causal GQA FlashDecoding),
//! `swiglu`/`add`/`add_bias`/`gelu_erf`, `embed`/`embed_q8_0`/`embed_q4_k`,
//! `transpose_heads`, and the KV-cache append `copy_range`.
//! Each kernel is validated against a CPU reference in `tests/resident.rs`.
//!
//! # Example
//!
//! ```no_run
//! use sapient_backends_wgpu::WgpuContext;
//!
//! let ctx = WgpuContext::new().expect("a GPU is available");
//! println!("running on {}", ctx.adapter_label());
//!
//! // out[1,2] = x[1,3] @ w[2,3]^T, GPU-resident
//! let x = ctx.upload_f32(&[1.0, 2.0, 3.0], "x");
//! let w = ctx.upload_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 1.0], "w");
//! let out = ctx.download_f32(&ctx.matmul_nt(&x, &w, 1, 3, 2)).unwrap();
//! assert_eq!(out, vec![1.0, 5.0]);
//! ```

mod context;
mod quant;
mod resident;

pub use context::{WgpuContext, WgpuError};
pub use quant::{GpuQ4KBuffer, GpuQ6KBuffer, GpuQ8Buffer};
pub use resident::GpuBuffer;
