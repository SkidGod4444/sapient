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
//! # Status — foundation
//!
//! This is the foundation layer: GPU device acquisition ([`WgpuContext`]) plus the
//! representative linear-projection kernels ([`WgpuContext::matmul_nt_f32`] and the
//! Q8_0-quantized [`WgpuContext::matmul_nt_q8_0`]), validated against a host
//! reference. The remaining transformer kernels (attention, RMSNorm, RoPE, SwiGLU,
//! embedding gather) and the `ForwardEngine` integration build on this base.
//!
//! # Example
//!
//! ```no_run
//! use sapient_backends_wgpu::WgpuContext;
//!
//! let ctx = WgpuContext::new().expect("a GPU is available");
//! println!("running on {}", ctx.adapter_label());
//!
//! // out[1,2] = x[1,3] @ w[2,3]^T
//! let x = [1.0, 2.0, 3.0];
//! let w = [1.0, 0.0, 0.0,   0.0, 1.0, 1.0];
//! let out = ctx.matmul_nt_f32(&x, &w, 1, 3, 2).unwrap();
//! assert_eq!(out, vec![1.0, 5.0]);
//! ```

mod context;
mod matmul;

pub use context::{WgpuContext, WgpuError};
pub use matmul::quantize_q8_0_rows;
