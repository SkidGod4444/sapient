// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! GPU device acquisition and lifetime management.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use thiserror::Error;

/// Errors raised while bringing up or running the wgpu backend.
#[derive(Debug, Error)]
pub enum WgpuError {
    #[error("no compatible GPU adapter found (need Vulkan, DX12, Metal, or GL)")]
    NoAdapter,
    #[error("failed to request GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("GPU buffer mapping failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),
    #[error("shape mismatch: {0}")]
    Shape(String),
}

/// A live GPU context: instance, adapter, logical device, and command queue.
///
/// One `WgpuContext` is created per process and shared across kernels. It runs on
/// whatever portable backend the platform provides — Vulkan or DX12 on
/// Intel/AMD/Nvidia (Linux/Windows), Metal on Apple — so the same WGSL kernels run
/// everywhere without change.
pub struct WgpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    adapter_name: String,
    backend: wgpu::Backend,
    /// Whether the device has the `shader-f16` feature (f16 storage in WGSL).
    /// Weights are stored f16 when true (half the VRAM), still f32-accumulated.
    pub(crate) has_f16: bool,
    /// Max single storage-buffer binding the adapter allows (wgpu default is only
    /// 128 MiB — too small for an lm_head/embedding tensor, so we raise it).
    pub(crate) max_binding_bytes: u64,
    /// Compiled compute pipelines, keyed by a stable label, so each WGSL kernel is
    /// compiled once and reused across every decode step.
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    /// Open command batch (Phase 7.4). While `Some`, kernels record into this
    /// encoder instead of submitting one queue submission each; `flush_batch`
    /// (called automatically by `download_f32`) submits the whole batch at once.
    /// A decode step is ~450 kernels — batching them cuts ~450 submissions per
    /// token to 1, removing the fixed per-submission CPU cost from the hot loop.
    batch: Mutex<Option<wgpu::CommandEncoder>>,
}

impl WgpuContext {
    /// Acquire a high-performance GPU adapter and logical device (blocking).
    ///
    /// Prefers a discrete GPU when one is present. Returns `WgpuError::NoAdapter`
    /// on a machine with no usable GPU (the caller should fall back to CPU).
    pub fn new() -> Result<Self, WgpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Adapter-only probe: does this machine have a usable GPU? Creates no
    /// logical device, so it is cheap enough for backend auto-selection to call
    /// before committing weights to a GPU engine build (which cannot fall back
    /// once the weight map is consumed).
    pub fn adapter_available() -> bool {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            });
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .is_some()
        })
    }

    async fn new_async() -> Result<Self, WgpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // Vulkan + Metal + DX12 (skips GL fallback)
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None, // headless compute — no window
            })
            .await
            .ok_or(WgpuError::NoAdapter)?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = info.backend;

        // Raise limits toward what the adapter actually supports. wgpu's defaults
        // cap a single storage binding at 128 MiB, which is smaller than one
        // lm_head/embedding tensor — use the adapter's real maxima so big weight
        // buffers bind. f16 halves weight VRAM (still f32-accumulated in shaders).
        let adapter_limits = adapter.limits();
        let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let required_features = if has_f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };
        let max_binding_bytes = adapter_limits.max_storage_buffer_binding_size as u64;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("sapient-wgpu-device"),
                    required_features,
                    required_limits: adapter_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        tracing::info!(
            adapter = %adapter_name, backend = ?backend, has_f16,
            max_binding_mib = max_binding_bytes / (1 << 20),
            "wgpu GPU device ready"
        );
        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
            has_f16,
            max_binding_bytes,
            pipelines: Mutex::new(HashMap::new()),
            batch: Mutex::new(None),
        })
    }

    /// Start (or continue) a command batch: until [`Self::flush_batch`], every
    /// kernel dispatch and buffer copy records into one shared command encoder
    /// instead of paying a queue submission each. Execution order is identical —
    /// WebGPU guarantees commands execute in recording order. Idempotent.
    pub fn begin_batch(&self) {
        let mut batch = self.batch.lock().unwrap();
        if batch.is_none() {
            *batch = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("batch"),
                    }),
            );
        }
    }

    /// Submit the open batch as a single queue submission (no-op when none is
    /// open). [`WgpuContext::download_f32`] calls this automatically, so readbacks
    /// always observe every recorded kernel.
    pub fn flush_batch(&self) {
        if let Some(enc) = self.batch.lock().unwrap().take() {
            self.queue.submit(Some(enc.finish()));
        }
    }

    /// Record commands via `f`: into the open batch when one exists, otherwise
    /// into an ephemeral encoder submitted immediately (the pre-batching
    /// behaviour, still used by tests and the Whisper engine).
    pub(crate) fn with_encoder(&self, f: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut batch = self.batch.lock().unwrap();
        if let Some(enc) = batch.as_mut() {
            f(enc);
        } else {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            f(&mut enc);
            self.queue.submit(Some(enc.finish()));
        }
    }

    /// Get-or-compile a compute pipeline for `label` from `wgsl` source (entry
    /// point `cs_main`). Compiled once per label, then cached — every decode step
    /// reuses the same pipeline (no per-call shader compilation).
    pub(crate) fn pipeline(&self, label: &str, wgsl: &str) -> Arc<wgpu::ComputePipeline> {
        if let Some(p) = self.pipelines.lock().unwrap().get(label) {
            return Arc::clone(p);
        }
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: "cs_main",
                compilation_options: Default::default(),
                cache: None,
            });
        let pipeline = Arc::new(pipeline);
        self.pipelines
            .lock()
            .unwrap()
            .insert(label.to_string(), Arc::clone(&pipeline));
        pipeline
    }

    /// True when the device supports f16 storage (`shader-f16`).
    pub fn has_f16(&self) -> bool {
        self.has_f16
    }

    /// Max single storage-buffer binding (bytes) the device allows. The engine
    /// uses this to decide when a weight tensor must be chunked across bindings.
    pub fn max_binding_bytes(&self) -> u64 {
        self.max_binding_bytes
    }

    /// Human-readable adapter description, e.g. "AMD Radeon RX 7600 (Vulkan)".
    pub fn adapter_label(&self) -> String {
        format!("{} ({:?})", self.adapter_name, self.backend)
    }

    /// The raw adapter name (e.g. "Intel(R) Arc(TM) A770 Graphics").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// The active wgpu backend (Vulkan / Dx12 / Metal / Gl).
    pub fn backend(&self) -> wgpu::Backend {
        self.backend
    }
}
