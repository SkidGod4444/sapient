//! GPU device acquisition and lifetime management.

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
}

impl WgpuContext {
    /// Acquire a high-performance GPU adapter and logical device (blocking).
    ///
    /// Prefers a discrete GPU when one is present. Returns `WgpuError::NoAdapter`
    /// on a machine with no usable GPU (the caller should fall back to CPU).
    pub fn new() -> Result<Self, WgpuError> {
        pollster::block_on(Self::new_async())
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("sapient-wgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        tracing::info!(adapter = %adapter_name, backend = ?backend, "wgpu GPU device ready");
        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
        })
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
