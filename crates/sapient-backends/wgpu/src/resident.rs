//! GPU-resident compute: tensors that live in GPU storage buffers and kernels
//! that operate buffer→buffer, so a forward pass uploads weights once and only
//! reads back the final logits (no per-op CPU↔GPU round-trips — the key to decode
//! throughput, mirroring the MLX engine's on-device graph).
//!
//! This is the foundation the `WgpuForwardEngine` builds on. Kernels are added
//! incrementally; each is validated against a CPU reference (see the crate tests).

use wgpu::util::DeviceExt;

use crate::context::{WgpuContext, WgpuError};

/// An f32 tensor resident in a GPU storage buffer. `len` is the element count;
/// callers track logical shape separately (kernels take explicit dims).
pub struct GpuBuffer {
    pub(crate) buf: wgpu::Buffer,
    /// Number of f32 elements.
    pub len: usize,
}

impl WgpuContext {
    /// Upload an f32 slice into a new GPU storage buffer (STORAGE|COPY_SRC|COPY_DST).
    pub fn upload_f32(&self, data: &[f32], label: &str) -> GpuBuffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        GpuBuffer {
            buf,
            len: data.len(),
        }
    }

    /// Allocate an uninitialized (zeroed) GPU storage buffer of `len` f32 elements.
    pub fn alloc_f32(&self, len: usize, label: &str) -> GpuBuffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuBuffer { buf, len }
    }

    /// Read a GPU buffer back to host (blocking). Used only for final logits in the
    /// hot path; freely in tests.
    pub fn download_f32(&self, b: &GpuBuffer) -> Result<Vec<f32>, WgpuError> {
        let size = (b.len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download-staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b.buf, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| WgpuError::Shape("map channel closed".into()))??;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Small uniform buffer from a `#[repr(C)]` Pod struct (kernel params).
    pub(crate) fn uniform<T: bytemuck::Pod>(&self, value: &T, label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Bind buffers `0..n` (storage) + a trailing uniform, dispatch `groups`
    /// workgroups of the cached `label`/`wgsl` pipeline. `storages` are bound at
    /// 0..k, the uniform at index k.
    pub(crate) fn dispatch(
        &self,
        label: &str,
        wgsl: &str,
        storages: &[&wgpu::Buffer],
        uniform: &wgpu::Buffer,
        groups: u32,
    ) {
        let pipeline = self.pipeline(label, wgsl);
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry> = storages
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: storages.len() as u32,
            resource: uniform.as_entire_binding(),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &entries,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    // ── Kernels ────────────────────────────────────────────────────────────────

    /// RMSNorm over each row: `out[r] = x[r] / rms(x[r]) * weight`, with f32
    /// accumulation. `x` is `[rows, dim]`, `weight` is `[dim]`.
    pub fn rms_norm(
        &self,
        x: &GpuBuffer,
        weight: &GpuBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> GpuBuffer {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            dim: u32,
            rows: u32,
            eps: f32,
            _pad: u32,
        }
        let out = self.alloc_f32(rows * dim, "rms_norm.out");
        let params = self.uniform(
            &Params {
                dim: dim as u32,
                rows: rows as u32,
                eps,
                _pad: 0,
            },
            "rms_norm.params",
        );
        self.dispatch(
            "rms_norm",
            include_str!("shaders/rms_norm.wgsl"),
            &[&x.buf, &weight.buf, &out.buf],
            &params,
            rows as u32,
        );
        out
    }
}
