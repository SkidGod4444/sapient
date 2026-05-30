//! Linear-projection kernels (`x @ Wᵀ`) on the GPU.
//!
//! Two variants are provided:
//!   - [`WgpuContext::matmul_nt_f32`]  — dense F32 weights (plumbing + reference).
//!   - [`WgpuContext::matmul_nt_q8_0`] — Q8_0-quantized weights (the representative
//!     real-inference kernel: int8 weights + per-32-block f32 scale, dequantized in
//!     the shader).
//!
//! Both compute `out[M, N] = x[M, K] @ W[N, K]ᵀ` (PyTorch `nn.Linear` layout).

use wgpu::util::DeviceExt;

use crate::context::{WgpuContext, WgpuError};

const Q8_0_BLOCK: usize = 32;

impl WgpuContext {
    /// Dense F32 linear projection: `out[M,N] = x[M,K] @ w[N,K]ᵀ`.
    pub fn matmul_nt_f32(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, WgpuError> {
        if x.len() != m * k {
            return Err(WgpuError::Shape(format!(
                "x has {} elems, expected M*K = {}",
                x.len(),
                m * k
            )));
        }
        if w.len() != n * k {
            return Err(WgpuError::Shape(format!(
                "w has {} elems, expected N*K = {}",
                w.len(),
                n * k
            )));
        }

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_nt_f32"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_nt_f32.wgsl").into()),
            });

        let x_buf = self.storage_init("x", bytemuck::cast_slice(x));
        let w_buf = self.storage_init("w", bytemuck::cast_slice(w));
        let dims = [m as u32, k as u32, n as u32, 0u32];
        let out = self.run_matmul(&shader, &[&x_buf, &w_buf], &dims, m * n, m, n)?;
        Ok(out)
    }

    /// Q8_0 quantized linear projection: `out[M,N] = x[M,K] @ dequant(W)[N,K]ᵀ`.
    ///
    /// `scales` is `[N * (K/32)]` (one f32 per row-block) and `qweights` is the int8
    /// weights repacked into `[N * (K/32) * 8]` u32 words (4 int8 per word). Use
    /// [`quantize_q8_0_rows`] to produce these from an F32 weight matrix.
    pub fn matmul_nt_q8_0(
        &self,
        x: &[f32],
        qweights: &[u32],
        scales: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, WgpuError> {
        if k % Q8_0_BLOCK != 0 {
            return Err(WgpuError::Shape(format!(
                "K={k} must be a multiple of the Q8_0 block size ({Q8_0_BLOCK})"
            )));
        }
        let nblocks = k / Q8_0_BLOCK;
        if scales.len() != n * nblocks {
            return Err(WgpuError::Shape(format!(
                "scales has {} elems, expected N*nblocks = {}",
                scales.len(),
                n * nblocks
            )));
        }
        if qweights.len() != n * nblocks * 8 {
            return Err(WgpuError::Shape(format!(
                "qweights has {} u32, expected N*nblocks*8 = {}",
                qweights.len(),
                n * nblocks * 8
            )));
        }

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_nt_q8_0"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/matmul_nt_q8_0.wgsl").into(),
                ),
            });

        let x_buf = self.storage_init("x", bytemuck::cast_slice(x));
        let qw_buf = self.storage_init("qw", bytemuck::cast_slice(qweights));
        let sc_buf = self.storage_init("scales", bytemuck::cast_slice(scales));
        let dims = [m as u32, k as u32, n as u32, 0u32];
        let out = self.run_matmul(&shader, &[&x_buf, &qw_buf, &sc_buf], &dims, m * n, m, n)?;
        Ok(out)
    }

    // ── internals ─────────────────────────────────────────────────────────────

    fn storage_init(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Bind the input buffers (in order) at bindings 0.., append a read_write output
    /// buffer and a uniform dims buffer at the final two bindings, dispatch a 2D grid
    /// over (M, N), and read the result back to the host.
    fn run_matmul(
        &self,
        shader: &wgpu::ShaderModule,
        inputs: &[&wgpu::Buffer],
        dims: &[u32; 4],
        out_len: usize,
        m: usize,
        n: usize,
    ) -> Result<Vec<f32>, WgpuError> {
        let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::cast_slice(dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Layout: inputs (storage, read) | output (storage, read_write) | dims (uniform)
        let mut layout_entries = Vec::new();
        for i in 0..inputs.len() {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let out_binding = inputs.len() as u32;
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: out_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let dims_binding = out_binding + 1;
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: dims_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let bind_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul-bgl"),
                entries: &layout_entries,
            });

        let mut bind_entries: Vec<wgpu::BindGroupEntry> = inputs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        bind_entries.push(wgpu::BindGroupEntry {
            binding: out_binding,
            resource: out_buf.as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: dims_binding,
            resource: dims_buf.as_entire_binding(),
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul-bg"),
            layout: &bind_layout,
            entries: &bind_entries,
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul-pl"),
                bind_group_layouts: &[&bind_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

        // Dispatch a 2D grid over (M, N) with 8×8 workgroups (matches the shader).
        let wg_x = m.div_ceil(8) as u32;
        let wg_y = n.div_ceil(8) as u32;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Stage the result for host readback.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map_async channel dropped")?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }
}

/// Quantize an F32 weight matrix `[N, K]` (row-major) into Q8_0 GPU buffers.
///
/// Returns `(qweights, scales)` ready for [`WgpuContext::matmul_nt_q8_0`]:
///   - `scales`   : `[N * (K/32)]` f32 — per-block absmax/127.
///   - `qweights` : `[N * (K/32) * 8]` u32 — int8 weights, 4 per word.
///
/// `K` must be a multiple of 32.
pub fn quantize_q8_0_rows(w: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % Q8_0_BLOCK, 0, "K must be a multiple of 32");
    assert_eq!(w.len(), n * k, "w must have N*K elements");
    let nblocks = k / Q8_0_BLOCK;
    let mut scales = Vec::with_capacity(n * nblocks);
    let mut qweights = Vec::with_capacity(n * nblocks * 8);

    for row in 0..n {
        for b in 0..nblocks {
            let base = row * k + b * Q8_0_BLOCK;
            let block = &w[base..base + Q8_0_BLOCK];
            let absmax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            scales.push(scale);
            // Pack 32 int8 into 8 u32 words, little-endian (byte 0 = lane 0).
            for w4 in 0..8 {
                let mut word = 0u32;
                for lane in 0..4 {
                    let q = (block[w4 * 4 + lane] * inv).round().clamp(-127.0, 127.0) as i32;
                    word |= ((q as u32) & 0xFF) << (lane * 8);
                }
                qweights.push(word);
            }
        }
    }
    (qweights, scales)
}
