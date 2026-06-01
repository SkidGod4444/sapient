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

    /// Upload a u32 slice (e.g. token ids) into a GPU storage buffer.
    pub fn upload_u32(&self, data: &[u32], label: &str) -> GpuBuffer {
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

    /// GPU→GPU copy of `len` f32 elements from `src[src_off..]` into `dst[dst_off..]`.
    /// No shader / no readback — a pure encoder copy. This is the KV-cache append
    /// primitive: write a freshly-computed K/V head slice into its slot in a
    /// pre-allocated `[n_kv_heads, max_seq, head_dim]` cache without leaving the GPU.
    pub fn copy_range(
        &self,
        dst: &GpuBuffer,
        dst_off: usize,
        src: &GpuBuffer,
        src_off: usize,
        len: usize,
    ) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_range"),
            });
        enc.copy_buffer_to_buffer(
            &src.buf,
            (src_off * 4) as u64,
            &dst.buf,
            (dst_off * 4) as u64,
            (len * 4) as u64,
        );
        self.queue.submit(Some(enc.finish()));
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
    ///
    /// `groups` is the logical 1-D workgroup count; it's tiled into 2-D
    /// `(gx, gy)` so counts above the 65535 per-dimension limit (e.g. lm_head with
    /// vocab≈152k rows) still dispatch. Kernels recover the linear index as
    /// `wg.x + wg.y * num_workgroups.x` and bounds-check against the true count.
    pub(crate) fn dispatch(
        &self,
        label: &str,
        wgsl: &str,
        storages: &[&wgpu::Buffer],
        uniform: &wgpu::Buffer,
        groups: u32,
    ) {
        let gx = groups.clamp(1, 65535);
        let gy = groups.div_ceil(gx);
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
            pass.dispatch_workgroups(gx, gy, 1);
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

    /// Resident linear projection `out[m,n] = x[m,k] @ w[n,k]^T` (w in HF `[n,k]`),
    /// GEMV-style cooperative reduction, f32 accumulation.
    pub fn matmul_nt(
        &self,
        x: &GpuBuffer,
        w: &GpuBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> GpuBuffer {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            m: u32,
            k: u32,
            n: u32,
            _pad: u32,
        }
        let out = self.alloc_f32(m * n, "matmul.out");
        let params = self.uniform(
            &P {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                _pad: 0,
            },
            "matmul.params",
        );
        self.dispatch(
            "matmul_nt",
            include_str!("shaders/matmul_nt.wgsl"),
            &[&x.buf, &w.buf, &out.buf],
            &params,
            (m * n) as u32,
        );
        out
    }

    /// Element-wise op over `n` elements. `op`: 0 = `a+b` (residual), 1 = SwiGLU
    /// `silu(a)*b`. Returns a new buffer of length `n`.
    fn ewise(&self, a: &GpuBuffer, b: &GpuBuffer, op: u32) -> GpuBuffer {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            op: u32,
            _b: u32,
            _c: u32,
        }
        let n = a.len;
        let out = self.alloc_f32(n, "ewise.out");
        let params = self.uniform(
            &P {
                n: n as u32,
                op,
                _b: 0,
                _c: 0,
            },
            "ewise.params",
        );
        self.dispatch(
            "elementwise",
            include_str!("shaders/elementwise.wgsl"),
            &[&a.buf, &b.buf, &out.buf],
            &params,
            (n as u32).div_ceil(256),
        );
        out
    }

    /// Residual add: `out = a + b`.
    pub fn add(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        self.ewise(a, b, 0)
    }

    /// SwiGLU: `out = silu(gate) * up` (element-wise).
    pub fn swiglu(&self, gate: &GpuBuffer, up: &GpuBuffer) -> GpuBuffer {
        self.ewise(gate, up, 1)
    }

    /// Apply RoPE in-place to `x` laid out as `[batch*n_heads*seq_len, head_dim]`
    /// (rows ordered so `row % seq_len` is the sequence-position index — i.e. the
    /// `[batch, n_heads, seq_len, head_dim]` tensor flattened). NEOX rotate_half
    /// over the first `rotary_dim` channels (pass `head_dim` for full rotation);
    /// matches the CPU `apply_rope_partial`. `positions[s]` is the absolute position
    /// of sequence slot `s`. Mutates `x.buf` directly — no new allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &GpuBuffer,
        positions: &[u32],
        n_heads: usize,
        seq_len: usize,
        head_dim: usize,
        rotary_dim: usize,
        base: f32,
    ) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            rows: u32,
            head_dim: u32,
            rotary_dim: u32,
            half: u32,
            seq_len: u32,
            base: f32,
            _p0: u32,
            _p1: u32,
        }
        let rows = x.len / head_dim; // = batch*n_heads*seq_len
        debug_assert_eq!(
            rows % seq_len,
            0,
            "rope: rows must be a multiple of seq_len"
        );
        debug_assert_eq!(rows / seq_len % n_heads, 0, "rope: row layout mismatch");
        let half = rotary_dim / 2;
        let pos_buf = self.upload_u32(positions, "rope.positions");
        let params = self.uniform(
            &P {
                rows: rows as u32,
                head_dim: head_dim as u32,
                rotary_dim: rotary_dim as u32,
                half: half as u32,
                seq_len: seq_len as u32,
                base,
                _p0: 0,
                _p1: 0,
            },
            "rope.params",
        );
        self.dispatch(
            "rope",
            include_str!("shaders/rope.wgsl"),
            &[&x.buf, &pos_buf.buf],
            &params,
            ((rows * half) as u32).div_ceil(256),
        );
    }

    /// Causal grouped-query flash attention. `q` is `[batch, n_heads, seq_q, head_dim]`,
    /// `k`/`v` are `[batch, n_kv_heads, kv_stride, head_dim]` of which the first `seq_k`
    /// positions are valid (cached prefix = `seq_k - seq_q`); `kv_stride` is the allocated
    /// capacity per kv-head (pass `seq_k` for a tightly-packed buffer, or the cache's
    /// `max_seq` for a pre-allocated KV cache). Returns `[batch, n_heads, seq_q, head_dim]`.
    /// Online softmax, f32 accumulation; matches the CPU `scaled_dot_product_attention`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        batch: usize,
        n_heads: usize,
        n_kv_heads: usize,
        seq_q: usize,
        seq_k: usize,
        kv_stride: usize,
        head_dim: usize,
        scale: f32,
    ) -> GpuBuffer {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            batch: u32,
            n_heads: u32,
            n_kv_heads: u32,
            seq_q: u32,
            seq_k: u32,
            head_dim: u32,
            kv_offset: u32,
            jcount: u32,
            kv_stride: u32,
            scale: f32,
            _p0: u32,
            _p1: u32,
        }
        let out = self.alloc_f32(batch * n_heads * seq_q * head_dim, "attn.out");
        let params = self.uniform(
            &P {
                batch: batch as u32,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                seq_q: seq_q as u32,
                seq_k: seq_k as u32,
                head_dim: head_dim as u32,
                kv_offset: (seq_k - seq_q) as u32,
                jcount: (head_dim as u32).div_ceil(128),
                kv_stride: kv_stride as u32,
                scale,
                _p0: 0,
                _p1: 0,
            },
            "attn.params",
        );
        self.dispatch(
            "attention",
            include_str!("shaders/attention.wgsl"),
            &[&q.buf, &k.buf, &v.buf, &out.buf],
            &params,
            (batch * n_heads * seq_q) as u32,
        );
        out
    }

    /// Embedding gather: `out[t,:] = table[ids[t], :]`. Returns `[n_tokens, dim]`.
    pub fn embed(
        &self,
        ids: &GpuBuffer,
        table: &GpuBuffer,
        n_tokens: usize,
        dim: usize,
    ) -> GpuBuffer {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n_tokens: u32,
            dim: u32,
            _b: u32,
            _c: u32,
        }
        let out = self.alloc_f32(n_tokens * dim, "embed.out");
        let params = self.uniform(
            &P {
                n_tokens: n_tokens as u32,
                dim: dim as u32,
                _b: 0,
                _c: 0,
            },
            "embed.params",
        );
        self.dispatch(
            "embed",
            include_str!("shaders/embed.wgsl"),
            &[&ids.buf, &table.buf, &out.buf],
            &params,
            n_tokens as u32,
        );
        out
    }
}
