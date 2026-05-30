# sapient-backends-wgpu

Cross-platform GPU backend for SAPIENT, built on [`wgpu`](https://wgpu.rs). The same
WGSL compute shaders run on every modern GPU through a portable API:

| Platform | GPU vendors | API wgpu uses |
|----------|-------------|----------------|
| Linux / Windows | **Intel, AMD, Nvidia** | Vulkan (Linux/Windows) · DX12 (Windows) |
| macOS | Apple, AMD | Metal |

A kernel validated on one machine (e.g. an M-series Mac via Metal) runs unchanged on
an Intel Arc or AMD Radeon card via Vulkan — no per-vendor code.

## Status: foundation

This crate currently provides the **foundation** the rest of the GPU forward pass
will build on:

- **`WgpuContext`** — acquires a high-performance GPU adapter and logical device on
  any platform (headless compute, no window). `adapter_label()` reports e.g.
  `"AMD Radeon RX 7600 (Vulkan)"` or `"Apple M4 (Metal)"`.
- **`matmul_nt_f32`** — dense `x @ Wᵀ` (plumbing + reference).
- **`matmul_nt_q8_0`** — the representative real-inference kernel: int8 weights with a
  per-32-element f32 scale, dequantized in the shader.
- **`quantize_q8_0_rows`** — host repack of an F32 weight matrix into the GPU buffers.

All three kernels are validated against a host reference in `tests/matmul.rs`
(they run on whatever GPU is present, and skip cleanly if none is).

```rust
use sapient_backends_wgpu::WgpuContext;

let ctx = WgpuContext::new()?;       // picks the discrete GPU if present
println!("GPU: {}", ctx.adapter_label());
let out = ctx.matmul_nt_f32(&x, &w, m, k, n)?;   // x[M,K] @ w[N,K]ᵀ
```

## Performance note

The foundation kernels prioritise **correctness and portability**, not yet
throughput. Each call currently rebuilds the pipeline and does a blocking readback,
and the matmul is a naive one-thread-per-output kernel — so a single isolated GEMV
measures only a few GFLOP/s, dominated by per-call overhead.

Real inference speed comes in the integration phase, which mirrors what
`MlxForwardEngine` does on Metal:

1. **Cache pipelines & bind-group layouts** — build once, reuse every step.
2. **Keep activations GPU-resident** — no per-op upload/readback; only the final
   logits come back to the host.
3. **Tiled / shared-memory kernels** — workgroup tiling, vectorised loads,
   subgroup reductions for the GEMV-heavy decode path.
4. **One submission per token** — batch the whole layer stack into a single command
   buffer, like the lazy-graph eval on Metal.

## Roadmap to a full GPU forward pass

| Phase | Deliverable |
|-------|-------------|
| ✅ Foundation | `WgpuContext` + F32/Q8_0 matmul, validated on GPU |
| — | Remaining kernels: RMSNorm, RoPE, SwiGLU, softmax/SDPA attention, embedding gather |
| — | Q4_K / Q4_0 dequant kernels (match the CPU K-quant paths) |
| — | `WgpuForwardEngine` in `sapient-models` — persistent buffers, cached pipelines, GPU-resident KV cache |
| — | Wire into `ForwardEngine` + `sapient devices` (auto-select on non-Apple GPUs) |
| — | Tiled kernels + perf tuning to approach native Vulkan throughput |

See [`docs/ROADMAP.md`](../../../docs/ROADMAP.md) for how this fits the overall plan.
