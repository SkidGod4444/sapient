# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential commands

```bash
# Build
cargo build --workspace          # debug (fast iteration)
cargo build --release -p sapient-cli   # release binary

# Test
cargo test --workspace           # all suites
cargo test -p sapient-backends-cpu --lib quant   # single crate + filter

# Lint (must pass before every push — the pre-push hook enforces this)
cargo fmt --all                  # auto-format (run before committing)
cargo clippy --workspace --all-targets -- -D warnings   # zero-warnings gate

# Run the local binary
./target/release/sapient chat openhorizon/phi-2
./target/release/sapient models

# Metal/GPU build (Apple Silicon only)
cargo build --release -p sapient-cli --features mlx
# Then colocate the shader library so MLX can find it at runtime:
cp $(find target/release -name 'mlx.metallib' | head -1) target/release/

# Publish to crates.io (rate-limit-aware, resumes if interrupted)
bash scripts/publish-all.sh

# Opt into native-CPU LLVM auto-vectorisation locally (do NOT commit this)
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

**Pre-push hook:** `.githooks/pre-push` runs `cargo fmt --all` then `cargo clippy -D warnings` automatically. Enable once per clone with `git config core.hooksPath .githooks`. Bypass with `SKIP_LINT=1 git push`.

## Architecture overview

SAPIENT is a pure-Rust edge LLM inference engine. The crates form a clear dependency stack:

```
sapient-cli              ← the `sapient` binary (chat, pull, run, models, update …)
  sapient-generate       ← Pipeline API (from_pretrained / from_gguf, generate, chat, stream)
    sapient-models       ← forward engines + weight loading
    sapient-hub          ← HF Hub client, curated registry, ModelInfo
    sapient-tokenizers   ← tokenizer + chat templates
    sapient-io           ← file format loaders (safetensors mmap, GGUF quant, ONNX)
    sapient-backends-cpu ← CPU kernels (matmul, attention, RoPE, quant dot-products)
    sapient-backends-metal ← MLX/Metal GPU backend (optional, `--features mlx`)
    sapient-backends-wgpu ← cross-platform GPU backend via wgpu (Vulkan/DX12/Metal) — Intel/AMD/Nvidia/Apple
  sapient-core           ← Tensor, DType, Shape, Buffer — used by everyone
```

`sapient-backends-wgpu` is the portable GPU path for non-Apple hardware (Intel/AMD/Nvidia on Linux/Windows). It is currently a **foundation**: `WgpuContext` device acquisition + validated `matmul_nt_f32` / `matmul_nt_q8_0` WGSL kernels. It is NOT yet wired into `ForwardEngine` — that's the integration phase (`WgpuForwardEngine`, see `docs/ROADMAP.md` Phase 3b). Develop/test the WGSL on any GPU including Apple Silicon (wgpu uses Metal there); the same shaders run on Intel/AMD via Vulkan.

`sapient-ir`, `sapient-runtime`, `sapient-scheduler`, `sapient-telemetry` power a separate *graph-execution* path used internally. The chat/generate path does **not** go through the IR; it uses the native forward engines directly. `sapient serve` (OpenAI-compatible HTTP server) is implemented in `sapient-cli/src/server.rs` and drives the `Pipeline` API directly — it does not use the IR runtime.

## Key design decisions to know

### Three forward engines wired to `Pipeline`
- **`LlamaForward`** — Llama, Mistral, Qwen2.5 (adds q/k/v biases), SmolLM2, TinyLlama (CPU + hybrid Metal)
- **`PhiForward`** — Phi-1/1.5/2 (LayerNorm+bias, parallel block, `partial_rotary_factor`) and Phi-3 (SwiGLU sequential path)
- **`MlxForwardEngine`** (`forward/mlx_engine.rs`, `cfg(macos + feature="mlx")`) — native lazy-graph Metal path for Llama/Qwen GGUF models. Auto-selected in `ForwardEngine::from_gguf_weights` when `use_mlx_engine(backend)` is true (Metal/Auto on Apple Silicon). All activations stay as `mlx_rs::Array`; one `mlx_rs::transforms::eval()` per decode step materialises logits + the whole KV cache. v0.3.5: ~187 tok/s decode + 21 ms TTFT on Qwen2.5-0.5B Q4 (9.4× the CPU path; lowest TTFT of any engine measured).

Architecture builder files in `sapient-models/src/architectures/` (gemma, gpt2, bert, mixtral …) build IR graphs for the graph-execution path; they are **not** used for live inference.

### MlxForwardEngine — critical invariants
- **RoPE axis:** `mlx_rs::fast::rope` uses dimension **−2** as the sequence-position axis. q/k/v MUST be transposed to `[1, n_heads, seq, head_dim]` (seq at −2) **before** RoPE — exactly like mlx-lm. Applying RoPE to `[1, seq, n_heads, head_dim]` scrambles positions across heads and produces garbage (every model collapses to one repeated token). This was the v0.3.4 fix.
- **KV cache** is stored in `[1, n_kv_heads, seq, head_dim]` layout and grown via `concatenate_axis` on axis 2.
- **GQA:** MLX's fused `scaled_dot_product_attention` handles grouped-query attention correctly (K/V not pre-tiled) — *once RoPE is on the right axis*. The earlier "fused SDPA mishandles GQA" symptom was the RoPE bug; with that fixed, the fused Metal kernel is both correct and faster than a manual matmul loop, so the engine uses it for all attention.
- **Engine reuse:** `Pipeline.engine` is `Arc<Mutex<ForwardEngine>>`. The streaming generators (`generate_stream`, `generate_stream_with_config`) clone the Arc and reuse the loaded engine — they must NOT rebuild it via `ForwardEngine::from_weight_paths_with_backend`, which re-quantizes the whole model and dominated TTFT (3 s on 1.5B). The non-streaming `generate_from_tokens_with_config` uses `tokio::task::block_in_place` on `self.engine.lock()`.

### Quantized storage (Phase 1)
`DType::Q4_0` and `DType::Q8_0` store raw ggml block bytes — no F32 expansion at load time. Key invariant: `as_bytes()` on non-quantized tensors returns the full buffer from `offset`; on quantized tensors it returns exactly `byte_count(numel)` bytes. Use `as_quant_blocks()` to access raw blocks, and `to_f32_vec()` to dequantize. `matmul_nt` dispatches on weight dtype: float weights use SGEMM, quant weights use per-block dot products.

### GGUF loading (Phase 4: mmap)
Three loading paths in `sapient-io/src/gguf.rs`: `parse_metadata_only` (header KV only, zero tensor alloc), `load_tensors_with_metadata` (heap CpuBuffer), `load_tensors_mmap` (OS-managed paging via `MmapBuffer` — Q4_0/Q8_0 zero-copy, K-quants dequantized from mmap bytes). GGUF dims are ggml column-major `[in, out]`; `map_gguf_tensors_to_hf` flips to HF `[out, in]`. The pipeline auto-detects mmap when file > 80% of available RAM (`available_ram_bytes()` reads `/proc/meminfo` on Linux, `sysctl` on macOS). `--mmap` flag forces it; `Pipeline::is_mmap()` reports which path was taken.

### Model registry (curated, not open)
`sapient-hub/src/registry.rs` contains a hardcoded `CATALOG` of `SupportedModel` entries. Every model resolves through `resolve_model_alias` — unrecognised names error with the catalog list. Fuzzy matching (prefix + Levenshtein) handles near-miss typos. Add a model by: (1) verify its arch is supported in a forward engine, (2) add a `SupportedModel` row with `openhorizon/*` branding.

### GGUF-only repos
When `from_pretrained` downloads a GGUF-only repo (no `config.json`), the hub client sets `config_path` to the GGUF file itself. The pipeline detects this via extension and routes to `from_gguf_opts`, bypassing `ModelInfo::from_config_file`.

### SIMD hot paths
`kernels/quant.rs` has four dispatch layers:
1. `dot_q4_0_block_f32` / `dot_q8_0_block_f32` — dispatch to NEON (`vfmaq_f32`) on `aarch64`, scalar otherwise.
2. `dot_q8_0_row_f32` — additionally dispatches to AVX2+FMA on `x86_64` with runtime `is_x86_feature_detected!`.
3. `dot_q4_k_block_f32` — NEON nibble-unpacking (`vshrq_n_u8` + `vandq_u8`) + `vfmaq_f32` FMA for Q4_K blocks.
4. Native F16 GEMV: F16 weights decoded in NEON registers via `vcvt_f32_f16`, no intermediate F32 allocation.

**Adaptive rayon chunking:** `gemv_chunk()` targets 4 tasks per core (not one task per output row). For a 151 936-row `lm_head` this avoids spawning 151 936 micro-tasks. `matmul_nt_q*` and `scaled_dot_product_attention` are parallelised with `rayon::par_iter_mut` / `par_chunks_mut` over the output dimension; `LlamaForward::forward_layer` uses `rayon::join` for parallel Q/K/V and gate/up projections.

**SDOT Q8_0 (v0.3.x):** `dot_q8_0_block_sdot` uses `core::arch::asm!` inline assembly to emit the ARMv8.4A `sdot` instruction. Marked `#[target_feature(enable = "neon,dotprod")]` — the `dotprod` feature is required by the assembler even though the call site already gates on `is_aarch64_feature_detected!("dotprod")`. Net gain is ~3% because Q8_0 GEMV is memory-bandwidth-bound (not compute-bound) on M-series UMA.

**Do not** add `-C target-cpu=native` to `.cargo/config.toml` for the `aarch64-apple-darwin` target — it causes `ring`'s compile-time const assertions to fail on CI runners.

### Q8_0 KV cache (in-place, zero allocation)
The KV cache is allocated as `DType::Q8_0` blocks (4× RAM reduction vs F32 for long contexts). Each decode step writes new K/V slices directly into the cache via `Tensor::as_bytes_mut()` — no per-step heap allocation. `as_bytes_mut()` is only valid on non-mmap tensors; the cache is always heap-allocated, so this invariant holds.

### Flash-Edge attention
`kernels/attention.rs` implements an online-softmax tiled attention algorithm that never materialises the full seq_q × seq_k score matrix. Working memory is O(head_dim). Uses NEON `vfmaq_f32` on `aarch64`. This replaces the previous naive `scaled_dot_product_attention` for the live-chat path.

### Tensor API additions (v0.2.9)
- `Tensor::from_f32_vec(Vec<f32>, shape)` — wraps a `Vec<f32>` as a tensor with zero copy (takes ownership).
- `Tensor::as_bytes_mut()` — mutable byte slice into the underlying buffer for in-place quantized writes.
- `CpuBuffer::from_f32_vec(Vec<f32>)` — low-level counterpart; wraps without copying.

### Online quantization at load time
F16/BF16 safetensors weights are auto-quantized to Q8_0 during loading (near-lossless, ~1.06 bytes/weight). This eliminates the F16→F32 expansion that previously dominated memory bandwidth for safetensors models.

### Speculative decoding
`SpeculativePipeline` wraps a draft and a target `Pipeline`. The draft model generates candidate tokens; `forward_all_logits` runs them through the target in a single batched forward pass for verification. Auto draft selection picks a smaller model from the registry automatically. Exposed via `sapient chat --speculative [--draft-model <alias>]` and `sapient serve --speculative`.

### OpenAI-compatible HTTP server (`sapient serve`)
`server.rs` in `sapient-cli` exposes a chat-completion server with the following endpoints:
- `GET /v1/models` — list loaded model(s)
- `POST /v1/chat/completions` — OpenAI-compatible chat
- `POST /v1/completions` — raw completion
- `GET /v1/health` — liveness check

No model is loaded at startup; the first API request triggers download + load (Ollama-style lazy loading).

### Hybrid Metal+CPU inference (v0.3.x)
Both `LlamaForward` and `PhiForward` support layer-split hybrid execution. `compute_backend_split()` / `compute_phi_backend_split()` run at model load and decide: full Metal, hybrid split, or CPU-only based on `(model_bytes × 1.5 ≤ RAM − 2 GB)`. The `forward_layer` is structured in **three borrow-safe phases** to satisfy the Rust borrow checker:
1. Pre-cache phase — borrows `&self.backend` (or `&self.cpu_fallback`) to compute norm, QKV, RoPE.
2. Cache phase — borrows `&mut self.cache[layer_idx]` (backend ref dropped).
3. Post-cache phase — re-borrows backend for attention + FFN.
Helper functions (`linear_with_bias_bk`, `mlp_phi2_bk`, `mlp_phi3_bk`) take explicit `bk: &LlmBackendDispatch` and `weights: &HashMap` so individual fields are borrowed rather than all of `self`.

### Metal SDPA head_dim compatibility
`mlx_sdpa_supported_head_dim(head_dim)` returns true only for {32, 64, 96, 128, 256}. MLX pre-compiles Metal SDPA shaders for this fixed set; any other value (e.g. Phi-2's 80) panics at runtime. `LlmBackendDispatch::from_kind_with_head_dim()` checks this at init: Auto silently falls back to CPU; explicit `--backend metal` returns a user-readable error.

### Stop-sequence handling
The streaming generator (`generate_stream`) buffers decoded text and withholds up to `max(stop_len)` bytes from the tail before emitting, preventing stop markers from leaking. Both EOS-by-token-id (multi-EOS, all candidates collected from the tokenizer vocab) and EOS-by-string are checked every step.

## Adding a new model architecture

1. Check if it's Llama-compatible (RMSNorm, SwiGLU, standard RoPE) — if so, just add a registry entry and it runs through `LlamaForward`.
2. If it needs a distinct forward pass: add `crates/sapient-models/src/forward/<arch>.rs`, add a variant to `ForwardEngine`, update `forward/mod.rs` dispatch, then add registry entries.
3. For GGUF: check that `map_gguf_tensor_name` in `gguf_weights.rs` covers the tensor naming; add `.bias` suffixes if the arch has projection biases.

## Version and release

- Version is set once in workspace `Cargo.toml` (`[workspace.package] version`); all crate `Cargo.toml` files inherit it via `.workspace = true`.
- Internal workspace deps also carry `version = "x.y.z"` for crates.io publishing.
- Release is triggered by pushing a `vX.Y.Z` tag; the workflow in `.github/workflows/release.yml` builds all platform binaries including a `-metal` variant for Apple Silicon.
- `scripts/publish-all.sh` publishes all 13 crates in dependency order, checking `static.crates.io` for the exact version before attempting each publish.


## Must follow

- always update the docs/PROJECT_GUIDE.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the CLAUDE.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the CONTRIBUTING.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the README.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the ROADMAP.md file when making changes to the codebase and keep it updated with the latest changes.