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
  sapient-core           ← Tensor, DType, Shape, Buffer — used by everyone
```

`sapient-ir`, `sapient-runtime`, `sapient-scheduler`, `sapient-telemetry` power a separate *graph-execution* path used by `sapient serve` (raw-tensor API server). The chat/generate path does **not** go through the IR; it uses the native forward engines directly.

## Key design decisions to know

### Two forward engines, not one per architecture
Only two engines are wired to `Pipeline`:
- **`LlamaForward`** — Llama, Mistral, Qwen2.5 (adds q/k/v biases), SmolLM2, TinyLlama
- **`PhiForward`** — Phi-1/1.5/2 (LayerNorm+bias, parallel block, `partial_rotary_factor`) and Phi-3 (SwiGLU sequential path)

Architecture builder files in `sapient-models/src/architectures/` (gemma, gpt2, bert, mixtral …) build IR graphs for the graph-execution path; they are **not** used for live inference.

### Quantized storage (Phase 1)
`DType::Q4_0` and `DType::Q8_0` store raw ggml block bytes — no F32 expansion at load time. Key invariant: `as_bytes()` on non-quantized tensors returns the full buffer from `offset`; on quantized tensors it returns exactly `byte_count(numel)` bytes. Use `as_quant_blocks()` to access raw blocks, and `to_f32_vec()` to dequantize. `matmul_nt` dispatches on weight dtype: float weights use SGEMM, quant weights use per-block dot products.

### GGUF loading (Phase 4: mmap)
Three loading paths in `sapient-io/src/gguf.rs`: `parse_metadata_only` (header KV only, zero tensor alloc), `load_tensors_with_metadata` (heap CpuBuffer), `load_tensors_mmap` (OS-managed paging via `MmapBuffer` — Q4_0/Q8_0 zero-copy, K-quants dequantized from mmap bytes). GGUF dims are ggml column-major `[in, out]`; `map_gguf_tensors_to_hf` flips to HF `[out, in]`. The pipeline auto-detects mmap when file > 80% of available RAM (`available_ram_bytes()` reads `/proc/meminfo` on Linux, `sysctl` on macOS). `--mmap` flag forces it; `Pipeline::is_mmap()` reports which path was taken.

### Model registry (curated, not open)
`sapient-hub/src/registry.rs` contains a hardcoded `CATALOG` of `SupportedModel` entries. Every model resolves through `resolve_model_alias` — unrecognised names error with the catalog list. Fuzzy matching (prefix + Levenshtein) handles near-miss typos. Add a model by: (1) verify its arch is supported in a forward engine, (2) add a `SupportedModel` row with `openhorizon/*` branding.

### GGUF-only repos
When `from_pretrained` downloads a GGUF-only repo (no `config.json`), the hub client sets `config_path` to the GGUF file itself. The pipeline detects this via extension and routes to `from_gguf_opts`, bypassing `ModelInfo::from_config_file`.

### SIMD hot paths
`kernels/quant.rs` has three dispatch layers:
1. `dot_q4_0_block_f32` / `dot_q8_0_block_f32` — dispatch to NEON on `aarch64`, scalar otherwise
2. `dot_q8_0_row_f32` — additionally dispatches to AVX2+FMA on `x86_64` with runtime `is_x86_feature_detected!`
3. `matmul_nt_q*` and `scaled_dot_product_attention` — parallelised with `rayon::par_iter_mut` / `par_chunks_mut` over the output dimension

**Do not** add `-C target-cpu=native` to `.cargo/config.toml` for the `aarch64-apple-darwin` target — it causes `ring`'s compile-time const assertions to fail on CI runners.

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