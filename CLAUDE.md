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

# NOTE: SAPIENT is NOT published to crates.io. Releases are GitHub binaries only
# (push a vX.Y.Z tag → .github/workflows/release.yml builds per-platform binaries).
# The previously-published crates have been yanked; `scripts/yank-all.sh` manages that.

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

### GGUF q/k RoPE permutation (critical for llama-arch GGUF)
llama.cpp's HF→GGUF converter **permutes the rows of `attn_q`/`attn_k`** for the `llama` architecture (Llama, Mistral, SmolLM, TinyLlama, …) because ggml uses NORM-style RoPE while HF/SAPIENT use NEOX-style (`rotate_half`). Loading those weights as-is makes RoPE scramble positions across each head → **incoherent token-salad output** (correct config, correct shapes, finite activations — just wrong). `gguf_weights::unpermute_llama_gguf_qk` (called from `ForwardEngine::from_gguf_weights` **only** for `ArchType::Llama`) inverts the permutation: HF row `h·D + a·(D/2) + b` ← GGUF row `h·D + b·2 + a`. It runs on any dtype (each output row is a contiguous `byte_count(in)` chunk). **Qwen2/Gemma GGUFs use NEOX RoPE — NOT permuted by the converter, so they must NOT be un-permuted** (the gate excludes them). Safetensors models are already HF layout. Regression test: `unpermute_qk_inverts_llama_cpp_permutation` in `gguf_weights.rs`.

### Tied-embedding GGUF models
GGUF metadata has no `tie_word_embeddings` flag, so models that tie the output projection to the input embedding (SmolLM2, Llama-3.2-1B/3B, small Qwen) simply omit `output.weight`. `resolve_lm_head` (in `weights.rs`) falls back to the embedding matrix when no explicit `lm_head`/`output` weight exists — otherwise those models fail to load with "missing lm_head.weight".

### KV-cache context window cap (OOM guard)
The KV cache is pre-allocated for `max_seq` positions up front. Modern models advertise huge contexts (Llama-3.1/DeepSeek-R1 = 131072) → ~9 GB of Q8_0 cache for an 8B model, OOM-killing a 16 GB device at **load** time. `common::kv_cache_ctx(model_max)` caps the allocation to `DEFAULT_KV_CACHE_CTX` (8192), overridable via the `SAPIENT_CTX` env var; longer conversations slide the window. Both `LlamaForward` and `PhiForward` cap `LayerCache.seq_len` to the allocated window so the sliding-window `update_kv_cache` never indexes past the cache.

### GGUF quant-file selection & tokenizer fallback
`sapient-hub/src/gguf.rs::gguf_preference_score` ranks GGUF files; **Q4_K_M is preferred over Q8_0** (edge sweet spot — ~40% smaller, near-lossless, uses the Q4_K matmul kernel; the old code picked Q8_0 and shipped an 8.5 GB file where a 4.9 GB one fits a 16 GB Pi). `tokenizer_fallback_model` resolves a HF tokenizer repo for GGUF-only models: it must point at **ungated** repos AND the **right version** — `meta-llama/*` returns 401 without a token, so Llama-3/DeepSeek fall back to `unsloth/Meta-Llama-3.1-8B-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. **Mistral: the GGUF catalog ships Mistral-7B-Instruct-v0.3 (vocab 32768), which REORDERED the vocab vs v0.1/v0.2 (32000) — the tokenizers are NOT interchangeable.** Loading the v0.1 tokenizer for a v0.3 GGUF mis-encodes the prompt and mis-decodes output into mixed-script token-salad (verified: v0.1 decodes v0.3 ids as `Г str — …らíses…レ`). So `mistral` defaults to `unsloth/mistral-7b-instruct-v0.3`; only an explicit `v0.1`/`v0.2` in the id falls back to `mistralai/Mistral-7B-v0.1`. Regression test: `mistral_tokenizer_fallback_defaults_to_v03`.

### Q6_K dequantization scale indexing (critical)
Q6_K stores **16 int8 scales per 256-weight super-block** (one per 16-element group) + one fp16 super-scale. Dequant must map weight `i` to scale `i/16`: within each 128-element half the four sub-groups (offsets 0/32/64/96) use scale offsets **+0/+2/+4/+6** with a further split at `l==16` (`is = l/16`), and the scale base advances by **8** per 128-block (matching ggml `dequantize_row_q6_K`). The original code used one scale per 32-group (`sc[ib..ib+4]`, `ib += 4`) — it only ever touched scales 0–7 and decoded Q6_K **wrong**, corrupting any Q6_K tensor. This is fixed in **three** places that must stay in sync: `sapient-core` `Tensor::to_f32_vec` (embedding lookup / `to_f32_cow`), `sapient-backends-cpu` `dot_q6_k_row_f32` (matmul), and `sapient-io` `dequantize_q6_k`. Regression test: `q6_k_scale_indexing_matches_ggml` in `kernels/quant.rs`. This bug was why Q4_K_M models that use Q6_K for the output/embedding (Llama-3.x, DeepSeek-R1-Distill-Llama, Mistral Q4_K_M) emitted token-salad — especially catastrophic for **tied-embedding** models (Llama-3.2-1B/3B) where the Q6_K embedding is the model input. qwen models were unaffected (their embeddings are Q4_K; Q6_K appears only in v_proj, where the error was tolerable).

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

**K-quant kernels are all NEON-vectorized (v0.3.9 — critical for RPi5/Cortex-A76 decode):** `dot_q4_k_row_q8_neon` (W4A8 SDOT — int8 activations via `quantize_row_to_i8_blocks` + `sdot`), `dot_q6_k_row_f32_neon`, and `dot_q5_k_row_f32_neon` (16-lane NEON). **Q6_K/Q5_K were previously pure scalar** while Q4_K was SIMD — and Q6_K is ~⅓ of a Q4_K_M model (lm_head + half of ffn_down + attn_v), so scalar Q6_K *dominated* decode and masked all Q4_K SIMD work. Vectorizing Q6_K gave +36–100% on the Pi (qwen-0.5B 4.5→6.1, 1.5B 1.3→1.9, mistral-7B 0.3→0.6 tok/s). Vectorizing Q5_K also surfaced+fixed a latent correctness bug: the scalar read the 5th bit as `qh[is/8]` (one bit per 32-element sub-block) instead of ggml's per-element `qh[l]` — would salad Q5_K_M models. Each NEON kernel is regression-tested bit-close to its scalar reference (`q6_k_neon_matches_scalar`, `q5_k_neon_matches_scalar`, `q4_k_w4a8_matches_f32_path`). **Lesson from the RPi5 perf hunt:** decode is memory-latency-bound, not compute-bound — SDOT, single-reduction, and multi-row/MLP kernels all gave ~0 once Q6_K was vectorized; the practical kernel ceiling is "no scalar K-quant kernels." Build the Pi binary on the Mac with `cargo zigbuild --release --target aarch64-unknown-linux-gnu` (rustls TLS means no openssl-sys cross-compile blocker); `RUSTFLAGS="-C target-cpu=cortex-a76"` is fine for the linux target (the ring/`target-cpu=native` caveat is only for `aarch64-apple-darwin`).

**SDOT Q8_0 (v0.3.x):** `dot_q8_0_block_sdot` uses `core::arch::asm!` inline assembly to emit the ARMv8.4A `sdot` instruction. Marked `#[target_feature(enable = "neon,dotprod")]` — the `dotprod` feature is required by the assembler even though the call site already gates on `is_aarch64_feature_detected!("dotprod")`. Net gain is ~3% because Q8_0 GEMV is memory-bandwidth-bound (not compute-bound) on M-series UMA. This is a W8A8 path (it quantizes the activation row to int8 too); activations are quantized **per 32-element block** (`quantize_row_to_i8_blocks`), matching the weight blocks. A single per-row activation scale (the old behavior) is set by outlier activation channels — common in LLMs — and collapses every normal-magnitude value to ~0, producing garbage; per-block scaling confines an outlier's damage to its own block (this is what llama.cpp does). Only Q8_0-*weight* models hit this path (online-quantized safetensors, or Q8_0 GGUFs); Q4_K GGUFs dequantize to F32 at load and never quantize activations. Tests: `sdot_q8_0_row_blockwise_survives_activation_outlier`, `matmul_nt_q8_0_matches_float`, `matmul_nt_q8_0_gguf_dimflip_matches_float`.

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
`SpeculativePipeline` wraps a draft and a target `Pipeline`. The draft proposes K tokens; the target verifies them in one cache-appending forward pass; accepted tokens are kept (rejection sampling corrects the distribution). Exposed via `sapient chat --speculative [--draft-model <alias>]` and `sapient serve --speculative [--draft-model <alias>]`.

**Engine reuse (serve-ready).** Like `Pipeline`, it reuses the loaded target+draft engines across requests: the streaming/non-streaming paths clone `self.target.engine_arc()` / `self.draft.engine_arc()` (`Arc<Mutex<ForwardEngine>>`) and **lock** them inside `spawn_blocking` — they do NOT rebuild engines via `from_weight_paths_with_backend` per request (that re-quantized the whole model and dominated TTFT). Has full `*_with_config` methods + `tokenizer()`/`arch()`/`is_mmap()`/`config()` accessors + `new_with_opts`/`with_auto_draft_with_opts` (backend/mmap). In `serve`, a resident model is `ServedModel::{Plain(Pipeline), Speculative(SpeculativePipeline)}` (in `server.rs`) and the LRU cache treats both uniformly.

**Cache-aware verification (critical invariant).** Target verification MUST use `forward_all_logits_cached` (NOT `forward_all_logits`). `forward_all_logits` calls `forward_hidden(.., use_cache=false)` which **resets the KV cache and starts at position 0** — verifying drafts with no prompt context, producing token-salad. This was a real bug (all speculative output was garbage). `forward_all_logits_cached` does `forward_hidden(.., true)`: it appends the draft tokens to the target KV (positions continue from the prompt) and returns per-position logits. The `spec_round` loop then rolls back rejected speculative tokens with `truncate_cache(n)`, maintaining the invariant "both target and draft KV caches hold exactly the committed tokens" and carrying each model's next-token logits across rounds. (MLX `truncate_cache` resets to 0 → no incremental rollback, so speculative uses the Llama/Phi CPU engines.)

**Vocab guard + family-aware auto-draft.** Draft and target must share a vocabulary (the draft proposes token IDs the target scores with its own logits). `new_with_opts` errors on a vocab mismatch instead of emitting garbage; `with_auto_draft` (`select_auto_draft`) picks a draft from the **same family** as the target (Qwen→`qwen2.5-0.5b`, SmolLM2→`smollm2-135m`).

### OpenAI-compatible HTTP server (`sapient serve`)
`server.rs` in `sapient-cli` exposes a chat-completion server with the following endpoints:
- `GET /v1/models` — list loaded model(s)
- `POST /v1/chat/completions` — OpenAI-compatible chat
- `POST /v1/completions` — raw completion
- `GET /v1/health` — liveness check

No model is loaded at startup; the first API request triggers download + load (Ollama-style lazy loading).

**Multi-model LRU cache (Phase 1, beats Ollama's single-resident model).** `ModelCache` in `server.rs` keeps the N most-recently-used models resident (`--max-models`, default 3) bounded by a RAM byte budget (`--cache-gb`, default ~70% of system RAM); switching back to a recent model is a cache hit (no download / re-quant / engine rebuild) instead of Ollama's cold reload. Design notes: each entry is `Arc<CachedModel>` so a streaming request keeps its model alive even if it's evicted mid-stream; `get_or_load` never holds the cache lock during the (slow) load or during inference, so cache hits and *other* models' requests run concurrently (only same-model inference serializes, on the engine's internal mutex); a `load_lock` serializes loads to prevent double-loading the same model on concurrent first-requests. Eviction is by both count (`max_models`) and bytes (LRU front evicted until both fit); size is estimated from on-disk weight size (`hub::cached_model_size`). `/v1/health` and `/v1/models` report `resident_models` + `active_model` (MRU). Measured: ~5× faster model switch-back on cache hit (cold reload of a large model is far higher — see deep-research notes).

**Admission control + prefix caching (Phases 3-4).** `inference_sem` (tokio `Semaphore`, `--max-concurrency`) bounds concurrent inferences. **Prefix/prompt KV caching**: `Pipeline::enable_prefix_cache()` (serve enables it) makes generation reuse the KV cache for the longest *token* prefix shared with the previous call (`ForwardEngine::truncate_cache(n)` + `Pipeline.last_prompt` + `common_prefix_len`), instead of `reset_cache()` + full re-prefill — multi-turn chat / shared system prompts skip re-prefilling history. Correct because only matching token IDs are reused (KV at `[0..P]` is identical → byte-identical greedy output) and same-model calls are serialized on the engine lock; off by default so CLI chat is unchanged. MLX `truncate_cache` falls back to reset (no reuse, still correct). **Deferred** (need engine work, see `docs/SERVING.md`): speculative-in-serve (SpeculativePipeline rebuilds engines per request — needs engine-reuse refactor first), continuous batching + PagedAttention (engine is single-sequence: `forward_logits(&[u32])`, KV `[1,n_kv,max_seq,hd]` — needs a batched-forward + block-pool rewrite).

### Hybrid Metal+CPU inference (v0.3.x)
Both `LlamaForward` and `PhiForward` support layer-split hybrid execution. `compute_backend_split()` / `compute_phi_backend_split()` run at model load and decide: full Metal, hybrid split, or CPU-only based on `(model_bytes × 1.5 ≤ RAM − 2 GB)`. The `forward_layer` is structured in **three borrow-safe phases** to satisfy the Rust borrow checker:
1. Pre-cache phase — borrows `&self.backend` (or `&self.cpu_fallback`) to compute norm, QKV, RoPE.
2. Cache phase — borrows `&mut self.cache[layer_idx]` (backend ref dropped).
3. Post-cache phase — re-borrows backend for attention + FFN.
Helper functions (`linear_with_bias_bk`, `mlp_phi2_bk`, `mlp_phi3_bk`) take explicit `bk: &LlmBackendDispatch` and `weights: &HashMap` so individual fields are borrowed rather than all of `self`.

### Metal SDPA head_dim compatibility
`mlx_sdpa_supported_head_dim(head_dim)` returns true only for {32, 64, 96, 128, 256}. MLX pre-compiles Metal SDPA shaders for this fixed set; any other value (e.g. Phi-2's 80) panics at runtime. `LlmBackendDispatch::from_kind_with_head_dim()` checks this at init: Auto silently falls back to CPU; explicit `--backend metal` returns a user-readable error.

### Chat input line editor (bracketed paste)
The interactive `sapient chat` REPL reads input via `rustyline::DefaultEditor` (`read_chat_line` in `sapient-cli/src/main.rs`), **not** `stdin().read_line()`. Plain `read_line` returns the instant it sees a newline, so any pasted text containing or ending with `\n` was submitted before the user pressed Enter (the v0.3.x paste bug). rustyline enables bracketed-paste mode by default, so a paste is inserted into the edit buffer as literal text (newlines included) and only a real Enter submits. `read_chat_line` returns `Ok(None)` on EOF / Ctrl-C / Ctrl-D (caller breaks) and feeds non-empty lines to history. The prompt is passed to `readline()` as `ui::user_prompt_str()` (styled string, ANSI handled by rustyline's cursor math). Both `chat_command` and `chat_speculative_command` share this path; piped/non-TTY input still works via rustyline's line-by-line fallback.

### Live Markdown rendering of replies (`markdown.rs`)
`sapient-cli/src/markdown.rs` (`StreamRenderer`) renders the assistant's streamed Markdown **as it streams** — prose/headings/lists/tables/inline styling via `termimad`, fenced code blocks via `syntect` 24-bit syntax highlighting (dim `│` gutter per line). Both chat loops route every token through `renderer.push(&token)` instead of `print!`.

**Commit-and-preview streaming.** Repainting the whole reply per token thrashes the screen and breaks once output scrolls past the viewport top (cursor can't move back up). Instead the renderer splits the reply into Markdown *blocks* (`complete_prefix_len`): **completed** blocks (separated by a blank line, or closed by a code fence) are rendered once and printed permanently; the **trailing incomplete** block is repainted in place each update via `\x1b[{n}A\r\x1b[0J` (move up over the preview, clear, reprint). `cursor_down_moves` computes the preview's row count *wrap-aware* (uses `console::measure_text_width`, which strips ANSI) so the cursor-up count is exact even when lines soft-wrap. A viewport guard commits the preview early if one in-progress block would exceed the screen height (rare, e.g. a huge unclosed code block). Repaints are throttled to ~30 fps unless a token contains `\n`.

**Raw / non-TTY fallback.** Rich rendering is disabled — and tokens stream as plain text — when stdout is not a terminal, when `NO_COLOR` is set, or with the `sapient chat --raw` flag. This keeps `sapient chat | tee log.txt` clean. Unit tests in `markdown.rs` cover block-boundary detection, wrap-aware row counting, and that `render()` emits ANSI styling + the code gutter.

### Stop-sequence handling
The streaming generator (`generate_stream`) buffers decoded text and withholds up to `max(stop_len)` bytes from the tail before emitting, preventing stop markers from leaking. Both EOS-by-token-id (multi-EOS, all candidates collected from the tokenizer vocab) and EOS-by-string are checked every step.

## Adding a new model architecture

1. Check if it's Llama-compatible (RMSNorm, SwiGLU, standard RoPE) — if so, just add a registry entry and it runs through `LlamaForward`.
2. If it needs a distinct forward pass: add `crates/sapient-models/src/forward/<arch>.rs`, add a variant to `ForwardEngine`, update `forward/mod.rs` dispatch, then add registry entries.
3. For GGUF: check that `map_gguf_tensor_name` in `gguf_weights.rs` covers the tensor naming; add `.bias` suffixes if the arch has projection biases.

## Version and release

- Version is set once in workspace `Cargo.toml` (`[workspace.package] version`); all crate `Cargo.toml` files inherit it via `.workspace = true`. Internal workspace deps still carry a matching `version = "x.y.z"` (kept in sync; required for path+version deps even though we don't publish).
- Release is triggered by pushing a `vX.Y.Z` tag; the workflow in `.github/workflows/release.yml` builds all platform binaries including a `-metal` variant for Apple Silicon.
- **SAPIENT is NOT published to crates.io.** Distribution is prebuilt GitHub release binaries (+ install script / Homebrew). The previously-published crates (0.1.11–0.3.1) have been yanked; `scripts/yank-all.sh` (idempotent, `--undo` to reverse) manages crates.io yanks. Do not re-introduce a publish step.


## Must follow

- always update the docs/PROJECT_GUIDE.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the CLAUDE.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the CONTRIBUTING.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the README.md file when making changes to the codebase and keep it updated with the latest changes.
- always update the ROADMAP.md file when making changes to the codebase and keep it updated with the latest changes.