# 🗺️ SAPIENT Roadmap — Huge Models on Small Devices

> **Mission:** run models that "shouldn't fit" on the hardware people actually own —
> laptops, Raspberry Pis, phones — with a one-line install and a great UX.
>
> The engine work below (quantization, mmap, SIMD, GPU offload) is the *price of entry*
> — llama.cpp already does it well. Our **moat** is the layer on top: pure-Rust
> portability, curated registry, modern CLI, and edge-specific automation
> (auto-pick quantization for available RAM, auto CPU/GPU offload, single static binary).

## Where we are (v0.3.5)
- ✅ **`MlxForwardEngine`** — native lazy-graph Metal forward pass for Llama/Qwen GGUF models. All activations stay on the GPU; one `eval()` per token; MLX fused SDPA. **~187 tok/s decode + 21 ms TTFT on Qwen2.5-0.5B Q4 (9.4× the CPU path); beats Ollama on 0.5B decode and has the lowest TTFT of any engine measured; within 1.3–1.5× of mlx-lm.** See [BENCHMARKS.md](BENCHMARKS.md).
- ✅ RoPE-axis correctness fix (transpose to `[1, n_heads, seq, head_dim]` before `fast::rope`).
- ✅ **Engine reuse** — pipeline holds the engine in `Arc<Mutex<…>>`; streaming no longer rebuilds/re-quantizes the model per call (**TTFT 30–44× faster**, 1.5B: 3 s → 70 ms).
- ✅ Correct CPU + Metal inference for Phi & Llama/Qwen families (F16/BF16 safetensors + GGUF Q4/Q8).
- ✅ Curated registry, modern CLI (`chat`, `pull`, `run`, `models`, `serve`, `reset`, `rm`, `update`, `devices`), self-update, published to crates.io.
- ✅ GGUF Q4_0/Q8_0/K-quant loading with mmap support (models larger than RAM).
- ✅ Flash-Edge attention (online-softmax, O(head_dim) memory, NEON).
- ✅ Q8_0 KV cache (in-place, 4× RAM reduction vs F32, zero per-step allocation).
- ✅ Online F16→Q8_0 quantization at load time (near-lossless, ~1.06 bytes/weight).
- ✅ Native F16 GEMV and NEON Q4_K GEMV; adaptive rayon chunking.
- ✅ SDOT Q8_0 kernel (ARMv8.4A `sdot` via inline asm, runtime-detected, ~3% net gain — bandwidth-bound).
- ✅ Speculative decoding (`sapient chat --speculative`).
- ✅ OpenAI-compatible HTTP server (`sapient serve`) with lazy model loading.
- ✅ Benchmark suite (`scripts/benchmark-compare.sh`, `scripts/gen-benchmark-report.py`).
- ✅ `sapient devices` — CPU/GPU detection, backend recommendations, hybrid Metal+CPU plan.
- ✅ Hybrid Metal+CPU layer-split inference for **both** LlamaForward and PhiForward.
- ✅ Phi-2 Metal crash fix — `mlx_sdpa_supported_head_dim()` gate prevents panic for unsupported head dims.
- ✅ Linux/Windows build fixes (cfg-gated `macos_gpu_name`, `dotprod` target_feature on SDOT functions).

## Guiding principles
1. **One PR/phase → one release.** Ship gradually; never a big-bang.
2. **Correctness is a gate.** Every phase adds/keeps a golden-output test (greedy decode of a known model → exact tokens). No release regresses output.
3. **Measure RAM and tok/s** every phase; numbers go in the release notes.
4. **CPU core first, accelerators second.** The quantized CPU engine is the shared foundation for *all four* targets.

---

## Phase 0 — Spike & de-risk  → `v0.1.x` ✅ DONE
Narrow proof before committing to the full build.
- ✅ Load one `Q4_0` GGUF, keep blocks quantized in memory (no F32 expansion).
- ✅ A single quantized `matmul_nt` (dequant-in-loop) for the linear layers only.
- ✅ Run a tiny model end-to-end; measure RAM (should ≈ file size) and tok/s.
- ✅ **Exit criteria met:** a Q4_0 linear path produces correct logits vs the F32 reference within tolerance.

## Phase 1 — Quantized CPU engine (foundation for every target)  → **`v0.2.0`** ✅ DONE
- ✅ `DType`: `Q4_0`, `Q8_0`, `Q4_K`, `Q5_0` storing raw quant blocks.
- ✅ Quantized `matmul_nt` / attention paths — never materialize F32 weights.
- ✅ GGUF loader; `from_gguf` wired into the Pipeline.
- ✅ mmap zero-copy: RAM ≈ file size.
- ✅ Auto-tokenizer fallback for GGUF repos.
- ✅ **Success metric met:** Q4_0/Q8_0 GGUF models run correctly in < 5 GB RAM.

## Phase 2 — CPU speed: SIMD + threading  → **`v0.2.x`** ✅ DONE (v0.2.9)
- ✅ SIMD quantized dot-products: **NEON** (Q4_0, Q8_0, Q4_K, native F16) + **AVX2** (x86).
- ✅ `rayon` threading; adaptive `gemv_chunk()` (4 tasks/core).
- ✅ `rayon::join` for parallel Q/K/V and gate/up projections.
- ✅ Flash-Edge attention (online-softmax, O(head_dim), NEON `vfmaq_f32`).
- ✅ Q8_0 KV cache (in-place, 4× RAM reduction, zero per-step allocation).
- ✅ Online F16→Q8_0 quantization at load time.
- ✅ Speculative decoding (`SpeculativePipeline`, auto draft selection).
- ✅ OpenAI-compatible `sapient serve` (lazy loading, `/v1/chat/completions`).
- ✅ **Success metric exceeded:** +89% (0.5B) and +138% (1.5B) tok/s vs v0.2.8 on M-series.

### Sprint 2b / Next CPU improvement (planned for v0.2.10)
SDOT integer arithmetic (ARMv8.4A — all M-series, Raspberry Pi 5):
- Replace i8→i16→i32→f32 widening (~10 NEON ops/8 weights) with `vdotq_s32` SDOT.
- Expected: ~4× compute improvement for Q8_0 dot products.
- Target: ~35–40 tok/s on 0.5B, ~18–20 tok/s on 1.5B.

## Phase 3 — Apple Silicon / Metal  → **`v0.3.0`–`v0.3.4`**
- ✅ Quantized matmul on MLX (`quantized_matmul`, group_size=64, 4-bit); unified memory.
- ✅ Native MLX attention + RoPE in `MlxForwardEngine` (no CPU fallback on the decode path).
- ✅ Auto CPU/GPU offload by model size & available memory (`use_mlx_engine` + hybrid split).
- ✅ **Decode throughput in the mlx-lm performance class** (187 tok/s @ 0.5B, beats Ollama).
- ✅ **Prefill / TTFT** — 21 ms @ 0.5B, 70 ms @ 1.5B (was 515 ms / 3 s). Root cause was the streaming path rebuilding the engine per call, not prefill compute (profiled at 64 ms). Fixed by reusing the loaded engine via `Arc<Mutex<…>>`.
- [ ] **Lower peak RAM** — store the token-embedding / `lm_head` table as MLX-Q4 and quantize weights without the transient F32 copy (currently ~1–1.5 GB vs mlx-lm's 0.3–1.0 GB).
- **Success metric:** a 7B–13B Q4 model interactive (> ~15 tok/s) on an M-series laptop.

## Phase 4 — Raspberry Pi / small ARM SBC  → **`v0.3.x`** (partially done)
The hardest, most differentiating CPU target (2–8 GB RAM).
- ✅ Bigger-than-RAM support via mmap paging.
- ✅ `aarch64` validation; NEON SIMD applies to Pi 4/5.
- [ ] Low-RAM tuning: minimal activation buffers, optional `Q4_K_S`.
- [ ] Document Pi 4/5 setup and expected tok/s.
- **Success metric:** run a 3B Q4 model on a 4 GB Pi 5 without OOM.

## Phase 4b — Multi-model server  → **`v0.3.x`**
`sapient serve` currently loads one model lazily. Extend to:
- [ ] Multiple simultaneous models in memory (switchable by `model` field in API request).
- [ ] LRU eviction when total model RAM exceeds a configurable budget.
- [ ] Streaming SSE for `POST /v1/chat/completions`.
- [ ] OpenAI-compatible `logprobs`, `n`, `stream` parameters.

## Phase 5 — Phones (iOS / Android)  → **`v0.4.0`**
Most constrained, biggest "wow".
- Library packaging: stable C FFI / UniFFI bindings; static lib for mobile.
- Mobile mmap + thermal/throttle-aware scheduling.
- Sample iOS (Swift) and Android (Kotlin/JNI) apps.
- **Success metric:** a 1–3B Q4 model running on-device in a demo app.

---

## Cross-cutting workstreams (continuous)
- **Correctness harness:** golden-token tests per architecture; CI gate.
- **Bench suite:** RAM + tok/s + time-to-first-token across targets; tracked over time.
- **UX automation:** `sapient` auto-selects a quantization that fits available RAM; `--mem` budget flag; clear "won't fit, try Q4" guidance.
- **Docs:** keep `PROJECT_GUIDE.md` and the README in sync each release.

## Definition of "leading the market"
Match llama.cpp on quantized edge inference (Phases 1–3), then win on:
**install in one line, run any curated model in one command, auto-fit the hardware, pure-Rust everywhere — including phones.**
