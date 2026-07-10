# 🗺️ SAPIENT Roadmap — Huge Models on Small Devices

> **Mission:** run models that "shouldn't fit" on the hardware people actually own —
> laptops, Raspberry Pis, phones — with a one-line install and a great UX.
>
> The engine work below (quantization, mmap, SIMD, GPU offload) is the *price of entry*
> — llama.cpp already does it well. Our **moat** is the layer on top: pure-Rust
> portability, curated registry, modern CLI, and edge-specific automation
> (auto-pick quantization for available RAM, auto CPU/GPU offload, single static binary).

## Where we are (v0.5.3)
- ✅ **Sparse MoE (Mixtral-class first cut)** — the credible "big models on edge"
  path: a 47B-A13B (Mixtral-8x7B) decodes at ~13B bandwidth cost on 32 GB+ devices
  (big Mac / Jetson Thor). Implemented as a per-layer `Ffn::{Dense, Moe}` branch
  **inside `LlamaForward`** (shared attention/KV/RoPE), detected by config not
  `ArchType` (a Mixtral GGUF is arch `llama`). Router = softmax→top-k→renorm
  (Mixtral order, numerically gated); expert-grouped batched SwiGLU experts.
  Handles **both** GGUF expert formats (stacked `*_exps` 3-D blob + older
  per-expert 2-D, both verified against real files) and safetensors. CPU-only for
  now (MLX/wgpu bail clearly). Registry `openhorizon/mixtral-8x7b-q4`. Extension
  points parsed but bailed-on: sigmoid/shared-expert routing (DeepSeek/GLM). Gated
  by routing unit tests + 3 coherence tests + an ignored Mixtral greedy e2e.
  **Verified end-to-end on a Jetson AGX Thor** (47B, pure Rust, zero CUDA): decode
  5.5 tok/s, RSS 25.6 GB (MoE now mmaps by default → ≈ file size), **0 quality
  loss** vs llama.cpp (greedy token-identical ~28 tokens). SAPIENT loads the
  classic per-expert Mixtral GGUFs current llama.cpp rejects. See
  [BENCHMARKS.md](BENCHMARKS.md).
- ✅ **GLM-4.5-Air (`Glm4Moe`) — the DeepSeek-V3-style sigmoid-gate MoE**, built on
  the Mixtral foundation and **decode-verified on Thor** (106B-A12B, pure Rust,
  zero CUDA, coherent output, decode 2.45 tok/s). New: sigmoid gate + aux-loss-free
  correction bias + always-on shared expert, partial RoPE 0.5, head_dim from
  `key_length`, MTP-layer cap, and **split-GGUF loading** (Q4_K_M is a 2-shard
  ~63 GB set) with a **zero-copy stacked-expert split** (per-expert mmap views, no
  heap copy — 7× decode over the byte-copy). `ArchType::Glm4Moe` (NEOX → no q/k
  unpermute). Registry `openhorizon/glm-4.5-air-q4` (96 GB+ device). Four
  real-model bugs the Thor run caught that synthetic tests couldn't. v0.5.3 added a
  fifth fix: quant types SAPIENT can't keep as packed blocks (GLM's Q5_0
  `ffn_down_exps`) now re-quantize to Q8_0 at load instead of F32-expanding —
  peak RSS 118 → 72 GB, decode 2.45 → 3.23 tok/s, prefill 5×; GLM-4.5-Air fits a
  96 GB device. GLM-5.2 stays out of scope (MLA + DeepSeek Sparse Attention +
  group-limited routing).
- ⏳ **Server-ARM decode kernels (parity project, NOT MoE-specific)** — the Thor
  benchmark surfaced that SAPIENT is ~3.16× behind llama.cpp on *dense* Neoverse
  CPU decode (bigger than the 1.8× MoE gap → MoE is fine), decomposing to ~1.94×
  single-core kernel quality (llama.cpp = Arm **KleidiAI** microkernels) × ~1.6×
  multicore scaling (per-GEMV rayon fork/join). The roadmap's "1.1–1.35× behind
  llama.cpp" holds on M4/Pi (NEON) but **not on SVE-class server ARM** (Graviton/
  Grace/Ampere/Thor) — never measured before. **SVE is a dead end** on Thor
  (128-bit = NEON width). Closing it = KleidiAI-class NEON microkernels + a
  lower-overhead decode threadpool (fewer parallel regions per token). Deep,
  bounded work; benefits all server ARM, not just MoE.
- 🚧 **Gemma3 engine** — gemma-3-1b/4b + **MedGemma-4B** (medical chat + medical
  image analysis via the Gemma3 multimodal path). New `Gemma3Forward` (QK-norm,
  sandwich norms, sliding/global attention) + a flash-attention NaN fix any
  sliding-window model needed. GGUF loading + perf work pending.
- 🚧 **Vision-language (Phase 12 first cut)** — `sapient see <image> -p "…"`:
  SmolVLM-256M (SigLIP tower + pixel-shuffle connector on new `forward/siglip.rs`,
  embedding-splice into the existing Llama engine). Golden test (red fixture → "Red")
  + numeric grid-orientation probe. v1: single global 512² image (no sub-image
  splitting yet). MedGemma requires a Gemma3 text engine — next engine project.
  **Server (12.3) done:** `/v1/chat/completions` accepts OpenAI image parts as
  base64 data URIs, routed through `VlmPipeline` in a third LRU cache;
  remote image URLs are refused by design.
- 🚧 **Streaming voice loop (Phase 10 first cut)** — incremental STT during speech
  (`LiveStt`, transcript ready at end-of-utterance), early-first-clause TTS handoff,
  barge-in (`SpeakerPlayback::clear` + mic monitor), per-turn latency breakdown.
  Perceived latency ~4.4 → ~3.1 s (M4 CPU); floor is now Kokoro first-fragment RTF.
- ✅ **On-device audio (Phase 6)** — `sapient transcribe` (Whisper STT), `sapient speak`
  (Kokoro-82M real-time TTS + Orpheus-3B), and `sapient converse` (live mic→STT→LLM→reply, with
  `--speak` voicing the reply via Kokoro). All pure-Rust, cross-platform, in the default binary.
- ✅ **One-shot `sapient chat -p "<text>"`** — single templated turn, reply-only to stdout (scriptable).
- ✅ **`MlxForwardEngine`** — native lazy-graph Metal forward pass for Llama/Qwen GGUF models. All activations stay on the GPU; one `eval()` per token; MLX fused SDPA. **~187 tok/s decode + 21 ms TTFT on Qwen2.5-0.5B Q4 (9.4× the CPU path); beats Ollama on 0.5B decode and has the lowest TTFT of any engine measured; within 1.3–1.5× of mlx-lm.** See [BENCHMARKS.md](BENCHMARKS.md).
- ✅ RoPE-axis correctness fix (transpose to `[1, n_heads, seq, head_dim]` before `fast::rope`).
- ✅ **Engine reuse** — pipeline holds the engine in `Arc<Mutex<…>>`; streaming no longer rebuilds/re-quantizes the model per call (**TTFT 30–44× faster**, 1.5B: 3 s → 70 ms).
- ✅ Correct CPU + Metal inference for Phi & Llama/Qwen families (F16/BF16 safetensors + GGUF Q4/Q8).
- ✅ Curated registry, modern CLI (`chat`, `transcribe`, `speak`, `converse`, `pull`, `run`, `models`, `serve`, `reset`, `rm`, `update`, `devices`, `stats`), self-update. Distributed as prebuilt GitHub release binaries (not crates.io).
- ✅ GGUF Q4_0/Q8_0/K-quant loading with mmap support (models larger than RAM).
- ✅ Flash-Edge attention (online-softmax, O(head_dim) memory, NEON).
- ✅ Q8_0 KV cache (in-place, 4× RAM reduction vs F32, zero per-step allocation).
- ✅ Online F16→Q8_0 quantization at load time (near-lossless, ~1.06 bytes/weight).
- ✅ Native F16 GEMV and NEON Q4_K GEMV; adaptive rayon chunking.
- ✅ SDOT Q8_0 kernel (ARMv8.4A `sdot` via inline asm, runtime-detected, ~3% net gain — bandwidth-bound).
- ✅ Speculative decoding (`sapient chat --speculative`).
- ✅ OpenAI-compatible HTTP server (`sapient serve`) with lazy loading + **multi-model LRU cache** (top-N resident, byte-budgeted; instant switch-back vs Ollama's cold reload).
- ✅ Benchmark suite (`scripts/benchmark-compare.sh`, `scripts/gen-benchmark-report.py`).
- ✅ `sapient devices` — CPU/GPU detection, backend recommendations, hybrid Metal+CPU plan.
- ✅ Hybrid Metal+CPU layer-split inference for **both** LlamaForward and PhiForward.
- ✅ Phi-2 Metal crash fix — `mlx_sdpa_supported_head_dim()` gate prevents panic for unsupported head dims.
- ✅ Linux/Windows build fixes (cfg-gated `macos_gpu_name`, `dotprod` target_feature on SDOT functions).
- ✅ Chat UX: paste-safe `rustyline` line editor (bracketed paste — multi-line pastes no longer auto-submit) and **live Markdown rendering** of replies (`termimad` prose + `syntect`-highlighted code blocks; `--raw` / non-TTY falls back to plain text).
- ✅ GGUF correctness fixes for llama-family models, **verified end-to-end on CPU through Llama-3.2-1B / Llama-3.1-8B / DeepSeek-R1-Distill-Llama-8B (Q4_K_M)**:
  - **Q6_K dequant scale-indexing fix** (the big one): the old code used one scale per 32-group and only touched 8 of the 16 super-block scales, decoding every Q6_K tensor wrong → token-salad for any Q4_K_M model that stores its output/embedding as Q6_K (Llama-3.x, DeepSeek, Mistral). Catastrophic for tied-embedding models. Fixed in all three Q6_K decoders + regression test.
  - **q/k RoPE un-permute** for `llama`-arch GGUFs (ggml NORM-RoPE → HF/NEOX layout).
  - **tied-embedding fallback** (SmolLM2 / Llama-3.2 GGUFs load).
  - **Q8_0 W8A8 per-block activation quantization** (outlier-robust).
  - **KV-cache context cap** (`SAPIENT_CTX`, default 8192) so 128K-context 8B models no longer OOM-kill at load.
  - **Q4_K_M preferred over Q8_0** in GGUF file selection (smaller, fits 16 GB edge devices); **ungated tokenizer fallbacks** (`unsloth/*`, `deepseek-ai/*` instead of gated `meta-llama/*`).

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

## Phase 3b — Cross-platform GPU (Intel / AMD / Nvidia on Linux & Windows)  → **`v0.3.x`**
Bring GPU acceleration to the machines Metal can't reach, via a portable compute API
(`wgpu` → Vulkan / DX12 / Metal). The **same WGSL kernels** run on Intel Arc, AMD
Radeon, Nvidia, and Apple — and are dev-tested on Apple Silicon (Metal under wgpu).
- ✅ **Foundation** (`crates/sapient-backends/wgpu`): `WgpuContext` device acquisition
  (adapter-max limits past the 128 MiB binding cap, `SHADER_F16`, pipeline cache).
- ✅ **Resident kernels** (`resident.rs` + `shaders/*.wgsl`): GPU-resident `GpuBuffer`,
  RMSNorm, GEMV `matmul_nt`, RoPE (NEOX partial-rotary), SwiGLU, residual add, embedding
  gather, causal GQA **FlashDecoding attention** (online softmax, `kv_stride`), and a
  `copy_range` KV-cache append — each validated bit-close to a CPU reference.
- ✅ **`WgpuForwardEngine`** in `sapient-models` (`--features wgpu`): weights upload once,
  GPU-resident KV cache, decode runs fully on-device, only logits read
  back. Wired into `ForwardEngine::Wgpu` + `LlmBackendKind::Wgpu` (`--backend wgpu`) for
  Llama/Qwen/Mistral (GGUF + safetensors). **Coherence proven**: logits match the CPU
  `LlamaForward` on a synthetic model (prompt + incremental decode, argmax + max_err<5e-3).
- ✅ **In-shader Q8_0 dequant** (Phase 7.1, `quant.rs` + `matmul_nt_q8_0.wgsl` /
  `embed_q8_0.wgsl`): raw ggml Q8_0 blocks upload as packed int8 `u32` words + f32
  scales (`GpuQ8Buffer`) — **no f32 expansion**; matmul/embed dequantize in-shader.
  F16/BF16 linears online-quantize to Q8_0 (same rule as the CPU engine); tied output
  projections reuse the embed buffer. Measured (SmolLM2-360M Q8_0, Apple M4 via
  wgpu→Metal): weights resident 1.6 GiB→**388 MiB** (≈ GGUF file size), peak RSS
  2.65→1.27 GB, decode 20.5→21.4 tok/s, TTFT 51→46 ms; greedy output token-identical
  to the f32 path. Gated by `wgpu_q8_0_logits_match_cpu_llama` + per-kernel dequant
  reference tests.
- ✅ **In-shader Q4_K dequant** (Phase 7.2, `matmul_nt_q4_k.wgsl` / `embed_q4_k.wgsl`):
  raw 144-byte super-blocks upload **verbatim** (word-aligned — zero repack); the
  shader decodes d/dmin + the packed 6-bit scale/min pairs (`get_scale_min_k4`) +
  4-bit nibbles, 0.5625 bytes/weight. Q4_K_M GGUFs now load mostly quantized
  (Qwen2.5-1.5B: 169/198 matrices). Measured (Qwen2.5-1.5B Q4_K_M, M4 16 GB):
  weights resident 6778→**2367 MiB**, peak footprint 14.7→**5.4 GB** — the f32
  baseline exhausted the machine and emitted an immediate-EOS empty reply; the
  Q4_K build answers correctly, matching CPU greedy byte-for-byte. Decode 11.3 tok/s
  (≈ CPU), TTFT 81 vs 89 ms. Gated by `wgpu_q4_k_logits_match_cpu_llama` (vs a
  host-dequantized f32 twin, max_err<5e-3) + random-bit per-kernel reference tests.
- ✅ **In-shader Q6_K dequant** (`matmul_nt_q6_k.wgsl` / `embed_q6_k.wgsl`): 210-byte
  blocks padded to 212 on upload (pure memcpy — word alignment only); the shader
  decodes the 4+2-bit quants and 16 **signed** int8 scales with the +0/+2/+4/+6
  per-128-half indexing mirrored from the fixed CPU `dequantize_row_q6_K`
  (random-bit reference tests pin every path). Q4_K_M GGUFs now load **fully
  quantized** (Qwen2.5-1.5B: 198/198): weights resident 2367→**1062 MiB** (≈ GGUF
  file size; 6.4× vs f32), peak footprint 5.4→**3.6 GB**, decode 11.3→**13.2 tok/s —
  the wgpu path now beats the NEON M4 CPU (11.7) at 1.13×**. TTFT 77 ms.
- ✅ **f16 KV cache** (Phase 7.3, `kv_append{,_f16}.wgsl` + templated attention):
  K/V stored as f16 halves packed two-per-`u32` word, written by a `kv_append`
  conversion kernel and read via core-WGSL `unpack2x16float` — **no `SHADER_F16`
  feature needed** (naga in wgpu 22 can't parse `enable f16;`), so it runs on every
  adapter. f32 accumulation unchanged. Half the bytes lifts the wgpu context cap
  **4096 → 8192** (`kv_cache_ctx` / `SAPIENT_CTX`) at the same memory; auto-on for
  even head_dim (all real models). Decode unchanged within noise at short context.
  Gated by an f16-rounded-reference kernel test + `wgpu_f16_kv_cache_matches_f32_kv_cache`.
- ✅ **Per-token command batching** (Phase 7.4, `begin_batch`/`flush_batch`):
  every kernel used to pay its own queue submission (~450/token); each decode
  token now records into one shared encoder and submits once. Measured
  back-to-back on M4/Metal: SmolLM2-360M **23.1→29.3 tok/s (+27%)**, TTFT
  40.5→35 ms; Qwen2.5-1.5B 12.0→12.5 tok/s (+4%), TTFT 86→80 ms. **Must flush
  per token** — batching a whole prompt's passes into one encoder stalls Metal.
  Shader-level fusion (norm→GEMV, gate/up→SwiGLU) evaluated and deferred: post-
  batching it would cut ~3 of ~450 kernels while multiplying shaders across 4
  weight formats; revisit if 7.6 discrete-GPU data shows launch-bound decode.
- ✅ **Batched prefill** (Phase 7.5, `forward_chunk` + multi-token `kv_append`):
  prompts process in 128-token chunks — transposes to heads-major for RoPE /
  KV-append / attention (`seq_q = chunk`, the FlashDecoding kernel handles it
  causally via `kv_offset`), last position sliced before the final norm; decode
  keeps the transpose-free `seq_q = 1` fast path. Measured (Qwen2.5-1.5B, ~640-token
  prompt, cold incl. load): time-to-first-token **87.9 → 58.5 s (1.5×)**, identical
  greedy reply. Gated by `wgpu_chunked_prefill_matches_per_token` (300-token prompt,
  chunk boundaries + pos0>0). **Known limitation:** matmuls are still GEMV-shaped,
  so weights are read `m×` per chunk — the multi-row/tiled GEMM epilogue that makes
  prefill weight traffic ∝ 1/chunk is the highest-value follow-up below.
- ✅ **Nvidia datapoint (7.6, Jetson AGX Thor via Vulkan, 2026-07-03)**: whole
  quantized WGSL stack correct on Vulkan first try (198/198 quantized, greedy
  matches Metal/CPU). 1.5B: CPU 2.2 → wgpu-quantized **10 tok/s (4.5×)**; but the
  **f32 path hits 19.6 tok/s** (bandwidth roofline) — the dequant kernels are
  **ALU-bound on Nvidia** (Q8_0 ≈ 0.9× f32, Q4_K/Q6_K ~0.5×). The ≥2×-f32 bar is
  NOT met on bandwidth-rich Thor-class hardware; quantized-resident's value there
  is the 6.4× memory cut. See BENCHMARKS.md for the full table.
- ✅ **Multi-row dequant GEMM (MT=8)** for all prefill matmuls (f32/Q8_0/Q4_K/
  Q6_K `_mt` shader variants): weight blocks decoded once per 8 x-rows. Measured
  1101-token cold prefill: Thor **485→57 s (~8.5×** — the full amortization
  factor, confirming GEMV prefill was dequant-ALU-bound on Nvidia); M4 Metal
  59.8→37.9 s (1.58×). Decode (m=1) untouched and unchanged on both.
- ✅ **Vectorized dequant** (unpack4x8snorm/unorm + dot in all six quant matmul
  shaders, norm constants folded into block scales): M4 1.5B decode 12.8→14.3
  tok/s (+12%); Thor neutral — which pins the remaining Nvidia m=1 gap on the
  GEMV **workgroup shape** (one output per 256-lane workgroup ⇒ ~1 word/lane +
  8-round reduction; f32 hides it behind 4× traffic), not instruction cost.
- [ ] **P5 (remaining)**: decode-GEMV shape rework for bandwidth-rich GPUs
  (fewer lanes per output / multiple outputs per workgroup — the measured
  Nvidia m=1 gap), then scratch-buffer/bind-group reuse,
  discrete-adapter pick, `sapient devices` listing, Linux/Windows CI, bench on
  real **Arc/AMD** cards (the remaining 7.6 vendors — and the original "done
  when" targets). (Q5_K/Q4_0 in-shader dequant only if a shipped model needs
  them; quantized Q8 KV cache only if long-context memory becomes the
  constraint.)
- **Success metric:** a Q4 model on an Intel Arc / AMD Radeon card decoding several×
  faster than that machine's CPU path, from the same single binary.

## Phase 4 — Raspberry Pi / small ARM SBC  → **`v0.3.x` – `v0.4.x`** (mostly done)
The hardest, most differentiating CPU target (2–8 GB RAM). (Continues as the
Notion roadmap's Phase 8 — "Own the Raspberry Pi".)
- ✅ Bigger-than-RAM support via mmap paging.
- ✅ `aarch64` validation; NEON SIMD applies to Pi 4/5. All hot dot-product paths
  are NEON (Q8_0 SDOT, Q4_K W4A8 SDOT, Q5_K/Q6_K 16-lane) — the v0.3.9 Pi perf
  hunt established "no scalar K-quant kernels" as the practical kernel ceiling
  (decode is memory-latency-bound; further SDOT conversions measured ~0).
- ✅ Low-RAM quant selection: **`SAPIENT_GGUF_QUANT=Q4_K_S`** (or any quant tag)
  overrides the Q4_K_M default when a 4 GB board needs the smaller file.
- ✅ **Thermal-aware sustained decode** (`sapient-backends-cpu/src/thermal.rs`):
  a hysteresis governor samples `/sys/class/thermal` (rate-limited, from the
  matmul dispatcher) and steps the GEMV parallelism target down one core at a
  time from 80 °C (floor: half the cores), restoring below 70 °C — backs off
  *before* the 85 °C firmware trip so passive boards degrade gracefully instead
  of collapsing. `SAPIENT_THERMAL=off|_HOT|_COOL|_PATH` to tune; inert on
  machines without thermal zones. Unit-tested against a fake sysfs; on-device
  Pi validation pending.
- ✅ `docs/PI.md`: setup, per-RAM guidance, thermal + voice-loop docs, and the
  measured Pi 5 table (0.5B 8.7 / 1B 8.3 / 1.5B 6.7 / 3B 3.4 tok/s post-fix);
  voice loop measured end-to-end via `converse --input` — re-measured on the
  v0.5.2 release binary (0.5B: STT 2.96 s + LLM 3.5 s + TTS 5.4 s ≈ 11.9 s
  sequential; 1.5B ≈ 12.6 s; Kokoro RTF ~2.4 is the dominant stage; the 2.4 s
  in-loop TTFT is an open observation — bare-chat TTFT is 116 ms). Pi 4 column:
  no hardware on hand; numbers welcome.
- ✅ **Minimal activation buffers (8.3) — closed with two findings.** (1) Ordinary
  per-step activation allocations are measured-zero: forcing all large allocs onto
  the reusable heap via `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=64M` changed
  Pi decode by 0.0% (8.7 tok/s in all four A/B runs) — glibc already recycles the
  repeating buffers, so no scratch-pool machinery was added. (2) The audit found
  the real per-step buffer catastrophe elsewhere: **embedding lookup dequantized
  the whole quantized table every token** (`to_f32_cow` on `[vocab, hidden]`).
  Now row-wise (`gather_row_f32`, bit-identical, regression-tested): Pi 5
  llama-3.2-1b **1.3→8.3 tok/s (6.4×)**, qwen-1.5b 1.9→6.7, llama-3b 0.8→3.4;
  M4 CPU llama-1b 6.6→38.7, qwen-1.5b 11.5→33.5. **The phase's success metric
  ("1B Q4 usable-interactive on Pi 5") is met.**
- **Success metric:** run a 3B Q4 model on a 4 GB Pi 5 without OOM.

## Phase 4b — Multi-model server  → **`v0.3.x`**
- [x] **Multi-model LRU residency** — keep the N most-recently-used models in memory (`--max-models`, default 3), switchable by the `model` field. Switch-back is a cache hit (no reload), ~5× faster than a cold load; beats Ollama's single-resident-model design.
- [x] **LRU eviction by count + RAM byte budget** (`--cache-gb`, default ~70% of system RAM).
- [x] **Streaming SSE** for `/v1/chat/completions` and `/v1/completions`; cache lock not held during inference, so different models serve concurrently.
- [x] **Admission control** — bounded inference concurrency (`--max-concurrency`, tokio semaphore) so bursts queue instead of oversubscribing.
- [x] **Prefix/prompt caching** — reuse the KV cache for the longest shared token prefix (multi-turn chat / shared system prompts skip re-prefilling history); byte-identical output, verified. `ForwardEngine::truncate_cache` + `Pipeline::enable_prefix_cache`.
- [x] **Speculative decoding wired into `serve`** (`--speculative [--draft-model <alias>]`). `SpeculativePipeline` reuses loaded target+draft engines across requests (`Arc<Mutex<ForwardEngine>>`, no per-request rebuild), gained `*_with_config` + accessors, and is cached via `ServedModel`. Also fixed a pre-existing correctness bug: target verification now uses a cache-aware forward (`forward_all_logits_cached` + `truncate_cache` rollback) instead of resetting the KV cache — output was previously token-salad. Vocab-mismatch guard + family-aware auto-draft.
- [ ] Continuous (in-flight) batching + parallel slots + chunked prefill; paged KV (block pool) — large single-sequence-engine rewrite.
- [ ] OpenAI-compatible `logprobs`, `n` parameters.

## Phase 5 — Phones (iOS / Android)  → **`v0.4.0`**
Most constrained, biggest "wow".
- Library packaging: stable C FFI / UniFFI bindings; static lib for mobile.
- Mobile mmap + thermal/throttle-aware scheduling.
- Sample iOS (Swift) and Android (Kotlin/JNI) apps.
- **Success metric:** a 1–3B Q4 model running on-device in a demo app.

---

## Phase 6 — On-device audio (STT → TTS → STS)  → **`v0.4.x`**
Cross-platform pure-Rust speech, the answer mlx-audio (Apple-only) and the
ONNX-wrapper crates (C++ dep) don't offer together.

- **6a — Whisper STT** ✅ DONE (CPU):
  - `sapient-audio` crate: decode/resample (`symphonia`+`rubato`) + Whisper log-mel
    front-end (`realfft`, slaney filterbank — numerically aligned to OpenAI/librosa).
  - `WhisperForward` engine + `AudioEngine` (encoder + decoder, growing self-attn KV
    cache, cross-attn K/V cached once per chunk) reusing `LlmBackendDispatch` for
    linear/layernorm/add. New kernels: `conv1d` (wraps `conv2d`), `gelu_erf` (exact
    erf GELU). Attention uses the CPU flash kernel with **explicit masks** (all-zeros
    for the non-causal encoder + cross-attn; causal for decoder self-attn).
  - `WhisperTokenizer` (control tokens + forced-prompt protocol + language detection),
    `TranscribePipeline`, `sapient transcribe <model> <audio>`, registry rows for
    `whisper-{tiny,base,small}`. Verified end-to-end on the JFK clip with `whisper-tiny`.
- **6b — GPU offload of the audio transformer body** ✅ DONE (`--features wgpu --backend wgpu`):
  - New WGSL kernels: `layer_norm` (with bias), exact-erf `gelu` (elementwise op=2),
    a broadcast `add_bias` (op=3), a `transpose_heads` (seq↔heads), and a `causal`
    flag on `attention` (non-causal for the encoder + cross-attn). All validated
    bit-close to CPU in `tests/resident.rs`.
  - `WhisperWgpuEngine` (`forward/whisper_wgpu.rs`) mirrors `WhisperForward` on the
    GPU: weights upload once as f32; encoder + decoder blocks (LayerNorm/matmul/
    attention/GELU/residual) run on-device; self-attn KV cache + cross-attn K/V are
    GPU-resident; only logits read back. mel/STFT/conv stay CPU (cheap, once/chunk).
  - `AudioEngine::WhisperWgpu` + `TranscribePipeline` wiring; verified end-to-end —
    `sapient transcribe whisper-tiny jfk.wav --backend wgpu` produces the identical
    transcript to CPU. Coherence test: `tests/whisper_wgpu_coherence.rs`.
  - **Perf note:** on small models / short clips the GPU path currently *trails* CPU
    (tiny 3.1 s vs 1.3 s, base 5.7 s vs 1.8 s end-to-end on M-series/Metal) — per-process
    GPU init + the one-token-at-a-time decoder with a logits read-back each step dominate
    the tiny GPU compute. **Batched prefill** (encode the
    whole forced prompt in one pass) and keeping logits/argmax on-GPU are the optimizations
    that make the GPU win on larger models / longer audio (tracked under 6c).
  - **Default (roadmap 10.4):** on a `wgpu`-feature build, `--backend auto` now
    routes Whisper to the wgpu engine when a GPU adapter actually exists
    (runtime probe, CPU fallback; MLX/Metal keeps precedence on Apple Silicon —
    `whisper_wants_wgpu` in `forward/mod.rs`). Explicit `--backend wgpu` is
    honored as before (and still errors clearly with no adapter). The M-series
    small-model caveat above stands — on Metal-capable Macs the mlx build keeps
    the faster CPU/Metal `WhisperForward`; the auto-wgpu default is for
    GPU-with-weak-CPU boxes (Jetson-class), where the GPU path wins.
- **6c — STT polish** ✅ DONE (branch `feat/audio-tts-sts`): ✅ `suppress_tokens`
  (from `generation_config.json`), ✅ streaming (`transcribe_stream` + live CLI),
  ✅ timestamp tokens + long-audio re-seek (`--timestamps`, ApplyTimestampRules),
  ✅ beam search (`--beam-size`, prefix-replay), ✅ batched prefill (already in the
  engines), ✅ `POST /v1/audio/transcriptions` serve endpoint.
- **6d — TTS** ✅ DONE (**pivoted from Kokoro to LM-codec/SNAC**): `sapient speak
  <model> "<text>" -o out.wav [--voice tara]`. The decisive finding was that an
  **LM-codec TTS** (a Llama-3.2 backbone — **Orpheus-3B** — emitting neural-audio-codec
  tokens, decoded by a small fully-convolutional **SNAC** decoder) reuses SAPIENT's
  existing `LlamaForward` + GGUF + quant + KV cache + sampling *wholesale*, needs
  **no G2P** (raw-text BPE, so no GPLv3 espeak), and collapses Kokoro's ~11 exacting
  kernels (BiLSTM/AdaIN/SineGen/ISTFT) to **ConvTranspose1d + Snake + weight-norm
  fold**. Shipped:
  - **`SnacDecoder`** (`forward/snac.rs`): RVQ-from-codes → conv stack → 24 kHz
    waveform; NoiseBlock omitted (stochastic). conv primitives `conv1d`/
    `conv_transpose1d`/`snake`; **validated bit-close to the torch reference
    (max_err ~2e-6)** via the ignored `snac_coherence` test.
  - **`normalize_snac_weights`**: loads the ungated **`mlx-community/snac_24khz`**
    safetensors mirror out-of-box (`HubClient::download_files`) — folds weight_norm,
    swaps MLX channel-last conv kernels to PyTorch layout, strips `.layers.` prefixes;
    also accepts `scripts/convert_snac_to_safetensors.py` output (or `SAPIENT_SNAC_DIR`).
  - **`SpeakPipeline`** + **`Pipeline::generate_token_ids`** (raw-token-id path) +
    `sapient speak`; Orpheus prompt protocol (`[128259] + tokenizer("{voice}: {text}")
    + [128009,128260,128261,128257]`, **BOS-included**), `orpheus_codes_to_snac`
    7-per-frame de-framing, `write_wav`. 8 voices (tara/leah/jess/leo/dan/mia/zac/zoe).
  - Verified **end-to-end** via the speak→transcribe round-trip (Orpheus speech →
    Whisper STT → original text). (Orpheus 3B Apache-2.0; OuteTTS-1.0 1B Llama but
    CC-BY-NC; Kani 400M but non-Llama LFM2.) Kokoro dropped — worst fit on every axis.
- **6e — STS** ✅ DONE: `EnergyVad` + `SentenceChunker` +
  `ConversePipeline` (STT→LLM→TTS, `Tts` trait) + `cpal` `MicCapture`/`SpeakerPlayback`
  (the `audio-io` feature, **on by default**) + `sapient converse <llm> [--stt] [--tts]
  [--language] [--system] [--speak]` (mic → VAD utterance → STT → streamed LLM reply → optional
  spoken reply; Ctrl-C to stop). Live UX: TTY mic-level meter, OS mic-permission request,
  token-by-token reply streaming, sentence-streamed TTS overlapped with generation, `--input`
  WAV benchmark path. **`--speak` voices the reply** (Kokoro by default — real-time; `--tts
  orpheus` for the richer 3B voice). `--stt` is validated to be a Whisper model.
  Remaining (optional): barge-in + `earshot` VAD upgrade.
- **6f — Kokoro-82M, the real-time TTS** ✅ DONE: the Orpheus/SNAC path (6d) is
  autoregressive (~0.18× real-time on Metal — too slow for live `converse`). Revisited
  Kokoro after a deep-research pass and **ported it pure-Rust** (`forward/kokoro/`):
  non-autoregressive StyleTTS2 + ISTFTNet, one forward pass, **RTF ≈ 0.79 (1.3×
  real-time) on M4 CPU**, ~12× faster than Orpheus. The ~11 "exacting kernels" feared
  in 6d were built + unit-tested (BiLSTM, iSTFT with 1,2,1 irfft + window² OLA, AdaLayerNorm,
  AdaIN1d, NSF SineGen, length-regulator) and the whole model is **validated stage-by-stage
  vs a PyTorch reference** (ALBERT 1e-5 … audio envelope 0.999). G2P via pure-Rust
  `misaki-rs` (no espeak). Weights: offline `.pth→safetensors` (`scripts/convert_kokoro_to_safetensors.py`)
  → mirror `sai1974dev/kokoro-82m-safetensors` (or `SAPIENT_KOKORO_DIR`). `KokoroTts: Tts`
  → `sapient speak kokoro-82m` + **`converse --speak` now defaults to Kokoro**. Apache-2.0,
  54 voices. (Supersedes the "Kokoro dropped" call in 6d — the LM-codec detour shipped a
  voice first; Kokoro shipped real-time.)
- **Success metric (6a):** `sapient transcribe whisper-base sample.wav` produces a
  correct transcript on CPU across macOS/Linux/Windows.

---

## Cross-cutting workstreams (continuous)
- **Correctness harness:** golden-token tests per architecture; CI gate.
- **Bench suite:** RAM + tok/s + time-to-first-token across targets; tracked over time.
- **UX automation:** `sapient` auto-selects a quantization that fits available RAM; `--mem` budget flag; clear "won't fit, try Q4" guidance.
- **Docs:** keep `PROJECT_GUIDE.md` and the README in sync each release.

## Definition of "leading the market"
Match llama.cpp on quantized edge inference (Phases 1–3), then win on:
**install in one line, run any curated model in one command, auto-fit the hardware, pure-Rust everywhere — including phones.**
