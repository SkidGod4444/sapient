# Changelog

Release notes for SAPIENT. The release workflow publishes each version's
section below as the GitHub release body.

## [Unreleased]

Model beta-test sweep (every downloaded model, long-form prompts) — four user-facing bugs found and fixed, plus new MLX debug tooling.

### 🐛 Correctness

- **Q5_K dequantization fixed in `sapient-core` `Tensor::to_f32_vec`**: the 5th
  bit was read from one `qh[is/8]` byte per 32-element sub-block instead of
  ggml's per-element `qh[l]` — corrupting every Q5_K tensor dequantized through
  `to_f32_cow` (the MLX requantize path). phi-4-mini (whose unsloth Q4_K_M GGUF
  stores q/k/v as Q5_K) emitted degenerate "mememe" output on `--backend
  metal`. Same bug class as the CPU scalar kernel fixed in v0.3.9; both copies
  now match, regression test `q5_k_dequant_high_bits_per_element`.
- **Phi-3/Phi-4 `<|end|>` added to `EOS_CANDIDATES`**: `<|end|>` is
  `special: true`, so `decode` strips it and it can never match as a stop
  *string* — with it missing from the EOS id list, phi-4-mini blew past its
  end-of-turn and rambled/repeated on every backend.
- **SmolLM2 GGUFs got the wrong builtin chat template**: Llama-arch + generic
  "llama" model_type fell into the LLAMA2 `[INST]` arm, but SmolLM2 is
  ChatML-trained — every chat reply came back empty. New `smollm` → ChatML arm
  in `builtin_template_for`.

### 💬 Chat UX

- **`sapient chat -n/--max-tokens <N>`** (default raised 512 → 2048 for chat),
  and a reply that stops at the cap now prints a truncation notice (stderr, so
  `chat -p` stdout stays scriptable) via the new
  `Pipeline::last_reply_truncated()` — long answers no longer silently cut off
  mid-sentence.

### 🔧 Debug tooling

- `SAPIENT_MLX_DISABLE=<op,…|all>` (force listed MLX ops onto the CPU
  reference kernel), `SAPIENT_MLX_VERIFY=1` (cross-check MLX `linear_3d`
  against CPU, print per-weight divergence), `SAPIENT_MLX_NO_QUANT=1` (force
  the F32 matmul path) — per-op bisection of wrong-numbers GPU kernels without
  rebuilding.

## [0.5.3] - 2026-07-09

Sparse MoE lands (Mixtral 47B and GLM-4.5-Air 106B on a Jetson, pure Rust,
zero CUDA), the server grows a vision API, Whisper picks the GPU by itself,
the CLI gets micro-interactions plus a 15-bug fix batch, and GGUF loading
stops F32-exploding exotic quants.

### 🧠 Sparse MoE — Mixtral-class + GLM-4.5-Air (#28)

- Big-MoE on edge: a per-layer `Ffn::{Dense, Moe}` branch inside the Llama
  engine (softmax → top-k → renorm routing), both GGUF expert layouts plus
  safetensors, CPU-first. **Mixtral-8x7B (47B) verified end-to-end on a
  Jetson AGX Thor — pure Rust, zero CUDA, greedy token-identical to
  llama.cpp** (decode 5.5 tok/s, RSS ≈ file size via mmap). SAPIENT loads the
  classic per-expert Mixtral GGUFs that current llama.cpp rejects.
- **GLM-4.5-Air (106B-A12B)**: sigmoid gate + aux-loss-free correction bias +
  always-on shared expert, partial RoPE, split-GGUF loading (2-shard ~63 GB)
  with zero-copy stacked-expert mmap views — decode-verified on Thor. With
  the Q8_0 re-quantization below it now fits a 96 GB device.

### 👁 Vision over HTTP — image parts in `/v1/chat/completions` (roadmap 12.3, #30)

- OpenAI-style content parts: `{"type":"text"}` + `{"type":"image_url"}` with
  **base64 data URIs** — the server never fetches remote image URLs (no
  surprise egress from your inference box). Plain-string clients are untouched.
- Vision requests run the `sapient see` engine (SigLIP tower + embedding
  splice) in a third LRU cache beside text/audio, sharing the load lock,
  admission control, and RAM budget. `usage` counts real text+image tokens.
- Verified end-to-end: smolvlm-256m answers the red-fixture PNG with "Red"
  over HTTP, streaming and non-streaming.
- Also: OpenAI `system`/`developer` roles now map to the chat template's
  system role (they were silently rendered as user turns).

### 🎙 Whisper auto-selects the GPU (roadmap 10.4, #30)

- On a `wgpu` build, `--backend auto` routes Whisper to the GPU engine when an
  adapter actually exists (runtime probe, CPU fallback; MLX/Metal keeps
  precedence on Apple Silicon). One gate covers `transcribe`, `converse`, and
  `POST /v1/audio/transcriptions`. Explicit `--backend wgpu` still errors
  clearly when no GPU is present.

### ✨ CLI micro-interactions + UX bug batch (#31)

- **Streaming cursor**: a dim `▍` rides the reply during live Markdown
  rendering. **Spinner receipts**: loads settle into `✓ model ready (768ms)`
  instead of vanishing; all spinners show live elapsed time (a Pi never looks
  hung). **Mic meter peak-hold** tick in `converse`. Truecolor gradient
  wordmark. Download completion prints size + duration.
- Fixed, among 15: `say`/`tts` aliases ran the **vision** command; the pull
  progress bar counted every quant in the repo (crawled to ~8%, then jumped to
  done); Ctrl-C was swallowed mid-converse-turn and killed chat at the prompt
  (now shell-conventional); `sapient serve` left a stale lock on Ctrl-C and
  loaded models in silence without `--verbose`; `<think>` reasoning could leak
  dim ANSI into the shell and leaked into `--raw` chat history; errors now
  print a root line + `↳` cause chain; tables align with unicode/styled cells;
  `stats` no longer streams escapes when piped.

### 📦 GGUF: unsupported quants re-quantize to Q8_0 at load (#29)

- Quantized GGUF types SAPIENT can't keep as packed blocks (e.g. **Q5_0** in
  unsloth "dynamic" quants) used to F32-expand on load. GLM-4.5-Air Q4_K_M
  carries 24 such `ffn_down_exps` tensors — **70 GB of heap**, 118 GB peak RSS
  on a 122 GB Jetson Thor. They now dequantize → re-quantize to Q8_0
  (~1.06 B/weight, near-lossless since 8 bits ⊇ the source's ≤6). Measured on
  Thor: **peak RSS 118 → 72 GB** (heap 66 → 18 GB), decode 2.45 → **3.23 tok/s**
  (+32%), prefill 0.80 → **3.94 tok/s** (5×) — the memory-pressure relief ends
  the page-thrash. GLM-4.5-Air now fits a **96 GB** device (was 128 GB+).
  Applies to both the mmap and heap loader paths.

### Pi 5 voice loop re-measured (roadmap 8.5, #30)

Same WAV-injected `converse --input` turn, v0.5.2 release binary, Pi 5 16 GB:
0.5B ≈ **11.9 s** sequential (STT 2.96 s · LLM 3.5 s · TTS 5.4 s), 1.5B ≈
12.6 s. STT improved 3.5 → 2.96 s vs v0.4.4; Kokoro (RTF ~2.4) remains the
dominant stage. Open observation: in-loop LLM TTFT is 2.4 s vs 116 ms
bare-chat — under investigation. Full tables in `docs/PI.md`.

### Still open

Intel Arc / AMD GPU datapoints (7.6 — `scripts/bench_gpu_7_6.sh` ships in the
release), Jetson Orin Nano decision gate (7b), Metal RAM tax (Phase 9),
mobile bindings (Phase 11).

## [0.5.2] - 2026-07-04

See the [v0.5.2 release notes](https://github.com/SkidGod4444/sapient/releases/tag/v0.5.2)
— vision-language (`sapient see`), Gemma3 engine + MedGemma, streaming voice
loop, W8A8 GEMM. (Earlier releases are documented on their GitHub release
pages.)
