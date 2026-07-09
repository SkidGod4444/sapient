# Changelog

Release notes for SAPIENT. The release workflow publishes each version's
section below as the GitHub release body.

## [0.5.3] - 2026-07-09

Three merged PRs: the server grows a vision API, Whisper picks the GPU by
itself, the CLI gets micro-interactions plus a 15-bug fix batch, and GGUF
loading stops F32-exploding exotic quants.

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
