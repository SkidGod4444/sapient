# Changelog

Release notes for SAPIENT. The release workflow publishes each version's
section below as the GitHub release body.

## [Unreleased]

**SAPIENT becomes an agent backend.** `sapient serve` now speaks OpenAI **tool
calling**, so the Vercel AI SDK, LangChain or the OpenAI SDK can drive it
unmodified — closing the last gap that forced users to keep a second engine
around for the action-selecting model. Speech gains an HTTP surface too.

### 🔧 Tool calling — `tools` / `tool_choice` on `/v1/chat/completions`

- **The models could already do this.** Every `qwen2.5-*` alias resolves to
  Qwen2.5-Instruct, whose chat template carries a `{%- if tools %}` branch and
  which is trained to answer with `<tool_call>{…}</tool_call>`. SAPIENT simply
  never told it: the Jinja context was built from `messages` alone, and an
  incoming `tools` array was dropped by serde. An agent loop pointed at SAPIENT
  would never actuate anything — and never say why.
- `tools` now flows render → `GenerationConfig` → server; the model's
  `<tool_call>` blocks are lifted back into OpenAI `tool_calls`; and the return
  leg (`role: "tool"` results, assistant turns with `"content": null`) is
  accepted. A malformed or truncated call stays visible as text rather than
  vanishing into an empty assistant turn.
- **`tool_choice` is binding, not advisory.** `"required"` and a named function
  *force* a call — implemented by prefilling the assistant turn, so prose is no
  longer a reachable continuation. This is what a caller needs when an answer
  must not come from imagination.
- **`$schema` is stripped from tool definitions.** Zod v4 — and therefore every
  Vercel AI SDK tool — tags each schema with a draft-07 dialect URL. Templates
  serialize tools verbatim into the model's preamble, and that URL alone is
  enough to push a small model off-distribution: Qwen2.5-1.5B stops emitting
  `<tool_call>` and answers in prose. The framework most likely to be pointed at
  SAPIENT was the one guaranteed to trip it.
- `minijinja` gains its `json` feature — without the `tojson` filter, every
  tool-aware chat template fails to render at all.

### 🔊 `POST /v1/audio/speech`

- Kokoro TTS over HTTP (54 voices, WAV/PCM), behind an LRU cache mirroring the
  STT one. `KokoroTts::synthesize_as` lets one cached engine serve every voice,
  and `sapient_audio::encode_wav` returns audio without staging a temp file.
  MP3 is rejected loudly rather than mislabelled as WAV bytes.

### ⚠️ For agents, model size is a correctness knob

Tool-calling quality falls off sharply below ~3B. Qwen2.5-1.5B answers
perception questions from imagination under `tool_choice: "auto"` — it will
describe a scene it never looked at rather than call the tool. 3B calls it.
Prefer 3B+ for agent work, or force the call with `tool_choice`.

Verified end-to-end against `ai@7.0.26` + `@ai-sdk/openai-compatible@3.0.9`: a
`ToolLoopAgent` drives multi-step tool use on both the streaming and
non-streaming paths.

## [0.6.0] - 2026-07-13

SAPIENT goes mobile: the engine runs **on-device** in Swift, Kotlin, and React
Native apps — **GPU by default** (Metal on iOS/macOS, Vulkan on Android) with
**engine-level thermal governance** — and the SDKs install the idiomatic way
per ecosystem: **SwiftPM by URL, Maven for Android, npm for TypeScript**. The
CPU-parity ladder closes with the Q8_K activation format (**Jetson Thor dense
decode +46% cumulative since v0.5.0**), and a model beta-test sweep fixes
four user-facing bugs.

### 📦 SDK distribution — how you get the SDKs

- **Swift**: add `https://github.com/openhorizon-labs/sapient-swift` in Xcode —
  its `Package.swift` points a checksum-pinned remote `binaryTarget` at the
  release's `SapientFFI.xcframework.zip` asset, re-pointed by the release
  workflow on every tag.
- **Android**: a git-hosted Maven repository at
  `openhorizon-labs/sapient-android` —
  `implementation("so.openhorizon:sapient:0.6.0")` (+ one `maven { url }`
  line); the AAR's POM carries JNA and kotlinx-coroutines as transitive
  deps. Maven Central is a later rung.
- **npm**: the TypeScript SDK publishes as **`@openhorizon-labs/sapient`**
  (the React Native on-device package shares the scope but stays
  repo-distributed for now: its native libs are monorepo build outputs).
- **release.yml**: `dist-swift`, `dist-android-maven`, and token-gated
  `publish-npm` jobs; the `openhorizon-labs/sapient` binary mirror waits for the
  mobile packaging jobs, so the SDK zips reliably reach it.
- **README**: mobile section rebuilt — per-platform install snippets +
  on-device screenshots (all three stacks) + a GPL-3.0 embedding note;
  `docs/MOBILE.md` gained a consumption-first quickstart.

### 🐛 Android on-device fixes (found by the first real emulator run)

- **`libsapient_ffi.so` linked the NDK's *shared* C++ runtime**
  (`libc++_shared.so`) — which nothing ships to consumer apps, so every app
  died at first load with `UnsatisfiedLinkError`. The C++ runtime is now
  static (`CXXSTDLIB=c++_static` + `-lc++abi`), and `package-android.sh`
  gates on it (readelf NEEDED + undefined-C++-symbol checks). Invisible to
  `assembleDebug`; only a real dlopen catches it.
- **`HF_HOME` was silently ignored by the model downloader** — `HubClient`
  used hf-hub's `ApiBuilder::new()`, which hard-codes the home-dir cache and
  panics (`Cache directory cannot be found`) on Android, where app processes
  have no home. Now `ApiBuilder::from_env()`: `set_cache_dir` / `HF_HOME`
  is honored on **every** platform (macOS/iOS previously worked only because
  a home dir happened to exist).
- Kotlin sample app **emulator-validated end-to-end** for the first time —
  a real streamed turn on `smollm2-135m-q4`, and it ran on
  **wgpu→Vulkan** (the emulator's SwiftShader software Vulkan), proving the
  quantized WGSL stack on Android. Screenshot in the README.

### 📱 Mobile & embedding SDKs — Phase 11 (#38, #39, #40, #43, #44, #46)

- **`sapient-ffi` (UniFFI)** — `LlmSession` chat + streaming-with-cancel over
  the existing `Pipeline` (prefix cache on), generating idiomatic **Swift**
  and **Kotlin**. Async exports (`load_session`, `chat_async`,
  `chat_stream_async`, `chat_messages_stream`) keep JS/Hermes hosts unblocked;
  `set_cache_dir` and `set_thermal_level` round out the embedding surface.
- **Packaging — one command per platform**: `scripts/package-swift.sh` →
  `SapientFFI.xcframework` (iOS device + simulator + macOS slices) inside a
  local Swift Package, gated by a compile-and-run smoke link;
  `scripts/package-android.sh` → a drop-in `com.android.library` Gradle
  module. **This is the first release to attach `sapient-swift.zip` and
  `sapient-android.zip`** (+ sha256) alongside the CLI binaries.
- **GPU on-device by default** — the mobile packages compile the wgpu backend
  in (**Metal on iOS/macOS, Vulkan on Android**; `--cpu-only` opts out).
  `Auto` probes for a usable adapter before routing to the GPU, so a broken
  driver or GPU-less emulator falls back to CPU instead of failing. Gate
  passed: a real inference turn inside the iOS-simulator app on wgpu→Metal,
  quantized-resident Q4_K/Q6_K weights + f16 KV cache.
- **Engine-level thermal governance** —
  `set_thermal_level(nominal|fair|serious|critical)` caps decode threads at
  full/¾/½/¼ of cores (the stricter of this and the sysfs governor wins). The
  sample apps carry the verified reference wiring: iOS
  `thermalStateDidChangeNotification` (+ Low Power Mode clamp) with its two
  documented traps handled, Android `PowerManager.addThermalStatusListener`
  with Google's ADPF mapping. MLC, llama.cpp-mobile, and MediaPipe ship no
  engine-side thermal response.
- **React Native on-device** — `@openhorizon-labs/sapient-react-native`:
  uniffi-bindgen-react-native generates the TS + JSI TurboModule straight
  from the FFI crate; the TypeScript SDK gained a `Transport` seam
  (`HttpTransport` unchanged default, `NativeTransport` runs the engine
  in-process). Example app defaults to on-device with a server-mode toggle.
- **TypeScript SDK** (`sdks/typescript`, `@openhorizon-labs/sapient`) —
  zero-dependency client for `sapient serve`: `chat`, SSE `chatStream` with
  cancel-on-break, `models`, `health`; injectable `fetch` (React Native
  streams via `expo/fetch`).
- **Three sample chat apps** (`examples/`) — SwiftUI (macOS + iOS), Jetpack
  Compose, and Expo/React-Native, all streaming with engine-side Stop and
  greedy sampling defaults; CI builds all three on every PR. Full build +
  personal-hardware safe-testing guide: `docs/MOBILE.md`.

### ⚡ CPU parity round 2 (#32, #33, #35, #36, #37)

- **Q8_K activation format, default ON** for the Q4_K and Q6_K int8 decode
  paths (one f32 scale per 256-element super-block; weight sub-scales
  combined in the integer domain — llama.cpp-precedented accuracy class):
  Jetson Thor 14-core dense decode **+44.5%** / prefill TTFT −16.3%
  (combined off→on), M4 qwen-1.5B +12.5%, Pi 5 +6.8%. `SAPIENT_Q8K_ACT=0`
  reverts. Every kernel bit-identity-gated against a scalar oracle.
- **Guided spin/park decode threadpool** replacing ~230 per-token rayon
  fork/joins: M4 llama-1B **+7.7%**, Thor 14-core +5.3%; topology-aware
  block claiming (block=1 on P/E-heterogeneous macOS, ~3 blocks/participant
  on homogeneous server ARM). Default ON for macOS and Linux/aarch64 ≥ 8
  threads; `SAPIENT_SPINPOOL=0` reverts.
- Precomputed per-row activation block-sums for Q4_K's `dmin·mn` term
  (bit-identical, ~+1% decode).
- **Cumulative since v0.5.0: Thor dense decode 22.4 → 32.8 tok/s (+46%);
  llama.cpp CPU decode gap 3.16× → ~2.6×.** The ladder's final rung
  (vectorized SMMLA combine) measured neutral and was reverted with the
  record; every falsified design is documented in `docs/BENCHMARKS.md`.

### 🐛 Correctness (model beta-test sweep)

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
