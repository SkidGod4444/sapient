# Changelog

Release notes for SAPIENT. The release workflow publishes each version's
section below as the GitHub release body.

## [Unreleased]

### 📚 Ten more models in the catalog

- Chat: `qwen2.5-7b-q4`, `qwen2.5-coder-3b`, `qwen2.5-coder-7b`, `phi-3.5-mini`,
  `deepseek-r1-1.5b`, `gemma-3-270m`, `smollm2-135m`.
- Speech-to-text: `whisper-medium`, `whisper-large-v3-turbo`.
- Vision: `smolvlm-500m`.
- Each was downloaded and run on the CPU build before it was listed.
- Fix: DeepSeek-R1 distills were prompted in the wrong chat format and echoed the
  question. They now use DeepSeek's own turn markers and stop at its end-of-turn
  token; the Qwen-based distills also load the right tokenizer.

## [0.6.3] - 2026-10-03

**Robot-policy results you can check, and a safer default.** SmolVLA running in
Sapient's 8-bit modes completes as many LIBERO-Spatial tasks as LeRobot's f32
reference (28/50 and 26/50 against 24/50, no measurable difference). A simulator test
with inference delay showed that overlapping slow inference with execution stalls
less but completes fewer tasks, so `sapient act --threshold auto` now runs one chunk
at a time once inference takes more than half a chunk. On phones, large models load
inside the iOS memory limit, and apps can download a model ahead of time with real
progress.

### 📱 Mobile: larger models, benchmarks and downloads

- GGUF models are always memory-mapped on iOS and Android, BF16 checkpoints are
  converted to 8-bit one tensor at a time, the wgpu engine frees each layer after
  uploading it, and models above 1.5B parameters get a 3072-token context window on
  phones (`context_length` overrides). Estimated peak for SmolLM2-1.7B: about 7 GB →
  2.9 GB; not yet measured on a device.
- New in the Swift/Kotlin API (`sapient-ffi`): `benchmark`, memory footprint and
  available-memory readings, `load_time_ms`, `context_length`, and `download_model`
  with byte progress and cancel. A downloaded model then loads with no network.
- Fix: a benchmark or raw completion between two chat turns garbled the next reply.

### 🤖 `sapient act`: safer automatic chunk requests

- `--threshold auto` now runs synchronously whenever inference takes more than half
  a chunk. It used to stay asynchronous until inference took a whole chunk, which
  stalls less but completes fewer tasks: in LIBERO-Spatial with a 1.6 s delay, 4 of
  30 episodes succeeded against 14 of 30 for synchronous execution. Below half a
  chunk nothing changes (no stalls, same success, faster episodes). A fixed
  `--threshold` value still gives the old behaviour.
- `scripts/vla_sim_eval.py --delay D --mode sync|auto` runs LIBERO with a simulated
  inference delay. Results and the long-run check of the stall model on a
  Raspberry Pi 5 (all eight points within 1 point) are in `docs/BENCHMARKS.md`.

### 📥 Download progress that tracks the download

- `download_model` progress jumped to 100 % about a second into a fresh
  download and stayed there: the downloader pre-sizes each partial file to
  its full length, and progress was measured by file length. It now counts
  the bytes actually written, never goes backwards, and ends at the total.
- The download size of a split GGUF (e.g. GLM-4.5-Air, two files) now counts
  every file.

### 🤖 SmolVLA task success in a simulator

- LIBERO-Spatial with the LIBERO-tuned SmolVLA (50 episodes each, identical noise):
  LeRobot's f32 reference 24/50, Sapient `fast` 28/50, `balanced` 26/50 — no
  measurable difference (paired p ≥ 0.29). `scripts/vla_sim_eval.py` reproduces it.
- Fix: checkpoints whose normalization files use different step numbers (such as
  `HuggingFaceVLA/smolvla_libero`) lost their action un-normalization. The file names
  are now read from the checkpoint's processor configs.
- `POST /v1/actions` accepts explicit start `noise` for exact comparison with other
  implementations.

### 📖 Paste-safe docs

- The README's GPU benchmark block started with an interactive `sapient chat`, so
  pasting it opened a chat and the benchmark only ran after `/exit`. It now uses a
  one-shot `chat -p`. The same fix applies to Quick start, the SmolVLA serve + request
  example (now two blocks: `serve` keeps running), the build-from-source steps and the
  TypeScript/mobile SDK snippets. The CLI reference notes that its example lists run one
  line at a time.

## [0.6.2] - 2026-10-02

**Robot actions, a smaller binary, and faster GPU decode.** `sapient act` runs the
SmolVLA vision-language-action policy (camera images + an instruction + robot state →
the next 50 actions), matches LeRobot to 2e-6 on real robot frames, and is served over
HTTP at `POST /v1/actions`. A chunk takes 0.6 s on an Apple M4 CPU and 3.3 s on a
Raspberry Pi 5. The default CPU binary drops from ~50 MB to ~17 MB, the 8-bit matrix
multiply is faster with bit-identical results, wgpu decode is up to 2.5× faster, and
`sapient serve` now explains an unavailable `--backend` instead of returning HTTP 500.

### 📦 Binary 52 MB → 17 MB

- Two-thirds of the binary was Kokoro's English pronunciation dictionaries and
  tagger weights (35.5 MB of JSON embedded by the `misaki-rs` crate). They are now
  gzip-compressed (7.6 MB) and **downloaded next to the Kokoro model** on the first
  `sapient speak` / `converse --speak`. CPU build on Apple Silicon: 52.5 MB → 16.7 MB.
- `--features embed-g2p` builds them into the binary instead (24.6 MB) for fully
  offline use.
- Speech output is byte-identical to before on three test voices (US and British),
  with either build.
- `misaki-rs` is vendored under `vendor/misaki-rs` (MIT) with those changes; see its
  `VENDORED.md` and `NOTICE`.

### 🤖 SmolVLA — robot actions (experimental)

- New `sapient act <image>... --task "…" [--state a,b,c] [--json]`: runs the SmolVLA
  vision-language-action policy (`lerobot/smolvla_base`, 450M) and prints a chunk of 50
  future actions. One or more camera images, an instruction, and the robot state go in.
- Checked against LeRobot's PyTorch implementation at every stage (image embedding,
  prefix K/V, one flow-matching step, the final chunk): the actions match to 4e-6 on the
  same inputs and start noise. `scripts/gen_smolvla_fixture.py` regenerates the reference.
- **Three precisions** (`--precision fast|balanced|exact`, also `"precision"` in the
  serve request). Per chunk, one camera: `fast` (all 8-bit, default) 0.6 s on an Apple
  M4 and 3.3 s on a Raspberry Pi 5 (its vision attention runs in int8); `balanced` (8-bit action expert only) 1.0 s / 5.9 s;
  `exact` (f32) 1.7 s / 11.9 s.
- Action error over eight synthetic observations (RMS, normalized units, typical action
  magnitude 0.36): `fast` 0.012, `balanced` 0.004; LeRobot's own default bf16 precision
  0.005 on the same observations. Task success is not measured.
- **`sapient serve` endpoint** `POST /v1/actions` (`task`, `images` as base64 data URIs,
  optional `state`, `seed`, `steps`): the policy stays loaded, so a call costs inference
  only. `sapient serve lerobot/smolvla_base` preloads it.
- `--steps N` / `"steps": N` runs fewer flow-matching steps. It is faster and much
  coarser: 5 steps halve the denoising time and move the actions by up to 0.24.
- The base checkpoint is a pretraining model meant for fine-tuning and ships no plain
  state/action statistics, so its actions are in normalized space; checkpoints that carry
  `observation.state.*` / `action.*` statistics are normalized and un-normalized.

### 🔧 Fixes from a teammate's GPU benchmark report

- **wgpu decode up to 2.5× faster** (Apple M4): one compute pass per token instead of one
  per kernel, and the decode matrix-vector kernels use their threads instead of idling
  ~80% of them. qwen2.5-0.5b 23 → 43 tok/s (on par with CPU), qwen2.5-1.5b-q4 13.5 → 34.
  Output is unchanged. wgpu is still slower than the CPU for 4-bit models on Apple
  Silicon.
- **`sapient serve --backend metal` on a build without Metal** (or `--backend wgpu`
  without GPU support) now refuses to start and says why and what to install, instead
  of answering every request with a bare HTTP 500. Load failures in other routes now
  include the underlying cause.
- `scripts/bench_wgpu.py` prints why a backend was skipped and carries on, instead of
  crashing; the first-request timeout allows for a model download.
- The chat hint for full-precision models said "4-8× faster"; measured it is about
  2–3× on CPU (qwen2.5-0.5b: 44 → 113 tok/s on an M4), and it is no longer shown when
  running on a GPU backend, where it isn't reliably true.
- Docs: shell examples no longer put `# comments` at the end of commands. In zsh, the
  macOS default, those became arguments (`zsh: unknown file attribute`). 63 commands
  fixed across the README, CONTRIBUTING and docs.
- The microphone-permission code uses `objc2`/`block2`; default builds no longer show
  the `block v0.1.6` future-incompatibility warning. Builds with `--features wgpu`
  still do, through wgpu 22 itself (fixed upstream in wgpu 30).

### 🤖 SmolVLA: real-data check and asynchronous chunking

- Checked on 24 real frames from an SO-100 dataset (two cameras): `exact` matches
  LeRobot to 2e-6; `fast` deviates by RMS 0.019 and `balanced` by 0.014, against 0.011
  for LeRobot's own bf16 default. No mode changes the error against the recorded
  actions (0.79). `scripts/smolvla_dataset_eval.py` produces the comparison data.
- `sapient act --simulate --hz H [--threshold T]`: a control loop that computes the next
  chunk while the current one executes (`AsyncActions` in the library) and reports
  stalls and inference latency. No stalls at 30 Hz on an Apple M4 or at 5 Hz on a
  Raspberry Pi 5.
- `--threshold auto` (default) picks when to request the next chunk from the measured
  latency: just in time when inference is fast, at once when it is moderately slow,
  and synchronous when it is slower than a chunk. It matched the best fixed setting at
  every rate tested on the M4 and the Pi 5, and computes ~20% fewer chunks than
  requesting as early as possible.

### ⚡ Faster 8-bit matrix multiply

- The Q8_0 GEMM used by vision towers, prefill and SmolVLA is 1.3–1.7× faster on an
  Apple M4 at 10 threads, with bit-identical results (4×4 register tile, in-place
  output, finer task split). `sapient see` image encode: about 555 → 397 ms on the M4.
  On a Raspberry Pi 5 a SmolVLA chunk went from 5.15 s to 3.65 s.

### 👁️ SmolVLM2-500M

- `sapient see --model smolvlm2-500m`: the SmolVLM2 family loads through the existing
  Idefics3 path (config `model_type: "smolvlm"`). It is the VLM that SmolVLA is built on.
- Its checkpoint is full F32 (1.9 GB); the SmolVLM loader now quantizes F32 linears to
  Q8_0 at load, as the engines already do for F16/BF16. Apple M4: image encode
  ~700 → 567 ms, 77-token prefill 328 → ~115 ms, answers byte-identical on the test images.
- The vision gates (`vlm_e2e`, `vlm_geometry_probe`) accept `SAPIENT_VLM_MODEL` and pass
  on both models; `scripts/bench_loop.py --vlm-model` picks the model to time.

## [0.6.1] - 2026-10-02

**A correctness and measurement release.** Qwen prompts no longer carry a stray
`<s>` token, image encoding is about twice as fast with bit-identical output
(Raspberry Pi 5: 7.3 s → 3.5 s per image), and the project now has a perplexity
gate that shows Sapient within 0.1–0.4% of llama.cpp on the same model files.
`openhorizon-labs/sapient` is the canonical download location.

### 🐛 Qwen prompts no longer start with a stray `<s>`

- The tokenizer wrapper chose its BOS by name; Qwen2.5's vocabulary has `<s>` as an
  ordinary token and the model has no BOS, so every Qwen prompt carried a stray token.
  BOS detection now requires a special added token (two regression tests).
- `sapient eval-ppl`: chunk boundaries now match llama.cpp for models with a BOS; new
  `--cached` and `--dump-nll`; `scripts/ppl_paired.py` gives paired intervals.
  Corrected quality numbers (Apple M4 CPU): Qwen2.5-1.5B 11.730 vs llama.cpp 11.720;
  Llama-3.2-1B 16.328 vs 16.259. The earlier figures in this file are superseded.
- `SAPIENT_F32_ACT=1` (diagnostic) forces f32 activations in the quantized matmuls.

### 🎯 Perplexity gate

- `sapient eval-ppl` (hidden) scores a text file with llama.cpp's perplexity protocol,
  so quality can be compared on the same GGUF. (The first numbers published here were
  taken with two protocol bugs and are superseded by the entry above.)
- `scripts/bench_loop.py` now records a `quality` section.

### 🔁 Optimisation-loop scaffolding

- `scripts/bench_loop.py` — one-command reproducible baseline (LLM decode / TTFT /
  peak RSS, llama.cpp on the same GGUF, vision encode p50/max/stdev, binary size);
  first result file in `benchmarks/`.
- Fixed: `llama-3.2-3b` and `mistral-7b` are now marked gated in `sapient models`.

### 📦 `openhorizon-labs/sapient` is the canonical download location

- `install.sh`, `install.ps1`, the README install commands and release badges, the
  Homebrew formula URLs, and the SwiftPM binary URL now all point at the
  `openhorizon-labs/sapient` release (the same place `sapient update` already
  used). Source links stay on `SkidGod4444/sapient`, where the code lives.

### 📖 README refresh

- New "What it is" and quick-start sections; CLI commands grouped by task; the
  performance section leads with one table and keeps every caveat in a
  "read before quoting" block; adds the 2026-10 vision numbers.
- Corrected: `serve` examples use the real default port (11435), the gated-model
  list, the `-q4` model table, and "pure Rust" wording.

### 👁️ Vision tower ~50% faster (bit-identical)

- SmolVLM image encode: **Apple M4 1140 → ~555 ms** (4 threads: 1410 → ~750 ms);
  **Raspberry Pi 5 7.3 s (v0.5.2) → 3.5 s**. Five bit-identical changes: the
  element-wise map (GELU) runs in parallel for large tensors; the head
  split/merge `permute` copies whole `head_dim` runs (also speeds LLM prefill);
  the blocked W8A8 GEMM processes four activation rows per weight row
  (`dot_q8_0_row_sdot_x4`); it now walks cache-sized activation panels (on the
  Pi the old loop was memory-bandwidth-bound); and the tower attention is tiled
  over query rows instead of materialising a 4 MB score matrix per head.
- `SAPIENT_VISION_TIMING=1` prints the tower's per-stage breakdown;
  `SAPIENT_Q8_PANEL_KB` / `SAPIENT_ATTN_TILE_KB` override the panel / tile
  sizes for tuning.
- Fixed: `--no-default-features` builds failed to compile (`server.rs` referenced
  the optional audio crate directly).

### 📏 Benchmark tooling and docs — measure what we say we measure

- **`sapient bench-llm` now reports decode-only throughput.** Greedy decode with an
  exact token count, `(tokens − 1) / (t_last − t_first)`; previously it divided the
  re-tokenized reply length by total time *including* TTFT and averaged every run.
  New `--warmup N` (default 1) runs are excluded from the means and listed
  separately in the JSON; "peak RSS" is now the real high-water mark (`getrusage`)
  instead of end-of-run RSS; local `.gguf` paths work as the help text always
  claimed. JSON keeps its existing keys and adds `method`, `warmup`, `threads`,
  `sapient_version`, per-run `e2e_tps`/`hit_eos`, and `summary.decode_tps`.
- **`scripts/gen-benchmark-report.py` no longer substitutes placeholder numbers**
  when an input file is missing — it exits with an error.
- **Docs:** `docs/BENCHMARKS.md` gains a method section (old-vs-new tool, thread
  asymmetry, Metal 4-bit re-quantization, no quality eval yet) and an independent
  v0.6.0 M4 reproduction with raw output committed under
  `docs/assets/bench_2026-10-01/`. Stale sections are dated and marked historical;
  the "22 MB binary", "beats the M4 CPU" (wgpu), "1.5× Ollama" and Pi 11.6 tok/s
  figures are corrected or caveated across README, PROJECT_GUIDE, PI, ROADMAP and
  SERVING_BENCHMARKS. The dead Homebrew tap line is removed from the README.

## [0.6.0] - 2026-07-14

**SAPIENT becomes an agent backend, and goes mobile.**

`sapient serve` now speaks OpenAI **tool calling**, so the Vercel AI SDK,
LangChain or the OpenAI SDK can drive it unmodified — closing the last gap that
forced users to keep a second engine around just for the action-selecting model.
Speech gains an HTTP surface too (`POST /v1/audio/speech`).

The engine also runs **on-device** in Swift, Kotlin and React Native apps —
**GPU by default** (Metal on iOS/macOS, Vulkan on Android) with **engine-level
thermal governance** — and the SDKs install the idiomatic way per ecosystem:
**SwiftPM by URL, Maven for Android, npm for TypeScript**. The CPU-parity ladder
closes with the Q8_K activation format (**Jetson Thor dense decode +46%
cumulative since v0.5.0**), and a model beta-test sweep fixes four user-facing
bugs.

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
