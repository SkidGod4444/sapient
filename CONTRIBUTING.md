# Contributing to SAPIENT

Thank you for your interest in contributing! SAPIENT is a Rust-native LLM inference engine, and every bug report, doc fix, and code contribution helps.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [License](#license)
- [Before you start](#before-you-start)
- [Getting set up](#getting-set-up)
- [Project structure](#project-structure)
- [Development workflow](#development-workflow)
- [Testing](#testing)
- [Formatting and linting](#formatting-and-linting)
- [Commit messages](#commit-messages)
- [Pull requests](#pull-requests)
- [Common contribution areas](#common-contribution-areas)
- [Adding a CLI command](#adding-a-cli-command)
- [Adding a model architecture](#adding-a-model-architecture)
- [Hub and download testing](#hub-and-download-testing)
- [Releases (maintainers)](#releases-maintainers)
- [Getting help](#getting-help)

---

## Code of conduct

Be respectful, constructive, and inclusive. We want this project to be welcoming to contributors at every experience level. Harassment, spam, and bad-faith arguments are not tolerated.

---

## License

SAPIENT is licensed under **[GPL-3.0-only](LICENSE)**.

By contributing code, documentation, or other materials, you agree that your contributions will be licensed under the same terms. If you modify SAPIENT and distribute binaries, those binaries must also comply with the GPL.

---

## Before you start

1. **Search existing issues** — someone may already be working on it.
2. **Open an issue first** for large changes (new backends, architecture rewrites, breaking API changes).
3. **Keep PRs focused** — one logical change per pull request is easier to review and merge.
4. **Prefer minimal diffs** — match existing style and conventions in the crate you are editing.

---

## Getting set up

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Rust | **1.82+** | Install via [rustup](https://rustup.rs/) — matches the workspace `rust-version` |
| Git | any recent | |
| ALSA dev headers | Linux only | `sudo apt-get install libasound2-dev` — needed because `sapient-cli`'s default `audio-io` feature (live `sapient converse`) links `cpal`/ALSA. macOS/Windows need nothing extra. |
| [just](https://github.com/casey/just) | optional | Task runner — see `justfile` |

### Clone and build

```bash
git clone https://github.com/SkidGod4444/sapient.git
cd sapient
cargo build --workspace
# No audio device libs / don't want the live `converse` command? Drop the
# default audio-io feature (chat/transcribe/speak/serve are unaffected):
#   cargo build -p sapient-cli --no-default-features
```

Run the CLI locally:

```bash
cargo run -p sapient-cli -- --help
cargo run -p sapient-cli -- chat <model>
```

Release binary (faster inference):

```bash
cargo build --release -p sapient-cli
./target/release/sapient --version
```

If you use `just`:

```bash
just build      # debug build
just release    # release build
just --list     # all tasks
```

---

## Project structure

SAPIENT is a Cargo workspace. Crates are layered from low-level primitives up to the user-facing API.

```
crates/
├── sapient-core/           # Tensors, dtypes, buffers
├── sapient-ir/             # Computation graph IR and passes
├── sapient-io/             # Safetensors, GGUF, ONNX loaders
├── sapient-backends/cpu/   # CPU inference kernels
├── sapient-backends/metal/ # Apple GPU backend (MLX)
├── sapient-backends/wgpu/  # Cross-platform GPU (Vulkan/DX12/Metal via wgpu)
├── sapient-scheduler/      # Request scheduling and batching
├── sapient-telemetry/      # Tracing and metrics
├── sapient-runtime/        # InferenceSession, Model
├── sapient-hub/            # HuggingFace Hub client
├── sapient-tokenizers/     # Tokenizers + chat templates + WhisperTokenizer
├── sapient-audio/          # Audio front-end: decode/resample + log-mel STFT (Whisper)
├── sapient-models/         # Forward engines (Llama, Phi, …) + AudioEngine (Whisper STT) + SnacDecoder (TTS)
├── sapient-generate/       # Pipeline API (from_pretrained, chat, stream) + TranscribePipeline + SpeakPipeline
├── sapient-ffi/            # Embedding surface: UniFFI → Swift/Kotlin bindings
│                           #   (staticlib/cdylib for iOS/Android; see docs/MOBILE.md)
└── sapient-cli/            # `sapient` binary (chat REPL uses a rustyline line
                            #   editor + markdown.rs live Markdown/code rendering)

sdks/typescript/            # @openhorizon/sapient — TS SDK for Node.js/React Native
                            #   (talks to `sapient serve`; npm test = tsc + node --test)
examples/                   # Sample chat apps: swift-chat (SwiftUI macOS+iOS),
                            #   android-chat (Compose), react-native-chat (Expo)
install.sh / install.ps1    # Install scripts (attached to releases)
Formula/sapient.rb          # Homebrew formula template
.github/workflows/          # CI and release automation
tests/                      # Workspace integration tests
```

**Dependency direction (simplified):**

```
sapient-cli → sapient-generate → sapient-hub, sapient-models, sapient-tokenizers
sapient-ffi → sapient-generate   (embedding surface — same layer as the CLI)
sapient-generate → sapient-runtime → sapient-backends-cpu, sapient-io, sapient-ir
```

When adding a dependency, keep layers acyclic — lower crates must not depend on higher ones.

### Key subsystems to know before contributing

**Forward engines, not one per architecture.** `LlamaForward`, `PhiForward`, and `Gemma3Forward`
are wired to the live chat `Pipeline`. Architecture builders in `sapient-models/src/architectures/`
target the IR graph path and are **not** used during inference. Adding a new model usually means
adding (or extending) a forward engine, not an architecture builder (unless it is for the graph
path). Some variants extend an existing engine rather than fork it: **Mixtral-class sparse MoE**
lives as a per-layer `Ffn::{Dense, Moe}` branch **inside `LlamaForward`** (shared attention/KV/RoPE;
only the FFN block differs), detected by config (`ModelInfo.moe`), not `ArchType` — a Mixtral GGUF
reports arch `llama`. See the MoE section in `CLAUDE.md` for the routing math, the two GGUF expert
formats, and the CPU-only backend gate.

**Audio (speech-to-text) is a separate path.** `WhisperForward`/`AudioEngine`
(`forward/whisper.rs`) and `TranscribePipeline` (`sapient-generate`) are independent of the text
`ForwardEngine`/`Pipeline` — `sapient transcribe` never touches the chat path. The front-end lives
in `sapient-audio` (CPU log-mel via `realfft`). Two traps when editing it: the CPU
`scaled_dot_product_attention` treats `mask=None` as **causal**, so the non-causal encoder and
cross-attention must pass an explicit all-zeros mask; and Whisper needs **exact erf GELU**
(`gelu_erf`), not the tanh `gelu`. Verify with `tests/whisper_coherence.rs` (synthetic, exact) and
the ignored `transcribe_e2e.rs` (real `whisper-tiny`).

**Text-to-speech also reuses the text engine, not a new one.** `sapient speak` runs **Orpheus-3B**
(a Llama-3.2 fine-tune) on the existing `LlamaForward`/GGUF path via `Pipeline::generate_token_ids`
(a raw-token-id generation method — no detokenize), then decodes the emitted SNAC audio-codec tokens
with the pure-Rust `SnacDecoder` (`forward/snac.rs`) → 24 kHz WAV. `SpeakPipeline` lives in
`sapient-generate/src/speak.rs`. Traps when editing it: the Orpheus prompt **must include the
tokenizer's BOS** (`encode_ids(.., true)` → realized `[128259, 128000, …]`) or the speech is
fluent-but-wrong; SNAC weights come from the ungated `mlx-community/snac_24khz` mirror and need
`normalize_snac_weights` (weight_norm fold + MLX→PyTorch conv-axis swap + `.layers.` strip); and the
`mlx` config omits `latent_dim`, so the decoder derives it from the conv-in weight shape. Validate
with `tests/snac_coherence.rs` (bit-close to torch) and the **speak→transcribe round-trip**.

**Kokoro-82M TTS (the real-time path).** `sapient-models/src/forward/kokoro/` is a pure-Rust
StyleTTS2 + ISTFTNet port — *non-autoregressive*, so it runs **real-time on CPU** (unlike the
autoregressive Orpheus/SNAC). New CPU kernels (BiLSTM, `torch.istft`-equivalent STFT/iSTFT,
AdaLayerNorm, AdaIN1d, NSF source, length-regulator) live in `kokoro/ops.rs`, each unit-tested vs a
reference. Weights: run `python3 scripts/convert_kokoro_to_safetensors.py --out ~/.cache/sapient-kokoro`
once (converts the upstream `.pth` pickle → safetensors; already hosted at `sai1974dev/kokoro-82m-safetensors`),
then point `SAPIENT_KOKORO_DIR` at it for the ignored coherence tests (`kokoro/stage_tests.rs`
validate every stage vs a committed PyTorch fixture; `sapient-generate/tests/kokoro_tts_e2e.rs` is the
text→audio + RTF check). G2P is the pure-Rust `misaki-rs` (no espeak). Editing trap: the NSF source
**omits** training-time noise/random-phase for determinism, so validate by **energy-envelope/spectrogram
correlation** (≈0.99), not max_err, plus the **speak→transcribe round-trip**.

**Quantized storage.** `DType::Q4_0` and `DType::Q8_0` store raw ggml block bytes. The key
invariant is that `as_bytes()` on a quantized tensor returns exactly `byte_count(numel)` bytes.
Use `as_quant_blocks()` to iterate raw blocks and `to_f32_vec()` to dequantize. Never call
`as_bytes_mut()` on a mmap-backed tensor (undefined behavior); it is only valid on heap-allocated
buffers such as the KV cache.

**GGUF q/k RoPE permutation.** llama.cpp permutes `attn_q`/`attn_k` rows for `llama`-arch GGUFs
(ggml NORM-style RoPE). SAPIENT uses HF/NEOX RoPE, so `gguf_weights::unpermute_llama_gguf_qk`
inverts it at load — without it those models emit incoherent token-salad. Do **not** apply it to
Qwen2/Gemma GGUFs (NEOX, not permuted) or to safetensors weights (already HF layout). If you add a
GGUF architecture, check whether its llama.cpp converter permutes q/k.

**Flash-Edge attention.** `kernels/attention.rs` uses an online-softmax tiled algorithm that
never materialises the full seq_q × seq_k score matrix (O(head_dim) working memory). If you
modify attention code, do not revert to the naive materialised path.

**Q8_0 KV cache.** The KV cache is allocated as Q8_0 blocks and updated in-place via
`Tensor::as_bytes_mut()`. Each decode step writes directly into the existing allocation — there
must be no `Vec` alloc inside the decode loop. Preserve this invariant when touching `kv_cache.rs`.

**`sapient serve` vs the graph runtime.** The OpenAI-compatible server in `sapient-cli/src/server.rs`
drives the `Pipeline` API directly. The `sapient-ir`/`sapient-runtime` graph execution path is
separate and is not involved in chat completions.

---

### SIMD kernel dispatch pattern

All SIMD kernels in `sapient-backends-cpu/src/kernels/quant.rs` follow a three-layer pattern:

1. **Scalar fallback** — always correct, no platform assumptions, used on x86 without AVX2 and on
   non-NEON ARM.
2. **`#[cfg(target_arch = "aarch64")]` static dispatch** — compiled unconditionally on aarch64
   (all Apple M-series, Raspberry Pi 64-bit). Uses `std::arch::aarch64::*` intrinsics. No runtime
   check needed because all aarch64 targets have NEON.
3. **`#[cfg(target_arch = "x86_64")]` + `is_x86_feature_detected!` runtime dispatch** — AVX2+FMA
   checked at runtime. The scalar fallback handles older x86_64 CPUs automatically.

When adding a new SIMD kernel:
- Put the scalar implementation first; it is the specification.
- Gate aarch64 blocks with `#[cfg(target_arch = "aarch64")]` (not `target_feature`).
- Use `is_x86_feature_detected!("avx2")` inside an `if` at runtime for x86_64.
- **Do not** add `-C target-cpu=native` or `target_feature = "+neon"` to `.cargo/config.toml`
  for the `aarch64-apple-darwin` target — it breaks `ring`'s compile-time const assertions on CI.

### Adaptive rayon chunking pattern

CPU kernels that parallelize over a large output dimension (e.g. `matmul_nt`, `gemv`) must use
`gemv_chunk()` rather than `par_iter_mut()` directly over individual output rows. The chunk size
targets **4 tasks per logical core** so that a 151 936-row `lm_head` produces ~(4 × cores) tasks
instead of 151 936 micro-tasks. This avoids rayon scheduler overhead dominating on large vocab
projections. When writing a new parallelised kernel, follow the `gemv_chunk()` pattern rather
than raw `par_chunks_mut` with a chunk size of 1.

---

## Development workflow

### 1. Create a branch

```bash
git checkout -b feat/my-feature
# or
git checkout -b fix/issue-123-download-lock
```

### 2. Make your changes

- Follow existing naming, error handling (`anyhow` in binaries, `thiserror` in libraries), and import style.
- Avoid unrelated refactors in the same PR.
- Add or update tests when behavior changes.
- Update `README.md` if user-facing CLI or install behavior changes.

### 3. Run checks locally

```bash
just fmt-check
just clippy
just test
```

Or without `just`:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -- -D warnings
cargo test --workspace
```

### 4. Push and open a PR

Push your branch and open a pull request against `main`. Fill in what changed, why, and how you tested it.

---

## Testing

### Run all tests

```bash
cargo test --workspace
```

### Run tests for one crate

```bash
cargo test -p sapient-hub
cargo test -p sapient-models
cargo test -p sapient-cli
```

### Hub integration tests (network)

Some Hub tests download public models and require network + write access to `~/.cache/huggingface/`:

```bash
cargo test -p sapient-hub --test download_parallel -- --test-threads=1
```

Run serially (`--test-threads=1`) when multiple tests touch the same cached model to avoid HF cache lock conflicts.

### FFI and TypeScript SDK tests

```bash
# sapient-ffi: unit tests, plus an ignored real-model e2e (downloads `smollm2-135m-q4`)
cargo test -p sapient-ffi
cargo test -p sapient-ffi --release -- --ignored

# TypeScript SDK (Node ≥ 18): tsc build + SSE/mock-serve suites
cd sdks/typescript && npm install && npm test
```

If you change `sapient-ffi`'s exported API, regenerate and eyeball the bindings
(they are not committed): see `docs/MOBILE.md` §4. The packaging scripts double
as integration tests — `./scripts/package-swift.sh --smoke` compiles and runs a
real macOS binary against the packaged XCFramework (CI runs this too), and
`./scripts/package-android.sh` verifies the `.so`'s uniffi exports. Both build
with the `wgpu` GPU feature by default (`--cpu-only` opts out); if the smoke
link fails on a new undefined symbol after a dependency change, add the
framework to BOTH the Package.swift template and the smoke `swiftc` line in
`package-swift.sh`. After repackaging, delete the consuming app's DerivedData —
Xcode does not re-link a changed xcframework at the same path. **Before
testing on a phone, read `docs/MOBILE.md` §5** — the safe-testing ladder for
personal hardware is a project rule, not a suggestion.

### Benchmarks

```bash
cargo bench -p sapient-backends-cpu
just bench
```

---

## Formatting and linting

CI enforces both. PRs that fail these checks will not merge.

```bash
cargo fmt --all              # auto-format
cargo fmt --all -- --check   # CI check (no writes)
cargo clippy --workspace --all-targets -- -D warnings
```

Rules of thumb:

- No `clippy` warnings — CI uses `-D warnings`.
- Prefer `debug!` / `tracing` for internal logs; keep CLI output user-friendly unless `-v` is passed.
- Do not commit secrets (`.env`, HF tokens, API keys).

---

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

[optional body]
```

Common types:

| Type | Use for |
|---|---|
| `feat` | New feature or CLI command |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Tooling, deps, CI |
| `ci` | CI workflow changes |

Examples:

```
feat: add sapient embed command for sentence vectors
fix: parse GitHub release tag_name correctly in install.sh
docs: document fast download env vars in README
ci(release): build Linux aarch64 on ARM runners
```

Keep the subject line under ~72 characters. Use the body for context, trade-offs, and breaking changes.

---

## Pull requests

### Checklist

Before requesting review, confirm:

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] User-facing changes are reflected in `README.md` (if applicable)
- [ ] No secrets or large binary blobs are included
- [ ] PR description explains **what**, **why**, and **how to test**

### Review expectations

- Maintainers may ask for tests, docs, or a smaller scope.
- Breaking changes need a clear migration note in the PR description.
- Installer changes (`install.sh`, `install.ps1`) should be tested manually on the target platform when possible.

### CI

Every PR runs:

- `cargo fmt --check`
- `cargo clippy` (warnings denied)
- `cargo test --workspace` on Linux and macOS (arm64 + x86_64)
- Windows release build of `sapient`
- Cross-compile check for `aarch64-unknown-linux-gnu`
- `cargo doc --workspace --no-deps`

---

## Common contribution areas

These are especially welcome:

| Area | Location | Notes |
|---|---|---|
| **Model architectures** | `crates/sapient-models/src/forward/` | Llama, Phi — forward engines for live chat |
| **CPU kernels** | `crates/sapient-backends-cpu/src/kernels/` | Attention (Flash-Edge), RoPE, matmul, quant dot-products |
| **SIMD dispatch** | `kernels/quant.rs` | NEON (aarch64) + AVX2 (x86_64) + scalar fallback |
| **Speculative decoding** | `crates/sapient-generate/src/speculative.rs` | Draft/target pipeline, engine reuse, cache-aware verify (`forward_all_logits_cached` + `truncate_cache`) |
| **HTTP server** | `crates/sapient-cli/src/server.rs` | OpenAI-compatible `/v1/chat/completions`; `ServedModel` (plain/speculative), LRU cache, `--speculative` |
| **Metal / GPU backend** | `crates/sapient-backends-metal/` | Apple Silicon (MLX); `MlxForwardEngine` |
| **Cross-platform GPU** | `crates/sapient-backends/wgpu/` | wgpu/WGSL (Vulkan/DX12/Metal); `WgpuForwardEngine` (`--features wgpu`, `--backend wgpu`). Kernels in `resident.rs` + `quant.rs` (Q8_0/Q4_K/Q6_K kept quantized on-device, in-shader dequant — Q4_K_M GGUFs load fully quantized) + `shaders/*.wgsl`, validated vs CPU in `tests/resident.rs` + `sapient-models/tests/wgpu_coherence.rs` (f32 + Q8_0 + mixed K-quant) |
| **Hub client** | `crates/sapient-hub/` | Downloads, caching, auth, registry |
| **CLI UX** | `crates/sapient-cli/` | Commands, terminal UI |
| **Tokenizers / chat templates** | `crates/sapient-tokenizers/` | HF tokenizer edge cases |
| **Docs & examples** | `README.md`, crate doc comments | |
| **Install scripts** | `install.sh`, `install.ps1` | Must work on fresh machines |
| **Benchmarks** | `scripts/benchmark-compare.sh`, `scripts/gen-benchmark-report.py` | Multi-engine comparison suite |

---

## Adding a CLI command

1. Add a variant to `Commands` in `crates/sapient-cli/src/main.rs` (use `clap` derive).
2. Wire it in the `match cli.command` block in `main()`.
3. Implement the handler function in `main.rs` or a submodule (`hub.rs`, `ui.rs`, …).
4. Add `--help` text via clap doc comments on the struct fields.
5. Update `README.md` and post-install help in `install.sh` / `install.ps1` if user-facing.

Example pattern:

```rust
/// Remove one cached model from this device.
#[command(name = "rm", visible_aliases = ["remove"])]
Rm {
    /// HuggingFace model ID to remove.
    model: String,
},
```

---

## Adding a model architecture

1. **Detect architecture** in `crates/sapient-hub/src/model_info.rs` — map HF `config.json` `model_type` to `ArchType`.
2. **Add forward engine** under `crates/sapient-models/src/forward/` if native inference is supported.
3. **Register weights loading** in `crates/sapient-models/src/weights.rs`.
4. **Wire Pipeline** in `crates/sapient-generate/src/pipeline.rs`.
5. **Add tests** in `crates/sapient-models/tests/` with a tiny fixture or mocked tensors where possible.

For text-only support of vision models, document limitations in CLI output rather than failing silently.

---

## Hub and download testing

Fast downloads are controlled by `LoadOptions` and env vars (see README **Fast Downloads** section):

```bash
SAPIENT_HUB_MAX_PARALLEL=4 sapient pull <model>
SAPIENT_FAST_DOWNLOAD=0 sapient pull <model>   # sequential mode
sapient -v pull <model>                        # verbose logs + file paths
```

When debugging Hub issues:

- Check `~/.cache/huggingface/hub/` for partial downloads (`.sync.part`, `.lock`).
- Use `sapient reset --stale` to clear incomplete downloads.
- Gated models require `sapient login` or `HF_TOKEN`.

---

## Releases (maintainers)

Releases are automated via GitHub Actions when a semver tag is pushed:

```bash
# 1. Bump version in root Cargo.toml (workspace.package.version)
#    and Formula/sapient.rb if needed

# 2. Commit, tag, push
git tag v0.3.2   # use actual version
git push origin main
git push origin v0.3.2
```

The [release workflow](.github/workflows/release.yml) builds binaries for:

- macOS (Apple Silicon + Intel)
- Linux (x86_64 + aarch64)
- Windows (x86_64 + ARM64)

and attaches `install.sh` / `install.ps1` to the GitHub Release.

Install URLs in docs should point to release assets:

```bash
curl -fsSL https://github.com/SkidGod4444/sapient/releases/latest/download/install.sh | sh
```

Do **not** use `raw.githubusercontent.com/.../main/install.sh` in user-facing docs — the CDN can serve stale scripts.

---

## Getting help

- **Bug reports & feature requests:** [GitHub Issues](https://github.com/SkidGod4444/sapient/issues)
- **Questions:** Open a Discussion or issue with the `question` label
- **Security issues:** Do not open public issues — contact maintainers privately

When filing a bug, include:

- OS and architecture (`uname -a` or Windows version)
- `sapient --version` output
- Full command you ran
- Expected vs actual behavior
- Relevant logs (`sapient -v …` for verbose output)

---

Thank you for helping make local LLM inference faster and more accessible.
