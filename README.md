<div align="center">
  <h1>⚡ SAPIENT</h1>
  <p><strong>A fast, pure-Rust edge inference engine for language, vision, and speech models — one command to install, one line to run</strong></p>
  <p>
    <a href="https://github.com/SkidGod4444/sapient/releases"><img src="https://img.shields.io/github/v/release/SkidGod4444/sapient" alt="Release"/></a>
    <a href="https://github.com/SkidGod4444/sapient/actions"><img src="https://github.com/SkidGod4444/sapient/workflows/CI/badge.svg" alt="CI"/></a>
    <img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License"/>
    <img src="https://img.shields.io/badge/rust-1.82%2B-orange" alt="MSRV"/>
    <img src="https://img.shields.io/github/downloads/SkidGod4444/sapient/total" alt="Downloads"/>
  </p>
  <p>
    <b>macOS · Linux · Windows</b> &nbsp;|&nbsp; No Python · No Docker · No CUDA required &nbsp;|&nbsp; <a href="https://sapient.openhorizon.so">sapient.openhorizon.so</a>
  </p>
</div>

---

## Install

### macOS & Linux (one command)

```bash
curl -fsSL https://github.com/SkidGod4444/sapient/releases/latest/download/install.sh | sh
```

> **Piped installs** go to `~/.local/bin`. If `sapient` is not found afterward, run:
> `export PATH="$HOME/.local/bin:$PATH"` and restart your terminal.

### Windows (PowerShell)

```powershell
irm https://github.com/SkidGod4444/sapient/releases/latest/download/install.ps1 | iex
```

> **Automatic GPU detection.** On x86_64 Linux/Windows the installer detects whether you
> have a graphics card and pulls the **GPU build** (`-gpu`, wgpu — Intel/AMD/Nvidia) when
> one is present, or the CPU build otherwise. Force a choice with `SAPIENT_VARIANT=cpu`
> (or `gpu`) on the `sh` install, or `$env:SAPIENT_VARIANT="cpu"` on Windows. Later,
> `sapient update` will ask which build you want whenever your machine has a GPU
> (or pass `--gpu` / `--cpu` / `--metal`).

### Homebrew (macOS)

```bash
brew install skidgod4444/tap/sapient
```

### Direct Download

Grab a pre-built binary for your platform from the [**latest release**](https://github.com/SkidGod4444/sapient/releases/latest):

| Platform | Binary |
|---|---|
| macOS (Apple Silicon) | `sapient-aarch64-apple-darwin.tar.gz` |
| macOS (Apple Silicon, Metal GPU) | `sapient-aarch64-apple-darwin-metal.tar.gz` |
| macOS (Intel) | `sapient-x86_64-apple-darwin.tar.gz` |
| Linux (x86_64) | `sapient-x86_64-unknown-linux-gnu.tar.gz` |
| Linux (x86_64, GPU — Intel/AMD/Nvidia via Vulkan) | `sapient-x86_64-unknown-linux-gnu-gpu.tar.gz` |
| Linux (ARM64 — Pi 4/5 64-bit OS, cloud ARM) | `sapient-aarch64-unknown-linux-gnu.tar.gz` |
| Windows (x86_64) | `sapient-x86_64-pc-windows-msvc.zip` |
| Windows (x86_64, GPU — Intel/AMD/Nvidia via DX12) | `sapient-x86_64-pc-windows-msvc-gpu.zip` |

> **Linux:** ARM64 binaries target 64-bit glibc systems (Pi 4/5 with Raspberry Pi OS 64-bit). 32-bit `armhf`/`armv7` is not built.
> **Raspberry Pi:** see [docs/PI.md](docs/PI.md) — per-RAM model guidance, the low-RAM quant override (`SAPIENT_GGUF_QUANT=Q4_K_S`), and the thermal governor that keeps sustained decode from collapsing on passive cooling (`SAPIENT_THERMAL=off` to disable).
>
> **`-gpu` binaries** add the cross-platform wgpu GPU backend (`--backend wgpu`); use them on any Intel/AMD/Nvidia GPU. On Linux they need the Vulkan loader (`libvulkan1`) and your GPU driver installed. The plain binaries are CPU-only. On Apple Silicon use the `-metal` binary instead.


---

## CLI — 30 Seconds to Running a Model

```bash
# See every model SAPIENT supports (the registry catalog)
sapient models

# Interactive chat — streaming replies, modern UI, paste-safe line editing
# Replies render as formatted Markdown live (headings, lists, **bold**, syntax-
# highlighted code blocks). Use --raw for plain text; auto-disabled when piped.
sapient chat openhorizon/phi-2
sapient chat openhorizon/phi-2 --raw                    # plain Markdown text
sapient chat openhorizon/qwen2.5-0.5b --backend auto   # auto | cpu | metal | wgpu
sapient chat openhorizon/qwen2.5-0.5b -p "Tell me a joke"  # one-shot: single turn, reply to stdout (scriptable)
sapient chat openhorizon/phi-4-mini -n 4096 -p "…"     # --max-tokens: per-reply cap (default 2048; capped replies print a notice)

# Speculative decoding (faster generation with a draft model)
sapient chat openhorizon/qwen2.5-1.5b --speculative
sapient chat openhorizon/qwen2.5-1.5b --speculative --draft-model openhorizon/qwen2.5-0.5b

# One-shot completion (Hub models need --prompt)
sapient run openhorizon/phi-2 --prompt "Explain transformers in simple terms"

# Speech-to-text — transcribe audio with Whisper (WAV/FLAC/MP3/OGG/M4A)
sapient transcribe whisper-base recording.wav             # streams text as it decodes
sapient transcribe whisper-small talk.mp3 --language en   # skip auto-detect
sapient transcribe whisper-tiny clip.flac --translate     # → English
sapient transcribe whisper-base long.wav --timestamps     # long-audio re-seek
sapient transcribe whisper-base clip.wav --beam-size 5    # beam search

# Text-to-speech — Kokoro-82M (~2× real-time on CPU: RTF 0.48 on M4; StyleTTS2 + ISTFTNet)
# Speaks aloud through the default output device AND writes the WAV. Add --no-play to only write.
sapient speak kokoro-82m "Hello, this is sapient speaking."             # plays + writes speech.wav
sapient speak kokoro-82m "The quick brown fox." --voice af_bella -o fox.wav
sapient speak kokoro-82m "Save it, don't play it." --no-play -o out.wav  # write only
#   54 voices (af_heart, af_bella, am_michael, bf_emma, …); pure-Rust G2P, no espeak

# Text-to-speech — Orpheus-3B (Llama-3.2 → SNAC codec; richer voice, slow on CPU)
sapient speak orpheus-3b "The quick brown fox." --voice leo -o fox.wav
#   voices: tara | leah | jess | leo | dan | mia | zac | zoe

# Vision — ask questions about an image, fully on-device
sapient see photo.jpg -p "What's in this picture?"          # SmolVLM-256M (default)
sapient see chart.png -p "Summarize this chart." --model gemma-3-4b
sapient see xray.png -p "Describe findings." --model medgemma-4b   # medical (gated: sapient login)

# Voice conversation — a STREAMING loop: speech is transcribed while you're
# still talking, the reply starts speaking after its first clause, and you can
# interrupt it mid-sentence (barge-in). ~2.4 s perceived reply latency on an
# M-series CPU; per-turn latency breakdown printed live.
# (Live mic; Linux needs libasound2-dev; macOS prompts for mic permission.)
sapient converse qwen2.5-1.5b --stt whisper-base
sapient converse qwen2.5-1.5b --speak   # speak replies aloud (Kokoro-82M)

# Live resource monitor — CPU cores, RAM, and disk used by SAPIENT
sapient stats        # (aliases: top, monitor) — Ctrl-C to exit

# Download a model to local cache
sapient pull openhorizon/phi-2

# List / remove downloaded models
sapient list
sapient rm openhorizon/phi-2   # remove one model
sapient reset                  # clear entire cache

# OpenAI-compatible HTTP server (lazy model load on first request)
sapient serve --port 8080
sapient serve --port 8080 --speculative

# Update sapient to the latest release
sapient update

# Gated models (Llama, Mistral) — set token first
sapient login

# Show config/architecture info for a model
sapient info openhorizon/phi-2

# Detect CPU/GPU, estimate tok/s, get a backend recommendation
sapient devices
sapient backend-info

# Verbose mode — show internal logs, file paths, and generation stats
sapient -v chat openhorizon/phi-2
```

Inside chat: type your message and press Enter. Use `/help` for commands, `/clear` to
reset the conversation, and `/exit` to quit.

---

## Fast Downloads

Sapient uses parallel HTTP range requests and concurrent shard downloads (via the Rust `hf-hub` client). Fast downloads are **on by default**.

| Variable | Default | Description |
|---|---|---|
| `SAPIENT_HUB_MAX_PARALLEL` | `min(CPU cores, 8)` | Concurrent download workers |
| `SAPIENT_HUB_CHUNK_SIZE` | `10000000` (10 MiB) | HTTP range chunk size |
| `SAPIENT_FAST_DOWNLOAD` | `1` | Set to `0` to disable parallel mode |

```bash
# Example: limit workers on a slow connection
SAPIENT_HUB_MAX_PARALLEL=2 sapient pull <model>
```

> **Note:** Python-only accelerators like `hf_xet` are not available in the Rust CLI. Sapient achieves similar gains through parallel range requests and concurrent multi-shard downloads.

---

## Rust API

SAPIENT is **not published to crates.io** — depend on it via git:

```toml
[dependencies]
sapient-generate = { git = "https://github.com/SkidGod4444/sapient" }
tokio = { version = "1", features = ["full"] }
```

```rust
use sapient_generate::Pipeline;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Downloads, caches, and runs — zero config needed
    let p = Pipeline::from_pretrained("openhorizon/phi-2").await?;
    println!("{}", p.generate("The key to good software is").await?);
    Ok(())
}
```

### Chat (Instruct Models)

```rust
use sapient_tokenizers::ChatMessage;

let p = Pipeline::from_pretrained("openhorizon/phi-2").await?;
let reply = p.chat(&[
    ChatMessage::system("You are a helpful coding assistant."),
    ChatMessage::user("Write a Rust function to reverse a string."),
]).await?;
println!("{reply}");
```

### Streaming

```rust
use futures::StreamExt;

let mut stream = p.generate_stream("Once upon a time").await;
while let Some(token) = stream.next().await {
    print!("{token}");
}
```

### Custom Sampling

```rust
use sapient_generate::{GenerationConfig, SamplingStrategy};

let cfg = GenerationConfig {
    max_new_tokens: 200,
    strategy: SamplingStrategy::TopP { p: 0.95, temperature: 0.8 },
    stop_sequences: vec!["<|end|>".into()],
    ..Default::default()
};
let text = p.generate_with_config("Write a haiku about Rust", &cfg).await?;
```

---

## Supported Models

SAPIENT ships a **curated registry** — every model below is one whose architecture is
implemented and verified in the native generation engine. Each `openhorizon/*` alias
resolves to the upstream Hugging Face repository it downloads from (the short alias
works too, e.g. `sapient chat qwen2.5-0.5b-q4`). Run `sapient models` to see this list
(and which models you've already downloaded) at any time — it's grouped into
**Text generation (chat)**, **Speech-to-text (transcribe)**, **Text-to-speech (speak)**,
and **Vision-language (see)** sections so it's clear which command each model is for.
Pointing a command at the wrong category fails fast with a clear hint (e.g.
`sapient speak whisper-small …` → "that's a speech-to-text model, use `sapient transcribe`").

### Text generation — `sapient chat`

| Alias | Family | Size | Notes |
|---|---|---|---|
| `openhorizon/phi-2` | Phi | 2.7B | Default; LayerNorm + partial RoPE |
| `openhorizon/phi-1.5` / `phi-1` | Phi | 1.3B | |
| `openhorizon/phi-2-q4` | Phi | 2.7B GGUF | |
| `openhorizon/phi-4-mini` | Phi | 3.8B Q4_K_M | |
| `openhorizon/qwen2.5-0.5b` | Qwen2.5 | 0.5B | Smallest chat model; great for quick tests |
| `openhorizon/qwen2.5-1.5b` / `-3b` | Qwen2.5 | 1.5B / 3B | |
| `openhorizon/qwen2.5-0.5b-q4` / `-1.5b-q4` | Qwen2.5 | 0.5B / 1.5B Q4_K_M | |
| `openhorizon/qwen2.5-coder-0.5b` / `-1.5b` | Qwen2.5 | 0.5B / 1.5B Q4_K_M | Code-tuned |
| `openhorizon/smollm2-135m-q4` | Llama | 135M Q4_K_M | Tiniest model in the catalog |
| `openhorizon/smollm2-360m` (+ `-q4`) | Llama | 360M | |
| `openhorizon/smollm2-1.7b` (+ `-q4`) | Llama | 1.7B | |
| `openhorizon/tinyllama-1.1b` | Llama | 1.1B | |
| `openhorizon/llama-3.2-1b` (+ `-q4`) | Llama | 1B | |
| `openhorizon/llama-3.2-3b` (+ `-q4`) | Llama | 3B | |
| `openhorizon/llama-3.1-8b-q4` | Llama | 8B Q4_K_M | |
| `openhorizon/deepseek-r1-8b` | Llama | 8B Q4_K_M | DeepSeek-R1-Distill |
| `openhorizon/mistral-7b` | Mistral | 7B | 13.5 GB safetensors — prefer `mistral-7b-q4` |
| `openhorizon/mistral-7b-q4` | Mistral | 7B Q4_K_M | |
| `openhorizon/gemma-3-1b` | Gemma3 | 1B | Gemma3 engine (QK-norm, sliding/global attention) |
| `openhorizon/gemma-3-4b` | Gemma3 (multimodal) | 4B | Also serves `sapient see` |
| `openhorizon/medgemma-4b` | Gemma3 (medical) | 4B | Medical Q&A + image analysis (gated — `sapient login`) |
| `openhorizon/mixtral-8x7b-q4` | Mixtral (sparse MoE) | 47B-A13B | 8 experts top-2; Q4_K_M ≈ 26 GB — needs a 32 GB+ device; CPU-only |
| `openhorizon/glm-4.5-air-q4` | GLM-4.5 (sigmoid-gate MoE) | 106B-A12B | 128 experts top-8 + shared expert; Q4_K_M ≈ 63 GB (2-shard split) — needs a 96 GB+ device; CPU-only |

The `-q4` aliases download a single quantized GGUF file — RAM ≈ file size, no F32
expansion — and are the right pick for edge devices.

### Speech-to-text — `sapient transcribe`

| Alias | Family | Size |
|---|---|---|
| `openhorizon/whisper-tiny` | Whisper | 39M |
| `openhorizon/whisper-base` | Whisper | 74M |
| `openhorizon/whisper-small` | Whisper | 244M |

Audio is decoded + resampled to 16 kHz in pure Rust (`symphonia`/`rubato`), turned into a
log-mel spectrogram, and run through a native Whisper encoder/decoder. Auto-detects the
spoken language; `--language <code>` forces it and `--translate` outputs English. On a
`-gpu` (wgpu) build, `--backend auto` picks the GPU automatically when an adapter exists
(CPU fallback; on Apple Silicon the CPU/Metal path keeps precedence).

### Text-to-speech — `sapient speak`

| Alias | Family | Size | Notes |
|---|---|---|---|
| `openhorizon/kokoro-82m` | StyleTTS2 + ISTFTNet | 82M | ~2× real-time on CPU; 54 voices; default for `converse --speak` |
| `openhorizon/orpheus-3b` | Llama/Orpheus → SNAC | 3B | Richer voice, slow on CPU; 8 voices |

### Vision-language — `sapient see`

| Alias | Family | Size | Notes |
|---|---|---|---|
| `openhorizon/smolvlm-256m` | SmolVLM (SigLIP + SmolLM2) | 256M | Default; ~3 s to first token on M4 |
| `openhorizon/gemma-3-4b` | Gemma3 multimodal | 4B | |
| `openhorizon/medgemma-4b` | Gemma3 medical | 4B | X-ray / dermatology / pathology (gated) |

Every model runs on the **CPU** backend on all platforms, loading safetensors
(F16/BF16/F32, auto-quantized to Q8_0 at load) or GGUF (Q4/Q5/Q6/Q8, mmap-able).
The `-metal` binary (Apple Silicon, MLX) and `-gpu` binaries (wgpu — Intel/AMD/Nvidia)
run chat models — and, on wgpu, Whisper — on the GPU; `--backend auto` picks the
compiled accelerator. To request another model, open an issue — adding one means
implementing and validating its architecture in `sapient-models`.

---

## Performance (Apple M4 16 GB · Raspberry Pi 5 · Jetson AGX Thor)

**Head-to-head vs llama.cpp and Ollama** (same GGUF file, same machine, same
session — re-measured on the **v0.5.3** binaries, 2026-07-09; method + full
tables in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| Apple M4 (Metal/GPU), decode tok/s | SAPIENT `-metal` | llama.cpp (Metal) | Ollama |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M | **90.6** | 111.3 | 60.4† |
| Qwen2.5-1.5B Q4_K_M | **82.2** | 88.4 | 86.3 |

† Ollama's default `llama3.2:1b` tag ships Q8_0, not Q4_K_M.

SAPIENT-Metal sits **within 10–20% of llama.cpp-Metal and beats Ollama by 1.5×
on the 1B** — with the **lowest TTFT of the three** (52–63 ms warm vs Ollama's
~130–150 ms) — from a single daemon-free ~22 MB binary. The
**`MlxForwardEngine`** runs the whole forward pass as one MLX lazy graph: every
activation stays on the GPU, one `eval()` per token.

**The CPU engine is within 1.3–1.6× of llama.cpp** (was 1.8–3.8× at v0.5.0) after
the v0.5.1 kernel ladder — multi-row GEMV, `Q4_K_R4` load-time weight repacking,
W6A8 SDOT Q6_K, and i8mm SMMLA prefill kernels, every rung bit-identity-gated:

| CPU decode, tok/s | SAPIENT | llama.cpp |
|---|---|---|
| Apple M4 — Llama-3.2-1B Q4_K_M | 56.7 | **83.1** |
| Apple M4 — Qwen2.5-1.5B Q4_K_M | 40.6 | **66.5** |
| Raspberry Pi 5 (16 GB) — Llama-3.2-1B Q4_K_M (v0.5.1 run) | 11.6 | **14.7** |

A Pi 5 went **1.3 → 11.6 tok/s (8.9×)** on this model across v0.5.0 + v0.5.1 — 1B-class
chat on a Pi is genuinely interactive. CPU prefill is 1.5× (M4) to 2× (Jetson Thor)
faster than v0.5.0 on long prompts. (These ratios hold on NEON-class CPUs — M-series,
Pi. SVE-class server ARM (Grace/Thor Neoverse) still trails llama.cpp's KleidiAI
microkernels ~3× on dense decode; closing that is its own roadmap project.)

### Sparse MoE — big models on small devices (v0.5.3)

A **47B Mixtral-8x7B** and a **106B GLM-4.5-Air** run fully on-device in pure Rust,
**zero CUDA**, on a Jetson AGX Thor (14× Neoverse, CPU path):

| Model (Q4_K_M GGUF) | Decode | Prefill | Peak RSS |
|---|---|---|---|
| Mixtral-8x7B (47B-A13B, ≈ 26 GB) | 5.5 tok/s | ~6–9 tok/s | 25.6 GB (mmap ≈ file size) |
| GLM-4.5-Air (106B-A12B, ≈ 63 GB split GGUF) | 3.2 tok/s | 3.9 tok/s | 72 GB — fits a 96 GB device |

**Zero quality loss:** greedy output is token-identical to llama.cpp on the same file
(~28 tokens, then benign f32-order drift) — and SAPIENT loads the classic per-expert
Mixtral GGUFs that current llama.cpp rejects. MoE models mmap by default (RSS ≈ file
size), and quant types SAPIENT can't keep as packed blocks (e.g. Q5_0 in "dynamic"
quants) re-quantize to Q8_0 at load instead of exploding to F32 (GLM peak RSS
118 → 72 GB). Full decomposition in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Serving head-to-head** (`sapient serve` vs Ollama vs vLLM, Apple M4 / Metal): SAPIENT
beats Ollama on TTFT (**4.2×**, 14 ms vs 59 ms), decode (**1.25×**), concurrent
throughput (**1.31×**, 1.9× lower p95), and model switch-back (**6×**). vLLM is a
datacenter-GPU engine and doesn't run on this edge box. Charts + method:
**[docs/SERVING_BENCHMARKS.md](docs/SERVING_BENCHMARKS.md)**.

![Decode throughput](docs/assets/decode_throughput.png)
![Time to first token](docs/assets/ttft.png)

Key improvements:
- **`MlxForwardEngine`** — all activations stay as `mlx_rs::Array`; one `eval()` per
  decode step; MLX fused SDPA for attention. Auto-selected for Llama/Qwen GGUF models on `--backend metal`.
- **`WgpuForwardEngine`** — cross-platform GPU (Vulkan/DX12/Metal) for Intel/AMD/Nvidia/Apple;
  GPU-resident weights + KV cache, on-device decode. `--backend wgpu` (build `--features wgpu`).
- **Engine reuse** — the pipeline holds the loaded engine in an `Arc<Mutex<…>>`; the
  streaming path no longer rebuilds it per call (TTFT dropped 30–44×, 1.5B: 3 s → 70 ms).
- **Flash-Edge attention** (CPU) — online-softmax, O(head_dim) memory, NEON `vfmaq_f32`.
- **Q8_0 KV cache** — 4× RAM reduction vs F32; zero per-step heap allocation.
- **Online quantization** — F16/BF16 safetensors weights auto-quantized to Q8_0 at load.
- **NEON int8 kernel ladder (v0.5.1)** — every K-quant matmul runs int8 `sdot`/`smmla`:
  `Q4_K_R4` load-time row-interleaved repacking (one contiguous weight stream per task),
  W4A8/W6A8 SDOT dot products, and i8mm SMMLA prefill kernels (two prompt tokens per
  weight pass on ARMv8.6 cores). Each kernel bit-identity-gated against a scalar oracle.
- **`sapient devices`** — detect CPU/GPU, estimate tok/s, recommend backend before loading a model.

### Cross-platform GPU (Intel / AMD / Nvidia)

Metal acceleration is Apple-only. To reach Intel Arc, AMD Radeon, and Nvidia GPUs on
Linux and Windows, SAPIENT has a portable GPU backend (`crates/sapient-backends/wgpu`)
built on [`wgpu`](https://wgpu.rs) — the **same WGSL compute shaders** run on Vulkan,
DX12, and Metal. The full forward pass (`WgpuForwardEngine`) is wired in: weights are
uploaded to the GPU once, the KV cache lives on-device, and each decode step runs
entirely on the GPU (RMSNorm, GEMV, RoPE, causal GQA FlashDecoding attention, SwiGLU)
with only the logits read back. Logits are validated to match the CPU engine.

```bash
# Build with the wgpu feature, then select the backend (Llama/Qwen/Mistral):
cargo build --release -p sapient-cli --features wgpu
./target/release/sapient chat openhorizon/qwen2.5-0.5b --backend wgpu
```

**Quantized weights stay quantized on the GPU** (Q8_0, Q4_K, Q6_K): raw ggml blocks
upload without f32 expansion and are dequantized inside the shader — a Q4_K_M GGUF
loads **fully quantized**, so VRAM ≈ the GGUF file size. Measured on Apple M4
(16 GB, wgpu→Metal): SmolLM2-360M Q8_0 weights resident 1.6 GiB → **388 MiB** with
greedy output token-identical to the f32 path; Qwen2.5-1.5B Q4_K_M weights resident
6.8 GiB → **1.06 GiB** (198/198 matrices quantized), peak process footprint
14.7 → 3.6 GB, decode 14.3 tok/s. On a 16 GB machine the old f32 path ran out of
memory at 1.5B (empty replies); the quantized-resident path answers correctly.
(On Apple Silicon the `-metal` MLX build is the fast path — wgpu's value is
Intel/AMD/Nvidia, where it's the only way to run these models quantized on the
GPU, and small-VRAM cards, where VRAM ≈ file size is the difference between
loading and not.) F16/BF16 safetensors linears are
online-quantized to Q8_0 on upload, same as the CPU engine.

The KV cache is **f16** (packed halves, f32 accumulation — works on any adapter,
no shader-f16 feature needed), which doubles the on-GPU context window to 8192 at
the same memory cost as the old f32@4096 cache. Each decoded token's kernels are
batched into **one queue submission** (was ~450), worth +27% decode on a 360M
model and +4% on 1.5B (M4/Metal).

Prompts prefill in 128-token batched chunks (1.5× faster time-to-first-token on
long prompts); decode runs one token at a time.

Current scope: Llama-family models. Tiled-GEMM prefill and buffer reuse are
tracked in [ROADMAP Phase 3b](docs/ROADMAP.md).

**Benchmark it on your machine.** `scripts/bench_wgpu.py` times TTFT and decode tok/s
across backends so you can see what your GPU buys you — works on any OS/vendor, needs
only Python's standard library:

```bash
python3 scripts/bench_wgpu.py                       # cpu vs wgpu (vs metal on a Mac)
python3 scripts/bench_wgpu.py --model openhorizon/qwen2.5-1.5b --tokens 128
python3 scripts/bench_wgpu.py --chart bench.png     # + a bar chart (needs matplotlib)
```

---

## HTTP Server — OpenAI-compatible

`sapient serve` starts an **OpenAI-compatible HTTP server** backed by the native chat
pipeline. No model is loaded at startup — the first API request triggers model download
and load automatically (Ollama-style lazy loading).

```bash
# Start the server (lazy model load on first request)
sapient serve --port 8080

# With speculative decoding enabled
sapient serve --port 8080 --speculative
```

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List loaded model(s) |
| `POST /v1/chat/completions` | OpenAI-compatible chat — plain text or **image parts** (base64 data URIs) |
| `POST /v1/completions` | Raw text completion |
| `POST /v1/audio/transcriptions` | OpenAI-compatible speech-to-text (multipart audio upload) |
| `GET /v1/health` | Liveness check |

`/v1/chat/completions` accepts OpenAI-style image content parts as **base64 data URIs**,
routed through the same vision engine as `sapient see` (smolvlm-256m, gemma-3-4b,
medgemma-4b). Remote image URLs are refused by design — your inference box never makes
surprise egress. The server keeps the N most-recently-used models resident (multi-model
LRU cache, `--max-models` / `--cache-gb`), so switching back to a recent model is
instant instead of a cold reload.

Example with `curl`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openhorizon/qwen2.5-0.5b-q4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The server is compatible with any OpenAI-client SDK or tool (LangChain, LlamaIndex, etc.)
by pointing the base URL at `http://localhost:8080/v1`.

---

## HuggingFace Token (Gated Models)

For models that require access approval (currently only `medgemma-4b` — accept
Google's Health AI Developer Foundations terms on Hugging Face first):

```bash
# Set via environment variable
export HF_TOKEN=hf_your_token_here

# Or set once via CLI
sapient login
```

---

## Architecture

Built in Rust for maximum performance, with zero dependencies on Python, ONNX Runtime, or CUDA.

```
sapient-cli               ← the `sapient` binary (chat, see, transcribe, speak, converse, serve, stats, …)
sapient-generate          ← Pipeline API — from_pretrained, generate, chat, stream
                             + SpeculativePipeline (draft+target speculative decoding),
                             TranscribePipeline (STT), SpeakPipeline (TTS),
                             VlmPipeline (vision), ConversePipeline (voice loop)
├── sapient-hub           ← HuggingFace Hub client — parallel downloads, auth, cache, curated registry
├── sapient-tokenizers    ← All HF tokenizer types + Jinja2 chat templates + Whisper tokenizer
├── sapient-models        ← Forward engines: Phi, Llama (Llama/Qwen2.5/SmolLM2/TinyLlama/Mistral
│                            + Mixtral/GLM sparse MoE), Gemma3, Whisper, SigLIP (vision),
│                            Kokoro + SNAC (TTS); MLX and wgpu GPU engines
├── sapient-audio         ← Audio decode/resample (symphonia+rubato), log-mel front-end, mic/speaker I/O
├── sapient-io            ← Safetensors (mmap), GGUF (Q4/Q5/Q6/Q8 quant), ONNX loaders
│
├── sapient-backends-cpu    ← CPU kernels: Flash-Edge attention, RoPE, RMSNorm/LayerNorm,
│                             NEON/AVX2 quantized GEMV (SDOT/SMMLA int8 ladder), thermal governor
├── sapient-backends-metal  ← Apple Silicon Metal/MLX backend (`--features mlx`)
└── sapient-backends-wgpu   ← Portable GPU backend — WGSL over Vulkan/DX12/Metal (`--features wgpu`)
```

> Generation runs through three validated text engines — **Phi**, **Llama** (which also
> serves Qwen2.5, SmolLM2, TinyLlama, Mistral, and the Mixtral/GLM sparse-MoE models),
> and **Gemma3** — plus dedicated engines for Whisper STT, Kokoro/SNAC TTS, and the
> SigLIP vision tower. `sapient serve` drives them directly via the `Pipeline` API
> (OpenAI-compatible). The IR-layer architecture builders (GPT-2, BERT, …) are graph
> scaffolding, not part of the live inference path.

---

## Build from Source

```bash
git clone https://github.com/SkidGod4444/sapient
cd sapient
cargo build --workspace --release

# Apple Silicon MLX GPU build:
# requires Xcode's Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)
cargo build -p sapient-cli --release --features mlx

# Binary will be at:
./target/release/sapient
```

---

## License

SAPIENT is licensed under the **[GNU General Public License v3.0](LICENSE)**.

You are free to use, study, share, and improve this software. Any modified version you distribute must also be open-source under the GPL-3.0.

---

## Contributing

Issues and PRs are very welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where contributions are especially appreciated:
- New forward engines (`crates/sapient-models/src/forward/`)
- WGSL GPU kernels (`crates/sapient-backends/wgpu/`)
- Quantization kernels (`crates/sapient-backends/cpu/src/kernels/`)
- Intel Arc / AMD GPU benchmark datapoints (`scripts/bench_gpu_7_6.sh`)
