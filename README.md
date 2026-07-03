<div align="center">
  <h1>⚡ SAPIENT</h1>
  <p><strong>A fast, pure-Rust edge inference engine for small language models — one command to install, one line to run</strong></p>
  <p>
    <a href="https://github.com/SkidGod4444/sapient/releases"><img src="https://img.shields.io/github/v/release/SkidGod4444/sapient" alt="Release"/></a>
    <a href="https://github.com/SkidGod4444/sapient/actions"><img src="https://github.com/SkidGod4444/sapient/workflows/CI/badge.svg" alt="CI"/></a>
    <img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License"/>
    <img src="https://img.shields.io/badge/rust-1.82%2B-orange" alt="MSRV"/>
    <img src="https://img.shields.io/github/downloads/SkidGod4444/sapient/total" alt="Downloads"/>
  </p>
  <p>
    <b>macOS · Linux · Windows</b> &nbsp;|&nbsp; No Python · No Docker · No CUDA required
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

# Text-to-speech — Kokoro-82M (real-time on CPU, non-autoregressive StyleTTS2 + ISTFTNet)
# Speaks aloud through the default output device AND writes the WAV. Add --no-play to only write.
sapient speak kokoro-82m "Hello, this is sapient speaking."             # plays + writes speech.wav
sapient speak kokoro-82m "The quick brown fox." --voice af_bella -o fox.wav
sapient speak kokoro-82m "Save it, don't play it." --no-play -o out.wav  # write only
#   54 voices (af_heart, af_bella, am_michael, bf_emma, …); pure-Rust G2P, no espeak

# Text-to-speech — Orpheus-3B (Llama-3.2 → SNAC codec; richer voice, slow on CPU)
sapient speak orpheus-3b "The quick brown fox." --voice leo -o fox.wav
#   voices: tara | leah | jess | leo | dan | mia | zac | zoe

# Voice conversation — mic → speech-to-text → LLM → reply (live mic; Linux needs libasound2-dev)
# Live mic meter + streamed reply; macOS prompts for mic permission on first run.
sapient converse openhorizon/qwen2.5-1.5b --stt whisper-base
sapient converse openhorizon/qwen2.5-1.5b --speak   # speak replies aloud (Kokoro-82M; real-time)

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
resolves to the upstream Hugging Face repository it downloads from. Run `sapient models`
to see this list (and which models you've already downloaded) at any time — it's grouped
into **Text generation (chat)**, **Speech-to-text (transcribe)**, and **Text-to-speech
(speak)** sections so it's clear which command each model is for. Pointing a command at the
wrong category fails fast with a clear hint (e.g. `sapient speak whisper-small …` → "that's
a speech-to-text model, use `sapient transcribe`").

| Alias | Family | Size | Notes |
|---|---|---|---|
| `openhorizon/phi-2` | Phi | 2.7B | Default; LayerNorm + partial RoPE |
| `openhorizon/phi-1.5` | Phi | 1.3B | |
| `openhorizon/phi-1` | Phi | 1.3B | |
| `openhorizon/qwen2.5-0.5b` | Qwen2.5 | 0.5B | Smallest; great for quick tests |
| `openhorizon/qwen2.5-1.5b` | Qwen2.5 | 1.5B | |
| `openhorizon/qwen2.5-3b` | Qwen2.5 | 3B | |
| `openhorizon/smollm2-360m` | Llama | 360M | |
| `openhorizon/smollm2-1.7b` | Llama | 1.7B | |
| `openhorizon/tinyllama-1.1b` | Llama | 1.1B | |
| `openhorizon/llama-3.2-1b` | Llama | 1B | |
| `openhorizon/llama-3.2-3b` | Llama | 3B | Gated — run `sapient login` |
| `openhorizon/mistral-7b` | Mistral | 7B | Gated — run `sapient login` |
| `whisper-tiny` | Whisper | 39M | Speech-to-text — `sapient transcribe` |
| `whisper-base` | Whisper | 74M | Speech-to-text — `sapient transcribe` |
| `whisper-small` | Whisper | 244M | Speech-to-text — `sapient transcribe` |
| `kokoro-82m` | StyleTTS2 + ISTFTNet | 82M | Text-to-speech — `sapient speak` (real-time on CPU) |
| `orpheus-3b` | Llama/Orpheus | 3B | Text-to-speech — `sapient speak` (richer voice, slow) |

**Speech-to-text:** Whisper models power `sapient transcribe <model> <audio>` on all platforms.
Audio is decoded + resampled to 16 kHz in pure Rust (`symphonia`/`rubato`), turned into a log-mel
spectrogram, and run through a native Whisper encoder/decoder. Auto-detects the spoken language;
`--language <code>` forces it and `--translate` outputs English. Runs on CPU by default, or on the
cross-platform GPU with `--backend wgpu` (build `--features wgpu`).

All text models run on the **CPU** backend everywhere; on Apple Silicon, building with
`--features mlx` enables the **Metal** GPU backend. Weights are loaded from Safetensors
(F16/BF16/F32). To request another model, open an issue — adding one means implementing
and validating its architecture in `sapient-models`.

---

## Performance (v0.3.5, Apple M4, 16 GB)

The **`MlxForwardEngine`** runs the whole Llama/Qwen forward pass as one MLX lazy
graph — every activation stays on the GPU, `eval()` runs once per token. Measured on
GGUF Q4 models (decode-only tok/s; steady-state TTFT):

| Model | CPU | **Metal** | Speedup | Decode vs Ollama / mlx-lm | TTFT vs Ollama / mlx-lm |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4 | 20 | **187 tok/s** | **9.4×** | beats 154 / 0.75× 249 | **21 ms** — best of all |
| Qwen2.5-1.5B Q4 | 11 | **74 tok/s** | **6.7×** | 0.95× 78 / 0.79× 94 | 70 ms vs 64 / 264 |

SAPIENT Metal **beats Ollama on 0.5B decode and has the lowest time-to-first-token of
any engine on 0.5B**, within **1.3–1.5× of mlx-lm** — from a single daemon-free 22 MB
binary. Full methodology, charts, and the remaining peak-RAM gap are in
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

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
- **NEON GEMV kernels** — native F16 (`vcvt_f32_f16`), Q4_K nibble-unpacking + FMA, SDOT Q8_0.
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
14.7 → 3.6 GB, decode **13.2 tok/s — 1.13× the NEON-optimized M4 CPU path**. On a
16 GB machine the old f32 path ran out of memory at 1.5B (empty replies); the
quantized-resident path answers correctly. F16/BF16 safetensors linears are
online-quantized to Q8_0 on upload, same as the CPU engine.

Current scope: Llama-family models, KV cache is f32, one token per submission. An
f16/quantized KV cache, kernel fusion, and batched prefill are tracked in
[ROADMAP Phase 3b](docs/ROADMAP.md).

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
| `POST /v1/chat/completions` | OpenAI-compatible chat completion |
| `POST /v1/completions` | Raw text completion |
| `POST /v1/audio/transcriptions` | OpenAI-compatible speech-to-text (multipart audio upload) |
| `GET /v1/health` | Liveness check |

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

For models that require access approval (Llama 3.2, Mistral):

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
sapient-generate          ← Pipeline API — from_pretrained, generate, chat, embed, stream
                             SpeculativePipeline — draft+target speculative decoding
├── sapient-hub           ← HuggingFace Hub client — parallel downloads, auth, cache, registry
├── sapient-tokenizers    ← All HF tokenizer types + Jinja2 chat templates
├── sapient-models        ← Forward engines: Phi (Phi-1/1.5/2) and Llama (Llama/Qwen2.5/SmolLM2/TinyLlama/Mistral)
│
├── sapient-runtime       ← InferenceSession — graph execution + telemetry
│   ├── sapient-ir        ← Computation graph IR + optimization passes
│   └── sapient-io        ← Safetensors, GGUF (Q4/Q8/Q5 dequant), ONNX loaders
│
├── sapient-backends-cpu    ← CPU kernels: Flash-Edge attention, RoPE, RMSNorm/LayerNorm,
│                             MatMul, NEON Q4_0/Q8_0/Q4_K/F16 GEMV, AVX2 Q8_0
└── sapient-backends-metal  ← Apple Silicon Metal/MLX backend (built with `--features mlx`)
```

> Generation runs through two validated forward engines — **Phi** and **Llama** (the
> latter also serves Qwen2.5, SmolLM2, TinyLlama and Mistral). `sapient serve` drives
> these engines directly via the `Pipeline` API (OpenAI-compatible). Additional architecture
> builders (Gemma, GPT-2, BERT, Mixtral) exist in the IR layer but are not yet wired into
> the chat/generation path.

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
- New model architecture builders (`crates/sapient-models/src/architectures/`)
- Apple Metal kernels (`crates/sapient-backends/metal/`)
- Quantization kernels (INT4, INT8 fused)
