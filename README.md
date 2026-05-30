<div align="center">
  <h1>⚡ SAPIENT</h1>
  <p><strong>A fast, pure-Rust edge inference engine for small language models — one command to install, one line to run</strong></p>
  <p>
    <a href="https://crates.io/crates/sapient-generate"><img src="https://img.shields.io/crates/v/sapient-generate.svg" alt="Crates.io"/></a>
    <a href="https://docs.rs/sapient-generate"><img src="https://docs.rs/sapient-generate/badge.svg" alt="docs.rs"/></a>
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
| Linux (ARM64 — Pi 4/5 64-bit OS, cloud ARM) | `sapient-aarch64-unknown-linux-gnu.tar.gz` |
| Windows (x86_64) | `sapient-x86_64-pc-windows-msvc.zip` |

> **Linux:** ARM64 binaries target 64-bit glibc systems (Pi 4/5 with Raspberry Pi OS 64-bit). 32-bit `armhf`/`armv7` is not built.


---

## CLI — 30 Seconds to Running a Model

```bash
# See every model SAPIENT supports (the registry catalog)
sapient models

# Interactive chat — streaming replies, modern UI
sapient chat openhorizon/phi-2
sapient chat openhorizon/qwen2.5-0.5b --backend auto   # auto | cpu | metal

# Speculative decoding (faster generation with a draft model)
sapient chat openhorizon/qwen2.5-1.5b --speculative
sapient chat openhorizon/qwen2.5-1.5b --speculative --draft-model openhorizon/qwen2.5-0.5b

# One-shot completion (Hub models need --prompt)
sapient run openhorizon/phi-2 --prompt "Explain transformers in simple terms"

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

Add to `Cargo.toml`:

```toml
[dependencies]
sapient-generate = "0.2"
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
to see this list (and which models you've already downloaded) at any time.

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
| `openhorizon/llama-3.2-1b` | Llama | 1B | Gated — run `sapient login` |
| `openhorizon/llama-3.2-3b` | Llama | 3B | Gated — run `sapient login` |
| `openhorizon/mistral-7b` | Mistral | 7B | Gated — run `sapient login` |

All models run on the **CPU** backend everywhere; on Apple Silicon, building with
`--features mlx` enables the **Metal** GPU backend. Weights are loaded from Safetensors
(F16/BF16/F32). To request another model, open an issue — adding one means implementing
and validating its architecture in `sapient-models`.

---

## Performance (v0.3.x, CPU, Apple M-series)

SAPIENT v0.3.x ships a fully overhauled inference engine. Measured on Apple M-series, GGUF Q8_0 models:

| Model | v0.2.8 | v0.3.x | Change |
|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | ~10 tok/s | ~18.9 tok/s | **+89%** |
| Qwen2.5-1.5B Q8_0 | ~4.2 tok/s | ~10.0 tok/s | **+138%** |

Key improvements:
- **Flash-Edge attention** — online-softmax, O(head_dim) working memory, NEON `vfmaq_f32`.
- **Q8_0 KV cache** — 4× RAM reduction vs F32; zero per-step heap allocation.
- **Online quantization** — F16/BF16 safetensors weights auto-quantized to Q8_0 at load.
- **NEON GEMV kernels** — native F16 (`vcvt_f32_f16`), Q4_K nibble-unpacking + FMA, SDOT Q8_0.
- **Adaptive rayon** — `gemv_chunk()` targets 4 tasks/core; avoids micro-task overhead.
- **Hybrid Metal+CPU** — large models that don't fully fit GPU are layer-split automatically (Llama + Phi).
- **`sapient devices`** — detect CPU/GPU, estimate tok/s, recommend backend before loading a model.

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
