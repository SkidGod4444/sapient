<div align="center">
  <h1>⚡ SAPIENT</h1>
  <p><strong>Run any HuggingFace LLM or SLM locally — one command to install, one line to run</strong></p>
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
| macOS (Intel) | `sapient-x86_64-apple-darwin.tar.gz` |
| Linux (x86_64) | `sapient-x86_64-unknown-linux-gnu.tar.gz` |
| Linux (ARM64 — Pi 4/5 64-bit OS, cloud ARM) | `sapient-aarch64-unknown-linux-gnu.tar.gz` |
| Windows (x86_64) | `sapient-x86_64-pc-windows-msvc.zip` |

> **Linux:** ARM64 binaries target 64-bit glibc systems (Pi 4/5 with Raspberry Pi OS 64-bit). 32-bit `armhf`/`armv7` is not built.


---

## CLI — 30 Seconds to Running a Model

```bash
# Interactive chat — streaming replies, clean UI
sapient chat <model>
sapient chat <model> --backend auto   # auto | cpu | metal

# One-shot completion (Hub models need --prompt)
sapient run <model> --prompt "Explain transformers in simple terms"
sapient run <model> --prompt "Explain transformers" --backend cpu

# Download a model to local cache
sapient pull <model>

# List / remove cached models
sapient list
sapient rm <model>          # remove one model
sapient reset               # clear entire cache

# Update sapient to the latest release
sapient update

# Gated models (Llama, Gemma) — set token first
sapient login

# Start an HTTP inference server (ONNX/GGUF files)
sapient serve <model> --port 8080

# Show info about a model
sapient info <model>
sapient backend-info

# Verbose mode — show internal logs and file paths
sapient -v pull <model>
```

Use `/exit` or `/quit` to leave chat. Type `/help` for chat commands.

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
sapient-generate = "0.1"
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

Sapient's native generation path is currently focused exclusively on optimizing the **openhorizon/phi-2** model for edge devices. Our built-in registry resolves `openhorizon/phi-2` directly to the Hugging Face repository, while providing $O(1)$ KV-caching optimizations.

| Registry Alias | Format | Backend |
|---|---|---|
| **`openhorizon/phi-2`** | Safetensors | CPU, MLX on Apple Silicon when built with `--features mlx` |

*Note: Other model builders exist in the IR layer but are not officially supported or validated in the current registry focus.*

---

## OpenAI-Compatible API

```bash
sapient serve openhorizon/phi-2 --port 8080
```

```bash
# Works with any OpenAI SDK — just change the base URL
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openhorizon/phi-2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Compatible with the **OpenAI Python SDK**, **LangChain**, **LlamaIndex**, and any tool that speaks the OpenAI API format.

---

## HuggingFace Token (Gated Models)

For models that require access approval (Llama 3, Gemma):

```bash
# Set via environment variable
export HF_TOKEN=hf_your_token_here

# Or set once via CLI
sapient login
```

---

## Architecture

Built in Rust for maximum performance, zero dependencies on Python, ONNX Runtime, or CUDA.

```
sapient-generate          ← Pipeline API — from_pretrained, generate, chat, embed, stream
├── sapient-hub           ← HuggingFace Hub client — parallel downloads, auth, cache
├── sapient-tokenizers    ← All HF tokenizer types + Jinja2 chat templates
├── sapient-models        ← Llama / Phi / Gemma / GPT-2 / BERT / Qwen / Mixtral builders
│
├── sapient-runtime       ← InferenceSession — execution + telemetry
│   ├── sapient-ir        ← Computation graph IR (90+ ops)
│   └── sapient-io        ← GGUF (Q4/Q8 dequant), Safetensors, ONNX loaders
│
├── sapient-backends-cpu    ← CPU kernels: GQA attention, RoPE, RMSNorm, MatMul...
└── sapient-backends-metal  ← macOS Metal backend selection and kernel integration point
```

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
