<div align="center">
  <h1>⚡ SAPIENT</h1>
  <p><strong>Run any HuggingFace LLM or SLM locally — one command to install, one line to run</strong></p>
  <p>
    <a href="https://crates.io/crates/sapient-generate"><img src="https://img.shields.io/crates/v/sapient-generate.svg" alt="Crates.io"/></a>
    <a href="https://docs.rs/sapient-generate"><img src="https://docs.rs/sapient-generate/badge.svg" alt="docs.rs"/></a>
    <a href="https://github.com/SkidGod4444/sapient/actions"><img src="https://github.com/SkidGod4444/sapient/workflows/CI/badge.svg" alt="CI"/></a>
    <img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License"/>
    <img src="https://img.shields.io/badge/rust-1.75%2B-orange" alt="MSRV"/>
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
curl -fsSL https://raw.githubusercontent.com/SkidGod4444/sapient/main/install.sh | sh
```

> **Piped installs** go to `~/.local/bin`. If `sapient` is not found afterward, run:
> `export PATH="$HOME/.local/bin:$PATH"` and restart your terminal.

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/SkidGod4444/sapient/main/install.ps1 | iex
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

> **Requires v0.1.1+** — reinstall with the [install script](#macos--linux-one-command) if `chat` or `pull` are missing.

```bash
# Interactive chat — just like Ollama
sapient chat meta-llama/Llama-3.2-1B-Instruct

# One-shot completion (Hub models need --prompt)
sapient run microsoft/phi-2 --prompt "Explain transformers in simple terms"

# Download a model to local cache
sapient pull TheBloke/Llama-2-7B-GGUF

# List cached models
sapient list

# Gated models (Llama, Gemma) — set token first
sapient login

# Start an HTTP inference server (ONNX/GGUF files)
sapient serve ./model.gguf --port 8080

# Show info about a model
sapient info google/gemma-2-2b-it
```

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
    let p = Pipeline::from_pretrained("microsoft/phi-2").await?;
    println!("{}", p.generate("The key to good software is").await?);
    Ok(())
}
```

### Chat (Instruct Models)

```rust
use sapient_tokenizers::ChatMessage;

let p = Pipeline::from_pretrained("meta-llama/Llama-3.2-1B-Instruct").await?;
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

All HuggingFace Hub models work with `sapient chat <model-id>` or `Pipeline::from_pretrained("<model-id>")`.

| Family | Example IDs | Format |
|---|---|---|
| **Llama 3** | `meta-llama/Llama-3.2-1B-Instruct` | GGUF / Safetensors |
| **Mistral** | `mistralai/Mistral-7B-Instruct-v0.3` | GGUF / Safetensors |
| **Phi** | `microsoft/phi-2`, `microsoft/Phi-3-mini-4k-instruct` | Safetensors |
| **Gemma** | `google/gemma-2-2b-it` | Safetensors |
| **Qwen** | `Qwen/Qwen2.5-1.5B-Instruct` | Safetensors |
| **GPT-2** | `openai-community/gpt2` | Safetensors |
| **BERT** | `sentence-transformers/all-MiniLM-L6-v2` | Safetensors |
| **Mixtral (MoE)** | `mistralai/Mixtral-8x7B-Instruct-v0.1` | GGUF / Safetensors |
| **Any GGUF** | `TheBloke/Llama-2-7B-GGUF` | Q4_0, Q8_0 |
| **Custom** | Any model with `config.json` + `tokenizer.json` | Any |

---

## OpenAI-Compatible API

```bash
sapient serve microsoft/phi-2 --port 8080
```

```bash
# Works with any OpenAI SDK — just change the base URL
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-2",
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
├── sapient-hub           ← HuggingFace Hub client — download, auth, cache, arch detection
├── sapient-tokenizers    ← All HF tokenizer types + Jinja2 chat templates
├── sapient-models        ← Llama / Phi / Gemma / GPT-2 / BERT / Qwen / Mixtral builders
│
├── sapient-runtime       ← InferenceSession — execution + telemetry
│   ├── sapient-ir        ← Computation graph IR (90+ ops)
│   └── sapient-io        ← GGUF (Q4/Q8 dequant), Safetensors, ONNX loaders
│
└── sapient-backends-cpu  ← CPU kernels: GQA attention, RoPE, RMSNorm, MatMul...
    └── sapient-backends-metal  ← 🚧 Apple Silicon GPU via Metal (coming soon)
```

---

## Build from Source

```bash
git clone https://github.com/SkidGod4444/sapient
cd sapient
cargo build --workspace --release

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
- Apple Metal / MLX GPU backend (`crates/sapient-backends/metal/`)
- Quantization kernels (INT4, INT8 fused)
