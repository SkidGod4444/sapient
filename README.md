<div align="center">
  <h1>⚡ SAPIENT</h1>
  <p><strong>Run any HuggingFace LLM or SLM in pure Rust — one line to load, one line to generate</strong></p>
  <p>
    <a href="https://crates.io/crates/sapient-runtime"><img src="https://img.shields.io/crates/v/sapient-runtime.svg" alt="Crates.io"/></a>
    <a href="https://docs.rs/sapient-runtime"><img src="https://docs.rs/sapient-runtime/badge.svg" alt="docs.rs"/></a>
    <a href="https://github.com/SkidGod4444/sapient/actions"><img src="https://github.com/SkidGod4444/sapient/workflows/CI/badge.svg" alt="CI"/></a>
    <img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="License"/>
    <img src="https://img.shields.io/badge/rust-1.75%2B-orange" alt="MSRV"/>
  </p>
</div>

SAPIENT is a **purpose-built LLM & SLM inference engine** written in Rust.  
It works like `llama.cpp` + `🤗 transformers` — but in a single, dependency-clean Rust crate.

---

## 30-Second Quickstart

```toml
[dependencies]
sapient-generate = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use sapient_generate::Pipeline;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Downloads, caches, and runs — zero config needed.
    let pipeline = Pipeline::from_pretrained("microsoft/phi-2").await?;
    let reply = pipeline.generate("The key to good software is").await?;
    println!("{reply}");
    Ok(())
}
```

```bash
cargo run --release
# ✅ Downloaded microsoft/phi-2 to ~/.cache/sapient/hub/
# ✅ Loaded tokenizer (BPE, vocab_size=51200)
# ✅ Detected architecture: Phi
# → "...clean abstractions, consistent naming, and thorough testing."
```

---

## Supported Models

All models on HuggingFace Hub are accessible via `Pipeline::from_pretrained("<model_id>")`.

| Model Family | HuggingFace IDs (examples) | Format | Status |
|---|---|---|---|
| **Llama 3** | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.1-8B` | GGUF / Safetensors | ✅ |
| **Llama 2** | `meta-llama/Llama-2-7b-chat-hf`, `NousResearch/Llama-2-7b-hf` | GGUF / Safetensors | ✅ |
| **Mistral** | `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mistral-7B-v0.1` | GGUF / Safetensors | ✅ |
| **Phi** | `microsoft/phi-2`, `microsoft/Phi-3-mini-4k-instruct` | Safetensors | ✅ |
| **Gemma** | `google/gemma-2-2b-it`, `google/gemma-7b` | Safetensors | ✅ |
| **Qwen** | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2-7B` | Safetensors | ✅ |
| **GPT-2** | `openai-community/gpt2`, `Salesforce/codegen-350M-mono` | Safetensors | ✅ |
| **BERT** | `google-bert/bert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2` | Safetensors | ✅ |
| **Mixtral** | `mistralai/Mixtral-8x7B-Instruct-v0.1` | GGUF / Safetensors | ✅ |
| **Quantized (GGUF)** | `TheBloke/Llama-2-7B-GGUF`, `bartowski/gemma-2-2b-it-GGUF` | GGUF Q4/Q8 | ✅ |
| **Custom LLMs** | Any model with a `config.json` + `tokenizer.json` | Any | ✅ |

> **Any HuggingFace model with a standard `config.json` works.** SAPIENT auto-detects the architecture and builds the correct inference graph.

---

## Usage Examples

### Basic Completion

```rust
use sapient_generate::Pipeline;

let p = Pipeline::from_pretrained("microsoft/phi-2").await?;
println!("{}", p.generate("Rust is great because").await?);
```

### Chat (Instruct Models)

```rust
use sapient_generate::Pipeline;
use sapient_tokenizers::ChatMessage;

let p = Pipeline::from_pretrained("meta-llama/Llama-3.2-1B-Instruct").await?;

let reply = p.chat(&[
    ChatMessage::system("You are a helpful coding assistant."),
    ChatMessage::user("Write a Rust function to reverse a string."),
]).await?;

println!("{reply}");
```

### Sampling Strategies

```rust
use sapient_generate::{Pipeline, GenerationConfig, SamplingStrategy};

let p = Pipeline::from_pretrained("microsoft/phi-2").await?;

// Top-P nucleus sampling with temperature
let config = GenerationConfig {
    max_new_tokens: 200,
    strategy: SamplingStrategy::TopP { p: 0.95, temperature: 0.8 },
    stop_sequences: vec!["<|endoftext|>".into()],
    ..Default::default()
};

let text = p.generate_with_config("Once upon a time", &config).await?;
println!("{text}");
```

### Streaming Output

```rust
use futures::StreamExt;
use sapient_generate::Pipeline;

let p = Pipeline::from_pretrained("meta-llama/Llama-3.2-1B-Instruct").await?;
let mut stream = p.generate_stream("The universe began").await;

while let Some(token) = stream.next().await {
    print!("{token}");
    std::io::stdout().flush().ok();
}
```

### Sentence Embeddings (BERT-style)

```rust
let p = Pipeline::from_pretrained("sentence-transformers/all-MiniLM-L6-v2").await?;
let embedding = p.embed("What is quantum entanglement?").await?;
println!("Embedding dim: {}", embedding.len()); // 384
```

### Load a Quantized GGUF Model

```rust
// Auto-fetches the Q4_K_M quant from TheBloke's repo
let p = Pipeline::from_pretrained("TheBloke/Llama-2-7B-GGUF").await?;
println!("{}", p.generate("The difference between TCP and UDP is").await?);
```

### Custom / Private Models

```rust
use sapient_generate::{Pipeline, LoadOptions};
use sapient_hub::LoadOptions as HubOptions;

let p = Pipeline::from_pretrained_with_opts(
    "my-org/my-custom-llm",
    LoadOptions {
        hub: HubOptions {
            token: Some("hf_your_token_here".into()),
            ..Default::default()
        },
        ..Default::default()
    },
).await?;
```

---

## CLI

```bash
# Install
cargo install sapient-cli

# Interactive chat REPL
sapient chat meta-llama/Llama-3.2-1B-Instruct

# One-shot completion
sapient run microsoft/phi-2 --prompt "Explain RoPE embeddings"

# Download to local cache
sapient download TheBloke/Llama-2-7B-GGUF

# Start an OpenAI-compatible HTTP server
sapient serve microsoft/phi-2 --port 8080

# POST /v1/infer  →  { "outputs": [...], "latency_ms": 12.3 }
curl http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"input_ids": {"shape": [1, 10], "data": [1,2,3,4,5,6,7,8,9,10]}}}'
```

---

## Architecture

```
sapient-generate          ← Pipeline::from_pretrained(), generate(), chat(), embed()
├── sapient-hub           ← HuggingFace Hub client — download, auth, cache, arch detection
├── sapient-tokenizers    ← BPE/WordPiece/SentencePiece + Jinja2 chat templates
├── sapient-models        ← Llama / Phi / Gemma / GPT-2 / BERT / Qwen / Mixtral graphs
│
├── sapient-runtime       ← InferenceSession — model execution + telemetry
│   ├── sapient-ir        ← Graph IR — 90+ ops including GQA, RoPE, MoE
│   ├── sapient-scheduler ← Dynamic batching (5ms micro-window)
│   └── sapient-io        ← GGUF (Q4/Q8 dequant), Safetensors, ONNX loaders
│
└── sapient-backends-cpu  ← CPU kernels: matmul, GQA attention, RoPE, LayerNorm…
    └── sapient-backends-metal  ← 🚧 Apple Silicon GPU (MLX — in progress)
```

---

## Crates

| Crate | Purpose |
|-------|---------|
| `sapient-generate` | **Start here** — Pipeline API, generation, sampling |
| `sapient-hub` | HuggingFace Hub downloads + auth |
| `sapient-tokenizers` | All HF tokenizer types + chat templates |
| `sapient-models` | Model graph builders (Llama, Phi, Gemma…) |
| `sapient-runtime` | Low-level InferenceSession |
| `sapient-core` | Tensor, Shape, DType |
| `sapient-ir` | Computation graph IR |
| `sapient-io` | GGUF/Safetensors/ONNX loaders |
| `sapient-cli` | `sapient` binary |

---

## Building

```bash
git clone https://github.com/SkidGod4444/sapient
cd sapient

# Build everything
cargo build --workspace --release

# Run all tests
cargo test --workspace

# Cross-compile for Raspberry Pi 4/5
cross build --target aarch64-unknown-linux-gnu --release
```

### HuggingFace Token

For gated models (Llama 3, Gemma), set your token:

```bash
export HF_TOKEN=hf_your_token_here
# or
echo "hf_your_token_here" > ~/.cache/huggingface/token
```

---

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT), at your option.

---

## Contributing

1. Fork the repository  
2. `cargo test --workspace` must pass  
3. Open a pull request

Issues and PRs for new model architectures are especially welcome!
