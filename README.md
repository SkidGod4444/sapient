<div align="center">
  <h1>⚡ SAPIENT</h1>
  <p><strong>High-performance ML inference engine for edge devices</strong></p>
  <p>
    <a href="https://crates.io/crates/sapient-runtime"><img src="https://img.shields.io/crates/v/sapient-runtime.svg" alt="Crates.io"/></a>
    <a href="https://docs.rs/sapient-runtime"><img src="https://docs.rs/sapient-runtime/badge.svg" alt="docs.rs"/></a>
    <a href="https://github.com/SkidGod4444/sapient/actions"><img src="https://github.com/SkidGod4444/sapient/workflows/CI/badge.svg" alt="CI"/></a>
    <img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="License"/>
    <img src="https://img.shields.io/badge/rust-1.75%2B-orange" alt="MSRV"/>
  </p>
</div>

SAPIENT is a modular, high-performance neural network inference engine written in Rust, designed for deployment on:
- **macOS** (Apple Silicon M1/M2/M3/M4 and Intel)
- **Raspberry Pi 4/5** (ARM64)
- **Linux x86_64** servers

---

## ✨ Features

| Feature | Status |
|---------|--------|
| **ONNX loading** (Opset 17+, hand-rolled protobuf) | ✅ |
| **GGUF loading** (Q4_0, Q8_0, F16, F32) | ✅ |
| **Safetensors loading** (zero-copy mmap) | ✅ |
| **CPU backend** — matmul, conv2d, softmax, LayerNorm, RMSNorm, GELU/SiLU | ✅ |
| **Dynamic batching** (5ms micro-window scheduler) | ✅ |
| **HTTP inference server** (`/v1/infer`, `/v1/batch_infer`) | ✅ |
| **Chrome trace profiler** (chrome://tracing JSON) | ✅ |
| **IR optimizer** — constant folding, DCE, layout, MatMul+Add fusion | ✅ |
| **Apple Accelerate BLAS** (M-series AMX acceleration) | 🚧 |
| **Metal / MLX backend** (Apple Silicon GPU) | 🚧 |
| **CUDA backend** | 🗓️ Planned |

---

## 🚀 Quick Start

### CLI

```bash
# Install from crates.io
cargo install sapient-cli

# Run inference
sapient run model.onnx --input input.json

# Benchmark batch sizes
sapient bench model.onnx --batch-sizes 1,4,8,16

# Start HTTP server
sapient serve model.onnx --port 8080

# Inspect graph as DOT
sapient inspect model.onnx
```

### Library

```toml
[dependencies]
sapient-runtime = "0.1"
```

```rust
use sapient_runtime::{Model, ModelConfig, InferenceSession, SessionOptions};
use sapient_core::Tensor;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // Load and optimize a model
    let model = Model::load("model.onnx".as_ref(), ModelConfig::default())?;

    // Create a session
    let session = InferenceSession::new(model, SessionOptions {
        telemetry: true,
        ..Default::default()
    })?;

    // Run inference
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4])?);

    let outputs = session.run(inputs)?;
    println!("Output shape: {:?}", outputs[0].shape().dims());
    Ok(())
}
```

### HTTP API

```bash
# POST /v1/infer
curl -X POST http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "x": { "shape": [1, 4], "data": [1.0, 2.0, 3.0, 4.0] }
    }
  }'

# Response:
{
  "outputs": [{ "shape": [1, 2], "dtype": "f32", "data": [0.3, 0.7] }],
  "latency_ms": 0.23
}
```

---

## 🏗️ Architecture

```
sapient-core           ← Tensor, Shape, DType, aligned Buffer
├── sapient-ir         ← Graph IR, 80+ OpTypes, optimization passes
│   └── sapient-backends-cpu   ← CPU kernels (matmul, conv2d, softmax…)
│       └── sapient-backends-metal  ← 🚧 MLX/Metal GPU (Apple Silicon)
│           └── sapient-scheduler   ← Dynamic batcher + async executor
│               └── sapient-io      ← ONNX / GGUF / Safetensors loaders
│                   └── sapient-telemetry  ← Tracing + metrics + profiler
│                       └── sapient-runtime   ← Model + InferenceSession
│                           └── sapient-cli   ← `sapient` binary + HTTP server
```

---

## 📦 Crates

| Crate | Description | Crates.io |
|-------|-------------|-----------|
| `sapient-runtime` | High-level API — load model, run session | [![](https://img.shields.io/crates/v/sapient-runtime)](https://crates.io/crates/sapient-runtime) |
| `sapient-core` | Tensor, Shape, DType, Buffer | [![](https://img.shields.io/crates/v/sapient-core)](https://crates.io/crates/sapient-core) |
| `sapient-ir` | Graph IR + optimization passes | [![](https://img.shields.io/crates/v/sapient-ir)](https://crates.io/crates/sapient-ir) |
| `sapient-io` | ONNX / GGUF / Safetensors loaders | [![](https://img.shields.io/crates/v/sapient-io)](https://crates.io/crates/sapient-io) |
| `sapient-scheduler` | Dynamic batch scheduler | [![](https://img.shields.io/crates/v/sapient-scheduler)](https://crates.io/crates/sapient-scheduler) |
| `sapient-telemetry` | Tracing, metrics, Chrome profiler | [![](https://img.shields.io/crates/v/sapient-telemetry)](https://crates.io/crates/sapient-telemetry) |
| `sapient-cli` | `sapient` CLI binary | [![](https://img.shields.io/crates/v/sapient-cli)](https://crates.io/crates/sapient-cli) |

---

## 🔧 Building

```bash
# Clone
git clone https://github.com/SkidGod4444/sapient
cd sapient

# Build (release)
cargo build --workspace --release

# Run tests
cargo test --workspace

# Benchmark CPU ops
cargo bench -p sapient-backends-cpu

# Cross-compile for Raspberry Pi 4/5
cross build --target aarch64-unknown-linux-gnu --release
```

### macOS GPU Acceleration

SAPIENT will automatically use Apple's **Accelerate** framework on macOS for BLAS-accelerated matrix math (AMX units on M-series):

```bash
cargo build --release -p sapient-backends-cpu --features accelerate
```

Full **MLX / Metal GPU** support (Apple Silicon) is in progress. Follow [`#metal-backend`](https://github.com/SkidGod4444/sapient/issues) for updates.

---

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first for significant changes.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/metal-backend`)
3. Run `cargo test --workspace` before submitting
4. Open a pull request
