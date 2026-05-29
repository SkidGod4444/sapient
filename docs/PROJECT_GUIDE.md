# 📚 The Big SAPIENT Guide

> A single, friendly tour of the whole project. We start *super* simple (imagine
> you're five), then go deeper, and finally walk through **every folder, every file,
> and every outside tool (dependency) we use** — and what each one is for.

---

## 1. What is SAPIENT? (the five-year-old version)

Imagine you have a very smart robot parrot. 🦜

- You **say something** to the parrot ("What color is the sky?").
- The parrot **thinks for a moment**.
- The parrot **says something back** ("The sky is blue!").

A computer program that can do this "talk back like it understands" trick is called a
**language model**. Famous big ones live on giant computers in the cloud. SAPIENT lets a
**small** version of that smart parrot live **right on your own laptop** — no internet
needed once it's downloaded, no giant cloud computer, no special graphics card required.

**SAPIENT is the machine that runs the parrot's brain on your computer.** That's it.

The fancy words for this are:
- **Edge inference engine** — "edge" means *your* device (not the cloud), "inference"
  means *running* an already-trained brain, and "engine" means *the thing that does the work*.
- **SLM** — "Small Language Model" — a parrot brain small enough to fit on a laptop.

SAPIENT is written in a programming language called **Rust** 🦀, which is loved for being
**fast** and **safe** (it rarely crashes).

---

## 2. How does the parrot actually "think"? (still pretty simple)

The parrot doesn't know words. It only knows **numbers**. So we play a translation game:

1. **Tokenizing** — We chop your sentence into little pieces called **tokens** (think:
   puzzle pieces — sometimes a whole word, sometimes part of a word) and give each piece a
   number. "Hello" might become the number `15496`.
2. **Embedding** — Each number is turned into a long list of numbers (a **vector**) that
   captures its "meaning." Similar words get similar lists.
3. **The layers (the thinking)** — These number-lists go through many **layers** of math.
   Each layer mixes the words together so the parrot understands how they relate ("sky"
   goes with "blue"). The two most important kinds of math here are:
   - **Attention** — every word gets to "look at" the other words and decide which ones
     matter. ("blue" pays attention to "sky".)
   - **A little neural network (MLP)** — squashes and stretches the numbers to find patterns.
4. **Predicting the next token** — After all the layers, the parrot produces a **score for
   every possible next token**. The highest score wins (or we roll dice weighted by the
   scores, to be creative). That winning token is the next piece of the answer.
5. **Repeat** — We add that new token to the sentence and do it all again to get the next
   word, and the next, until the parrot says "I'm done" (a special **end token**).

That loop — predict one token, add it, predict again — is how the whole answer gets written
one piece at a time. When you see words *streaming* onto your screen, that's this loop running.

### A few more words you'll meet
- **Tensor** — just a fancy word for "a box of numbers" (a list, a grid, or a cube of numbers).
- **Weights** — the millions of numbers the parrot *learned* during training. This is the
  "brain." We download these from Hugging Face. They never change while running.
- **KV cache** — a memory notebook 📒. Without it, the parrot would re-read the whole
  conversation for every new word (slow!). The cache lets it remember its earlier work so
  each new word is fast.
- **RoPE (Rotary Position Embedding)** — a trick to tell the parrot **where** each word is
  in the sentence (1st, 2nd, 3rd…), because order matters ("dog bites man" ≠ "man bites dog").
- **Quantization** — squishing the brain's numbers to be smaller (e.g. using tiny 4-bit
  numbers instead of big ones) so the model fits in less memory. SAPIENT can read these
  squished formats (GGUF Q4/Q5/Q8).

---

## 3. The journey of one chat message (the whole engine in one picture)

Here's what happens when you type `sapient chat openhorizon/phi-2` and say "Hi":

```
You type "Hi"
   │
   ▼
[sapient-cli]            The app you run in the terminal. Shows the pretty UI,
                         reads your message.
   │
   ▼
[sapient-hub]            "Do we have this model on disk? No? Download it from
                         Hugging Face." Saves the brain (weights) + tokenizer.
   │
   ▼
[sapient-tokenizers]     Wraps your message in a chat template and turns it into
                         token numbers.
   │
   ▼
[sapient-generate]       The conductor 🎼. Runs the predict-one-token loop, decides
                         which token to pick, streams text back, knows when to stop.
   │
   ▼
[sapient-models]         The actual parrot brain logic: run the layers (attention +
                         MLP) for Phi or Llama-style models, using the weights.
   │
   ▼
[sapient-backends-cpu]   The number-crunching muscles 💪. Does the heavy math
                         (matrix multiply, attention, RoPE, normalization) fast.
   │
   ▼
Tokens come back → turned into text → streamed to your screen as "Hi there!"
```

`sapient-core` is the shared toolbox used by *everyone* above (it defines what a "tensor"
is). `sapient-telemetry` quietly measures speed. `sapient-ir`, `sapient-io`,
`sapient-runtime`, and `sapient-scheduler` power the lower-level "graph" mode (for running
ONNX/GGUF computation graphs and a raw inference server).

---

## 4. The crates (folders of code), one by one

SAPIENT is split into small libraries called **crates** (Rust's word for a package). Each
crate has one job. Splitting things up keeps each part easy to understand and test.

Below, for each crate, you get: **what it's for**, then **every file inside it**.

### 🧱 `sapient-core` — the shared toolbox
The most basic building blocks every other crate uses. If SAPIENT were LEGO, this is the
box of basic bricks.
- `lib.rs` — the front door: lists what this crate shares with others.
- `tensor.rs` — defines the **Tensor** (the "box of numbers"). Shapes, data types, slicing,
  reshaping, and converting half-precision (F16/BF16) numbers to full F32. The heart of the toolbox.
- `buffer.rs` — the raw block of memory a tensor's numbers actually live in (kept aligned so
  the CPU can read it quickly).
- `dtype.rs` — the list of number **types** we support: F32 (big/accurate), F16 & BF16
  (half-size), and integers. Knows how many bytes each takes.
- `shape.rs` — describes a tensor's **shape** (e.g. "3 rows × 4 columns") and the math to
  walk through it (strides).
- `error.rs` — the shared list of things that can go wrong (e.g. "shapes don't match"), so
  errors read nicely everywhere.

### 🔌 `sapient-ir` — the computation graph (advanced mode)
Describes a model as a **graph** of math operations (like a flowchart: this op feeds into
that op). Used by the ONNX/GGUF "graph" path, not the main chat path.
- `lib.rs` — front door.
- `op.rs` — the catalog of operations (Add, MatMul, Softmax, …).
- `node.rs` — one box in the flowchart (an op plus its inputs/outputs).
- `graph.rs` — the whole flowchart and how to build/connect it.
- `shape_inference.rs` — figures out the shape of each tensor as data flows through, before
  running anything.
- `passes/` — automatic **optimizers** that rewrite the graph to be faster:
  - `passes/mod.rs` — lists the passes.
  - `passes/constant_folding.rs` — pre-computes parts that never change.
  - `passes/dead_code.rs` — deletes ops whose results nobody uses.
  - `passes/fusion.rs` — merges several small ops into one bigger, faster op.
  - `passes/layout.rs` — arranges data in memory for faster access.

### 💾 `sapient-io` — reading model files from disk
Knows how to open the file formats that store model brains.
- `lib.rs` — front door.
- `safetensors.rs` — reads **Safetensors** files (the main, modern weight format; safe & fast).
- `gguf.rs` — reads **GGUF** files, including **dequantizing** squished Q4/Q5/Q8 numbers back
  into normal numbers.
- `onnx.rs` — reads **ONNX** model graphs (a cross-tool standard format).

### 🔤 `sapient-tokenizers` — words ↔ numbers
Turns text into tokens and back, and formats chat conversations.
- `lib.rs` — front door.
- `tokenizer.rs` — wraps Hugging Face's tokenizer; finds the special **start/end tokens**
  (including *all* end tokens a model uses, like `<|im_end|>`) so generation stops correctly.
- `chat.rs` — applies **chat templates** (the Jinja2 recipe that wraps your message with
  role markers like `<|im_start|>user`). Has built-in templates for ChatML, Llama, Gemma, etc.

### 🌐 `sapient-hub` — downloading & managing models
Talks to Hugging Face, downloads model files, caches them, and keeps the **registry** of
which models SAPIENT supports.
- `lib.rs` — front door.
- `registry.rs` — the **curated list** of supported models. Maps friendly `openhorizon/…`
  aliases to real Hugging Face repos (e.g. `openhorizon/phi-2` → `microsoft/phi-2`).
- `client.rs` — the high-level "download this model" client.
- `download.rs` — the fast downloader (parallel chunks); reads the `SAPIENT_HUB_*` env vars.
- `cache.rs` — where downloaded files are stored on your disk.
- `resolver.rs` — figures out *which* files a model needs (config, tokenizer, weight shards).
- `model_info.rs` — reads a model's `config.json` into a tidy `ModelInfo` (layers, heads,
  RoPE settings, `partial_rotary_factor`, etc.).
- `gguf.rs` — hub-side helpers for GGUF repositories.

### 🧠 `sapient-models` — the parrot brain logic
The real generation math: how to run a Phi or Llama-style model layer by layer.
- `lib.rs` — front door.
- `weights.rs` — loads weight tensors from Safetensors and finds them by name (handles
  prefixes, biases, and tied embeddings).
- `gguf_weights.rs` — maps GGUF tensor names to the names the engine expects.
- `registry.rs` — builds an IR graph for a model type (graph mode).
- `forward/` — the **forward pass** (running the model to get an answer):
  - `forward/mod.rs` — picks the right engine for a model type.
  - `forward/common.rs` — shared building blocks: embedding lookup, **linear layers**
    (`matmul_nt`), normalization, RoPE (full + partial), attention, bias-add.
  - `forward/backend.rs` — the backend interface (CPU vs Metal) and the default helpers
    like "linear with bias" and "partial RoPE."
  - `forward/llama.rs` — the **Llama engine** (also runs Qwen2.5, SmolLM2, TinyLlama,
    Mistral): RMSNorm, RoPE, SwiGLU MLP, optional Q/K/V biases for Qwen.
  - `forward/phi.rs` — the **Phi engine**: LayerNorm with biases, partial RoPE, parallel
    attention+MLP block, and the `<final_layernorm>` + `lm_head` bias.
- `architectures/` — graph **builders** for many model types (used by the IR/graph path).
  Note: only Phi and Llama are wired into live chat today; the rest are scaffolding.
  - `llama.rs`, `phi.rs`, `qwen.rs`, `gemma.rs`, `gpt2.rs`, `bert.rs`, `mixtral.rs`, `mod.rs`.

### 🎼 `sapient-generate` — the conductor
Ties everything together into the simple `Pipeline` you call. Runs the token loop, picks
tokens, streams text, and stops at the right time.
- `lib.rs` — front door; exposes `GenerationConfig` and `SamplingStrategy`.
- `pipeline.rs` — the `Pipeline`: load a model, `generate`, `chat`, `generate_stream`,
  `embed`. Handles chat templates, stop sequences, and **multi-EOS** stopping.
- `sampler.rs` — **how to pick the next token**: greedy (highest score), temperature,
  top-k, top-p, and repetition penalty.
- `kv_cache.rs` — the memory notebook (KV cache) helpers.

### 🗓️ `sapient-scheduler` — running many requests (server mode)
Batches and schedules inference requests so a server can handle several at once.
- `lib.rs` — front door.
- `request.rs` — one inference request (with priority/deadline fields).
- `batcher.rs` — groups multiple requests into one batch.
- `scheduler.rs` — decides what runs when.
- `executor.rs` — actually runs the batches.

### ⚙️ `sapient-runtime` — the graph runtime
Runs an IR graph end-to-end with a session object (the engine behind `sapient serve`).
- `lib.rs` — front door.
- `model.rs` — loads a model graph + its weights.
- `session.rs` — `InferenceSession`: feed inputs, get outputs, with timing.

### 📊 `sapient-telemetry` — measuring speed & health
Optional metrics, tracing, and profiling so you can see how fast things run.
- `lib.rs` — front door.
- `telemetry.rs` — sets up logging/tracing.
- `metrics.rs` — counters and histograms (e.g. tokens/sec).
- `profiler.rs` — simple timers for sections of code.

### 💪 `sapient-backends-cpu` — the CPU number-crunching muscles
The fast math that runs on any CPU. This is where most of the real work happens during chat.
- `lib.rs` — front door.
- `backend.rs` — dispatches each operation to the right kernel.
- `pool.rs` — reuses memory buffers so we don't constantly allocate/free (faster).
- `kernels/` — the individual math routines ("kernels"):
  - `kernels/mod.rs` — lists the kernels.
  - `kernels/matmul.rs` — **matrix multiply** + `matmul_nt` (the linear-layer core) + `gemm`.
  - `kernels/attention.rs` — **attention** + grouped-query attention + causal masking.
  - `kernels/rope.rs` — **RoPE** position trick (full and partial/Phi versions).
  - `kernels/softmax.rs` — turns scores into probabilities (stable version).
  - `kernels/layernorm.rs` — **LayerNorm** and **RMSNorm** (keep numbers well-behaved).
  - `kernels/reduce.rs` — sums/means/maxes across a dimension.
  - `kernels/elementwise.rs` — add/multiply/etc. and activations (GELU, SiLU…).
  - `kernels/conv2d.rs` — 2D convolution (for vision-style ops).

### 🍎 `sapient-backends-metal` — Apple Silicon GPU
The hook for running on a Mac's GPU via Apple's **MLX**. Enabled when built with
`--features mlx`; otherwise the engine falls back to the CPU kernels.
- `lib.rs` — front door.
- `backend.rs` — Metal/MLX backend detection and integration point.

### 🖥️ `sapient-cli` — the app you actually run
The `sapient` command-line program: parses commands, shows the modern UI, and calls the
libraries above.
- `main.rs` — defines all commands (`chat`, `pull`, `run`, `list`, `models`, `info`,
  `serve`, `login`, `update`, …) and wires them up.
- `ui.rs` — the **modern terminal UI**: banner, colored role "chip" badges, spinners,
  tables, success/error messages, and the tokens/sec stat line.
- `hub.rs` — CLI-side model management (list cached, remove, login, resolve paths).
- `progress.rs` — the live download progress bar.
- `server.rs` — the raw-tensor HTTP server (`/v1/infer`, `/v1/health`, …) for graph files.
- `update.rs` — `sapient update`: self-updates the binary from GitHub releases.

---

## 5. Every dependency (outside tool) and what it does

We don't build everything from scratch — we stand on great open-source libraries. Here's
**every external crate** we depend on, grouped by purpose, in plain language.

### Core utilities (used widely)
| Crate | What it does for us |
|---|---|
| `thiserror` | Lets us define tidy, readable error types. |
| `anyhow` | Easy "something went wrong" error handling in app code. |
| `serde` / `serde_json` | Convert structs ↔ JSON (config files, API messages). |
| `bincode` | Compact binary save/load (used by the IR). |
| `bytemuck` | Safely reinterpret bytes as numbers (e.g. raw bytes → f32). |
| `half` | The F16 / BF16 half-size number types. |
| `num-traits` | Generic math over different number types. |
| `ordered-float` | Floats that can be sorted / used as map keys (IR constants). |
| `uuid` | Unique IDs for scheduler requests. |
| `tracing` | Structured logging (the "what's happening" messages). |

### Async & parallel (doing many things at once)
| Crate | What it does for us |
|---|---|
| `tokio` | The async runtime — powers downloads, the server, streaming. |
| `tokio-stream` | Streams of values over time (streaming tokens). |
| `futures` | Building blocks for async code. |
| `async-trait` | Lets traits have `async` methods. |
| `rayon` | Easy CPU multi-threading (splits math across cores). |
| `flume` | Fast channels for passing work between threads. |
| `parking_lot` | Faster locks (mutexes) than the standard ones. |
| `num_cpus` | Counts your CPU cores (to size download workers/threads). |

### Math & compute
| Crate | What it does for us |
|---|---|
| `matrixmultiply` | Fast, pure-Rust matrix multiply — the core of every linear layer. |
| `blas-src` / `cblas-sys` | Optional link to a system BLAS for extra matrix speed. |

### Model formats & Hugging Face
| Crate | What it does for us |
|---|---|
| `memmap2` | Memory-maps big weight files (read without loading all into RAM). |
| `prost` | Decodes Protobuf (the ONNX file format). |
| `hf-hub` | Downloads models from the Hugging Face Hub. |
| `tokenizers` | Hugging Face's tokenizer engine (text ↔ tokens). |
| `minijinja` | Runs Jinja2 chat templates (formats conversations). |

### Networking & downloads
| Crate | What it does for us |
|---|---|
| `reqwest` | HTTP client (download files, query the Hub API). |
| `ureq` | A simpler blocking HTTP client (used by self-update). |
| `indicatif` | Pretty progress bars and spinners. |
| `console` | Terminal styling (colors, the modern chat UI). |
| `sha2` | SHA-256 hashing (checksums). |
| `dirs` | Finds the right cache/home folders on each OS. |
| `flate2` / `tar` / `zip` | Unpack downloaded `.tar.gz` / `.zip` archives (self-update). |

### CLI & server
| Crate | What it does for us |
|---|---|
| `clap` | Parses command-line arguments and builds `--help`. |
| `axum` | The web framework for `sapient serve`. |
| `tower` / `tower-http` | Middleware for the server (CORS, tracing). |

### Telemetry (measuring)
| Crate | What it does for us |
|---|---|
| `tracing-subscriber` | Decides where log messages go and how they look. |
| `metrics` | Records numbers like tokens/sec and request counts. |
| `metrics-exporter-prometheus` | Exposes those numbers for Prometheus to scrape. |
| `opentelemetry` / `opentelemetry_sdk` / `opentelemetry-otlp` | Send traces to observability tools. |

### Apple Silicon (macOS, optional)
| Crate | What it does for us |
|---|---|
| `mlx-rs` | Rust bindings to Apple's MLX framework — runs math on the Mac GPU (only when built with `--features mlx`). |

### Testing & benchmarking (developer-only)
| Crate | What it does for us |
|---|---|
| `criterion` | Benchmarks (measures performance precisely). |
| `proptest` | Property-based testing (throws many random inputs at code). |
| `approx` | Compares floating-point numbers "close enough" in tests. |
| `tempfile` | Temporary files/folders during tests. |
| `log` / `env_logger` | Simple logging used in some places. |

---

## 6. How to build and run it yourself

```bash
# 1) Get the code
git clone https://github.com/SkidGod4444/sapient
cd sapient

# 2) Build the app (CPU version — works everywhere)
cargo build --release -p sapient-cli
# the program is now at ./target/release/sapient

# 3) See which models are supported
./target/release/sapient models

# 4) Chat! (downloads the model the first time)
./target/release/sapient chat openhorizon/phi-2

# Apple Silicon GPU build (optional, macOS only):
cargo build --release -p sapient-cli --features mlx
```

Useful chat commands while chatting: `/help`, `/clear` (forget the conversation), `/exit`.

---

## 7. How the pieces depend on each other (the map)

```
sapient-cli  ──────────────► everything (it's the app)
   │
   ├── sapient-generate  ─► sapient-models, sapient-tokenizers, sapient-hub,
   │                        sapient-runtime, sapient-io, sapient-backends-cpu
   │
   ├── sapient-models    ─► sapient-hub, sapient-io, sapient-ir, sapient-backends-cpu
   ├── sapient-runtime   ─► sapient-scheduler, sapient-io, sapient-telemetry, sapient-ir
   ├── sapient-scheduler ─► sapient-ir, sapient-backends-cpu
   ├── sapient-backends-cpu ─► sapient-ir
   ├── sapient-ir / sapient-io / sapient-tokenizers / sapient-hub ─► sapient-core
   └── sapient-core      ─► (nobody — it's the foundation)
```

Read it top-down: the app uses the conductor, the conductor uses the brain and the
muscles, and everyone shares the basic toolbox at the bottom.

---

## 8. Glossary (quick reference)

- **Token** — a small chunk of text (word or word-part) the model reads as a number.
- **Tensor** — a box of numbers (list / grid / cube).
- **Weights** — the model's learned numbers (its "brain"). Downloaded, never changed at runtime.
- **Forward pass** — running the model once to get the next-token scores.
- **Attention** — the step where words "look at" each other to understand context.
- **MLP / SwiGLU / GELU / SiLU** — small math networks/activations inside each layer.
- **LayerNorm / RMSNorm** — keep the numbers from getting too big or too small.
- **RoPE** — tells the model the position of each token.
- **KV cache** — a memory of past work so each new token is fast.
- **Logits** — the raw scores for every possible next token (before picking one).
- **Sampling** — how we choose the next token from the scores (greedy, top-k, top-p…).
- **EOS** — "end of sequence" token: the model's way of saying "I'm done."
- **Quantization** — storing weights with fewer bits to save memory (GGUF Q4/Q5/Q8).
- **Backend** — where the math runs: CPU (everywhere) or Metal/MLX (Mac GPU).
- **Crate** — a Rust package/library.
- **IR (Intermediate Representation)** — a flowchart of math ops used by the graph runtime.

---

---

## 9. Performance guide — how to get fast inference

### Recommended: GGUF quantized models

For **CPU inference** on any platform (Linux, Raspberry Pi, etc.), always use a GGUF
quantized model rather than F16 safetensors:

| Model | Format | RAM needed | Typical tok/s (Apple M4, CPU) |
|---|---|---|---|
| `openhorizon/qwen2.5-0.5b-q4` | GGUF Q8_0 | ~640 MB | ~17 tok/s |
| `openhorizon/phi-2-q4` | GGUF Q4_K_M | ~1.5 GB | ~5 tok/s |
| `openhorizon/phi-2` | F16 safetensors | ~5.2 GB | ~0.8 tok/s |

The F16 safetensors path is slow on CPU because every token requires converting
large weight matrices from F16 → F32 (phi-2 MLP weights are 52 MB each × 64 per layer).
GGUF Q4/Q8 keeps weights quantized in memory and dequantizes one 32-element block at
a time inside the dot product, so memory bandwidth is 4–8× lower.

### Apple Silicon: Metal GPU

Build with `--features mlx` to enable the Metal GPU backend. MLX uses Apple Silicon's
unified memory — there's no CPU↔GPU copy overhead. The engine picks Metal automatically
when the model fits in memory (`sapient backend-info` shows the capacity).

Key changes shipped in Phase 2/3:
- **Phase 2**: rayon parallel dot products across output rows + NEON SIMD (Q4_0, Q8_0).
- **Phase 3**: MLX persistent weight cache (upload each weight to GPU once, reuse per token),
  native MLX `gqa_attention` via `fast::scaled_dot_product_attention` (removes CPU fallback),
  auto backend selection by available unified memory.

### Linux / NVIDIA (DGX, cloud)

CUDA is not yet supported. Until it is, use GGUF Q4/Q8 models on CPU — they run the
rayon + NEON parallel kernels and are the fastest CPU path. The DGX Spark (ARM64 Grace)
also has NEON, so the Q8_0 path gets the full SIMD benefit.

*Happy hacking! If anything here ever stops matching the code, the code wins — please open
a PR to fix the docs.* 🦜
