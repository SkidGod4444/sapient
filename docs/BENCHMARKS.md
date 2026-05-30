# SAPIENT Benchmarks — Metal, CPU, vs mlx-lm & Ollama

> Generated: 2026-05-30 · Hardware: **Apple M4 · 16 GB RAM · macOS 26.5 aarch64**
> SAPIENT **v0.3.4** · mlx-lm 0.31.2 · Ollama 0.12.x

---

## TL;DR

The v0.3.4 `MlxForwardEngine` (native lazy-graph Metal forward pass) lands SAPIENT's
GPU decode throughput **ahead of Ollama on the 0.5B model** and within **1.3–1.5×
of mlx-lm** — the Apple-native reference — while staying a single 22 MB daemon-free
binary.

| Axis | SAPIENT | Best alternative | Notes |
|---|---|---|---|
| Decode tok/s — 0.5B (Metal) | **168** | mlx-lm 249 / Ollama 154 | **beats Ollama** |
| Decode tok/s — 1.5B (Metal) | **70** | mlx-lm 94 / Ollama 78 | competitive |
| CPU → Metal speedup | **6.5–8.6×** | — | same binary, `--backend metal` |
| Binary size | **22 MB** | Ollama 28 MB | single static binary |
| Daemon required | **No** | Ollama: yes | direct process |
| Peak RAM — 0.5B | 2.3 GB | mlx-lm 0.33 GB | see *Memory* below |

![Decode throughput](assets/decode_throughput.png)

**The honest story:** mlx-lm is still the fastest path on Apple Silicon — it loads
pre-quantized 4-bit weights straight into the GPU and has a hand-tuned prefill.
SAPIENT now matches the *class* of performance (same MLX kernels, same lazy-graph
strategy) from pure Rust, beats Ollama on small-model decode, and needs no daemon.
Where SAPIENT still trails: **prompt prefill / TTFT** and **peak RAM** (it dequantizes
GGUF → F32 → MLX-Q4 at load and keeps the embedding table in F32).

---

## What changed in v0.3.4

The headline fix: **RoPE was being applied on the wrong tensor axis.**
`mlx_rs::fast::rope` treats dimension −2 as the sequence-position axis, but the
engine fed it `[1, seq, n_heads, head_dim]` — so dim −2 was `n_heads`, assigning a
different rotary position to each *head* instead of the same position to all heads
at a sequence step. Positional encoding was scrambled and every model collapsed to
repeatedly emitting a single token. Transposing to `[1, n_heads, seq, head_dim]`
*before* RoPE (matching mlx-lm) restored coherent output — and unlocked the
throughput numbers below.

---

## CPU → Metal speedup

Same binary, same GGUF weights, just `--backend metal`:

![SAPIENT CPU vs Metal](assets/sapient_speedup.png)

| Model | CPU (NEON) | Metal (MLX) | Speedup |
|---|---|---|---|
| Qwen2.5-0.5B Q4 | 19.6 tok/s | **167.9 tok/s** | **8.6×** |
| Qwen2.5-1.5B Q4 | 10.8 tok/s | **70.3 tok/s** | **6.5×** |

---

## Full comparison

Decode throughput is measured **decode-only** — `generated_tokens ÷ (total_time −
TTFT)` — so prefill time does not dilute the steady-state rate. Prompt: a 58-token
request for a 200-word backprop explanation; 200 tokens generated.

### Qwen2.5-0.5B (4-bit)

| Engine | Backend | Decode tok/s | TTFT | Peak RAM |
|---|---|---|---|---|
| mlx-lm | Metal | **248.6** | 39 ms | **0.33 GB** |
| **SAPIENT** | **Metal** | **167.9** | 515 ms | 2.28 GB |
| Ollama | Metal | 153.7 | 28 ms | — (daemon) |
| SAPIENT | CPU | 19.6 | 296 ms | 1.39 GB |

### Qwen2.5-1.5B (4-bit)

| Engine | Backend | Decode tok/s | TTFT | Peak RAM |
|---|---|---|---|---|
| mlx-lm | Metal | **94.2** | 264 ms | **0.95 GB** |
| Ollama | Metal | 77.9 | 64 ms | — (daemon) |
| **SAPIENT** | **Metal** | **70.3** | 2997 ms | 2.13 GB |
| SAPIENT | CPU | 10.8 | 1233 ms | 0.85 GB |

```
Decode tok/s — Qwen2.5-0.5B
  mlx-lm   Metal  █████████████████████████  249
  SAPIENT  Metal  █████████████████░░░░░░░░░  168   ← beats Ollama
  Ollama   Metal  ███████████████░░░░░░░░░░░  154
  SAPIENT  CPU    ██░░░░░░░░░░░░░░░░░░░░░░░░░   20

Decode tok/s — Qwen2.5-1.5B
  mlx-lm   Metal  █████████████████████████   94
  Ollama   Metal  █████████████████████░░░░   78
  SAPIENT  Metal  ███████████████████░░░░░░   70
  SAPIENT  CPU    ███░░░░░░░░░░░░░░░░░░░░░░░   11
```

---

## Known gaps (and why)

**1. TTFT / prefill is slow (515 ms @ 0.5B, ~3 s @ 1.5B).**
The decode path is now fast, but the prompt-prefill forward still trails the
reference engines by a wide margin. Two causes: the GQA attention runs as a
per-KV-head 4D matmul loop (mlx_rs 0.25.3's fused SDPA mishandles grouped-query
attention), and weights are converted GGUF → F32 → MLX-Q4 on the way in. Optimising
prefill is the top item on the [roadmap](../ROADMAP.md).

**2. Peak RAM is ~2 GB regardless of model (vs mlx-lm's 0.3–1.0 GB).**
mlx-lm memory-maps native 4-bit safetensors. SAPIENT dequantizes GGUF K-quants to
F32 to feed `mlx_rs::ops::quantize`, and keeps the token-embedding / `lm_head`
matrix in F32. Storing those as MLX-Q4 and quantizing weights without the F32
intermediate would close most of this gap.

**3. mlx-lm is still 1.3–1.5× faster on decode.**
It is the Apple-native reference with a hand-tuned prefill and zero format
conversion. Matching it from a portable Rust + GGUF stack is the long-term target;
landing in the same performance class is the v0.3.4 milestone.

---

## Binary & deployment

| Metric | SAPIENT | Ollama | mlx-lm |
|---|---|---|---|
| Distribution | single 22 MB binary | 28 MB + daemon | Python + venv |
| Daemon required | **No** | `ollama serve` | No (library) |
| Runtime deps | none (static) | none | Python 3.9+, MLX |
| Works on Linux / ARM SBC | **Yes** (CPU/NEON) | Yes | No (Apple only) |
| GPU backend | Metal (`--features mlx`) | Metal | Metal |

SAPIENT is the only one of the three that is a single dependency-free binary *and*
runs the same code on a Raspberry Pi (CPU/NEON) and an M-series Mac (Metal).

---

## Reproducibility

```bash
# 1. Build the Metal binary and colocate the shader library
cargo build --release -p sapient-cli --features mlx
cp "$(find target/release -name 'mlx.metallib' | head -1)" target/release/

# 2. SAPIENT — CPU and Metal, decode-only throughput
PROMPT="Write a detailed 200-word explanation of how neural networks learn through backpropagation, including the role of gradients and the chain rule."
for backend in cpu metal; do
  ./target/release/sapient bench-llm openhorizon/qwen2.5-0.5b-q4 \
    --prompt "$PROMPT" --max-tokens 200 --runs 3 --backend $backend --json \
    > results/sapient_${backend}_0.5b.json
done

# 3. mlx-lm reference (pip install mlx-lm)
python3 -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "$PROMPT" --max-tokens 200

# 4. Ollama reference (ollama serve &; ollama pull qwen2.5:0.5b)
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:0.5b","prompt":"'"$PROMPT"'","options":{"num_predict":200},"stream":false}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin,strict=False); print(d['eval_count']/(d['eval_duration']/1e9),'tok/s')"

# 5. Regenerate the charts in this report
python3 scripts/gen-benchmark-charts.py
```

The raw per-run JSON for this report lives in `results/v033/`.

---

## Guidance by use case

**M-series Mac, want max speed:** mlx-lm (or Ollama) edge SAPIENT on raw decode and
prefill. Reach for SAPIENT when you also want a daemon-free single binary or plan to
ship the *same* tool to non-Apple hardware.

**Raspberry Pi / ARM SBC / constrained edge:** SAPIENT, clearly — 22 MB static
binary, NEON kernels, mmap for bigger-than-RAM models, no Python, no daemon.

**CI / scripting / embedded automation:** SAPIENT's direct-process model (no server
lifecycle) is the simplest to wire up.

**Apple Silicon decode at small model sizes:** SAPIENT Metal now beats Ollama on the
0.5B model and trails mlx-lm by ~1.5× — a viable single-binary GPU option.

---

> *Real measurements taken 2026-05-30 on Apple M4, 16 GB RAM, macOS 26.5 aarch64.*
> *We publish the engines that beat us openly — credibility outlasts cherry-picking.*
