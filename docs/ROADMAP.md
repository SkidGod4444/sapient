# 🗺️ SAPIENT Roadmap — Huge Models on Small Devices

> **Mission:** run models that "shouldn't fit" on the hardware people actually own —
> laptops, Raspberry Pis, phones — with a one-line install and a great UX.
>
> The engine work below (quantization, mmap, SIMD, GPU offload) is the *price of entry*
> — llama.cpp already does it well. Our **moat** is the layer on top: pure-Rust
> portability, curated registry, modern CLI, and edge-specific automation
> (auto-pick quantization for available RAM, auto CPU/GPU offload, single static binary).

## Where we are (v0.1.11)
- ✅ Correct CPU + Metal inference for Phi & Llama/Qwen families (F16/BF16 safetensors).
- ✅ Curated registry, modern CLI, self-update, published to crates.io.
- ❌ **Weights are materialized at 2–4 bytes/param** — GGUF expands to F32, safetensors stays F16.
  A 7B model needs ~14–28 GB RAM. We literally cannot fit big models yet.
- ❌ No quantized compute dtype/kernels; the Pipeline can't load GGUF.

## Guiding principles
1. **One PR/phase → one release.** Ship gradually; never a big-bang.
2. **Correctness is a gate.** Every phase adds/keeps a golden-output test (greedy decode of a known model → exact tokens). No release regresses output.
3. **Measure RAM and tok/s** every phase; numbers go in the release notes.
4. **CPU core first, accelerators second.** The quantized CPU engine is the shared foundation for *all four* targets.

---

## Phase 0 — Spike & de-risk  → `v0.1.x` (no public release)
Narrow proof before committing to the full build.
- Load one `Q4_0` GGUF, keep blocks quantized in memory (no F32 expansion).
- A single quantized `matmul_nt` (dequant-in-loop) for the linear layers only.
- Run a tiny model end-to-end; measure RAM (should ≈ file size) and tok/s.
- **Exit criteria:** a Q4_0 linear path produces correct logits vs the F32 reference within tolerance.

## Phase 1 — Quantized CPU engine (foundation for every target)  → **`v0.2.0`**
The unlock. Generic CPU (x86 + ARM), no SIMD yet.
- `DType`: add `Q4_0`, `Q8_0` (then `Q4_K`, `Q5_0`) storing raw quant blocks. → `dtype.rs`, `buffer.rs`, `tensor.rs`
- Quantized `matmul_nt` / attention paths that dequantize per-block inside the dot loop — never materialize F32 weights. → `kernels/`
- GGUF loader keeps tensors quantized; **wire `from_gguf` into the Pipeline** so GGUF models run.
- **mmap, zero-copy:** tensors reference the mmap'd file → **RAM ≈ file size.**
- Auto-tokenizer fallback for GGUF repos that omit tokenizer files.
- **Success metric:** run a 7B `Q4_0` GGUF in **< 5 GB RAM** on a laptop, correct output.
- Registry: add a few GGUF quant entries (e.g. TinyLlama/Qwen Q4).

## Phase 2 — CPU speed: SIMD + threading  → **`v0.2.x`**
Make quantized inference *usable*, benefiting laptops, Pi, and phones alike.
- SIMD quantized dot-products: **NEON** (Apple/ARM/Pi) + **AVX2** (x86), behind `cfg`.
- Q8 activation quantization for fast integer dot-products (llama.cpp pattern).
- `rayon` threading across rows/heads; tune the buffer pool.
- KV-cache quantization (Q8) → longer context in less RAM.
- **Success metric:** ≥ 5–10× tok/s vs the Phase-1 scalar path on the same model.

## Phase 3 — Apple Silicon / Metal  → **`v0.3.0`**
Builds on the MLX work already landed.
- Quantized matmul on MLX (or Metal kernels); exploit unified memory.
- Native MLX attention + RoPE (remove the current CPU fallback for those ops).
- Auto CPU/GPU offload by model size & available memory.
- **Success metric:** a 7B–13B Q4 model interactive (> ~15 tok/s) on an M-series laptop.

## Phase 4 — Raspberry Pi / small ARM SBC  → **`v0.3.x`**
The hardest, most differentiating CPU target (2–8 GB RAM).
- Bigger-than-RAM support via mmap paging / per-layer streaming.
- Low-RAM tuning: minimal activation buffers, optional `Q4_K_S`.
- `armv7`/`aarch64` validation; document Pi 4/5 setup.
- **Success metric:** run a 3B Q4 model on a 4 GB Pi 5 without OOM.

## Phase 5 — Phones (iOS / Android)  → **`v0.4.0`**
Most constrained, biggest "wow".
- Library packaging: stable C FFI / UniFFI bindings; static lib for mobile.
- Mobile mmap + thermal/throttle-aware scheduling.
- Sample iOS (Swift) and Android (Kotlin/JNI) apps.
- **Success metric:** a 1–3B Q4 model running on-device in a demo app.

---

## Cross-cutting workstreams (continuous)
- **Correctness harness:** golden-token tests per architecture; CI gate.
- **Bench suite:** RAM + tok/s + time-to-first-token across targets; tracked over time.
- **UX automation:** `sapient` auto-selects a quantization that fits available RAM; `--mem` budget flag; clear "won't fit, try Q4" guidance.
- **Docs:** keep `PROJECT_GUIDE.md` and the README in sync each release.

## Definition of "leading the market"
Match llama.cpp on quantized edge inference (Phases 1–3), then win on:
**install in one line, run any curated model in one command, auto-fit the hardware, pure-Rust everywhere — including phones.**
