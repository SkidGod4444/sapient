# SAPIENT on the Raspberry Pi

SAPIENT ships a native `aarch64-unknown-linux-gnu` binary with all the NEON
K-quant kernels (Q4_K W4A8 SDOT, Q6_K/Q5_K 16-lane NEON, Q8_0 SDOT) and the full
voice stack (`converse` works out of the box — the ARM release is built natively
with ALSA). This page is the Pi 4/5 playbook: setup, model choices for each RAM
size, thermal behaviour, and the measured numbers.

> Status: setup + tuning guidance are current; the measured-throughput table is
> being filled in on a reference Pi 5 (8 GB, active cooler) — entries marked TBD.

## Setup (Pi 4/5, 64-bit Raspberry Pi OS)

```bash
# One-command install (detects aarch64, grabs the native ARM binary):
curl -fsSL https://github.com/SkidGod4444/sapient/releases/latest/download/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

sapient models          # catalog
sapient chat qwen2.5-0.5b-q4          # first chat (downloads ~400 MB)
```

No Python, no Docker, no swap tuning needed for the models recommended below —
the engine caps the KV-cache allocation (`SAPIENT_CTX`, default 8192) and
memory-maps GGUFs bigger than ~80% of free RAM automatically.

## Which model for which Pi

| Device | RAM | Recommended | Notes |
|---|---|---|---|
| Pi 5 8 GB | 8 GB | `qwen2.5-1.5b-q4`, `llama-3.2-3b` (Q4_K_M) | 3B is the comfort ceiling |
| Pi 5 4 GB | 4 GB | `qwen2.5-1.5b-q4`; 3B with `SAPIENT_GGUF_QUANT=Q4_K_S` | the smaller `_S` file leaves headroom |
| Pi 4 4 GB | 4 GB | `qwen2.5-0.5b-q4`, `smollm2-360m-q4` | Cortex-A72 has no `dotprod`; slower NEON path |

**Low-RAM quant override (Phase 8.2):** `SAPIENT_GGUF_QUANT=Q4_K_S sapient pull
<model>` forces the smaller `_S` variant of a repo instead of the default
Q4_K_M — worth ~15% file size when a 4 GB board is tight. `SAPIENT_CTX=4096`
halves the KV-cache allocation if you don't need long prompts.

## Measured throughput (decode tok/s, 64-token greedy)

Reproduce with: `python3 scripts/bench_wgpu.py --backends cpu --model <alias> --tokens 64`

Reference device: **Pi 5 16 GB**, Raspberry Pi OS 64-bit, sapient v0.4.4
(measured 2026-07-03; sustained decode plateaus at ~75 °C on this board).

| Model | Pi 5 16 GB | vs v0.4.4 release | TTFT |
|---|---|---|---|
| qwen2.5-0.5b-q4 | **8.7 tok/s** | = (embed was never quantized) | 116 ms |
| llama-3.2-1b-q4 | **8.3 tok/s** | **6.4×** (was 1.3) | 119 ms |
| qwen2.5-1.5b-q4 | **6.7 tok/s** | **3.5×** (was 1.9) | 148 ms |
| llama-3.2-3b-q4 | **3.4 tok/s** | **4.3×** (was 0.8) | 303 ms |
| mistral-7b Q4_K_M (mmap) | ~0.6 (v0.3.9 measurement) | — | — |

The big jumps come from the Phase-8 embedding fix: the engine used to
dequantize the **entire** quantized embedding table every decode step
(`to_f32_cow` on a `[vocab, hidden]` GGUF table — ~0.8 GB of Q6_K dequant per
token for Llama-3.2-1B's tied 128k-vocab embedding). Embedding lookup is now
row-wise; only the tokens actually processed are dequantized. 1B-class chat on
a Pi 5 is genuinely interactive now, and even 3B is usable.

Sustained (6-minute soak, back-to-back 64-token generations, 0.5B): steady
**8.70 tok/s** with the SoC plateauing at 71–75 °C — no throttling on this
board, and the thermal governor (below) correctly stays inert at its default
80 °C threshold with zero throughput cost (measured on vs off: identical).

## Thermal behaviour (Phase 8.4)

Sustained decode pins all four cores; on a passive Pi the SoC reaches the 85 °C
firmware trip and every core hard-throttles — throughput collapses and
oscillates. SAPIENT now ships a **thermal governor**: it samples
`/sys/class/thermal` twice a second during inference and, from 80 °C, steps the
decode parallelism down one core at a time (never below half the cores),
restoring cores once the SoC cools below 70 °C. Backing off *before* the trip
point keeps the clocks up, so sustained throughput degrades gracefully instead
of collapsing.

- Watch it: `sapient -v chat …` logs a one-time
  `thermal: 80.x °C — backing decode off to 3/4 threads` warning.
- Tune it: `SAPIENT_THERMAL_HOT=75 SAPIENT_THERMAL_COOL=65 sapient chat …`
- Disable it: `SAPIENT_THERMAL=off` (e.g. when benchmarking peak, actively cooled).

Validated on a Pi 5 (16 GB, 2026-07-03):
- **Inert when cool** — at the default 80 °C threshold on a board that plateaus
  at ~75 °C, 6-minute soaks with the governor on vs off produced identical
  throughput (8.70 tok/s) and identical temperature curves. Zero cost.
- **Engages under load** — forcing the threshold inside the plateau
  (`SAPIENT_THERMAL_HOT=72`) stepped decode down within seconds of crossing
  72 °C: 8.70 → 8.23 tok/s, held stable for the rest of the soak. Only **−5%**
  throughput for the shed cores, because Pi decode is memory-latency-bound —
  which is exactly why backing off cores is nearly free in tokens/sec while
  cutting package power. On a passive board heading for the 85 °C trip, that
  trade is what prevents the hard-throttle collapse.

## The full voice loop on a Pi 5

```bash
# mic → Whisper STT → LLM → spoken reply (Kokoro TTS, CPU real-time class):
sapient converse qwen2.5-1.5b-q4 --stt whisper-tiny --speak
```

Notes:
- The ARM release binary includes audio I/O (ALSA); a USB mic/speaker or a
  HAT works out of the box. List devices with `arecord -l`.
- `whisper-tiny` keeps STT latency reasonable on the Pi; `whisper-base` is
  noticeably more accurate but slower.
- Replies stream sentence-by-sentence into TTS, so speech starts before the
  LLM finishes. Measured on the Pi 5 (v0.4.4 release binary, WAV-injected turn via
  `converse --input`): STT 3.5 s (whisper-tiny, 0.6× realtime) + LLM 2.1 s
  (0.5B) + TTS 5.3 s (Kokoro, RTF 2.34) ≈ **10.9 s per turn** — functional and
  correct end-to-end, not yet conversational-speed on Pi hardware (streaming
  overlap is roadmap Phase 10; the embedding fix above will also cut the LLM
  stage on the next release).

## Bigger models than RAM

GGUFs larger than ~80% of available RAM are memory-mapped automatically
(`--mmap` forces it): the OS pages weights on demand, trading throughput for
fitting at all — mistral-7B Q4_K_M (4.1 GB) runs on an 8 GB Pi 5 this way at
~0.6 tok/s. Practical, not comfortable; stay ≤3B for interactive use.
