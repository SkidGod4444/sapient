# Non-AR Duplex-Codec De-Risking Spike

> Status: **spike in progress** — synthesis-cost gate measured (Mac/CPU);
> streaming-decoder + look-ahead gate is the next step. Backs Paper 1
> (`~/Documents/Importants/My Research Papers/paper1_duplex_codec.tex`, §7).

## The question

Paper 1's thesis: a CPU-real-time, GPU-less, cross-platform **full-duplex** voice
agent is feasible only with a **non-autoregressive (non-AR)** synthesizer, because
an autoregressive neural codec cannot hold the duplex latency budget on a CPU
(measured: Orpheus AR RTF 5.5 on Metal / 17.4 on CPU vs Kokoro non-AR RTF ~0.76
on CPU). The spike must answer, before any codec is trained:

> Can the non-AR synthesizer be driven in **streaming chunks** small enough to
> begin speaking within the duplex budget (Moshi ≈ 200 ms; GO threshold ≤ 3× =
> 600 ms), while sustaining RTF < 1?

## Harness

`crates/sapient-generate/examples/duplex_spike.rs` (run:
`cargo run --release -p sapient-generate --example duplex_spike`). It drives the
**shipped Kokoro-82M** non-AR decoder over chunks of increasing size and reports
per-chunk synthesis latency, RTF, and a GO/NO-GO verdict. It also defines the
`StreamingDuplexSynth` interface that the trained streaming decoder must satisfy.

## Measured results (Apple M4, CPU)

Per-chunk synthesis (best of 3), whole-utterance Kokoro:

| chunk      | audio (s) | wall (ms) | RTF   |
|------------|-----------|-----------|-------|
| 1 word     | 1.40      | 811.7     | 0.580 |
| 3 words    | 1.75      | 1070.4    | 0.612 |
| 6 words    | 2.25      | 1428.9    | 0.635 |
| 12 words   | 3.75      | 2417.5    | 0.645 |
| ~24 words  | 8.58      | 5630.4    | 0.657 |

Per-stage breakdown for "Hello there friend." (1.75 s audio,
`SAPIENT_KOKORO_TIMING=1`):

| stage         | ms    | note |
|---------------|-------|------|
| albert        | 50.7  | backbone (amortizable) |
| prosody       | 70.9  | backbone (amortizable) |
| f0/n          | 68.6  | backbone (amortizable) |
| text_encoder  | 17.3  | backbone (amortizable) |
| **decoder**   | **871.0** | **per-chunk (ISTFTNet conv stack)** |
| └ src+convs   | 365   | upsampling convs |
| └ resblocks   | 420   | AdaIN residual blocks (the hot loop) |
| └ post+istft  | 18    | |

## Findings (the spike's actual value)

1. **RTF is sustainable** (~0.58–0.66 < 1.0) — non-AR synthesis is faster than
   real time, confirming the AR-ceiling escape.
2. **Naive whole-utterance chunking is NO-GO.** Kokoro pads short input to a
   minimum-length utterance, so even "1 word" produces 1.4 s of audio at an
   811 ms floor — above the 600 ms budget. You cannot get a cheap 0.3 s chunk
   from the current API.
3. **The bottleneck is the DECODER, not the backbone** (this reverses the
   initial hypothesis). The decoder is **~80%** of the cost (871 of ~1080 ms);
   ALBERT + prosody + f0/n + text_encoder are only **~20%** (~207 ms) and are
   **per-utterance, hence amortizable** across a stream.
4. **Decoder cost is ~linear in chunk audio length:** subtracting the ~207 ms
   fixed backbone, the decoder runs at **~430–630 ms per second of audio**. A
   0.3 s chunk therefore extrapolates to **~165 ms** of decoder work — *within*
   the 600 ms budget, and approaching Moshi's 200 ms.

## Device-class result: Raspberry Pi 5 (Cortex-A76)

Same sentence ("Hello there friend.", 1.75 s audio), `sapient` 0.4.4 release
binary, `SAPIENT_KOKORO_TIMING=1`:

| stage         | M4 (ms) | Pi 5 (ms) | Pi/M4 |
|---------------|---------|-----------|-------|
| albert        | 50.7    | 184.0     | 3.6×  |
| prosody       | 70.9    | 146.5     | 2.1×  |
| f0/n          | 68.6    | 158.8     | 2.3×  |
| text_encoder  | 17.3    | 40.1      | 2.3×  |
| **decoder**   | 871.0   | **3410.3**| 3.9×  |
| synth total   | ~1080   | ~3940     | 3.6×  |
| **synth RTF** | **0.62**| **2.25**  | —     |

**The "CPU-real-time" claim is device-class-dependent.** On a laptop-class M4 CPU
non-AR synthesis is real-time (RTF 0.62); on a Pi 5 it is **2.25× too slow** even
for whole-utterance non-AR synthesis. The decoder dominates on both (87% on Pi).
Decoder cost: ~498 ms/s of audio on M4, **~1949 ms/s on Pi**. A 0.3 s streamed
chunk with the backbone amortized extrapolates to ~165 ms on M4 (within budget)
but **~585 ms on the Pi** — at the very edge of the 600 ms budget *before* adding
look-ahead, and before the STT+LLM stages contend for the same 4 cores. So on the
Pi, streaming-duplex non-AR is **borderline/NO-GO without decoder optimization**.

## Refined verdict

- **Whole-utterance chunking:** NO-GO on every device.
- **Streaming decoder on small (~0.3 s) chunks with amortized backbone:**
  **device-class-dependent.** On laptop-class CPU (M4): **PLAUSIBLE GO** by
  extrapolation (~165 ms/chunk). On a Pi 5: **borderline/NO-GO** (~585 ms/chunk
  before look-ahead and before STT+LLM contention) — needs decoder optimization
  first. Both pending the two unmeasured quantities below.

## Next steps (to convert PLAUSIBLE → measured GO/NO-GO)

1. **Implement `StreamingKokoroDecoder: StreamingDuplexSynth`** that runs the
   ISTFTNet decoder on a rolling latent window with a fixed right-context margin
   (mirror `speak.rs::emit_stable` / `STREAM_MARGIN_FRAMES`), amortizing
   ALBERT/prosody/text_encoder across the stream. This requires exposing a
   decoder-only entry point in `kokoro/decoder.rs` (currently the pipeline is
   coupled in `KokoroModel::synthesize_ids`).
2. **Measure decoder-only per-chunk latency** for 0.2–0.4 s chunks (validate the
   ~165 ms extrapolation directly).
3. **Measure minimum stable look-ahead**: append right-context and find where the
   emitted prefix stops changing (the decoder's conv receptive field). The full
   latency is `decoder_per_chunk + look_ahead`; GO iff ≤ 600 ms (stretch 200 ms).
4. **Optimize the resblocks** (420 ms hot loop): they are already rayon-parallel;
   next levers are SIMD on the AdaIN/Snake path or a smaller decoder.
5. **Cross-backend + Pi:** rerun on Raspberry Pi 5 (CPU) and via WGSL/WebGPU to
   fill Paper 1's cross-backend latency/energy table.

## Kill criterion

If, after amortizing the backbone, decoder-only per-chunk latency + minimum
stable look-ahead for a ≤0.4 s chunk exceeds ~600 ms on commodity CPU, the
streaming non-AR duplex path is not viable and the fallback is a tightened
cascade (which the verified evidence does **not** establish is inferior).
