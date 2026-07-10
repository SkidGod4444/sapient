#!/usr/bin/env python3
"""Generate the README benchmark charts from docs/assets/bench_v053.json.

Produces two PNGs in docs/assets/:
  - decode_throughput.png — decode tok/s, Metal (GPU) and CPU panels,
    SAPIENT vs llama.cpp vs Ollama, same GGUF quant, same machine
  - ttft.png              — warm time-to-first-token per engine

The data file is the committed record of the measured runs (hardware, date,
method in its `meta` block). Re-measure before regenerating: `sapient bench-llm
<model> --json`, `llama-bench -m <gguf> -p 0 -n 128 -o json`, and the Ollama
`/api/generate` timings, per the protocol in docs/BENCHMARKS.md.

Usage:
  python3 scripts/gen-benchmark-charts.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
DATA = os.path.join(ASSETS, "bench_v053.json")

with open(DATA) as f:
    data = json.load(f)
meta = data["meta"]

# Palette: categorical slots validated for CVD separation + lightness band
# (sub-3:1 contrast on aqua/yellow is relieved by direct value labels).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COLORS = {
    "SAPIENT": "#2a78d6",  # blue — slot 1
    "SAPIENT (CPU)": "#2a78d6",  # same entity, same hue — hatch is the discriminator
    "llama.cpp": "#1baf7a",  # aqua — slot 2
    "Ollama": "#eda100",  # yellow — slot 3
}
HATCHES = {"SAPIENT (CPU)": "///"}


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)


def grouped_bars(ax, models, engines, values, unit, label_fmt="{:.0f}", marks=None):
    n = len(engines)
    bar_w = 0.8 / n
    x = range(len(models))
    for i, eng in enumerate(engines):
        vals = [values[m][eng] for m in models]
        offs = [xi + (i - (n - 1) / 2) * bar_w for xi in x]
        bars = ax.bar(
            offs, vals, bar_w * 0.92, label=eng, color=COLORS[eng],
            edgecolor=SURFACE, linewidth=1.0, hatch=HATCHES.get(eng, ""),
        )
        for b, v, m in zip(bars, vals, models):
            suffix = (marks or {}).get((m, eng), "")
            ax.text(
                b.get_x() + b.get_width() / 2, v, label_fmt.format(v) + suffix,
                ha="center", va="bottom", fontsize=8.5, color=INK,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylabel(unit, fontsize=9.5, color=INK_2)
    style_axis(ax)


# ── Chart 1: decode throughput, Metal + CPU panels ────────────────────────
decode = data["decode"]
models = list(decode["metal"].keys())

fig, (ax_gpu, ax_cpu) = plt.subplots(
    1, 2, figsize=(10.5, 4.6), sharey=True, facecolor=SURFACE
)
grouped_bars(ax_gpu, models, ["SAPIENT", "llama.cpp", "Ollama"], decode["metal"],
             "decode tokens / sec",
             marks={("Llama-3.2-1B Q4_K_M", "Ollama"): "†"})
ax_gpu.set_title("Apple Metal (GPU)", fontsize=11, color=INK, pad=10)
grouped_bars(ax_cpu, models, ["SAPIENT", "llama.cpp"], decode["cpu"], "")
ax_cpu.set_title("CPU (4 threads)", fontsize=11, color=INK, pad=10)
ax_gpu.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)
ax_cpu.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)

fig.suptitle(
    f"Decode throughput — same Q4_K_M GGUF, {meta['hardware']}   (higher is better)",
    fontsize=12.5, color=INK, fontweight="bold", y=1.00,
)
fig.text(
    0.01, 0.005,
    f"SAPIENT v{meta['sapient_version']} (-metal / CPU builds) · {meta['llamacpp']} · "
    f"Ollama {meta['ollama']} · {meta['date']} · {meta['method_decode']} "
    f"† Ollama's default llama3.2:1b tag ships Q8_0, not Q4_K_M.",
    fontsize=7, color=MUTED,
)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(os.path.join(ASSETS, "decode_throughput.png"), dpi=140,
            facecolor=SURFACE, bbox_inches="tight")
print("wrote docs/assets/decode_throughput.png")

# ── Chart 2: warm TTFT ────────────────────────────────────────────────────
ttft = data["ttft"]
fig2, ax2 = plt.subplots(figsize=(8.2, 4.4), facecolor=SURFACE)
grouped_bars(ax2, list(ttft.keys()), ["SAPIENT", "SAPIENT (CPU)", "Ollama"], ttft,
             "time to first token (ms), warm", label_fmt="{:.0f} ms")
ax2.set_title(
    f"Time to first token — warm model, {meta['hardware']}   (lower is better)",
    fontsize=12.5, color=INK, fontweight="bold", pad=12,
)
fig2.text(
    0.01, 0.005,
    f"SAPIENT = streamed first token (bench-llm, warm-run mean) · Ollama = total − eval "
    f"duration proxy · llama.cpp omitted (llama-bench reports no TTFT) · {meta['date']}",
    fontsize=7, color=MUTED,
)
fig2.tight_layout(rect=(0, 0.04, 1, 1))
fig2.savefig(os.path.join(ASSETS, "ttft.png"), dpi=140,
             facecolor=SURFACE, bbox_inches="tight")
print("wrote docs/assets/ttft.png")
