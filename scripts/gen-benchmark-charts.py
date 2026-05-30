#!/usr/bin/env python3
"""Generate benchmark comparison charts from results/v033/summary.json.

Produces two PNGs in docs/assets/:
  - decode_throughput.png  — grouped bar chart of decode tok/s per engine/model
  - sapient_speedup.png    — SAPIENT CPU vs Metal speedup per model

Usage:
  python3 scripts/gen-benchmark-charts.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "results", "v033", "summary.json")
ASSETS = os.path.join(ROOT, "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)

with open(SUMMARY) as f:
    data = json.load(f)

results = data["results"]
meta = data["meta"]

# Brand palette
COLORS = {
    "SAPIENT Metal": "#7d7dbd",  # SAPIENT purple
    "SAPIENT CPU": "#b8b8d8",  # light purple
    "Ollama Metal": "#d97757",  # terracotta
    "mlx-lm Metal": "#ffc107",  # amber
}

# ── Chart 1: decode throughput, grouped by model ──────────────────────────────
models = ["0.5b", "1.5b"]
engines = ["SAPIENT Metal", "mlx-lm Metal", "Ollama Metal", "SAPIENT CPU"]

fig, ax = plt.subplots(figsize=(10, 5.5))
n_eng = len(engines)
bar_w = 0.19
x = range(len(models))

for i, eng in enumerate(engines):
    vals = []
    for m in models:
        v = next(
            (r["decode_tps"] for r in results if r["model"] == m and r["engine"] == eng),
            0,
        )
        vals.append(v)
    offsets = [xi + (i - (n_eng - 1) / 2) * bar_w for xi in x]
    bars = ax.bar(offsets, vals, bar_w, label=eng, color=COLORS[eng], edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels([f"Qwen2.5-{m.upper()}" for m in models], fontsize=12)
ax.set_ylabel("Decode throughput (tokens/sec)", fontsize=12)
ax.set_title(
    f"Decode Throughput — {meta['hardware']}, {meta['ram_gb']} GB RAM\n"
    f"SAPIENT v{meta['sapient_version']} · higher is better",
    fontsize=13,
    fontweight="bold",
)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "decode_throughput.png"), dpi=140)
print("wrote docs/assets/decode_throughput.png")

# ── Chart 2: SAPIENT CPU → Metal speedup ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
cpu_vals = [next(r["decode_tps"] for r in results if r["model"] == m and r["engine"] == "SAPIENT CPU") for m in models]
metal_vals = [next(r["decode_tps"] for r in results if r["model"] == m and r["engine"] == "SAPIENT Metal") for m in models]

x2 = range(len(models))
w = 0.35
b1 = ax2.bar([xi - w / 2 for xi in x2], cpu_vals, w, label="CPU (NEON)", color=COLORS["SAPIENT CPU"], edgecolor="white")
b2 = ax2.bar([xi + w / 2 for xi in x2], metal_vals, w, label="Metal (MLX engine)", color=COLORS["SAPIENT Metal"], edgecolor="white")

for bars in (b1, b2):
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

for i, m in enumerate(models):
    speedup = metal_vals[i] / cpu_vals[i]
    ax2.text(i, max(metal_vals[i], cpu_vals[i]) + 12, f"{speedup:.1f}× faster", ha="center", fontsize=11, fontweight="bold", color=COLORS["SAPIENT Metal"])

ax2.set_xticks(list(x2))
ax2.set_xticklabels([f"Qwen2.5-{m.upper()}" for m in models], fontsize=12)
ax2.set_ylabel("Decode throughput (tokens/sec)", fontsize=12)
ax2.set_title(
    f"SAPIENT CPU → Metal speedup (v{meta['sapient_version']})\n{meta['hardware']} · {meta['os']}",
    fontsize=13,
    fontweight="bold",
)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(axis="y", alpha=0.25, linestyle="--")
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, max(metal_vals) * 1.25)
fig2.tight_layout()
fig2.savefig(os.path.join(ASSETS, "sapient_speedup.png"), dpi=140)
print("wrote docs/assets/sapient_speedup.png")
