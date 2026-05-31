#!/usr/bin/env python3
"""Head-to-head serving benchmark: SAPIENT vs Ollama vs vLLM.

All three expose an OpenAI-compatible `/v1/chat/completions` endpoint, so we
drive them through the identical client path on the identical hardware and the
identical model family (Qwen2.5-0.5B-Instruct) for a fair edge comparison.

Metrics (per engine):
  • warm TTFT (ms)            — time to first streamed token, model already loaded
  • decode throughput (tok/s) — sustained generation rate (est, ~4 chars/token)
  • concurrency throughput    — aggregate tok/s with N parallel requests
  • switch-back TTFT (ms)     — return to a previously-used model (resident vs reload)

Run the three servers first (see docs/BENCHMARKS.md), then:

    python3 scripts/bench_compete.py --out docs/assets

Produces PNG bar charts + a results.json in --out. Only engines that respond are
included; unreachable engines are skipped and noted.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict


# ── HTTP / streaming ───────────────────────────────────────────────────────────


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


@dataclass
class Stream:
    ttft_s: float
    total_s: float
    chars: int

    @property
    def est_tokens(self) -> float:
        return self.chars / 4.0

    @property
    def decode_tok_s(self) -> float:
        return self.est_tokens / max(self.total_s - self.ttft_s, 1e-9)


def stream_chat(url: str, model: str, content: str, max_tokens: int,
                temperature: float = 0.0) -> Stream:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    ttft = None
    chars = 0
    resp = urllib.request.urlopen(req, timeout=600)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue
        c = obj.get("choices", [{}])[0].get("delta", {}).get("content")
        if c:
            if ttft is None:
                ttft = time.perf_counter() - start
            chars += len(c)
    total = time.perf_counter() - start
    return Stream(ttft if ttft is not None else total, total, chars)


# ── Per-engine benchmark ─────────────────────────────────────────────────────


@dataclass
class EngineResult:
    name: str
    warm_ttft_ms: float = 0.0
    decode_tok_s: float = 0.0
    conc_tok_s: float = 0.0
    conc_p95_ms: float = 0.0
    switchback_ttft_ms: float | None = None
    note: str = ""


def bench_engine(name: str, url: str, model: str, max_tokens: int, conc: int) -> EngineResult:
    r = EngineResult(name=name)
    # Warm-up (also pays any lazy load).
    stream_chat(url, model, "Say hello.", max_tokens=8)
    # Warm single request.
    s = stream_chat(url, model, "Write a short paragraph about the ocean.", max_tokens)
    r.warm_ttft_ms = s.ttft_s * 1000
    r.decode_tok_s = s.decode_tok_s
    # Concurrency.
    t0 = time.perf_counter()
    outs: list[Stream] = []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(stream_chat, url, model, "Explain gravity in two sentences.", max_tokens)
                for _ in range(conc)]
        for f in as_completed(futs):
            outs.append(f.result())
    wall = time.perf_counter() - t0
    r.conc_tok_s = sum(o.est_tokens for o in outs) / max(wall, 1e-9)
    lat = sorted(o.total_s for o in outs)
    r.conc_p95_ms = lat[min(len(lat) - 1, int(len(lat) * 0.95))] * 1000
    return r


def bench_switchback(name: str, url: str, model_a: str, model_b: str) -> float | None:
    """A → B → A; return A's switch-back TTFT (resident = fast, reload = slow)."""
    try:
        stream_chat(url, model_a, "hi", max_tokens=4)   # ensure A loaded
        stream_chat(url, model_b, "hi", max_tokens=4)   # load B (may evict A)
        back = stream_chat(url, model_a, "hi", max_tokens=4)  # return to A
        return back.ttft_s * 1000
    except Exception as e:
        print(f"  [{name}] switch-back skipped: {e}")
        return None


# ── Plotting ───────────────────────────────────────────────────────────────────


def make_charts(results: list[EngineResult], out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    colors = {"SAPIENT": "#2dd4bf", "Ollama": "#a78bfa", "vLLM": "#f59e0b"}
    cs = [colors.get(n, "#888") for n in names]

    def bar(values, title, ylabel, fname, fmt="{:.0f}", logy=False):
        fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=140)
        bars = ax.bar(names, values, color=cs, width=0.6, edgecolor="#222", linewidth=0.6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, " " + fmt.format(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path)
        plt.close(fig)
        print(f"  wrote {path}")

    bar([r.warm_ttft_ms for r in results], "Warm TTFT (lower = better)",
        "ms", "bench_ttft.png", "{:.0f} ms", logy=True)
    bar([r.decode_tok_s for r in results], "Decode throughput (higher = better)",
        "tokens/s", "bench_throughput.png", "{:.1f}")
    bar([r.conc_tok_s for r in results], "Concurrent throughput (4 parallel)",
        "tokens/s", "bench_concurrency.png", "{:.1f}")

    sb = [r for r in results if r.switchback_ttft_ms is not None]
    if sb:
        fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=140)
        nm = [r.name for r in sb]
        vals = [r.switchback_ttft_ms for r in sb]
        bars = ax.bar(nm, vals, color=[colors.get(n, "#888") for n in nm],
                      width=0.6, edgecolor="#222", linewidth=0.6)
        ax.set_title("Model switch-back TTFT (lower = better)", fontsize=13, fontweight="bold")
        ax.set_ylabel("ms")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f" {v:.0f} ms",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, "bench_switchback.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  wrote {path}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sapient-url", default="http://localhost:11600")
    ap.add_argument("--sapient-model", default="openhorizon/qwen2.5-0.5b-q4")
    ap.add_argument("--sapient-model-b", default="openhorizon/qwen2.5-1.5b-q4")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="qwen2.5:0.5b")
    ap.add_argument("--ollama-model-b", default="phi3:mini")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="docs/assets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    engines = [
        ("SAPIENT", args.sapient_url, args.sapient_model, args.sapient_model_b),
        ("Ollama", args.ollama_url, args.ollama_model, args.ollama_model_b),
        ("vLLM", args.vllm_url, args.vllm_model, None),
    ]

    results: list[EngineResult] = []
    for name, url, model, model_b in engines:
        if not _reachable(url):
            print(f"[{name}] unreachable at {url} — skipping")
            continue
        print(f"[{name}] benchmarking {model} @ {url} …")
        r = bench_engine(name, url, model, args.max_tokens, args.concurrency)
        if model_b:
            r.switchback_ttft_ms = bench_switchback(name, url, model, model_b)
        results.append(r)
        print(f"  TTFT {r.warm_ttft_ms:.0f}ms · decode {r.decode_tok_s:.1f} tok/s · "
              f"conc {r.conc_tok_s:.1f} tok/s · switchback "
              f"{('%.0fms' % r.switchback_ttft_ms) if r.switchback_ttft_ms else 'n/a'}")

    if not results:
        raise SystemExit("no engines reachable")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")
    make_charts(results, args.out)


if __name__ == "__main__":
    main()
