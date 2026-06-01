#!/usr/bin/env python3
"""Cross-platform GPU-vs-CPU benchmark for SAPIENT.

Measures time-to-first-token (TTFT) and decode throughput (tok/s) for the same
model across backends — `cpu`, `wgpu` (Vulkan/DX12/Metal), and `metal` (Apple MLX,
macOS only) — so you can see what your GPU buys you on *your* machine, whatever the
vendor (Intel / AMD / Nvidia / Apple).

It drives the `sapient serve` OpenAI-compatible server over HTTP with streaming, so
the numbers reflect the real inference path. Uses only the Python standard library
(no pip installs) for the measurement; matplotlib is optional, only for the chart.

Usage:
    # Build a GPU-capable binary first (once):
    #   cargo build --release -p sapient-cli --features wgpu
    python3 scripts/bench_wgpu.py
    python3 scripts/bench_wgpu.py --model openhorizon/qwen2.5-1.5b --tokens 128
    python3 scripts/bench_wgpu.py --backends cpu,wgpu --bin ./target/release/sapient
    python3 scripts/bench_wgpu.py --chart bench_wgpu.png

Notes:
  * The first request to `serve` lazily downloads + loads the model; the script does
    a warmup request per backend before timing, so load time is excluded.
  * `wgpu` requires a binary built with `--features wgpu`. On Linux you also need the
    Vulkan loader (`libvulkan1`) and a GPU driver installed at runtime.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


def find_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in ("./target/release/sapient", "./target/debug/sapient"):
        if shutil.which(cand) or _is_file(cand):
            return cand
    onpath = shutil.which("sapient")
    if onpath:
        return onpath
    sys.exit(
        "could not find the `sapient` binary — build it with\n"
        "    cargo build --release -p sapient-cli --features wgpu\n"
        "or pass --bin <path>."
    )


def _is_file(p: str) -> bool:
    import os

    return os.path.isfile(p)


@dataclass
class Result:
    backend: str
    ttft_ms: float
    decode_toks: int
    decode_s: float

    @property
    def tok_s(self) -> float:
        return self.decode_toks / self.decode_s if self.decode_s > 0 else 0.0


def wait_health(port: int, timeout_s: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}/v1/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    return False


def stream_chat(port: int, model: str, prompt: str, max_tokens: int) -> tuple[float, int, float]:
    """Returns (ttft_seconds, tokens_decoded, decode_seconds)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    first_tok_t: float | None = None
    last_tok_t = start
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                now = time.perf_counter()
                if first_tok_t is None:
                    first_tok_t = now
                n_tokens += 1
                last_tok_t = now
    if first_tok_t is None:
        return (0.0, 0, 0.0)
    ttft = first_tok_t - start
    decode_s = max(last_tok_t - first_tok_t, 1e-9)
    # n_tokens counts streamed chunks (≈ tokens); the first chunk is the TTFT one,
    # so tokens decoded *after* first is n_tokens-1 over the decode window.
    return (ttft, max(n_tokens - 1, 0), decode_s)


def run_backend(binary: str, backend: str, model: str, prompt: str, tokens: int, port: int) -> Result | None:
    cmd = [binary, "serve", "--backend", backend, "--port", str(port)]
    print(f"\n=== {backend} ===\n$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_health(port):
            print(f"  server did not become healthy — skipping {backend}")
            return None
        print("  warmup (downloads/loads the model on first request)…")
        stream_chat(port, model, prompt, max_tokens=8)  # warmup excludes load time
        print("  timing…")
        ttft, n, decode_s = stream_chat(port, model, prompt, max_tokens=tokens)
        if n == 0:
            print(f"  no tokens produced — skipping {backend}")
            return None
        r = Result(backend, ttft * 1000.0, n, decode_s)
        print(f"  TTFT {r.ttft_ms:7.1f} ms   decode {r.tok_s:6.1f} tok/s  ({n} tok in {decode_s:.2f}s)")
        return r
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def print_table(results: list[Result]) -> None:
    print("\n" + "=" * 56)
    print(f"{'backend':<10}{'TTFT (ms)':>14}{'decode (tok/s)':>18}")
    print("-" * 56)
    cpu = next((r for r in results if r.backend == "cpu"), None)
    for r in results:
        speedup = ""
        if cpu and cpu.tok_s > 0 and r.backend != "cpu":
            speedup = f"  ({r.tok_s / cpu.tok_s:.2f}× CPU)"
        print(f"{r.backend:<10}{r.ttft_ms:>14.1f}{r.tok_s:>18.1f}{speedup}")
    print("=" * 56)


def make_chart(results: list[Result], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed — skipping chart ({path})")
        return
    backends = [r.backend for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(backends, [r.tok_s for r in results], color="#4C9AFF")
    ax1.set_title("Decode throughput (tok/s) — higher is better")
    ax1.set_ylabel("tok/s")
    ax2.bar(backends, [r.ttft_ms for r in results], color="#FF8B00")
    ax2.set_title("Time to first token (ms) — lower is better")
    ax2.set_ylabel("ms")
    fig.suptitle(f"SAPIENT backends — {platform.system()} {platform.machine()}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"chart written to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openhorizon/qwen2.5-0.5b", help="model alias or path")
    ap.add_argument("--prompt", default="Explain how a transformer language model works, in a few sentences.")
    ap.add_argument("--tokens", type=int, default=96, help="tokens to decode when timing")
    ap.add_argument("--bin", default=None, help="path to the sapient binary")
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument(
        "--backends",
        default=None,
        help="comma-separated (default: cpu,wgpu plus metal on macOS)",
    )
    ap.add_argument("--chart", default=None, help="write a PNG bar chart to this path")
    args = ap.parse_args()

    binary = find_binary(args.bin)
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    else:
        backends = ["cpu", "wgpu"]
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            backends.append("metal")

    print(f"SAPIENT backend benchmark — {platform.system()} {platform.machine()}")
    print(f"binary : {binary}")
    print(f"model  : {args.model}")
    print(f"tokens : {args.tokens}   prompt: {args.prompt!r}")
    print(f"backends: {', '.join(backends)}")

    results: list[Result] = []
    for i, b in enumerate(backends):
        r = run_backend(binary, b, args.model, args.prompt, args.tokens, args.port + i)
        if r:
            results.append(r)

    if not results:
        sys.exit("no backend produced a result")
    print_table(results)
    if args.chart:
        make_chart(results, args.chart)


if __name__ == "__main__":
    main()
