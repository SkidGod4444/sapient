#!/usr/bin/env python3
"""SAPIENT `serve` benchmark harness.

Measures the serving features that set `sapient serve` apart from Ollama:

  1. TTFT + decode latency           — single streamed request
  2. Model switch-back (cache hit)   — Ollama cold-reloads on every switch; we
                                        keep N models resident (LRU), so coming
                                        back to a recent model skips reload
  3. Prefix / prompt KV caching      — a multi-turn chat that extends a shared
                                        prefix only re-prefills the new suffix
  4. Concurrency scaling             — N parallel requests: aggregate throughput
                                        + p50/p95 latency under admission control

Talks to a *running* server over the OpenAI-compatible HTTP API (stdlib only —
no third-party deps). Start one first, e.g.:

    sapient serve openhorizon/qwen2.5-0.5b-q4 --backend cpu --port 11500
    # (no preload also works — models load on first request)

then:

    python3 scripts/bench_serve.py --url http://localhost:11500 \
        --models openhorizon/qwen2.5-0.5b-q4,openhorizon/qwen2.5-1.5b-q4

Use `--spawn ./target/release/sapient` to let the harness start/stop the server
itself (handy for speculative runs via `--serve-args "--speculative"`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


# ── HTTP helpers ───────────────────────────────────────────────────────────────


def _post(url: str, path: str, payload: dict, timeout: float = 600.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def wait_for_health(url: str, timeout: float = 60.0) -> bool:
    """Block until GET /v1/health returns 200 (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/v1/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


@dataclass
class StreamResult:
    ttft_s: float            # time to first content token
    total_s: float           # time until the stream closed
    chunks: int              # number of content deltas received
    chars: int               # total characters streamed
    text: str = ""

    @property
    def decode_s(self) -> float:
        return max(self.total_s - self.ttft_s, 1e-9)

    @property
    def est_tokens(self) -> float:
        # ~4 chars/token is a reasonable English estimate (no client tokenizer).
        return self.chars / 4.0

    @property
    def est_tok_s(self) -> float:
        return self.est_tokens / self.decode_s


def stream_chat(url: str, model: str | None, messages: list[dict],
                max_tokens: int, temperature: float = 0.0) -> StreamResult:
    """Fire a streaming chat completion; measure TTFT and total time."""
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        payload["model"] = model

    start = time.perf_counter()
    ttft = None
    chunks = 0
    chars = 0
    text_parts: list[str] = []

    resp = _post(url, "/v1/chat/completions", payload)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if body == "[DONE]":
            break
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue
        delta = obj.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            if ttft is None:
                ttft = time.perf_counter() - start
            chunks += 1
            chars += len(content)
            text_parts.append(content)
    total = time.perf_counter() - start
    if ttft is None:
        ttft = total
    return StreamResult(ttft, total, chunks, chars, "".join(text_parts))


# ── Benchmarks ───────────────────────────────────────────────────────────────


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)

    def head(self, title: str):
        self.lines.append("")
        self.lines.append(f"━━ {title} " + "━" * max(0, 60 - len(title)))

    def row(self, label: str, value: str):
        self.lines.append(f"  {label:<34} {value}")

    def render(self) -> str:
        return "\n".join(self.lines)


def bench_single(url: str, model: str, max_tokens: int, rep: Report):
    rep.head(f"1. Single-request latency · {model}")
    msgs = [{"role": "user", "content": "Write a short paragraph about the sea."}]
    # Warm the model first (cold load excluded from the latency number).
    cold = stream_chat(url, model, msgs, max_tokens=8)
    warm = stream_chat(url, model, msgs, max_tokens=max_tokens)
    rep.row("cold first-request TTFT", f"{cold.ttft_s*1000:8.0f} ms  (includes load)")
    rep.row("warm TTFT", f"{warm.ttft_s*1000:8.0f} ms")
    rep.row("decode throughput (est)", f"{warm.est_tok_s:8.1f} tok/s  (~{warm.est_tokens:.0f} tok)")
    rep.row("total wall time", f"{warm.total_s*1000:8.0f} ms")


def bench_switch(url: str, model_a: str, model_b: str, rep: Report):
    rep.head("2. Model switch-back (LRU cache hit vs reload)")
    short = [{"role": "user", "content": "Say hi."}]
    # Ensure A then B are both resident (default --max-models 3 keeps both).
    a_cold = stream_chat(url, model_a, short, max_tokens=8)
    b_cold = stream_chat(url, model_b, short, max_tokens=8)
    # Switch back to A — should be a cache hit (no download / re-quant / reload).
    a_warm = stream_chat(url, model_a, short, max_tokens=8)
    rep.row(f"{model_a} first load TTFT", f"{a_cold.ttft_s*1000:8.0f} ms")
    rep.row(f"{model_b} first load TTFT", f"{b_cold.ttft_s*1000:8.0f} ms")
    rep.row(f"{model_a} switch-back TTFT", f"{a_warm.ttft_s*1000:8.0f} ms  (cache hit)")
    if a_warm.ttft_s > 0:
        rep.row("switch-back speedup", f"{a_cold.ttft_s / a_warm.ttft_s:8.1f}×  vs its own cold load")


def bench_prefix_cache(url: str, model: str, rep: Report):
    rep.head("3. Prefix / prompt KV caching (multi-turn)")
    # A big shared system prompt — the expensive part to prefill.
    big_system = ("You are a meticulous assistant. " * 120).strip()
    sys_msg = {"role": "system", "content": big_system}
    turn1 = [sys_msg, {"role": "user", "content": "Name one color."}]
    r1 = stream_chat(url, model, turn1, max_tokens=8)
    # Turn 2 extends the same prefix (system + turn1 + assistant1 + user2).
    turn2 = turn1 + [
        {"role": "assistant", "content": r1.text or "Blue."},
        {"role": "user", "content": "Now name one animal."},
    ]
    r2 = stream_chat(url, model, turn2, max_tokens=8)
    rep.row("turn 1 TTFT (full prefill)", f"{r1.ttft_s*1000:8.0f} ms")
    rep.row("turn 2 TTFT (prefix reused)", f"{r2.ttft_s*1000:8.0f} ms")
    if r2.ttft_s > 0:
        rep.row("multi-turn TTFT speedup", f"{r1.ttft_s / r2.ttft_s:8.1f}×")


def bench_concurrency(url: str, model: str, n: int, max_tokens: int, rep: Report):
    rep.head(f"4. Concurrency · {n} parallel requests · {model}")
    msgs = [{"role": "user", "content": "Explain gravity in two sentences."}]
    stream_chat(url, model, msgs, max_tokens=8)  # warm
    start = time.perf_counter()
    results: list[StreamResult] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(stream_chat, url, model, msgs, max_tokens) for _ in range(n)]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - start
    lat = sorted(r.total_s for r in results)
    total_tok = sum(r.est_tokens for r in results)
    rep.row("requests completed", f"{len(results)}")
    rep.row("wall time (all)", f"{wall*1000:8.0f} ms")
    rep.row("aggregate throughput (est)", f"{total_tok/max(wall,1e-9):8.1f} tok/s")
    rep.row("latency p50 / p95", f"{lat[len(lat)//2]*1000:.0f} / {lat[min(len(lat)-1, int(len(lat)*0.95))]*1000:.0f} ms")
    rep.row("TTFT p50", f"{sorted(r.ttft_s for r in results)[len(results)//2]*1000:8.0f} ms")


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Benchmark `sapient serve`.")
    ap.add_argument("--url", default="http://localhost:11500")
    ap.add_argument("--models", default="openhorizon/qwen2.5-0.5b-q4",
                    help="comma-separated model aliases (>=2 enables the switch-back test)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--spawn", default=None,
                    help="path to the sapient binary; if set, start/stop the server here")
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--serve-args", default="", help="extra args passed to `sapient serve`")
    ap.add_argument("--json", default=None, help="also write raw report text to this path")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        sys.exit("need at least one --models entry")

    proc = None
    if args.spawn:
        cmd = [args.spawn, "serve", "--port", str(args.port), "--backend", args.backend]
        cmd += args.serve_args.split()
        url = f"http://localhost:{args.port}"
        print(f"[harness] spawning: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        url = args.url

    try:
        if not wait_for_health(url, timeout=30):
            sys.exit(f"server at {url} did not become healthy")

        rep = Report()
        rep.lines.append("")
        rep.lines.append("⚡ SAPIENT serve — benchmark report")
        rep.row("endpoint", url)
        rep.row("models", ", ".join(models))
        rep.row("backend / max_tokens / concurrency",
                f"{args.backend} / {args.max_tokens} / {args.concurrency}")

        bench_single(url, models[0], args.max_tokens, rep)
        if len(models) >= 2:
            bench_switch(url, models[0], models[1], rep)
        else:
            rep.head("2. Model switch-back (skipped — pass >=2 --models)")
        bench_prefix_cache(url, models[0], rep)
        bench_concurrency(url, models[0], args.concurrency, args.max_tokens, rep)

        out = rep.render()
        print(out)
        if args.json:
            with open(args.json, "w") as f:
                f.write(out + "\n")
            print(f"\n[harness] report written to {args.json}")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
