#!/usr/bin/env bash
# Phase 7.6 cross-vendor GPU benchmark — run this on a machine with an Intel Arc,
# AMD Radeon, or Nvidia card (Linux; Windows users: see docs/BENCHMARKS.md for the
# equivalent PowerShell steps). Produces `bench-7_6-<gpu>.txt` — paste it into the
# Phase 7 PR or an issue.
#
# What it does:
#   1. Detects the GPU + Vulkan driver.
#   2. Builds (or reuses) the wgpu-enabled binary from this checkout.
#   3. Captures the engine's resident-VRAM line (verifies "VRAM ≈ GGUF file size"
#      and that all matrices load quantized).
#   4. Greedy-decode correctness probe (must answer "Paris").
#   5. `bench_wgpu.py` cpu-vs-wgpu on two models:
#        - openhorizon/smollm2-360m-q4  (Q8_0 — in-shader Q8_0 path)
#        - openhorizon/qwen2.5-1.5b-q4  (Q4_K_M — Q4_K + Q6_K paths, f16 KV)
#      Phase 7's "done when": 1.5B Q4 > 15 tok/s on a mid-range Arc/AMD card.
#
# Requirements: Rust toolchain (or SAPIENT_BIN pointing at a prebuilt wgpu binary),
# python3, ~2 GB disk for model downloads, Vulkan loader + GPU driver
# (`sudo apt install libvulkan1` + vendor driver on Linux).
#
# Usage:
#   scripts/bench_gpu_7_6.sh                 # build + bench
#   SAPIENT_BIN=/path/to/sapient scripts/bench_gpu_7_6.sh   # reuse a binary

set -euo pipefail
cd "$(dirname "$0")/.."

TOKENS="${TOKENS:-64}"
MODELS=("openhorizon/smollm2-360m-q4" "openhorizon/qwen2.5-1.5b-q4")
PORT="${PORT:-18234}"

# ── 1. GPU / driver info ─────────────────────────────────────────────────────
GPU_DESC="unknown-gpu"
if command -v vulkaninfo >/dev/null 2>&1; then
    GPU_DESC=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= //') || true
elif command -v lspci >/dev/null 2>&1; then
    GPU_DESC=$(lspci | grep -m1 -iE 'vga|3d controller' | sed 's/.*: //') || true
fi
SAFE_GPU=$(echo "$GPU_DESC" | tr -cs '[:alnum:]' '-' | sed 's/^-//;s/-$//' | cut -c1-40)
OUT="bench-7_6-${SAFE_GPU:-gpu}.txt"

# ── 2. Binary ────────────────────────────────────────────────────────────────
if [ -n "${SAPIENT_BIN:-}" ]; then
    BIN="$SAPIENT_BIN"
else
    echo "building sapient with --features wgpu (first build takes a few minutes)…"
    cargo build --release -p sapient-cli --features wgpu
    BIN="./target/release/sapient"
fi

{
    echo "# SAPIENT Phase 7.6 GPU benchmark"
    echo "date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host:    $(uname -srm)"
    echo "gpu:     $GPU_DESC"
    echo "binary:  $("$BIN" --version 2>/dev/null || echo "$BIN")"
    echo "commit:  $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo
} | tee "$OUT"

wait_health() {
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && return 0
        sleep 2
    done
    echo "server did not come up on :$PORT" | tee -a "$OUT"; return 1
}

for MODEL in "${MODELS[@]}"; do
    echo "════ $MODEL ════" | tee -a "$OUT"

    # ── 3+4. Engine line (VRAM, quantized count, KV dtype) + greedy probe ────
    LOG=$(mktemp)
    "$BIN" --verbose serve --backend wgpu --port "$PORT" >"$LOG" 2>&1 &
    SRV=$!
    wait_health
    REPLY=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"temperature\":0,\"max_tokens\":16,\
             \"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}]}" \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' \
        || echo "REQUEST-FAILED")
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true
    grep -E "wgpu GPU device ready|WgpuForwardEngine ready" "$LOG" \
        | sed 's/\x1b\[[0-9;]*m//g' | tee -a "$OUT" || true
    # Correctness signal: the 1.5B model must answer "Paris"; the 360M model is
    # too small to stay on topic (it rambles identically on every backend) — for
    # it the engine line + throughput are the data points.
    echo "greedy probe: '$REPLY'" | tee -a "$OUT"
    rm -f "$LOG"

    # ── 5. Throughput: cpu vs wgpu ───────────────────────────────────────────
    python3 scripts/bench_wgpu.py --model "$MODEL" --backends cpu,wgpu \
        --tokens "$TOKENS" --bin "$BIN" 2>&1 | tee -a "$OUT"
    echo | tee -a "$OUT"
done

echo "results written to $OUT — please attach it to the Phase 7 PR (SkidGod4444/sapient#17)."
