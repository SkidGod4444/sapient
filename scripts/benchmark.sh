#!/usr/bin/env bash
# benchmark.sh — SAPIENT vs Ollama LLM benchmark comparison
#
# Usage:
#   ./scripts/benchmark.sh                  # default: qwen2.5-0.5b
#   ./scripts/benchmark.sh --model 1.5b     # compare 1.5B model pair
#   ./scripts/benchmark.sh --tokens 100     # generate more tokens per run
#   ./scripts/benchmark.sh --runs 5         # more statistical runs
#   ./scripts/benchmark.sh --out results/   # write JSON to directory
#
# Requirements:
#   - sapient binary in PATH (or ./target/release/sapient)
#   - ollama installed and server running (ollama serve &)
#   - jq >= 1.6
#   - python3 (for report generation)
#
# Outputs:
#   - Prints side-by-side comparison table
#   - Writes sapient_result.json and ollama_result.json to --out dir

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_SIZE="0.5b"
MAX_TOKENS=50
RUNS=3
OUT_DIR="."
PROMPT="Explain quantum entanglement in one sentence."

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)  MODEL_SIZE="$2"; shift 2 ;;
        --tokens) MAX_TOKENS="$2"; shift 2 ;;
        --runs)   RUNS="$2"; shift 2 ;;
        --out)    OUT_DIR="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Model pair selection ──────────────────────────────────────────────────────
case "$MODEL_SIZE" in
    0.5b)
        SAPIENT_MODEL="openhorizon/qwen2.5-0.5b-q4"
        OLLAMA_MODEL="qwen2.5:0.5b"
        ;;
    1.5b)
        SAPIENT_MODEL="openhorizon/qwen2.5-1.5b-q4"
        OLLAMA_MODEL="qwen2.5:1.5b"
        ;;
    smol-360m)
        SAPIENT_MODEL="openhorizon/smollm2-360m-q4"
        OLLAMA_MODEL="smollm2:360m"
        ;;
    smol-1.7b)
        SAPIENT_MODEL="openhorizon/smollm2-1.7b-q4"
        OLLAMA_MODEL="smollm2:1.7b"
        ;;
    *)
        echo "Unknown model size '$MODEL_SIZE'. Use: 0.5b, 1.5b, smol-360m, smol-1.7b"
        exit 1
        ;;
esac

# ── Find sapient binary ───────────────────────────────────────────────────────
if command -v sapient &>/dev/null; then
    SAPIENT="sapient"
elif [[ -f "./target/release/sapient" ]]; then
    SAPIENT="./target/release/sapient"
else
    echo "Error: sapient not found in PATH or ./target/release/sapient"
    echo "Build with: cargo build --release -p sapient-cli"
    exit 1
fi

mkdir -p "$OUT_DIR"

SAPIENT_JSON="$OUT_DIR/sapient_result.json"
OLLAMA_JSON="$OUT_DIR/ollama_result.json"
REPORT_MD="$OUT_DIR/benchmark_report.md"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAPIENT vs Ollama Benchmark"
echo "  Model pair : $SAPIENT_MODEL  ↔  $OLLAMA_MODEL"
echo "  Prompt     : $PROMPT"
echo "  Runs       : $RUNS   Max tokens: $MAX_TOKENS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── SAPIENT benchmark ─────────────────────────────────────────────────────────
echo ""
echo "▶ Running SAPIENT bench-llm …"
$SAPIENT bench-llm "$SAPIENT_MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --runs "$RUNS" \
    --mmap \
    --json > "$SAPIENT_JSON"

echo "  Saved → $SAPIENT_JSON"

# ── Ollama benchmark ──────────────────────────────────────────────────────────
echo ""
echo "▶ Running Ollama benchmark …"

OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"

# Check Ollama is reachable
if ! curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "  Warning: Ollama server not reachable at $OLLAMA_URL"
    echo "  Start it with: ollama serve"
    echo "  Skipping Ollama comparison — SAPIENT results saved to $SAPIENT_JSON"
    exit 0
fi

# Pull the model if not already local
if ! ollama list 2>/dev/null | grep -q "^$OLLAMA_MODEL"; then
    echo "  Pulling $OLLAMA_MODEL …"
    ollama pull "$OLLAMA_MODEL"
fi

# Warm-up run (not measured)
echo "  Warm-up run (discarded)…"
curl -sf "$OLLAMA_URL/api/generate" \
    -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"hi\",\"options\":{\"num_predict\":5},\"stream\":false}" \
    > /dev/null

# Measured runs
OLLAMA_RUNS="[]"
for i in $(seq 1 "$RUNS"); do
    printf "  Run %d/%d…\r" "$i" "$RUNS"
    RESP=$(curl -sf "$OLLAMA_URL/api/generate" \
        -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"$PROMPT\",\"options\":{\"num_predict\":$MAX_TOKENS},\"stream\":false}")

    EVAL_COUNT=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('eval_count',0))")
    EVAL_DUR=$(echo "$RESP"   | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('eval_duration',1))")
    LOAD_DUR=$(echo "$RESP"   | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('load_duration',0))")
    PEVAL_DUR=$(echo "$RESP"  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('prompt_eval_duration',0))")

    TPS=$(python3 -c "print(round($EVAL_COUNT/($EVAL_DUR/1e9),1))")
    TTFT_MS=$(python3 -c "print(int($PEVAL_DUR/1e6))")
    LOAD_MS=$(python3 -c "print(int($LOAD_DUR/1e6))")

    RUN_JSON="{\"run\":$i,\"ttft_ms\":$TTFT_MS,\"elapsed_ms\":$(python3 -c "print(int($EVAL_DUR/1e6))"),\"total_tokens\":$EVAL_COUNT,\"tps\":$TPS}"
    OLLAMA_RUNS=$(python3 -c "
import json, sys
runs = json.loads('$OLLAMA_RUNS')
runs.append(json.loads('''$RUN_JSON'''))
print(json.dumps(runs))
")
done
echo "                    "  # clear \r line

# Compute summary stats
python3 - "$OLLAMA_MODEL" "$OLLAMA_RUNS" "$OLLAMA_JSON" <<'PYEOF'
import json, sys

model = sys.argv[1]
runs = json.loads(sys.argv[2])
out_path = sys.argv[3]

mean_ttft = int(sum(r["ttft_ms"] for r in runs) / len(runs)) if runs else 0
mean_tps  = round(sum(r["tps"] for r in runs) / len(runs), 1) if runs else 0

result = {
    "model": model,
    "backend": "ollama (llama.cpp)",
    "load_time_ms": runs[0]["ttft_ms"] if runs else 0,
    "runs": runs,
    "summary": {
        "mean_ttft_ms": mean_ttft,
        "mean_tps": mean_tps,
        "peak_rss_mb": None,  # Ollama server is a separate process; RSS not tracked here
    }
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"  Saved → {out_path}")
PYEOF

# ── Print comparison table ────────────────────────────────────────────────────
python3 - "$SAPIENT_JSON" "$OLLAMA_JSON" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f: s = json.load(f)
with open(sys.argv[2]) as f: o = json.load(f)

ss = s["summary"]
os_ = o["summary"]

def bar(val, max_val, width=20, char="█"):
    filled = int(round(val / max_val * width)) if max_val else 0
    return char * filled + "░" * (width - filled)

print("\n" + "━"*60)
print("  Benchmark Results")
print("━"*60)
print(f"  {'Metric':<28} {'SAPIENT':>12} {'Ollama':>12}  Winner")
print("  " + "─"*56)

def row(label, sval, oval, lower_is_better=True, unit=""):
    try:
        sv, ov = float(sval), float(oval)
    except (TypeError, ValueError):
        print(f"  {label:<28} {str(sval)+unit:>12} {str(oval)+unit:>12}  —")
        return
    if lower_is_better:
        winner = "✓ SAPIENT" if sv < ov else ("✓ Ollama" if ov < sv else "tie")
    else:
        winner = "✓ SAPIENT" if sv > ov else ("✓ Ollama" if ov > sv else "tie")
    print(f"  {label:<28} {str(sval)+unit:>12} {str(oval)+unit:>12}  {winner}")

row("Load time",        s.get("load_time_ms","?"), o.get("load_time_ms","?"), lower_is_better=True,  unit=" ms")
row("Mean TTFT",        ss["mean_ttft_ms"],         os_["mean_ttft_ms"],       lower_is_better=True,  unit=" ms")
row("Mean tok/s",       ss["mean_tps"],              os_["mean_tps"],           lower_is_better=False, unit="")
if ss.get("peak_rss_mb"):
    row("Peak RSS",     ss["peak_rss_mb"],           "N/A (server)",            lower_is_better=True,  unit=" MB")

print("  " + "─"*56)
print(f"\n  Models: {s['model']}  vs  {o['model']}")
print(f"  Runs:   {len(s['runs'])} per engine")
print("━"*60 + "\n")
PYEOF

echo "Run 'python3 scripts/gen-benchmark-report.py --sapient $SAPIENT_JSON --ollama $OLLAMA_JSON --out $REPORT_MD' to generate the full markdown report."
