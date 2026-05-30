#!/usr/bin/env bash
# =============================================================================
# benchmark-compare.sh — Portable SAPIENT vs llama.cpp / Ollama / llamafile
#
# Designed to run on:
#   - macOS Intel (x86_64) or Apple Silicon (aarch64)
#   - Linux x86_64  (DGX H200, cloud VMs)
#   - Linux aarch64 (DGX Spark, Raspberry Pi 5)
#
# Measures (per engine, per model):
#   load_ms      — wall time from process start to model ready
#   ttft_ms      — time from prompt send to first output byte
#   decode_tps   — tokens/second during generation (after first token)
#   peak_rss_mb  — max resident RAM during inference
#   binary_mb    — size of the engine binary on disk
#
# Usage:
#   ./scripts/benchmark-compare.sh [OPTIONS]
#
# Options:
#   --engines   sapient,llamacpp,ollama,llamafile   (default: sapient,llamacpp,ollama)
#   --models    0.5b,8b                             (default: 0.5b)
#   --tokens    50                                  (tokens to generate per run)
#   --runs      3                                   (measured runs per engine+model)
#   --out       ./results/benchmark                 (output directory for JSON)
#   --no-install                                    (skip engine installation)
#   --cuda                                          (enable CUDA for llama.cpp on Linux)
#
# Output:
#   results/benchmark/
#     system_info.json
#     sapient_<model>.json
#     llamacpp_<model>.json
#     ollama_<model>.json
#     llamafile_<model>.json
#     summary.md
#
# After running, generate the report:
#   python3 scripts/gen-benchmark-report.py --dir results/benchmark/ --out docs/BENCHMARKS.md
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
ENGINES="sapient,llamacpp,ollama"
MODELS_ARG="0.5b"
MAX_TOKENS=50
RUNS=3
OUT_DIR="./results/benchmark"
SKIP_INSTALL=false
USE_CUDA=false
PROMPT_SHORT="What is the capital of France? Answer in one word."
PROMPT_MEDIUM="Explain quantum entanglement in one sentence."
PROMPT_LONG="You are a helpful assistant. The user asks: Explain the difference between supervised and unsupervised machine learning, including 2 examples of each."

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --engines)    ENGINES="$2";       shift 2 ;;
    --models)     MODELS_ARG="$2";    shift 2 ;;
    --tokens)     MAX_TOKENS="$2";    shift 2 ;;
    --runs)       RUNS="$2";          shift 2 ;;
    --out)        OUT_DIR="$2";       shift 2 ;;
    --no-install) SKIP_INSTALL=true;  shift   ;;
    --cuda)       USE_CUDA=true;      shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"

# ── System detection ──────────────────────────────────────────────────────────
OS=$(uname -s)
ARCH=$(uname -m)
NCORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

detect_ram_gb() {
  if [[ "$OS" == "Darwin" ]]; then
    sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}'
  else
    awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || echo 0
  fi
}

detect_gpu() {
  if [[ "$OS" == "Darwin" ]]; then
    system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | cut -d: -f2 | xargs
  elif command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
  else
    echo "none"
  fi
}

detect_cpu() {
  if [[ "$OS" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m
  else
    grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || uname -m
  fi
}

RAM_GB=$(detect_ram_gb)
GPU=$(detect_gpu)
CPU_NAME=$(detect_cpu)
SAPIENT_VERSION=$(sapient --version 2>/dev/null | awk '{print $2}' || echo "unknown")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAPIENT Engine Benchmark Suite"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OS:      $OS $ARCH"
echo "  CPU:     $CPU_NAME"
echo "  RAM:     ${RAM_GB} GB"
echo "  GPU:     $GPU"
echo "  Cores:   $NCORES"
echo "  Engines: $ENGINES"
echo "  Models:  $MODELS_ARG"
echo "  Tokens:  $MAX_TOKENS (× $RUNS runs)"
echo "  Output:  $OUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Save system info
python3 -c "
import json, datetime
info = {
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'os': '$OS', 'arch': '$ARCH', 'cpu': '$CPU_NAME',
    'ram_gb': $RAM_GB, 'gpu': '$GPU', 'cores': $NCORES,
    'sapient_version': '$SAPIENT_VERSION',
}
with open('$OUT_DIR/system_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print('  system_info.json written')
"

# ── Helper: RSS measurement ───────────────────────────────────────────────────
# Returns peak RSS in MB by sampling /proc/<pid>/status or `ps`
measure_peak_rss() {
  local pid=$1
  local peak=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ "$OS" == "Linux" ]]; then
      rss=$(awk '/VmRSS/{print $2}' /proc/"$pid"/status 2>/dev/null | head -1 || echo 0)
    else
      rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0)
    fi
    [[ "$rss" -gt "$peak" ]] && peak=$rss
    sleep 0.1
  done
  echo $((peak / 1024))  # kB → MB
}

# ── Helper: ms since epoch ────────────────────────────────────────────────────
now_ms() { python3 -c "import time; print(int(time.time()*1000))"; }

# ── Model resolution ──────────────────────────────────────────────────────────
resolve_gguf_for_llamacpp() {
  local model_size=$1
  case "$model_size" in
    0.5b) echo "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q8_0.gguf" ;;
    1.5b) echo "Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q8_0.gguf" ;;
    3b)   echo "unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf" ;;
    8b)   echo "unsloth/Llama-3.1-8B-Instruct-GGUF:Llama-3.1-8B-Instruct-Q4_K_M.gguf" ;;
    *) echo "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q8_0.gguf" ;;
  esac
}

resolve_sapient_model() {
  case "$1" in
    0.5b) echo "openhorizon/qwen2.5-0.5b-q4" ;;
    1.5b) echo "openhorizon/qwen2.5-1.5b-q4" ;;
    3b)   echo "openhorizon/llama-3.2-3b-q4" ;;
    8b)   echo "openhorizon/llama-3.1-8b-q4" ;;
    *)    echo "openhorizon/qwen2.5-0.5b-q4" ;;
  esac
}

resolve_ollama_model() {
  case "$1" in
    0.5b) echo "qwen2.5:0.5b" ;;
    1.5b) echo "qwen2.5:1.5b" ;;
    3b)   echo "llama3.2:3b"  ;;
    8b)   echo "llama3.1:8b"  ;;
    *)    echo "qwen2.5:0.5b" ;;
  esac
}

# ── Install engines ───────────────────────────────────────────────────────────
LLAMA_CPP_DIR="$OUT_DIR/llama.cpp"
LLAMAFILE_DIR="$OUT_DIR/llamafile"

install_llamacpp() {
  if command -v llama-bench &>/dev/null; then
    echo "  llama.cpp already installed ($(which llama-bench))"
    return
  fi
  if [[ -f "$LLAMA_CPP_DIR/build/bin/llama-bench" ]]; then
    echo "  llama.cpp already built at $LLAMA_CPP_DIR"
    return
  fi

  echo "  Installing llama.cpp from source..."

  # System deps
  if [[ "$OS" == "Linux" ]]; then
    if command -v apt-get &>/dev/null; then
      sudo apt-get install -y build-essential cmake curl libcurl4-openssl-dev 2>/dev/null || true
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y gcc gcc-c++ cmake curl libcurl-devel 2>/dev/null || true
    fi
  fi

  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR" 2>/dev/null \
    || git -C "$LLAMA_CPP_DIR" pull --depth 1 2>/dev/null || true

  CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF"
  if [[ "$USE_CUDA" == "true" ]] && command -v nvcc &>/dev/null; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON"
    echo "  Building with CUDA support"
  elif [[ "$OS" == "Darwin" ]]; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_METAL=ON"
    echo "  Building with Metal support"
  fi

  cmake "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" $CMAKE_FLAGS
  cmake --build "$LLAMA_CPP_DIR/build" -j"$NCORES" \
    --target llama-bench llama-cli llama-server 2>&1 | tail -3
  echo "  llama.cpp built: $LLAMA_CPP_DIR/build/bin/"
}

install_ollama() {
  if command -v ollama &>/dev/null; then
    echo "  Ollama already installed ($(ollama --version 2>&1 | head -1))"
    return
  fi
  echo "  Installing Ollama..."
  curl -fsSL https://ollama.ai/install.sh | sh 2>&1 | tail -3
}

install_llamafile() {
  local bin="$LLAMAFILE_DIR/llamafile"
  if [[ -f "$bin" ]]; then
    echo "  llamafile already at $bin"
    return
  fi
  mkdir -p "$LLAMAFILE_DIR"
  echo "  Downloading llamafile..."
  # Mozilla's llamafile is a self-contained llama.cpp executable
  LLAMAFILE_VER="0.8.14"
  LLAMAFILE_URL="https://github.com/Mozilla-Ocho/llamafile/releases/download/$LLAMAFILE_VER/llamafile-$LLAMAFILE_VER"
  curl -L "$LLAMAFILE_URL" -o "$bin" 2>/dev/null
  chmod +x "$bin"
  echo "  llamafile installed at $bin"
}

find_llama_bench() {
  if command -v llama-bench &>/dev/null; then echo "llama-bench"
  elif [[ -f "$LLAMA_CPP_DIR/build/bin/llama-bench" ]]; then echo "$LLAMA_CPP_DIR/build/bin/llama-bench"
  else echo ""; fi
}

find_llama_cli() {
  if command -v llama-cli &>/dev/null; then echo "llama-cli"
  elif [[ -f "$LLAMA_CPP_DIR/build/bin/llama-cli" ]]; then echo "$LLAMA_CPP_DIR/build/bin/llama-cli"
  else echo ""; fi
}

# ── Download model file (for llama.cpp / llamafile) ───────────────────────────
download_gguf() {
  local repo_file=$1   # "Org/Repo:filename.gguf"
  local out_dir="$OUT_DIR/models"
  mkdir -p "$out_dir"

  local repo="${repo_file%%:*}"
  local fname="${repo_file##*:}"
  local out="$out_dir/$fname"

  if [[ -f "$out" ]]; then
    echo "$out"
    return
  fi

  echo "  Downloading $fname from $repo..." >&2
  if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$repo" "$fname" --local-dir "$out_dir" --quiet 2>/dev/null
  elif python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('$repo', '$fname', local_dir='$out_dir', quiet=True)
print(path)
"
  else
    # Direct URL download
    local url="https://huggingface.co/$repo/resolve/main/$fname"
    curl -L "$url" -o "$out" --progress-bar
  fi
  echo "$out"
}

# ── Binary size helper ────────────────────────────────────────────────────────
binary_mb() {
  local bin=$1
  if [[ -f "$bin" ]]; then
    ls -l "$bin" | awk '{print int($5/1024/1024)}'
  else
    echo 0
  fi
}

# ── SAPIENT benchmark ─────────────────────────────────────────────────────────
run_sapient() {
  local model_size=$1
  local model=$(resolve_sapient_model "$model_size")
  local sapient_bin
  sapient_bin=$(command -v sapient 2>/dev/null || echo "")
  if [[ -z "$sapient_bin" ]]; then
    echo "  [SKIP] sapient not found in PATH"
    return
  fi

  local out_file="$OUT_DIR/sapient_${model_size}.json"
  echo ""
  echo "▶ SAPIENT — $model"

  "$sapient_bin" bench-llm "$model" \
    --prompt "$PROMPT_MEDIUM" \
    --max-tokens "$MAX_TOKENS" \
    --runs "$RUNS" \
    --mmap \
    --json > "$out_file"

  echo "  Saved → $out_file"
  python3 -c "
import json
with open('$out_file') as f: d = json.load(f)
s = d.get('summary', {})
print(f\"  load={d.get('load_time_ms','?')}ms  ttft={s.get('mean_ttft_ms','?')}ms  tps={s.get('mean_tps','?')}  rss={s.get('peak_rss_mb','?')}MB\")
"
}

# ── llama.cpp benchmark ───────────────────────────────────────────────────────
run_llamacpp() {
  local model_size=$1
  local bench=$(find_llama_bench)
  local cli=$(find_llama_cli)

  if [[ -z "$bench" && -z "$cli" ]]; then
    echo "  [SKIP] llama.cpp not found — run with --engines without llamacpp or install first"
    return
  fi

  local gguf_ref
  gguf_ref=$(resolve_gguf_for_llamacpp "$model_size")
  local model_file
  model_file=$(download_gguf "$gguf_ref")

  if [[ ! -f "$model_file" ]]; then
    echo "  [SKIP] Could not download model file for llama.cpp"
    return
  fi

  local out_file="$OUT_DIR/llamacpp_${model_size}.json"
  local bin_mb=0
  [[ -n "$bench" ]] && bin_mb=$(binary_mb "$(which "$bench" 2>/dev/null || echo "$bench")")

  echo ""
  echo "▶ llama.cpp — $gguf_ref"
  echo "  (binary: ${bin_mb}MB)"

  # llama-bench: measures prompt eval (pp) and token generation (tg) throughput
  # -p 0 = no prompt eval benchmark, -n $MAX_TOKENS = gen tokens
  # --output json for structured output
  local bench_out
  if [[ -n "$bench" ]]; then
    bench_out=$("$bench" \
      -m "$model_file" \
      -p 512 \
      -n "$MAX_TOKENS" \
      -r "$RUNS" \
      --output json 2>/dev/null) || true
  fi

  # Also measure TTFT with llama-cli
  local ttft_ms=0
  if [[ -n "$cli" ]]; then
    local t0 t1
    t0=$(now_ms)
    "$cli" -m "$model_file" \
      -p "$PROMPT_MEDIUM" \
      -n "$MAX_TOKENS" \
      -t "$NCORES" \
      --no-display-prompt \
      -ngl 0 \
      2>/tmp/llama_timing.txt 1>/dev/null || true
    t1=$(now_ms)

    # llama-cli outputs timing to stderr, parse it
    local pp_ms tg_ms tg_tok
    pp_ms=$(grep "prompt eval time" /tmp/llama_timing.txt 2>/dev/null | grep -oP '\d+\.\d+ ms' | head -1 | awk '{print int($1)}' || echo 0)
    tg_ms=$(grep "eval time" /tmp/llama_timing.txt 2>/dev/null | tail -1 | grep -oP '\d+\.\d+ ms' | head -1 | awk '{print int($1)}' || echo 0)
    tg_tok=$(grep "eval time" /tmp/llama_timing.txt 2>/dev/null | tail -1 | grep -oP '\d+ runs' | awk '{print $1}' || echo 0)
    ttft_ms=$pp_ms
  fi

  # Save results
  python3 - "$out_file" "$model_size" "$gguf_ref" "$bin_mb" "$bench_out" "$ttft_ms" <<'PYEOF'
import json, sys

out_file   = sys.argv[1]
model_size = sys.argv[2]
model_ref  = sys.argv[3]
bin_mb     = int(sys.argv[4])
bench_json = sys.argv[5]
ttft_ms    = int(sys.argv[6])

# Parse llama-bench JSON if available
tg_tps = None
pp_tps = None
try:
    rows = json.loads(bench_json) if bench_json.strip() else []
    for r in rows:
        if r.get('test', '') == 'tg':
            tg_tps = round(r.get('avg_ts', 0), 1)
        elif r.get('test', '') == 'pp':
            pp_tps = round(r.get('avg_ts', 0), 1)
except Exception:
    pass

result = {
    "engine": "llama.cpp",
    "model": model_ref,
    "model_size": model_size,
    "binary_mb": bin_mb,
    "ttft_ms": ttft_ms,
    "decode_tps": tg_tps,
    "prefill_tps": pp_tps,
    "peak_rss_mb": None,  # not measured separately
}
with open(out_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved → {out_file}")
if tg_tps: print(f"  ttft={ttft_ms}ms  decode={tg_tps}tps  prefill={pp_tps}tps")
PYEOF
}

# ── Ollama benchmark ──────────────────────────────────────────────────────────
run_ollama() {
  local model_size=$1
  local ollama_model
  ollama_model=$(resolve_ollama_model "$model_size")
  local out_file="$OUT_DIR/ollama_${model_size}.json"

  local OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"

  echo ""
  echo "▶ Ollama — $ollama_model"

  # Start Ollama if not running
  if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    if command -v ollama &>/dev/null; then
      echo "  Starting ollama serve..."
      ollama serve >/tmp/ollama.log 2>&1 &
      OLLAMA_PID=$!
      sleep 5
    else
      echo "  [SKIP] Ollama not installed and not running"
      return
    fi
  fi

  # Pull model if not present
  if ! ollama list 2>/dev/null | grep -q "^$ollama_model"; then
    echo "  Pulling $ollama_model..."
    ollama pull "$ollama_model" 2>&1 | tail -2 || {
      echo "  [SKIP] Failed to pull $ollama_model"
      return
    }
  fi

  # Warm-up (discard)
  curl -sf "$OLLAMA_URL/api/generate" \
    -d "{\"model\":\"$ollama_model\",\"prompt\":\"hi\",\"options\":{\"num_predict\":3},\"stream\":false}" \
    >/dev/null 2>&1 || true

  local bin_mb=0
  command -v ollama &>/dev/null && bin_mb=$(binary_mb "$(which ollama)")

  # Measured runs
  python3 - "$out_file" "$ollama_model" "$model_size" "$OLLAMA_URL" \
    "$PROMPT_MEDIUM" "$MAX_TOKENS" "$RUNS" "$bin_mb" <<'PYEOF'
import json, sys, urllib.request, time, statistics

out_file    = sys.argv[1]
model       = sys.argv[2]
model_size  = sys.argv[3]
base_url    = sys.argv[4]
prompt      = sys.argv[5]
max_tokens  = int(sys.argv[6])
n_runs      = int(sys.argv[7])
bin_mb      = int(sys.argv[8])

runs = []
for i in range(n_runs):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "options": {"num_predict": max_tokens},
        "stream": False
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        d = json.loads(resp.read())
        eval_count = d.get("eval_count", 0)
        eval_dur   = d.get("eval_duration", 1)   # ns
        pp_dur     = d.get("prompt_eval_duration", 0)  # ns
        load_dur   = d.get("load_duration", 0)   # ns
        tps = round(eval_count / (eval_dur / 1e9), 1) if eval_dur else 0
        run = {
            "run": i + 1,
            "ttft_ms":   int(pp_dur / 1e6),
            "elapsed_ms": int(eval_dur / 1e6),
            "load_ms":   int(load_dur / 1e6),
            "total_tokens": eval_count,
            "tps":       tps,
        }
        runs.append(run)
        print(f"  run {i+1}: ttft={run['ttft_ms']}ms  tps={tps}  load={run['load_ms']}ms")
    except Exception as e:
        print(f"  run {i+1} failed: {e}")

if runs:
    mean_ttft = int(statistics.mean(r["ttft_ms"] for r in runs))
    mean_tps  = round(statistics.mean(r["tps"] for r in runs), 1)
    result = {
        "engine": "ollama",
        "model": model,
        "model_size": model_size,
        "binary_mb": bin_mb,
        "load_time_ms": runs[0].get("load_ms", 0),
        "ttft_ms": mean_ttft,
        "decode_tps": mean_tps,
        "runs": runs,
        "summary": {"mean_ttft_ms": mean_ttft, "mean_tps": mean_tps},
    }
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_file}")
    print(f"  Mean: ttft={mean_ttft}ms  tps={mean_tps}")
PYEOF
}

# ── llamafile benchmark ───────────────────────────────────────────────────────
run_llamafile() {
  local model_size=$1
  local bin="$LLAMAFILE_DIR/llamafile"

  if [[ ! -f "$bin" ]]; then
    echo "  [SKIP] llamafile not found (run without --no-install to download)"
    return
  fi

  local gguf_ref
  gguf_ref=$(resolve_gguf_for_llamacpp "$model_size")
  local model_file
  model_file=$(download_gguf "$gguf_ref")

  if [[ ! -f "$model_file" ]]; then
    echo "  [SKIP] Could not download model for llamafile"
    return
  fi

  local out_file="$OUT_DIR/llamafile_${model_size}.json"
  local bin_mb
  bin_mb=$(binary_mb "$bin")

  echo ""
  echo "▶ llamafile — $gguf_ref"
  echo "  (binary: ${bin_mb}MB)"

  local t0 t1
  t0=$(now_ms)
  "$bin" \
    -m "$model_file" \
    -p "$PROMPT_MEDIUM" \
    -n "$MAX_TOKENS" \
    -t "$NCORES" \
    --no-display-prompt \
    2>/tmp/llamafile_timing.txt 1>/dev/null || true
  t1=$(now_ms)

  local elapsed=$((t1 - t0))
  local pp_ms tg_tps
  pp_ms=$(grep "prompt eval time" /tmp/llamafile_timing.txt 2>/dev/null | grep -oP '\d+\.\d+ ms' | head -1 | awk '{print int($1)}' || echo 0)
  tg_tps=$(grep "eval time" /tmp/llamafile_timing.txt 2>/dev/null | grep -oP '\d+\.\d+ tokens/s' | head -1 | awk '{print $1}' || echo 0)

  python3 -c "
import json
result = {
    'engine': 'llamafile',
    'model': '$gguf_ref',
    'model_size': '$model_size',
    'binary_mb': $bin_mb,
    'ttft_ms': $pp_ms,
    'decode_tps': float('$tg_tps') if '$tg_tps' else None,
    'total_elapsed_ms': $elapsed,
}
with open('$out_file', 'w') as f:
    json.dump(result, f, indent=2)
print('  Saved → $out_file')
if '$tg_tps':
    print(f'  ttft=${pp_ms}ms  tps=$tg_tps')
"
}

# ── Main run loop ─────────────────────────────────────────────────────────────
# Install requested engines
if [[ "$SKIP_INSTALL" == "false" ]]; then
  echo ""
  echo "── Installing / verifying engines ──────────────────────────────────────"
  IFS=',' read -ra ENGINE_LIST <<< "$ENGINES"
  for eng in "${ENGINE_LIST[@]}"; do
    case "$eng" in
      llamacpp|llama_cpp|llama.cpp) install_llamacpp ;;
      ollama)                        install_ollama ;;
      llamafile)                     install_llamafile ;;
      sapient)                       : ;;  # assume already installed
    esac
  done
fi

# Run benchmarks
IFS=',' read -ra MODEL_LIST <<< "$MODELS_ARG"
IFS=',' read -ra ENGINE_LIST <<< "$ENGINES"

echo ""
echo "── Running benchmarks ───────────────────────────────────────────────────"

for model_size in "${MODEL_LIST[@]}"; do
  echo ""
  echo "━━━ Model: ${model_size} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  for eng in "${ENGINE_LIST[@]}"; do
    case "$eng" in
      sapient)                       run_sapient   "$model_size" ;;
      llamacpp|llama_cpp|llama.cpp)  run_llamacpp  "$model_size" ;;
      ollama)                        run_ollama    "$model_size" ;;
      llamafile)                     run_llamafile "$model_size" ;;
      *) echo "  Unknown engine: $eng" ;;
    esac
  done
done

# ── Side-by-side summary table ────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 - "$OUT_DIR" "${MODEL_LIST[@]}" <<'PYEOF'
import json, sys, os, glob

out_dir = sys.argv[1]
models  = sys.argv[2:]

engines_order = ["sapient", "llamacpp", "ollama", "llamafile"]

for model_size in models:
    print(f"\n  Model: {model_size}")
    print(f"  {'Engine':<14} {'TTFT':>8} {'Tok/s':>8} {'Load':>8} {'Binary':>8}  Winner")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    results = {}
    for eng in engines_order:
        fpath = os.path.join(out_dir, f"{eng}_{model_size}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                results[eng] = json.load(f)

    if not results:
        print("  (no results)")
        continue

    def val(r, *keys):
        for k in keys:
            v = r.get(k)
            if v is not None:
                s = r.get("summary", {})
                if isinstance(v, dict):
                    continue
                return v
            # try summary
            v2 = r.get("summary", {}).get(k)
            if v2 is not None:
                return v2
        return None

    min_ttft = min((val(r, "ttft_ms", "mean_ttft_ms") or 99999) for r in results.values())
    max_tps  = max((val(r, "decode_tps", "mean_tps") or 0) for r in results.values())

    for eng, r in results.items():
        ttft   = val(r, "ttft_ms", "mean_ttft_ms")
        tps    = val(r, "decode_tps", "mean_tps")
        load   = val(r, "load_time_ms", "load_ms")
        binmb  = r.get("binary_mb")

        ttft_s  = f"{ttft}ms"   if ttft  is not None else "?"
        tps_s   = f"{tps}"      if tps   is not None else "?"
        load_s  = f"{load}ms"   if load  is not None else "?"
        bin_s   = f"{binmb}MB"  if binmb is not None else "?"

        wins = []
        if ttft is not None and ttft == min_ttft: wins.append("TTFT")
        if tps  is not None and tps  == max_tps:  wins.append("tok/s")
        win_str = "+".join(wins) if wins else ""

        print(f"  {eng:<14} {ttft_s:>8} {tps_s:>8} {load_s:>8} {bin_s:>8}  {win_str}")

print()
PYEOF

echo ""
echo "  JSON results: $OUT_DIR/"
echo "  Generate full report:"
echo "    python3 scripts/gen-benchmark-report.py --dir $OUT_DIR --out docs/BENCHMARKS.md"
echo ""
echo "  To run on a remote system, scp this script and run:"
echo "    scp scripts/benchmark-compare.sh user@host:"
echo "    ssh user@host './benchmark-compare.sh --engines sapient,llamacpp,ollama --models 0.5b,8b'"
