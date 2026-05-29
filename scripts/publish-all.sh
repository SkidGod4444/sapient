#!/usr/bin/env bash
# Publish the workspace to crates.io crate-by-crate, in dependency order.
#
# `cargo publish --workspace` is NOT resumable (it errors on the first
# already-published crate), so we publish each crate individually instead:
#   - already-published crates are skipped,
#   - crates.io's new-crate rate limit (429) is waited out and retried.
# Idempotent and safe to re-run until everything is live.
set -uo pipefail
cd "$(dirname "$0")/.."

# Dependency-topological order.
CRATES=(
  sapient-core
  sapient-ir
  sapient-telemetry
  sapient-tokenizers
  sapient-backends-cpu
  sapient-backends-metal
  sapient-io
  sapient-hub
  sapient-scheduler
  sapient-models
  sapient-runtime
  sapient-generate
  sapient-cli
)

for c in "${CRATES[@]}"; do
  # Skip if already on crates.io.
  code=$(curl -s -A "sapient-publish" -o /dev/null -w "%{http_code}" \
    "https://crates.io/api/v1/crates/$c/0.1.11")
  if [ "$code" = "200" ]; then
    echo "✓ $c already published — skipping"
    continue
  fi

  while true; do
    echo "=== publishing $c ($(date -u)) ==="
    out=$(cargo publish -p "$c" 2>&1)
    rc=$?
    echo "$out" | grep -iE "Uploaded|error|429|Too Many|already exists" || true

    if [ $rc -eq 0 ]; then
      echo "✓ published $c"
      break
    fi
    if echo "$out" | grep -qiE "already exists"; then
      echo "✓ $c already exists — continuing"
      break
    fi
    if echo "$out" | grep -qiE "429|Too Many Requests"; then
      echo "rate-limited on $c; sleeping 11 minutes…"
      sleep 660
      continue
    fi
    echo "=== FATAL publishing $c ==="
    echo "$out" | tail -30
    exit 1
  done
done

echo "=== ALL 13 CRATES PUBLISHED ==="
