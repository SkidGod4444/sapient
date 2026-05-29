#!/usr/bin/env bash
# Publish the workspace to crates.io crate-by-crate, in dependency order.
#
# `cargo publish --workspace` is NOT resumable (it errors on the first
# already-published crate), so we publish each crate individually instead:
#   - already-published versions are skipped (uses the /download endpoint to
#     check the exact version, not just whether the crate name exists),
#   - crates.io's new-crate rate limit (429) is waited out and retried.
# Idempotent and safe to re-run until everything is live.
set -uo pipefail
cd "$(dirname "$0")/.."

VERSION=$(grep '^version' Cargo.toml | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
echo "Publishing workspace version: $VERSION"

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

is_published() {
  local crate="$1"
  # The /download endpoint redirects (3xx) when the version exists; 404 when not.
  local code
  code=$(curl -s -A "sapient-publish" -o /dev/null -w "%{http_code}" -L \
    "https://static.crates.io/crates/$crate/$crate-$VERSION.crate")
  [ "$code" = "200" ]
}

for c in "${CRATES[@]}"; do
  if is_published "$c"; then
    echo "✓ $c@$VERSION already published — skipping"
    continue
  fi

  while true; do
    echo "=== publishing $c@$VERSION ($(date -u)) ==="
    out=$(cargo publish -p "$c" 2>&1)
    rc=$?
    echo "$out" | grep -iE "Uploaded|error|429|Too Many|already exists" || true

    if [ $rc -eq 0 ]; then
      echo "✓ published $c@$VERSION"
      break
    fi
    if echo "$out" | grep -qiE "already exists"; then
      echo "✓ $c@$VERSION already exists — continuing"
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

echo "=== ALL 13 CRATES PUBLISHED @ $VERSION ==="
