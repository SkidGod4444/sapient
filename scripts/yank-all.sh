#!/usr/bin/env bash
# Yank every published version of every SAPIENT crate from crates.io.
#
# We no longer publish to crates.io. crates.io does NOT allow deleting a crate
# (only the crates.io team can, by request); `cargo yank` is the supported way to
# "remove" a release — it prevents NEW dependency resolution against the version
# while keeping it downloadable for existing users (reproducibility guarantee).
#
# Requires a crates.io token (`cargo login`, stored in ~/.cargo/credentials.toml).
# Idempotent: re-yanking an already-yanked version is a no-op. Pass `--undo` to
# reverse (un-yank) every version instead.
set -uo pipefail

UNDO=""
if [[ "${1:-}" == "--undo" ]]; then
  UNDO="--undo"
  echo "Mode: UN-YANK (reversing)"
else
  echo "Mode: YANK"
fi

CRATES=(
  sapient-core
  sapient-ir
  sapient-telemetry
  sapient-tokenizers
  sapient-backends-cpu
  sapient-backends-metal
  sapient-backends-wgpu
  sapient-io
  sapient-hub
  sapient-scheduler
  sapient-models
  sapient-runtime
  sapient-generate
  sapient-cli
)

for crate in "${CRATES[@]}"; do
  # Fetch published version numbers from the crates.io API.
  versions=$(curl -s "https://crates.io/api/v1/crates/${crate}" \
    -H "User-Agent: sapient-yank" \
    | python3 -c "import sys,json
try:
    d=json.load(sys.stdin)
    print('\n'.join(v['num'] for v in d.get('versions',[])))
except Exception:
    pass")

  if [[ -z "$versions" ]]; then
    echo "○ ${crate}: not published — skipping"
    continue
  fi

  while IFS= read -r v; do
    [[ -z "$v" ]] && continue
    echo "→ cargo yank ${UNDO} ${crate}@${v}"
    cargo yank ${UNDO} --version "$v" "$crate" || echo "  (failed; continuing)"
    sleep 1  # be gentle with crates.io rate limits
  done <<< "$versions"
done

echo "Done."
