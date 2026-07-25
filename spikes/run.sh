#!/usr/bin/env bash
# Build + run a Phase-0 spike crate reproducibly in the tdb-spikes image,
# with cached cargo registry + workspace target/ so rebuilds are incremental.
#
# Usage:  ./run.sh <crate> [extra cargo-run args...]
#   crate ∈ { lance-branch | layer-index | tokenizer }
#
# Full output is tee'd to evidence/logs/<crate>.log (never suppressed);
# the caller surfaces the relevant lines.
set -euo pipefail

CRATE="${1:?usage: run.sh <crate> [args...]}"; shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/evidence/logs/${CRATE}.log"
mkdir -p "$HERE/evidence/logs"

# Named volumes: cargo registry/git (download cache) + workspace target (compile cache).
docker volume create tdb-spike-cargo  >/dev/null
docker volume create tdb-spike-target >/dev/null

# Ensure the image exists (cheap if already built).
docker build -t tdb-spikes "$HERE" 2>&1 | tee -a "$LOG" >/dev/null

# The tokenizer spike needs the model tokenizer.json mounted read-only.
EXTRA_MOUNT=()
if [[ "$CRATE" == "tokenizer" ]]; then
  EXTRA_MOUNT=(-v "$HERE/tokenizer/tokenizer.json:/data/tokenizer.json:ro")
  set -- /data/tokenizer.json "$@"
fi

echo "=== $(date -u +%FT%TZ) building+running spike: $CRATE ===" | tee -a "$LOG"
docker run --rm \
  -v "$HERE":/spikes \
  -v tdb-spike-cargo:/usr/local/cargo/registry \
  -v tdb-spike-target:/target \
  -e CARGO_TARGET_DIR=/target \
  "${EXTRA_MOUNT[@]}" \
  --entrypoint cargo \
  tdb-spikes \
  run --release -p "spike-${CRATE}" -- "$@" 2>&1 | tee -a "$LOG"
