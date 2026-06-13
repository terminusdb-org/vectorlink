#!/usr/bin/env bash
# Build the engine, start it with an Ollama embeddings backend, run the mocha
# HTTP-contract/integration suite against the LIVE server, then tear down.
# Self-contained so `make pr` can gate on integration tests with no manual setup.
set -euo pipefail

CARGO_VOLUME="${CARGO_VOLUME:-tdb-search-cargo}"
BUILD_IMAGE="${BUILD_IMAGE:-tdb-search-build:local}"
CONTAINER="tdb-search-itest-$$"
NETWORK="tdb-search-itest-net-$$"
HOST_PORT="${HOST_PORT:-8089}"
EMBED_CONTAINER="tdb-search-itest-embed-$$"
EMBED_MODEL="${TDB_SEARCH_MODEL:-nomic-embed-v2}"
EMBED_DIM="${TDB_SEARCH_DIM:-768}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

cleanup() {
  # WHY: always remove test containers even if a step fails.
  # INVARIANT: containers are throwaway (unique names per run).
  # CONSEQUENCE: if removal itself fails, nothing downstream depends on them.
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$EMBED_CONTAINER" >/dev/null 2>&1 || true
  docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Create an isolated network for the test containers.
docker network create "$NETWORK" >/dev/null 2>&1 || true

# Build profile: debug for fast iteration (incremental), release for production.
BUILD_PROFILE="${BUILD_PROFILE:-debug}"
if [ "$BUILD_PROFILE" = "release" ]; then
  CARGO_BUILD_FLAGS="--release"
  BINARY_PATH="/work/target/release/tdb-search"
  echo "→ building RELEASE binary in $BUILD_IMAGE"
else
  CARGO_BUILD_FLAGS=""
  BINARY_PATH="/work/target/debug/tdb-search"
  echo "→ building DEBUG binary in $BUILD_IMAGE (incremental, fast iteration)"
fi

TARGET_VOLUME="${TARGET_VOLUME:-tdb-search-target}"
docker run --rm \
  --user "$HOST_UID:$HOST_GID" \
  -e HOME=/tmp/build-home \
  -e CARGO_HOME=/cargo-registry \
  -v "$ROOT":/work \
  -v "$CARGO_VOLUME":/cargo-registry \
  -v "$TARGET_VOLUME":/work/target \
  -w /work \
  "$BUILD_IMAGE" cargo build $CARGO_BUILD_FLAGS

echo "→ starting Ollama embeddings sidecar"
# Check if there's already an Ollama with the model available on the compose stack.
# If tdb-search-embeddings-1 is running, connect the engine to its network directly.
OLLAMA_URL=""
EXTRA_DOCKER_ARGS=""
EMBED_NET=$(docker inspect tdb-search-embeddings-1 --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null || true)
if [ -n "$EMBED_NET" ] && curl -fsS "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  echo "  (reusing compose Ollama on network $EMBED_NET)"
  OLLAMA_URL="http://tdb-search-embeddings-1:11434"
elif curl -fsS "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  echo "  (reusing host Ollama at localhost:11434)"
  OLLAMA_URL="http://host.docker.internal:11434"
  EXTRA_DOCKER_ARGS="--add-host=host.docker.internal:host-gateway"
else
  # Start a fresh Ollama and pull the model.
  echo "  (no Ollama found; starting a fresh one)"
  docker run -d --name "$EMBED_CONTAINER" \
    --network "$NETWORK" \
    -v tdb-search-ollama-itest:/root/.ollama \
    ollama/ollama:latest >/dev/null

  # Wait for Ollama to come up.
  for _ in $(seq 1 60); do
    if docker exec "$EMBED_CONTAINER" curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  # Pull the model (skip if already present).
  echo "  pulling $EMBED_MODEL model (may take a while on first run)..."
  docker exec "$EMBED_CONTAINER" ollama pull "$EMBED_MODEL" >/dev/null 2>&1 || true
  OLLAMA_URL="http://$EMBED_CONTAINER:11434"
fi

echo "→ starting engine (container $CONTAINER on host port $HOST_PORT)"
# Use the compose network (to reach Ollama sidecar) or the test network.
ENGINE_NETWORK="${EMBED_NET:-$NETWORK}"

# shellcheck disable=SC2086
docker run -d --name "$CONTAINER" \
  --user "$HOST_UID:$HOST_GID" \
  -e HOME=/tmp/build-home \
  --network "$ENGINE_NETWORK" \
  $EXTRA_DOCKER_ARGS \
  -v "$ROOT":/work \
  -v "$TARGET_VOLUME":/work/target \
  -w /work \
  -p "$HOST_PORT":8080 \
  -e TDB_SEARCH_ADMIN_USER=admin \
  -e TDB_SEARCH_ADMIN_SECRET=root \
  -e TDB_SEARCH_EMBED_PROVIDER=openai_compatible \
  -e "TDB_SEARCH_EMBED_URL=$OLLAMA_URL" \
  -e "TDB_SEARCH_MODEL=$EMBED_MODEL" \
  -e "TDB_SEARCH_DIM=$EMBED_DIM" \
  -e TDB_SEARCH_DATA_DIR=/tmp/tdb-search-itest-data \
  -e TDB_SEARCH_TOKENIZER_PATH=/work/spikes/tokenizer/tokenizer.json \
  "$BUILD_IMAGE" "$BINARY_PATH" >/dev/null

# If using a fresh Ollama on the test network, also connect the engine there.
if [ "$ENGINE_NETWORK" != "$NETWORK" ]; then
  docker network connect "$NETWORK" "$CONTAINER" 2>/dev/null || true
fi

echo "→ waiting for engine readiness (poll, no fixed sleep)"
ready=""
for _ in $(seq 1 100); do
  if curl -fsS "http://localhost:$HOST_PORT/health/live" >/dev/null 2>&1; then
    ready="yes"; break
  fi
  sleep 0.3
done
if [ -z "$ready" ]; then
  echo "✗ engine did not become live; logs:" >&2
  docker logs "$CONTAINER" >&2 || true
  exit 1
fi
echo "  engine live at http://localhost:$HOST_PORT"

echo "→ running mocha integration suite against http://localhost:$HOST_PORT"
TDB_SEARCH_URL="http://localhost:$HOST_PORT" \
  TDB_SEARCH_ADMIN_USER=admin \
  TDB_SEARCH_ADMIN_SECRET=root \
  npx mocha --timeout 60000
