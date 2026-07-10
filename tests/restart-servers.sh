#!/usr/bin/env bash
set -euo pipefail

# Vectorlink integration test server restart script
# Starts TerminusDB on port 7373 and vectorlink on port 7374

TDB_PORT="${TERMINUSDB_PORT:-7373}"
VL_PORT="${VECTORLINK_PORT:-7374}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TDB_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)/terminusdb"
TDB_BIN="${TDB_BIN:-$TDB_HOME/terminusdb}"
TDB_PLUGINS="${TDB_PLUGINS:-$TDB_HOME/plugins}"
VL_BIN="${VL_BIN:-$(cd "$(dirname "$0")/.." && pwd)/target/release/terminusdb-semantic-indexer}"
TDB_STORE="${TDB_STORE:-/tmp/vl_tdb_store}"
VL_DIR="${VL_DIR:-/tmp/vl_index_dir}"
TDB_LOG="${TDB_LOG:-/tmp/vl_tdb_server.log}"
VL_LOG="${VL_LOG:-/tmp/vl_server.log}"

echo "Stopping existing processes on ports $TDB_PORT and $VL_PORT..."
kill "$(lsof -ti :"$TDB_PORT" 2>/dev/null)" 2>/dev/null || true
kill "$(lsof -ti :"$VL_PORT" 2>/dev/null)" 2>/dev/null || true
sleep 2

echo "Cleaning storage..."
rm -rf "$TDB_STORE" "$VL_DIR" /tmp/vl_plugins
mkdir -p "$TDB_STORE" "$VL_DIR" /tmp/vl_plugins
# Copy only the vectorlink plugin to avoid conflicts with tdb_search.pl
cp "$TDB_PLUGINS/vectorlink.pl" /tmp/vl_plugins/ 2>/dev/null || true

echo "Starting TerminusDB on port $TDB_PORT..."
nohup env \
  TERMINUSDB_SERVER_PORT="$TDB_PORT" \
  TERMINUSDB_SERVER_DB_PATH="$TDB_STORE" \
  TERMINUSDB_PLUGINS_PATH=/tmp/vl_plugins \
  TERMINUSDB_INDEXER_BACKEND=http_vectorlink \
  TERMINUSDB_SEMANTIC_INDEXER_ENDPOINT="http://localhost:$VL_PORT" \
  TERMINUSDB_INSECURE_USER_HEADER_ENABLED=true \
  TERMINUSDB_INSECURE_USER_HEADER=x-terminusdb-user \
  "$TDB_BIN" serve -m root \
  > "$TDB_LOG" 2>&1 &
TDB_PID=$!
disown
echo "  TerminusDB PID: $TDB_PID, log: $TDB_LOG"

echo "Waiting for TerminusDB to start..."
for i in $(seq 1 15); do
  if curl -s -o /dev/null -w '%{http_code}' -u admin:root "http://localhost:$TDB_PORT/api/info" 2>/dev/null | grep -q 200; then
    echo "  TerminusDB is up"
    break
  fi
  sleep 1
done

echo "Starting vectorlink on port $VL_PORT..."
nohup "$VL_BIN" serve \
  --content-endpoint "http://localhost:$TDB_PORT/api/index" \
  --user-forward-header x-terminusdb-user \
  --directory "$VL_DIR" \
  --port "$VL_PORT" \
  > "$VL_LOG" 2>&1 &
VL_PID=$!
disown
echo "  Vectorlink PID: $VL_PID, log: $VL_LOG"

echo "Waiting for vectorlink to start..."
for i in $(seq 1 10); do
  if curl -s -o /dev/null -w '%{http_code}' "http://localhost:$VL_PORT/statistics" 2>/dev/null | grep -q 200; then
    echo "  Vectorlink is up"
    break
  fi
  sleep 1
done

echo ""
echo "Both services are running:"
echo "  TerminusDB:  http://localhost:$TDB_PORT  (log: $TDB_LOG)"
echo "  Vectorlink:  http://localhost:$VL_PORT  (log: $VL_LOG)"
echo ""
echo "Run tests with:"
echo "  cd tests && TERMINUSDB_BASE_URL=http://localhost:$TDB_PORT VECTORLINK_BASE_URL=http://localhost:$VL_PORT npx mocha"
