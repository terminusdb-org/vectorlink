#!/bin/bash

# vectorlink-server.sh
# Manages a local vectorlink server (port 7372) and the paired TerminusDB
# test server (port 7373) for integration testing.
#
# restart stops and starts both servers so the full stack is fresh.
# Mirrors the terminusdb-test-server.sh pattern: start/stop/restart/status/logs.
# Tracks vectorlink PID in .vectorlink-test.pid, logs to .vectorlink-test.log.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$SCRIPT_DIR/.vectorlink-test.pid"
LOG_FILE="$SCRIPT_DIR/.vectorlink-test.log"

# TerminusDB test server paths (sibling repo)
TDB_REPO_ROOT="${TERMINUSDB_REPO_ROOT:-$(cd "$PROJECT_ROOT/../terminusdb" 2>/dev/null && pwd)}"
TDB_SCRIPT="$TDB_REPO_ROOT/tests/terminusdb-test-server.sh"

# Defaults — overridable via environment
SERVER_PORT="${VECTORLINK_PORT:-7372}"
ADMIN_USER="${VECTORLINK_ADMIN_USER:-admin}"
ADMIN_SECRET="${VECTORLINK_ADMIN_SECRET:-root}"
DATA_DIR="${VECTORLINK_DATA_DIR:-/tmp/vectorlink-data}"
TOKENIZER_PATH="${VECTORLINK_TOKENIZER_PATH:-$PROJECT_ROOT/assets/tokenizer.json.bz2}"
EMBED_URL="${VECTORLINK_EMBED_URL:-http://127.0.0.1:11434}"
MODEL="${VECTORLINK_MODEL:-nomic-embed-text-v2-moe}"
DIM="${VECTORLINK_DIM:-768}"
BINARY="${VECTORLINK_BINARY:-$PROJECT_ROOT/target/release/vectorlink}"

function start_server() {
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "vectorlink is already running (PID: $pid)"
            return 0
        else
            echo "Stale PID file found. Cleaning up..."
            rm -f "$PID_FILE"
        fi
    fi

    # Check port
    if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "ERROR: Port $SERVER_PORT is already in use:"
        lsof -Pi :$SERVER_PORT -sTCP:LISTEN
        return 1
    fi

    # Build if binary doesn't exist
    if [ ! -f "$BINARY" ]; then
        echo "Building vectorlink (release)..."
        cd "$PROJECT_ROOT"
        cargo build --release
    fi

    echo "Starting vectorlink server..."

    cd "$PROJECT_ROOT"
    export VECTORLINK_PORT="$SERVER_PORT"
    export VECTORLINK_ADMIN_USER="$ADMIN_USER"
    export VECTORLINK_ADMIN_SECRET="$ADMIN_SECRET"
    export VECTORLINK_DATA_DIR="$DATA_DIR"
    export VECTORLINK_TOKENIZER_PATH="$TOKENIZER_PATH"
    export VECTORLINK_EMBED_URL="$EMBED_URL"
    export VECTORLINK_MODEL="$MODEL"
    export VECTORLINK_DIM="$DIM"
    export VECTORLINK_EMBED_CACHE_SIZE="${VECTORLINK_EMBED_CACHE_SIZE:-3000000}"

    # Start in a new session so it survives the script exiting
    python3 -c "
import os, subprocess, sys
log_file = sys.argv[1]
pid_file = sys.argv[2]
cmd = sys.argv[3:]
proc = subprocess.Popen(cmd, stdout=open(log_file, 'w'), stderr=subprocess.STDOUT, start_new_session=True, env=os.environ)
with open(pid_file, 'w') as f:
    f.write(str(proc.pid))
" "$LOG_FILE" "$PID_FILE" "$BINARY"

    local pid
    pid=$(cat "$PID_FILE")
    echo "vectorlink starting (PID: $pid)..."
    echo "  Port:        $SERVER_PORT"
    echo "  Data dir:    $DATA_DIR"
    echo "  Model:       $MODEL"
    echo "  Embed URL:   $EMBED_URL"
    echo "  Logs:        $LOG_FILE"

    # Wait for readiness
    echo -n "Waiting for server to be ready"
    local max_wait=30
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo " ✗"
            echo "ERROR: Server process died unexpectedly"
            echo "Check logs: $LOG_FILE"
            cat "$LOG_FILE"
            rm -f "$PID_FILE"
            return 1
        fi

        # vectorlink has no health endpoint; check if port is listening
        if curl -s -f --max-time 2 -o /dev/null "http://127.0.0.1:${SERVER_PORT}/" 2>/dev/null; then
            :
        fi
        # Check if port is accepting connections
        if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo " ✓"
            echo "vectorlink is ready!"
            echo "  URL: http://127.0.0.1:${SERVER_PORT}"
            return 0
        fi
        echo -n "."
        sleep 0.5
        waited=$((waited + 1))
    done

    echo " ✗"
    echo "ERROR: Server failed to start within ${max_wait}s"
    echo "Check logs: $LOG_FILE"
    cat "$LOG_FILE"
    stop_server
    return 1
}

function stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. vectorlink may not be running."
        return 0
    fi

    local pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping vectorlink (PID: $pid)..."
        kill "$pid"

        local max_wait=140
        local waited=0
        while ps -p "$pid" > /dev/null 2>&1 && [ $waited -lt $max_wait ]; do
            sleep 1
            waited=$((waited + 1))
        done

        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Forcing shutdown..."
            kill -9 "$pid" 2>/dev/null || true
        fi

        echo "Server stopped."
    else
        echo "Server not running (stale PID file)."
    fi

    rm -f "$PID_FILE"
}

function restart_server() {
    stop_server
    sleep 1
    start_server
}

function start_terminusdb() {
    if [ -z "$TDB_REPO_ROOT" ] || [ ! -f "$TDB_SCRIPT" ]; then
        echo "TerminusDB repo not found at $TDB_REPO_ROOT — skipping."
        echo "Set TERMINUSDB_REPO_ROOT to enable paired server management."
        return 0
    fi
    echo "Starting paired TerminusDB test server (port ${TERMINUSDB_SERVER_PORT:-7373})..."
    # Configure the TerminusDB indexer to push to this vectorlink instance.
    export TERMINUSDB_INDEXER_BACKEND="${TERMINUSDB_INDEXER_BACKEND:-http_vectorlink}"
    export TERMINUSDB_VECTORLINK_ENDPOINT="${TERMINUSDB_VECTORLINK_ENDPOINT:-http://127.0.0.1:${SERVER_PORT}}"
    export TERMINUSDB_SERVER_PORT="${TERMINUSDB_SERVER_PORT:-7373}"
    echo "  Indexer backend:  $TERMINUSDB_INDEXER_BACKEND"
    echo "  Indexer endpoint: $TERMINUSDB_VECTORLINK_ENDPOINT"
    "$TDB_SCRIPT" start
}

function stop_terminusdb() {
    if [ -z "$TDB_REPO_ROOT" ] || [ ! -f "$TDB_SCRIPT" ]; then
        return 0
    fi
    export TERMINUSDB_SERVER_PORT="${TERMINUSDB_SERVER_PORT:-7373}"
    "$TDB_SCRIPT" stop 2>/dev/null || true
}

function restart_terminusdb() {
    if [ -z "$TDB_REPO_ROOT" ] || [ ! -f "$TDB_SCRIPT" ]; then
        return 0
    fi
    export TERMINUSDB_INDEXER_BACKEND="${TERMINUSDB_INDEXER_BACKEND:-http_vectorlink}"
    export TERMINUSDB_VECTORLINK_ENDPOINT="${TERMINUSDB_VECTORLINK_ENDPOINT:-http://127.0.0.1:${SERVER_PORT}}"
    export TERMINUSDB_SERVER_PORT="${TERMINUSDB_SERVER_PORT:-7373}"
    "$TDB_SCRIPT" restart 2>/dev/null || true
}

function status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "vectorlink is running (PID: $pid)"
            echo "  URL:    http://127.0.0.1:${SERVER_PORT}"
            echo "  Model:  $MODEL"
            echo "  Logs:   $LOG_FILE"
        else
            echo "vectorlink is not running (stale PID file)"
        fi
    else
        echo "vectorlink is not running"
    fi
    if [ -n "$TDB_REPO_ROOT" ] && [ -f "$TDB_SCRIPT" ]; then
        export TERMINUSDB_SERVER_PORT="${TERMINUSDB_SERVER_PORT:-7373}"
        "$TDB_SCRIPT" status 2>/dev/null || true
    fi
}

function logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -n "${1:-50}" "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
        return 1
    fi
}

# Main command dispatcher
case "${1:-}" in
    start)
        start_server
        start_terminusdb
        ;;
    stop)
        stop_server
        stop_terminusdb
        ;;
    restart)
        restart_server
        restart_terminusdb
        ;;
    status)
        status
        ;;
    logs)
        shift
        logs "$@"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [N]}"
        echo ""
        echo "Commands:"
        echo "  start       - Start vectorlink (7372) and TerminusDB (7373)"
        echo "  stop        - Stop both servers"
        echo "  restart     - Restart both servers"
        echo "  status      - Check if servers are running"
        echo "  logs [N]    - Show last N log lines (default: 50)"
        echo ""
        echo "Environment variables:"
        echo "  VECTORLINK_PORT           - vectorlink listen port (default: 7372)"
        echo "  TERMINUSDB_REPO_ROOT      - Path to terminusdb repo (default: sibling)"
        echo "  VECTORLINK_ADMIN_USER     - Admin user (default: admin)"
        echo "  VECTORLINK_ADMIN_SECRET   - Admin secret (default: root)"
        echo "  VECTORLINK_DATA_DIR       - Data directory (default: /tmp/vectorlink-data)"
        echo "  VECTORLINK_TOKENIZER_PATH - Tokenizer path (default: assets/tokenizer.json.bz2)"
        echo "  VECTORLINK_EMBED_URL      - Ollama URL (default: http://127.0.0.1:11434)"
        echo "  VECTORLINK_MODEL          - Embedding model (default: nomic-embed-text-v2-moe)"
        echo "  VECTORLINK_DIM            - Embedding dimension (default: 768)"
        echo "  TERMINUSDB_INDEXER_BACKEND       - Indexer backend (default: http_vectorlink)"
        echo "  TERMINUSDB_VECTORLINK_ENDPOINT   - Indexer push target (default: http://127.0.0.1:7372)"
        exit 1
        ;;
esac
