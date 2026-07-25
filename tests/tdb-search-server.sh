#!/bin/bash

# tdb-search-server.sh
# Manages a local tdb-search server (port 7372) and the paired TerminusDB
# test server (port 7373) for integration testing.
#
# restart stops and starts both servers so the full stack is fresh.
# Mirrors the terminusdb-test-server.sh pattern: start/stop/restart/status/logs.
# Tracks tdb-search PID in .tdb-search-test.pid, logs to .tdb-search-test.log.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$SCRIPT_DIR/.tdb-search-test.pid"
LOG_FILE="$SCRIPT_DIR/.tdb-search-test.log"

# TerminusDB test server paths (sibling repo)
TDB_REPO_ROOT="${TERMINUSDB_REPO_ROOT:-$(cd "$PROJECT_ROOT/../terminusdb" 2>/dev/null && pwd)}"
TDB_SCRIPT="$TDB_REPO_ROOT/tests/terminusdb-test-server.sh"

# Defaults — overridable via environment
SERVER_PORT="${TDB_SEARCH_PORT:-7372}"
ADMIN_USER="${TDB_SEARCH_ADMIN_USER:-admin}"
ADMIN_SECRET="${TDB_SEARCH_ADMIN_SECRET:-root}"
DATA_DIR="${TDB_SEARCH_DATA_DIR:-/tmp/tdb-search-data}"
TOKENIZER_PATH="${TDB_SEARCH_TOKENIZER_PATH:-$PROJECT_ROOT/assets/tokenizer.json.bz2}"
EMBED_URL="${TDB_SEARCH_EMBED_URL:-http://127.0.0.1:11434}"
MODEL="${TDB_SEARCH_MODEL:-nomic-embed-text-v2-moe}"
DIM="${TDB_SEARCH_DIM:-768}"
BINARY="${TDB_SEARCH_BINARY:-$PROJECT_ROOT/target/release/tdb-search}"

function start_server() {
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "tdb-search is already running (PID: $pid)"
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
        echo "Building tdb-search (release)..."
        cd "$PROJECT_ROOT"
        cargo build --release
    fi

    echo "Starting tdb-search server..."

    cd "$PROJECT_ROOT"
    export TDB_SEARCH_PORT="$SERVER_PORT"
    export TDB_SEARCH_ADMIN_USER="$ADMIN_USER"
    export TDB_SEARCH_ADMIN_SECRET="$ADMIN_SECRET"
    export TDB_SEARCH_DATA_DIR="$DATA_DIR"
    export TDB_SEARCH_TOKENIZER_PATH="$TOKENIZER_PATH"
    export TDB_SEARCH_EMBED_URL="$EMBED_URL"
    export TDB_SEARCH_MODEL="$MODEL"
    export TDB_SEARCH_DIM="$DIM"
    export TDB_SEARCH_EMBED_CACHE_SIZE="${TDB_SEARCH_EMBED_CACHE_SIZE:-3000000}"

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
    echo "tdb-search starting (PID: $pid)..."
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

        # tdb-search has no health endpoint; check if port is listening
        if curl -s -f --max-time 2 -o /dev/null "http://127.0.0.1:${SERVER_PORT}/" 2>/dev/null; then
            :
        fi
        # Check if port is accepting connections
        if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo " ✓"
            echo "tdb-search is ready!"
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
        echo "No PID file found. tdb-search may not be running."
        return 0
    fi

    local pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping tdb-search (PID: $pid)..."
        kill "$pid"

        local max_wait=130
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
    # Configure the TerminusDB indexer to push to this tdb-search instance.
    export TERMINUSDB_INDEXER_BACKEND="${TERMINUSDB_INDEXER_BACKEND:-http_tdb_search}"
    export TERMINUSDB_TDB_SEARCH_ENDPOINT="${TERMINUSDB_TDB_SEARCH_ENDPOINT:-http://127.0.0.1:${SERVER_PORT}}"
    export TERMINUSDB_SERVER_PORT="${TERMINUSDB_SERVER_PORT:-7373}"
    echo "  Indexer backend:  $TERMINUSDB_INDEXER_BACKEND"
    echo "  Indexer endpoint: $TERMINUSDB_TDB_SEARCH_ENDPOINT"
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
    export TERMINUSDB_INDEXER_BACKEND="${TERMINUSDB_INDEXER_BACKEND:-http_tdb_search}"
    export TERMINUSDB_TDB_SEARCH_ENDPOINT="${TERMINUSDB_TDB_SEARCH_ENDPOINT:-http://127.0.0.1:${SERVER_PORT}}"
    export TERMINUSDB_SERVER_PORT="${TERMINUSDB_SERVER_PORT:-7373}"
    "$TDB_SCRIPT" restart 2>/dev/null || true
}

function status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "tdb-search is running (PID: $pid)"
            echo "  URL:    http://127.0.0.1:${SERVER_PORT}"
            echo "  Model:  $MODEL"
            echo "  Logs:   $LOG_FILE"
        else
            echo "tdb-search is not running (stale PID file)"
        fi
    else
        echo "tdb-search is not running"
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
        echo "  start       - Start tdb-search (7372) and TerminusDB (7373)"
        echo "  stop        - Stop both servers"
        echo "  restart     - Restart both servers"
        echo "  status      - Check if servers are running"
        echo "  logs [N]    - Show last N log lines (default: 50)"
        echo ""
        echo "Environment variables:"
        echo "  TDB_SEARCH_PORT           - tdb-search listen port (default: 7372)"
        echo "  TERMINUSDB_REPO_ROOT      - Path to terminusdb repo (default: sibling)"
        echo "  TDB_SEARCH_ADMIN_USER     - Admin user (default: admin)"
        echo "  TDB_SEARCH_ADMIN_SECRET   - Admin secret (default: root)"
        echo "  TDB_SEARCH_DATA_DIR       - Data directory (default: /tmp/tdb-search-data)"
        echo "  TDB_SEARCH_TOKENIZER_PATH - Tokenizer path (default: assets/tokenizer.json.bz2)"
        echo "  TDB_SEARCH_EMBED_URL      - Ollama URL (default: http://127.0.0.1:11434)"
        echo "  TDB_SEARCH_MODEL          - Embedding model (default: nomic-embed-text-v2-moe)"
        echo "  TDB_SEARCH_DIM            - Embedding dimension (default: 768)"
        echo "  TERMINUSDB_INDEXER_BACKEND       - Indexer backend (default: http_tdb_search)"
        echo "  TERMINUSDB_TDB_SEARCH_ENDPOINT   - Indexer push target (default: http://127.0.0.1:7372)"
        exit 1
        ;;
esac
