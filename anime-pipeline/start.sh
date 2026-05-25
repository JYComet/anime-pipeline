#!/bin/bash
# ============================================================
#  Anime Pipeline — Quick Start (Linux/macOS)
#  Run setup.sh first if this is a fresh install.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# If venv doesn't exist, redirect to setup
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not found. Running setup first..."
    echo ""
    exec bash "$PROJECT_ROOT/setup.sh"
fi

echo "============================================"
echo "  Anime Pipeline Server"
echo "============================================"
echo ""

# Free port 5800 if in use
if command -v fuser &>/dev/null; then
    PID=$(fuser 5800/tcp 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Port 5800 in use by PID $PID. Killing..."
        kill -9 "$PID" 2>/dev/null || true
        sleep 1
    fi
elif command -v lsof &>/dev/null; then
    PID=$(lsof -ti :5800 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Port 5800 in use by PID $PID. Killing..."
        kill -9 "$PID" 2>/dev/null || true
        sleep 1
    fi
fi

echo "Starting server..."
echo "Frontend: http://localhost:5800"
echo ""

cd "$PROJECT_ROOT/scripts"
exec "$VENV_PYTHON" server.py
