#!/usr/bin/env bash
# ============================================================
#  Video Pipeline — Linux/macOS 一键启动脚本
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

VENV_DIR="$PROJECT_ROOT/venv"
PYTHON="$VENV_DIR/bin/python"

# Check if venv exists
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "        Please run: bash setup.sh"
    echo ""
    exit 1
fi

# Check if config exists
if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
    echo "[ERROR] config.yaml not found at $PROJECT_ROOT"
    echo "        Please ensure config.yaml exists."
    exit 1
fi

# Set Torch Hub cache directory for offline model loading
export TORCH_HOME="$PROJECT_ROOT/checkpoints"

# Fix PyTorch encoding
export PYTHONUTF8=1

echo ""
echo "============================================"
echo "  Video Pipeline — Starting..."
echo "============================================"
echo ""
echo "  Press Ctrl+C at any time to stop gracefully."
echo "============================================"
echo ""

# Launch pipeline with all arguments passed through
exec "$PYTHON" "$PROJECT_ROOT/main.py" "$@"
