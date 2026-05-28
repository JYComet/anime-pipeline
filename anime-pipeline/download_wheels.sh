#!/bin/bash
# ============================================================
#  Offline Wheel Downloader (Linux/macOS)
#  Run on any machine WITH internet access.
#  Then copy offline_wheels/ to the target server.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/offline_wheels"

echo "============================================"
echo "  Offline Wheel Downloader"
echo "============================================"
echo ""
echo "  Output: $WHEELS_DIR"
echo ""

mkdir -p "$WHEELS_DIR"

# Find a working pip
if command -v pip3 &>/dev/null; then
    PIP="pip3"
elif command -v pip &>/dev/null; then
    PIP="pip"
else
    echo "Error: pip not found. Install python3-pip first."
    exit 1
fi

echo "Testing PyPI connectivity..."
if ! "$PIP" install --upgrade pip -q 2>/dev/null; then
    echo "[WARNING] PyPI may be slow or unreachable."
fi

echo ""
echo "[1/3] Downloading pip + setuptools + wheel (bootstrap)..."
"$PIP" download pip setuptools wheel -d "$WHEELS_DIR" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-deps || {
    echo ""
    echo "ERROR: Cannot reach PyPI at all."
    echo "  Use a VPN, different network, or set proxy: export HTTP_PROXY=http://proxy:port"
    echo ""
    exit 1
}

echo ""
echo "[2/3] Downloading all dependencies..."
python3 "$SCRIPT_DIR/tools/download_wheels.py" "$@"

echo ""
echo "[3/3] Done!"
echo "  $(ls "$WHEELS_DIR"/*.whl 2>/dev/null | wc -l) wheels downloaded"
echo ""
echo "  For Linux offline install, first download Linux wheels:"
echo "    python3 tools/download_wheels.py --linux"
echo ""
echo "  Then on server: ./setup.sh"
echo "============================================"
