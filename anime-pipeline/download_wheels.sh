#!/bin/bash
# ============================================================
#  Download all Python wheels for Linux offline install.
#  Run on any Linux machine WITH internet access.
#  Then copy offline_wheels/ to the target server.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/offline_wheels"
REQ="$SCRIPT_DIR/requirements.txt"
ASR_REQ="$SCRIPT_DIR/requirements-asr.txt"

# Use Tsinghua mirror (faster in China, bypasses GFW)
PIP_TRUST="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

echo "============================================"
echo "  Offline Wheel Downloader (Linux)"
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
if ! "$PIP" install --upgrade pip $PIP_TRUST -q 2>/dev/null; then
    echo "[WARNING] PyPI may be slow or unreachable."
fi

# ============================================================
# Step 1: pip + setuptools (must succeed for bootstrap)
# ============================================================
echo ""
echo "[1/4] Downloading pip + setuptools..."
"$PIP" download pip setuptools wheel -d "$WHEELS_DIR" $PIP_TRUST --no-deps || {
    echo ""
    echo "ERROR: Cannot reach PyPI at all."
    echo ""
    echo "  Possible fixes:"
    echo "  1. Use a VPN or different network"
    echo "  2. Set proxy: export HTTP_PROXY=http://proxy:port"
    echo "  3. Try a mirror:"
    echo "     $PIP download -i https://pypi.tuna.tsinghua.edu.cn/simple pip setuptools -d offline_wheels"
    echo ""
    exit 1
}

# ============================================================
# Step 2: Core dependencies (current platform)
# ============================================================
# Step 2: Core dependencies — two-pass for cross-platform coverage
# Pass A: normal download (deps resolved correctly)
# Pass B: Linux binary wheels (overwrites platform-specific ones)
# ============================================================
echo ""
echo "[2/4] Core dependencies..."
"$PIP" download -r "$REQ" -d "$WHEELS_DIR" $PIP_TRUST || echo "[WARNING] Some packages failed."

if [ "$(uname -s)" != "Linux" ]; then
    echo "        Adding Linux binary wheels..."
    "$PIP" download -r "$REQ" \
        -d "$WHEELS_DIR" \
        --no-deps \
        --platform manylinux_2_17_x86_64 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --python-version 310 \
        $PIP_TRUST 2>/dev/null || echo "[WARNING] Some Linux wheels failed."
fi

# ============================================================
# Step 4: ASR (optional)
# ============================================================
echo ""
echo "[4/4] ASR dependencies (optional)..."
"$PIP" download -r "$ASR_REQ" -d "$WHEELS_DIR" $PIP_TRUST 2>/dev/null || echo "[WARNING] Some ASR packages failed."

echo ""
echo "============================================"
echo "  Done!"
echo "  $(ls "$WHEELS_DIR"/*.whl 2>/dev/null | wc -l) wheels downloaded"
echo ""
echo "  Copy to server:"
echo "  scp -r offline_wheels/ devuser@192.168.103.163:/mnt/project/ComicCut/anime-pipeline/"
echo ""
echo "  Then on server: ./setup.sh"
echo "============================================"
