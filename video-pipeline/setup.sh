#!/usr/bin/env bash
# ============================================================
#  Video Pipeline — Linux/macOS 一键配置脚本
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "============================================"
echo "  Video Pipeline — One-Click Setup"
echo "============================================"
echo ""
echo "  Project : $PROJECT_ROOT"
echo ""

# ============================================================
# Step 1 — Check Python
# ============================================================
echo "[1/8] Checking Python installation..."

PYTHON_BIN=""
for p in python3 python; do
    if command -v "$p" &>/dev/null; then
        PYVER=$("$p" --version 2>&1)
        PYTHON_BIN="$p"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python not found. Please install Python 3.10+."
    echo "        Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "        CentOS/RHEL:   sudo yum install python3 python3-pip"
    echo "        macOS:         brew install python3"
    exit 1
fi

echo "        Found: $PYVER  ($PYTHON_BIN)"

# ============================================================
# Step 2 — Create virtual environment
# ============================================================
echo ""
echo "[2/8] Setting up virtual environment..."

VENV_DIR="$PROJECT_ROOT/venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if [ -f "$PYTHON" ]; then
    echo "        venv already exists, skipping creation."
else
    echo "        Creating venv at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "        venv created."
fi

"$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1

# ============================================================
# Step 3 — Create data directories
# ============================================================
echo ""
echo "[3/8] Creating data directories..."

for d in \
    "$PROJECT_ROOT/data/input" \
    "$PROJECT_ROOT/data/output" \
    "$PROJECT_ROOT/data/temp" \
    "$PROJECT_ROOT/data/logs"; do
    mkdir -p "$d"
done
echo "        All data directories ready."

# ============================================================
# Step 4 — Detect CUDA and install PyTorch
# ============================================================
echo ""
echo "[4/8] Detecting GPU and installing PyTorch..."

if command -v nvidia-smi &>/dev/null; then
    echo "        NVIDIA GPU detected."
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
    echo "        Installing PyTorch with CUDA 12.6 support..."
    "$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 || {
        echo "[WARNING] PyTorch CUDA install failed. Trying CPU version..."
        "$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
else
    echo "        No NVIDIA GPU detected."
    echo "        Installing PyTorch CPU version..."
    "$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ============================================================
# Step 5 — Install Python dependencies
# ============================================================
echo ""
echo "[5/8] Installing Python dependencies..."

"$PIP" install -r "$PROJECT_ROOT/requirements.txt" || {
    echo "[WARNING] Some packages failed to install."
    echo "          Check the output above for details."
}

echo "        Dependencies installed."

# ============================================================
# Step 6 — Verify ffmpeg
# ============================================================
echo ""
echo "[6/8] Checking ffmpeg..."

if command -v ffmpeg &>/dev/null; then
    FFVER=$(ffmpeg -version 2>&1 | head -1)
    echo "        ffmpeg found: $FFVER"
else
    echo ""
    echo "  [WARNING] ffmpeg not found on PATH!"
    echo ""
    echo "  ffmpeg is REQUIRED for audio/video processing."
    echo "  Install it:"
    echo "    Ubuntu/Debian: sudo apt install ffmpeg"
    echo "    CentOS/RHEL:   sudo yum install ffmpeg"
    echo "    macOS:         brew install ffmpeg"
    echo ""
    echo "  Or set the full path in config.yaml: tools.ffmpeg"
    echo ""
fi

# ============================================================
# Step 7 — Copy Demucs model from original project (if exists)
# ============================================================
echo ""
echo "[7/8] Checking Demucs model files..."

ORIGIN_CHECKPOINTS="$SCRIPT_DIR/../anime-pipeline/checkpoints/torch_hub"
ORIGIN_CHECKPOINTS2="$SCRIPT_DIR/../anime-pipeline/checkpoints/hub/checkpoints"
LOCAL_CHECKPOINTS="$PROJECT_ROOT/checkpoints/hub/checkpoints"

mkdir -p "$LOCAL_CHECKPOINTS"
LOCAL_COUNT=$(find "$LOCAL_CHECKPOINTS" -maxdepth 1 -name "*.th" 2>/dev/null | wc -l)

if [ "$LOCAL_COUNT" -gt 0 ]; then
    echo "        Model files already present ($LOCAL_COUNT files)."
else
    COPIED=0
    if ls "$ORIGIN_CHECKPOINTS"/*.th >/dev/null 2>&1; then
        ORIGIN_COUNT=$(ls "$ORIGIN_CHECKPOINTS"/*.th 2>/dev/null | wc -l)
        echo "        Copying $ORIGIN_COUNT model files from original project..."
        cp "$ORIGIN_CHECKPOINTS"/*.th "$LOCAL_CHECKPOINTS/"
        COPIED=1
    elif ls "$ORIGIN_CHECKPOINTS2"/*.th >/dev/null 2>&1; then
        ORIGIN_COUNT=$(ls "$ORIGIN_CHECKPOINTS2"/*.th 2>/dev/null | wc -l)
        echo "        Copying $ORIGIN_COUNT model files from original project (new structure)..."
        cp "$ORIGIN_CHECKPOINTS2"/*.th "$LOCAL_CHECKPOINTS/"
        COPIED=1
    fi

    if [ "$COPIED" -eq 1 ]; then
        echo "        Models copied successfully."
    else
        echo ""
        echo "  [INFO] Demucs htdemucs_ft model not found locally."
        echo ""
        echo "  On first run, Demucs will automatically download the model"
        echo "  from PyTorch Hub (~330 MB). This requires internet access."
        echo ""
        echo "  To skip this, copy .th files from the original project:"
        echo "    $ORIGIN_CHECKPOINTS"
        echo "  into:"
        echo "    $LOCAL_CHECKPOINTS"
        echo ""
    fi
fi

# ============================================================
# Step 8 — Done
# ============================================================
echo ""
echo "[8/8] Setup complete!"
echo ""
echo "============================================"
echo "  Video Pipeline setup finished!"
echo ""
echo "  To start processing:"
echo "    - Run: bash start.sh"
echo "    - Or:  venv/bin/python main.py"
echo ""
echo "  Edit config.yaml to customize settings."
echo "============================================"
echo ""

# Make start script executable
chmod +x "$PROJECT_ROOT/start.sh" 2>/dev/null || true
