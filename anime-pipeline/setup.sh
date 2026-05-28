#!/bin/bash
# ============================================================
#  Anime Pipeline — One-Click Setup & Launch (Linux/macOS)
#  Supports both online and offline install modes.
#  Auto-detects paths for new environments.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
COMICUT_ROOT="$(dirname "$PROJECT_ROOT")"

VENV_DIR="$PROJECT_ROOT/venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
SETUP_DONE_FILE="$PROJECT_ROOT/.setup_done"
WHEELS_DIR="$PROJECT_ROOT/offline_wheels"
SETTINGS_FILE="$PROJECT_ROOT/data/settings.json"
DATA_DIR="$PROJECT_ROOT/data"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================"
echo "  Anime Pipeline — One-Click Setup"
echo "============================================"
echo ""
echo "  Project : $PROJECT_ROOT"
echo "  Parent  : $COMICUT_ROOT"

# Detect offline mode
if [ -d "$WHEELS_DIR" ] && ls "$WHEELS_DIR"/*.whl &>/dev/null 2>&1; then
    OFFLINE=1
    WHEEL_COUNT=$(ls "$WHEELS_DIR"/*.whl 2>/dev/null | wc -l)
    echo "  Mode    : offline ($WHEEL_COUNT wheels found)"
else
    OFFLINE=0
    echo "  Mode    : online"
fi
echo ""

# ============================================================
# Step 1 — Check Python
# ============================================================
echo "[1/10] Checking Python installation..."

PYTHON3=""
for p in python3 python; do
    if command -v "$p" &>/dev/null; then
        ver=$("$p" --version 2>&1)
        PYTHON3="$p"
        echo "        Found: $ver  ($(command -v "$p"))"
        break
    fi
done

if [ -z "$PYTHON3" ]; then
    echo -e "${RED}[ERROR] Python not found on PATH.${NC}"
    echo "        Install Python 3.10+: sudo apt install python3"
    exit 1
fi

# Check system pip3 (fallback if venv fails)
SYSTEM_PIP=""
if command -v pip3 &>/dev/null; then
    SYSTEM_PIP="pip3"
elif "$PYTHON3" -m pip --version &>/dev/null 2>&1; then
    SYSTEM_PIP="$PYTHON3 -m pip"
fi

# ============================================================
# Step 2 — Create / activate virtual environment
# ============================================================
echo ""
echo "[2/10] Setting up virtual environment..."

if [ -f "$PYTHON" ] && "$PYTHON" -m pip --version &>/dev/null 2>&1; then
    echo "        venv already exists and working."
else
    if [ -f "$PYTHON" ]; then
        echo "        Broken venv detected, recreating..."
        rm -rf "$VENV_DIR"
    fi
    echo "        Creating venv at $VENV_DIR ..."

    if "$PYTHON3" -m venv --copies --without-pip "$VENV_DIR" 2>/dev/null; then
        :
    else
        # lib64 symlink fails on network mounts; build venv manually
        rm -rf "$VENV_DIR"
        PY_PREFIX=$("$PYTHON3" -c "import sys; print(sys.base_prefix)")
        mkdir -p "$VENV_DIR/bin"
        cp "$(command -v "$PYTHON3")" "$VENV_DIR/bin/python"
        cp -r "$PY_PREFIX/lib" "$VENV_DIR/lib" 2>/dev/null || true
        if [ -d "$PY_PREFIX/lib64" ]; then
            cp -r "$PY_PREFIX/lib64" "$VENV_DIR/lib64" 2>/dev/null || true
        elif [ ! -e "$VENV_DIR/lib64" ]; then
            cp -r "$VENV_DIR/lib" "$VENV_DIR/lib64" 2>/dev/null || true
        fi
        echo "home = $PY_PREFIX" > "$VENV_DIR/pyvenv.cfg"
        echo "include-system-site-packages = false" >> "$VENV_DIR/pyvenv.cfg"
        echo "version = $("$PYTHON3" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" >> "$VENV_DIR/pyvenv.cfg"
    fi

    # Bootstrap pip into venv
    if [ -f "$PYTHON" ] && ! "$PYTHON" -m pip --version &>/dev/null 2>&1; then
        echo "        Installing pip into venv..."
        PIP_INSTALLED=0

        # Method 1: ensurepip
        "$PYTHON" -m ensurepip --upgrade 2>/dev/null && PIP_INSTALLED=1

        # Method 2: pip wheel from offline_wheels
        if [ $PIP_INSTALLED -eq 0 ]; then
            PIP_WHL=$(ls "$WHEELS_DIR"/pip-*.whl 2>/dev/null | head -1)
            if [ -n "$PIP_WHL" ]; then
                "$PYTHON" -c "
import zipfile, sys
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(sys.argv[2])
" "$PIP_WHL" /tmp/pip_bootstrap 2>/dev/null && \
                PYTHONPATH=/tmp/pip_bootstrap "$PYTHON" -m pip install --no-index --force-reinstall --no-deps "$PIP_WHL" 2>/dev/null && PIP_INSTALLED=1
                rm -rf /tmp/pip_bootstrap
            fi
        fi

        # Method 3: download get-pip.py (online only)
        if [ $PIP_INSTALLED -eq 0 ] && [ "$OFFLINE" -eq 0 ]; then
            curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py 2>/dev/null && \
                "$PYTHON" /tmp/get-pip.py 2>/dev/null && PIP_INSTALLED=1
            rm -f /tmp/get-pip.py
        fi

        # Method 4: apt install python3-venv (online)
        if [ $PIP_INSTALLED -eq 0 ] && [ "$OFFLINE" -eq 0 ]; then
            if command -v apt &>/dev/null; then
                sudo apt install -y python3-venv 2>/dev/null && \
                    "$PYTHON" -m ensurepip --upgrade 2>/dev/null && PIP_INSTALLED=1
            fi
        fi
    fi

    if [ ! -f "$PYTHON" ] || ! "$PYTHON" -m pip --version &>/dev/null 2>&1; then
        echo ""
        echo -e "${RED}[ERROR] Cannot create venv with pip.${NC}"
        echo ""
        echo "        Run on a machine with internet access first:"
        echo ""
        echo -e "          ${GREEN}./download_wheels.sh --linux${NC}"
        echo ""
        echo "        This downloads all needed packages to offline_wheels/"
        echo "        Then re-run ./setup.sh — it will install from local files."
        exit 1
    fi
    echo "        venv ready."
fi

# Upgrade pip (offline: skip, online: upgrade)
if [ "$OFFLINE" -eq 0 ]; then
    "$PYTHON" -m pip install --upgrade pip -q 2>/dev/null || true
fi

# ============================================================
# Step 3 — Install Python dependencies
# ============================================================
echo ""
echo "[3/10] Installing Python dependencies..."

if [ -f "$SETUP_DONE_FILE" ]; then
    if "$PYTHON" -c "import fastapi" &>/dev/null 2>&1; then
        echo "        Setup already completed — skipping to launch."
        echo "        (Delete .setup_done to force full reinstall.)"
        cd "$PROJECT_ROOT/scripts"
        exec "$PYTHON" server.py
    else
        echo "        Setup marker found but packages missing. Reinstalling..."
        rm -f "$SETUP_DONE_FILE"
    fi
fi

FAILED_PKGS=""

install_pkg() {
    local req="$1"
    if "$PIP" install --no-index --find-links="$WHEELS_DIR" --only-binary=:all: "$req" 2>/dev/null; then
        return 0
    else
        FAILED_PKGS="$FAILED_PKGS  $req"
        return 1
    fi
}

if [ "$OFFLINE" -eq 1 ]; then
    echo "        Installing from offline wheels..."
    INSTALL_FLAGS="--no-index --find-links=$WHEELS_DIR"
else
    echo "        Installing from PyPI..."
    INSTALL_FLAGS=""
fi

# Install one by one — a single missing dep won't block the rest
while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    install_pkg "$line" || echo "        SKIPPED: $line"
done < "$PROJECT_ROOT/requirements.txt"

if [ -n "$FAILED_PKGS" ]; then
    echo -e "${YELLOW}[WARNING] Failed to install:$FAILED_PKGS${NC}"
    echo "          Run tools/download_wheels.py --linux on a machine with internet."
fi

# Verify at least fastapi is installed
if "$PYTHON" -c "import fastapi" &>/dev/null 2>&1; then
    echo "        Core dependencies installed."
else
    echo -e "${RED}[ERROR] fastapi failed to install. Server cannot start.${NC}"
    echo "        Run: python tools/download_wheels.py --linux on a machine with internet"
    exit 1
fi

# ============================================================
# Step 4 — Optional ASR dependencies
# ============================================================
echo ""
echo "[4/10] Optional ASR dependencies..."

ASR_REQ="$PROJECT_ROOT/requirements-asr.txt"
if [ -f "$ASR_REQ" ]; then
    echo "        ASR (speech recognition) packages are optional but large (~4GB)."
    echo "        Required for: auto-subtitle generation, ASR model comparison."
    echo ""
    read -r -p "        Install ASR dependencies? [y/N]: " ASR_INSTALL
    if [ "$ASR_INSTALL" = "y" ] || [ "$ASR_INSTALL" = "Y" ]; then
        FAILED_ASR=""
        while IFS= read -r line || [ -n "$line" ]; do
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            if ! "$PIP" install --no-index --find-links="$WHEELS_DIR" --only-binary=:all: "$line" 2>/dev/null; then
                FAILED_ASR="$FAILED_ASR  $line"
            fi
        done < "$ASR_REQ"
        if [ -n "$FAILED_ASR" ]; then
            echo -e "${YELLOW}[WARNING] ASR packages failed:$FAILED_ASR${NC}"
        fi
    else
        echo "        Skipped."
    fi
fi

# ============================================================
# Step 5 — System tools
# ============================================================
echo ""
echo "[5/10] Checking system tools..."

check_tool() {
    local name="$1"
    local pkg="${2:-$1}"
    if command -v "$name" &>/dev/null; then
        echo "        $name : found ($(command -v "$name"))"
        return 0
    else
        echo -e "        $name : ${RED}MISSING${NC}"
        MISSING_TOOLS+=("$pkg")
        return 1
    fi
}

MISSING_TOOLS=()
check_tool ffmpeg ffmpeg
check_tool ffprobe ffmpeg
check_tool mkvextract mkvtoolnix
check_tool mkvmerge mkvtoolnix
check_tool mkvinfo mkvtoolnix
check_tool aria2c aria2

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    UNIQUE_PKGS=($(printf '%s\n' "${MISSING_TOOLS[@]}" | sort -u))
    echo ""
    echo "        Missing: ${UNIQUE_PKGS[*]}"
    if [ "$OFFLINE" -eq 1 ]; then
        echo -e "        ${YELLOW}Server is offline — install these manually when possible.${NC}"
        echo "        Commands: sudo apt install -y ${UNIQUE_PKGS[*]}"
    elif command -v apt &>/dev/null; then
        read -r -p "        Install via apt? [Y/n]: " APT_INSTALL
        if [ "$APT_INSTALL" != "n" ] && [ "$APT_INSTALL" != "N" ]; then
            sudo apt update 2>/dev/null || true
            sudo apt install -y "${UNIQUE_PKGS[@]}" || true
        fi
    fi
fi

# ============================================================
# Step 6 — ClearerVoice-Studio (audio denoising)
# ============================================================
echo ""
echo "[6/10] Checking ClearerVoice-Studio..."

CV_DIR="$COMICUT_ROOT/ClearerVoice-Studio-main/ClearerVoice-Studio-main"
CV_CLEARVOICE="$CV_DIR/clearvoice"

if [ -f "$CV_CLEARVOICE/clearvoice.py" ]; then
    echo "        Found at: $CV_DIR"
else
    echo "        ClearerVoice-Studio not found."
    if [ "$OFFLINE" -eq 1 ]; then
        echo -e "        ${YELLOW}Copy the folder from your online machine to:${NC}"
        echo "        $COMICUT_ROOT/ClearerVoice-Studio-main/"
    else
        echo "        It will be cloned from GitHub (~200MB)."
        read -r -p "        Clone now? [Y/n]: " CV_CLONE
        if [ "$CV_CLONE" != "n" ] && [ "$CV_CLONE" != "N" ]; then
            CV_TEMP="$COMICUT_ROOT/ClearerVoice-Studio-temp"
            CV_PARENT="$COMICUT_ROOT/ClearerVoice-Studio-main"
            rm -rf "$CV_TEMP"
            if git clone --depth 1 https://github.com/modelscope/ClearerVoice-Studio.git "$CV_TEMP" 2>&1; then
                mkdir -p "$CV_PARENT"
                mv "$CV_TEMP" "$CV_DIR"
                rm -rf "$CV_TEMP"
                echo "        Cloned successfully."
            else
                echo -e "${YELLOW}[WARNING] Git clone failed.${NC}"
            fi
        fi
    fi
fi

if [ -f "$CV_CLEARVOICE/clearvoice.py" ]; then
    echo "        Installing ClearVoice package..."
    "$PIP" install -e "$CV_DIR" -q 2>/dev/null || true
    echo "        ClearVoice ready."
fi

# ============================================================
# Step 7 — Post-install fixes
# ============================================================
echo ""
echo "[7/10] Post-install fixes..."

# Fix torch GPU compatibility — downgrade to CPU if CUDA fails
if "$PYTHON" -c "import torch; torch.cuda.is_available()" &>/dev/null 2>&1; then
    echo "        torch GPU: OK"
else
    if "$PYTHON" -c "import torch" &>/dev/null 2>&1; then
        cu_ver=$("$PYTHON" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
        echo "        torch GPU not available (CUDA $cu_ver), checking driver..."
        if "$PYTHON" -c "import torch; torch.cuda.init()" 2>/dev/null; then
            echo "        GPU OK — will use CUDA."
        else
            echo -e "        ${YELLOW}GPU driver mismatch, switching to CPU torch...${NC}"
            "$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall -q 2>/dev/null && echo "        Switched to torch CPU." || echo "        Could not switch — manual fix needed."
        fi
    fi
fi

# Pre-copy demucs models from project to cache
MODEL_SRC="$PROJECT_ROOT/checkpoints/torch_hub"
MODEL_DST="$HOME/.cache/torch/hub/checkpoints"
if [ -d "$MODEL_SRC" ] && ls "$MODEL_SRC"/*.th &>/dev/null 2>&1; then
    mkdir -p "$MODEL_DST"
    cp -u "$MODEL_SRC"/*.th "$MODEL_DST/" 2>/dev/null || true
    echo "        Demucs models: pre-loaded to cache."
else
    echo "        Demucs models: will download on first use."
fi

# ============================================================
# Step 8 — Create data directories
# ============================================================
echo ""
echo "[8/10] Creating data directories..."

mkdir -p "$DATA_DIR"/{downloads,subtitles,clips,temp,approved,cleaned,cleaned_unreviewed,denoised_approved,stitched,pipelinevideo,hotwords,情绪,情绪降噪}
mkdir -p "$DATA_DIR"/asr/{audio,subtitles}
mkdir -p "$DATA_DIR"/asr_compare/{subtitles,audio,discarded}
mkdir -p "$DATA_DIR"/asr_compare_output
mkdir -p "$DATA_DIR"/mfa/{raw_wav,wav,txt,aligned,post,filtered,validate}

echo "        Data directories ready."

# ============================================================
# Step 9 — Clean stale settings + path configuration
# ============================================================
echo ""
echo "[9/10] Checking configuration..."

STALE_DETECTED=0

if [ -f "$SETTINGS_FILE" ]; then
    # Check if CLIPS_DIR in settings points to a non-existent path
    if "$PYTHON" -c "
import json, os
try:
    d = json.load(open('$SETTINGS_FILE', encoding='utf-8'))
    paths = d.get('paths', {})
    clip = paths.get('CLIPS_DIR', '')
    if clip and not os.path.isdir(clip):
        exit(0)
    exit(1)
except:
    exit(1)
" 2>/dev/null; then
        echo -e "        ${YELLOW}Stale settings.json detected (paths from another machine).${NC}"
        echo "        Removing old settings — fresh defaults will be used."
        rm "$SETTINGS_FILE"
        STALE_DETECTED=1
    else
        echo "        settings.json OK for this environment."
    fi
else
    echo "        Using default paths (auto-detected from project location)."
fi

if [ "$STALE_DETECTED" -eq 1 ]; then
    echo ""
    echo "  =============================================================="
    echo "  Fresh environment detected!"
    echo "  Default data directories are under: $DATA_DIR"
    echo ""
    echo "  To customize (e.g., different drives), use the Web UI:"
    echo "  Settings > File Paths"
    echo ""
    echo "  Current defaults:"
    echo "    Downloads : $DATA_DIR/downloads"
    echo "    Clips     : $DATA_DIR/clips"
    echo "    Cleaned   : $DATA_DIR/cleaned"
    echo "  =============================================================="
    echo ""
fi

# ============================================================
# Step 10 — Launch server
# ============================================================
echo ""
echo "[10/10] Starting Anime Pipeline Server..."

if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}[ERROR] Python not found in venv.${NC}"
    exit 1
fi

echo ""
echo "============================================"
echo "  Frontend : http://localhost:5800"
echo "  API docs : http://localhost:5800/docs"
echo "  Data root: $DATA_DIR"
echo "============================================"
echo ""

date > "$SETUP_DONE_FILE"

# Free port 5800
if command -v fuser &>/dev/null; then
    PID=$(fuser 5800/tcp 2>/dev/null || true)
    [ -n "$PID" ] && kill -9 "$PID" 2>/dev/null && sleep 1
elif command -v lsof &>/dev/null; then
    PID=$(lsof -ti :5800 2>/dev/null || true)
    [ -n "$PID" ] && kill -9 "$PID" 2>/dev/null && sleep 1
fi

# Fix PyTorch encoding issues
export PYTHONUTF8=1

exec "$PYTHON" "$PROJECT_ROOT/scripts/server.py"
