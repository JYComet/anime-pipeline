#!/bin/bash
# ============================================================
#  Anime Pipeline — One-Click Setup & Launch (Linux/macOS)
#  Supports both online and offline install modes.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
COMICUT_ROOT="$(dirname "$PROJECT_ROOT")"

VENV_DIR="$PROJECT_ROOT/venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
SETUP_DONE_FILE="$PROJECT_ROOT/.setup_done"
WHEELS_DIR="$PROJECT_ROOT/offline_wheels"

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
echo "[1/8] Checking Python installation..."

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
echo "[2/8] Setting up virtual environment..."

if [ -f "$PYTHON" ] && "$PYTHON" -m pip --version &>/dev/null 2>&1; then
    echo "        venv already exists and working."
else
    if [ -f "$PYTHON" ]; then
        echo "        Broken venv detected, recreating..."
        rm -rf "$VENV_DIR"
    fi
    echo "        Creating venv at $VENV_DIR ..."

    if "$PYTHON3" -m venv --copies "$VENV_DIR" 2>/dev/null; then
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

        # Method 1: ensurepip (needs python3-venv system package)
        "$PYTHON" -m ensurepip --upgrade 2>/dev/null && PIP_INSTALLED=1

        # Method 2: pip wheel from offline_wheels (uses Python zipfile, no unzip needed)
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

        # Method 3: download get-pip.py (online only, skipped in offline mode)
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
        echo "        This server appears to have no internet access."
        echo "        To fix this, run on your Windows machine first:"
        echo ""
        echo -e "          ${GREEN}download_wheels.bat${NC}"
        echo ""
        echo "        This downloads all needed packages to offline_wheels/"
        echo "        (the folder is shared via the network mount)."
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
echo "[3/8] Installing Python dependencies..."

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

if [ "$OFFLINE" -eq 1 ]; then
    echo "        Installing from offline wheels..."
    "$PIP" install --no-index --find-links="$WHEELS_DIR" -r "$PROJECT_ROOT/requirements.txt" || {
        echo -e "${YELLOW}[WARNING] Some packages failed offline.${NC}"
        echo "          Try running download_wheels.bat on Windows to refresh."
    }
else
    echo "        Installing from PyPI..."
    "$PIP" install -r "$PROJECT_ROOT/requirements.txt" || {
        echo -e "${YELLOW}[WARNING] Some packages failed to install.${NC}"
    }
fi
echo "        Core dependencies installed."

# ============================================================
# Step 4 — Optional ASR dependencies
# ============================================================
echo ""
echo "[4/8] Optional ASR dependencies..."

ASR_REQ="$PROJECT_ROOT/requirements-asr.txt"
if [ -f "$ASR_REQ" ]; then
    echo "        ASR (speech recognition) packages are optional but large (~4GB)."
    echo "        Required for: auto-subtitle generation, ASR model comparison."
    echo ""
    read -r -p "        Install ASR dependencies? [y/N]: " ASR_INSTALL
    if [ "$ASR_INSTALL" = "y" ] || [ "$ASR_INSTALL" = "Y" ]; then
        if [ "$OFFLINE" -eq 1 ]; then
            "$PIP" install --no-index --find-links="$WHEELS_DIR" -r "$ASR_REQ" 2>/dev/null || {
                echo -e "${YELLOW}[WARNING] Some ASR wheels missing offline.${NC}"
                echo "          Re-run download_wheels.bat on Windows to include them."
            }
        else
            "$PIP" install -r "$ASR_REQ" || {
                echo -e "${YELLOW}[WARNING] Some ASR packages failed to install.${NC}"
            }
        fi
    else
        echo "        Skipped."
    fi
fi

# ============================================================
# Step 5 — System tools
# ============================================================
echo ""
echo "[5/8] Checking system tools..."

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
echo "[6/8] Checking ClearerVoice-Studio..."

CV_DIR="$COMICUT_ROOT/ClearerVoice-Studio-main/ClearerVoice-Studio-main"
CV_CLEARVOICE="$CV_DIR/clearvoice"

if [ -f "$CV_CLEARVOICE/clearvoice.py" ]; then
    echo "        Found at: $CV_DIR"
else
    echo "        ClearerVoice-Studio not found."
    if [ "$OFFLINE" -eq 1 ]; then
        echo -e "        ${YELLOW}Copy the folder from your Windows machine to:${NC}"
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
# Step 7 — Clean up stale settings.json
# ============================================================
echo ""
echo "[7/8] Checking configuration..."

SETTINGS_FILE="$PROJECT_ROOT/data/settings.json"
if [ -f "$SETTINGS_FILE" ]; then
    if grep -q '[A-Z]:\\\\' "$SETTINGS_FILE" 2>/dev/null; then
        echo "        Removing old settings.json with Windows absolute paths..."
        rm "$SETTINGS_FILE"
    else
        echo "        settings.json OK."
    fi
else
    echo "        Using default paths."
fi

# ============================================================
# Step 8 — Launch server
# ============================================================
echo ""
echo "[8/8] Starting Anime Pipeline Server..."

if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}[ERROR] Python not found in venv.${NC}"
    exit 1
fi

echo ""
echo "============================================"
echo "  Frontend : http://192.168.103.163:5800"
echo "  API docs : http://192.168.103.163:5800/docs"
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

exec "$PYTHON" "$PROJECT_ROOT/scripts/server.py"
