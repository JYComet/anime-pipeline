#!/usr/bin/env bash
# =============================================================================
# ASR Compare Tool — Linux / macOS One-Click Launcher
# =============================================================================
# Usage:
#   bash run.sh                                              # full auto deploy + run
#   bash run.sh --no-delete                                  # pass args to ASR tool
#   bash run.sh --match-threshold 85 --language ja
#
# Or deploy separately:
#   python deploy.py                    # check + install
#   python deploy.py --force-cpu        # CPU-only mode
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  ASR Compare Tool"
echo "  FireRed ASR2 <> Qwen3-ASR (API)"
echo "========================================"
echo ""
echo "[1/2] Checking environment and installing dependencies..."
echo "========================================"

python deploy.py || {
    echo ""
    echo "========================================"
    echo "Deploy failed! Check errors above."
    echo "You can also run: python deploy.py --force-cpu"
    echo "========================================"
    exit 1
}

echo ""
echo "========================================"
echo "[2/2] Starting ASR comparison..."
echo "========================================"
echo ""
echo "Config: config.yaml"
echo "Log:    asr_compare.log"
echo "Press Ctrl+C to cancel during processing."
echo "========================================"
echo ""

python asr_compare.py "$@"

echo ""
echo "========================================"
echo "Task completed."
echo "========================================"
