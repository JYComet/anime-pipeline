#!/usr/bin/env python3
"""
ASR Compare Tool — One-Click Deploy
====================================
Auto-detects OS, checks environment, installs dependencies.

Usage:
    python deploy.py              # full deploy: check + install
    python deploy.py --check      # check only, no install
    python deploy.py --install    # install only, skip checks
    python deploy.py --check       # check environment only
"""
import os
import sys
import platform
import subprocess
import shutil
import re
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
MODELS_DIR = PROJECT_ROOT / "models"


# ── OS Detection ─────────────────────────────────────────────────────────────
def detect_os():
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "macos"
    return system.lower()


# ── Print Helpers ─────────────────────────────────────────────────────────────
HEADER = "=" * 60

def print_header(title: str):
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


def print_ok(msg: str):
    print(f"  [OK]    {msg}")


def print_warn(msg: str):
    print(f"  [WARN]  {msg}")


def print_err(msg: str):
    print(f"  [ERROR] {msg}")


def print_info(msg: str):
    print(f"  [INFO]  {msg}")


# ── Python Check ─────────────────────────────────────────────────────────────
def check_python():
    print_header("Python Environment")
    ver = sys.version_info
    py_path = sys.executable
    print_info(f"Python {ver.major}.{ver.minor}.{ver.micro} — {py_path}")
    if ver < (3, 10):
        print_err("Python 3.10+ is required. Please upgrade.")
        return False
    print_ok(f"Python {ver.major}.{ver.minor} meets minimum (3.10+)")
    return True


# ── Pip Check ─────────────────────────────────────────────────────────────────
def check_pip():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            print_ok(f"pip available: {result.stdout.strip()[:60]}")
            return True
        print_err("pip not working")
        return False
    except Exception as e:
        print_err(f"pip check failed: {e}")
        return False


# ── CUDA Check ────────────────────────────────────────────────────────────────
def check_cuda():
    print_header("CUDA / GPU")
    try:
        import subprocess as sp
        r = sp.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                      "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            info = r.stdout.strip().split("\n")[0]
            print_ok(f"GPU detected: {info}")
            # Try to get CUDA version
            try:
                nvcc = sp.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
                m = re.search(r"release (\d+\.\d+)", nvcc.stdout)
                if m:
                    cuda_ver = m.group(1)
                    print_info(f"CUDA Toolkit: {cuda_ver}")
                    major = int(cuda_ver.split(".")[0])
                    if major >= 12:
                        return "cu121"
                    elif major >= 11:
                        return "cu118"
            except Exception:
                pass
            # Default: use CUDA 11.8 wheels (most compatible)
            print_info("CUDA version not detected from nvcc, defaulting to cu118")
            return "cu118"
        else:
            print_err("No NVIDIA GPU detected. FireRed ASR2 requires a GPU.")
            print_err("This tool cannot run on CPU. Please install a NVIDIA GPU and drivers.")
            return None
    except FileNotFoundError:
        print_err("nvidia-smi not found — no NVIDIA GPU or driver installed.")
        print_err("This tool cannot run on CPU. Please install NVIDIA drivers.")
        return None
    except Exception as e:
        print_err(f"GPU check failed: {e}")
        return None
        return None


# ── FFmpeg Check ──────────────────────────────────────────────────────────────
def check_ffmpeg():
    print_header("External Tools")
    all_ok = True
    for tool in ["ffmpeg", "ffprobe"]:
        path = shutil.which(tool)
        if path:
            try:
                r = subprocess.run([tool, "-version"], capture_output=True, text=True, timeout=10)
                ver_line = r.stdout.split("\n")[0] if r.stdout else "?"
                print_ok(f"{tool}: {ver_line[:70]}")
            except Exception:
                print_ok(f"{tool}: found at {path}")
        else:
            print_err(f"{tool}: NOT FOUND in PATH. Audio format conversion will fail.")
            all_ok = False
    return all_ok


# ── Model Files Check ─────────────────────────────────────────────────────────
def check_models():
    print_header("Model Files")
    all_ok = True

    # Silero VAD
    silero_model = MODELS_DIR / "silero_vad" / "silero_vad.onnx"
    if silero_model.exists():
        size_mb = silero_model.stat().st_size / (1024 * 1024)
        print_ok(f"Silero VAD: {silero_model} ({size_mb:.1f} MB)")
    else:
        print_err(f"Silero VAD: NOT FOUND at {silero_model}")
        all_ok = False

    # FireRed ASR2 source
    firered_src = PROJECT_ROOT / "FireRedASR2S" / "fireredasr2s"
    if firered_src.is_dir():
        print_ok(f"FireRed source: {firered_src}")
    else:
        print_err(f"FireRed source: NOT FOUND at {firered_src}")
        all_ok = False

    # FireRed models
    firered_models = MODELS_DIR / "firered_asr2"
    expected_dirs = ["FireRedVAD", "FireRedLID", "FireRedASR2-AED", "FireRedPunc"]
    if firered_models.is_dir():
        for d in expected_dirs:
            sub = firered_models / d
            if sub.is_dir():
                total = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
                size_mb = total / (1024 * 1024)
                print_ok(f"FireRed {d}: {size_mb:.0f} MB")
            else:
                print_warn(f"FireRed {d}: not found (FireRed ASR will fail)")
                all_ok = False
    else:
        print_err(f"FireRed models dir NOT FOUND: {firered_models}")
        all_ok = False

    return all_ok


# ── Mirrors ────────────────────────────────────────────────────────────────────
_PIP_MIRRORS = [
    # Default PyPI
    ("default", []),
    # Tsinghua (mainland China)
    ("Tsinghua", ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]),
    # Aliyun (mainland China)
    ("Aliyun",  ["-i", "https://mirrors.aliyun.com/pypi/simple/"]),
    # USTC (mainland China)
    ("USTC",    ["-i", "https://pypi.mirrors.ustc.edu.cn/simple/"]),
]

# ── Dependency Install ────────────────────────────────────────────────────────
def install_deps(cuda_tag=None, force_cpu=False):
    print_header("Installing Dependencies")

    if not REQUIREMENTS.exists():
        print_err(f"requirements.txt not found at {REQUIREMENTS}")
        return False

    if force_cpu:
        print_err("CPU mode is not supported. This tool requires a CUDA GPU.")
        return False

    # Build extra-index-url for PyTorch CUDA wheels
    if cuda_tag == "cu121":
        print_info("Using CUDA 12.1 PyTorch wheels")
        pytorch_extra = ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
    elif cuda_tag == "cu118":
        print_info("Using CUDA 11.8 PyTorch wheels")
        pytorch_extra = ["--extra-index-url", "https://download.pytorch.org/whl/cu118"]
    elif cuda_tag is None:
        print_err("No CUDA GPU detected. Cannot proceed with installation.")
        print_err("This tool requires a NVIDIA GPU with CUDA support.")
        return False
    else:
        pytorch_extra = ["--extra-index-url", f"https://download.pytorch.org/whl/{cuda_tag}"]

    # Try mirrors in order: default PyPI first, then Chinese mirrors as fallback
    for mirror_name, mirror_args in _PIP_MIRRORS:
        if mirror_name == "default":
            print_info("Trying default PyPI...")
        else:
            print_info(f"Trying {mirror_name} mirror...")

        cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        cmd += mirror_args
        cmd += pytorch_extra

        print_info(f"Running: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                print()
                print_ok(f"All dependencies installed successfully (via {mirror_name})")
                return True
            else:
                print_warn(f"pip install failed via {mirror_name} (exit code {result.returncode})")
        except Exception as e:
            print_warn(f"pip install error via {mirror_name}: {e}")

    print()
    print_err("All mirrors exhausted. Manual install required.")
    print_info("Try: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
    return False


# ── Post-Install Verification ─────────────────────────────────────────────────
def verify_imports():
    print_header("Verifying Installed Packages")

    # Each entry: (import_name, display_name, is_required)
    tests = [
        # Our tool core
        ("numpy",       "numpy",                 True),
        ("soundfile",   "soundfile",             True),
        ("yaml",        "PyYAML",                True),
        ("tqdm",        "tqdm",                  True),
        # Deep learning
        ("torch",       "torch",                 True),
        ("torchaudio",  "torchaudio",            True),
        # Qwen3 API
        ("dashscope",   "dashscope",             True),
        # Silero VAD
        ("silero_vad",  "silero-vad",            True),
        ("onnxruntime", "onnxruntime",           True),
        # FireRed ASR2
        ("transformers",           "transformers",          True),
        ("cn2an",                  "cn2an",                 False),  # optional: WER calc
        ("kaldiio",                "kaldiio",               True),
        ("kaldi_native_fbank",     "kaldi_native_fbank",    True),
        ("sentencepiece",          "sentencepiece",         True),
        ("textgrid",               "textgrid",              False),  # optional: CLI only
        ("peft",                   "peft",                  False),  # optional: LLM LoRA
    ]

    all_ok = True
    for mod, display, required in tests:
        try:
            __import__(mod)
            print_ok(display)
        except ImportError as e:
            if required:
                print_err(f"{display}: NOT INSTALLED")
                all_ok = False
            else:
                print_warn(f"{display}: not installed (optional)")

    # Show torch CUDA status
    try:
        import torch
        if torch.cuda.is_available():
            print_ok(f"torch CUDA: available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print_warn("torch CUDA: NOT available (CPU-only mode)")
    except Exception:
        pass

    return all_ok


# ── Config Check ──────────────────────────────────────────────────────────────
def check_config():
    print_header("Configuration")
    try:
        import yaml
    except ImportError:
        print_err("PyYAML not installed — cannot check config")
        return False

    config_path = PROJECT_ROOT / "config.yaml"
    local_path = PROJECT_ROOT / "config.local.yaml"

    config_file = local_path if local_path.exists() else config_path
    if not config_file.exists():
        print_err(f"Config file not found: {config_path}")
        return False

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_key = cfg.get("api", {}).get("dashscope_api_key", "")
    if api_key:
        print_ok(f"API Key configured ({'*' * 20})")
    else:
        print_warn("API Key NOT set — Qwen3 ASR API calls will fail")
        print_info("Edit config.yaml and set dashscope_api_key")

    paths = cfg.get("paths", {})
    for k in ["audio_input_dir", "output_dir"]:
        v = paths.get(k, "")
        print_info(f"  {k}: {v}")

    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os_name = detect_os()
    print_header(f"ASR Compare Tool — Deploy ({os_name})")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Platform:     {platform.platform()}")

    args = sys.argv[1:]
    do_check = "--check" in args or "--install" not in args
    do_install = "--install" in args or "--check" not in args

    # ── Checks ──
    if do_check:
        check_python()
        check_pip()
        cuda_tag = check_cuda()
        if cuda_tag is None:
            print()
            print_err("GPU check failed. This tool requires a CUDA-capable NVIDIA GPU.")
            print_err("Cannot continue without GPU support.")
            return
        check_ffmpeg()
        check_models()
        check_config()
    else:
        cuda_tag = check_cuda()

    # ── Install ──
    if do_install:
        if cuda_tag is None:
            print_err("Installation aborted: no CUDA GPU available.")
            return
        ok = install_deps(cuda_tag)
        if ok:
            verify_imports()

    # ── Summary ──
    print_header("Deploy Complete")
    print(f"  Run the tool: python asr_compare.py")
    print(f"  Or double-click: run.bat (Windows) / bash run.sh (Linux/Mac)")
    print(HEADER)


if __name__ == "__main__":
    main()
