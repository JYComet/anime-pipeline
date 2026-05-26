"""
Fill in missing Linux wheels in offline_wheels/.
Run after download_wheels.bat on Windows.
"""
import subprocess, sys, os, re
from collections import defaultdict

WHEELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_wheels")
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TRUSTED = ["--trusted-host", "pypi.tuna.tsinghua.edu.cn"]

# Windows-only packages (no Linux equivalent)
SKIP = {"pywin32", "winrt", "pywinpty", "winsdk", "windows-curses"}

def run(cmd):
    subprocess.run(cmd, check=False)

# Group wheels by package name -> list of platform tags
pkg_tags = defaultdict(list)
whl_re = re.compile(r"^(.+?)-(.+?)-(.+?)-(.+?)-(.+?)\.whl$")

for f in os.listdir(WHEELS_DIR):
    m = whl_re.match(f)
    if not m:
        continue
    name, version, _, py_tag, abi_platform = m.groups()
    pkg_tags[name.lower()].append((f, abi_platform))

# Find packages missing Linux or pure-Python wheels
need_linux = []
for name, wheels in sorted(pkg_tags.items()):
    if name in SKIP:
        continue
    platforms = {plat for _, plat in wheels}
    has_any = any("none-any" in p for p in platforms)
    has_linux = any("manylinux" in p for p in platforms)
    if has_any or has_linux:
        continue  # already covered
    # Only Windows wheels exist
    need_linux.append(name)

# Also fix torch: download CPU version (no CUDA deps) for Linux
print("Downloading torch CPU wheels (no CUDA needed)...\n")
subprocess.run([
    sys.executable, "-m", "pip", "download", "torch", "torchaudio",
    "-d", WHEELS_DIR, "--no-deps",
    "--platform", "manylinux_2_17_x86_64",
    "--only-binary=:all:", "--python-version", "310",
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "--trusted-host", "download.pytorch.org"
], check=False)

# And CUDA deps just in case torch CUDA version exists
for cuda_pkg in [
    "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-cupti-cu12",
    "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12", "nvidia-cufft-cu12",
    "nvidia-cusparse-cu12", "nvidia-cusolver-cu12",
    "nvidia-nccl-cu12", "nvidia-nvjitlink-cu12",
]:
    subprocess.run([
        sys.executable, "-m", "pip", "download", cuda_pkg,
        "-d", WHEELS_DIR, "--no-deps",
        "--platform", "manylinux_2_17_x86_64",
        "--only-binary=:all:", "--python-version", "310",
        "-i", MIRROR, "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
    ], check=False)

# Tokenizers (needed by transformers for ASR)
subprocess.run([
    sys.executable, "-m", "pip", "download", "tokenizers",
    "-d", WHEELS_DIR, "--no-deps",
    "--platform", "manylinux_2_17_x86_64",
    "--only-binary=:all:", "--python-version", "310",
    "-i", MIRROR, "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
], check=False)

if not need_linux:
    print("\nAll packages have Linux or pure-Python wheels. Done.")
    sys.exit(0)

print(f"\n{len(need_linux)} packages need Linux wheels:\n")
for n in need_linux:
    print(f"  {n}")

print(f"\nDownloading from {MIRROR}...\n")
for name in need_linux:
    print(f"  [{name}]")
    # Try binary Linux wheel first
    rc = subprocess.run([
        sys.executable, "-m", "pip", "download", name,
        "-d", WHEELS_DIR, "--no-deps",
        "--platform", "manylinux_2_17_x86_64",
        "--platform", "manylinux2014_x86_64",
        "--only-binary=:all:", "--python-version", "310",
        "-i", MIRROR, "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
    ], check=False)
    if rc.returncode != 0:
        # Try pure Python
        subprocess.run([
            sys.executable, "-m", "pip", "download", name,
            "-d", WHEELS_DIR, "--no-deps", "--no-binary=:all:",
            "-i", MIRROR, "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
        ], check=False)

print("\nDone. Run setup.sh on the server now.")
