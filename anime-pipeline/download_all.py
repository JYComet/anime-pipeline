"""
One-shot: download ALL wheels consistently (no version mismatches).
1. Binary wheels first (current platform, consistent resolution)
2. Source dists for packages with no binary wheel
3. Linux binary wheels for compiled packages
4. ASR deps
"""
import subprocess, sys, os, re
from pathlib import Path

ROOT = Path(__file__).parent
WHEELS = ROOT / "offline_wheels"
REQ = ROOT / "requirements.txt"
ASR_REQ = ROOT / "requirements-asr.txt"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
MIRROR_ARGS = ["-i", MIRROR, "--trusted-host", "pypi.tuna.tsinghua.edu.cn"]

WHEELS.mkdir(exist_ok=True)
PIP = [sys.executable, "-m", "pip"]
DOWNLOAD = PIP + ["download", "-d", str(WHEELS)] + MIRROR_ARGS

def run(cmd):
    print(f"  $ {' '.join(cmd[:5])}...")
    subprocess.run(cmd, check=False)

# ============================================================
# Step 1: Download everything — consistent version resolution
# pip download doesn't build source dists, so it won't hang
# ============================================================
print("[1/5] Downloading all dependencies (consistent resolution)...")
run(DOWNLOAD + ["-r", str(REQ)])
run(DOWNLOAD + ["-r", str(ASR_REQ)])

# ============================================================
# Step 2: Source distributions for packages still missing
# ============================================================
print("\n[2/5] Source distributions for missing packages...")
# Get packages from requirements that have no wheel in offline_wheels
whl_re = re.compile(r"^(.+?)-(.+?)-(.+?)-(.+?)-(.+?)\.whl$")
have_wheel = set()
for f in os.listdir(WHEELS):
    m = whl_re.match(f)
    if m:
        have_wheel.add(m.group(1).lower().replace("_", "-"))

for req_file in [REQ, ASR_REQ]:
    for line in open(req_file, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(">=")[0].split("==")[0].split("[")[0].strip().lower().replace("_", "-")
        if name not in have_wheel:
            print(f"  downloading source for: {name}")
            run(DOWNLOAD + ["--no-binary=:all:", name])

# ============================================================
# Step 3: Linux binary wheels — batched for speed
# ============================================================
print("\n[3/5] Linux binary wheels...")
LINUX_BASE = [
    "--no-deps", "--only-binary=:all:",
    "--platform", "manylinux_2_17_x86_64",
    "--platform", "manylinux2014_x86_64",
    "--python-version", "310",
]

# Find packages that need Linux wheels (compiled, no Linux wheel yet)
already_linux = set()
need_linux = set()
for f in os.listdir(WHEELS):
    m = whl_re.match(f)
    if not m:
        continue
    name = m.group(1).lower().replace("_", "-")
    plat = m.group(5)
    if "manylinux" in plat:
        already_linux.add(name)
    elif "none-any" not in plat:
        need_linux.add(name)

need_linux -= already_linux  # skip if Linux wheel already exists

if need_linux:
    print(f"  {len(need_linux)} packages need Linux wheels, batching...")
    # Batch into groups of 50 to avoid command line being too long
    pkgs = sorted(need_linux)
    batch_size = 50
    for i in range(0, len(pkgs), batch_size):
        batch = pkgs[i:i+batch_size]
        print(f"  batch {i//batch_size + 1}: {len(batch)} packages")
        run(DOWNLOAD + LINUX_BASE + batch)
else:
    print("  All Linux wheels already present.")

# ============================================================
# Step 4: Torch CPU (no CUDA needed)
# ============================================================
print("\n[4/5] Torch CPU wheels...")
TORCH_DOWNLOAD = PIP + ["download", "-d", str(WHEELS),
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "--trusted-host", "download.pytorch.org"]
for t in ["torch", "torchaudio"]:
    run(TORCH_DOWNLOAD + LINUX + [t])

# ============================================================
# Step 5: Special dependencies (tokenizers etc.)
# ============================================================
print("\n[5/5] Special dependencies...")
for extra in ["tokenizers", "safetensors", "onnxruntime"]:
    run(DOWNLOAD + LINUX + [extra])

# ============================================================
# Done
# ============================================================
whl_count = len(list(WHEELS.glob("*.whl")))
src_count = len(list(WHEELS.glob("*.tar.gz")))
print(f"\nDone. {whl_count} wheels + {src_count} source dists.")
print("Now run on server: rm -f .setup_done && rm -rf venv && ./setup.sh")
