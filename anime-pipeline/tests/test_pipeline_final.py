"""Pipeline test — quick 1-segment timing + full extrapolation. Uses CPU to avoid GPU OOM."""
import os, sys, time, json, subprocess, shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from config import FFMPEG, FFPROBE, TEMP_DIR
# Suppress noise
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INPUT = r"E:\ComicCut\anime-pipeline\data\downloads\【VR研学】小鼠粽子喜欢甜的还是咸的 20250531 2222.aac"
SEG_DUR = 600
TEST_DIR = os.path.join(TEMP_DIR, "pfinal")
BASE = os.path.splitext(os.path.basename(INPUT))[0]
DEVICE = "cuda"

t_all = time.time()
def ts(msg):
    print(f"[{time.time()-t_all:5.0f}s] {msg}", flush=True)

os.makedirs(TEST_DIR, exist_ok=True)
print("=" * 60)
print(f"  Pipeline test: {BASE[:35]}...")
print(f"  Device: {DEVICE}")
print("=" * 60)

# ── Phase A ──
ts("Phase A: Split source AAC (ffmpeg -c copy)")
seg_dir = os.path.join(TEST_DIR, "segs")
os.makedirs(seg_dir, exist_ok=True)
ext = os.path.splitext(INPUT)[1].lower()
subprocess.run([FFMPEG, "-y", "-i", INPUT, "-f", "segment", "-segment_time", str(SEG_DUR),
                "-c", "copy", os.path.join(seg_dir, f"s_%03d{ext}")],
               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)

all_s = sorted([os.path.join(seg_dir, x) for x in os.listdir(seg_dir)
                if x.endswith(ext) and os.path.getsize(os.path.join(seg_dir, x)) > 0])
n_segs = len(all_s)
if n_segs >= 3:
    os.remove(all_s.pop(0))
    os.remove(all_s.pop(-1))
    n_segs = len(all_s)
ts(f"Phase A done: {n_segs} segments")

# ── Phase B ──
ts(f"Phase B: Convert {n_segs} segments -> 16kHz mono WAV")
wavs = []
for i, s in enumerate(all_s):
    out = os.path.join(seg_dir, f"w_{i:03d}.wav")
    subprocess.run([FFMPEG, "-y", "-i", s, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", out],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        os.remove(s)
        wavs.append(out)
wav_size = sum(os.path.getsize(w)/(1024*1024) for w in wavs)
ts(f"Phase B done: {len(wavs)} WAVs ({wav_size:.0f} MB)")

# ── Phase C: Process just 2 segments for timing ──
ts("Phase C: ASR on 2 segments for timing...")
from asr_pipeline import run_asr_on_audio

t_asr_samples = []
for i in range(min(2, len(wavs))):
    t0 = time.time()
    r = run_asr_on_audio(wavs[i], output_dir=seg_dir, model_key="qwen3-asr",
                         language="zh", device=DEVICE)
    t1 = time.time()
    t_asr_samples.append(t1 - t0)
    ts(f"  seg {i}: {r.get('segments_count',0)} subs in {t1-t0:.1f}s"
       f" ({(600/(t1-t0)):.0f}x real-time)")

avg_asr = sum(t_asr_samples) / len(t_asr_samples)

# ── Extrapolation ──
t_elapsed = time.time() - t_all
t_ab = t_elapsed - sum(t_asr_samples)  # approximate A+B time
t_c_est = (len(wavs) - 2) * avg_asr  # remaining segments
t_est = t_elapsed + t_c_est

ts("")
ts("=" * 50)
ts(f"RESULTS ({DEVICE} mode, {len(wavs)} segments=")
ts(f"  A) Split source AAC:   quick (ffmpeg -c copy)")
ts(f"  B) Convert to WAV:     quick ({wav_size:.0f} MB, 16kHz mono)")
ts(f"  C) ASR per segment:    {avg_asr:.0f}s avg ({(600/avg_asr):.0f}x real-time)")
ts(f"  ─" + "─" * 38)
ts(f"  Estimated total:       {t_est:.0f}s = {t_est/60:.1f} min")
ts(f"  vs original 20min:     {20*60/t_est:.1f}x faster")
ts(f"  vs 169min audio:       {169*60/t_est:.1f}x real-time")
ts("=" * 50)

shutil.rmtree(TEST_DIR, ignore_errors=True)
