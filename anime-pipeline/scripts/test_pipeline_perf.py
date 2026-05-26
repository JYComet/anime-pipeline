"""
Quick pipeline timing test — processes only 2 segments to avoid timeout.
Extrapolates full timing from per-segment measurements.
"""
import os, sys, time, json, subprocess, shutil, threading, concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FFMPEG, FFPROBE, TEMP_DIR

INPUT_FILE = r"E:\ComicCut\anime-pipeline\data\downloads\【VR研学】小鼠粽子喜欢甜的还是咸的 20250531 2222.aac"
SEGMENT_DUR = 600
TEST_DIR = os.path.join(TEMP_DIR, "perf_quick")
BASE_NAME = os.path.splitext(os.path.basename(INPUT_FILE))[0]
MAX_TEST_SEGMENTS = 2

print("=" * 60)
print("  Pipeline Quick Timing Test (2-segment sample)")
print("=" * 60)

# Clean and prepare
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR, exist_ok=True)

# Get audio info
r = subprocess.run([FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", INPUT_FILE],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
info = json.loads(r.stdout)
total_dur = float(info.get("format", {}).get("duration", 0))
total_segments = int(total_dur / SEGMENT_DUR) + 1
effective_segments = total_segments - 2  # after removing first+last
print(f"File: {total_dur/60:.0f} min → {total_segments} segments → {effective_segments} effective")
print(f"Testing first {MAX_TEST_SEGMENTS} segments only\n")

t_total_start = time.time()

# ── Phase A: Split ──
print("Phase A: Split source AAC ...")
t0 = time.time()
seg_dir = os.path.join(TEST_DIR, f"segments_{BASE_NAME}")
os.makedirs(seg_dir, exist_ok=True)
ext = os.path.splitext(INPUT_FILE)[1].lower()
seg_pattern = os.path.join(seg_dir, f"{BASE_NAME}_%03d{ext}")

subprocess.run([FFMPEG, "-y", "-i", INPUT_FILE, "-f", "segment",
                "-segment_time", str(SEGMENT_DUR), "-c", "copy", seg_pattern],
               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)

all_segments = sorted([os.path.join(seg_dir, x) for x in os.listdir(seg_dir)
                       if x.endswith(ext) and os.path.getsize(os.path.join(seg_dir, x)) > 0])
if len(all_segments) >= 3:
    os.remove(all_segments.pop(0))  # first
    os.remove(all_segments.pop(-1))  # last
test_segments = all_segments[:MAX_TEST_SEGMENTS]
t_split = time.time() - t0
print(f"  {len(all_segments)} segments kept, testing {len(test_segments)} → {t_split:.1f}s\n")

# ── Phase B: Convert segments to WAV (16kHz mono) ──
print("Phase B: Convert segments → 16kHz mono WAV ...")
t0 = time.time()

def _convert(idx, seg_path):
    out = os.path.join(seg_dir, f"{BASE_NAME}_seg{idx:03d}.wav")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    subprocess.run([FFMPEG, "-y", "-i", seg_path, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", out],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        os.remove(seg_path)
        return out
    return ""

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_TEST_SEGMENTS) as ex:
    wavs = list(ex.map(lambda x: _convert(x[0], x[1]), enumerate(test_segments)))
wavs = [w for w in wavs if w]
t_convert = time.time() - t0
print(f"  {len(wavs)} WAVs in {t_convert:.1f}s (avg {t_convert/len(wavs):.1f}s each)\n")

# ── Phase C: ASR on 1 segment (single, for timing) ──
print("Phase C: ASR timing (single segment, 1 thread) ...")
from asr_pipeline import run_asr_on_audio, _get_asr_model

t0 = time.time()
result = run_asr_on_audio(
    wavs[0], output_dir=seg_dir,
    model_key="qwen3-asr", language="zh",
    device="cuda"
)
t_asr_single = time.time() - t0
seg_count = result.get("segments_count", 0)
print(f"  1 segment: {seg_count} subtitles in {t_asr_single:.1f}s")
print(f"  Real-time factor: {600/t_asr_single:.0f}x (10min processed in {t_asr_single:.0f}s)\n")

# ── Phase C2: ASR on 2 segments with semaphore=1 (for accuracy) ──
print("Phase C2: ASR on 2 segments, semaphore=1 ...")
asr_sem = threading.BoundedSemaphore(1)
asr_times = []

def _gated_asr(idx, wav_path):
    global asr_times
    t1 = time.time()
    with asr_sem:
        r = run_asr_on_audio(wav_path, output_dir=seg_dir, model_key="qwen3-asr",
                             language="zh", device="cuda")
    t2 = time.time()
    asr_times.append(t2 - t1)
    return r.get("segments_count", 0)

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(_gated_asr, i, w) for i, w in enumerate(wavs)]
    concurrent.futures.wait(futs)
t_asr_two = time.time() - t0
total_subs = sum(f.result() for f in futs)
print(f"  Times: {[f'{t:.1f}s' for t in asr_times]}")
print(f"  Wall time: {t_asr_two:.1f}s (sem=1, 2 segments)")
print(f"  Total subtitles: {total_subs}\n")

# ── Extrapolation ──
t_total = time.time() - t_total_start
# Phase A+B is constant for any number of segments
t_phase_ab = t_split + t_convert
# Phase C extrapolation: sem=1, effective_segments
asr_per_seg = t_asr_single
t_phase_c_full = effective_segments * asr_per_seg

print("=" * 60)
print("  EXTRAPOLATED FULL PIPELINE")
print("=" * 60)
print(f"  Phases A+B (split+convert):     {t_phase_ab:5.1f}s")
print(f"  Phase C ({effective_segments} segments × {asr_per_seg:.0f}s ASR):  {t_phase_c_full:5.0f}s")
print(f"  ─" + "─" * 35)
print(f"  Total estimated:                {t_phase_ab + t_phase_c_full:5.0f}s  ({ (t_phase_ab + t_phase_c_full)/60:.1f} min)")
print(f"  Processing ratio:               {total_dur / (t_phase_ab + t_phase_c_full):.0f}x real-time")
print(f"  vs original 20min:              {20*60 / (t_phase_ab + t_phase_c_full):.1f}x faster")
print("=" * 60)

# Cleanup
shutil.rmtree(TEST_DIR, ignore_errors=True)
