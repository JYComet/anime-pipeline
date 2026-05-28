"""
Pipeline test with per-segment progress reporting.
Runs split+convert first (fast), then ASR segment-by-segment.
"""
import os, sys, time, json, subprocess, shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from config import FFMPEG, FFPROBE, TEMP_DIR

INPUT_FILE = r"E:\ComicCut\anime-pipeline\data\downloads\【VR研学】小鼠粽子喜欢甜的还是咸的 20250531 2222.aac"
SEGMENT_DUR = 600
TEST_DIR = os.path.join(TEMP_DIR, "perf_async")
BASE_NAME = os.path.splitext(os.path.basename(INPUT_FILE))[0]

def log(msg):
    t = time.time() - t_start
    print(f"[{t:5.0f}s] {msg}", flush=True)

os.makedirs(TEST_DIR, exist_ok=True)
t_start = time.time()

print("=" * 50)
print(f"Pipeline test: {BASE_NAME[:40]}...")
print("=" * 50, flush=True)

# Phase A: Split
log("Phase A: Splitting source AAC...")
seg_dir = os.path.join(TEST_DIR, "segments")
os.makedirs(seg_dir, exist_ok=True)
ext = os.path.splitext(INPUT_FILE)[1].lower()

subprocess.run([FFMPEG, "-y", "-i", INPUT_FILE, "-f", "segment",
                "-segment_time", str(SEGMENT_DUR), "-c", "copy",
                os.path.join(seg_dir, f"seg_%03d{ext}")],
               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)

all_segs = sorted([os.path.join(seg_dir, x) for x in os.listdir(seg_dir)
                   if x.endswith(ext) and os.path.getsize(os.path.join(seg_dir, x)) > 0])
if len(all_segs) >= 3:
    os.remove(all_segs.pop(0))
    os.remove(all_segs.pop(-1))
log(f"Phase A done: {len(all_segs)} segments ({all_segs[0]!r})")

# Phase B: Convert all to WAV
log("Phase B: Converting to 16kHz mono WAV...")
wavs = []
for i, seg in enumerate(all_segs):
    out = os.path.join(seg_dir, f"wav_{i:03d}.wav")
    subprocess.run([FFMPEG, "-y", "-i", seg, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", out],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        os.remove(seg)
        wavs.append(out)
log(f"Phase B done: {len(wavs)} WAVs ({sum(os.path.getsize(w)/1e6 for w in wavs):.0f} MB total)")

# Pre-load ASR model once
log("Phase C: Loading ASR model...")
from asr_pipeline import _get_asr_model, _get_vad_model, run_vad
import soundfile as sf
import numpy as np
import torch

asr_model = _get_asr_model("qwen3-asr", device="cuda")
vad_model = _get_vad_model(device="cuda")
log(f"Models loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ASR on each segment, reporting progress
log(f"Phase D: ASR on {len(wavs)} segments...")
asr_times = []
total_subs = 0

for i, wav in enumerate(wavs):
    t1 = time.time()

    # VAD
    vad_segs = run_vad(wav, device="cuda")
    # Transcribe each VAD segment
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    from asr_pipeline import _QWEN3_LANG_MAP
    seg_texts = 0
    for j, (start_ms, end_ms) in enumerate(vad_segs):
        start_samp = max(0, int(start_ms * sr / 1000))
        end_samp = min(len(audio), int(end_ms * sr / 1000))
        if end_samp <= start_samp:
            continue
        chunk = audio[start_samp:end_samp]
        try:
            results = asr_model.transcribe([(chunk, sr)], language="Chinese")
            for tr in results:
                if (tr.text or "").strip():
                    seg_texts += 1
        except Exception:
            pass

    t2 = time.time()
    asr_times.append(t2 - t1)
    total_subs += seg_texts
    log(f"  seg {i+1}/{len(wavs)}: {len(vad_segs)} VAD → {seg_texts} texts in {t2-t1:.1f}s "
        f"({600/(t2-t1) if (t2-t1) > 0 else 0:.0f}x real-time)")

# Summary
t_total = time.time() - t_start
asr_total = sum(asr_times)
log("=" * 50)
log(f"RESULTS:")
log(f"  Phases A+B (split+convert): {t_total - asr_total:.0f}s")
log(f"  Phase C (model loading):    ??s")
log(f"  Phase D ({len(wavs)} segments ASR): {asr_total:.0f}s ({asr_total/len(wavs):.0f}s/seg)")
log(f"  Total subtitles: {total_subs}")
log(f"  Total wall time:  {t_total:.0f}s ({t_total/60:.1f} min)")
log(f"  Real-time factor: {10160/t_total:.0f}x")
log("=" * 50)

shutil.rmtree(TEST_DIR, ignore_errors=True)
