"""
ASR subtitle extraction pipeline.
Extracts audio from video, runs VAD + SenseVoiceSmall via FunASR, generates SRT subtitles.
"""
import os
import sys

# Ensure ffmpeg is on PATH for funasr's internal audio loading
_QUICKCUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "QuickCut")
if os.path.isdir(_QUICKCUT_DIR) and _QUICKCUT_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _QUICKCUT_DIR + os.pathsep + os.environ.get("PATH", "")

import subprocess
import json
import time
import threading
import re
import shutil

import numpy as np
import soundfile as sf

from config import FFMPEG, FFPROBE, DATA_DIR, SUBTITLE_DIR

# --- ASR directories ---
ASR_DIR = os.path.join(DATA_DIR, "asr")
ASR_AUDIO_DIR = os.path.join(ASR_DIR, "audio")
ASR_SUBTITLE_DIR = os.path.join(ASR_DIR, "subtitles")

for d in [ASR_DIR, ASR_AUDIO_DIR, ASR_SUBTITLE_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Available ASR models ---
ASR_MODELS = {
    "sensevoice": {
        "name": "SenseVoiceSmall",
        "model_id": "iic/SenseVoiceSmall",
        "description": "阿里 SenseVoice 多语言模型，支持中/英/日/韩/粤语等",
        "languages": ["auto", "zh", "en", "ja", "ko", "yue"],
    },
}

# --- Model singletons ---
_asr_model = None
_asr_model_name = None
_vad_model = None
_model_lock = threading.Lock()

# SenseVoice special token pattern: <|lang|>, <|EMO_XXX|>, <|Event_XXX|>, etc.
_TAG_RE = re.compile(r"<\|\s*([^|>]+)\s*\|>")


def _get_vad_model(device="cpu"):
    """Load (or reuse) the VAD model. Thread-safe lazy singleton."""
    global _vad_model
    with _model_lock:
        if _vad_model is not None:
            return _vad_model
        from funasr import AutoModel
        _vad_model = AutoModel(model="fsmn-vad", device=device)
        return _vad_model


def _get_asr_model(model_key="sensevoice", device="cpu"):
    """Load (or reuse) the ASR model. Thread-safe lazy singleton."""
    global _asr_model, _asr_model_name
    with _model_lock:
        if _asr_model is not None and _asr_model_name == model_key:
            return _asr_model

        model_info = ASR_MODELS.get(model_key)
        if model_info is None:
            raise ValueError(f"Unknown ASR model: {model_key}")

        from funasr import AutoModel
        _asr_model = AutoModel(
            model=model_info["model_id"],
            trust_remote_code=True,
            device=device,
        )
        _asr_model_name = model_key
        return _asr_model


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))
    except Exception:
        return 0


def extract_audio_to_wav(video_path: str, output_wav: str = "") -> str:
    """Extract audio from video as 16kHz mono 16-bit WAV for ASR.

    Returns the path to the WAV file, or empty string on failure.
    """
    if not output_wav:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_wav = os.path.join(ASR_AUDIO_DIR, f"{base}.wav")

    if os.path.exists(output_wav) and os.path.getsize(output_wav) > 0:
        return output_wav

    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    cmd = [
        FFMPEG, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav,
    ]

    result = subprocess.run(
        cmd, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=600,
    )

    if result.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 0:
        return output_wav

    if result.stderr:
        print(f"[asr] ffmpeg error: {result.stderr[:300]}")
    return ""


def _clean_tags(text: str) -> str:
    """Remove SenseVoice special tags like <|zh|>, <|EMO_XXX|>, leaving only transcription text."""
    return _TAG_RE.sub("", text).strip()


def run_vad(audio_path: str, device="cpu") -> list:
    """Run VAD and return speech segments as [(start_ms, end_ms), ...] in chronological order."""
    vad = _get_vad_model(device=device)
    results = vad.generate(input=audio_path)
    if not results or not isinstance(results, list):
        return []

    segments = []
    for item in results:
        if isinstance(item, dict):
            vals = item.get("value", [])
            for seg in vals:
                if isinstance(seg, (list, tuple)) and len(seg) == 2:
                    segments.append((int(seg[0]), int(seg[1])))

    segments.sort(key=lambda s: s[0])
    return segments


def vad_asr_pipeline(
    audio_path: str,
    model_key: str = "sensevoice",
    language: str = "ja",
    device: str = "cpu",
    progress_callback=None,
) -> list:
    """Run VAD → split audio → ASR on each speech segment → return [{text, start_ms, end_ms}].

    This is the core subtitle generation pipeline.
    """
    # Step 1: Run VAD
    if progress_callback:
        progress_callback("vad", 10)

    vad_segments = run_vad(audio_path, device=device)
    if not vad_segments:
        return []

    # Step 2: Load full audio into memory
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Step 3: Process each VAD segment through ASR
    asr_model = _get_asr_model(model_key, device=device)
    lang = None if language == "auto" else language
    total = len(vad_segments)
    results = []

    # Process in small batches for efficiency — sensevoice handles batches natively
    batch_size = 16
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        audio_batch = []
        time_batch = []  # (start_ms, end_ms) for each item in this batch

        for i in range(batch_start, batch_end):
            start_ms, end_ms = vad_segments[i]
            start_sample = int(start_ms * sr / 1000)
            end_sample = int(end_ms * sr / 1000)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if end_sample <= start_sample:
                continue

            chunk = audio[start_sample:end_sample]
            audio_batch.append(chunk)
            time_batch.append((start_ms, end_ms))

        if not audio_batch:
            continue

        # Run ASR on batch
        batch_results = asr_model.generate(input=audio_batch, language=lang)

        for j, item in enumerate(batch_results):
            if not isinstance(item, dict):
                continue
            text = _clean_tags(item.get("text", ""))
            if not text:
                continue
            start_ms, end_ms = time_batch[j]
            results.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})

        if progress_callback:
            pct = 10 + int((batch_end / total) * 80)
            progress_callback("asr", pct)

    return results


def _ms_to_srt_timestamp(ms: float) -> str:
    """Convert milliseconds to SRT timestamp HH:MM:SS,mmm."""
    total_sec = ms / 1000.0
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    secs = int(total_sec % 60)
    millis = int(total_sec * 1000) % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list, output_srt: str) -> str:
    """Write segments to an SRT subtitle file. Returns the file path."""
    os.makedirs(os.path.dirname(output_srt), exist_ok=True)

    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            idx = i + 1
            text = seg.get("text", "").strip()
            if not text:
                continue

            start_ms = seg.get("start_ms", 0)
            end_ms = seg.get("end_ms", 0)
            start_str = _ms_to_srt_timestamp(start_ms)
            end_str = _ms_to_srt_timestamp(end_ms)

            f.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")

    return output_srt


def run_asr_pipeline(
    video_path: str,
    model_key: str = "sensevoice",
    language: str = "ja",
    device: str = "cpu",
    progress_callback=None,
) -> dict:
    """Full ASR pipeline: extract audio → VAD → ASR per segment → generate SRT.

    Returns dict with keys: audio_path, srt_path, segments_count, model, model_name, duration_sec.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Step 1: Extract audio from video
    if progress_callback:
        progress_callback("extracting_audio", 5)
    t0 = time.time()

    audio_path = extract_audio_to_wav(video_path)
    if not audio_path:
        raise RuntimeError("音频提取失败 — ffmpeg 未能从视频中提取音频")

    duration = get_audio_duration(audio_path)
    t1 = time.time()

    # Step 2: VAD + ASR
    if progress_callback:
        progress_callback("loading_models", 7)

    segments = vad_asr_pipeline(
        audio_path,
        model_key=model_key,
        language=language,
        device=device,
        progress_callback=progress_callback,
    )
    t2 = time.time()

    # Step 3: Generate SRT
    if progress_callback:
        progress_callback("generating_srt", 93)

    model_label = ASR_MODELS[model_key]["name"]
    srt_path = os.path.join(ASR_SUBTITLE_DIR, f"{base_name}_{model_key}.srt")

    counter = 1
    while os.path.exists(srt_path):
        srt_path = os.path.join(ASR_SUBTITLE_DIR, f"{base_name}_{model_key}_{counter}.srt")
        counter += 1

    if segments:
        segments_to_srt(segments, srt_path)
    else:
        # Write empty file to indicate processing happened but no speech found
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("")

    # Copy SRT to the shared subtitles directory so it appears in the extract page
    srt_filename = os.path.basename(srt_path)
    srt_copy_path = os.path.join(SUBTITLE_DIR, srt_filename)
    shutil.copy2(srt_path, srt_copy_path)

    if progress_callback:
        progress_callback("completed", 100)

    return {
        "audio_path": audio_path,
        "srt_path": srt_path,
        "segments_count": len(segments),
        "model": model_key,
        "model_name": model_label,
        "duration_sec": round(duration, 1),
        "extract_time_sec": round(t1 - t0, 1),
        "asr_time_sec": round(t2 - t1, 1),
    }
