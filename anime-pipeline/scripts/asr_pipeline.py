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

from config import (
    FFMPEG, FFPROBE, DATA_DIR, SUBTITLE_DIR,
    ASR_COMPARE_DIR, ASR_COMPARE_SUBTITLE_DIR, ASR_COMPARE_AUDIO_DIR,
    ASR_COMPARE_OUTPUT_DIR, ASR_COMPARE_DISCARD_DIR,
)

# --- ASR directories ---
ASR_DIR = os.path.join(DATA_DIR, "asr")
ASR_AUDIO_DIR = os.path.join(ASR_DIR, "audio")
ASR_SUBTITLE_DIR = os.path.join(ASR_DIR, "subtitles")

for d in [ASR_DIR, ASR_AUDIO_DIR, ASR_SUBTITLE_DIR,
          ASR_COMPARE_DIR, ASR_COMPARE_SUBTITLE_DIR, ASR_COMPARE_AUDIO_DIR,
          ASR_COMPARE_OUTPUT_DIR, ASR_COMPARE_DISCARD_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Available ASR models ---
ASR_MODELS = {
    "sensevoice": {
        "name": "SenseVoiceSmall",
        "model_id": "iic/SenseVoiceSmall",
        "description": "阿里 SenseVoice 多语言模型，支持中/英/日/韩/粤语等",
        "languages": ["auto", "zh", "en", "ja", "ko", "yue"],
        "abbr": "sensevoice",
    },
    "qwen3-asr": {
        "name": "Qwen3-ASR-1.7B",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "description": "Qwen3 ASR 多语言模型，支持中/英/日/韩/粤语等",
        "languages": ["auto", "zh", "en", "ja", "ko", "yue"],
        "abbr": "qwen3",
    },
    "parakeet-ja": {
        "name": "Parakeet-TDT-0.6B-ja",
        "model_id": "nvidia/parakeet-tdt_ctc-0.6b-ja",
        "description": "NVIDIA Parakeet TDT 日语特化模型 (NeMo)",
        "languages": ["ja"],
        "abbr": "parakeet",
        "framework": "nemo",
    },
    "cohere-transcribe": {
        "name": "Cohere Transcribe",
        "model_id": "CohereLabs/cohere-transcribe-03-2026",
        "description": "Cohere Transcribe 2B ASR 模型，支持14种语言含日语",
        "languages": ["ja", "en", "zh", "ko", "fr", "de", "it", "es", "pt", "el", "nl", "pl", "ar", "vi"],
        "abbr": "cohere",
        "framework": "transformers",
    },
}

# Models used for ASR comparison
COMPARE_MODELS = ["qwen3-asr", "cohere-transcribe"]

# --- Model singletons ---
_asr_models: dict[str, object] = {}  # keyed by model_key
_vad_model = None
_model_lock = threading.Lock()

# Language code -> full name mapping for Qwen3-ASR
_QWEN3_LANG_MAP = {
    "auto": None, "zh": "Chinese", "en": "English", "ja": "Japanese",
    "ko": "Korean", "yue": "Cantonese", "fr": "French", "de": "German",
    "es": "Spanish", "ru": "Russian", "ar": "Arabic", "th": "Thai",
    "vi": "Vietnamese", "it": "Italian", "pt": "Portuguese",
    "id": "Indonesian", "ms": "Malay", "nl": "Dutch", "pl": "Polish",
    "ro": "Romanian", "sv": "Swedish", "tr": "Turkish", "fi": "Finnish",
    "cs": "Czech", "da": "Danish", "el": "Greek", "hi": "Hindi",
    "hu": "Hungarian", "mk": "Macedonian", "fa": "Persian", "tl": "Filipino",
}

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


def _get_asr_model(model_key="qwen3-asr", device="cpu"):
    """Load (or reuse) the ASR model. Thread-safe lazy singleton, cached per model_key."""
    with _model_lock:
        if model_key in _asr_models:
            return _asr_models[model_key]

        model_info = ASR_MODELS.get(model_key)
        if model_info is None:
            raise ValueError(f"Unknown ASR model: {model_key}")

        # Qwen3-ASR uses its own package (qwen-asr), not funasr
        if model_key in ("qwen3-asr",):
            from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
            model = Qwen3ASRModel.from_pretrained(
                model_info["model_id"],
                max_inference_batch_size=16,
            )
            if device != "cpu":
                try:
                    model = model.cuda()
                except Exception:
                    pass
            _asr_models[model_key] = model
            return model

        # NeMo-based models (Parakeet)
        if model_info.get("framework") == "nemo":
            import nemo.collections.asr as nemo_asr
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_info["model_id"],
            )
            if device != "cpu":
                try:
                    model = model.cuda()
                except Exception:
                    pass
            _asr_models[model_key] = model
            return model

        # HuggingFace transformers-based models (Cohere Transcribe)
        if model_info.get("framework") == "transformers":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_info["model_id"], trust_remote_code=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_info["model_id"],
                trust_remote_code=True,
                device_map="auto" if device != "cpu" else "cpu",
                dtype="auto",
            )
            _asr_models[model_key] = (model, processor)
            return (model, processor)

        from funasr import AutoModel
        model = AutoModel(
            model=model_info["model_id"],
            trust_remote_code=True,
            device=device,
        )
        _asr_models[model_key] = model
        return model


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
    model_key: str = "qwen3-asr",
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

    # Process in small batches. Qwen3 needs batch_size=1 due to variable-length inputs.
    batch_size = 1 if model_key in ("qwen3-asr",) else 16
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

        # Run ASR on batch — use transcribe() for Qwen3, generate() for others
        if hasattr(asr_model, "transcribe"):
            # Qwen3-ASR: transcribe accepts (ndarray, sr) tuples, needs full language names
            audio_inputs = [(chunk, sr) for chunk in audio_batch]
            qwen3_lang = _QWEN3_LANG_MAP.get(lang) if lang else None
            transcribe_results = asr_model.transcribe(audio_inputs, language=qwen3_lang)
            for j, tr in enumerate(transcribe_results):
                text = (tr.text or "").strip()
                if not text:
                    continue
                start_ms, end_ms = time_batch[j]
                results.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
        else:
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


def _nemo_asr_pipeline(audio_path: str, model_key: str, language: str, device: str) -> list:
    """Run NeMo-based ASR on the full audio file and return a single-segment result.

    NeMo Parakeet models handle VAD internally and transcribe entire files.
    """
    duration_ms = int(get_audio_duration(audio_path) * 1000)
    asr_model = _get_asr_model(model_key, device=device)
    # NeMo transcribe returns a list of transcriptions (one per file)
    results = asr_model.transcribe([audio_path])
    if not results or not isinstance(results, list):
        return []
    text = results[0].text if hasattr(results[0], 'text') else str(results[0])
    text = text.strip()
    if not text:
        return []
    return [{"text": text, "start_ms": 0, "end_ms": duration_ms}]


def _transformers_asr_pipeline(audio_path: str, model_key: str, language: str, device: str) -> list:
    """Run transformers-based ASR (Cohere Transcribe) on the full audio file.

    Cohere Transcribe handles audio loading, resampling, and chunking internally.
    Returns a single-segment result with the full transcription text.
    """
    import torch
    from transformers.audio_utils import load_audio

    duration_ms = int(get_audio_duration(audio_path) * 1000)
    model, processor = _get_asr_model(model_key, device=device)

    # Load and process audio (automatically resampled to 16kHz mono)
    audio = load_audio(audio_path, sampling_rate=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=language)
    inputs = {k: v.to(model.device, dtype=model.dtype) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    text = processor.decode(outputs[0], skip_special_tokens=True).strip()

    if not text:
        return []
    return [{"text": text, "start_ms": 0, "end_ms": duration_ms}]


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
    model_key: str = "qwen3-asr",
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


def run_asr_on_audio(
    audio_path: str,
    output_dir: str,
    model_key: str = "qwen3-asr",
    language: str = "ja",
    device: str = "cpu",
    progress_callback=None,
) -> dict:
    """Run ASR directly on an audio file (no extraction step).

    The output SRT is named after the audio file (<basename>.srt) and
    saved to ``output_dir``.

    Returns dict with keys: audio_name, srt_path, segments_count, model_name, duration_sec.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    t0 = time.time()

    if progress_callback:
        progress_callback("loading_models", "加载模型中...")

    segments = vad_asr_pipeline(
        audio_path,
        model_key=model_key,
        language=language,
        device=device,
        progress_callback=progress_callback,
    )
    t1 = time.time()

    if progress_callback:
        progress_callback("generating_srt", "生成字幕文件...")

    os.makedirs(output_dir, exist_ok=True)
    srt_path = os.path.join(output_dir, base_name + ".srt")

    counter = 1
    while os.path.exists(srt_path):
        srt_path = os.path.join(output_dir, f"{base_name}_{counter}.srt")
        counter += 1

    if segments:
        segments_to_srt(segments, srt_path)
    else:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("")

    if progress_callback:
        progress_callback("completed", "完成")

    model_label = ASR_MODELS.get(model_key, {}).get("name", model_key)

    return {
        "audio_name": os.path.basename(audio_path),
        "srt_path": srt_path,
        "segments_count": len(segments),
        "model_name": model_label,
        "duration_sec": round(t1 - t0, 1),
    }


# --- ASR Comparison ---

def srt_to_plain_text(srt_path: str) -> str:
    """Extract normalized plain text from an SRT file for comparison."""
    if not os.path.exists(srt_path):
        return ""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip index numbers and timestamp lines
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        lines.append(line)

    text = " ".join(lines)
    # Normalize: remove punctuation, normalize whitespace, lowercase
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    # Remove all punctuation/symbols, keep only word characters and spaces
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N") or cat == "Zs":
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    text = "".join(cleaned)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.lower()


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compare_srt_texts(srt_path1: str, srt_path2: str) -> float:
    """Compare two SRT files and return difference percentage (0-100).

    Returns 0.0 for identical text, 100.0 for completely different text.
    """
    text1 = srt_to_plain_text(srt_path1)
    text2 = srt_to_plain_text(srt_path2)

    if not text1 and not text2:
        return 0.0
    if not text1 or not text2:
        return 100.0

    dist = _levenshtein(text1, text2)
    max_len = max(len(text1), len(text2))
    return round((dist / max_len) * 100, 1)


def compare_asr_pipeline(
    audio_path: str,
    language: str = "ja",
    device: str = "cpu",
    progress_callback=None,
    source_dir: str = "",
    model_a: str = "qwen3-asr",
    model_b: str = "cohere-transcribe",
) -> dict:
    """Run two selected ASR models on a single WAV file and compare results.

    Takes a WAV file directly (no extraction needed).
    SRT files are placed in a subdirectory named after source_dir to avoid mixing.
    Returns dict with keys: audio_path, results (per-model), diff_percent,
    flagged, srt_paths, duration_sec.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    duration = get_audio_duration(audio_path)
    compare_models = [model_a, model_b]

    # SRT output goes into a source-dir subfolder to keep different folders separate
    srt_out_dir = os.path.join(ASR_COMPARE_SUBTITLE_DIR, source_dir) if source_dir else ASR_COMPARE_SUBTITLE_DIR
    os.makedirs(srt_out_dir, exist_ok=True)

    if progress_callback:
        progress_callback("vad", 5)

    model_results = {}
    srt_paths = {}

    for i, model_key in enumerate(compare_models):
        model_info = ASR_MODELS[model_key]
        abbr = model_info.get("abbr", model_key)
        framework = model_info.get("framework", "funasr")

        if progress_callback:
            progress_callback(f"asr_{model_key}", 10 + i * 40)

        if framework == "nemo":
            segments = _nemo_asr_pipeline(
                audio_path,
                model_key=model_key,
                language=language,
                device=device,
            )
        elif framework == "transformers":
            segments = _transformers_asr_pipeline(
                audio_path,
                model_key=model_key,
                language=language,
                device=device,
            )
        else:
            segments = vad_asr_pipeline(
                audio_path,
                model_key=model_key,
                language=language,
                device=device,
                progress_callback=None,
            )

        srt_path = os.path.join(srt_out_dir, f"{base_name}_{abbr}.srt")
        counter = 1
        while os.path.exists(srt_path):
            srt_path = os.path.join(srt_out_dir, f"{base_name}_{abbr}_{counter}.srt")
            counter += 1

        if segments:
            segments_to_srt(segments, srt_path)
        else:
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("")

        model_results[model_key] = {
            "model_key": model_key,
            "model_name": model_info["name"],
            "abbr": abbr,
            "segments_count": len(segments),
            "srt_path": srt_path,
        }
        srt_paths[model_key] = srt_path

    if progress_callback:
        progress_callback("comparing", 90)

    # Check for empty results (0 segments = 0 KB SRT) from either model
    empty_models = []
    for model_key, mr in model_results.items():
        if mr["segments_count"] == 0:
            empty_models.append(mr["model_name"])

    # Compare the two SRT outputs
    diff_percent = compare_srt_texts(
        srt_paths[compare_models[0]], srt_paths[compare_models[1]]
    )
    flagged = diff_percent > 10.0 or len(empty_models) > 0

    if progress_callback:
        progress_callback("completed", 100)

    return {
        "audio_path": audio_path,
        "audio_name": base_name,
        "duration_sec": round(duration, 1),
        "results": model_results,
        "srt_paths": srt_paths,
        "diff_percent": diff_percent,
        "flagged": flagged,
        "empty_models": empty_models,
    }
