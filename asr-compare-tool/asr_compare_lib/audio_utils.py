"""
Audio utilities: VAD, format conversion, segment extraction.
"""
import os
import json
import threading
import subprocess
import uuid

import numpy as np
import soundfile as sf


def get_audio_duration(audio_path: str, ffprobe: str = "ffprobe") -> float:
    """Get audio duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))
    except Exception:
        return 0


def convert_to_pcm_wav(input_path: str, output_path: str,
                       ffmpeg: str = "ffmpeg") -> None:
    """Convert any audio to 16kHz mono s16le WAV via ffmpeg."""
    subprocess.run(
        [ffmpeg, "-y", "-i", input_path, "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", output_path],
        capture_output=True, check=True,
    )


# Formats that soundfile (libsndfile) cannot decode
_SF_UNSUPPORTED = {'.aac', '.mp3', '.m4a', '.wma', '.opus', '.wv'}


def ensure_pcm_wav(audio_path: str, ffmpeg: str = "ffmpeg",
                   temp_dir: str = None) -> str:
    """Ensure audio is 16kHz mono s16le WAV. Converts via ffmpeg if needed.

    Returns path to a valid PCM WAV (may be the original or a temp file).
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in _SF_UNSUPPORTED:
        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(audio_path), ".temp")
        os.makedirs(temp_dir, exist_ok=True)
        out_path = os.path.join(temp_dir, f"{uuid.uuid4().hex[:8]}.wav")
        convert_to_pcm_wav(audio_path, out_path, ffmpeg)
        return out_path

    # Check if already correct format
    try:
        info = sf.info(audio_path)
        if info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16":
            return audio_path
    except Exception:
        pass

    # Convert
    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(audio_path), ".temp")
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f"{uuid.uuid4().hex[:8]}.wav")
    convert_to_pcm_wav(audio_path, out_path, ffmpeg)
    return out_path


# ---------------------------------------------------------------------------
# VAD (Voice Activity Detection)
# ---------------------------------------------------------------------------
_silero_model = None
_silero_lock = threading.Lock()

from . import PROJECT_ROOT

# Default model path relative to this project
_SILERO_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "silero_vad")


def _load_silero_vad_local(model_dir: str = None):
    """Load Silero VAD ONNX model from a local directory.

    Uses OnnxWrapper from silero_vad package. The model file (silero_vad.onnx)
    is loaded from the given directory instead of the pip package's data dir.
    """
    global _silero_model
    if _silero_model is not None:
        return _silero_model

    with _silero_lock:
        if _silero_model is not None:
            return _silero_model

        from silero_vad.utils_vad import OnnxWrapper

        model_dir = model_dir or _SILERO_MODEL_DIR
        model_path = os.path.join(model_dir, "silero_vad.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                "Place silero_vad.onnx in models/silero_vad/"
            )
        _silero_model = OnnxWrapper(model_path, force_onnx_cpu=True)
        return _silero_model


def run_vad_silero(audio_path: str, model_dir: str = None) -> list:
    """Run Silero VAD and return speech segments as [(start_ms, end_ms), ...].

    Parameters tuned for clean studio speech (anime dubbing):
      - threshold=0.5: speech probability threshold
      - min_speech_duration_ms=250: minimum speech segment
      - min_silence_duration_ms=100: minimum silence gap
      - speech_pad_ms=30: padding around speech boundaries
    """
    import torch
    from silero_vad import get_speech_timestamps

    model = _load_silero_vad_local(model_dir)

    audio_np, sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import tempfile
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tf.close()
        try:
            convert_to_pcm_wav(audio_path, tf.name)
            audio_np, _sr = sf.read(tf.name, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            sr = _sr
        finally:
            try:
                os.unlink(tf.name)
            except OSError:
                pass

    wav = torch.from_numpy(audio_np)
    segments = get_speech_timestamps(
        wav, model,
        threshold=0.5,
        sampling_rate=16000,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
        return_seconds=False,
    )

    return [(int(s["start"] * 1000 / 16000), int(s["end"] * 1000 / 16000))
            for s in segments]


# ---------------------------------------------------------------------------
# Segment chunking
# ---------------------------------------------------------------------------
def chunk_vad_segments(vad_segments: list, min_s: float = 9.0,
                       max_s: float = 16.0) -> list:
    """Split VAD segments into chunks of roughly min_s-max_s seconds.

    Returns [(start_ms, end_ms), ...] in chronological order.
    """
    if not vad_segments:
        return []

    MIN_MS = int(min_s * 1000)
    MAX_MS = int(max_s * 1000)

    # Merge pass: combine very short adjacent segments
    merged = []
    buf_start, buf_end = vad_segments[0]
    for s, e in vad_segments[1:]:
        gap = s - buf_end
        total = e - buf_start
        if gap < 2000 and total <= MAX_MS:
            buf_end = e
        else:
            merged.append((buf_start, buf_end))
            buf_start, buf_end = s, e
    merged.append((buf_start, buf_end))

    # Split pass: chunk long segments
    chunks = []
    for start_ms, end_ms in merged:
        dur_ms = end_ms - start_ms
        if dur_ms <= MAX_MS:
            chunks.append((start_ms, end_ms))
        else:
            n = max(1, round(dur_ms / ((MIN_MS + MAX_MS) / 2)))
            piece = dur_ms / n
            for i in range(n):
                s = int(start_ms + i * piece)
                e = int(start_ms + (i + 1) * piece) if i < n - 1 else end_ms
                chunks.append((s, e))

    return chunks


# ---------------------------------------------------------------------------
# SRT output
# ---------------------------------------------------------------------------
def _ms_to_srt_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp HH:MM:SS,mmm."""
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list, output_srt: str) -> str:
    """Write segments to an SRT subtitle file. Returns the file path."""
    os.makedirs(os.path.dirname(output_srt) if os.path.dirname(output_srt) else ".", exist_ok=True)

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
