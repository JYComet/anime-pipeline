"""
Audio normalization pipeline:
  Step 1: Reverb detection → discard if present
  Step 2: Silence ratio filter → discard if too much silence
  Step 3: VAD (Voice Activity Detection) → filter noise events
  Step 4: PAD (Pad/Trim silence) → normalize to 0.5s silence at edges
"""
import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
import threading
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# Config
# ============================================================
SR = 16000
HOP_MS = 0.01  # 10ms per frame
FRAME_LEN = 2048
HOP_LEN = int(SR * HOP_MS)  # 160 samples

# Step 1: Reverb
TAIL_ENERGY_RATIO_THRESH = 0.08
ZCR_TAIL_THRESH = 3000  # Hz

# Step 2: Silence
SIL_VOL_THRESHOLD = 0.006
SIL_LEN_THRESHOLD = 0.3  # seconds
MAX_SILENCE_RATIO = 0.5

# Step 3: VAD
RMS_THRESHOLD = 0.005
NON_SPEECH_DURATION_LIMIT = 1.0  # seconds
NON_SPEECH_ENERGY_THRESHOLD = 0.001

# Step 4: PAD
TARGET_SILENCE_DURATION = 0.5
SILENCE_THRESHOLD = 0.01


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    DISCARDED = "discarded"
    ERROR = "error"


@dataclass
class AudioJob:
    job_id: str
    name: str
    source_path: str
    status: str = "pending"
    steps: list[dict] = field(default_factory=list)
    output_path: str = ""
    progress: float = 0


# ============================================================
# Helpers
# ============================================================
def _rms(audio: np.ndarray, frame_len: int = FRAME_LEN, hop_len: int = HOP_LEN) -> np.ndarray:
    """Compute RMS energy per frame."""
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    rms_vals = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        rms_vals[i] = np.sqrt(np.mean(frame ** 2))
    return rms_vals


def _create_silence(duration_s: float, sr: int = SR) -> torch.Tensor:
    return torch.zeros(1, int(duration_s * sr))


# ============================================================
# Step 1: Reverb Detection
# ============================================================
def detect_reverb(audio_path: str) -> tuple[bool, dict]:
    """Detect if audio has noticeable reverb/tail.

    Returns (has_reverb, debug_info).
    """
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    if len(y) < sr * 0.5:  # too short
        return False, {"reason": "too_short"}

    rms = _rms(y)
    if len(rms) == 0:
        return False, {"reason": "no_frames"}

    peak = np.max(rms)

    # Find tail: last frame where energy > peak * 0.1
    tail_start_idx = len(rms) - 1
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] > peak * 0.1:
            tail_start_idx = i
            break

    tail_frames = len(rms) - tail_start_idx
    if tail_frames < int(0.5 / HOP_MS):  # tail < 0.5s → no reverb
        return False, {"reason": "tail_too_short", "tail_frames": int(tail_frames)}

    # Tail energy ratio
    tail_rms = np.mean(rms[tail_start_idx:])
    total_rms = np.mean(rms)
    energy_ratio = tail_rms / (total_rms + 1e-9)

    # Tail zero-crossing rate
    tail_samples = y[tail_start_idx * HOP_LEN:]
    zcr = librosa.feature.zero_crossing_rate(
        tail_samples, frame_length=FRAME_LEN, hop_length=HOP_LEN
    )[0]
    avg_zcr = np.mean(zcr) * sr / 2

    has_reverb = energy_ratio > TAIL_ENERGY_RATIO_THRESH or avg_zcr > ZCR_TAIL_THRESH
    return has_reverb, {
        "energy_ratio": round(float(energy_ratio), 4),
        "avg_zcr": round(float(avg_zcr), 1),
        "tail_frames": int(tail_frames),
    }


# ============================================================
# Step 2: Silence Ratio Filter
# ============================================================
def check_silence_ratio(audio_path: str) -> tuple[bool, dict]:
    """Check if audio has too much silence. Returns (is_bad, info)."""
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    total_len = len(y)
    if total_len == 0:
        return True, {"reason": "empty"}

    rms = _rms(y)
    if len(rms) == 0:
        return True, {"reason": "no_frames"}

    frame_len_s = HOP_MS  # seconds per frame
    min_sil_frames = int(SIL_LEN_THRESHOLD / frame_len_s)

    # Find continuous silence regions
    is_silent = rms < SIL_VOL_THRESHOLD
    total_silent_frames = np.sum(is_silent)
    silence_ratio = total_silent_frames / len(rms)

    is_bad = silence_ratio > MAX_SILENCE_RATIO
    return is_bad, {
        "silence_ratio": round(float(silence_ratio), 4),
        "threshold": MAX_SILENCE_RATIO,
        "total_frames": int(len(rms)),
        "silent_frames": int(total_silent_frames),
    }


# ============================================================
# Combined BGM / Reverb analysis
# ============================================================
def analyze_clip_bgm(audio_path: str) -> dict:
    """Analyze a clip for BGM (background music) and reverb characteristics.

    Returns a dict with:
        has_bgm: bool — likely has background music
        has_reverb: bool — likely has reverb/echo
        bgm_score: int — 0-6, higher = more likely BGM/reverb
        silence_ratio: float
        energy_ratio: float
        avg_zcr: float
        details: str — human-readable explanation
    """
    result = {
        "has_bgm": False,
        "has_reverb": False,
        "bgm_score": 0,
        "silence_ratio": 0.0,
        "energy_ratio": 0.0,
        "avg_zcr": 0.0,
        "details": "",
    }

    # Load audio
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception:
        result["details"] = "failed to load audio"
        return result

    if len(y) < sr * 0.3:
        result["details"] = "too short"
        return result

    rms = _rms(y)
    if len(rms) == 0:
        result["details"] = "no frames"
        return result

    # --- Silence ratio ---
    is_silent = rms < SIL_VOL_THRESHOLD
    silence_ratio = float(np.sum(is_silent) / len(rms))
    result["silence_ratio"] = round(silence_ratio, 4)

    # --- Reverb detection ---
    peak = np.max(rms)
    tail_start_idx = len(rms) - 1
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] > peak * 0.1:
            tail_start_idx = i
            break

    tail_frames = len(rms) - tail_start_idx
    energy_ratio = 0.0
    avg_zcr = 0.0
    has_reverb = False

    if tail_frames >= int(0.5 / HOP_MS):
        tail_rms = np.mean(rms[tail_start_idx:])
        total_rms = np.mean(rms)
        energy_ratio = float(tail_rms / (total_rms + 1e-9))
        result["energy_ratio"] = round(energy_ratio, 4)

        tail_samples = y[tail_start_idx * HOP_LEN:]
        if len(tail_samples) > FRAME_LEN:
            zcr = librosa.feature.zero_crossing_rate(
                tail_samples, frame_length=FRAME_LEN, hop_length=HOP_LEN
            )[0]
            avg_zcr = float(np.mean(zcr) * sr / 2)
            result["avg_zcr"] = round(avg_zcr, 1)

        has_reverb = energy_ratio > TAIL_ENERGY_RATIO_THRESH or avg_zcr > ZCR_TAIL_THRESH

    result["has_reverb"] = has_reverb

    # --- Energy variance (coefficient of variation) ---
    # Dialogue: words → attack/decay → high RMS variance
    # BGM: sustained music → low RMS variance
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_cv = rms_std / (rms_mean + 1e-9)
    is_steady = rms_cv < 0.8  # low variance → steady energy → likely BGM

    # --- Compute BGM score (duration-aware) ---
    score = 0
    reasons = []
    duration = len(y) / sr

    # Score = silence score + energy steadiness score + reverb score
    # All must agree for short clips; any can contribute for long clips.

    if duration < 3.0:
        # Short clips: need BOTH low silence AND steady energy to suspect BGM
        if silence_ratio < 0.05 and is_steady:
            score += 2
            reasons.append("短片段连续音频")
        elif silence_ratio < 0.05 and not is_steady:
            # Low silence but variable energy → likely dialogue (no pauses)
            pass
    elif duration < 6.0:
        if silence_ratio < 0.08 and is_steady:
            score += 3
            reasons.append("连续低静音(" + str(round(silence_ratio * 100)) + "%)")
        elif silence_ratio < 0.08:
            score += 1  # low silence but variable → maybe
            reasons.append("低静音波动(" + str(round(silence_ratio * 100)) + "%)")
        elif silence_ratio < 0.15 and is_steady:
            score += 1
    else:
        # Long clips: low silence strongly suggests BGM
        if silence_ratio < 0.10:
            score += 3
            reasons.append("极低静音比(" + str(round(silence_ratio * 100)) + "%)")
        elif silence_ratio < 0.18:
            score += 2
            reasons.append("较低静音比(" + str(round(silence_ratio * 100)) + "%)")
        elif silence_ratio < 0.30:
            score += 1

    # Reverb always counts heavily
    if has_reverb:
        score += 3
        if energy_ratio > TAIL_ENERGY_RATIO_THRESH:
            reasons.append("混响尾音")
        if avg_zcr > ZCR_TAIL_THRESH:
            reasons.append("尾音高频")

    # Long clips with sustained audio → extra confidence
    if duration > 8.0 and silence_ratio < 0.15:
        score += 1
        reasons.append("长片段持续音频")

    result["bgm_score"] = score
    result["has_bgm"] = score >= 4

    if reasons:
        result["details"] = "; ".join(reasons)
    elif score > 0:
        result["details"] = "轻微(" + str(score) + "分)"
    else:
        result["details"] = "正常对话"

    return result


# ============================================================
# Step 3: VAD (Voice Activity Detection)
# ============================================================
def vad_filter(audio_path: str) -> tuple[bool, dict]:
    """VAD-based noise event detection. Returns (has_noise_issue, info)."""
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    if len(y) < sr * 0.3:
        return False, {"reason": "too_short"}

    rms = _rms(y)
    if len(rms) == 0:
        return False, {"reason": "no_frames"}

    # Find non-speech intervals (below threshold)
    is_speech = rms >= RMS_THRESHOLD
    non_speech_intervals = []
    in_ns = False
    ns_start = 0
    for i, speech in enumerate(is_speech):
        if not speech and not in_ns:
            ns_start = i
            in_ns = True
        elif speech and in_ns:
            non_speech_intervals.append((ns_start * HOP_LEN, i * HOP_LEN))
            in_ns = False
    if in_ns:
        non_speech_intervals.append((ns_start * HOP_LEN, len(y)))

    # Check each non-speech interval
    for start, end in non_speech_intervals:
        seg_dur = (end - start) / sr
        if seg_dur < NON_SPEECH_DURATION_LIMIT:
            continue
        seg_wav = y[start:end]
        seg_rms_val = np.sqrt(np.mean(seg_wav ** 2)) if len(seg_wav) > 0 else 0
        if seg_rms_val < NON_SPEECH_ENERGY_THRESHOLD:
            continue
        # Long + loud non-speech = noise event
        return True, {
            "reason": "NOISE_EVENT_LONG_AND_LOUD",
            "segment_duration": round(seg_dur, 2),
            "segment_rms": round(float(seg_rms_val), 6),
            "total_non_speech": len(non_speech_intervals),
        }

    return False, {"non_speech_intervals": len(non_speech_intervals)}


# ============================================================
# Step 4: PAD (Pad/Trim silence to 0.5s at edges)
# ============================================================
def pad_normalize(audio_path: str, output_path: str) -> str:
    """Normalize silence to 0.5s at beginning and end."""
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    wav = torch.from_numpy(y).unsqueeze(0).float()
    target_samples = int(TARGET_SILENCE_DURATION * sr)

    # Find beginning silence
    rms_vals = _rms(y)
    beg_frames = 0
    for v in rms_vals:
        if v < SILENCE_THRESHOLD:
            beg_frames += 1
        else:
            break
    beginning_samples = beg_frames * HOP_LEN

    # Find ending silence
    end_frames = 0
    for v in reversed(rms_vals):
        if v < SILENCE_THRESHOLD:
            end_frames += 1
        else:
            break
    ending_samples = end_frames * HOP_LEN

    info = {"beginning_silence": round(beginning_samples / sr, 3),
            "ending_silence": round(ending_samples / sr, 3)}

    # Trim/pad beginning
    if beginning_samples > target_samples:
        start_idx = beginning_samples - target_samples
        wav = wav[:, start_idx:]
        info["beginning_action"] = "trimmed"
    elif beginning_samples < target_samples:
        pad_needed = target_samples - beginning_samples
        wav = torch.cat([_create_silence(pad_needed / sr, sr), wav], dim=1)
        info["beginning_action"] = "padded"
    else:
        info["beginning_action"] = "unchanged"

    # Trim/pad ending
    if ending_samples > target_samples:
        keep_len = wav.shape[1] - (ending_samples - target_samples)
        wav = wav[:, :keep_len]
        info["ending_action"] = "trimmed"
    elif ending_samples < target_samples:
        pad_needed = target_samples - ending_samples
        wav = torch.cat([wav, _create_silence(pad_needed / sr, sr)], dim=1)
        info["ending_action"] = "padded"
    else:
        info["ending_action"] = "unchanged"

    # Save using soundfile (more reliable than torchaudio for WAV)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_np = wav.squeeze(0).numpy()
    sf.write(output_path, audio_np, sr)
    return output_path


# ============================================================
# Full Pipeline
# ============================================================
_audio_jobs: dict[str, AudioJob] = {}
_audio_lock = threading.Lock()


def run_audio_pipeline(job_id: str, name: str, source_path: str, output_dir: str) -> AudioJob:
    """Run the full audio normalization pipeline on a single WAV file."""
    job = AudioJob(job_id=job_id, name=name, source_path=source_path)
    job.status = "running"
    job.progress = 5
    with _audio_lock:
        _audio_jobs[job_id] = job

    base = os.path.splitext(name)[0]

    # Step 1: Reverb detection
    job.progress = 10
    has_reverb, rev_info = detect_reverb(source_path)
    job.steps.append({"step": "reverb", "status": "passed" if not has_reverb else "discarded",
                       "info": rev_info})
    if has_reverb:
        job.status = "discarded"
        job.steps.append({"step": "result", "status": "discarded",
                          "message": f"检测到混响 (energy_ratio={rev_info.get('energy_ratio', '?')})，已丢弃"})
        job.progress = 100
        with _audio_lock:
            _audio_jobs[job_id] = job
        return job

    # Step 2: Silence ratio
    job.progress = 30
    is_silent, sil_info = check_silence_ratio(source_path)
    job.steps.append({"step": "silence_filter", "status": "passed" if not is_silent else "discarded",
                       "info": sil_info})
    if is_silent:
        job.status = "discarded"
        job.steps.append({"step": "result", "status": "discarded",
                          "message": f"静音占比过高 ({sil_info.get('silence_ratio', '?')})，已丢弃"})
        job.progress = 100
        with _audio_lock:
            _audio_jobs[job_id] = job
        return job

    # Step 3: VAD
    job.progress = 50
    has_noise, vad_info = vad_filter(source_path)
    job.steps.append({"step": "vad", "status": "passed" if not has_noise else "discarded",
                       "info": vad_info})
    if has_noise:
        job.status = "discarded"
        job.steps.append({"step": "result", "status": "discarded",
                          "message": f"检测到噪声事件 ({vad_info.get('reason', '?')})，已丢弃"})
        job.progress = 100
        with _audio_lock:
            _audio_jobs[job_id] = job
        return job

    # Step 4: PAD
    job.progress = 70
    out_path = os.path.join(output_dir, f"{base}_norm.wav")
    try:
        pad_normalize(source_path, out_path)
        job.output_path = out_path
        job.steps.append({"step": "pad", "status": "passed",
                           "message": f"静音规范化完成 → {os.path.basename(out_path)}"})
        job.status = "completed"
        job.progress = 100
        job.steps.append({"step": "result", "status": "passed",
                          "message": f"全部通过，输出: {os.path.basename(out_path)}"})
    except Exception as e:
        job.steps.append({"step": "pad", "status": "error", "message": str(e)})
        job.status = "error"

    with _audio_lock:
        _audio_jobs[job_id] = job
    return job


def get_job(job_id: str) -> AudioJob | None:
    return _audio_jobs.get(job_id)


def get_all_jobs() -> list[dict]:
    return [{
        "job_id": j.job_id,
        "name": j.name,
        "status": j.status,
        "steps": j.steps,
        "output_path": j.output_path,
        "progress": j.progress,
    } for j in _audio_jobs.values()]


def batch_process(audio_files: list[tuple[str, str]], output_dir: str) -> list[str]:
    """Process multiple WAV files. Returns list of job IDs."""
    import uuid
    job_ids = []
    for path, name in audio_files:
        if not os.path.exists(path):
            continue
        jid = uuid.uuid4().hex[:12]
        job_ids.append(jid)
        t = threading.Thread(target=run_audio_pipeline, args=(jid, name, path, output_dir), daemon=True)
        t.start()
    return job_ids
