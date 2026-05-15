"""
Audio denoising pipeline using ClearerVoice-Studio.
Steps are dynamically selected and ordered by the frontend.
"""
import os
import sys
import numpy as np
import soundfile as sf

_CLEARVOICE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ClearerVoice-Studio-main", "ClearerVoice-Studio-main", "clearvoice"
)
if _CLEARVOICE_PATH not in sys.path:
    sys.path.insert(0, _CLEARVOICE_PATH)

# Singleton model instances (lazy-loaded)
_se_model = None
_sr_model = None


def _get_se_model():
    global _se_model
    if _se_model is None:
        from clearvoice import ClearVoice
        _se_model = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    return _se_model


def _get_sr_model():
    global _sr_model
    if _sr_model is None:
        from clearvoice import ClearVoice
        _sr_model = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])
    return _sr_model


# ============================================================
# Step Registry
# ============================================================
# Each step function: (current_path, output_dir, base_name) -> StepResult
# StepResult: {"success": bool, "output_path": str, "discard_reason": str, "info": dict}

STEP_REGISTRY = {}  # Populated below

DEFAULT_STEPS = ["enhance", "super_resolve", "reverb", "silence", "vad", "pad"]

STEP_LABELS = {
    "enhance": "语音增强",
    "super_resolve": "超分辨率",
    "reverb": "混响检测",
    "silence": "静音过滤",
    "vad": "VAD 噪声检测",
    "pad": "静音规范化",
}


def _step_enhance(current_path, output_dir, base):
    """Speech Enhancement: MossFormer2_SE_48K."""
    out = os.path.join(output_dir, f"{base}_enhanced.wav")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return {"success": True, "output_path": out}
    try:
        cv = _get_se_model()
        wav = cv(input_path=current_path, online_write=False)
        cv.write(wav, output_path=out)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return {"success": True, "output_path": out}
        return {"success": False, "discard_reason": "语音增强写入失败"}
    except Exception as e:
        return {"success": False, "discard_reason": f"语音增强异常: {str(e)[:100]}"}


def _step_super_resolve(current_path, output_dir, base):
    """Speech Super Resolution: MossFormer2_SR_48K."""
    out = os.path.join(output_dir, f"{base}_sr.wav")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return {"success": True, "output_path": out}
    try:
        cv = _get_sr_model()
        wav = cv(input_path=current_path, online_write=False)
        cv.write(wav, output_path=out)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return {"success": True, "output_path": out}
        return {"success": True, "output_path": current_path, "info": {"fallback": "SR skipped"}}
    except Exception:
        return {"success": True, "output_path": current_path, "info": {"fallback": "SR unavailable"}}


def _step_reverb(current_path, output_dir, base):
    """Reverb detection."""
    from audio_pipeline import detect_reverb
    has_reverb, info = detect_reverb(current_path)
    if has_reverb:
        return {"success": False, "discard_reason": f"混响 (energy_ratio={info.get('energy_ratio','?')})", "info": info}
    return {"success": True, "output_path": current_path, "info": info}


def _step_silence(current_path, output_dir, base):
    """Silence ratio filter."""
    from audio_pipeline import check_silence_ratio
    is_bad, info = check_silence_ratio(current_path)
    if is_bad:
        return {"success": False, "discard_reason": f"静音比例过高 ({info.get('silence_ratio','?')})", "info": info}
    return {"success": True, "output_path": current_path, "info": info}


def _step_vad(current_path, output_dir, base):
    """VAD noise event detection."""
    from audio_pipeline import vad_filter
    has_noise, info = vad_filter(current_path)
    if has_noise:
        return {"success": False, "discard_reason": f"VAD噪声事件 ({info.get('reason','?')})", "info": info}
    return {"success": True, "output_path": current_path, "info": info}


def _step_pad(current_path, output_dir, base):
    """PAD silence normalization to 0.5s edges."""
    from audio_pipeline import pad_normalize
    out = os.path.join(output_dir, f"{base}_norm.wav")
    try:
        pad_normalize(current_path, out)
        return {"success": True, "output_path": out}
    except Exception as e:
        return {"success": False, "discard_reason": f"PAD失败: {str(e)[:100]}"}


STEP_REGISTRY = {
    "enhance": _step_enhance,
    "super_resolve": _step_super_resolve,
    "reverb": _step_reverb,
    "silence": _step_silence,
    "vad": _step_vad,
    "pad": _step_pad,
}


# ============================================================
# Dynamic Pipeline
# ============================================================
def run_full_denoise(input_path: str, output_dir: str, on_step=None, steps=None) -> dict:
    """Run denoise pipeline with dynamically selected and ordered steps.

    Args:
        input_path: Path to input WAV file.
        output_dir: Directory for output files.
        on_step: Callback(step_key, status, message).
        steps: List of step keys in desired order.
               Default: ["enhance", "super_resolve", "reverb", "silence", "vad", "pad"]

    Returns:
        {success, output_path, steps: [{step, status, message, info?}], discard_reason}
    """
    if steps is None:
        steps = DEFAULT_STEPS

    # Validate step keys
    valid_steps = [s for s in steps if s in STEP_REGISTRY]
    if not valid_steps:
        return {"success": False, "output_path": "", "steps": [],
                "discard_reason": "没有选择任何有效步骤"}

    all_steps = []
    base = os.path.splitext(os.path.basename(input_path))[0]
    current_path = input_path

    def _record(step_key, status, message, info=None):
        s = {"step": step_key, "status": status, "message": message}
        if info:
            s["info"] = info
        all_steps.append(s)
        if on_step:
            on_step(step_key, status, message)

    _record("pipeline", "running", f"开始降噪 ({len(valid_steps)} 步)")

    for sk in valid_steps:
        label = STEP_LABELS.get(sk, sk)
        fn = STEP_REGISTRY[sk]

        _record(sk, "running", label)
        try:
            result = fn(current_path, output_dir, base)
            if result.get("success"):
                current_path = result.get("output_path", current_path)
                _record(sk, "passed", label, info=result.get("info"))
            else:
                reason = result.get("discard_reason", label + " 未通过")
                _record(sk, "discarded", reason, info=result.get("info"))
                _record("result", "discarded", reason)
                return {"success": False, "output_path": "", "steps": all_steps,
                        "discard_reason": reason}
        except Exception as e:
            msg = f"{label}异常: {str(e)[:100]}"
            _record(sk, "error", msg)
            _record("result", "error", msg)
            return {"success": False, "output_path": "", "steps": all_steps,
                    "discard_reason": msg}

    _record("result", "passed", "全部通过")
    return {"success": True, "output_path": current_path, "steps": all_steps,
            "discard_reason": ""}
