"""
ASR subtitle extraction pipeline.
Extracts audio from video, runs VAD + SenseVoiceSmall via FunASR, generates SRT subtitles.
"""
import os
import sys
import subprocess
import json
import time
import threading
import re
import shutil

import numpy as np
import soundfile as sf
import torch
import concurrent.futures

from config import (
    COMICUT_ROOT, FFMPEG, FFPROBE, DATA_DIR, SUBTITLE_DIR, TEMP_DIR,
    ASR_DIR, ASR_AUDIO_DIR, ASR_SUBTITLE_DIR,
    ASR_COMPARE_DIR, ASR_COMPARE_SUBTITLE_DIR, ASR_COMPARE_AUDIO_DIR,
    ASR_COMPARE_OUTPUT_DIR, ASR_COMPARE_DISCARD_DIR,
    ASR_COMPARE_SEGMENTS_DIR, ASR_COMPARE_KEPT_DIR,
    HF_CACHE_DIR, MS_CACHE_DIR,
)

# Ensure ffmpeg is on PATH for funasr's internal audio loading
_QUICKCUT_DIR = os.path.join(COMICUT_ROOT, "QuickCut")
if os.path.isdir(_QUICKCUT_DIR) and _QUICKCUT_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _QUICKCUT_DIR + os.pathsep + os.environ.get("PATH", "")

# --- Monkey-patch Modelscope patcher bug ---
# The modelscope patcher monkey-patches transformers' get_class_from_dynamic_module
# and tries to mutate the *args tuple with args[0]=..., which raises:
#   TypeError: 'tuple' object does not support item assignment
# This breaks all models loaded with trust_remote_code=True (cohere-transcribe, etc.)
# when modelscope is installed. We fix this by converting tuple args to a mutable list
# before the patcher body runs.
try:
    import modelscope.utils.hf_util.patcher as _ms_patcher_mod
    _ms_orig = _ms_patcher_mod.get_class_from_dynamic_module

    def _ms_fixed_get_class_from_dynamic_module(class_reference, *args, **kwargs):
        return _ms_orig(class_reference, *list(args), **kwargs)

    _ms_patcher_mod.get_class_from_dynamic_module = _ms_fixed_get_class_from_dynamic_module
except Exception:
    pass

# --- Monkey-patch transformers torch.load safety check ---
# transformers 4.57+ requires torch>=2.6 for torch.load with weights_only=True
# (CVE-2025-32434). This project uses torch 2.5.x. We patch the check at every
# import site to allow loading legacy .bin checkpoints (needed by firered-asr2).
try:
    import transformers.utils.import_utils as _tf_iu
    _tf_iu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as _tf_mu
    _tf_mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# Extensions that soundfile (libsndfile) cannot decode — need ffmpeg pre-conversion
_SF_UNSUPPORTED = {'.aac', '.mp3', '.m4a', '.wma', '.opus', '.wv'}


def _convert_to_pcm_wav(input_path: str, output_path: str) -> None:
    """Convert any audio to 16kHz mono s16le WAV via ffmpeg."""
    subprocess.run(
        [FFMPEG, '-y', '-i', input_path, '-ar', '16000', '-ac', '1',
         '-sample_fmt', 's16', output_path],
        capture_output=True, check=True,
    )


# --- Available ASR models ---
ASR_MODELS = {
    "qwen3-asr": {
        "name": "Qwen3-ASR-1.7B",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "description": "Qwen3 多语言模型，中/英/日/韩/粤语",
        "languages": ["auto", "zh", "en", "ja", "ko", "yue"],
        "abbr": "qwen3",
    },
    "cohere-transcribe": {
        "name": "Cohere Transcribe",
        "model_id": "CohereLabs/cohere-transcribe-03-2026",
        "description": "Cohere Transcribe 多语言模型，14 种语言",
        "languages": ["auto", "zh", "en", "ja", "ko", "de", "fr", "es", "pt", "ar", "ru", "hi", "tr", "vi", "nl", "id"],
        "abbr": "cohere",
        "framework": "transformers",
    },
    "whisper-base": {
        "name": "Whisper Base",
        "model_id": "openai/whisper-base",
        "description": "OpenAI Whisper Base，99 种语言，需 16kHz 单声道",
        "languages": ["auto", "zh", "en", "ja", "ko", "de", "fr", "es", "pt", "ar", "ru", "hi", "tr", "vi", "nl", "id", "it"],
        "abbr": "whisper",
        "framework": "whisper",
    },
    "firered-asr2": {
        "name": "FireRedASR2-AED",
        "model_id": "",
        "description": "FireRedASR2 AED 模型，中/英文，自带 VAD+LID+标点",
        "languages": ["auto", "zh", "en"],
        "abbr": "firered",
        "framework": "firered",
    },
    "sensevoice-small": {
        "name": "SenseVoiceSmall",
        "model_id": "iic/SenseVoiceSmall",
        "description": "阿里 SenseVoiceSmall，中/英/日/韩/粤语，含情感/事件标签",
        "languages": ["auto", "zh", "en", "ja", "ko", "yue"],
        "abbr": "svs",
        "framework": "funasr",
    },
    "paraformer-large": {
        "name": "Paraformer-Large",
        "model_id": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "description": "阿里 Paraformer-Large，中文普通话专用，自带 VAD+标点",
        "languages": ["zh"],
        "abbr": "pf",
        "framework": "funasr",
    },
    "qwen3-asr-api": {
        "name": "Qwen3-ASR-Flash (API)",
        "model_id": "",
        "description": "阿里云百炼 Qwen3-ASR-Flash，中文/多语言，API 调用",
        "languages": ["zh"],
        "abbr": "qwen3-api",
        "framework": "api",
        "api_model": "qwen3-asr-flash",
    },
    "paraformer-v2-api": {
        "name": "Paraformer语音识别-v2 (API)",
        "model_id": "",
        "description": "阿里云百炼 Paraformer语音识别-v2，中文普通话，API 调用",
        "languages": ["zh"],
        "abbr": "pfv2-api",
        "framework": "api",
        "api_model": "paraformer-v2",
    },
    "paraformer-8k-api": {
        "name": "Paraformer语音识别-8k-v1 (API)",
        "model_id": "",
        "description": "阿里云百炼 Paraformer语音识别-8k-v1，中文普通话 8kHz，API 调用",
        "languages": ["zh"],
        "abbr": "pf8k-api",
        "framework": "api",
        "api_model": "paraformer-8k-v1",
    },
}

# Models used for ASR comparison
COMPARE_MODELS = ["qwen3-asr", "cohere-transcribe", "whisper-base", "firered-asr2", "sensevoice-small", "paraformer-large", "qwen3-asr-api", "paraformer-v2-api", "paraformer-8k-api"]

# --- Model singletons ---
_asr_models: dict[tuple, object] = {}  # keyed by (model_key, device, use_fp16, use_flash_attn)
_vad_model = None
_model_lock = threading.Lock()
_compile_works = None  # None=untested, True=works, False=unavailable


def _check_torch_compile():
    """Pre-flight test: can torch.compile actually work on this system?

    torch.compile requires Triton (Linux) or MSVC (Windows) for GPU codegen.
    On systems lacking both, we skip compiling to avoid runtime crashes.
    """
    global _compile_works
    if _compile_works is not None:
        return _compile_works
    try:
        import os
        os.environ.setdefault("PYTHONUTF8", "1")  # fix GBK encoding on Chinese Windows
        t = torch.tensor([1.0], device="cuda")

        @torch.compile(mode="default")
        def _test(x):
            return x * 2

        _test(t)
        _compile_works = True
    except Exception:
        _compile_works = False
    return _compile_works

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


def _get_vad_model(device="cuda"):
    """Load (or reuse) the VAD model. Thread-safe lazy singleton."""
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for VAD model. CPU fallback is disabled.")
    global _vad_model
    with _model_lock:
        if _vad_model is not None:
            return _vad_model
        from funasr import AutoModel
        _vad_model = AutoModel(model="fsmn-vad", device=device, model_dir=MS_CACHE_DIR)
        return _vad_model


def _get_asr_model(model_key="qwen3-asr", device="cuda", use_fp16=True, use_flash_attn=True, use_compile=True):
    """Load (or reuse) the ASR model. Thread-safe lazy singleton, cached per model_key.

    Optimizations (all enabled by default, gracefully fall back):
      - use_fp16: load in fp16/bf16 for 2x faster inference
      - use_flash_attn: enable FlashAttention 2 for faster transformer layers
      - use_compile: torch.compile for 30-50% faster repeated forward passes
    """
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for ASR model. CPU fallback is disabled.")

    # Cache key encodes optimization flags so different configs don't collide
    cache_key = (model_key, device, use_fp16, use_flash_attn)

    # Fast path: already loaded
    if cache_key in _asr_models:
        return _asr_models[cache_key]

    with _model_lock:
        # Double-check after acquiring lock
        if cache_key in _asr_models:
            return _asr_models[cache_key]

        model_info = ASR_MODELS.get(model_key)
        if model_info is None:
            raise ValueError(f"Unknown ASR model: {model_key}")

        # Determine dtype
        dtype = None
        if use_fp16 and device != "cpu":
            try:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                dtype = None

        # FlashAttention / SDPA — only on GPU
        attn_kwargs = {}
        if use_flash_attn and device != "cpu":
            try:
                import flash_attn  # noqa: F401
                attn_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                try:
                    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                        torch.backends.cuda.enable_flash_sdp(True)
                    attn_kwargs["attn_implementation"] = "sdpa"
                except Exception:
                    pass

        # Qwen3-ASR uses its own package (qwen-asr), not funasr
        if model_key in ("qwen3-asr",):
            from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

            load_kwargs = {"max_inference_batch_size": 16}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            load_kwargs.update(attn_kwargs)

            model = Qwen3ASRModel.from_pretrained(
                model_info["model_id"],
                cache_dir=HF_CACHE_DIR,
                local_files_only=True,
                **load_kwargs,
            )
            if device != "cpu":
                try:
                    model.model = model.model.cuda()
                except Exception:
                    pass
            _asr_models[cache_key] = model
        elif model_info.get("framework") == "nemo":
            import nemo.collections.asr as nemo_asr
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_info["model_id"],
            )
            if device != "cpu":
                try:
                    model = model.cuda()
                except Exception:
                    pass
            _asr_models[cache_key] = model
        elif model_info.get("framework") in ("transformers", "whisper"):
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            load_kwargs = {"trust_remote_code": True}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            load_kwargs.update(attn_kwargs)
            if device != "cpu":
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = "cpu"

            processor = AutoProcessor.from_pretrained(model_info["model_id"], trust_remote_code=True,
                                                      cache_dir=HF_CACHE_DIR, local_files_only=True)
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_info["model_id"],
                    cache_dir=HF_CACHE_DIR,
                    local_files_only=True,
                    **load_kwargs,
                )
            except (ValueError, RuntimeError) as e:
                # Some models (e.g. cohere-transcribe) don't support SDPA/flash_attn.
                # Fall back to eager attention implementation.
                err_msg = str(e)
                if "attn_implementation" in err_msg or "attention" in err_msg.lower():
                    fallback_kwargs = {k: v for k, v in load_kwargs.items()
                                       if k != "attn_implementation"}
                    fallback_kwargs["attn_implementation"] = "eager"
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_info["model_id"],
                        cache_dir=HF_CACHE_DIR,
                        local_files_only=True,
                        **fallback_kwargs,
                    )
                else:
                    raise
            _asr_models[cache_key] = (model, processor)
        elif model_info.get("framework") == "firered":
            import config as _cfg
            models_dir = getattr(_cfg, 'FIRERED_ASR2_MODELS_DIR', os.path.join(DATA_DIR, "models", "firered_asr2"))
            firered_path = getattr(_cfg, 'FIRERED_ASR2S_PATH', os.path.join(COMICUT_ROOT, "FireRedASR2S"))

            if firered_path not in sys.path:
                sys.path.insert(0, firered_path)

            from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
            from fireredasr2s.fireredasr2 import FireRedAsr2Config
            from fireredasr2s.fireredvad import FireRedVadConfig
            from fireredasr2s.fireredlid import FireRedLidConfig
            from fireredasr2s.fireredpunc import FireRedPuncConfig

            gpu = device != "cpu"
            vad_cfg = FireRedVadConfig(use_gpu=gpu)
            lid_cfg = FireRedLidConfig(use_gpu=gpu)
            asr_cfg = FireRedAsr2Config(
                use_gpu=gpu,
                use_half=use_fp16 and gpu,
                return_timestamp=True,
            )
            punc_cfg = FireRedPuncConfig(use_gpu=gpu)

            sys_cfg = FireRedAsr2SystemConfig(
                os.path.join(models_dir, "FireRedVAD", "VAD"),
                os.path.join(models_dir, "FireRedLID"),
                "aed",
                os.path.join(models_dir, "FireRedASR2-AED"),
                os.path.join(models_dir, "FireRedPunc"),
                vad_cfg, lid_cfg, asr_cfg, punc_cfg,
                enable_vad=1, enable_lid=1, enable_punc=1,
            )
            model = FireRedAsr2System(sys_cfg)
            _asr_models[cache_key] = model
        else:
            from funasr import AutoModel
            model = AutoModel(
                model=model_info["model_id"],
                trust_remote_code=True,
                device=device,
                model_dir=MS_CACHE_DIR,
            )
            _asr_models[cache_key] = model

    # torch.compile outside the lock — it's slow and doesn't need mutual exclusion
    if use_compile and _check_torch_compile():
        try:
            if model_key in ("qwen3-asr",):
                model.model = torch.compile(model.model, mode="reduce-overhead")
            elif model_info.get("framework") in ("nemo", "transformers", "whisper"):
                compiled = torch.compile(model, mode="reduce-overhead")
                if model_info.get("framework") in ("transformers", "whisper"):
                    _asr_models[cache_key] = (compiled, processor)
                else:
                    _asr_models[cache_key] = compiled
        except Exception:
            try:
                if model_key in ("qwen3-asr",):
                    model.model = torch.compile(model.model, mode="default")
                elif model_info.get("framework") in ("nemo", "transformers", "whisper"):
                    compiled = torch.compile(model, mode="default")
                    if model_info.get("framework") in ("transformers", "whisper"):
                        _asr_models[cache_key] = (compiled, processor)
                    else:
                        _asr_models[cache_key] = compiled
            except Exception:
                pass

    return _asr_models[cache_key]


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


def run_vad(audio_path: str, device="cuda") -> list:
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


def _pad_and_batch(segments_with_times, audio, sr, max_batch_size=16, max_length_ratio=3.0):
    """Group VAD segments by similar length and pad to batch.

    Sorts segments by duration, groups them so the longest/shortest ratio
    in each batch stays below max_length_ratio, then pads to equal length.
    This minimizes wasted compute on padding while enabling batched GPU inference.

    Yields (padded_audio_batch, time_batch, orig_lengths).
    """
    # Sort by duration (shortest first) so similar lengths cluster together
    indexed = [(end_ms - start_ms, start_ms, end_ms,
                int(start_ms * sr / 1000), int(end_ms * sr / 1000))
               for (start_ms, end_ms) in segments_with_times]
    indexed.sort(key=lambda x: x[0])

    i = 0
    n = len(indexed)
    while i < n:
        # Determine batch: group segments with similar length
        batch_end = min(i + max_batch_size, n)
        # Ensure length ratio within batch is bounded
        min_dur = indexed[i][0]
        for j in range(i + 1, batch_end):
            if indexed[j][0] > min_dur * max_length_ratio:
                batch_end = j
                break

        group = indexed[i:batch_end]
        chunks = []
        times = []
        orig_lens = []

        for _, start_ms, end_ms, start_samp, end_samp in group:
            start_samp = max(0, start_samp)
            end_samp = min(len(audio), end_samp)
            chunk = audio[start_samp:end_samp] if end_samp > start_samp else np.zeros(160, dtype=np.float32)
            chunks.append(chunk)
            times.append((start_ms, end_ms))
            orig_lens.append(len(chunk))

        max_len = max(orig_lens)
        # Pad all chunks to max_len
        padded = np.zeros((len(chunks), max_len), dtype=np.float32)
        for ci, ch in enumerate(chunks):
            padded[ci, :len(ch)] = ch

        yield padded, times, orig_lens
        i = batch_end


def _preload_asr_model_async(model_key="qwen3-asr", device="cuda"):
    """Start loading the ASR model in a background thread.

    Call this before running VAD so the model is ready (or nearly ready)
    by the time VAD completes. Returns a concurrent.futures.Future.
    """
    def _load():
        return _get_asr_model(model_key, device=device)
    return concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(_load)


_SENTENCE_END_RE = re.compile(r".*[。！？.!?]$")


def _merge_segments_by_text(segments: list) -> list:
    """Merge consecutive short VAD fragments into natural sentences.

    Uses two criteria (either triggers a merge):
    1. Text doesn't end with sentence-ending punctuation (。！？.!?)
    2. Segment duration is very short (< 3s), suggesting a mid-sentence pause

    Segments shorter than 3 seconds are merged with the next segment.
    Combined segments longer than 45 seconds are NOT merged further.
    """
    if not segments or len(segments) <= 1:
        return segments

    merged = []
    buf = None  # accumulates text/timestamps

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        if buf is None:
            buf = dict(seg)
            continue

        buf_dur = (buf["end_ms"] - buf["start_ms"]) / 1000
        seg_dur = (seg["end_ms"] - seg["start_ms"]) / 1000
        combined_dur = (seg["end_ms"] - buf["start_ms"]) / 1000

        # Don't merge if combined would be too long
        if combined_dur > 45:
            merged.append(buf)
            buf = dict(seg)
            continue

        is_sentence_end = bool(_SENTENCE_END_RE.match(buf["text"]))
        buf_is_short = buf_dur < 3.0
        seg_is_short = seg_dur < 3.0

        # Merge if: buffer doesn't end with punctuation, or either segment is very short
        if not is_sentence_end or buf_is_short or seg_is_short:
            buf["text"] = buf["text"] + text
            buf["end_ms"] = seg["end_ms"]
        else:
            merged.append(buf)
            buf = dict(seg)

    if buf is not None:
        merged.append(buf)

    return merged


def _is_hotword_hallucination(text: str, audio_chunk, hotword_terms: set) -> bool:
    """Detect if ASR output is a hotword hallucination during silence.

    When given hotwords as context, some ASR models (Qwen3-ASR) may output
    the hotwords themselves as transcription text during silent / near-silent
    segments.  This filter checks two conditions:
      1. The text is dominated by hotword terms (>50% of characters).
      2. The audio energy is very low (<1% of full-scale RMS).
    If both are true the segment is discarded.
    """
    if not hotword_terms or not text:
        return False

    # Condition 1: hotword coverage ratio
    matched_chars = 0
    remaining = text
    for term in hotword_terms:
        pos = 0
        while True:
            pos = remaining.find(term, pos)
            if pos == -1:
                break
            matched_chars += len(term)
            pos += len(term)
    total_chars = len(text.replace(" ", "").replace(",", "").replace("，", "").replace("。", "").replace("！", "").replace("？", "").replace(".", "").replace("!", "").replace("?", ""))
    if total_chars == 0:
        return False
    hw_ratio = matched_chars / total_chars

    # Condition 2: audio energy (RMS relative to full-scale float32)
    import numpy as np
    if len(audio_chunk) == 0:
        return hw_ratio > 0.5
    rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
    # Full-scale sine RMS ≈ 0.707; threshold at ~0.7% ≈ 0.005
    is_silent = rms < 0.005

    return hw_ratio > 0.5 and is_silent


def vad_asr_pipeline(
    audio_path: str,
    model_key: str = "qwen3-asr",
    language: str = "zh",
    device: str = "cuda",
    progress_callback=None,
    hotwords: str = "",
) -> list:
    """Run VAD → split audio → ASR on each speech segment → return [{text, start_ms, end_ms}].

    This is the core subtitle generation pipeline.
    Optimized with: batched VAD segments, fp16 inference, FlashAttention 2, torch.compile.

    Args:
        hotwords: Optional context string with proper nouns / domain terms to improve
                  recognition accuracy. Passed as system prompt to Qwen3-ASR.
    """
    # Step 1: Start ASR model preloading in background (opt 7: parallel loading)
    asr_future = _preload_asr_model_async(model_key, device)

    # Step 2: Run VAD while ASR model loads in background
    if progress_callback:
        progress_callback("vad", 10)

    vad_segments = run_vad(audio_path, device=device)
    if not vad_segments:
        return []

    # Step 3: Load full audio into memory
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Step 4: Wait for ASR model to finish loading (if not already done)
    _ = asr_future.result()
    asr_model = _get_asr_model(model_key, device=device)
    lang = None if language == "auto" else language
    total = len(vad_segments)
    results = []

    # Step 5: Process in padded batches (opt 1: batched inference)
    # Qwen3-ASR-1.7B is too large for VAD-segment batching on 8GB GPUs (OOM).
    # Use batch_size=1 for Qwen3; other models (SenseVoice, Parakeet, etc.) benefit from batching.
    is_qwen3 = model_key in ("qwen3-asr",)

    # Build hotword term set for silence hallucination filtering
    _hotword_terms = set()
    if hotwords:
        for t in re.split(r"[,，、\s]+", hotwords):
            t = t.strip()
            if t:
                _hotword_terms.add(t)

    if is_qwen3:
        # Qwen3: one VAD segment at a time (model is 1.7B, OOM risk with batching)
        for i, (start_ms, end_ms) in enumerate(vad_segments):
            start_sample = max(0, int(start_ms * sr / 1000))
            end_sample = min(len(audio), int(end_ms * sr / 1000))
            if end_sample <= start_sample:
                continue
            chunk = audio[start_sample:end_sample]
            try:
                qwen3_lang = _QWEN3_LANG_MAP.get(lang) if lang else None
                tr_results = asr_model.transcribe([(chunk, sr)], language=qwen3_lang, context=hotwords)
                for tr in tr_results:
                    text = (tr.text or "").strip()
                    if text and not _is_hotword_hallucination(text, chunk, _hotword_terms):
                        results.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
            except Exception:
                continue
            if progress_callback and total > 0:
                pct = 10 + int(((i + 1) / total) * 80)
                progress_callback("asr", pct)
    else:
        # Other models (SenseVoice etc.): batched inference
        for padded_batch, time_batch, orig_lens in _pad_and_batch(
            vad_segments, audio, sr, max_batch_size=16
        ):
            if padded_batch.size == 0:
                continue
            batch_inputs = [(padded_batch[bi, :orig_lens[bi]], sr) for bi in range(len(time_batch))]
            batch_results = asr_model.generate(input=[inp[0] for inp in batch_inputs], language=lang)
            for j, item in enumerate(batch_results):
                if not isinstance(item, dict):
                    continue
                text = _clean_tags(item.get("text", ""))
                if not text:
                    continue
                start_ms, end_ms = time_batch[j]
                results.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
            if progress_callback:
                processed = len(results) if results else 0
                pct = 10 + int((processed / total) * 80) if total > 0 else 50
                progress_callback("asr", pct)

    return _merge_segments_by_text(results)


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


def _whisper_asr_pipeline(audio_path: str, model_key: str, language: str, device: str) -> list:
    """Run OpenAI Whisper via HuggingFace transformers on the full audio file.

    Whisper handles feature extraction and language detection internally.
    Returns a single-segment result with the full transcription text.
    """
    import librosa

    duration_ms = int(get_audio_duration(audio_path) * 1000)
    model, processor = _get_asr_model(model_key, device=device)

    audio, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
    if len(audio) < sr * 0.1:
        return []

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    if hasattr(model, 'device'):
        input_features = input_features.to(model.device, dtype=model.dtype)

    gen_kwargs = {"max_new_tokens": 448}
    if language and language != "auto":
        try:
            gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )
        except Exception:
            pass  # fall back to auto-detect

    with torch.no_grad():
        predicted_ids = model.generate(input_features, **gen_kwargs)
    text = processor.decode(predicted_ids[0], skip_special_tokens=True).strip()

    if not text:
        return []
    return [{"text": text, "start_ms": 0, "end_ms": duration_ms}]


def _firered_asr_pipeline(audio_path: str, model_key: str, language: str, device: str) -> list:
    """Run FireRedASR2 on the full audio file and return timestamped segments.

    FireRedASR2S includes built-in VAD, so it handles segmentation internally.
    Returns a list of {text, start_ms, end_ms} dicts.
    """
    # FireRedASR2S requires exactly 16kHz mono int16 WAV — resample if needed
    work_path = audio_path
    temp_wav = None
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        if info.samplerate != 16000 or info.channels != 1:
            os.makedirs(TEMP_DIR, exist_ok=True)
            temp_wav = os.path.join(TEMP_DIR, f"firered_{os.path.basename(audio_path)}")
            _convert_to_pcm_wav(audio_path, temp_wav)
            work_path = temp_wav

        asr_model = _get_asr_model(model_key, device=device)
        result = asr_model.process(work_path)

        segments = []
        for sent in result.get("sentences", []):
            text = sent.get("text", "").strip()
            if text:
                segments.append({
                    "text": text,
                    "start_ms": sent["start_ms"],
                    "end_ms": sent["end_ms"],
                })

        # Fallback: if sentences is empty, use top-level text + vad_segments_ms
        if not segments and result.get("text"):
            vad_segs = result.get("vad_segments_ms", [])
            full_text = result["text"].strip()
            if vad_segs and len(vad_segs) == 1:
                segments.append({
                    "text": full_text,
                    "start_ms": vad_segs[0][0],
                    "end_ms": vad_segs[0][1],
                })

        return segments
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


def _api_asr_pipeline(audio_path: str, model_key: str, language: str, device: str, hotwords: str = "") -> list:
    """Run API-based ASR via Alibaba Cloud DashScope (Bailian) on audio file.

    Returns a list of {text, start_ms, end_ms} dicts, or a single-segment result
    when sentence-level timestamps are not available.
    """
    import requests
    import json as _json
    import base64
    from config import DASHSCOPE_API_KEY, DASHSCOPE_API_BASE

    model_info = ASR_MODELS[model_key]
    api_model = model_info["api_model"]

    if not DASHSCOPE_API_KEY:
        raise RuntimeError("未配置阿里云百炼 API Key，请在设置中配置 DASHSCOPE_API_KEY")

    duration_ms = int(get_audio_duration(audio_path) * 1000)

    # Read audio and encode as base64 (DashScope supports base64 audio in JSON body)
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"{DASHSCOPE_API_BASE.rstrip('/')}/api/v1/services/audio/asr/transcription"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    params: dict = {}
    if language and language != "auto":
        params["language"] = language
    if hotwords:
        params["hotwords"] = hotwords

    payload = {
        "model": api_model,
        "input": {"audio": audio_b64},
    }
    if params:
        payload["parameters"] = params

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        if resp.status_code != 200:
            error_msg = resp.text[:500]
            try:
                err = resp.json()
                error_msg = err.get("message", err.get("code", resp.text[:200]))
            except Exception:
                pass
            raise RuntimeError(f"API调用失败 ({resp.status_code}): {error_msg}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API请求失败: {e}") from e

    result = resp.json()
    output = result.get("output", {})

    # Parse DashScope ASR response: handle transcripts with optional sentences
    segments: list = []
    transcripts = output.get("transcripts", [])
    if not transcripts and isinstance(output, dict):
        # Some models return flat output.text
        flat_text = output.get("text", "").strip()
        if flat_text:
            return [{"text": flat_text, "start_ms": 0, "end_ms": duration_ms}]

    for transcript in transcripts:
        text = transcript.get("text", "").strip()
        sentences = transcript.get("sentences", [])

        if sentences:
            for sent in sentences:
                sent_text = sent.get("text", "").strip()
                if sent_text:
                    segments.append({
                        "text": sent_text,
                        "start_ms": sent.get("begin_time", sent.get("start_ms", 0)),
                        "end_ms": sent.get("end_time", 0),
                    })
        elif text:
            segments.append({
                "text": text,
                "start_ms": 0,
                "end_ms": duration_ms,
            })

    return segments


def _api_transcribe_segment(audio_path: str, model_key: str, language: str, device: str, hotwords: str = "") -> str:
    """Transcribe a short audio segment via DashScope API. Returns plain text."""
    segs = _api_asr_pipeline(audio_path, model_key, language, device, hotwords)
    return " ".join(seg.get("text", "").strip() for seg in segs if seg.get("text", "").strip())


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
    language: str = "zh",
    device: str = "cuda",
    progress_callback=None,
    hotwords: str = "",
) -> dict:
    """Full ASR pipeline: extract audio → VAD → ASR per segment → generate SRT.

    Returns dict with keys: audio_path, srt_path, segments_count, model, model_name, duration_sec.

    Args:
        hotwords: Optional context string with proper nouns for Qwen3-ASR.
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

    framework = ASR_MODELS[model_key].get("framework", "funasr")
    if framework == "api":
        segments = _api_asr_pipeline(audio_path, model_key, language, device, hotwords)
    elif framework == "nemo":
        segments = _nemo_asr_pipeline(audio_path, model_key, language, device)
    elif framework == "transformers":
        segments = _transformers_asr_pipeline(audio_path, model_key, language, device)
    elif framework == "whisper":
        segments = _whisper_asr_pipeline(audio_path, model_key, language, device)
    elif framework == "firered":
        segments = _firered_asr_pipeline(audio_path, model_key, language, device)
    else:
        segments = vad_asr_pipeline(
            audio_path,
            model_key=model_key,
            language=language,
            device=device,
            progress_callback=progress_callback,
            hotwords=hotwords,
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
        "segments_count": len(segments) if segments else 0,
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
    language: str = "zh",
    device: str = "cuda",
    progress_callback=None,
    hotwords: str = "",
) -> dict:
    """Run ASR directly on an audio file (no extraction step).

    The output SRT is named after the audio file (<basename>.srt) and
    saved to ``output_dir``.

    Args:
        hotwords: Optional context string with proper nouns for Qwen3-ASR.

    Returns dict with keys: audio_name, srt_path, segments_count, model_name, duration_sec.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    ext = os.path.splitext(audio_path)[1].lower()
    t0 = time.time()

    # Convert unsupported formats (AAC, MP3, etc.) to temp PCM WAV
    work_path = audio_path
    temp_wav = None
    try:
        if ext in _SF_UNSUPPORTED:
            import uuid
            os.makedirs(TEMP_DIR, exist_ok=True)
            temp_wav = os.path.join(TEMP_DIR, f"asr_{uuid.uuid4().hex[:8]}.wav")
            if progress_callback:
                progress_callback("converting", f"转换音频: {ext} → wav")
            _convert_to_pcm_wav(audio_path, temp_wav)
            work_path = temp_wav

        if progress_callback:
            progress_callback("loading_models", "加载模型中...")

        framework = ASR_MODELS[model_key].get("framework", "funasr")
        if framework == "nemo":
            segments = _nemo_asr_pipeline(work_path, model_key, language, device)
        elif framework == "transformers":
            segments = _transformers_asr_pipeline(work_path, model_key, language, device)
        elif framework == "firered":
            segments = _firered_asr_pipeline(work_path, model_key, language, device)
        else:
            segments = vad_asr_pipeline(
                work_path,
                model_key=model_key,
                language=language,
                device=device,
                progress_callback=progress_callback,
                hotwords=hotwords,
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
            "segments_count": len(segments) if segments else 0,
            "model_name": model_label,
            "duration_sec": round(t1 - t0, 1),
        }
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


class _CancelPipeline(Exception):
    """Raised when a pipeline is cancelled mid-processing. Caught by the caller to mark the job as cancelled."""
    pass


# --- ASR Comparison ---

# ---------------------------------------------------------------------------
# Simplified → Traditional character mapping (common pairs used in ASR output).
# When either model outputs traditional characters, they map to simplified so
# the comparison is script-agnostic.
# ---------------------------------------------------------------------------
_S2T_MAP: dict[str, str] = {}
def _build_s2t_map():
    """Lazily build the simplified→traditional mapping on first use."""
    if _S2T_MAP:
        return _S2T_MAP
    # Each group: first character is the simplified form; ALL subsequent
    # characters are traditional / regional variants that map to it.
    # Focused on the ≈300 most frequent characters found in ASR output.
    groups = """
        说説 来來 国國 时時 对對 会會 过過 个個 们們
        为為爲 学學 开開 关關 门門 车車 长長 见見 贝貝
        页頁 风風 飞飛 马馬 鱼魚 鸟鳥 龙龍 东東 爱愛
        笔筆 变變 边邊 宾賓 仓倉 产產 尝嘗 厂廠 处处處
        从從 达達 带帶 当當噹 党黨 导導 灯燈 敌敵 点點
        电電 动動 独獨 断斷 队隊 尔爾 范範 丰豐 妇婦
        刚剛 纲綱 给給 广廣 规規 汉漢 号號 后後 华華
        欢歡 还還 击擊 极極 几幾 计計 济濟 价價 坚堅
        间間 检檢 简簡 剑劍 节節 紧緊 进進 惊驚 旧舊
        举舉 剧劇 据據 决決 军軍 蓝藍 乐樂 离離 礼禮
        连連 练練 两兩 疗療 邻鄰 灵靈 领領 刘劉 录錄
        绿綠 论論 妈媽 买買 卖賣 满满滿 没沒 梦夢 难難
        脑腦 宁寧 农農 暖暖暖 盘盤 钱錢 强強 轻轻輕 热熱
        认認 伤傷 声声聲 师師 实實 识識 势勢 试試 书書
        术術 树樹 双雙 岁歲 孙孫 体體 条條 铁鐵 听聽
        头頭 图圖 万万萬 网網 问問 无無 线線 乡鄉 写寫
        谢謝 兴興 选選 压壓 严嚴 颜顏 业業 医醫 艺藝
        阴陰 应應 拥擁 优優 邮郵 圆圆圓 运運 杂雜 战戰
        张張 阵陣 证證 织織 职職 质質 众眾 转轉 装裝
        资資 总總 组組 讲講 误誤 员員 显顯 调調 议議
        谈談 读讀 诗詩 词詞 语語 课課 谁誰 让讓 记記
        话話 请請 胜勝 卫衛 洁潔 显顯 响響 预預 页頁
        发髮發 回迴 汇匯彙 尽儘盡 历歷曆 台臺颱檯
        复復複 团團糰 脏脏髒臟 云雲 制製 面麵麪
        里裡裏 准準 后後 只隻 征徵 系係繫 钟鐘鍾
    """.split()
    for group in groups:
        s = group[0]  # simplified / canonical form
        for t in group[1:]:  # traditional variants
            _S2T_MAP[t] = s
    # Edge cases not covered by groups above
    _S2T_MAP["喫"] = "吃"
    _S2T_MAP["鎗"] = "枪"
    _S2T_MAP["兇"] = "凶"
    _S2T_MAP["採"] = "采"
    _S2T_MAP["綵"] = "彩"
    _S2T_MAP["瀋"] = "沈"
    _S2T_MAP["誌"] = "志"
    _S2T_MAP["慾"] = "欲"
    return _S2T_MAP


# ---------------------------------------------------------------------------
# CJK character variant mapping — normalize equivalent glyphs to a single form
# ---------------------------------------------------------------------------
_CJK_VARIANT_MAP: dict[str, str] = {}
def _build_cjk_variant_map():
    """Lazily build the CJK variant mapping."""
    if _CJK_VARIANT_MAP:
        return _CJK_VARIANT_MAP
    variants = {
        # Common variants (same character, different encoding/region)
        "爲": "為", "峯": "峰", "羣": "群", "峽": "峡",
        "麪": "面", "祕": "秘", "薑": "姜", "禦": "御",
        "綫": "线", "跡": "迹", "蹟": "迹",
        # Japanese kanji → common form
        "弐": "二", "壱": "一", "弌": "一", "拾": "十",
    }
    _CJK_VARIANT_MAP.update(variants)
    return _CJK_VARIANT_MAP


# MFA-style punctuation symbols (same as mfa/postprocess_textgrids.py)
_MFA_PUNCT_SYMBOLS = set("〜～")  # wave dash, fullwidth tilde


def _normalize_text_mfa(text: str) -> str:
    """Normalize text the same way MFA does — strip punctuation only.

    Matches the approach in mfa/postprocess_textgrids.py:
    remove Unicode punctuation (category P) + wave dash / fullwidth tilde.
    No NFKC, no kana shift, no homophone mapping.
    """
    import unicodedata
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or ch in _MFA_PUNCT_SYMBOLS:
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison — deep normalization.

    Steps applied in order:
      1. NFKC — fullwidth/halfwidth, ligatures, compatibility chars
      2. Simplified-Traditional mapping — unify script variants (说/説 → 说)
      3. CJK variant characters (爲/為 → 為, 峯/峰 → 峰)
      4. Japanese katakana → hiragana (カタカナ → かたかな)
      5. Third-person pronouns (他/她/它/祂/牠 → 他)
      6. Common ASR homophone pairs (的/得/地 → 的, 在/再 → 在, 做/作 → 做)
      7. Remove spaces between CJK characters (今天 天气 → 今天天气)
      8. Strip punctuation — keep only letters, numbers, and spaces
      9. Lowercase — English case-insensitivity
     10. Collapse whitespace
    """
    import unicodedata
    import re

    # 1. NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Simplified-Traditional mapping
    s2t = _build_s2t_map()
    text = "".join(s2t.get(ch, ch) for ch in text)

    # 3. CJK variant characters
    cjk_var = _build_cjk_variant_map()
    text = "".join(cjk_var.get(ch, ch) for ch in text)

    # 4. Japanese katakana → hiragana (Unicode block shift: 0x60)
    #    Katakana U+30A1–U+30F6  →  Hiragana U+3041–U+3096
    #    Katakana U+30F7–U+30FA (small ku/shi/su/to) handled separately
    def _kana_shift(ch):
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            return chr(cp - 0x60)
        # Small katakana (ㇰㇱㇲㇳ etc.) — less common, map approximately
        if 0x30F7 <= cp <= 0x30FA:
            return chr(0x3041 + (cp - 0x30F7))  # approximate mapping
        # Katakana punctuation: ・→ (remove), ー→ (keep as is for now)
        if cp == 0x30FB:  # ・ katakana middle dot
            return " "
        if cp == 0x30FC:  # ー long vowel mark
            return ""  # remove long vowel mark for comparison
        return ch
    text = "".join(_kana_shift(ch) for ch in text)

    # 5. Third-person pronouns
    text = re.sub(r'[他她它祂牠]', '他', text)

    # 6. Common ASR homophone pairs (semantically equivalent in most contexts)
    text = re.sub(r'[得地]', '的', text)       # 的/得/地 → 的
    text = re.sub(r'再', '在', text)            # 在/再 → 在
    text = re.sub(r'作', '做', text)            # 做/作 → 做
    text = re.sub(r'[吗嘛]', '么', text)         # 吗/嘛/么 → 么 (sentence-final particles)

    # 7. Remove spaces between CJK characters (but preserve word boundaries
    #    between CJK and Latin scripts).
    #    CJK: 一-鿿 (unified), 㐀-䶿 (ext-A),
    #         豈-﫿 (compat), ぀-ゟ (hiragana),
    #         ゠-ヿ (katakana — already shifted to hiragana above),
    #         가-힯 (hangul)
    _CJK = re.compile(
        r'([一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ가-힯])'
        r'\s+'
        r'(?=[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ가-힯])'
    )
    text = _CJK.sub(r'\1', text)

    # 8. Strip punctuation — keep only letters, numbers, and spaces
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N") or cat == "Zs":
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    text = "".join(cleaned)

    # 9 & 10. Lowercase and collapse whitespace
    return " ".join(text.split()).lower()


def _parse_srt_timestamp(ts: str) -> int:
    """Parse SRT timestamp HH:MM:SS,mmm to milliseconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s_parts = parts[2].split(".")
    s = int(s_parts[0])
    ms = int(s_parts[1]) if len(s_parts) > 1 else 0
    return h * 3600000 + m * 60000 + s * 1000 + ms


def parse_srt_to_segments(srt_path: str) -> list:
    """Parse an SRT file into a list of segment dicts.

    Each segment: {index, start_ms, end_ms, text, normalized_text}
    """
    segments = []
    if not os.path.exists(srt_path):
        return segments

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Line 0: index number
        idx_line = lines[0].strip()
        if not idx_line.isdigit():
            continue
        idx = int(idx_line)
        # Line 1: timestamp "00:00:00,000 --> 00:00:02,500"
        ts_line = lines[1].strip()
        if "-->" not in ts_line:
            continue
        parts = ts_line.split("-->")
        start_ms = _parse_srt_timestamp(parts[0])
        end_ms = _parse_srt_timestamp(parts[1])
        # Remaining lines: text (may span multiple lines)
        text = " ".join(line.strip() for line in lines[2:] if line.strip())
        if not text:
            continue
        segments.append({
            "index": idx,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "normalized_text": _normalize_text_mfa(text),
        })

    return segments


def _time_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return overlap duration in ms between two time ranges."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def align_segments_by_time(segs_a: list, segs_b: list, min_overlap_ratio: float = 0.3) -> dict:
    """Align segments from two SRTs by time overlap.

    Returns:
        aligned_pairs: list of (seg_a, seg_b, overlap_ms) for matched pairs
        unmatched_a: list of seg_a with no match
        unmatched_b: list of seg_b with no match
    """
    aligned_pairs = []
    used_b = set()

    for seg_a in segs_a:
        best_b = None
        best_overlap = 0
        for j, seg_b in enumerate(segs_b):
            if j in used_b:
                continue
            overlap = _time_overlap(
                seg_a["start_ms"], seg_a["end_ms"],
                seg_b["start_ms"], seg_b["end_ms"]
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_b = (j, seg_b)

        if best_b and best_overlap > 0:
            seg_b_dur = best_b[1]["end_ms"] - best_b[1]["start_ms"]
            seg_a_dur = seg_a["end_ms"] - seg_a["start_ms"]
            min_dur = min(seg_a_dur, seg_b_dur) if seg_b_dur > 0 else 0
            overlap_ratio = best_overlap / min_dur if min_dur > 0 else 0
            if overlap_ratio >= min_overlap_ratio:
                used_b.add(best_b[0])
                aligned_pairs.append((seg_a, best_b[1], best_overlap))

    unmatched_a = [s for s in segs_a if not any(s is pair[0] for pair in aligned_pairs)]
    unmatched_b = [s for j, s in enumerate(segs_b) if j not in used_b]

    return {
        "aligned_pairs": aligned_pairs,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
    }


def srt_to_plain_text(srt_path: str) -> str:
    """Extract normalized plain text from an SRT file for comparison."""
    segments = parse_srt_to_segments(srt_path)
    if not segments:
        return ""
    return " ".join(seg["normalized_text"] for seg in segments)


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


_ENGLISH_RE = re.compile(r"[a-zA-Z]{3,}")  # 3+ consecutive Latin letters

def _has_english(text: str) -> bool:
    """Check if text contains English words (3+ consecutive ASCII letters)."""
    return bool(_ENGLISH_RE.search(text))


def compare_srt_texts(srt_path1: str, srt_path2: str) -> float:
    """Compare two SRT files and return difference percentage (0-100).

    Returns 0.0 for identical text, 100.0 for completely different text.
    If either output contains English text (likely hallucination),
    returns 100.0 diff regardless of content.
    """
    text1 = srt_to_plain_text(srt_path1)
    text2 = srt_to_plain_text(srt_path2)

    if not text1 and not text2:
        return 0.0
    if not text1 or not text2:
        return 100.0

    # English in either output = zero match (model hallucination)
    if _has_english(text1) or _has_english(text2):
        return 100.0

    dist = _levenshtein(text1, text2)
    max_len = max(len(text1), len(text2))
    return round((dist / max_len) * 100, 1)


def _compute_diff_chunks(text_a: str, text_b: str) -> list:
    """Compute character-level diff between two texts using LCS backtracking.

    Returns a list of chunk dicts:
      {type: "equal"|"diff", text_a: str, text_b: str}
    Groups consecutive same-type operations for readability.
    """
    a = text_a or ""
    b = text_b or ""
    if not a and not b:
        return []
    if not a:
        return [{"type": "diff", "text_a": "", "text_b": b}]
    if not b:
        return [{"type": "diff", "text_a": a, "text_b": ""}]

    m, n = len(a), len(b)
    # Build LCS DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    # Backtrack to produce diff ops
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append(("equal", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            ops.append(("diff", "", b[j - 1]))
            j -= 1
        else:
            ops.append(("diff", a[i - 1], ""))
            i -= 1
    ops.reverse()

    # Group consecutive same-type ops into chunks
    chunks = []
    for op, ca, cb in ops:
        if not chunks or chunks[-1]["type"] != op:
            chunks.append({"type": op, "text_a": ca, "text_b": cb})
        else:
            chunks[-1]["text_a"] += ca
            chunks[-1]["text_b"] += cb

    return chunks


def compare_srt_sentences(srt_path1: str, srt_path2: str) -> dict:
    """Compare two SRT files sentence-by-sentence with time-based alignment.

    Returns a dict with match_rate, overall diff and per-sentence breakdown
    including character-level diff_chunks for highlighting.
    """
    segs_a = parse_srt_to_segments(srt_path1)
    segs_b = parse_srt_to_segments(srt_path2)

    if not segs_a and not segs_b:
        return {
            "overall_diff_percent": 0.0,
            "match_rate": 100.0,
            "sentence_results": [],
            "matched_count": 0,
            "unmatched_a": 0,
            "unmatched_b": 0,
        }
    if not segs_a or not segs_b:
        return {
            "overall_diff_percent": 100.0,
            "match_rate": 0.0,
            "sentence_results": [],
            "matched_count": 0,
            "unmatched_a": len(segs_a),
            "unmatched_b": len(segs_b),
        }

    alignment = align_segments_by_time(segs_a, segs_b)
    sentence_results = []
    total_weight = 0
    weighted_diff_sum = 0.0

    # Process aligned pairs
    for seg_a, seg_b, overlap_ms in alignment["aligned_pairs"]:
        start_ms = max(seg_a["start_ms"], seg_b["start_ms"])
        end_ms = min(seg_a["end_ms"], seg_b["end_ms"])

        t_a = seg_a["normalized_text"]
        t_b = seg_b["normalized_text"]
        if not t_a and not t_b:
            diff = 0.0
        elif not t_a or not t_b:
            diff = 100.0
        else:
            dist = _levenshtein(t_a, t_b)
            max_len = max(len(t_a), len(t_b))
            diff = round((dist / max_len) * 100, 1)

        weight = max(len(t_a), len(t_b))
        weighted_diff_sum += diff * weight
        total_weight += weight

        # Compute character diff for this sentence pair
        diff_chunks = _compute_diff_chunks(seg_a["text"], seg_b["text"])

        sentence_results.append({
            "idx_a": seg_a["index"],
            "idx_b": seg_b["index"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text_a": seg_a["text"],
            "text_b": seg_b["text"],
            "diff_percent": diff,
            "match_rate": round(100.0 - diff, 1),
            "flagged": diff > 10.0,
            "diff_chunks": diff_chunks,
        })

    # Process unmatched segments from A
    for seg_a in alignment["unmatched_a"]:
        weight = len(seg_a["normalized_text"])
        weighted_diff_sum += 100.0 * weight
        total_weight += weight
        sentence_results.append({
            "idx_a": seg_a["index"],
            "idx_b": None,
            "start_ms": seg_a["start_ms"],
            "end_ms": seg_a["end_ms"],
            "text_a": seg_a["text"],
            "text_b": "",
            "diff_percent": 100.0,
            "match_rate": 0.0,
            "flagged": True,
            "diff_chunks": [{"type": "diff", "text_a": seg_a["text"], "text_b": ""}],
        })

    # Process unmatched segments from B
    for seg_b in alignment["unmatched_b"]:
        weight = len(seg_b["normalized_text"])
        weighted_diff_sum += 100.0 * weight
        total_weight += weight
        sentence_results.append({
            "idx_a": None,
            "idx_b": seg_b["index"],
            "start_ms": seg_b["start_ms"],
            "end_ms": seg_b["end_ms"],
            "text_a": "",
            "text_b": seg_b["text"],
            "diff_percent": 100.0,
            "match_rate": 0.0,
            "flagged": True,
            "diff_chunks": [{"type": "diff", "text_a": "", "text_b": seg_b["text"]}],
        })

    overall_diff = round(weighted_diff_sum / total_weight, 1) if total_weight > 0 else 0.0

    # Sort results by time
    sentence_results.sort(key=lambda r: r["start_ms"])

    return {
        "overall_diff_percent": overall_diff,
        "match_rate": round(100.0 - overall_diff, 1),
        "sentence_results": sentence_results,
        "matched_count": len(alignment["aligned_pairs"]),
        "unmatched_a": len(alignment["unmatched_a"]),
        "unmatched_b": len(alignment["unmatched_b"]),
    }


def compare_text_files(file_path_a: str, file_path_b: str, filter_english: bool = True) -> dict:
    """Compare two text/SRT files and return comparison results.

    For .srt files: delegates to compare_srt_sentences() for time-aligned,
    sentence-level comparison with character diff chunks.
    For .txt/.ass/.ssa/.vtt files: splits into lines, compares line-by-line
    using Levenshtein distance and LCS diff chunks.

    If filter_english is True and either file contains English text (3+
    consecutive ASCII letters), forces match_rate to 0%.
    """
    ext_a = os.path.splitext(file_path_a)[1].lower()
    ext_b = os.path.splitext(file_path_b)[1].lower()

    # Helper to read file content
    def _read_all(path):
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # English filter check
    if filter_english:
        content_a = _read_all(file_path_a)
        content_b = _read_all(file_path_b)
        if _has_english(content_a) or _has_english(content_b):
            return {
                "overall_diff_percent": 100.0,
                "match_rate": 0.0,
                "flagged": True,
                "sentence_results": [],
                "matched_count": 0,
                "unmatched_a": 0,
                "unmatched_b": 0,
                "file_a": file_path_a,
                "file_b": file_path_b,
            }

    if ext_a == ".srt" and ext_b == ".srt":
        result = compare_srt_sentences(file_path_a, file_path_b)
        result["file_a"] = file_path_a
        result["file_b"] = file_path_b
        result["flagged"] = result.get("overall_diff_percent", 0) > 10.0
        return result

    # Plain text / other formats: read and compare line-by-line
    def _read_lines(path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n\r") for line in f]

    lines_a = _read_lines(file_path_a)
    lines_b = _read_lines(file_path_b)

    full_a = "\n".join(lines_a)
    full_b = "\n".join(lines_b)

    if not full_a and not full_b:
        overall_diff = 0.0
    elif not full_a or not full_b:
        overall_diff = 100.0
    else:
        dist = _levenshtein(full_a, full_b)
        max_len = max(len(full_a), len(full_b))
        overall_diff = round((dist / max_len) * 100, 1)

    sentence_results = []
    max_lines = max(len(lines_a), len(lines_b))
    total_weight = 0
    weighted_diff_sum = 0.0

    for i in range(max_lines):
        ta = lines_a[i] if i < len(lines_a) else ""
        tb = lines_b[i] if i < len(lines_b) else ""
        if not ta and not tb:
            diff = 0.0
        elif not ta or not tb:
            diff = 100.0
        else:
            dist = _levenshtein(ta, tb)
            max_l = max(len(ta), len(tb))
            diff = round((dist / max_l) * 100, 1) if max_l > 0 else 0.0

        weight = max(len(ta), len(tb))
        weighted_diff_sum += diff * weight
        total_weight += weight

        diff_chunks = _compute_diff_chunks(ta, tb)
        sentence_results.append({
            "idx_a": i + 1 if i < len(lines_a) and ta else None,
            "idx_b": i + 1 if i < len(lines_b) and tb else None,
            "start_ms": 0,
            "end_ms": 0,
            "text_a": ta,
            "text_b": tb,
            "diff_percent": diff,
            "match_rate": round(100.0 - diff, 1),
            "flagged": diff > 10.0,
            "diff_chunks": diff_chunks,
        })

    matched_count = min(len(lines_a), len(lines_b))
    return {
        "overall_diff_percent": overall_diff,
        "match_rate": round(100.0 - overall_diff, 1),
        "flagged": overall_diff > 10.0,
        "sentence_results": sentence_results,
        "matched_count": matched_count,
        "unmatched_a": max(0, len(lines_a) - len(lines_b)),
        "unmatched_b": max(0, len(lines_b) - len(lines_a)),
        "file_a": file_path_a,
        "file_b": file_path_b,
    }


def compare_asr_pipeline(
    audio_path: str,
    language: str = "ja",
    device: str = "cuda",
    progress_callback=None,
    source_dir: str = "",
    model_a: str = "qwen3-asr",
    model_b: str = "cohere-transcribe",
    hotwords: str = "",
    cancel_check=None,  # callable → bool; if True, abort processing
    filter_english: bool = True,
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
        progress_callback("loading_models", 5)

    # Preload both ASR models BEFORE any processing.
    # This is critical when running in background/daemon threads (e.g. the server's
    # comparison endpoints), because FunASR / ModelScope model downloads can hang
    # indefinitely when triggered from daemon threads.
    # Preloading here also gives the user immediate feedback that models are loading
    # rather than an unexplained hang after VAD completes.
    for model_key in compare_models:
        if cancel_check and cancel_check():
            raise _CancelPipeline("Cancelled during model loading")
        model_info = ASR_MODELS.get(model_key)
        if model_info is None:
            raise ValueError(f"Unknown ASR model: {model_key}")
        if model_info.get("framework") == "api":
            continue  # API models don't need local preloading
        try:
            _get_asr_model(model_key, device=device, use_compile=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ASR model '{model_key}' ({model_info['name']}): {e}"
            ) from e

    if cancel_check and cancel_check():
        raise _CancelPipeline("Cancelled before warm-up")

    # Warm-up: run a short dummy inference on each model to fully initialize
    # GPU contexts before spawning threads. Critical for firered-asr2 which
    # must initialize CUDA kernels in the main thread's context.
    import tempfile as _tempfile2
    import numpy as _np2
    _dummy2 = (_np2.sin(2 * _np2.pi * _np2.linspace(300, 3000, 32000) / 16000
                       * _np2.arange(32000)) * 0.05).astype(_np2.float32)
    for mk in compare_models:
        if ASR_MODELS.get(mk, {}).get("framework") == "api":
            continue  # API models don't need GPU warm-up
        try:
            with _tempfile2.NamedTemporaryFile(suffix=".wav", delete=False) as _tf2:
                sf.write(_tf2.name, _dummy2, 16000, subtype="PCM_16")
                _transcribe_segment(_tf2.name, mk, language, device, "")
            os.unlink(_tf2.name)
        except Exception:
            pass

    if progress_callback:
        progress_callback("vad", 8)

    if cancel_check and cancel_check():
        raise _CancelPipeline("Cancelled before ASR inference")

    model_results = {}
    srt_paths = {}

    # Run both ASR models in parallel — each model processes the same audio
    # independently. On GPUs with sufficient VRAM (e.g. RTX 4090 24GB), both
    # models can reside in memory and run concurrently, nearly halving the
    # total comparison time.
    _cuda_dev = 0
    if device != "cpu" and torch.cuda.is_available():
        try:
            _cuda_dev = torch.cuda.current_device()
        except Exception:
            _cuda_dev = 0

    def _run_one_model(model_key):
        """Run a single ASR model on the audio file. Returns (model_key, segments)."""
        # Ensure this thread has the correct CUDA device context.
        # CUDA contexts are thread-local; without this, GPU tensors loaded
        # in the main thread may be inaccessible from thread-pool workers.
        if device != "cpu" and torch.cuda.is_available():
            try:
                torch.cuda.set_device(_cuda_dev)
            except Exception:
                pass

        model_info = ASR_MODELS[model_key]
        framework = model_info.get("framework", "funasr")

        if framework == "api":
            segs = _api_asr_pipeline(audio_path, model_key=model_key,
                                     language=language, device=device, hotwords=hotwords)
        elif framework == "nemo":
            segs = _nemo_asr_pipeline(audio_path, model_key=model_key,
                                       language=language, device=device)
        elif framework == "transformers":
            segs = _transformers_asr_pipeline(audio_path, model_key=model_key,
                                               language=language, device=device)
        elif framework == "whisper":
            segs = _whisper_asr_pipeline(audio_path, model_key=model_key,
                                          language=language, device=device)
        elif framework == "firered":
            segs = _firered_asr_pipeline(audio_path, model_key=model_key,
                                          language=language, device=device)
        else:
            segs = vad_asr_pipeline(audio_path, model_key=model_key,
                                     language=language, device=device,
                                     progress_callback=None, hotwords=hotwords)
        return model_key, segs

    # Execute both models concurrently via thread pool.
    # Use a timeout loop on wait() so we can check cancellation every second
    # while the ASR models are running (which may take 10-60+ seconds each).
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _executor:
        _futures = {
            _executor.submit(_run_one_model, mk): mk
            for mk in compare_models
        }
        _pending = set(_futures.keys())
        while _pending:
            if cancel_check and cancel_check():
                # Don't cancel futures that are already running (Python can't
                # kill threads), but stop processing results and raise.
                raise _CancelPipeline("Cancelled during ASR inference")
            _done, _pending = concurrent.futures.wait(
                _pending, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for _future in _done:
                mk, segments = _future.result()
                model_info = ASR_MODELS[mk]
                abbr = model_info.get("abbr", mk)

                if progress_callback:
                    progress_callback(f"asr_{mk}", 10 + len(model_results) * 40)

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

                # Save plain-text TXT alongside SRT (no timestamps).
                # Naming: {base_name}_{abbr}.txt (e.g., xxx_qwen3-asr.txt)
                txt_path = os.path.join(srt_out_dir, f"{base_name}_{abbr}.txt")
                plain_text = "\n".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()) if segments else ""
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(plain_text)

                model_results[mk] = {
                    "model_key": mk,
                    "model_name": model_info["name"],
                    "abbr": abbr,
                    "segments_count": len(segments) if segments else 0,
                    "srt_path": srt_path,
                    "txt_path": txt_path,
                }
                srt_paths[mk] = srt_path

    if progress_callback:
        progress_callback("comparing", 90)

    # Check for empty results (0 segments = 0 KB SRT) from either model
    empty_models = []
    for model_key, mr in model_results.items():
        if mr["segments_count"] == 0:
            empty_models.append(mr["model_name"])

    # If filter enabled and either output contains English, match rate = 0%
    if filter_english:
        eng_a = _has_english(srt_to_plain_text(srt_paths[compare_models[0]]))
        eng_b = _has_english(srt_to_plain_text(srt_paths[compare_models[1]]))
        eng_flag = eng_a or eng_b
    else:
        eng_flag = False
    if eng_flag:
        diff_percent = 100.0
        match_rate = 0.0
        sentence_results = []
        matched_count = 0
        unmatched_a = 0
        unmatched_b = 0
        flagged = True
    else:
        # Sentence-level comparison
        sentence_cmp = compare_srt_sentences(
            srt_paths[compare_models[0]], srt_paths[compare_models[1]]
        )
        diff_percent = sentence_cmp["overall_diff_percent"]
        match_rate = sentence_cmp["match_rate"]
        sentence_results = sentence_cmp["sentence_results"]
        matched_count = sentence_cmp["matched_count"]
        unmatched_a = sentence_cmp["unmatched_a"]
        unmatched_b = sentence_cmp["unmatched_b"]
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
        "match_rate": match_rate,
        "flagged": flagged,
        "empty_models": empty_models,
        "sentence_results": sentence_results,
        "matched_count": matched_count,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
    }


def _chunk_vad_segments(vad_segments: list, min_s: float = 9.0, max_s: float = 16.0) -> list:
    """Split VAD segments into chunks of roughly min_s–max_s seconds.

    Long segments are split evenly; short adjacent segments are merged when
    the combined duration stays under max_s.
    Returns [(start_ms, end_ms), ...] in chronological order.
    """
    if not vad_segments:
        return []

    MIN_MS = int(min_s * 1000)
    MAX_MS = int(max_s * 1000)

    # Merged pass: combine very short adjacent segments
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


def _transcribe_segment(audio_path: str, model_key: str, language: str, device: str, hotwords: str = "") -> str:
    """Transcribe a short audio file with a single ASR model. Returns plain text."""
    model_info = ASR_MODELS[model_key]
    framework = model_info.get("framework", "funasr")

    if framework == "api":
        return _api_transcribe_segment(audio_path, model_key, language, device, hotwords)

    if model_key in ("qwen3-asr",):
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        model = _get_asr_model(model_key, device=device)
        lang = _QWEN3_LANG_MAP.get(language) if language and language != "auto" else None
        results = model.transcribe([(audio, sr)], language=lang, context=hotwords)
        return " ".join((r.text or "").strip() for r in results if (r.text or "").strip())

    elif framework == "firered":
        # FireRedASR2S requires exactly 16kHz mono int16 WAV — resample if needed
        work_path = audio_path
        try:
            info = sf.info(audio_path)
            if info.samplerate != 16000 or info.channels != 1:
                os.makedirs(TEMP_DIR, exist_ok=True)
                import uuid as _uuid
                temp_wav = os.path.join(TEMP_DIR, f"firered_seg_{_uuid.uuid4().hex[:8]}.wav")
                _convert_to_pcm_wav(audio_path, temp_wav)
                work_path = temp_wav
        except Exception:
            work_path = audio_path  # best-effort: try original path

        model = _get_asr_model(model_key, device=device)
        result = model.process(work_path)
        segs = result.get("sentences", [])
        if not segs and result.get("text"):
            return result["text"].strip()
        text = " ".join(s.get("text", "").strip() for s in segs if s.get("text", "").strip())
        # Detect silent firered failures — the internal ASR model catches all
        # exceptions and returns empty text (see FireRedAsr2.transcribe()).
        # When this happens, log details so the caller can retry or alert.
        if not text:
            import logging as _logging
            _logging.getLogger("asr_pipeline").warning(
                "firered-asr2 returned empty for %s (VAD segs=%d, dur=%.1fs)",
                os.path.basename(audio_path),
                len(result.get("vad_segments_ms", [])),
                result.get("dur_s", 0),
            )
        return text

    elif framework in ("transformers", "whisper"):
        import torch
        from transformers.audio_utils import load_audio
        model, processor = _get_asr_model(model_key, device=device)
        audio = load_audio(audio_path, sampling_rate=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=language)
        inputs = {k: v.to(model.device, dtype=model.dtype) for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": 256}
        # Whisper language handling
        if framework == "whisper" and language and language != "auto":
            try:
                gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
            except Exception:
                pass
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        return processor.decode(outputs[0], skip_special_tokens=True).strip()

    else:
        # funasr / default: use AutoModel
        model = _get_asr_model(model_key, device=device)
        try:
            results = model.generate(input=audio_path)
        except Exception:
            # Fallback: load audio and transcribe directly
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            results = model.generate(input=audio)
        if isinstance(results, list) and results:
            texts = []
            for r in results:
                t = (r.get("text", "") if isinstance(r, dict) else (r.text if hasattr(r, "text") else str(r))).strip()
                if t:
                    texts.append(t)
            return " ".join(texts)
        return ""


def segment_and_compare_pipeline(
    audio_path: str,
    language: str = "zh",
    device: str = "cuda",
    model_a: str = "qwen3-asr",
    model_b: str = "cohere-transcribe",
    hotwords: str = "",
    segment_min_s: float = 9.0,
    segment_max_s: float = 16.0,
    progress_callback=None,
    source_dir: str = "",
    cancel_check=None,  # callable → bool; if True, abort processing
    filter_english: bool = True,
    on_segment=None,  # callable(seg_index, total, partial_result) called after each segment
) -> dict:
    """Split audio into 10-15s segments via VAD, run two ASR models on each, compare.

    1. Run VAD to find speech regions
    2. Chunk speech into segment_min_s–segment_max_s pieces
    3. Save each chunk as WAV in ASR_COMPARE_SEGMENTS_DIR/{audio_name}/
    4. Run model_a and model_b on each segment
    5. Compare normalized texts, compute match_rate
    6. Return per-segment results

    Returns dict with keys: audio_path, audio_name, source_dir, duration_sec,
    segment_count, segments (list of per-segment dicts), model_a, model_b.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    duration = get_audio_duration(audio_path)
    duration_ms = int(duration * 1000)

    # Output dir for segment WAVs
    seg_out_dir = os.path.join(ASR_COMPARE_SEGMENTS_DIR, base_name)
    os.makedirs(seg_out_dir, exist_ok=True)

    if progress_callback:
        progress_callback("loading_models", 2)

    # Preload both ASR models BEFORE VAD and processing.
    # Critical when running in background/daemon threads — model downloads
    # from FunASR/ModelScope can hang indefinitely in daemon threads.
    # Preloading gives immediate feedback that models are loading.
    compare_models = [model_a, model_b]
    for mk in compare_models:
        if cancel_check and cancel_check():
            raise _CancelPipeline("Cancelled during model loading")
        mi = ASR_MODELS.get(mk)
        if mi is None:
            raise ValueError(f"Unknown ASR model: {mk}")
        if mi.get("framework") == "api":
            continue  # API models don't need local preloading
        try:
            _get_asr_model(mk, device=device, use_compile=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ASR model '{mk}' ({mi['name']}): {e}"
            ) from e

    if cancel_check and cancel_check():
        raise _CancelPipeline("Cancelled before warm-up")

    # Warm-up: run a short dummy inference on each model to fully initialize
    # GPU contexts and internal components (critical for firered-asr2 which
    # lazy-loads VAD/LID/ASR/Punc sub-models on first process() call).
    # Without this, the first parallel segment may fail silently due to CUDA
    # context conflicts when both models initialize GPU state simultaneously.
    #
    # IMPORTANT: We use a short sine-sweep (NOT all-zeros silence) because
    # firered's process() runs VAD first — silence produces 0 VAD segments,
    # so the ASR/LID/Punc sub-models are NEVER exercised during warm-up.
    # A 2-second 300→3000 Hz sweep reliably triggers VAD speech detection,
    # ensuring the FULL pipeline (VAD→ASR→LID→Punc) initializes GPU state.
    import tempfile as _tempfile
    import numpy as _np
    _dummy_len = 32000  # 2 seconds @ 16kHz
    _dummy = (_np.sin(2 * _np.pi * _np.linspace(300, 3000, _dummy_len) / 16000
                      * _np.arange(_dummy_len)) * 0.05).astype(_np.float32)
    for mk in compare_models:
        if ASR_MODELS.get(mk, {}).get("framework") == "api":
            continue  # API models don't need GPU warm-up
        try:
            with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf:
                sf.write(_tf.name, _dummy, 16000, subtype="PCM_16")
                _transcribe_segment(_tf.name, mk, language, device, "")
            os.unlink(_tf.name)
        except Exception:
            pass  # warm-up failure is non-fatal

    # For firered-asr2 specifically, verify the warm-up actually exercised the
    # ASR model. If VAD didn't detect the sweep (unlikely but possible), force
    # a direct ASR transcribe call to ensure the model is fully initialized.
    for mk in compare_models:
        mi = ASR_MODELS.get(mk, {})
        if mi.get("framework") != "firered":
            continue
        try:
            _fmodel = _get_asr_model(mk, device=device)
            # Direct ASR warm-up: pass a 1s dummy tensor through the ASR model
            # to compile CUDA kernels and initialize cuDNN handles in THIS thread.
            _dummy_asr = _np.random.randn(16000).astype(_np.float32) * 0.001
            _fmodel.asr.transcribe(["warmup_asr"], [(16000, _dummy_asr)])
        except Exception:
            pass  # best-effort; segment processing will retry if needed

    if progress_callback:
        progress_callback("vad", 5)

    # Step 1: VAD
    vad_segments = run_vad(audio_path, device=device)
    if not vad_segments:
        # Fallback: treat whole audio as one segment (only if short enough)
        if duration <= segment_max_s * 2:
            vad_segments = [(0, duration_ms)]
        else:
            # Split whole audio into fixed chunks
            chunk_ms = int((segment_min_s + segment_max_s) / 2 * 1000)
            vad_segments = [(i * chunk_ms, min((i + 1) * chunk_ms, duration_ms))
                            for i in range((duration_ms + chunk_ms - 1) // chunk_ms)]

    # Step 2: Chunk into 10-15s pieces
    chunks = _chunk_vad_segments(vad_segments, segment_min_s, segment_max_s)
    if not chunks:
        chunks = [(0, duration_ms)]

    # Step 3: Load full audio, extract and save each segment
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    segment_files = []
    for i, (start_ms, end_ms) in enumerate(chunks):
        start_samp = max(0, int(start_ms * sr / 1000))
        end_samp = min(len(audio), int(end_ms * sr / 1000))
        if end_samp <= start_samp:
            continue
        chunk = audio[start_samp:end_samp]
        seg_name = f"{base_name}_seg{i + 1:03d}.wav"
        seg_path = os.path.join(seg_out_dir, seg_name)
        sf.write(seg_path, chunk, sr, subtype="PCM_16")
        segment_files.append({
            "index": i + 1,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_s": round((end_ms - start_ms) / 1000, 1),
            "wav_path": seg_path,
            "name": seg_name,
        })

    total = len(segment_files)
    if progress_callback:
        progress_callback("segments_ready", 10)

    # Step 4 & 5: Run both models on each segment and compare
    model_a_info = ASR_MODELS[model_a]
    model_b_info = ASR_MODELS[model_b]
    segment_results = []

    for idx, seg in enumerate(segment_files):
        if cancel_check and cancel_check():
            raise _CancelPipeline("Cancelled during segment processing")
        if progress_callback:
            progress_callback(f"processing_{idx + 1}", 10 + int((idx / total) * 85))

        # Run both models on the SAME segment in parallel
        text_a = ""
        text_b = ""
        error_a = None
        error_b = None

        # Determine CUDA device index for per-thread context initialization.
        # CUDA contexts are thread-local; each thread must set its own device
        # to ensure GPU tensors are accessible. Without this, firered-asr2 can
        # silently fail when its first ASR forward pass runs in a sub-thread.
        _cuda_idx = 0
        if device != "cpu" and torch.cuda.is_available():
            try:
                _cuda_idx = torch.cuda.current_device()
            except Exception:
                _cuda_idx = 0

        def _transcribe_a():
            nonlocal text_a, error_a
            try:
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.set_device(_cuda_idx)
                text_a = _transcribe_segment(seg["wav_path"], model_a, language, device, hotwords)
            except Exception as e:
                error_a = str(e)

        def _transcribe_b():
            nonlocal text_b, error_b
            try:
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.set_device(_cuda_idx)
                text_b = _transcribe_segment(seg["wav_path"], model_b, language, device, hotwords)
            except Exception as e:
                error_b = str(e)

        # Execute both transcriptions concurrently
        t_a = threading.Thread(target=_transcribe_a, daemon=True)
        t_b = threading.Thread(target=_transcribe_b, daemon=True)
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        # Save TXT transcriptions in a txt/ subfolder alongside segment WAVs.
        # Naming: {seg_name_without_ext}_{model_abbr}.txt
        # e.g., xxx_seg001_qwen3-asr.txt, xxx_seg001_firered-asr2.txt
        txt_dir = os.path.join(seg_out_dir, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        seg_base = os.path.splitext(seg["name"])[0]  # e.g., "xxx_seg001"
        txt_path_a = os.path.join(txt_dir, f"{seg_base}_{model_a_info.get('abbr', model_a)}.txt")
        txt_path_b = os.path.join(txt_dir, f"{seg_base}_{model_b_info.get('abbr', model_b)}.txt")
        with open(txt_path_a, "w", encoding="utf-8") as _f:
            _f.write((text_a or "").strip())
        with open(txt_path_b, "w", encoding="utf-8") as _f:
            _f.write((text_b or "").strip())

        # Retry: if firered-asr2 returned empty in a sub-thread (silent GPU
        # failure — see FireRedAsr2.transcribe() which swallows exceptions),
        # retry it once in the main thread where the CUDA context is primary.
        # Only firered has this silent-failure mode; other models raise
        # exceptions on real errors.
        for _retry_mk, _is_model_a in ((model_a, True), (model_b, False)):
            _retry_text = text_a if _is_model_a else text_b
            _retry_err = error_a if _is_model_a else error_b
            if _retry_text or _retry_err:
                continue  # already got a result or a hard error
            _retry_mi = ASR_MODELS.get(_retry_mk, {})
            if _retry_mi.get("framework") != "firered":
                continue  # only firered has known silent-failure mode
            try:
                _retry_result = _transcribe_segment(
                    seg["wav_path"], _retry_mk, language, device, hotwords
                )
                if _retry_result:
                    if _is_model_a:
                        text_a = _retry_result
                    else:
                        text_b = _retry_result
            except Exception as _retry_exc:
                if _is_model_a:
                    error_a = str(_retry_exc)
                else:
                    error_b = str(_retry_exc)

        # Compare
        norm_a = _normalize_text_mfa(text_a)
        norm_b = _normalize_text_mfa(text_b)

        # English filter: if enabled and either side has English → 0% match
        if filter_english and (_has_english(norm_a) or _has_english(norm_b)):
            diff_percent = 100.0
        elif not norm_a and not norm_b:
            diff_percent = 0.0
        elif not norm_a or not norm_b:
            diff_percent = 100.0
        else:
            dist = _levenshtein(norm_a, norm_b)
            max_len = max(len(norm_a), len(norm_b))
            diff_percent = round((dist / max_len) * 100, 1)

        match_rate = round(100.0 - diff_percent, 1)
        diff_chunks = _compute_diff_chunks(text_a, text_b) if (text_a or text_b) else []

        segment_results.append({
            "seg_index": seg["index"],
            "seg_name": seg["name"],
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "duration_s": seg["duration_s"],
            "wav_path": seg["wav_path"],
            "text_a": text_a,
            "text_b": text_b,
            "txt_path_a": txt_path_a,
            "txt_path_b": txt_path_b,
            "error_a": error_a,
            "error_b": error_b,
            "diff_percent": diff_percent,
            "match_rate": match_rate,
            "flagged": diff_percent > 20.0,  # 20% diff = 80% match threshold
            "diff_chunks": diff_chunks,
            "user_action": None,
        })

        if on_segment:
            partial = {
                "audio_path": audio_path,
                "audio_name": base_name,
                "source_dir": source_dir,
                "duration_sec": round(duration, 1),
                "segment_count": len(segment_results),
                "model_a": {"key": model_a, "name": model_a_info["name"], "abbr": model_a_info.get("abbr", model_a)},
                "model_b": {"key": model_b, "name": model_b_info["name"], "abbr": model_b_info.get("abbr", model_b)},
                "segments_dir": seg_out_dir,
                "segments": list(segment_results),
            }
            try:
                on_segment(idx + 1, total, partial)
            except Exception:
                pass

    if progress_callback:
        progress_callback("completed", 100)

    return {
        "audio_path": audio_path,
        "audio_name": base_name,
        "source_dir": source_dir,
        "duration_sec": round(duration, 1),
        "segment_count": len(segment_results),
        "model_a": {"key": model_a, "name": model_a_info["name"], "abbr": model_a_info.get("abbr", model_a)},
        "model_b": {"key": model_b, "name": model_b_info["name"], "abbr": model_b_info.get("abbr", model_b)},
        "segments_dir": seg_out_dir,
        "segments": segment_results,
    }
