"""
Shared configuration for the anime pipeline.
All paths, tool locations, and settings are centralized here.
"""
import os
import sys
import json

# --- Platform detection ---
_IS_WIN = sys.platform == "win32"
_EXE = ".exe" if _IS_WIN else ""

# --- Project root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMICUT_ROOT = os.path.dirname(PROJECT_ROOT)
VIDEO_DIR = os.path.join(COMICUT_ROOT, "video")

# --- Data directories ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")
SUBTITLE_DIR = os.path.join(DATA_DIR, "subtitles")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# --- Model cache: redirect all downloads into project folder ---
_MODELS_DIR = os.path.join(DATA_DIR, "models")
_HF_CACHE = os.path.join(_MODELS_DIR, "huggingface")
_MS_CACHE = os.path.join(_MODELS_DIR, "modelscope")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_CACHE)
os.environ.setdefault("MODELSCOPE_CACHE", _MS_CACHE)
os.environ["HF_HUB_OFFLINE"] = "1"          # block all HuggingFace network access
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # transformers offline mode

APPROVED_DIR = os.path.join(DATA_DIR, "approved")  # clips that pass audio review
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")    # denoised audio output (reviewed clips)
CLEANED_UNREVIEWED_DIR = os.path.join(DATA_DIR, "cleaned_unreviewed")  # denoised from unreviewed
DENOISED_APPROVED_DIR = os.path.join(DATA_DIR, "denoised_approved")    # final approved denoised audio
EMOTION_DIR = os.path.join(DATA_DIR, "情绪")              # emotion classification output
EMOTION_DENOISE_DIR = os.path.join(DATA_DIR, "情绪降噪")   # denoise emotion classification output

# --- External tools ---
# Each tool checks a local bundled path first, then falls back to system PATH.
# On Windows the binaries carry .exe; on Linux/macOS they do not.

import shutil as _shutil

def _find_tool(name, *preferred_dirs):
    """Find a tool binary, checking preferred_dirs first, then PATH."""
    fname = f"{name}{_EXE}"
    for d in preferred_dirs:
        p = os.path.join(d, fname) if d else fname
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Fallback to PATH (shutil.which handles _EXE automatically on Windows)
    found = _shutil.which(name)
    return found or fname

MKVTOOLNIX_DIR = os.path.join(COMICUT_ROOT, "mkvtoolnix")
MKVEXTRACT = _find_tool("mkvextract", MKVTOOLNIX_DIR)
MKVMERGE = _find_tool("mkvmerge", MKVTOOLNIX_DIR)
MKVINFO = _find_tool("mkvinfo", MKVTOOLNIX_DIR)

QUICKCUT_DIR = os.path.join(COMICUT_ROOT, "QuickCut")
FFMPEG = _find_tool("ffmpeg", QUICKCUT_DIR)
FFPROBE = _find_tool("ffprobe", QUICKCUT_DIR)

# --- API ---
ANIMEGARDEN_API = "https://api.animes.garden"
RESOURCES_ENDPOINT = f"{ANIMEGARDEN_API}/resources"
ANIMEGARDEN_RESOURCE_DETAIL = f"{ANIMEGARDEN_API}/resource"

# --- Download ---
# aria2c — bundled in project tools/ directory, or system PATH
ARIA2C = _find_tool("aria2c", os.path.join(PROJECT_ROOT, "tools"))

# qBittorrent — primary download backend
_QB_LOCAL_DIR = os.path.join(PROJECT_ROOT, "tools", "qbittorrent")
_QB_PATH = _find_tool("qbittorrent", _QB_LOCAL_DIR)
if _QB_PATH != os.path.join(_QB_LOCAL_DIR, f"qbittorrent{_EXE}"):
    QBITTORRENT_EXE = _QB_PATH  # found on PATH
elif _IS_WIN:
    for _pf in [r"C:\Program Files\qBittorrent", r"C:\Program Files (x86)\qBittorrent"]:
        _p = os.path.join(_pf, "qbittorrent.exe")
        if os.path.exists(_p):
            QBITTORRENT_EXE = _p
            break
    else:
        QBITTORRENT_EXE = "qbittorrent.exe"
else:
    QBITTORRENT_EXE = "qbittorrent"

# BitComet — alternative download backend (Windows only)
_BIT_LOCAL_DIR = os.path.join(PROJECT_ROOT, "tools", "BitComet")
_BIT_PATH = _find_tool("BitComet", _BIT_LOCAL_DIR)
if _BIT_PATH != os.path.join(_BIT_LOCAL_DIR, f"BitComet{_EXE}"):
    BITCOMET_EXE = _BIT_PATH
elif _IS_WIN:
    for _pf in [r"C:\Program Files\BitComet", r"C:\Program Files (x86)\BitComet"]:
        _p = os.path.join(_pf, "BitComet.exe")
        if os.path.exists(_p):
            BITCOMET_EXE = _p
            break
    else:
        BITCOMET_EXE = "BitComet.exe"
else:
    BITCOMET_EXE = "BitComet"

# --- Video splitting ---
# Hardware acceleration: auto-detect, or force one of: nvenc, amf, qsv, none
HW_ACCEL = "auto"
# Output segment naming format
SEGMENT_NAME_FMT = "{title}_S{index:03d}_{start}_{end}.mp4"

# --- Default request headers ---
HEADERS = {
    "User-Agent": "AnimePipeline/1.0",
    "Accept": "application/json",
}

STITCHED_DIR = os.path.join(DATA_DIR, "stitched")
PIPELINE_VIDEO_DIR = os.path.join(DATA_DIR, "pipelinevideo")

# --- Video splitting / cutting ---
SPLIT_DIR = os.path.join(DATA_DIR, "split")

# --- MFA (Montreal Forced Aligner) directories ---
MFA_DIR = os.path.join(DATA_DIR, "mfa")
MFA_RAW_WAV_DIR = os.path.join(MFA_DIR, "raw_wav")
MFA_WAV_DIR = os.path.join(MFA_DIR, "wav")
MFA_TXT_DIR = os.path.join(MFA_DIR, "txt")
MFA_ALIGNED_DIR = os.path.join(MFA_DIR, "aligned")
MFA_JSONL_DIR = os.path.join(MFA_DIR, "jsonl")
MFA_POST_DIR = os.path.join(MFA_DIR, "post")
MFA_FILTERED_DIR = os.path.join(MFA_DIR, "filtered")
MFA_VALIDATE_DIR = os.path.join(MFA_DIR, "validate")
MFA_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts", "mfa")
MFA_MODELS_DIR = os.path.join(MFA_DIR, "models")
MFA_TEMP_DIR = os.path.join(MFA_DIR, "temp")
MFA_ENERGY_DIR = os.path.join(MFA_DIR, "energy")
MFA_DICT_PATH = os.path.join(MFA_MODELS_DIR, "pretrained_models", "dictionary", "japanese_mfa.dict")
MFA_DICT_PATH_ZH = os.path.join(MFA_MODELS_DIR, "pretrained_models", "dictionary", "fullpinyin_enword.dict")

# --- Hotword configurations ---
HOTWORDS_DIR = os.path.join(DATA_DIR, "hotwords")

# --- Settings overrides ---
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

# Only these path keys are user-configurable and shown in settings UI
_USER_PATH_KEYS = {
    'DOWNLOAD_DIR', 'SUBTITLE_DIR', 'CLIPS_DIR', 'TEMP_DIR',
    'APPROVED_DIR', 'CLEANED_DIR', 'CLEANED_UNREVIEWED_DIR',
    'DENOISED_APPROVED_DIR', 'STITCHED_DIR',
    'EMOTION_DIR', 'EMOTION_DENOISE_DIR',
    'ASR_DIR', 'ASR_AUDIO_DIR', 'ASR_SUBTITLE_DIR',
    'ASR_COMPARE_DIR', 'ASR_COMPARE_SUBTITLE_DIR', 'ASR_COMPARE_AUDIO_DIR',
    'ASR_COMPARE_OUTPUT_DIR', 'ASR_COMPARE_DISCARD_DIR',
    'ASR_COMPARE_SEGMENTS_DIR', 'ASR_COMPARE_KEPT_DIR',
    'TXT_COMPARE_DIR', 'TXT_COMPARE_OUTPUT_DIR',
    'TXT_COMPARE_DISCARD_DIR', 'TXT_COMPARE_DEFAULT_PATH',
    'HOTWORDS_DIR',
    'PIPELINE_VIDEO_DIR',
    'SPLIT_DIR',
    'MFA_RAW_WAV_DIR', 'MFA_WAV_DIR', 'MFA_TXT_DIR',
    'MFA_ALIGNED_DIR', 'MFA_JSONL_DIR', 'MFA_POST_DIR', 'MFA_FILTERED_DIR',
    'MFA_VALIDATE_DIR', 'MFA_MODELS_DIR', 'MFA_TEMP_DIR', 'MFA_ENERGY_DIR', 'MFA_DICT_PATH', 'MFA_DICT_PATH_ZH',
}

# Non-path config keys stored in settings.json (model/language selections etc.)
_CONFIG_KEYS = {
    'ASR_DEFAULT_MODEL': 'qwen3-asr',
    'ASR_DEFAULT_LANGUAGE': 'zh',
    'ASR_COMPARE_MODEL_A': 'qwen3-asr',
    'ASR_COMPARE_MODEL_B': 'firered-asr2',
    'ASR_DEFAULT_HOTWORDS': '',
    'ASR_COMPARE_MATCH_THRESHOLD': '90',  # flag when match rate < this %
    'FIRERED_ASR2_MODELS_DIR': os.path.join(DATA_DIR, "models", "firered_asr2"),
    'FIRERED_ASR2S_PATH': os.path.join(COMICUT_ROOT, "FireRedASR2S"),
    'DASHSCOPE_API_KEY': '',
    'DASHSCOPE_API_BASE': 'https://dashscope.aliyuncs.com',
    'MFA_PYTHON': sys.executable,
    'MFA_EXECUTABLE': _find_tool('mfa',
        os.path.expanduser("~/miniconda3/bin"),
        os.path.expanduser("~/anaconda3/bin"),
        "/opt/conda/bin", ""),
    'MFA_DEFAULT_ACOUSTIC': 'japanese_mfa',
    'MFA_DEFAULT_DICTIONARY': 'japanese_mfa',
    'MFA_DEFAULT_NUM_JOBS': '8',
}

# Populate module-level defaults from _CONFIG_KEYS
for _k, _v in _CONFIG_KEYS.items():
    if _k not in globals():
        globals()[_k] = _v

# Registry of all configurable path variables: (name, current_value)
_PATH_VARS = {}


def _register_path_vars():
    """Collect all user-configurable _DIR path variables."""
    if _PATH_VARS:
        return
    for key in _USER_PATH_KEYS:
        val = globals().get(key, '')
        if isinstance(val, str):
            _PATH_VARS[key] = val


# Populate eagerly so GET /api/settings works before first save/load
_register_path_vars()


def _is_stale_settings(paths: dict) -> bool:
    """Check if settings.json paths are from another machine.

    Returns True if any path override points to a non-existent directory,
    indicating the settings were copied from a different environment.
    """
    if not paths:
        return False
    # Check CLIPS_DIR as the canary — if it doesn't exist, likely all paths are stale
    clip = paths.get('CLIPS_DIR', '')
    if clip and not os.path.isdir(clip):
        return True
    # Extra check: if DOWNLOAD_DIR override exists but doesn't match current root
    dl = paths.get('DOWNLOAD_DIR', '')
    if dl and not os.path.isdir(dl):
        return True
    return False


def _reset_to_default_paths():
    """Reset all user-configurable path vars to their computed defaults."""
    global PROJECT_ROOT, COMICUT_ROOT, DATA_DIR
    # Recompute root-derived paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COMICUT_ROOT = os.path.dirname(PROJECT_ROOT)

    new_defaults = {
        'DOWNLOAD_DIR': os.path.join(DATA_DIR, "downloads"),
        'SUBTITLE_DIR': os.path.join(DATA_DIR, "subtitles"),
        'CLIPS_DIR': os.path.join(DATA_DIR, "clips"),
        'TEMP_DIR': os.path.join(DATA_DIR, "temp"),
        'APPROVED_DIR': os.path.join(DATA_DIR, "approved"),
        'CLEANED_DIR': os.path.join(DATA_DIR, "cleaned"),
        'CLEANED_UNREVIEWED_DIR': os.path.join(DATA_DIR, "cleaned_unreviewed"),
        'DENOISED_APPROVED_DIR': os.path.join(DATA_DIR, "denoised_approved"),
        'STITCHED_DIR': os.path.join(DATA_DIR, "stitched"),
        'EMOTION_DIR': os.path.join(DATA_DIR, "情绪"),
        'EMOTION_DENOISE_DIR': os.path.join(DATA_DIR, "情绪降噪"),
        'ASR_DIR': os.path.join(DATA_DIR, "asr"),
        'ASR_AUDIO_DIR': os.path.join(DATA_DIR, "asr", "audio"),
        'ASR_SUBTITLE_DIR': os.path.join(DATA_DIR, "asr", "subtitles"),
        'ASR_COMPARE_DIR': os.path.join(DATA_DIR, "asr_compare"),
        'ASR_COMPARE_SUBTITLE_DIR': os.path.join(DATA_DIR, "asr_compare", "subtitles"),
        'ASR_COMPARE_AUDIO_DIR': os.path.join(DATA_DIR, "asr_compare", "audio"),
        'ASR_COMPARE_OUTPUT_DIR': os.path.join(DATA_DIR, "asr_compare_output"),
        'ASR_COMPARE_DISCARD_DIR': os.path.join(DATA_DIR, "asr_compare", "discarded"),
        'ASR_COMPARE_SEGMENTS_DIR': os.path.join(DATA_DIR, "asr_compare", "segments"),
        'ASR_COMPARE_KEPT_DIR': os.path.join(DATA_DIR, "asr_compare", "kept"),
        'ASR_COMPARE_DEFAULT_PATH': '/mnt/nas/Persons/jiangyichen/Enhanced',
        'TXT_COMPARE_DIR': os.path.join(DATA_DIR, "txt_compare"),
        'TXT_COMPARE_OUTPUT_DIR': os.path.join(DATA_DIR, "txt_compare_output"),
        'TXT_COMPARE_DISCARD_DIR': os.path.join(DATA_DIR, "txt_compare", "discarded"),
        'TXT_COMPARE_DEFAULT_PATH': '',
        'HOTWORDS_DIR': os.path.join(DATA_DIR, "hotwords"),
        'PIPELINE_VIDEO_DIR': os.path.join(DATA_DIR, "pipelinevideo"),
        'SPLIT_DIR': os.path.join(DATA_DIR, "split"),
        'MFA_RAW_WAV_DIR': os.path.join(DATA_DIR, "mfa", "raw_wav"),
        'MFA_WAV_DIR': os.path.join(DATA_DIR, "mfa", "wav"),
        'MFA_TXT_DIR': os.path.join(DATA_DIR, "mfa", "txt"),
        'MFA_ALIGNED_DIR': os.path.join(DATA_DIR, "mfa", "aligned"),
        'MFA_POST_DIR': os.path.join(DATA_DIR, "mfa", "post"),
        'MFA_FILTERED_DIR': os.path.join(DATA_DIR, "mfa", "filtered"),
        'MFA_VALIDATE_DIR': os.path.join(DATA_DIR, "mfa", "validate"),
        'MFA_MODELS_DIR': os.path.join(DATA_DIR, "mfa", "models"),
        'MFA_TEMP_DIR': os.path.join(DATA_DIR, "mfa", "temp"),
        'MFA_ENERGY_DIR': os.path.join(DATA_DIR, "mfa", "energy"),
        'MFA_DICT_PATH': os.path.join(DATA_DIR, "mfa", "models", "pretrained_models", "dictionary", "japanese_mfa.dict"),
        'MFA_DICT_PATH_ZH': os.path.join(DATA_DIR, "mfa", "models", "pretrained_models", "dictionary", "fullpinyin_enword.dict"),
    }
    for key, val in new_defaults.items():
        globals()[key] = val
        if key in _PATH_VARS:
            _PATH_VARS[key] = val


def load_settings():
    """Load user settings overrides from settings.json and patch module globals.

    If settings.json contains paths from another machine (non-existent dirs),
    ignores the stale paths and falls back to auto-computed defaults.
    Non-path config (model selections, steps) is preserved across migrations.
    """
    _register_path_vars()
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
    except Exception:
        return

    paths = data.get('paths', {})
    stale = _is_stale_settings(paths)

    if stale:
        # Stale settings from another machine — reset paths to current defaults,
        # but preserve non-path config (model choices, step order, etc.)
        print("[config] Stale settings.json detected (paths point to another machine).")
        print("[config] Resetting paths to current environment defaults.")
        _reset_to_default_paths()
    else:
        for key, val in paths.items():
            if key in _USER_PATH_KEYS and isinstance(val, str) and val:
                globals()[key] = val
                _PATH_VARS[key] = val

    # Non-path settings — always load (survive machine migration)
    steps = data.get('denoise_default_steps', None)
    if steps is not None:
        globals()['DENOISE_DEFAULT_STEPS'] = steps
    pv_steps = data.get('pv_default_steps', None)
    if pv_steps is not None:
        globals()['PV_DEFAULT_STEPS'] = pv_steps

    config_vals = data.get('config', {})
    for key, val in config_vals.items():
        if key in _CONFIG_KEYS and isinstance(val, str) and val:
            globals()[key] = val


def save_settings(paths: dict, denoise_default_steps=None, pv_default_steps=None, config_vals=None):
    """Save settings to settings.json and patch module globals."""
    _register_path_vars()
    for key, val in paths.items():
        if key in _USER_PATH_KEYS and isinstance(val, str) and val:
            globals()[key] = val
            _PATH_VARS[key] = val
    if denoise_default_steps is not None:
        globals()['DENOISE_DEFAULT_STEPS'] = denoise_default_steps
    if pv_default_steps is not None:
        globals()['PV_DEFAULT_STEPS'] = pv_default_steps
    if config_vals is not None:
        for key, val in config_vals.items():
            if key in _CONFIG_KEYS and isinstance(val, str) and val:
                globals()[key] = val

    data = {'paths': {k: globals()[k] for k in _USER_PATH_KEYS if k in globals()}}
    if denoise_default_steps is not None or 'DENOISE_DEFAULT_STEPS' in globals():
        data['denoise_default_steps'] = globals().get('DENOISE_DEFAULT_STEPS', [])
    if pv_default_steps is not None or 'PV_DEFAULT_STEPS' in globals():
        data['pv_default_steps'] = globals().get('PV_DEFAULT_STEPS', [])
    data['config'] = {k: globals().get(k, _CONFIG_KEYS[k]) for k in _CONFIG_KEYS}

    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    tmp = SETTINGS_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SETTINGS_FILE)

# ASR directories
ASR_DIR = os.path.join(DATA_DIR, "asr")
ASR_AUDIO_DIR = os.path.join(ASR_DIR, "audio")
ASR_SUBTITLE_DIR = os.path.join(ASR_DIR, "subtitles")

# ASR comparison directories
ASR_COMPARE_DIR = os.path.join(DATA_DIR, "asr_compare")
ASR_COMPARE_SUBTITLE_DIR = os.path.join(ASR_COMPARE_DIR, "subtitles")
ASR_COMPARE_AUDIO_DIR = os.path.join(ASR_COMPARE_DIR, "audio")
ASR_COMPARE_OUTPUT_DIR = os.path.join(DATA_DIR, "asr_compare_output")
ASR_COMPARE_DISCARD_DIR = os.path.join(ASR_COMPARE_DIR, "discarded")
ASR_COMPARE_SEGMENTS_DIR = os.path.join(ASR_COMPARE_DIR, "segments")
ASR_COMPARE_KEPT_DIR = os.path.join(ASR_COMPARE_DIR, "kept")
# Default browse path for ASR comparison tab (can be a network share / UNC path)
ASR_COMPARE_DEFAULT_PATH = _CONFIG_KEYS.get('ASR_COMPARE_DEFAULT_PATH', '')

# Text comparison directories
TXT_COMPARE_DIR = os.path.join(DATA_DIR, "txt_compare")
TXT_COMPARE_OUTPUT_DIR = os.path.join(DATA_DIR, "txt_compare_output")
TXT_COMPARE_DISCARD_DIR = os.path.join(TXT_COMPARE_DIR, "discarded")
TXT_COMPARE_DEFAULT_PATH = _CONFIG_KEYS.get('TXT_COMPARE_DEFAULT_PATH', '')

# Model cache directories (all downloads go into project folder)
MODELS_DIR = _MODELS_DIR
ASR_MODELS_DIR = os.path.join(_MODELS_DIR, "asr")
HF_CACHE_DIR = _HF_CACHE
MS_CACHE_DIR = _MS_CACHE

# Ensure all data directories exist
for d in [DOWNLOAD_DIR, SUBTITLE_DIR, CLIPS_DIR, TEMP_DIR, APPROVED_DIR, CLEANED_DIR, CLEANED_UNREVIEWED_DIR, DENOISED_APPROVED_DIR, STITCHED_DIR, PIPELINE_VIDEO_DIR, ASR_DIR, ASR_AUDIO_DIR, ASR_SUBTITLE_DIR, ASR_COMPARE_DIR, ASR_COMPARE_SUBTITLE_DIR, ASR_COMPARE_AUDIO_DIR, ASR_COMPARE_OUTPUT_DIR, ASR_COMPARE_DISCARD_DIR, ASR_COMPARE_SEGMENTS_DIR, ASR_COMPARE_KEPT_DIR, TXT_COMPARE_DIR, TXT_COMPARE_OUTPUT_DIR, TXT_COMPARE_DISCARD_DIR, EMOTION_DIR, EMOTION_DENOISE_DIR, SPLIT_DIR, MFA_DIR, MFA_RAW_WAV_DIR, MFA_WAV_DIR, MFA_TXT_DIR, MFA_ALIGNED_DIR, MFA_JSONL_DIR, MFA_POST_DIR, MFA_FILTERED_DIR, MFA_VALIDATE_DIR, MFA_MODELS_DIR, MFA_TEMP_DIR, MFA_ENERGY_DIR]:
    os.makedirs(d, exist_ok=True)


def detect_hw_accel() -> str:
    """Detect available hardware acceleration for ffmpeg encoding."""
    import subprocess
    try:
        result = subprocess.run(
            [FFMPEG, "-hide_banner", "-encoders"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30
        )
        encoders = result.stdout
        if "h264_nvenc" in encoders:
            return "nvenc"
        elif "h264_amf" in encoders:
            return "amf"
        elif "h264_qsv" in encoders:
            return "qsv"
    except Exception:
        pass
    return "libx264"  # software fallback
