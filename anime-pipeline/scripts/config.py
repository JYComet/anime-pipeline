"""
Shared configuration for the anime pipeline.
All paths, tool locations, and settings are centralized here.
"""
import os
import sys
import json

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
APPROVED_DIR = os.path.join(DATA_DIR, "approved")  # clips that pass audio review
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")    # denoised audio output (reviewed clips)
CLEANED_UNREVIEWED_DIR = os.path.join(DATA_DIR, "cleaned_unreviewed")  # denoised from unreviewed
DENOISED_APPROVED_DIR = os.path.join(DATA_DIR, "denoised_approved")    # final approved denoised audio
EMOTION_DIR = os.path.join(DATA_DIR, "情绪")              # emotion classification output
EMOTION_DENOISE_DIR = os.path.join(DATA_DIR, "情绪降噪")   # denoise emotion classification output

# --- External tools ---
MKVTOOLNIX_DIR = os.path.join(COMICUT_ROOT, "mkvtoolnix")
MKVEXTRACT = os.path.join(MKVTOOLNIX_DIR, "mkvextract.exe")
MKVMERGE = os.path.join(MKVTOOLNIX_DIR, "mkvmerge.exe")
MKVINFO = os.path.join(MKVTOOLNIX_DIR, "mkvinfo.exe")

QUICKCUT_DIR = os.path.join(COMICUT_ROOT, "QuickCut")
FFMPEG = os.path.join(QUICKCUT_DIR, "ffmpeg.exe")
FFPROBE = os.path.join(QUICKCUT_DIR, "ffprobe.exe")

# --- API ---
ANIMEGARDEN_API = "https://api.animes.garden"
RESOURCES_ENDPOINT = f"{ANIMEGARDEN_API}/resources"
ANIMEGARDEN_RESOURCE_DETAIL = f"{ANIMEGARDEN_API}/resource"

# --- Download ---
# aria2c — bundled in project tools/ directory (legacy fallback)
_ARIA2C_LOCAL = os.path.join(PROJECT_ROOT, "tools", "aria2c.exe")
ARIA2C = _ARIA2C_LOCAL if os.path.exists(_ARIA2C_LOCAL) else "aria2c"

# qBittorrent — primary download backend
_QBITTORRENT_LOCAL = os.path.join(PROJECT_ROOT, "tools", "qbittorrent", "qbittorrent.exe")
if os.path.exists(_QBITTORRENT_LOCAL):
    QBITTORRENT_EXE = _QBITTORRENT_LOCAL
else:
    _QBITTORRENT_PF = r"C:\Program Files\qBittorrent\qbittorrent.exe"
    _QBITTORRENT_PFX86 = r"C:\Program Files (x86)\qBittorrent\qbittorrent.exe"
    if os.path.exists(_QBITTORRENT_PF):
        QBITTORRENT_EXE = _QBITTORRENT_PF
    elif os.path.exists(_QBITTORRENT_PFX86):
        QBITTORRENT_EXE = _QBITTORRENT_PFX86
    else:
        QBITTORRENT_EXE = "qbittorrent.exe"

# BitComet — alternative download backend
_BITCOMET_LOCAL = os.path.join(PROJECT_ROOT, "tools", "BitComet", "BitComet.exe")
if os.path.exists(_BITCOMET_LOCAL):
    BITCOMET_EXE = _BITCOMET_LOCAL
else:
    _BITCOMET_PF = r"C:\Program Files\BitComet\BitComet.exe"
    _BITCOMET_PFX86 = r"C:\Program Files (x86)\BitComet\BitComet.exe"
    if os.path.exists(_BITCOMET_PF):
        BITCOMET_EXE = _BITCOMET_PF
    elif os.path.exists(_BITCOMET_PFX86):
        BITCOMET_EXE = _BITCOMET_PFX86
    else:
        BITCOMET_EXE = "BitComet.exe"

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
    'PIPELINE_VIDEO_DIR',
}

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


def load_settings():
    """Load user settings overrides from settings.json and patch module globals."""
    _register_path_vars()
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
    except Exception:
        return

    paths = data.get('paths', {})

    # Detect stale settings from another machine: if the CLIPS_DIR override
    # points to a path that doesn't exist, ignore all overrides and use defaults.
    if paths:
        _clip = paths.get('CLIPS_DIR', '')
        if _clip and not os.path.isdir(_clip):
            # Stale settings detected — keep defaults, save corrected on next save
            steps = data.get('denoise_default_steps', None)
            if steps is not None:
                globals()['DENOISE_DEFAULT_STEPS'] = steps
            return

    for key, val in paths.items():
        if key in _USER_PATH_KEYS and isinstance(val, str) and val:
            globals()[key] = val
            _PATH_VARS[key] = val

    steps = data.get('denoise_default_steps', None)
    if steps is not None:
        globals()['DENOISE_DEFAULT_STEPS'] = steps


def save_settings(paths: dict, denoise_default_steps=None):
    """Save settings to settings.json and patch module globals."""
    _register_path_vars()
    for key, val in paths.items():
        if key in _USER_PATH_KEYS and isinstance(val, str) and val:
            globals()[key] = val
            _PATH_VARS[key] = val
    if denoise_default_steps is not None:
        globals()['DENOISE_DEFAULT_STEPS'] = denoise_default_steps

    data = {'paths': {k: globals()[k] for k in _USER_PATH_KEYS if k in globals()}}
    if denoise_default_steps is not None or 'DENOISE_DEFAULT_STEPS' in globals():
        data['denoise_default_steps'] = globals().get('DENOISE_DEFAULT_STEPS', [])

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

# Ensure all data directories exist
for d in [DOWNLOAD_DIR, SUBTITLE_DIR, CLIPS_DIR, TEMP_DIR, APPROVED_DIR, CLEANED_DIR, CLEANED_UNREVIEWED_DIR, DENOISED_APPROVED_DIR, STITCHED_DIR, PIPELINE_VIDEO_DIR, ASR_DIR, ASR_AUDIO_DIR, ASR_SUBTITLE_DIR, ASR_COMPARE_DIR, ASR_COMPARE_SUBTITLE_DIR, ASR_COMPARE_AUDIO_DIR, ASR_COMPARE_OUTPUT_DIR, ASR_COMPARE_DISCARD_DIR, EMOTION_DIR, EMOTION_DENOISE_DIR]:
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
