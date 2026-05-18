"""
Shared configuration for the anime pipeline.
All paths, tool locations, and settings are centralized here.
"""
import os
import sys

# --- Project root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMICUT_ROOT = os.path.dirname(PROJECT_ROOT)

# --- Data directories ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")
SUBTITLE_DIR = os.path.join(DATA_DIR, "subtitles")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
APPROVED_DIR = os.path.join(DATA_DIR, "approved")  # clips that pass audio review
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")    # denoised audio output (reviewed clips)
CLEANED_UNREVIEWED_DIR = os.path.join(DATA_DIR, "cleaned_unreviewed")  # denoised from unreviewed

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
QBITTORRENT_EXE = r"C:\Program Files\qBittorrent\qbittorrent.exe"

# BitComet — alternative download backend
BITCOMET_EXE = r"C:\Program Files\BitComet\BitComet.exe"

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

# Ensure all data directories exist
for d in [DOWNLOAD_DIR, SUBTITLE_DIR, CLIPS_DIR, TEMP_DIR, APPROVED_DIR, CLEANED_DIR, CLEANED_UNREVIEWED_DIR, STITCHED_DIR]:
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
