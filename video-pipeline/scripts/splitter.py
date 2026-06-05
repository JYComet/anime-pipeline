"""
Duration-based audio/video splitting using ffmpeg.
Extracted from the original Anime Pipeline server.py _step_duration_split.
"""
import os
import subprocess

from .config_loader import get_tool, get_processing

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}


def get_media_info(file_path):
    """Get duration and size for a media file via ffprobe."""
    import json
    ffprobe = get_tool("ffprobe")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", file_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            return {
                "duration_s": float(fmt.get("duration", 0)),
                "size_mb": os.path.getsize(file_path) / 1024 / 1024,
            }
    except Exception:
        pass
    return {"duration_s": 0, "size_mb": os.path.getsize(file_path) / 1024 / 1024}


def split_by_duration(src_path, seg_dir, base_name, logger=None):
    """Split a media file into fixed-duration segments.

    Strategy (same as original Anime Pipeline):
    - WAV source: split and resample to unified format in one pass
    - Video source (MP4/MKV/etc.): extract audio + convert to WAV + split in one pass
    - Audio source (AAC/MP3/FLAC/etc.): split with ``-c copy``, convert later

    First and last segments are discarded by default (opening/closing noise).

    Args:
        src_path: Path to the source media file.
        seg_dir: Directory to write segments into.
        base_name: Base filename (without extension) for naming segments.
        logger: Optional logger instance.

    Returns:
        dict with keys:
            - segments: list of segment file paths (may need WAV conversion)
            - seg_ext: extension of the segment files
            - needs_convert: bool, whether segments need WAV conversion
            - sample_rate: target sample rate
            - channels: target channels
            - error: error message string (empty if success)
    """
    proc = get_processing()
    segment_dur = proc.get("segment_duration", 600)
    keep_ends = proc.get("keep_ends", False)
    opt_sr = proc.get("sample_rate", 32000)
    opt_ch = proc.get("channels", 1)
    ffmpeg = get_tool("ffmpeg")

    log = logger.info if logger else print

    ext = os.path.splitext(src_path)[1].lower()

    if ext == ".wav":
        seg_pattern = os.path.join(seg_dir, f"{base_name}_%03d.wav")
        cmd = [
            ffmpeg, "-y", "-i", src_path,
            "-f", "segment", "-segment_time", str(segment_dur),
            "-acodec", "pcm_s16le",
            "-ar", str(opt_sr), "-ac", str(opt_ch),
            seg_pattern,
        ]
        seg_ext = ".wav"
        needs_convert = False
    elif ext in VIDEO_EXTENSIONS:
        seg_pattern = os.path.join(seg_dir, f"{base_name}_%03d.wav")
        cmd = [
            ffmpeg, "-y", "-i", src_path, "-vn",
            "-f", "segment", "-segment_time", str(segment_dur),
            "-acodec", "pcm_s16le",
            "-ar", str(opt_sr), "-ac", str(opt_ch),
            seg_pattern,
        ]
        seg_ext = ".wav"
        needs_convert = False
    else:
        seg_ext = ext if ext else ".aac"
        seg_pattern = os.path.join(seg_dir, f"{base_name}_%03d{seg_ext}")
        cmd = [
            ffmpeg, "-y", "-i", src_path,
            "-f", "segment", "-segment_time", str(segment_dur),
            "-c", "copy", seg_pattern,
        ]
        needs_convert = True

    log(f"  [切分] 执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        if result.returncode != 0 and result.stderr:
            stderr_msg = result.stderr.strip()[-300:]
            if logger:
                logger.warning(f"  [切分] ffmpeg stderr: {stderr_msg}")
            else:
                print(f"  [切分] ffmpeg stderr: {stderr_msg}")

        segments = sorted([
            os.path.join(seg_dir, x) for x in os.listdir(seg_dir)
            if x.endswith(seg_ext) and os.path.getsize(os.path.join(seg_dir, x)) > 0
        ])

        if len(segments) >= 3 and not keep_ends:
            removed_first = segments.pop(0)
            removed_last = segments.pop(-1)
            os.remove(removed_first)
            os.remove(removed_last)
            log(f"  [切分] 去除首尾分段: 原始{len(segments) + 2}段 → {len(segments)}段")

        if segments:
            log(f"  [切分] 完成: {len(segments)}段 (每段{segment_dur}秒)")
            return {
                "segments": segments,
                "seg_ext": seg_ext,
                "needs_convert": needs_convert,
                "sample_rate": opt_sr,
                "channels": opt_ch,
                "error": "",
            }
        else:
            return {
                "segments": [], "seg_ext": seg_ext,
                "needs_convert": needs_convert,
                "sample_rate": opt_sr, "channels": opt_ch,
                "error": "切分未产生有效片段",
            }
    except subprocess.TimeoutExpired:
        return {"segments": [], "seg_ext": seg_ext, "needs_convert": False,
                "sample_rate": opt_sr, "channels": opt_ch,
                "error": f"切分超时 (>600秒)"}
    except Exception as e:
        return {"segments": [], "seg_ext": seg_ext, "needs_convert": False,
                "sample_rate": opt_sr, "channels": opt_ch,
                "error": str(e)}
