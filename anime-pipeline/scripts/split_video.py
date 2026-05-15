"""
Video splitting module.
Splits MKV/MP4 videos into clips based on subtitle timing.
Uses ffmpeg with hardware acceleration for fast processing.
"""
import os
import re
import subprocess
import math
import threading
from typing import Optional
from dataclasses import dataclass, field

# Only one split at a time (ffmpeg GPU encoding is resource-heavy)
_split_semaphore = threading.BoundedSemaphore(1)

from config import FFMPEG, FFPROBE, CLIPS_DIR, TEMP_DIR, detect_hw_accel


@dataclass
class SubtitleEntry:
    """A single subtitle entry with timing."""
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str
    style: str = ""


@dataclass
class SubtitleFile:
    """Parsed subtitle file."""
    path: str
    format: str  # 'ass' or 'srt'
    entries: list[SubtitleEntry] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        if self.entries:
            return self.entries[-1].end
        return 0.0


def parse_ass_time(time_str: str) -> float:
    """Convert ASS time format (H:MM:SS.cc) to seconds."""
    # ASS format: 0:00:00.00 or 1:23:45.67
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
    # SRT format: 00:00:00,000
    time_str = time_str.strip().replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def parse_subtitle_file(sub_path: str) -> Optional[SubtitleFile]:
    """Parse an ASS or SRT subtitle file and extract entries with timing.

    Returns SubtitleFile with all entries, or None on failure.
    """
    if not os.path.exists(sub_path):
        print(f"Subtitle file not found: {sub_path}")
        return None

    ext = os.path.splitext(sub_path)[1].lower()

    with open(sub_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if ext in (".ass", ".ssa"):
        return _parse_ass(lines, sub_path)
    elif ext in (".srt",):
        return _parse_srt(lines, sub_path)
    else:
        print(f"Unknown subtitle format: {ext}")
        return None


def _parse_ass(lines: list[str], path: str) -> SubtitleFile:
    """Parse ASS/SSA subtitle format."""
    entries = []
    idx = 0
    for line in lines:
        if not line.startswith("Dialogue:"):
            continue
        # Split by comma, but preserve commas in the Text field (last field)
        # Format: Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        content = line[len("Dialogue:"):].strip()
        # Split only the first 9 commas (10 fields, last field is Text which may contain commas)
        parts = content.split(",", 9)
        if len(parts) < 10:
            continue
        start = parse_ass_time(parts[1].strip())
        end = parse_ass_time(parts[2].strip())
        style = parts[3].strip()
        text = parts[9].strip()
        if text:
            entries.append(SubtitleEntry(
                index=idx,
                start=start,
                end=end,
                text=_clean_ass_text(text),
                style=style,
            ))
            idx += 1

    return SubtitleFile(path=path, format="ass", entries=entries)


def _clean_ass_text(text: str) -> str:
    """Remove ASS override tags like {\\pos(100,200)} or {\\fad(300,300)}."""
    # Remove override blocks
    text = re.sub(r'\{[^}]*\}', '', text)
    # Remove \N line breaks, replace with space
    text = text.replace('\\N', ' ').replace('\\n', ' ')
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _parse_srt(lines: list[str], path: str) -> SubtitleFile:
    """Parse SRT subtitle format."""
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Try to parse entry number
        try:
            index = int(line)
        except ValueError:
            i += 1
            continue

        i += 1
        if i >= len(lines):
            break

        # Timing line: 00:00:00,000 --> 00:00:05,000
        timing_line = lines[i].strip()
        timing_match = re.match(
            r'(\d+:\d+:\d+[,\.]\d+)\s*-->\s*(\d+:\d+:\d+[,\.]\d+)',
            timing_line
        )
        if not timing_match:
            i += 1
            continue

        start = parse_srt_time(timing_match.group(1))
        end = parse_srt_time(timing_match.group(2))
        i += 1

        # Text lines (until empty line or next number)
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        text = " ".join(text_lines)
        # Remove HTML tags from SRT
        text = re.sub(r'<[^>]+>', '', text)

        entries.append(SubtitleEntry(
            index=index - 1,
            start=start,
            end=end,
            text=text,
        ))

    return SubtitleFile(path=path, format="srt", entries=entries)


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds using ffprobe."""
    cmd = [
        FFPROBE, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return 0.0


def get_hw_accel_params(hw_type: str = "auto") -> dict:
    """Get ffmpeg hardware acceleration parameters.

    Returns dict with keys: hwaccel_in, video_encoder, extra_flags
    """
    if hw_type == "auto":
        hw_type = detect_hw_accel()

    configs = {
        "nvenc": {
            "hwaccel_in": [],
            "video_encoder": "h264_nvenc",
            "extra_flags": ["-rc", "constqp", "-qp", "23", "-pix_fmt", "yuv420p"],
        },
        "amf": {
            "hwaccel_in": ["-hwaccel", "d3d11va"],
            "video_encoder": "h264_amf",
            "extra_flags": ["-quality", "quality"],
        },
        "qsv": {
            "hwaccel_in": ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"],
            "video_encoder": "h264_qsv",
            "extra_flags": ["-preset", "fast"],
        },
        "libx264": {
            "hwaccel_in": [],
            "video_encoder": "libx264",
            "extra_flags": ["-preset", "ultrafast", "-crf", "18"],
        },
    }

    return configs.get(hw_type, configs["libx264"])


def split_video_by_subtitle(
    video_path: str,
    subtitle_path: str,
    output_dir: str = "",
    padding: float = 0.1,  # seconds to pad before/after
    hw_accel: str = "auto",
    min_duration: float = 0.5,  # skip entries shorter than this
    on_progress=None,
    cancel_event=None,  # threading.Event to signal cancellation
) -> list[str]:
    """Split a video into clips based on subtitle timing.

    Each subtitle entry becomes a separate video clip.

    Args:
        video_path: Path to the source video file
        subtitle_path: Path to the ASS/SRT subtitle file
        output_dir: Where to write the clips
        padding: Extra seconds to pad around each segment
        hw_accel: Hardware acceleration type (nvenc/amf/qsv/libx264/auto)
        min_duration: Minimum segment duration in seconds
        on_progress: Optional callback(current, total, text)
        cancel_event: Optional threading.Event — set to abort splitting

    Returns:
        List of paths to the created video clips.
    """
    # Acquire split lock — only one video splits at a time
    _split_semaphore.acquire()
    try:
        return _split_video_by_subtitle_impl(
            video_path, subtitle_path, output_dir, padding,
            hw_accel, min_duration, on_progress, cancel_event,
        )
    finally:
        _split_semaphore.release()


def _split_video_by_subtitle_impl(
    video_path: str,
    subtitle_path: str,
    output_dir: str = "",
    padding: float = 0.1,
    hw_accel: str = "auto",
    min_duration: float = 0.5,
    on_progress=None,
    cancel_event=None,
) -> list[str]:
    # Parse subtitles
    sub = parse_subtitle_file(subtitle_path)
    if not sub or not sub.entries:
        print(f"No subtitle entries found in {subtitle_path}")
        return []

    print(f"Parsed {len(sub.entries)} subtitle entries from {os.path.basename(subtitle_path)}")

    # Setup output
    if not output_dir:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(CLIPS_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Get HW acceleration config
    hw_config = get_hw_accel_params(hw_accel)
    hw_type = detect_hw_accel() if hw_accel == "auto" else hw_accel

    # Verify video duration
    video_duration = get_video_duration(video_path)
    if video_duration <= 0:
        print(f"Warning: Could not determine video duration for {video_path}")

    # Build ffmpeg base command parts
    # We cut with -ss before -i for fast seeking when possible
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    clips = []
    total = len(sub.entries)

    for i, entry in enumerate(sub.entries):
        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            print(f"[split] Cancelled after {len(clips)} clips")
            break

        start = max(0, entry.start - padding)
        end = min(video_duration or 999999, entry.end + padding)
        duration = end - start

        # Skip very short segments
        if duration < min_duration:
            continue

        # Sanitize text for filename (limit to 20 chars)
        safe_text = re.sub(r'[\\/*?:"<>|]', '', entry.text)[:20].strip()
        output_name = f"{base_name}_S{i+1:03d}_{safe_text}.mp4"
        output_path = os.path.join(output_dir, output_name)

        # ffmpeg command: fast seek before input, then encode segment
        cmd = [FFMPEG, "-y"]

        # Hardware decode input
        cmd.extend(hw_config["hwaccel_in"])

        # Fast seek (before input for speed)
        cmd.extend(["-ss", str(start)])

        # Input
        cmd.extend(["-i", video_path])

        # Duration to copy
        cmd.extend(["-t", str(duration)])

        # Map video and audio only (skip subtitles, fonts, etc.)
        cmd.extend(["-map", "0:v", "-map", "0:a?"])

        # Copy audio (much faster), encode video with HW
        cmd.extend(["-c:a", "copy"])

        # Video encoder with HW acceleration
        cmd.extend(["-c:v", hw_config["video_encoder"]])
        cmd.extend(hw_config["extra_flags"])

        # Avoid sync issues
        cmd.extend(["-avoid_negative_ts", "make_zero"])

        cmd.append(output_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8', errors='replace',
                timeout=300,  # 5 min per clip
            )
            if result.returncode == 0 and os.path.exists(output_path):
                clips.append(output_path)
                if on_progress:
                    on_progress(i + 1, total, entry.text[:30])
            else:
                # Try without hardware acceleration as fallback
                fallback_cmd = [FFMPEG, "-y",
                    "-ss", str(start),
                    "-i", video_path,
                    "-t", str(duration),
                    "-map", "0:v", "-map", "0:a?",
                    "-c:a", "copy",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                    "-avoid_negative_ts", "make_zero",
                    output_path,
                ]
                subprocess.run(fallback_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
                if os.path.exists(output_path):
                    clips.append(output_path)
        except subprocess.TimeoutExpired:
            print(f"  Timeout on segment {i+1}: {entry.text[:30]}")
        except Exception as e:
            print(f"  Error on segment {i+1}: {e}")

    print(f"Created {len(clips)} video clips in {output_dir}")
    return clips


def split_video_single_range(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    hw_accel: str = "auto",
) -> bool:
    """Cut a single time range from a video.

    Args:
        video_path: Source video
        start: Start time in seconds
        end: End time in seconds
        output_path: Output file path
        hw_accel: Hardware acceleration type

    Returns:
        True on success, False on failure.
    """
    hw_config = get_hw_accel_params(hw_accel)
    duration = end - start

    cmd = [FFMPEG, "-y"]
    cmd.extend(hw_config["hwaccel_in"])
    cmd.extend(["-ss", str(start)])
    cmd.extend(["-i", video_path])
    cmd.extend(["-t", str(duration)])
    cmd.extend(["-map", "0:v", "-map", "0:a?"])
    cmd.extend(["-c:a", "copy"])
    cmd.extend(["-c:v", hw_config["video_encoder"]])
    cmd.extend(hw_config["extra_flags"])
    cmd.extend(["-avoid_negative_ts", "make_zero"])
    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    return result.returncode == 0 and os.path.exists(output_path)


def filter_short_clips(
    clip_paths: list[str],
    min_duration: float = 2.0,
    on_progress=None,
) -> dict:
    """Delete clips shorter than min_duration seconds.

    Args:
        clip_paths: List of paths to video clip files
        min_duration: Minimum duration in seconds (clips shorter than this are deleted)
        on_progress: Optional callback(current, total)

    Returns:
        dict with keys: kept (list of remaining paths), deleted (count), total_before, total_after
    """
    kept = []
    deleted = 0
    total = len(clip_paths)

    for i, path in enumerate(clip_paths):
        if on_progress:
            on_progress(i + 1, total)

        if not os.path.exists(path):
            continue

        duration = get_video_duration(path)
        if duration <= 0:
            # Could not determine duration, keep the file
            kept.append(path)
            continue

        if duration < min_duration:
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                kept.append(path)  # Keep if can't delete
        else:
            kept.append(path)

    return {
        "kept": kept,
        "deleted": deleted,
        "total_before": total,
        "total_after": len(kept),
    }
