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


def deduplicate_entries(entries: list[SubtitleEntry]) -> list[SubtitleEntry]:
    """Split overlapping subtitle entries into non-overlapping segments with unique text.

    Anime ASS subtitles often have accumulated on-screen text: each new
    dialogue line repeats previous text plus new content, and their time
    ranges overlap (e.g. entry A covers 0-2s with "A", entry B covers
    0-4s with "A B"). Without dedup this produces overlapping clips
    where clip 2's video includes clip 1's content.

    This function splits them so each sentence appears in exactly one clip:

    - Entries are sorted by start time, then by text length (shortest
      first) so the "base" sentence precedes its accumulated extensions.
    - When entry B's text starts with entry A's text, the shared prefix
      is stripped from B and B's start time is moved to A's end.
    - Fully contained entries (time and text both covered) are skipped.
    - Unrelated overlapping entries get start-time adjustment to avoid
      time overlap.

    The original entries are not modified — a new list is returned.
    """
    if not entries:
        return []

    # Sort by start time; for same start, shorter text first so that
    # the base sentence (e.g. "A") comes before its extension ("A B").
    sorted_entries = sorted(entries, key=lambda e: (e.start, len(e.text)))

    cleaned = []
    for entry in sorted_entries:
        if not cleaned:
            cleaned.append(SubtitleEntry(
                index=0, start=entry.start, end=entry.end,
                text=entry.text, style=entry.style,
            ))
            continue

        prev = cleaned[-1]

        # No time overlap — add as independent entry
        if entry.start >= prev.end:
            cleaned.append(SubtitleEntry(
                index=len(cleaned), start=entry.start, end=entry.end,
                text=entry.text, style=entry.style,
            ))
            continue

        # Fully contained within the previous entry's time range — skip
        if entry.start >= prev.start and entry.end <= prev.end:
            continue

        # Overlap — entry extends beyond prev.end.
        # Strip accumulated text prefix if present.
        text = entry.text
        if text.startswith(prev.text):
            suffix = text[len(prev.text):].strip()
            if not suffix:
                continue  # identical text after stripping prefix
            text = suffix

        # Push start forward to avoid time overlap with previous entry
        start = max(entry.start, prev.end)
        if start >= entry.end:
            continue

        cleaned.append(SubtitleEntry(
            index=len(cleaned), start=start, end=entry.end,
            text=text, style=entry.style,
        ))

    return cleaned


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
    deduplicate: bool = True,  # merge/remove overlapping entries
    on_progress=None,
    cancel_event=None,  # threading.Event to signal cancellation
) -> list[str]:
    """Split a video into clips based on subtitle timing.

    Each subtitle entry becomes a separate video clip.

    If deduplicate=True, overlapping entries are merged or skipped
    before splitting, avoiding duplicate/overlapping clips.

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
            hw_accel, min_duration, deduplicate, on_progress, cancel_event,
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
    deduplicate: bool = True,
    on_progress=None,
    cancel_event=None,
) -> list[str]:
    # Parse subtitles
    sub = parse_subtitle_file(subtitle_path)
    if not sub or not sub.entries:
        print(f"No subtitle entries found in {subtitle_path}")
        return []

    original_count = len(sub.entries)

    # Apply deduplication if requested
    if deduplicate:
        sub.entries = deduplicate_entries(sub.entries)
        removed = original_count - len(sub.entries)
        print(f"Deduplicated: {original_count} -> {len(sub.entries)} entries ({removed} removed/merged)")

    print(f"Parsed {len(sub.entries)} subtitle entries from {os.path.basename(subtitle_path)}")

    is_audio = video_path.lower().endswith('.wav')

    # Setup output
    if not output_dir:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        if is_audio:
            # For audio, default to sibling 'clips' folder
            parent_dir = os.path.dirname(os.path.dirname(video_path))
            output_dir = os.path.join(parent_dir, "clips", base_name)
        else:
            output_dir = os.path.join(CLIPS_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Get HW acceleration config (video only)
    hw_config = get_hw_accel_params(hw_accel) if not is_audio else {}
    hw_type = detect_hw_accel() if hw_accel == "auto" and not is_audio else hw_accel

    # Verify duration
    video_duration = get_video_duration(video_path)
    if video_duration <= 0:
        print(f"Warning: Could not determine duration for {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_ext = ".wav" if is_audio else ".mp4"

    clips = []
    total = len(sub.entries)

    for i, entry in enumerate(sub.entries):
        if cancel_event and cancel_event.is_set():
            print(f"[split] Cancelled after {len(clips)} clips")
            break

        start = max(0, entry.start - padding)
        end = min(video_duration or 999999, entry.end + padding)
        duration = end - start

        if duration < min_duration:
            continue

        safe_text = re.sub(r'[\\/*?:"<>|]', '', entry.text)[:20].strip()
        index_str = f"_S{i+1:03d}"
        output_name = f"{base_name}{index_str}_{safe_text}{output_ext}"
        output_path = os.path.join(output_dir, output_name)

        # Build ffmpeg command
        if is_audio:
            cmd = [FFMPEG, "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-map", "0:a",
                "-c:a", "pcm_s16le",
                output_path,
            ]
        else:
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

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=300,
            )
            if result.returncode == 0 and os.path.exists(output_path):
                clips.append(output_path)
                # Write individual SRT file for this segment
                srt_name = f"{base_name}{index_str}_{safe_text}.srt"
                srt_path = os.path.join(output_dir, srt_name)
                _write_single_srt(srt_path, i + 1, entry, start, end)
                if on_progress:
                    on_progress(i + 1, total, entry.text[:30])
            else:
                # Fallback
                if is_audio:
                    fb_cmd = [FFMPEG, "-y",
                        "-ss", str(start),
                        "-i", video_path,
                        "-t", str(duration),
                        "-map", "0:a",
                        "-c:a", "pcm_s16le",
                        output_path,
                    ]
                else:
                    fb_cmd = [FFMPEG, "-y",
                        "-ss", str(start),
                        "-i", video_path,
                        "-t", str(duration),
                        "-map", "0:v", "-map", "0:a?",
                        "-c:a", "copy",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                        "-avoid_negative_ts", "make_zero",
                        output_path,
                    ]
                subprocess.run(fb_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
                if os.path.exists(output_path):
                    clips.append(output_path)
                    srt_name = f"{base_name}{index_str}_{safe_text}.srt"
                    srt_path = os.path.join(output_dir, srt_name)
                    _write_single_srt(srt_path, i + 1, entry, start, end)
        except subprocess.TimeoutExpired:
            print(f"  Timeout on segment {i+1}: {entry.text[:30]}")
        except Exception as e:
            print(f"  Error on segment {i+1}: {e}")

    print(f"Created {len(clips)} clips in {output_dir}")
    return clips


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _write_single_srt(output_path: str, index: int, entry, clip_start: float, clip_end: float):
    """Write a single-segment SRT file for one clip."""
    relative_start = entry.start - clip_start
    relative_end = entry.end - clip_start
    if relative_start < 0:
        relative_start = 0.0
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{index}\n")
            f.write(f"{_format_srt_time(relative_start)} --> {_format_srt_time(relative_end)}\n")
            f.write(f"{entry.text}\n\n")
    except Exception as e:
        print(f"  Warning: failed to write SRT {output_path}: {e}")


def split_video_single_range(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    hw_accel: str = "auto",
) -> bool:
    """Cut a single time range from a video or audio file.

    Args:
        video_path: Source video or audio file
        start: Start time in seconds
        end: End time in seconds
        output_path: Output file path
        hw_accel: Hardware acceleration type (ignored for audio)

    Returns:
        True on success, False on failure.
    """
    is_audio = video_path.lower().endswith('.wav')
    duration = end - start

    if is_audio:
        cmd = [FFMPEG, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-map", "0:a",
            "-c:a", "pcm_s16le",
            output_path,
        ]
    else:
        hw_config = get_hw_accel_params(hw_accel)
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


def trim_video(
    video_path: str,
    start: float,
    end: float,
    output_dir: str = "",
    hw_accel: str = "auto",
) -> list[str]:
    """Trim a video to a specific time range, discarding everything outside.

    Args:
        video_path: Path to the source video file
        start: Start time in seconds (keep from this point)
        end: End time in seconds (keep until this point)
        output_dir: Where to write the trimmed clip
        hw_accel: Hardware acceleration type

    Returns:
        List containing the path to the trimmed clip, or empty list on failure.
    """
    _split_semaphore.acquire()
    try:
        if not output_dir:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(CLIPS_DIR, base_name)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # Format times for filename: 1m30s → 1m30s
        def _fmt(t):
            m = int(t // 60)
            s = t % 60
            if m > 0:
                return f"{m}m{int(s)}s" if s == int(s) else f"{m}m{s:.1f}s"
            return f"{int(s)}s" if s == int(s) else f"{s:.1f}s"

        output_ext = ".wav" if video_path.lower().endswith('.wav') else ".mp4"
        output_name = f"{base_name}_trim_{_fmt(start)}-{_fmt(end)}{output_ext}"
        output_path = os.path.join(output_dir, output_name)

        ok = split_video_single_range(video_path, start, end, output_path, hw_accel)
        return [output_path] if ok else []
    finally:
        _split_semaphore.release()


def split_video_by_duration(
    video_path: str,
    segment_duration: float,
    output_dir: str = "",
    hw_accel: str = "auto",
    output_ext: str = ".mp4",
    start_offset: float = 0.0,
    end_time: float = 0.0,
    on_progress=None,
    cancel_event=None,
) -> list[str]:
    """Split a video into equal-duration chunks.

    Each output segment is approximately segment_duration seconds long
    (the last segment may be shorter).

    Args:
        video_path: Path to the source video file
        segment_duration: Duration of each segment in seconds
        output_dir: Where to write the clips
        hw_accel: Hardware acceleration type (nvenc/amf/qsv/libx264/auto)
        output_ext: Output container format (default .mp4)
        start_offset: Start time in seconds (skip beginning of video)
        end_time: End time in seconds (0 = use full duration)
        on_progress: Optional callback(current, total)
        cancel_event: Optional threading.Event to abort

    Returns:
        List of paths to the created video clips.
    """
    # Acquire split semaphore
    _split_semaphore.acquire()
    try:
        return _split_video_by_duration_impl(
            video_path, segment_duration, output_dir, hw_accel,
            output_ext, start_offset, end_time, on_progress, cancel_event,
        )
    finally:
        _split_semaphore.release()


def _split_video_by_duration_impl(
    video_path: str,
    segment_duration: float,
    output_dir: str = "",
    hw_accel: str = "auto",
    output_ext: str = ".mp4",
    start_offset: float = 0.0,
    end_time: float = 0.0,
    on_progress=None,
    cancel_event=None,
) -> list[str]:
    if segment_duration <= 0:
        print(f"Invalid segment duration: {segment_duration}")
        return []

    is_audio = video_path.lower().endswith('.wav')
    if is_audio:
        output_ext = ".wav"

    video_duration = get_video_duration(video_path)
    if video_duration <= 0:
        print(f"Could not determine video duration for {video_path}")
        return []

    effective_end = min(end_time if end_time > 0 else video_duration, video_duration)
    effective_start = max(0, start_offset)
    total_cut_duration = effective_end - effective_start
    if total_cut_duration <= 0:
        print(f"Invalid time range: start={effective_start} end={effective_end}")
        return []

    total_segments = int(math.ceil(total_cut_duration / segment_duration))

    # Setup output
    if not output_dir:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(CLIPS_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    hw_config = get_hw_accel_params(hw_accel)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    clips = []
    for i in range(total_segments):
        if cancel_event and cancel_event.is_set():
            print(f"[split-duration] Cancelled after {len(clips)} clips")
            break

        seg_start = effective_start + i * segment_duration
        seg_end = min(seg_start + segment_duration, effective_end)
        seg_dur = seg_end - seg_start

        if seg_dur < 0.5:
            continue

        output_name = f"{base_name}_D{i+1:04d}{output_ext}"
        output_path = os.path.join(output_dir, output_name)

        if is_audio:
            cmd = [FFMPEG, "-y",
                "-ss", str(seg_start),
                "-i", video_path,
                "-t", str(seg_dur),
                "-map", "0:a",
                "-c:a", "pcm_s16le",
                output_path,
            ]
        else:
            cmd = [FFMPEG, "-y"]
            cmd.extend(hw_config["hwaccel_in"])
            cmd.extend(["-ss", str(seg_start)])
            cmd.extend(["-i", video_path])
            cmd.extend(["-t", str(seg_dur)])
            cmd.extend(["-map", "0:v", "-map", "0:a?"])
            cmd.extend(["-c:a", "copy"])
            cmd.extend(["-c:v", hw_config["video_encoder"]])
            cmd.extend(hw_config["extra_flags"])
            cmd.extend(["-avoid_negative_ts", "make_zero"])
            cmd.append(output_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=300,
            )
            if result.returncode == 0 and os.path.exists(output_path):
                clips.append(output_path)
                if on_progress:
                    on_progress(i + 1, total_segments)
            else:
                # Fallback: stream copy
                if is_audio:
                    fb_cmd = [FFMPEG, "-y",
                        "-ss", str(seg_start),
                        "-i", video_path,
                        "-t", str(seg_dur),
                        "-map", "0:a",
                        "-c:a", "pcm_s16le",
                        output_path,
                    ]
                else:
                    fb_cmd = [FFMPEG, "-y",
                        "-ss", str(seg_start),
                        "-i", video_path,
                        "-t", str(seg_dur),
                        "-map", "0:v", "-map", "0:a?",
                        "-c", "copy",
                        "-avoid_negative_ts", "make_zero",
                        output_path,
                    ]
                subprocess.run(fb_cmd, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', timeout=300)
                if os.path.exists(output_path):
                    clips.append(output_path)
        except subprocess.TimeoutExpired:
            print(f"  Timeout on segment {i+1}")
        except Exception as e:
            print(f"  Error on segment {i+1}: {e}")

    print(f"Created {len(clips)} duration-based clips in {output_dir}")
    return clips


def split_video_by_size(
    video_path: str,
    target_size_mb: float,
    output_dir: str = "",
    hw_accel: str = "auto",
    output_ext: str = ".mp4",
    on_progress=None,
    cancel_event=None,
) -> list[str]:
    """Split a video into chunks of approximately target_size_mb each.

    Calculates the video bitrate and determines the duration per segment
    needed to hit the target file size. Final sizes may vary slightly
    depending on content complexity.

    Args:
        video_path: Path to the source video file
        target_size_mb: Approximate target size per segment in megabytes
        output_dir: Where to write the clips
        hw_accel: Hardware acceleration type
        output_ext: Output container format (default .mp4)
        on_progress: Optional callback(current, total)
        cancel_event: Optional threading.Event to abort

    Returns:
        List of paths to the created video clips.
    """
    _split_semaphore.acquire()
    try:
        return _split_video_by_size_impl(
            video_path, target_size_mb, output_dir, hw_accel,
            output_ext, on_progress, cancel_event,
        )
    finally:
        _split_semaphore.release()


def _split_video_by_size_impl(
    video_path: str,
    target_size_mb: float,
    output_dir: str = "",
    hw_accel: str = "auto",
    output_ext: str = ".mp4",
    on_progress=None,
    cancel_event=None,
) -> list[str]:
    if target_size_mb <= 0:
        print(f"Invalid target size: {target_size_mb} MB")
        return []

    # Get video bitrate using ffprobe
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
        )
        if result.returncode != 0:
            print(f"ffprobe failed for {video_path}")
            return []

        import json
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        video_bitrate = int(fmt.get("bit_rate", 0))
        video_duration = float(fmt.get("duration", 0))

        if video_duration <= 0:
            print(f"Could not determine video duration")
            return []
        if video_bitrate <= 0:
            # Estimate: file_size / duration
            file_size_bytes = os.path.getsize(video_path)
            video_bitrate = int((file_size_bytes * 8) / video_duration)
            if video_bitrate <= 0:
                print(f"Could not estimate video bitrate")
                return []

        # Calculate segment duration: target_size (in bits) / bitrate
        target_size_bits = target_size_mb * 8 * 1024 * 1024
        segment_duration = target_size_bits / video_bitrate

        # Cap segment duration at video duration
        segment_duration = min(segment_duration, video_duration * 0.95)
        if segment_duration < 1.0:
            segment_duration = 1.0  # minimum 1 second

        print(f"Video: {video_duration:.1f}s, bitrate: {video_bitrate/1000:.0f}kbps, "
              f"target {target_size_mb}MB -> segment {segment_duration:.1f}s")

        return _split_video_by_duration_impl(
            video_path=video_path,
            segment_duration=segment_duration,
            output_dir=output_dir,
            hw_accel=hw_accel,
            output_ext=output_ext,
            start_offset=0.0,
            end_time=0.0,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )
    except Exception as e:
        print(f"Error in split_video_by_size: {e}")
        return []
