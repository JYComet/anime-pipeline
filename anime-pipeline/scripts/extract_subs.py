"""
Subtitle extraction module.
Uses mkvextract from mkvtoolnix to extract subtitle tracks from MKV files.
"""
import os
import re
import time
import subprocess
import json
import threading
from typing import Optional

from config import MKVEXTRACT, MKVINFO, MKVMERGE, FFMPEG, SUBTITLE_DIR, TEMP_DIR


def get_mkv_tracks(mkv_path: str) -> list[dict]:
    """Get all track information from an MKV file using mkvmerge -J (JSON output).

    Returns list of tracks: {id, type, codec, language, name, codec_id}
    """
    cmd = [MKVMERGE, "-J", mkv_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=60)
        if result.returncode != 0:
            print(f"mkvmerge error: {result.stderr}")
            return []

        data = json.loads(result.stdout)
        tracks = []
        for t in data.get("tracks", []):
            props = t.get("properties", {})
            tracks.append({
                "id": t.get("id", 0),
                "type": t.get("type", ""),
                "codec": t.get("codec", ""),
                "codec_id": props.get("codec_id", ""),
                "language": props.get("language", ""),
                "language_ietf": props.get("language_ietf", ""),
                "name": props.get("track_name", ""),
            })
        return tracks
    except Exception as e:
        print(f"Failed to get MKV tracks: {e}")
        return []


def find_subtitle_tracks(tracks: list[dict], language: str = "") -> list[dict]:
    """Filter tracks to find subtitle tracks, optionally by language.

    Args:
        tracks: List of track dicts from mkvinfo
        language: Language code to filter (chi, zho, zh, ja, etc.).
                  Empty = all subtitle tracks.

    Chinese language codes commonly found in anime:
      - chi (Chinese)
      - zho (Chinese, ISO 639-2)
      - zh / zh-CN / zh-TW
    """
    subtitle_tracks = []
    chinese_codes = {"chi", "zho", "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant",
                     "chinese", "cn", "tc", "sc"}

    for track in tracks:
        if track.get("type") != "subtitles":
            continue

        track_lang = track.get("language", "").lower()
        track_name = track.get("name", "").lower()

        if not language:
            subtitle_tracks.append(track)
        elif language.lower() in chinese_codes:
            # Match any Chinese variant
            if track_lang in chinese_codes:
                subtitle_tracks.append(track)
            elif any(c in track_lang for c in ["chi", "zho", "zh", "cn"]):
                subtitle_tracks.append(track)
            elif any(c in track_name for c in ["中文", "简体", "繁體", "chs", "cht", "sc", "tc", "chi", "zho", "zh"]):
                subtitle_tracks.append(track)
        elif track_lang.startswith(language.lower()):
            subtitle_tracks.append(track)

    return subtitle_tracks


def extract_subtitle_track(
    mkv_path: str,
    track_id: int,
    output_path: str = "",
    cancel_event=None,
) -> Optional[str]:
    """Extract a single subtitle track from an MKV file using ffmpeg.

    Args:
        mkv_path: Path to the MKV file
        track_id: Track number (1-based, as shown in mkvinfo/mkvmerge)
        output_path: Where to write the subtitle file.
        cancel_event: Optional threading.Event — checked before extraction.

    Returns:
        Path to the extracted subtitle file, or None on failure.
    """
    if cancel_event and cancel_event.is_set():
        return None

    if not output_path:
        base = os.path.splitext(os.path.basename(mkv_path))[0]
        output_path = os.path.join(SUBTITLE_DIR, f"{base}_track{track_id}.ass")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try mkvextract first (instant for subtitles)
    proc = subprocess.run(
        [MKVEXTRACT, "tracks", mkv_path, f"{track_id}:{output_path}"],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=30,
    )
    if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    # Fallback to ffmpeg
    if cancel_event and cancel_event.is_set():
        return None

    # ffmpeg subtitle stream index is 0-based
    stream_idx = 0
    tracks = get_mkv_tracks(mkv_path)
    sub_count = 0
    for t in tracks:
        if t.get("type") != "subtitles":
            continue
        if t.get("id") == track_id:
            stream_idx = sub_count
            break
        sub_count += 1

    proc2 = subprocess.run(
        [FFMPEG, "-y", "-i", mkv_path, "-map", f"0:s:{stream_idx}", "-c:s", "copy", output_path],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=60,
    )
    if proc2.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    print(f"Subtitle extraction failed for track {track_id}")
    return None


def extract_all_chinese_subs(mkv_path: str, cancel_event=None) -> list[str]:
    """Extract Simplified Chinese subtitle tracks from an MKV file.

    Prefers zh-Hans (Simplified Chinese), skips zh-Hant (Traditional).
    If no embedded tracks found, checks for external SRT/ASS files.
    Returns list of paths to extracted subtitle files.
    """
    tracks = get_mkv_tracks(mkv_path)
    sub_tracks = find_subtitle_tracks(tracks, language="chi")

    if not sub_tracks:
        sub_tracks = find_subtitle_tracks(tracks, language="zho")
    if not sub_tracks:
        sub_tracks = find_subtitle_tracks(tracks, language="")

    # Prefer Simplified Chinese over Traditional
    simplified = []
    traditional = []
    for t in sub_tracks:
        ietf = t.get("language_ietf", "").lower()
        name = t.get("name", "").lower()
        if "hans" in ietf or any(kw in name for kw in ["简", "chs", "sc", "gb", "cn", "简体"]):
            simplified.append(t)
        elif "hant" in ietf or any(kw in name for kw in ["繁", "cht", "tc", "big5", "hk", "tw", "繁體"]):
            traditional.append(t)
        else:
            simplified.append(t)

    chosen = simplified if simplified else sub_tracks
    extracted = []

    if chosen:
        for track in chosen:
            track_id = track.get("id", 0)
            codec = track.get("codec", "")
            codec_id = track.get("codec_id", "")
            if any(c in codec.lower() + codec_id.lower()
                   for c in ["ass", "ssa", "substation"]):
                ext = ".ass"
            elif "subrip" in (codec.lower() + codec_id.lower()):
                ext = ".srt"
            else:
                ext = ".ass"

            base = os.path.splitext(os.path.basename(mkv_path))[0]
            lang = track.get("language", "unknown")
            output = os.path.join(SUBTITLE_DIR, f"{base}_track{track_id}_{lang}{ext}")

            result = extract_subtitle_track(mkv_path, track_id, output, cancel_event=cancel_event)
            if result:
                extracted.append(result)
                print(f"  Extracted subtitle: {os.path.basename(result)}")

    # Fallback: check for external subtitle files next to the MKV
    if not extracted:
        import shutil
        base_no_ext = os.path.splitext(mkv_path)[0]
        mkv_dir = os.path.dirname(mkv_path)
        mkv_basename = os.path.basename(base_no_ext)

        # Look for same-name SRT/ASS files in same directory
        for ext in [".srt", ".ass", ".ssa"]:
            # Check exact name match
            candidate = base_no_ext + ext
            if os.path.exists(candidate):
                dest = os.path.join(SUBTITLE_DIR, os.path.basename(candidate))
                if not os.path.exists(dest):
                    os.makedirs(SUBTITLE_DIR, exist_ok=True)
                    shutil.copy2(candidate, dest)
                extracted.append(dest)
                print(f"  Using external subtitle: {os.path.basename(candidate)}")
                break

    return extracted


def list_all_tracks(mkv_path: str) -> list[dict]:
    """List all tracks in human-readable format. Useful for debugging."""
    tracks = get_mkv_tracks(mkv_path)
    info = []
    for t in tracks:
        info.append({
            "id": t.get("id"),
            "type": t.get("type"),
            "codec": t.get("codec"),
            "language": t.get("language", "unknown"),
            "name": t.get("name", ""),
        })
    return info
