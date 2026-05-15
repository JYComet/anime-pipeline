"""
MP4 to WAV audio conversion using ffmpeg.
"""
import os
import subprocess
from config import FFMPEG


def mp4_to_wav(mp4_path: str, wav_path: str = "") -> str:
    """Extract audio from an MP4 file and save as 16-bit PCM WAV.

    Args:
        mp4_path: Path to the source MP4 file.
        wav_path: Optional output path. If empty, replaces .mp4 extension with .wav.

    Returns:
        Path to the created WAV file, or empty string on failure.
    """
    if not wav_path:
        wav_path = os.path.splitext(mp4_path)[0] + ".wav"

    if os.path.exists(wav_path):
        return wav_path  # already converted

    cmd = [
        FFMPEG, "-y",
        "-i", mp4_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM WAV
        "-ar", "44100",           # 44.1kHz sample rate
        "-ac", "2",               # stereo
        wav_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path

    # Log stderr on failure
    if result.stderr:
        print(f"[convert_audio] ffmpeg error: {result.stderr[:300]}")
    return ""


def batch_convert(directory: str) -> list[str]:
    """Convert all MP4 files in a directory to WAV.

    Returns list of created WAV paths.
    """
    wavs = []
    if not os.path.isdir(directory):
        return wavs
    for f in sorted(os.listdir(directory)):
        if not f.lower().endswith(".mp4"):
            continue
        mp4 = os.path.join(directory, f)
        wav = mp4_to_wav(mp4)
        if wav:
            wavs.append(wav)
    return wavs
