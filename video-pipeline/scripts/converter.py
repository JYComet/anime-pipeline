"""
Audio/video to WAV conversion using ffmpeg.
Extracted from the original Anime Pipeline convert_audio.py and server.py.
"""
import os
import subprocess

from .config_loader import get_tool, get_processing


def convert_to_wav(input_path, output_path="", sample_rate=None, channels=None, logger=None):
    """Convert any media file to 16-bit PCM WAV.

    For video files: extracts audio track via ``-vn``.
    For audio files: transcodes to WAV.

    Args:
        input_path: Path to source file (video or audio).
        output_path: Optional output path. Auto-generated if empty.
        sample_rate: Output sample rate Hz. Uses config default if None.
        channels: Output channels (1=mono, 2=stereo). Uses config default if None.
        logger: Optional logger instance.

    Returns:
        Path to created WAV file, or empty string on failure.
    """
    proc = get_processing()
    if sample_rate is None:
        sample_rate = proc.get("sample_rate", 32000)
    if channels is None:
        channels = proc.get("channels", 1)

    if not output_path:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".wav"

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        if logger:
            logger.info(f"  [转换] 跳过，已存在: {os.path.basename(output_path)}")
        return output_path

    ffmpeg = get_tool("ffmpeg")
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_path,
    ]

    log = logger.info if logger else print

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            log(f"  [转换] 完成: {os.path.basename(output_path)} ({sample_rate}Hz/{'单声道' if channels == 1 else '立体声'})")
            return output_path
        if result.stderr:
            stderr_msg = result.stderr.strip()[-200:]
            if logger:
                logger.warning(f"  [转换] ffmpeg stderr: {stderr_msg}")
            else:
                print(f"  [转换] ffmpeg stderr: {stderr_msg}")
        return ""
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"  [转换] 超时: {os.path.basename(input_path)}")
        return ""
    except Exception as e:
        if logger:
            logger.error(f"  [转换] 失败: {os.path.basename(input_path)} — {e}")
        return ""


def batch_convert_segments(segments, seg_dir, base_name, sample_rate, channels, ffmpeg_sem, logger):
    """Convert non-WAV segments to WAV in parallel using threads.

    Args:
        segments: List of segment file paths.
        seg_dir: Directory containing segments.
        base_name: Base filename for naming converted segments.
        sample_rate: Target sample rate.
        channels: Target channels.
        ffmpeg_sem: threading.BoundedSemaphore for ffmpeg concurrency control.
        logger: Logger instance.

    Returns:
        List of WAV segment paths (some may be empty string on failure).
    """
    import concurrent.futures

    ffmpeg_path = get_tool("ffmpeg")

    def _convert_one(idx, seg_path):
        if not os.path.exists(seg_path):
            return ""
        wav_out = os.path.join(seg_dir, f"{base_name}_seg{idx:03d}.wav")
        if os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
            return wav_out
        try:
            with ffmpeg_sem:
                subprocess.run(
                    [ffmpeg_path, "-y", "-i", seg_path, "-vn",
                     "-acodec", "pcm_s16le",
                     "-ar", str(sample_rate), "-ac", str(channels),
                     wav_out],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=120,
                )
            if os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
                try:
                    os.remove(seg_path)  # clean up original-codec segment
                except OSError:
                    pass
                return wav_out
            return ""
        except Exception as e:
            logger.warning(f"  [转换] 片段 {idx} 失败: {e}")
            return ""

    workers = min(len(segments), 8)
    logger.info(f"  [转换] 并行转换 {len(segments)} 段 → {sample_rate}Hz 单声道 (workers={workers})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_convert_one, i, p) for i, p in enumerate(segments)]

    result_map = {}
    for i, fut in enumerate(futures):
        result_map[i] = fut.result()

    return [result_map[i] for i in sorted(result_map)]
