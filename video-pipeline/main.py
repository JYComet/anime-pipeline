#!/usr/bin/env python3
"""
Video Pipeline — 视频管线独立脚本
====================================
从 Anime Pipeline 原项目中提取的独立视频处理管线。

功能:
  1. 时长切分 (duration_split) — 将长视频/音频按固定时长切分为片段
  2. WAV 转换 (convert) — 统一转换为 32kHz 单声道 16-bit PCM WAV
  3. BGM 分离 (music_separate) — 使用 Demucs htdemucs_ft 分离人声

用法:
  python main.py                          # 处理 config.yaml 中 input_dir 下的所有文件
  python main.py -i /path/to/input        # 处理指定目录
  python main.py -i /path/to/file.mp4     # 处理单个文件
  python main.py -c /path/to/config.yaml  # 使用指定配置文件
  python main.py --dry-run                # 预览模式，只显示将要处理的文件
"""
import os
import sys
import time
import signal
import logging
import argparse
import shutil
import threading
import concurrent.futures

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scripts.config_loader import (
    load_config, get_path, get_processing, get_model_pool, get_concurrency, get_device, get_logging,
)
from scripts.splitter import split_by_duration, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS
from scripts.converter import batch_convert_segments
from scripts.bgm_separator import separate_vocals, get_pool_size, shutdown_pool

# ============================================================
# Global state
# ============================================================
_shutdown_requested = False
_stats = {
    "total_files": 0,
    "completed_files": 0,
    "error_files": 0,
    "total_segments": 0,
    "completed_segments": 0,
    "error_segments": 0,
    "start_time": 0,
}
_stats_lock = threading.Lock()

# Logger
logger = logging.getLogger("video-pipeline")


def setup_logging():
    """Configure logging based on config.yaml settings."""
    log_cfg = get_logging()
    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    logger.setLevel(logging.DEBUG)  # Root logger captures all; handlers filter

    # Console handler — terse, real-time
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # File handler — detailed
    if log_cfg.get("file_log", True):
        log_dir = get_path("log_dir")
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{ts}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_level_name = log_cfg.get("file_log_level", "DEBUG")
        file_handler.setLevel(getattr(logging, file_level_name.upper(), logging.DEBUG))
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")

        # Clean old logs
        max_logs = log_cfg.get("max_log_files", 10)
        try:
            log_files = sorted([
                f for f in os.listdir(log_dir) if f.startswith("pipeline_") and f.endswith(".log")
            ])
            for old in log_files[:-max_logs]:
                os.remove(os.path.join(log_dir, old))
        except Exception:
            pass


def print_header():
    """Print a startup banner with current configuration."""
    proc = get_processing()
    pool = get_model_pool()
    device = get_device()

    lines = [
        "=" * 70,
        "  Video Pipeline — 视频管线处理脚本",
        "=" * 70,
        f"  输入目录 : {get_path('input_dir')}",
        f"  输出目录 : {get_path('output_dir')}",
        f"  临时目录 : {get_path('temp_dir')}",
        f"  日志目录 : {get_path('log_dir')}",
        f"  切分时长 : {proc.get('segment_duration', 600)} 秒/段",
        f"  保留首尾 : {'是' if proc.get('keep_ends') else '否（自动去除）'}",
        f"  输出格式 : {proc.get('sample_rate', 32000)}Hz / {'立体声' if proc.get('channels') == 2 else '单声道'}",
        f"  Demucs实例: {pool.get('demucs_instances', 2)}",
        f"  运行设备 : {device.upper()}",
        "-" * 70,
    ]
    for line in lines:
        logger.info(line)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("再次收到中断信号，强制退出...")
        shutdown_pool()
        os._exit(1)
    _shutdown_requested = True
    logger.info("\n收到中断信号 (Ctrl+C)，正在优雅退出...")
    logger.info("等待当前任务完成（再次按 Ctrl+C 强制退出）...")


def scan_files(input_path):
    """Scan for supported media files.

    Args:
        input_path: File path or directory path.

    Returns:
        List of dicts: [{name, path, size_mb, duration_s}]
    """
    from scripts.splitter import get_media_info

    files = []

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS or ext in AUDIO_EXTENSIONS:
            info = get_media_info(input_path)
            files.append({
                "name": os.path.basename(input_path),
                "path": input_path,
                "size_mb": info.get("size_mb", 0),
                "duration_s": info.get("duration_s", 0),
            })
        else:
            logger.warning(f"不支持的文件格式: {input_path}")
    elif os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            fpath = os.path.join(input_path, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in VIDEO_EXTENSIONS or ext in AUDIO_EXTENSIONS:
                info = get_media_info(fpath)
                files.append({
                    "name": fname,
                    "path": fpath,
                    "size_mb": info.get("size_mb", 0),
                    "duration_s": info.get("duration_s", 0),
                })
    else:
        logger.error(f"路径不存在: {input_path}")

    return files


def format_duration(seconds):
    """Format seconds to H:MM:SS."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def process_file(file_info, semaphores, temp_dir, output_dir):
    """Process a single file through the full pipeline.

    Pipeline: duration_split -> convert segments -> BGM separate each segment.

    Args:
        file_info: dict with name, path, etc.
        semaphores: dict of threading.BoundedSemaphore for concurrency control.
        temp_dir: Temporary directory for intermediate files.
        output_dir: Final output directory.

    Returns:
        True if successful, False on error.
    """
    global _shutdown_requested

    fname = file_info["name"]
    fpath = file_info["path"]
    base_name = os.path.splitext(fname)[0]

    logger.info(f"[{fname}] 开始处理")

    t0 = time.time()

    # Per-file temp directory
    file_temp = os.path.join(temp_dir, base_name)
    seg_dir = os.path.join(file_temp, "segments")
    os.makedirs(seg_dir, exist_ok=True)

    try:
        # ============================================================
        # Step 1: Duration Split
        # ============================================================
        if _shutdown_requested:
            logger.info(f"[{fname}] 已取消")
            return False

        with semaphores["split"]:
            logger.info(f"[{fname}] 步骤 1/3: 时长切分...")
            split_result = split_by_duration(fpath, seg_dir, base_name, logger)

        if split_result["error"]:
            logger.error(f"[{fname}] 切分失败: {split_result['error']}")
            return False

        segments = split_result["segments"]
        if not segments:
            logger.warning(f"[{fname}] 切分未产生任何片段")
            return False

        with _stats_lock:
            _stats["total_segments"] += len(segments)

        logger.info(f"[{fname}] 切分完成: {len(segments)} 段")

        # ============================================================
        # Step 2: Convert segments to WAV (if needed)
        # ============================================================
        if _shutdown_requested:
            return False

        if split_result["needs_convert"]:
            logger.info(f"[{fname}] 步骤 2/3: WAV 转换...")
            opt_sr = split_result["sample_rate"]
            opt_ch = split_result["channels"]
            converted = batch_convert_segments(
                segments, seg_dir, base_name,
                opt_sr, opt_ch, semaphores["ffmpeg"], logger,
            )
            # Filter out failed conversions
            segments = [c for c in converted if c]
            if not segments:
                logger.error(f"[{fname}] 所有片段转换失败")
                return False
            logger.info(f"[{fname}] 转换完成: {len(segments)} 段")
        else:
            logger.info(f"[{fname}] 步骤 2/3: WAV 转换 (跳过，已是 WAV 格式)")

        # ============================================================
        # Step 3: BGM Separation (per segment, parallel)
        # ============================================================
        if _shutdown_requested:
            return False

        seg_count = len(segments)
        pool_size = get_pool_size()
        step_workers = min(seg_count, pool_size)

        logger.info(f"[{fname}] 步骤 3/3: BGM 分离 ({seg_count} 段, {step_workers} 并行)...")

        voc_seg_dir = os.path.join(file_temp, "vocals_segments")
        os.makedirs(voc_seg_dir, exist_ok=True)

        seg_error = [False]
        seg_done_count = [0]
        seg_lock = threading.Lock()

        def _bgm_one(idx, seg_path):
            if _shutdown_requested or seg_error[0]:
                return
            result = separate_vocals(seg_path, voc_seg_dir, logger)
            with seg_lock:
                if result and os.path.exists(result):
                    seg_done_count[0] += 1
                else:
                    seg_error[0] = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=step_workers) as executor:
            futures = [executor.submit(_bgm_one, i, p) for i, p in enumerate(segments)]
            while futures:
                done, futures = concurrent.futures.wait(
                    futures, timeout=2,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if _shutdown_requested:
                    for f in futures:
                        f.cancel()
                    break

        if seg_error[0] and seg_done_count[0] == 0:
            logger.error(f"[{fname}] BGM 分离全部失败")
            return False
        elif seg_error[0]:
            logger.warning(f"[{fname}] BGM 分离: {seg_done_count[0]}/{seg_count} 段成功, 部分失败")

        # ============================================================
        # Publish results to output directory
        # ============================================================
        file_output = os.path.join(output_dir, base_name)
        os.makedirs(file_output, exist_ok=True)

        published = 0
        for f in sorted(os.listdir(voc_seg_dir)):
            if f.endswith("_vocals.wav"):
                src = os.path.join(voc_seg_dir, f)
                dst = os.path.join(file_output, f)
                try:
                    shutil.copy2(src, dst)
                    published += 1
                except Exception as e:
                    logger.warning(f"[{fname}] 复制失败: {f} — {e}")

        # Clean temp
        try:
            shutil.rmtree(file_temp, ignore_errors=True)
        except Exception:
            pass

        elapsed = time.time() - t0
        logger.info(f"[{fname}] 完成! {published} 个人声文件 (耗时 {format_duration(elapsed)})")

        with _stats_lock:
            _stats["completed_files"] += 1

        return True

    except Exception:
        import traceback
        logger.error(f"[{fname}] 处理异常:\n{traceback.format_exc()}")
        with _stats_lock:
            _stats["error_files"] += 1
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video Pipeline — 视频管线处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 使用默认配置处理 input_dir
  python main.py -i ./my_videos           # 处理指定目录
  python main.py -i ./video.mp4           # 处理单个文件
  python main.py -c my_config.yaml        # 使用自定义配置
  python main.py --dry-run                # 预览模式
        """,
    )
    parser.add_argument("-i", "--input", help="输入文件或目录 (覆盖配置文件中的 input_dir)")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，只显示将要处理的文件")
    args = parser.parse_args()

    # Load config
    config_path = args.config
    load_config(config_path)

    # Setup logging
    setup_logging()

    # Resolve input
    if args.input:
        input_path = args.input
        if not os.path.isabs(input_path):
            input_path = os.path.abspath(input_path)
    else:
        input_path = get_path("input_dir")

    if not os.path.exists(input_path):
        logger.error(f"输入路径不存在: {input_path}")
        sys.exit(1)

    # Scan files
    logger.info("扫描输入文件...")
    files = scan_files(input_path)
    if not files:
        logger.error(f"未找到支持的媒体文件: {input_path}")
        logger.info(f"支持的格式: 视频 — {', '.join(sorted(VIDEO_EXTENSIONS))}")
        logger.info(f"             音频 — {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    # Print file list
    total_dur = sum(f.get("duration_s", 0) for f in files)
    logger.info(f"找到 {len(files)} 个文件 (总时长: {format_duration(total_dur)})")
    for i, f in enumerate(files):
        dur = format_duration(f.get("duration_s", 0))
        size = f"{f.get('size_mb', 0):.0f} MB"
        logger.info(f"  {i + 1}. {f['name']}  [{dur}, {size}]")

    if args.dry_run:
        logger.info("[预览模式] 不执行实际处理。移除 --dry-run 以开始处理。")
        sys.exit(0)

    # Print config header
    print_header()

    # GPU check
    device = get_device()
    if device != "cpu":
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA GPU 不可用！BGM 分离需要 GPU。")
            logger.error("请将 config.yaml 中 device 设为 'cpu'，或安装 CUDA 版本的 PyTorch。")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Confirm start
    print()
    response = input("按 Enter 开始处理，或输入 q 退出: ").strip().lower()
    if response == "q":
        logger.info("用户取消")
        sys.exit(0)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Prepare directories
    output_dir = get_path("output_dir")
    temp_dir = get_path("temp_dir")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Concurrency semaphores
    conc = get_concurrency()
    semaphores = {
        "ffmpeg": threading.BoundedSemaphore(conc.get("ffmpeg_max", 8)),
        "split": threading.BoundedSemaphore(conc.get("split_max", 6)),
    }
    pool_size = get_pool_size()

    # Init stats
    with _stats_lock:
        _stats["total_files"] = len(files)
        _stats["start_time"] = time.time()

    logger.info("=" * 70)
    logger.info("开始处理...")
    logger.info(f"并发设置: ffmpeg={conc.get('ffmpeg_max', 8)}, "
                f"split={conc.get('split_max', 6)}, "
                f"BGM实例池={pool_size}")

    # Pre-load Demucs model pool
    from scripts.bgm_separator import _init_pool
    _init_pool()

    # Process files sequentially (each file's segments run in parallel internally)
    for f in files:
        if _shutdown_requested:
            logger.info("处理已取消")
            break

        logger.info("-" * 50)
        try:
            process_file(f, semaphores, temp_dir, output_dir)
        except Exception:
            import traceback
            logger.error(f"[{f['name']}] 致命错误:\n{traceback.format_exc()}")
            with _stats_lock:
                _stats["error_files"] += 1

    # Summary
    shutdown_pool()

    with _stats_lock:
        total = _stats["total_files"]
        done = _stats["completed_files"]
        err = _stats["error_files"]
        elapsed = time.time() - _stats["start_time"]

    logger.info("=" * 70)
    logger.info("处理完成!")
    logger.info(f"  文件: {total} 个 ({done} 成功, {err} 失败)")
    logger.info(f"  总耗时: {format_duration(elapsed)}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 70)

    if _shutdown_requested:
        logger.info("(用户中断)")


if __name__ == "__main__":
    main()
