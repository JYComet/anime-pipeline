#!/usr/bin/env python3
"""
ASR Compare Tool — FireRed ASR2 vs Qwen3-ASR (API)

Compares two ASR models on a batch of audio files using the exact same
segmented comparison pipeline as the original ComicCut project.

Usage:
    python asr_compare.py                          # uses config.yaml
    python asr_compare.py -c config.local.yaml      # custom config
    python asr_compare.py --audio-dir ./my_audio    # override input dir
    python asr_compare.py --help
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path so asr_compare_lib is importable
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def setup_logging(level: str = "INFO", log_file: str = "asr_compare.log"):
    """Configure logging to both console and file."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Formatter
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File handler
    if log_file:
        log_path = os.path.join(_PROJECT_ROOT, log_file)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

    return root_logger


def load_config(config_path: str) -> dict:
    """Load YAML config, with optional local override."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check for local override
    local_path = config_path.replace(".yaml", ".local.yaml")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f)
        _deep_merge(config, local)

    return config


def _deep_merge(base: dict, override: dict):
    """Recursively merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def main():
    parser = argparse.ArgumentParser(
        description="ASR Compare Tool — FireRed ASR2 vs Qwen3-ASR (API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asr_compare.py
  python asr_compare.py -c config.local.yaml
  python asr_compare.py --audio-dir ./my_audio --output-dir ./my_output
  python asr_compare.py --no-delete --match-threshold 85
        """,
    )
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(_PROJECT_ROOT, "config.yaml"),
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--audio-dir",
        help="Override audio input directory from config",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config",
    )
    parser.add_argument(
        "--api-key",
        help="Override DashScope API key from config",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        help="Override match threshold (default: 90)",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Keep all results, do not auto-delete below-threshold audio",
    )
    parser.add_argument(
        "--language",
        help="Override language (zh, en, ja, auto)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda"],
        help="Device: only 'cuda' is supported",
    )
    parser.add_argument(
        "--hotwords",
        default="",
        help="Context words for ASR (comma-separated)",
    )
    parser.add_argument(
        "--no-segments",
        action="store_true",
        help="Do not keep segment-level WAV + TXT files",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (use plain text output)",
    )

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("Copy config.yaml and edit it, or use -c to specify a path.")
        sys.exit(1)

    config = load_config(args.config)

    # Apply CLI overrides
    if args.audio_dir:
        config["paths"]["audio_input_dir"] = args.audio_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.api_key:
        config["api"]["dashscope_api_key"] = args.api_key
    if args.match_threshold is not None:
        config["compare"]["match_threshold"] = args.match_threshold
    if args.no_delete:
        config["compare"]["delete_below_threshold"] = False
    if args.language:
        config["asr"]["language"] = args.language
    if args.device:
        config["asr"]["device"] = args.device
    if args.hotwords:
        config["asr"]["hotwords"] = args.hotwords
    if args.no_segments:
        config["compare"]["keep_segments"] = False

    # Setup logging
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file", "asr_compare.log")
    setup_logging(log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("ASR Compare Tool — Starting")
    logger.info("Time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Config: %s", args.config)

    # Resolve relative paths against project root
    for key in ["audio_input_dir", "output_dir", "firered_source_path", "firered_models_dir"]:
        p = config["paths"].get(key, "")
        if p and not os.path.isabs(p):
            config["paths"][key] = os.path.join(_PROJECT_ROOT, p)

    # ── GPU enforcement ──
    device = config["asr"]["device"]
    if device != "cuda":
        logger.error("Device must be 'cuda'. CPU is not supported.")
        sys.exit(1)
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA GPU not available. FireRed ASR2 requires a NVIDIA GPU.")
            sys.exit(1)
        logger.info("GPU: %s (VRAM: %.0f GB)", torch.cuda.get_device_name(0),
                   torch.cuda.get_device_properties(0).total_mem / (1024**3))
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    logger.info("Audio input dir : %s", config["paths"]["audio_input_dir"])
    logger.info("Output dir      : %s", config["paths"]["output_dir"])
    logger.info("Match threshold : %d%%", config["compare"]["match_threshold"])
    logger.info("Auto-delete     : %s", config["compare"]["delete_below_threshold"])
    logger.info("Language        : %s", config["asr"]["language"])
    logger.info("Device          : %s", device)
    logger.info("VAD engine      : %s", config["vad"]["engine"])
    logger.info("Filter English  : %s", config["compare"]["filter_english"])

    # Create output dir
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)

    # Run pipeline
    from asr_compare_lib.pipeline import ASRComparePipeline

    pipeline = ASRComparePipeline(config)

    try:
        summary = pipeline.run(use_progress=not args.no_progress)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        pipeline.cancel()
        summary = {"total": 0, "processed": 0, "kept": 0, "deleted": 0,
                   "failed": 0, "results": []}
    except Exception as e:
        _exc_name = type(e).__name__
        if _exc_name == "_CancelPipeline":
            logger.info("Pipeline cancelled")
        else:
            logger.error("Pipeline failed: %s", e, exc_info=True)
        summary = {"total": 0, "processed": 0, "kept": 0, "deleted": 0,
                   "failed": 0, "results": []}
        sys.exit(1)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Total files    : %d", summary["total"])
    logger.info("  Processed      : %d", summary["processed"])
    logger.info("  Kept           : %d", summary["kept"])
    logger.info("  Deleted        : %d", summary["deleted"])
    logger.info("  Failed         : %d", summary["failed"])
    logger.info("=" * 60)

    # Print per-file details
    for r in summary.get("results", []):
        if "error" in r:
            logger.info("  [FAIL] %s — %s", r["audio_name"], r["error"])
        else:
            flag = " [FLAGGED]" if r.get("overall_flagged") else ""
            logger.info("  [OK] %s — match=%.1f%%%s (%.1fs)",
                       r["audio_name"], r.get("overall_match_rate", 0),
                       flag, r.get("elapsed_sec", 0))

    logger.info("Log saved to: %s", os.path.join(_PROJECT_ROOT, log_file))


if __name__ == "__main__":
    main()
