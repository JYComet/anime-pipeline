#!/usr/bin/env python3
"""
Export FireRedASR2-AED Conformer encoder to ONNX and TensorRT engine.

The encoder accounts for ~70% of FireRed's inference time. Converting it to
TensorRT gives 2-3x speedup on the encoder alone, and ~1.5-2x end-to-end.

Requirements:
    pip install tensorrt onnx
    # TensorRT must match your CUDA version (cu121/cu124/cu126/...)

Usage:
    python export_firered_trt.py \
        --model-dir models/firered_asr2/FireRedASR2-AED \
        --output-dir models/firered_asr2/FireRedASR2-AED-TRT

This creates:
    models/firered_asr2/FireRedASR2-AED-TRT/
        encoder.plan          # TensorRT engine for the Conformer encoder
        cmvn.ark              # copied from source model dir
        dict.txt              # copied from source model dir
        train_bpe1000.model   # copied from source model dir
        model.pth.tar         # copied (decoder weights still needed)
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# FireRed source is bundled in the project
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIRERED_PATH = os.path.join(_PROJECT_DIR, "FireRedASR2S")
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
if _FIRERED_PATH not in sys.path:
    sys.path.insert(0, _FIRERED_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_firered_trt")


def get_parser():
    p = argparse.ArgumentParser(
        description="Export FireRedASR2-AED encoder to TensorRT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True,
                   help="Path to FireRedASR2-AED model directory (contains model.pth.tar)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for TensorRT engine and model files")
    p.add_argument("--opset-version", type=int, default=17,
                   help="ONNX opset version")
    p.add_argument("--opt-batch-size", type=int, default=16,
                   help="Optimization batch size for TRT profile")
    p.add_argument("--max-batch-size", type=int, default=64,
                   help="Max batch size for TRT profile")
    p.add_argument("--keep-onnx", action="store_true",
                   help="Keep intermediate ONNX file")
    return p


def load_encoder(model_dir: str):
    """Load the FireRedASR2-AED encoder from checkpoint."""
    from fireredasr2s.fireredasr2.asr import load_fireredasr_aed_model

    model_path = os.path.join(model_dir, "model.pth.tar")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint once - load_fireredasr_aed_model also calls torch.load internally,
    # but we need idim from the package, so just load explicitly here to avoid double-load
    logger.info("Loading checkpoint (this may take a while for large files on network storage)...")
    pkg = torch.load(model_path, map_location="cpu", weights_only=False)
    idim = pkg["args"].idim
    logger.info("Checkpoint loaded. Building model...")

    model = load_fireredasr_aed_model(model_path)
    encoder = model.encoder
    encoder.eval()
    encoder.half()  # fp16 for TensorRT
    logger.info("Encoder ready. idim=%d, params=%.1fM",
                idim, sum(p.numel() for p in encoder.parameters()) / 1e6)
    return encoder, idim


def export_onnx(encoder, onnx_path: str, idim: int, opset: int = 17):
    """Export encoder to ONNX."""
    logger.info("Exporting encoder to ONNX...")

    seq_len = 400
    batch_size = 1
    dummy_input = torch.randn(batch_size, seq_len, idim, dtype=torch.float16)
    dummy_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

    torch.onnx.export(
        encoder,
        (dummy_input, dummy_lengths),
        onnx_path,
        opset_version=opset,
        input_names=["padded_input", "input_lengths"],
        output_names=["enc_output", "output_lengths", "src_mask"],
        dynamic_axes={
            "padded_input": {0: "batch_size", 1: "seq_len"},
            "input_lengths": {0: "batch_size"},
            "enc_output": {0: "batch_size", 1: "seq_len_out"},
            "output_lengths": {0: "batch_size"},
            "src_mask": {0: "batch_size", 2: "seq_len_out"},
        },
    )
    logger.info(f"ONNX exported to {onnx_path}")


def build_trt_engine(onnx_path: str, engine_path: str, idim: int,
                     opt_batch: int = 16, max_batch: int = 64):
    """Build TensorRT engine from ONNX model."""
    import tensorrt as trt

    logger.info("Building TensorRT engine (this may take 10-30 minutes for Conformer)...")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    profile = builder.create_optimization_profile()
    min_seq, opt_seq, max_seq = 50, 400, 3000

    # padded_input: (batch, seq, idim)
    profile.set_shape("padded_input",
                      (1, min_seq, idim),
                      (opt_batch, opt_seq, idim),
                      (max_batch, max_seq, idim))
    # input_lengths: (batch,)
    profile.set_shape("input_lengths",
                      (1,), (opt_batch,), (max_batch,))
    config.add_optimization_profile(profile)

    # Build serialized engine
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    logger.info(f"TensorRT engine saved to {engine_path}")


def main():
    args = get_parser().parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir

    if not os.path.isdir(model_dir):
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load encoder
    encoder, idim = load_encoder(model_dir)

    # Step 1: Export ONNX
    onnx_path = os.path.join(output_dir, "encoder.fp16.onnx")
    export_onnx(encoder, onnx_path, idim, args.opset_version)

    # Step 2: Build TensorRT engine
    engine_path = os.path.join(output_dir, "encoder.plan")
    build_trt_engine(onnx_path, engine_path, idim,
                     args.opt_batch_size, args.max_batch_size)

    # Step 3: Copy required model files
    for fname in ["cmvn.ark", "dict.txt", "train_bpe1000.model", "model.pth.tar"]:
        src = os.path.join(model_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info("Copied %s", fname)

    # Step 4: Clean up ONNX unless requested
    if not args.keep_onnx and os.path.exists(onnx_path):
        os.remove(onnx_path)
        logger.info("Removed intermediate ONNX file")

    logger.info("=" * 50)
    logger.info("Export complete!")
    logger.info("TensorRT engine: %s", engine_path)
    logger.info("")
    logger.info("To use it in anime-pipeline, set in config:")
    logger.info("  FIRERED_TRT_ENGINE_DIR = '%s'", output_dir)
    logger.info("")
    logger.info("Or use via CLI: --model firered-asr2-trt")


if __name__ == "__main__":
    main()
