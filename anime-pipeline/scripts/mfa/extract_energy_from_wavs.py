#!/usr/bin/env python3
"""
Extract energy features directly from wav files.

This script is intentionally self-contained. It does not depend on the
original project's audio/utils package structure and only requires a wav
directory plus standard third-party audio libraries.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract energy features from wav files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root. Reads from <data_dir>/wavs and writes to <data_dir>/energy by default.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        help="Directory that contains wav files. Overrides the input path from --data_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for .npy energy outputs. Defaults to <data_dir>/energy or sibling energy/ of --wav_dir.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=32000,
        help="Target sample rate. Input audio is resampled when needed.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Target frame rate used to crop the feature length.",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=1024,
        help="FFT size for spectrogram extraction.",
    )
    parser.add_argument(
        "--win_length",
        type=int,
        default=1024,
        help="Window length for spectrogram extraction.",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=None,
        help="Hop length. Defaults to sample_rate // fps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute files even if the target .npy already exists.",
    )
    return parser.parse_args()


def resolve_paths(args):
    if not args.data_dir and not args.wav_dir:
        raise ValueError("Either --data_dir or --wav_dir is required.")

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None

    if args.wav_dir:
        wav_dir = Path(args.wav_dir).expanduser().resolve()
    else:
        wav_dir = data_dir / "wavs"

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif data_dir is not None:
        output_dir = data_dir / "energy"
    else:
        output_dir = wav_dir.parent / "energy"

    return wav_dir, output_dir


def load_audio_mono_resampled(wav_path, target_sr):
    audio, sample_rate = sf.read(str(wav_path), always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("empty audio")

    wav_tensor = torch.from_numpy(audio)
    if sample_rate != target_sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sample_rate, target_sr)

    return wav_tensor.contiguous(), target_sr


def compute_energy(wav_tensor, sample_rate, fps, spec_transform):
    duration_sec = wav_tensor.shape[0] / float(sample_rate)
    target_frames = max(1, int(duration_sec * fps))

    spec = spec_transform(wav_tensor)
    energy = torch.norm(torch.abs(spec), dim=-2)

    if energy.ndim != 1:
        energy = energy.reshape(-1)

    if energy.numel() == 0:
        return np.zeros((target_frames,), dtype=np.float32)

    if energy.numel() < target_frames:
        pad_value = energy[-1]
        energy = torch.nn.functional.pad(
            energy, (0, target_frames - energy.numel()), value=float(pad_value)
        )
    else:
        energy = energy[:target_frames]

    return energy.cpu().numpy().astype(np.float32, copy=False)


def extract_energy(
    wav_dir,
    output_dir,
    sample_rate,
    fps,
    n_fft,
    win_length,
    hop_length,
    force=False,
):
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"Input wav directory does not exist: {wav_dir}")

    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {wav_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=None,
    )

    processed = 0
    skipped = 0
    failed = []

    for wav_path in tqdm(wav_files, desc="Extracting energy"):
        save_path = output_dir / f"{wav_path.stem}.npy"
        if save_path.exists() and not force:
            skipped += 1
            continue

        try:
            wav_tensor, _ = load_audio_mono_resampled(wav_path, sample_rate)
            energy = compute_energy(wav_tensor, sample_rate, fps, spec_transform)
            np.save(save_path, energy)
            processed += 1
        except Exception as exc:
            failed.append((wav_path.name, str(exc)))

    return {
        "input_count": len(wav_files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "output_dir": output_dir,
    }


def main():
    args = parse_args()

    try:
        wav_dir, output_dir = resolve_paths(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    hop_length = args.hop_length or (args.sample_rate // args.fps)
    if hop_length <= 0:
        print("Error: hop_length must be positive.", file=sys.stderr)
        return 1

    print("=" * 80)
    print("Energy Extraction")
    print("=" * 80)
    print(f"Input wav dir : {wav_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Sample rate   : {args.sample_rate}")
    print(f"FPS           : {args.fps}")
    print(f"n_fft         : {args.n_fft}")
    print(f"win_length    : {args.win_length}")
    print(f"hop_length    : {hop_length}")
    print(f"Force         : {args.force}")

    try:
        stats = extract_energy(
            wav_dir=wav_dir,
            output_dir=output_dir,
            sample_rate=args.sample_rate,
            fps=args.fps,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=hop_length,
            force=args.force,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("-" * 80)
    print(f"Total wavs    : {stats['input_count']}")
    print(f"Processed     : {stats['processed']}")
    print(f"Skipped       : {stats['skipped']}")
    print(f"Failed        : {len(stats['failed'])}")

    if stats["failed"]:
        print("Failed files:")
        for name, reason in stats["failed"][:10]:
            print(f"  {name}: {reason}")
        if len(stats["failed"]) > 10:
            print(f"  ... and {len(stats['failed']) - 10} more")
        return 1

    print("Energy extraction finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
