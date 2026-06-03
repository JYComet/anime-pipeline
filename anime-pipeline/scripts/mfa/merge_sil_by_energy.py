#!/usr/bin/env python3
"""
Merge short sil phones in TextGrid files based on energy thresholds.

For each short "sil" interval in the phones tier:
1. It must be shorter than --max_sil_duration.
2. Its non-zero energy mean must be greater than
   previous_phone_nonzero_mean * --threshold.

If both conditions pass, the sil interval is merged into the previous phone.
The matching <eps> interval in the words tier is also merged into the previous
word when an exact time match is found.
"""

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge short sil intervals in TextGrid files using energy."
    )
    parser.add_argument(
        "--energy_dir",
        type=str,
        required=True,
        help="Directory containing energy .npy files.",
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        required=True,
        help="Directory containing source TextGrid files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for merged TextGrid outputs.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Frame rate used by the energy features.",
    )
    parser.add_argument(
        "--max_sil_duration",
        type=float,
        default=0.2,
        help="Only sil intervals shorter than this duration are candidates.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Merge when sil_mean_nonzero > prev_mean_nonzero * threshold.",
    )
    parser.add_argument(
        "--eps_token",
        type=str,
        default="<eps>",
        help="Token used for silence in the words tier.",
    )
    parser.add_argument(
        "--sil_token",
        type=str,
        default="sil",
        help="Token used for silence in the phones tier.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance when matching word intervals to sil intervals.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would change. Do not write output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file merge details.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_sil_duration <= 0:
        parser.error("--max_sil_duration must be positive")
    if args.threshold < 0:
        parser.error("--threshold must be non-negative")
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    return args


def read_textgrid(textgrid_path):
    content = textgrid_path.read_text(encoding="utf-8")
    match = re.search(
        r'File type = "ooTextFile"\s+'
        r'Object class = "TextGrid"\s+'
        r'xmin = ([\d.]+)\s+'
        r'xmax = ([\d.]+)\s+'
        r'tiers\? <exists>\s+'
        r'size = (\d+)',
        content,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Invalid TextGrid header: {textgrid_path}")

    tiers = {}
    for tier_name in ("words", "phones"):
        tier_match = re.search(
            rf'name = "{re.escape(tier_name)}"\s+.*?intervals: size = (\d+)(.*?)(?=\n\s*item \[\d+\]:|$)',
            content,
            re.DOTALL,
        )
        if not tier_match:
            raise ValueError(f"Missing {tier_name} tier: {textgrid_path}")

        intervals = []
        for item in re.finditer(
            r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"',
            tier_match.group(2),
            re.DOTALL,
        ):
            intervals.append(
                {
                    "xmin": float(item.group(1)),
                    "xmax": float(item.group(2)),
                    "text": item.group(3),
                }
            )
        tiers[tier_name] = intervals

    return {
        "content": content,
        "xmin": float(match.group(1)),
        "xmax": float(match.group(2)),
        "words": tiers["words"],
        "phones": tiers["phones"],
    }


def format_float(value):
    if isinstance(value, int):
        return str(value)
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_tier_block(item_index, name, xmin, xmax, intervals):
    lines = [
        f"    item [{item_index}]:",
        '        class = "IntervalTier" ',
        f'        name = "{name}" ',
        f"        xmin = {format_float(xmin)} ",
        f"        xmax = {format_float(xmax)} ",
        f"        intervals: size = {len(intervals)} ",
    ]
    for idx, interval in enumerate(intervals, start=1):
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {format_float(interval['xmin'])} ",
                f"            xmax = {format_float(interval['xmax'])} ",
                f'            text = "{interval["text"]}" ',
            ]
        )
    return "\n".join(lines)


def write_textgrid(data):
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {format_float(data['xmin'])} ",
        f"xmax = {format_float(data['xmax'])} ",
        "tiers? <exists> ",
        "size = 2 ",
        "item []: ",
        build_tier_block(1, "words", data["xmin"], data["xmax"], data["words"]),
        build_tier_block(2, "phones", data["xmin"], data["xmax"], data["phones"]),
        "",
    ]
    return "\n".join(lines)


def time_to_frame_range(xmin, xmax, fps, frame_count):
    start = int(math.floor(xmin * fps))
    end = int(math.ceil(xmax * fps))

    start = max(0, min(start, frame_count))
    end = max(0, min(end, frame_count))
    if end <= start:
        end = min(frame_count, start + 1)
    return start, end


def nonzero_mean(segment):
    if segment.size == 0:
        return 0.0
    nonzero = segment[segment > 0]
    if nonzero.size == 0:
        return 0.0
    return float(nonzero.mean())


def find_matching_word_index(words, sil_interval, eps_token, tolerance):
    for idx, word in enumerate(words):
        if word["text"] != eps_token:
            continue
        if (
            abs(word["xmin"] - sil_interval["xmin"]) <= tolerance
            and abs(word["xmax"] - sil_interval["xmax"]) <= tolerance
        ):
            return idx
    return None


def collect_merge_ops(textgrid_data, energy, args):
    phones = textgrid_data["phones"]
    words = textgrid_data["words"]
    frame_count = len(energy)

    merge_ops = []
    stats = {
        "candidate_short_sil_count": 0,
        "merged_sil_count": 0,
        "skipped_by_energy_threshold": 0,
        "word_eps_match_failures": 0,
    }

    for idx, sil_interval in enumerate(phones):
        if sil_interval["text"] != args.sil_token:
            continue

        duration = sil_interval["xmax"] - sil_interval["xmin"]
        if duration >= args.max_sil_duration:
            continue

        stats["candidate_short_sil_count"] += 1

        if idx == 0:
            stats["skipped_by_energy_threshold"] += 1
            continue

        prev_interval = phones[idx - 1]
        sil_start, sil_end = time_to_frame_range(
            sil_interval["xmin"], sil_interval["xmax"], args.fps, frame_count
        )
        prev_start, prev_end = time_to_frame_range(
            prev_interval["xmin"], prev_interval["xmax"], args.fps, frame_count
        )

        sil_mean = nonzero_mean(energy[sil_start:sil_end])
        prev_mean = nonzero_mean(energy[prev_start:prev_end])

        if not (sil_mean > prev_mean * args.threshold):
            stats["skipped_by_energy_threshold"] += 1
            continue

        word_index = find_matching_word_index(
            words, sil_interval, args.eps_token, args.tolerance
        )
        if word_index is None:
            stats["word_eps_match_failures"] += 1

        merge_ops.append(
            {
                "sil_phone_index": idx,
                "prev_phone_index": idx - 1,
                "sil_interval": sil_interval.copy(),
                "matched_word_index": word_index,
                "sil_mean": sil_mean,
                "prev_mean": prev_mean,
            }
        )
        stats["merged_sil_count"] += 1

    return merge_ops, stats


def apply_merge_ops(textgrid_data, merge_ops):
    phones = [interval.copy() for interval in textgrid_data["phones"]]
    words = [interval.copy() for interval in textgrid_data["words"]]

    for op in sorted(merge_ops, key=lambda item: item["sil_phone_index"], reverse=True):
        sil_idx = op["sil_phone_index"]
        prev_idx = op["prev_phone_index"]
        if sil_idx >= len(phones) or prev_idx < 0:
            continue

        phones[prev_idx]["xmax"] = phones[sil_idx]["xmax"]
        del phones[sil_idx]

        word_idx = op["matched_word_index"]
        if word_idx is not None and 0 < word_idx < len(words):
            words[word_idx - 1]["xmax"] = words[word_idx]["xmax"]
            del words[word_idx]

    merged = {
        "xmin": textgrid_data["xmin"],
        "xmax": textgrid_data["xmax"],
        "words": words,
        "phones": phones,
    }
    return merged


def process_textgrid(textgrid_path, energy_dir, output_dir, args):
    result = {
        "file": textgrid_path.name,
        "changed": False,
        "missing_energy": False,
        "parse_failed": False,
        "candidate_short_sil_count": 0,
        "merged_sil_count": 0,
        "skipped_by_energy_threshold": 0,
        "word_eps_match_failures": 0,
    }

    energy_path = energy_dir / f"{textgrid_path.stem}.npy"
    output_path = output_dir / textgrid_path.name

    if not energy_path.is_file():
        result["missing_energy"] = True
        if not args.dry_run:
            shutil.copyfile(textgrid_path, output_path)
        return result

    try:
        textgrid_data = read_textgrid(textgrid_path)
    except Exception:
        result["parse_failed"] = True
        if not args.dry_run:
            shutil.copyfile(textgrid_path, output_path)
        return result

    energy = np.load(energy_path)
    energy = np.asarray(energy, dtype=np.float32).reshape(-1)

    merge_ops, stats = collect_merge_ops(textgrid_data, energy, args)
    result.update(stats)
    result["changed"] = bool(merge_ops)

    if args.verbose:
        print(
            f"{textgrid_path.name}: candidates={stats['candidate_short_sil_count']} "
            f"merged={stats['merged_sil_count']} "
            f"threshold_skips={stats['skipped_by_energy_threshold']} "
            f"word_miss={stats['word_eps_match_failures']}"
        )
        for op in merge_ops[:20]:
            sil_interval = op["sil_interval"]
            print(
                f"  merge sil [{sil_interval['xmin']:.3f}, {sil_interval['xmax']:.3f}] "
                f"mean={op['sil_mean']:.4f} prev_mean={op['prev_mean']:.4f}"
            )

    if not args.dry_run:
        if merge_ops:
            merged = apply_merge_ops(textgrid_data, merge_ops)
            output_path.write_text(write_textgrid(merged), encoding="utf-8")
        else:
            shutil.copyfile(textgrid_path, output_path)

    return result


def main():
    args = parse_args()

    energy_dir = Path(args.energy_dir).expanduser().resolve()
    textgrid_dir = Path(args.textgrid_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not energy_dir.is_dir():
        raise FileNotFoundError(f"Energy directory not found: {energy_dir}")
    if not textgrid_dir.is_dir():
        raise FileNotFoundError(f"TextGrid directory not found: {textgrid_dir}")

    textgrid_files = sorted(textgrid_dir.glob("*.TextGrid"))
    if not textgrid_files:
        raise FileNotFoundError(f"No .TextGrid files found in: {textgrid_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "total_textgrids": len(textgrid_files),
        "matched_energy_files": 0,
        "missing_energy_files": 0,
        "parse_failed_files": 0,
        "candidate_short_sil_count": 0,
        "merged_sil_count": 0,
        "skipped_by_energy_threshold": 0,
        "word_eps_match_failures": 0,
        "files_with_changes": 0,
        "files_without_changes": 0,
    }

    for textgrid_path in tqdm(textgrid_files, desc="Processing TextGrid"):
        result = process_textgrid(textgrid_path, energy_dir, output_dir, args)

        if result["missing_energy"]:
            totals["missing_energy_files"] += 1
            totals["files_without_changes"] += 1
            continue

        totals["matched_energy_files"] += 1

        if result["parse_failed"]:
            totals["parse_failed_files"] += 1
            totals["files_without_changes"] += 1
            continue

        totals["candidate_short_sil_count"] += result["candidate_short_sil_count"]
        totals["merged_sil_count"] += result["merged_sil_count"]
        totals["skipped_by_energy_threshold"] += result["skipped_by_energy_threshold"]
        totals["word_eps_match_failures"] += result["word_eps_match_failures"]

        if result["changed"]:
            totals["files_with_changes"] += 1
        else:
            totals["files_without_changes"] += 1

    print(f"TextGrid dir: {textgrid_dir}")
    print(f"Energy dir: {energy_dir}")
    if not args.dry_run:
        print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Total TextGrid files: {totals['total_textgrids']}")
    print(f"Matched energy files: {totals['matched_energy_files']}")
    print(f"Missing energy files: {totals['missing_energy_files']}")
    print(f"Parse failed files: {totals['parse_failed_files']}")
    print(f"Candidate short sil count: {totals['candidate_short_sil_count']}")
    print(f"Merged sil count: {totals['merged_sil_count']}")
    print(f"Skipped by energy threshold: {totals['skipped_by_energy_threshold']}")
    print(f"Word <eps> match failures: {totals['word_eps_match_failures']}")
    print(f"Files with changes: {totals['files_with_changes']}")
    print(f"Files without changes: {totals['files_without_changes']}")


if __name__ == "__main__":
    main()
