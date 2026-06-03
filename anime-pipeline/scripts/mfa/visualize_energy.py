#!/usr/bin/env python3
"""
Visualize 1D energy features along frame and time axes, with optional
TextGrid phone overlays that emphasize "sil" alignment.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize an energy .npy file and save it as an image."
    )
    parser.add_argument(
        "--energy_path",
        type=str,
        default=None,
        help="Path to a 1D .npy energy file.",
    )
    parser.add_argument(
        "--energy_dir",
        type=str,
        default=None,
        help="Directory that contains .npy energy files for batch visualization.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output image path for single-file mode. Defaults to <energy_path stem>_plot.png.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for batch mode. Defaults to <energy_dir>/plots.",
    )
    parser.add_argument(
        "--textgrid_path",
        type=str,
        default=None,
        help="Path to the matching TextGrid file in single-file mode.",
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        default=None,
        help="Directory containing matching TextGrid files in batch mode.",
    )
    parser.add_argument(
        "--overlay_textgrid",
        action="store_true",
        help="Overlay phone intervals from TextGrid on the energy figure.",
    )
    parser.add_argument(
        "--focus_sil",
        action="store_true",
        help="Highlight the sil phone and treat it as the primary alignment signal.",
    )
    parser.add_argument(
        "--show_non_sil_labels",
        action="store_true",
        help="Show labels for non-sil phones when their duration is long enough.",
    )
    parser.add_argument(
        "--min_phone_label_sec",
        type=float,
        default=0.10,
        help="Minimum duration in seconds before a non-sil phone label is drawn.",
    )
    parser.add_argument(
        "--sil_token",
        type=str,
        default="sil",
        help="Phone label treated as silence for highlighting.",
    )
    parser.add_argument(
        "--sil_context_mode",
        type=str,
        default="adjacent_phones",
        choices=["sil_only", "adjacent_phones", "adjacent_words"],
        help=(
            "When --focus_sil is enabled: show only sil, show sil plus adjacent "
            "phones, or show sil plus adjacent phones and neighboring word labels."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Feature frame rate, used to map frames to seconds.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom chart title.",
    )
    args = parser.parse_args()
    if not args.energy_path and not args.energy_dir:
        parser.error("one of --energy_path or --energy_dir is required")
    if args.energy_path and args.energy_dir:
        parser.error("--energy_path and --energy_dir cannot be used together")
    if args.output_path and args.energy_dir:
        parser.error("--output_path only works with --energy_path")
    if args.textgrid_path and args.energy_dir:
        parser.error("--textgrid_path only works with --energy_path")
    if args.textgrid_dir and args.energy_path:
        parser.error("--textgrid_dir only works with --energy_dir")
    return args


def load_energy(energy_path):
    energy = np.load(energy_path)
    energy = np.asarray(energy, dtype=np.float32).reshape(-1)
    if energy.size == 0:
        raise ValueError("energy file is empty")
    return energy


def default_output_path(energy_path):
    return energy_path.with_name(f"{energy_path.stem}_plot.png")


def candidate_textgrid_paths(base_dir, stem):
    return [
        base_dir / "textgrid" / f"{stem}.TextGrid",
        base_dir / "textgrid" / f"{stem}.textgrid",
        base_dir / "TextGrid" / f"{stem}.TextGrid",
        base_dir / "TextGrid" / f"{stem}.textgrid",
    ]


def infer_textgrid_path(energy_path):
    search_roots = []
    if energy_path.parent.name.lower() == "energy":
        search_roots.append(energy_path.parent.parent)
    search_roots.append(energy_path.parent)

    seen = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        for candidate in candidate_textgrid_paths(root, energy_path.stem):
            if candidate.is_file():
                return candidate
    return None


def resolve_single_textgrid_path(energy_path, textgrid_path=None):
    if textgrid_path:
        resolved = Path(textgrid_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"TextGrid file not found: {resolved}")
        return resolved

    inferred = infer_textgrid_path(energy_path)
    if inferred is None:
        raise FileNotFoundError(
            f"Could not infer a TextGrid file for {energy_path.name}. "
            "Use --textgrid_path to provide it explicitly."
        )
    return inferred


def resolve_batch_textgrid_dir(energy_dir, textgrid_dir=None):
    if textgrid_dir:
        resolved = Path(textgrid_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"TextGrid directory not found: {resolved}")
        return resolved

    candidates = [
        energy_dir.parent / "textgrid",
        energy_dir.parent / "TextGrid",
        energy_dir / "textgrid",
        energy_dir / "TextGrid",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not infer a TextGrid directory for {energy_dir}. "
        "Use --textgrid_dir to provide it explicitly."
    )


def parse_textgrid_tier(content, tier_name):
    match = re.search(
        rf'name = "{re.escape(tier_name)}"\s+.*?intervals: size = \d+(.*?)(?=\n\s*item \[\d+\]:|$)',
        content,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"{tier_name} tier not found")

    intervals = []
    for item in re.finditer(
        r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"',
        match.group(1),
        re.DOTALL,
    ):
        intervals.append(
            {
                "xmin": float(item.group(1)),
                "xmax": float(item.group(2)),
                "text": item.group(3),
            }
        )

    return intervals


def parse_textgrid_data(textgrid_path):
    content = textgrid_path.read_text(encoding="utf-8")
    try:
        words = parse_textgrid_tier(content, "words")
        phones = parse_textgrid_tier(content, "phones")
    except ValueError as exc:
        raise ValueError(f"{exc} in {textgrid_path}") from exc

    return {"words": words, "phones": phones}


def split_phone_intervals(phone_intervals, sil_token):
    sil_intervals = []
    other_intervals = []
    for interval in phone_intervals:
        if interval["text"] == sil_token:
            sil_intervals.append(interval)
        else:
            other_intervals.append(interval)
    return sil_intervals, other_intervals


def build_sil_summary(sil_intervals):
    if not sil_intervals:
        return "sil_count=0  total_sil=0.00s  longest_sil=0.00s"

    durations = [max(0.0, item["xmax"] - item["xmin"]) for item in sil_intervals]
    return (
        f"sil_count={len(sil_intervals)}  total_sil={sum(durations):.2f}s  "
        f"longest_sil={max(durations):.2f}s"
    )


def find_neighbor_word(words, sil_interval, side, eps_token="<eps>"):
    tol = 1e-4
    lexical_words = [item for item in words if item["text"] and item["text"] != eps_token]
    if side == "left":
        candidate = None
        for word in lexical_words:
            if word["xmax"] <= sil_interval["xmin"] + tol:
                candidate = word
        return candidate

    for word in lexical_words:
        if word["xmin"] >= sil_interval["xmax"] - tol:
            return word
    return None


def collect_sil_neighbor_context(phones, sil_token, words=None):
    contexts = []

    for idx, phone in enumerate(phones):
        if phone["text"] != sil_token:
            continue

        prev_phone = None
        next_phone = None

        for left_idx in range(idx - 1, -1, -1):
            if phones[left_idx]["text"] != sil_token:
                prev_phone = phones[left_idx]
                break

        for right_idx in range(idx + 1, len(phones)):
            if phones[right_idx]["text"] != sil_token:
                next_phone = phones[right_idx]
                break

        context = {
            "sil": phone,
            "prev_phone": prev_phone,
            "next_phone": next_phone,
            "prev_word": None,
            "next_word": None,
        }

        if words:
            context["prev_word"] = find_neighbor_word(words, phone, "left")
            context["next_word"] = find_neighbor_word(words, phone, "right")

        contexts.append(context)

    return contexts


def draw_phone_reference(ax, intervals, fps, show_labels=False, min_label_sec=0.10):
    transform = ax.get_xaxis_transform()
    y_track = -0.12
    label_y = -0.18

    for interval in intervals:
        start_frame = max(0.0, interval["xmin"] * fps)
        end_frame = max(start_frame + 0.25, interval["xmax"] * fps)
        ax.plot(
            [start_frame, end_frame],
            [y_track, y_track],
            transform=transform,
            clip_on=False,
            color="#9aa0a6",
            linewidth=1.0,
            alpha=0.6,
        )

        duration_sec = max(0.0, interval["xmax"] - interval["xmin"])
        if show_labels and duration_sec >= min_label_sec:
            mid_frame = (start_frame + end_frame) * 0.5
            ax.text(
                mid_frame,
                label_y,
                interval["text"],
                transform=transform,
                clip_on=False,
                ha="center",
                va="top",
                fontsize=6,
                color="#5f6368",
            )


def draw_adjacent_phone_context(ax, contexts, fps, show_words=False):
    if not contexts:
        return

    transform = ax.get_xaxis_transform()
    phone_track = -0.03
    left_label_levels = [-0.11, -0.145, -0.18]
    right_label_levels = [-0.145, -0.18, -0.215]
    word_track = -0.23
    word_label_y = -0.27

    seen_word_keys = set()
    seen_phone_keys = set()
    last_label_mid = {"#1b9e77": None, "#8e44ad": None}
    last_label_level = {"#1b9e77": 0, "#8e44ad": 0}

    def pick_label_y(color, mid_frame):
        levels = left_label_levels if color == "#1b9e77" else right_label_levels
        prev_mid = last_label_mid[color]
        prev_level = last_label_level[color]

        if prev_mid is None or abs(mid_frame - prev_mid) >= 14.0:
            level = 0
        else:
            level = (prev_level + 1) % len(levels)

        last_label_mid[color] = mid_frame
        last_label_level[color] = level
        return levels[level]

    def draw_phone(phone, color, y_track):
        if phone is None:
            return
        key = (phone["xmin"], phone["xmax"], phone["text"], color)
        if key in seen_phone_keys:
            return
        seen_phone_keys.add(key)

        phone_start = max(0.0, phone["xmin"] * fps)
        phone_end = max(phone_start + 0.25, phone["xmax"] * fps)
        ax.axvspan(phone_start, phone_end, color=color, alpha=0.08, zorder=0)
        ax.axvline(phone_start, color=color, linewidth=0.75, alpha=0.45)
        ax.axvline(phone_end, color=color, linewidth=0.75, alpha=0.45)
        ax.plot(
            [phone_start, phone_end],
            [y_track, y_track],
            transform=transform,
            clip_on=False,
            color=color,
            linewidth=2.6,
            alpha=0.9,
        )
        mid_frame = (phone_start + phone_end) * 0.5
        ax.text(
            mid_frame,
            pick_label_y(color, mid_frame),
            phone["text"],
            transform=transform,
            clip_on=False,
            ha="center",
            va="top",
            fontsize=4.6,
            color=color,
        )

    def draw_word(word, color):
        if word is None:
            return
        key = (word["xmin"], word["xmax"], word["text"], color)
        if key in seen_word_keys:
            return
        seen_word_keys.add(key)
        start_frame = max(0.0, word["xmin"] * fps)
        end_frame = max(start_frame + 0.25, word["xmax"] * fps)
        ax.plot(
            [start_frame, end_frame],
            [word_track, word_track],
            transform=transform,
            clip_on=False,
            color=color,
            linewidth=2.2,
            alpha=0.75,
        )
        ax.text(
            (start_frame + end_frame) * 0.5,
            word_label_y,
            word["text"],
            transform=transform,
            clip_on=False,
            ha="center",
            va="top",
            fontsize=6.5,
            color=color,
        )

    for context in contexts:
        draw_phone(context["prev_phone"], "#1b9e77", phone_track)
        draw_phone(context["next_phone"], "#8e44ad", phone_track)

        if show_words:
            draw_word(context["prev_word"], "#1b9e77")
            draw_word(context["next_word"], "#8e44ad")


def draw_sil_overlay(ax, sil_intervals, fps, max_frame, sil_token):
    transform = ax.get_xaxis_transform()
    y_track = -0.03
    label_y = -0.08

    for interval in sil_intervals:
        start_frame = max(0.0, interval["xmin"] * fps)
        end_frame = min(max_frame, max(start_frame + 0.25, interval["xmax"] * fps))

        ax.axvspan(start_frame, end_frame, color="#f2994a", alpha=0.18, zorder=0)
        ax.axvline(start_frame, color="#d35400", linewidth=0.9, alpha=0.7)
        ax.axvline(end_frame, color="#d35400", linewidth=0.9, alpha=0.7)
        ax.plot(
            [start_frame, end_frame],
            [y_track, y_track],
            transform=transform,
            clip_on=False,
            color="#d35400",
            linewidth=3.0,
            alpha=0.9,
        )

        duration_sec = max(0.0, interval["xmax"] - interval["xmin"])
        mid_frame = (start_frame + end_frame) * 0.5
        ax.text(
            mid_frame,
            label_y,
            f"{sil_token}\n{duration_sec:.2f}s",
            transform=transform,
            clip_on=False,
            ha="center",
            va="top",
            fontsize=7,
            color="#a84300",
        )


def plot_energy(
    energy,
    output_path,
    fps,
    title,
    phone_intervals=None,
    word_intervals=None,
    focus_sil=False,
    show_non_sil_labels=False,
    min_phone_label_sec=0.10,
    sil_token="sil",
    sil_context_mode="adjacent_phones",
):
    frames = np.arange(len(energy), dtype=np.float32)
    duration_sec = len(energy) / fps

    fig_width = min(max(24.0, duration_sec * 4.0), 48.0)
    fig_height = 5.0
    if phone_intervals and focus_sil:
        fig_height = 5.9 if sil_context_mode == "adjacent_words" else 5.5
    elif phone_intervals:
        fig_height = 5.2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    max_frame = max(len(energy) - 1, 1)
    y_max = max(float(energy.max()), 1e-6)

    sil_intervals = []
    other_intervals = []
    neighbor_contexts = []
    if phone_intervals:
        sil_intervals, other_intervals = split_phone_intervals(phone_intervals, sil_token)
        if focus_sil:
            draw_sil_overlay(ax, sil_intervals, fps, max_frame, sil_token)
            if sil_context_mode != "sil_only":
                neighbor_contexts = collect_sil_neighbor_context(
                    phone_intervals,
                    sil_token,
                    words=word_intervals or [],
                )
                draw_adjacent_phone_context(
                    ax,
                    neighbor_contexts,
                    fps,
                    show_words=(sil_context_mode == "adjacent_words"),
                )
        else:
            draw_phone_reference(
                ax,
                other_intervals,
                fps,
                show_labels=show_non_sil_labels,
                min_label_sec=min_phone_label_sec,
            )

    ax.plot(frames, energy, color="#1f77b4", linewidth=1.6, label="Energy", zorder=3)
    ax.fill_between(frames, energy, 0, color="#1f77b4", alpha=0.18, zorder=2)

    ax.set_xlabel(f"Frame Index ({fps:.2f} fps)")
    ax.set_ylabel("Energy")
    ax.set_xlim(0, max_frame)
    ax.set_ylim(0, y_max * 1.15)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    chart_title = title or f"Energy Curve: {output_path.stem}"
    ax.set_title(chart_title)

    sec_ax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x / fps, lambda x: x * fps),
    )
    sec_ax.set_xlabel("Time (seconds)")

    summary = (
        f"frames={len(energy)}  duration={duration_sec:.2f}s  "
        f"min={energy.min():.4f}  max={energy.max():.4f}  mean={energy.mean():.4f}"
    )
    ax.text(
        0.01,
        0.97,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    if phone_intervals and focus_sil:
        ax.text(
            0.99,
            0.97,
            (
                f"{build_sil_summary(sil_intervals)}  "
                f"context_marks={len(neighbor_contexts)}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#f0c08b"},
        )

    if phone_intervals:
        if focus_sil:
            if sil_context_mode == "sil_only":
                bottom = 0.24
            elif sil_context_mode == "adjacent_words":
                bottom = 0.38
            else:
                bottom = 0.31
            fig.subplots_adjust(bottom=bottom, top=0.88)
        else:
            fig.subplots_adjust(bottom=0.28, top=0.88)
    else:
        fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.min_phone_label_sec < 0:
        raise ValueError("min_phone_label_sec must be non-negative")

    overlay_enabled = (
        args.overlay_textgrid
        or args.focus_sil
        or args.textgrid_path is not None
        or args.textgrid_dir is not None
    )

    if args.energy_path:
        energy_path = Path(args.energy_path).expanduser().resolve()
        if not energy_path.is_file():
            raise FileNotFoundError(f"Energy file not found: {energy_path}")

        output_path = (
            Path(args.output_path).expanduser().resolve()
            if args.output_path
            else default_output_path(energy_path)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        energy = load_energy(energy_path)
        phone_intervals = None
        word_intervals = None
        textgrid_used = None
        if overlay_enabled:
            textgrid_used = resolve_single_textgrid_path(energy_path, args.textgrid_path)
            textgrid_data = parse_textgrid_data(textgrid_used)
            phone_intervals = textgrid_data["phones"]
            word_intervals = textgrid_data["words"]

        plot_energy(
            energy,
            output_path,
            args.fps,
            args.title or f"Energy Curve: {energy_path.stem}",
            phone_intervals=phone_intervals,
            word_intervals=word_intervals,
            focus_sil=args.focus_sil,
            show_non_sil_labels=args.show_non_sil_labels,
            min_phone_label_sec=args.min_phone_label_sec,
            sil_token=args.sil_token,
            sil_context_mode=args.sil_context_mode,
        )

        print(f"Saved plot: {output_path}")
        print(
            f"Frames: {len(energy)}, Duration: {len(energy) / args.fps:.2f}s, "
            f"Min: {energy.min():.4f}, Max: {energy.max():.4f}, Mean: {energy.mean():.4f}"
        )
        if textgrid_used is not None:
            sil_count = sum(1 for item in phone_intervals if item["text"] == args.sil_token)
            print(f"TextGrid: {textgrid_used}")
            print(f"Phones: {len(phone_intervals)}, {args.sil_token}: {sil_count}")
        return

    energy_dir = Path(args.energy_dir).expanduser().resolve()
    if not energy_dir.is_dir():
        raise FileNotFoundError(f"Energy directory not found: {energy_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else energy_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_files = sorted(energy_dir.glob("*.npy"))
    if not energy_files:
        raise FileNotFoundError(f"No .npy files found in: {energy_dir}")

    textgrid_dir = None
    if overlay_enabled:
        textgrid_dir = resolve_batch_textgrid_dir(energy_dir, args.textgrid_dir)

    processed = 0
    failed = []
    with_textgrid = 0
    missing_textgrid = 0
    failed_textgrid_parse = 0
    files_without_sil = 0

    for energy_path in energy_files:
        output_path = output_dir / f"{energy_path.stem}_plot.png"
        try:
            energy = load_energy(energy_path)
            title = args.title or f"Energy Curve: {energy_path.stem}"

            phone_intervals = None
            word_intervals = None
            if overlay_enabled:
                candidates = [
                    textgrid_dir / f"{energy_path.stem}.TextGrid",
                    textgrid_dir / f"{energy_path.stem}.textgrid",
                ]
                matched = next((item for item in candidates if item.is_file()), None)
                if matched is None:
                    missing_textgrid += 1
                else:
                    try:
                        textgrid_data = parse_textgrid_data(matched)
                        phone_intervals = textgrid_data["phones"]
                        word_intervals = textgrid_data["words"]
                        with_textgrid += 1
                        if not any(item["text"] == args.sil_token for item in phone_intervals):
                            files_without_sil += 1
                    except Exception:
                        failed_textgrid_parse += 1
                        phone_intervals = None
                        word_intervals = None

            plot_energy(
                energy,
                output_path,
                args.fps,
                title,
                phone_intervals=phone_intervals,
                word_intervals=word_intervals,
                focus_sil=args.focus_sil,
                show_non_sil_labels=args.show_non_sil_labels,
                min_phone_label_sec=args.min_phone_label_sec,
                sil_token=args.sil_token,
                sil_context_mode=args.sil_context_mode,
            )
            processed += 1
        except Exception as exc:
            failed.append((energy_path.name, str(exc)))

    print(f"Batch output dir: {output_dir}")
    print(f"Total files: {len(energy_files)}")
    print(f"Processed: {processed}")
    print(f"Failed: {len(failed)}")
    if overlay_enabled:
        print(f"TextGrid dir: {textgrid_dir}")
        print(f"With TextGrid: {with_textgrid}")
        print(f"Missing TextGrid: {missing_textgrid}")
        print(f"Failed TextGrid parse: {failed_textgrid_parse}")
        print(f"Files without {args.sil_token}: {files_without_sil}")
    if failed:
        for name, reason in failed[:10]:
            print(f"  {name}: {reason}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
