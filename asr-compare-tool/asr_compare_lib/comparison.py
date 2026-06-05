"""
ASR text comparison: Levenshtein distance, diff computation, SRT parsing & alignment.
Exact replication of the comparison logic from asr_pipeline.py.
"""
import os
import re

from .normalization import normalize_text_mfa

# 3+ consecutive Latin letters = likely hallucination in Chinese output
_ENGLISH_RE = re.compile(r"[a-zA-Z]{3,}")


def has_english(text: str) -> bool:
    """Check if text contains English words (3+ consecutive ASCII letters)."""
    return bool(_ENGLISH_RE.search(text))


def levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings (optimized single-row DP)."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_diff_chunks(text_a: str, text_b: str) -> list:
    """Compute character-level diff between two texts using LCS backtracking.

    Returns list of {type: "equal"|"diff", text_a: str, text_b: str}.
    """
    a = text_a or ""
    b = text_b or ""
    if not a and not b:
        return []
    if not a:
        return [{"type": "diff", "text_a": "", "text_b": b}]
    if not b:
        return [{"type": "diff", "text_a": a, "text_b": ""}]

    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append(("equal", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            ops.append(("diff", "", b[j - 1]))
            j -= 1
        else:
            ops.append(("diff", a[i - 1], ""))
            i -= 1
    ops.reverse()

    chunks = []
    for op, ca, cb in ops:
        if not chunks or chunks[-1]["type"] != op:
            chunks.append({"type": op, "text_a": ca, "text_b": cb})
        else:
            chunks[-1]["text_a"] += ca
            chunks[-1]["text_b"] += cb

    return chunks


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------
def _parse_srt_timestamp(ts: str) -> int:
    """Parse SRT timestamp HH:MM:SS,mmm to milliseconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s_parts = parts[2].split(".")
    s = int(s_parts[0])
    if len(s_parts) > 1:
        ms_str = s_parts[1].strip()
        ms = int(ms_str.ljust(3, "0")[:3])
    else:
        ms = 0
    return h * 3600000 + m * 60000 + s * 1000 + ms


def parse_srt_to_segments(srt_path: str) -> list:
    """Parse an SRT file into a list of segment dicts.

    Each segment: {index, start_ms, end_ms, text, normalized_text}
    """
    segments = []
    if not os.path.exists(srt_path):
        return segments

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        idx_line = lines[0].strip()
        if not idx_line.isdigit():
            continue
        idx = int(idx_line)
        ts_line = lines[1].strip()
        if "-->" not in ts_line:
            continue
        parts = ts_line.split("-->")
        start_ms = _parse_srt_timestamp(parts[0])
        end_ms = _parse_srt_timestamp(parts[1])
        text = " ".join(line.strip() for line in lines[2:] if line.strip())
        if not text:
            continue
        segments.append({
            "index": idx,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "normalized_text": normalize_text_mfa(text),
        })

    return segments


def srt_to_plain_text(srt_path: str) -> str:
    """Extract normalized plain text from an SRT file for comparison."""
    segments = parse_srt_to_segments(srt_path)
    if not segments:
        return ""
    return " ".join(seg["normalized_text"] for seg in segments)


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------
def _time_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return overlap duration in ms between two time ranges."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def align_segments_by_time(segs_a: list, segs_b: list,
                           min_overlap_ratio: float = 0.3) -> dict:
    """Align segments from two SRTs by time overlap."""
    aligned_pairs = []
    used_b = set()

    for seg_a in segs_a:
        best_b = None
        best_overlap = 0
        for j, seg_b in enumerate(segs_b):
            if j in used_b:
                continue
            overlap = _time_overlap(
                seg_a["start_ms"], seg_a["end_ms"],
                seg_b["start_ms"], seg_b["end_ms"]
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_b = (j, seg_b)

        if best_b and best_overlap > 0:
            seg_b_dur = best_b[1]["end_ms"] - best_b[1]["start_ms"]
            seg_a_dur = seg_a["end_ms"] - seg_a["start_ms"]
            min_dur = min(seg_a_dur, seg_b_dur) if seg_b_dur > 0 else 0
            overlap_ratio = best_overlap / min_dur if min_dur > 0 else 0
            if overlap_ratio >= min_overlap_ratio:
                used_b.add(best_b[0])
                aligned_pairs.append((seg_a, best_b[1], best_overlap))

    unmatched_a = [s for s in segs_a if not any(s is pair[0] for pair in aligned_pairs)]
    unmatched_b = [s for j, s in enumerate(segs_b) if j not in used_b]

    return {
        "aligned_pairs": aligned_pairs,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
    }


# ---------------------------------------------------------------------------
# SRT sentence-level comparison (whole-file mode)
# ---------------------------------------------------------------------------
def compare_srt_sentences(srt_path_a: str, srt_path_b: str,
                          match_threshold: float = 90.0) -> dict:
    """Compare two SRT files sentence-by-sentence with time-based alignment."""
    segs_a = parse_srt_to_segments(srt_path_a)
    segs_b = parse_srt_to_segments(srt_path_b)

    if not segs_a and not segs_b:
        return {
            "overall_diff_percent": 0.0, "match_rate": 100.0,
            "sentence_results": [], "matched_count": 0,
            "unmatched_a": 0, "unmatched_b": 0, "flagged": False,
        }
    if not segs_a or not segs_b:
        return {
            "overall_diff_percent": 100.0, "match_rate": 0.0,
            "sentence_results": [], "matched_count": 0,
            "unmatched_a": len(segs_a), "unmatched_b": len(segs_b),
            "flagged": True,
        }

    alignment = align_segments_by_time(segs_a, segs_b)
    sentence_results = []
    total_weight = 0
    weighted_diff_sum = 0.0

    for seg_a, seg_b, overlap_ms in alignment["aligned_pairs"]:
        start_ms = max(seg_a["start_ms"], seg_b["start_ms"])
        end_ms = min(seg_a["end_ms"], seg_b["end_ms"])

        t_a = seg_a["normalized_text"]
        t_b = seg_b["normalized_text"]
        if not t_a and not t_b:
            diff = 0.0
        elif not t_a or not t_b:
            diff = 100.0
        else:
            dist = levenshtein(t_a, t_b)
            max_len = max(len(t_a), len(t_b))
            diff = round((dist / max_len) * 100, 1)

        weight = max(len(t_a), len(t_b))
        weighted_diff_sum += diff * weight
        total_weight += weight

        diff_chunks = compute_diff_chunks(seg_a["text"], seg_b["text"])
        sentence_results.append({
            "idx_a": seg_a["index"], "idx_b": seg_b["index"],
            "start_ms": start_ms, "end_ms": end_ms,
            "text_a": seg_a["text"], "text_b": seg_b["text"],
            "diff_percent": diff, "match_rate": round(100.0 - diff, 1),
            "flagged": (100.0 - diff) < match_threshold,
            "diff_chunks": diff_chunks,
        })

    for seg_a in alignment["unmatched_a"]:
        weight = len(seg_a["normalized_text"])
        weighted_diff_sum += 100.0 * weight
        total_weight += weight
        sentence_results.append({
            "idx_a": seg_a["index"], "idx_b": None,
            "start_ms": seg_a["start_ms"], "end_ms": seg_a["end_ms"],
            "text_a": seg_a["text"], "text_b": "",
            "diff_percent": 100.0, "match_rate": 0.0, "flagged": True,
            "diff_chunks": [{"type": "diff", "text_a": seg_a["text"], "text_b": ""}],
        })

    for seg_b in alignment["unmatched_b"]:
        weight = len(seg_b["normalized_text"])
        weighted_diff_sum += 100.0 * weight
        total_weight += weight
        sentence_results.append({
            "idx_a": None, "idx_b": seg_b["index"],
            "start_ms": seg_b["start_ms"], "end_ms": seg_b["end_ms"],
            "text_a": "", "text_b": seg_b["text"],
            "diff_percent": 100.0, "match_rate": 0.0, "flagged": True,
            "diff_chunks": [{"type": "diff", "text_a": "", "text_b": seg_b["text"]}],
        })

    overall_diff = round(weighted_diff_sum / total_weight, 1) if total_weight > 0 else 0.0
    match_rate = round(100.0 - overall_diff, 1)
    sentence_results.sort(key=lambda r: r["start_ms"])

    return {
        "overall_diff_percent": overall_diff,
        "match_rate": match_rate,
        "flagged": match_rate < match_threshold,
        "sentence_results": sentence_results,
        "matched_count": len(alignment["aligned_pairs"]),
        "unmatched_a": len(alignment["unmatched_a"]),
        "unmatched_b": len(alignment["unmatched_b"]),
    }


# ---------------------------------------------------------------------------
# Per-segment text comparison (segmented mode)
# ---------------------------------------------------------------------------
def compare_segment_texts(text_a: str, text_b: str, match_threshold: float = 90.0,
                          filter_english: bool = True) -> dict:
    """Compare two segment texts, return diff/match info."""
    norm_a = normalize_text_mfa(text_a)
    norm_b = normalize_text_mfa(text_b)

    if filter_english and (has_english(norm_a) or has_english(norm_b)):
        diff_percent = 100.0
    elif not norm_a and not norm_b:
        diff_percent = 0.0
    elif not norm_a or not norm_b:
        diff_percent = 100.0
    else:
        dist = levenshtein(norm_a, norm_b)
        max_len = max(len(norm_a), len(norm_b))
        diff_percent = round((dist / max_len) * 100, 1)

    match_rate = round(100.0 - diff_percent, 1)
    diff_chunks = compute_diff_chunks(text_a, text_b) if (text_a or text_b) else []

    return {
        "diff_percent": diff_percent,
        "match_rate": match_rate,
        "flagged": match_rate < match_threshold,
        "diff_chunks": diff_chunks,
        "norm_a": norm_a,
        "norm_b": norm_b,
    }
