r"""
对 MFA 输出的 TextGrid 做最终后处理。

这个脚本做什么：
- 读取 MFA align 输出的 TextGrid，并生成五层：raw_text、tokenized_text、words、romaji、phones。
- raw_text 来自 JSONL 原文；tokenized_text 来自 generate_wav_txt.py 生成的 TXT。
- words 和 phones 以 MFA 输出为准；romaji 使用 pykakasi 从 MFA words 转换。
- 问号会尽量插入到对应静音 interval，作为独立时间段。
- 自动修复一类高置信错位：`っ + て` 等短词后接长静音，且 `t/e` 被 MFA 放进静音前端。
- 把所有静默标签统一为 `<sp0>`、`<sp1>`、`<sp2>`、`<sp3>`，包括 phone 层，不保留 `sil/<sil>/<eps>`。
- 把明显异常样本输出到 filtered：`<sp3>`、`spn`、异常长/短 phone、word/phone 覆盖异常等。
- 输出 `postprocess_report.jsonl`，记录 warning、问号填充、修复、过滤原因和输出路径。

输入：
- `--jsonl`：原始 JSONL，用于读取完整原句和 wav 路径。
- `--txt-dir`：generate_wav_txt.py 生成的 TXT 文件夹。
- `--textgrid-dir`：MFA align 输出的原始 TextGrid 文件夹。
- `--wav-dir`：处理后的 wav 文件夹；用于自动修复短词后接长静音的错位问题。

输出：
- `--output-dir`：通过后处理和过滤的最终 TextGrid。
- `--filtered-dir`：被过滤掉的 TextGrid。
- `<output-dir>/postprocess_report.jsonl`：每条数据的处理报告。

核心可选参数：
- `--text-key` / `--wav-key`：JSONL 中文本和 wav 路径字段名，默认 `text` / `wav_file`。
- `--overwrite`：覆盖已有输出，并清理 post/filtered 两侧的同名旧文件。
- `--fix-short-multi-unit` / `--no-fix-short-multi-unit`：是否启用 `っ + て` 这类错位自动修复，默认启用。
- `--fix-short-word-sec`：被视为短词的最大时长，默认 `0.25` 秒。
- `--fix-min-silence-sec`：短词后方静音至少多长才尝试修复，默认 `0.8` 秒。
- `--fix-search-sec`：在后方静音中搜索真实发音的最长窗口，默认 `0.5` 秒。
- `--fix-threshold-ratio`：能量检测阈值倍率，默认 `4.0`。
- `--fix-t-split-ratio`：把能量段前多少比例分给 `t/tː`，默认 `0.22`。
- `--filter-suspicious-alignment` / `--no-filter-suspicious-alignment`：是否启用可疑对齐过滤，默认启用。
- `--filter-long-word-sec`：长词过滤参考阈值，默认 `1.2` 秒。
- `--filter-remaining-long-consonant-sec`：剩余超长辅音 phone 阈值，默认 `0.32` 秒。
- `--filter-remaining-long-vowel-sec`：剩余超长元音 phone 阈值，默认 `0.42` 秒。
- `--filter-remaining-short-phone-sec`：极短 phone 阈值，默认 `0.018` 秒。
- `--filter-remaining-min-phone-coverage`：word 内 phone 覆盖率最低阈值，默认 `0.35`。
- `--filter-remaining-edge-gap-sec`：word 边界与 phone 边界允许的最大空隙，默认 `0.28` 秒。
- `--copy-errors`：处理出错时，把原始 TextGrid 复制到 filtered 目录。

使用示例：
python scripts/postprocess_textgrids.py ^
  --jsonl data/wav/generated.jsonl ^
  --txt-dir data/txt_a_dict ^
  --textgrid-dir data/aligned_a_dict ^
  --output-dir data/aligned_a_dict_post ^
  --filtered-dir data/aligned_a_dict_filtered ^
  --wav-dir data/wav ^
  --overwrite


"""

import argparse
import array
import json
import math
import shutil
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    from pykakasi import kakasi
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pykakasi is not installed. Install it with:\n"
        "  pip install pykakasi"
    ) from exc

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

'''
python scripts/postprocess_textgrids.py ^
  --jsonl data/wav/generated.jsonl ^
  --txt-dir data/txt_c ^
  --textgrid-dir data/aligned_c ^
  --output-dir data/aligned_c_post ^
  --filtered-dir data/aligned_c_filtered ^
  --overwrite
'''


SILENCE_LABELS = {"<eps>", "<sil>", "sil", "<sp0>", "<sp1>", "<sp2>", "<sp3>"}
QUESTION_MARKS = {"?", "\uff1f"}
PUNCT_LIKE_SYMBOLS = set("\u301c\uff5e")
TARGET_PHONE_PAIRS = {("t", "e"), ("t:", "e"), ("t\u02d0", "e")}
SHORT_WORDS = {
    "\u3088", "\u306d", "\u3060", "\u3066", "\u3068", "\u306e", "\u306b",
    "\u304c", "\u3092", "\u306f", "\u3078", "\u3082",
    "yo", "ne", "da", "te", "to", "no", "ni", "ga", "o", "wa", "e", "mo",
}
VOWEL_PHONES = {"a", "i", "u", "e", "o", "\u0259", "\u0258", "\u026f", "\u0274"}


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str

    @property
    def duration(self) -> float:
        return self.xmax - self.xmin


@dataclass
class Tier:
    name: str
    xmin: float
    xmax: float
    intervals: list[Interval]


@dataclass
class TextGrid:
    xmin: float
    xmax: float
    tiers: list[Tier]


def is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P") or char in PUNCT_LIKE_SYMBOLS


def strip_punctuation(text: str) -> str:
    return "".join(char for char in text if not is_punctuation(char))


def parse_textgrid(path: Path) -> TextGrid:
    lines = path.read_text(encoding="utf-8").splitlines()
    xmin = 0.0
    xmax = 0.0
    tiers: list[Tier] = []
    current: Tier | None = None
    pending_xmin: float | None = None
    pending_xmax: float | None = None
    in_items = False
    in_interval = False

    for raw_line in lines:
        line = raw_line.strip()
        if line == "item []:":
            in_items = True
            continue
        if not in_items:
            if line.startswith("xmin = "):
                xmin = float(line.split("=", 1)[1])
            elif line.startswith("xmax = "):
                xmax = float(line.split("=", 1)[1])
            continue
        if line.startswith("item ["):
            if current is not None:
                tiers.append(current)
            current = Tier(name="", xmin=xmin, xmax=xmax, intervals=[])
            pending_xmin = None
            pending_xmax = None
            in_interval = False
        elif current is not None and line.startswith("name = "):
            current.name = unquote_textgrid(line.split("=", 1)[1].strip())
        elif current is not None and line.startswith("xmin = "):
            value = float(line.split("=", 1)[1])
            if in_interval:
                pending_xmin = value
            else:
                current.xmin = value
        elif current is not None and line.startswith("xmax = "):
            value = float(line.split("=", 1)[1])
            if in_interval:
                pending_xmax = value
            else:
                current.xmax = value
        elif current is not None and line.startswith("intervals ["):
            pending_xmin = None
            pending_xmax = None
            in_interval = True
        elif current is not None and line.startswith("text = "):
            text = unquote_textgrid(line.split("=", 1)[1].strip())
            if pending_xmin is None or pending_xmax is None:
                raise ValueError(f"Malformed interval near line: {raw_line}")
            current.intervals.append(Interval(pending_xmin, pending_xmax, text))
            pending_xmin = None
            pending_xmax = None
            in_interval = False

    if current is not None:
        tiers.append(current)
    if not tiers:
        raise ValueError(f"No tiers found in {path}")
    return TextGrid(xmin=xmin, xmax=xmax, tiers=tiers)


def unquote_textgrid(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('""', '"')


def quote_textgrid(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def write_textgrid(textgrid: TextGrid, path: Path) -> None:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {format_time(textgrid.xmin)} ",
        f"xmax = {format_time(textgrid.xmax)} ",
        "tiers? <exists> ",
        f"size = {len(textgrid.tiers)} ",
        "item []: ",
    ]
    for tier_index, tier in enumerate(textgrid.tiers, start=1):
        lines.extend(
            [
                f"    item [{tier_index}]:",
                '        class = "IntervalTier" ',
                f"        name = {quote_textgrid(tier.name)} ",
                f"        xmin = {format_time(tier.xmin)} ",
                f"        xmax = {format_time(tier.xmax)} ",
                f"        intervals: size = {len(tier.intervals)} ",
            ]
        )
        for interval_index, interval in enumerate(tier.intervals, start=1):
            lines.extend(
                [
                    f"        intervals [{interval_index}]:",
                    f"            xmin = {format_time(interval.xmin)} ",
                    f"            xmax = {format_time(interval.xmax)} ",
                    f"            text = {quote_textgrid(interval.text)} ",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_time(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def silence_by_duration(interval: Interval) -> str:
    duration = interval.duration
    if duration < 0.2:
        return "<sp0>"
    if duration < 0.5:
        return "<sp1>"
    if duration < 1.5:
        return "<sp2>"
    return "<sp3>"


def relabel_silences(intervals: list[Interval]) -> list[Interval]:
    relabeled = []
    for interval in intervals:
        text = interval.text.strip()
        if is_silence_mark(text):
            text = silence_by_duration(interval)
        relabeled.append(Interval(interval.xmin, interval.xmax, text))
    return relabeled


def relabel_all_silences(textgrid: TextGrid) -> TextGrid:
    return TextGrid(
        textgrid.xmin,
        textgrid.xmax,
        [
            Tier(tier.name, tier.xmin, tier.xmax, relabel_silences(tier.intervals))
            for tier in textgrid.tiers
        ],
    )


def load_texts(jsonl_path: Path, text_key: str, wav_key: str) -> dict[str, str]:
    texts = {}
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get(text_key)
            wav_file = item.get(wav_key)
            if not isinstance(text, str) or not isinstance(wav_file, str):
                raise ValueError(f"Line {line_no} missing {text_key!r} or {wav_key!r}")
            texts[Path(wav_file).stem] = text
    return texts


def load_wav_paths(jsonl_path: Path, wav_key: str) -> dict[str, Path]:
    wav_paths = {}
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            wav_file = item.get(wav_key)
            if isinstance(wav_file, str):
                wav_paths[Path(wav_file).stem] = Path(wav_file)
    return wav_paths


def attach_question_marks(raw_text: str, tokens: list[str]) -> tuple[list[str], list[str]]:
    base_tokens = [strip_punctuation(token) for token in tokens if strip_punctuation(token)]
    punctuated = base_tokens[:]
    warnings = []
    raw_index = 0
    last_token_index: int | None = None

    for token_index, token in enumerate(base_tokens):
        for char in token:
            while raw_index < len(raw_text) and is_punctuation(raw_text[raw_index]):
                if raw_text[raw_index] in QUESTION_MARKS:
                    target = last_token_index if last_token_index is not None else token_index
                    punctuated[target] += raw_text[raw_index]
                raw_index += 1
            if raw_index >= len(raw_text) or raw_text[raw_index] != char:
                warnings.append(
                    f"question insertion alignment mismatch at token {token_index + 1}"
                )
                return tokens[:], warnings
            raw_index += 1
        last_token_index = token_index

    while raw_index < len(raw_text):
        if raw_text[raw_index] in QUESTION_MARKS and punctuated:
            punctuated[-1] += raw_text[raw_index]
        raw_index += 1

    return punctuated, warnings


QUESTION_MARK = "\uff1f"


def is_silence_mark(text: str) -> bool:
    text = text.strip()
    return text in SILENCE_LABELS or text.startswith("<sp")


def token_to_romaji(token: str, kakasi_converter) -> str:
    if token in QUESTION_MARKS:
        return QUESTION_MARK
    if is_silence_mark(token):
        return token
    base = strip_punctuation(token)
    if not base:
        return token
    converted = kakasi_converter.convert(base)
    romaji = "".join(item.get("hepburn", item.get("orig", "")) for item in converted)
    return romaji or base


def make_romaji_tier(words_tier: Tier, kakasi_converter) -> Tier:
    intervals = []
    for interval in words_tier.intervals:
        intervals.append(
            Interval(
                interval.xmin,
                interval.xmax,
                token_to_romaji(interval.text, kakasi_converter),
            )
        )
    return Tier("romaji", words_tier.xmin, words_tier.xmax, intervals)


def locate_question_tokens(raw_text: str, words: list[str]) -> tuple[list[dict], list[str]]:
    warnings = []
    questions = []
    raw_index = 0
    last_word_index: int | None = None
    clean_words = [strip_punctuation(word) for word in words]

    for word_index, word in enumerate(clean_words):
        if not word:
            continue

        while raw_index < len(raw_text) and is_punctuation(raw_text[raw_index]):
            if raw_text[raw_index] in QUESTION_MARKS:
                questions.append(
                    {
                        "mark": QUESTION_MARK,
                        "prev_word_index": last_word_index,
                        "next_word_index": word_index,
                    }
                )
            raw_index += 1

        for char in word:
            while raw_index < len(raw_text) and is_punctuation(raw_text[raw_index]):
                if raw_text[raw_index] in QUESTION_MARKS:
                    questions.append(
                        {
                            "mark": QUESTION_MARK,
                            "prev_word_index": last_word_index,
                            "next_word_index": word_index,
                        }
                    )
                raw_index += 1
            if raw_index >= len(raw_text) or raw_text[raw_index] != char:
                warnings.append(
                    f"question interval alignment mismatch at MFA word {word_index + 1}"
                )
                return [], warnings
            raw_index += 1
        last_word_index = word_index

    while raw_index < len(raw_text):
        if raw_text[raw_index] in QUESTION_MARKS:
            questions.append(
                {
                    "mark": QUESTION_MARK,
                    "prev_word_index": last_word_index,
                    "next_word_index": None,
                }
            )
        raw_index += 1

    return questions, warnings


def find_silence_interval_for_question(
    words_tier: Tier,
    non_silence_interval_indices: list[int],
    question: dict,
) -> int | None:
    prev_word_index = question["prev_word_index"]
    next_word_index = question["next_word_index"]

    if prev_word_index is None and next_word_index is None:
        return None
    if prev_word_index is None:
        start = 0
        end = non_silence_interval_indices[next_word_index]
    elif next_word_index is None:
        start = non_silence_interval_indices[prev_word_index] + 1
        end = len(words_tier.intervals)
    else:
        start = non_silence_interval_indices[prev_word_index] + 1
        end = non_silence_interval_indices[next_word_index]

    for interval_index in range(start, end):
        if is_silence_mark(words_tier.intervals[interval_index].text):
            return interval_index
    return None


def sync_phone_question_interval(phones_tier: Tier, start_time: float, end_time: float) -> Tier:
    eps = 1e-4
    intervals = []
    for phone in phones_tier.intervals:
        if phone.xmin >= start_time - eps and phone.xmax <= end_time + eps:
            intervals.append(Interval(phone.xmin, phone.xmax, QUESTION_MARK))
        else:
            intervals.append(phone)
    return Tier(phones_tier.name, phones_tier.xmin, phones_tier.xmax, intervals)


def add_question_intervals(
    raw_text: str,
    words_tier: Tier,
    phones_tier: Tier,
) -> tuple[Tier, Tier, list[str], dict]:
    warnings = []
    words = non_silence_texts(words_tier)
    questions, question_warnings = locate_question_tokens(raw_text, words)
    warnings.extend(question_warnings)
    if question_warnings:
        return words_tier, phones_tier, warnings, {
            "question_total": len([char for char in raw_text if char in QUESTION_MARKS]),
            "question_filled": 0,
            "question_dropped": len([char for char in raw_text if char in QUESTION_MARKS]),
        }

    word_intervals = [Interval(i.xmin, i.xmax, i.text) for i in words_tier.intervals]
    updated_words = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, word_intervals)
    updated_phones = phones_tier
    non_silence_indices = [
        index for index, interval in enumerate(updated_words.intervals)
        if not is_silence_mark(interval.text)
    ]
    filled = 0
    dropped = 0
    used_silence_indices = set()

    for question in questions:
        interval_index = find_silence_interval_for_question(
            updated_words, non_silence_indices, question
        )
        if interval_index is None or interval_index in used_silence_indices:
            dropped += 1
            warnings.append("question dropped: no matching silence interval")
            continue
        used_silence_indices.add(interval_index)
        target = updated_words.intervals[interval_index]
        updated_words.intervals[interval_index] = Interval(
            target.xmin, target.xmax, question["mark"]
        )
        updated_phones = sync_phone_question_interval(updated_phones, target.xmin, target.xmax)
        filled += 1

    return updated_words, updated_phones, warnings, {
        "question_total": len(questions),
        "question_filled": filled,
        "question_dropped": dropped,
    }


def get_filter_reasons(textgrid: TextGrid) -> list[str]:
    reasons = []
    has_sp3 = False
    has_spn = False
    for tier in textgrid.tiers:
        for interval in tier.intervals:
            text = interval.text.strip()
            if text == "<sp3>":
                has_sp3 = True
            if text.lower() == "spn":
                has_spn = True
    if has_sp3:
        reasons.append("sp3")
    if has_spn:
        reasons.append("spn")
    return reasons


def non_silence_texts(tier: Tier) -> list[str]:
    return [
        interval.text
        for interval in tier.intervals
        if not is_silence_mark(interval.text)
    ]


def tier_by_name(textgrid: TextGrid, name: str) -> Tier | None:
    for tier in textgrid.tiers:
        if tier.name.lower() == name.lower():
            return tier
    return None


def normalize_phone(text: str) -> str:
    return text.strip().lower().replace("\u02d0", ":")


def is_vowel_phone(text: str) -> bool:
    normalized = normalize_phone(text).replace(":", "")
    return normalized in VOWEL_PHONES


def overlapping_indices(tier: Tier, start: float, end: float, eps: float = 1e-4) -> list[int]:
    return [
        index for index, interval in enumerate(tier.intervals)
        if interval.xmax > start + eps and interval.xmin < end - eps
    ]


def overlapping_intervals(tier: Tier, start: float, end: float, eps: float = 1e-4) -> list[Interval]:
    return [
        interval for interval in tier.intervals
        if interval.xmax > start + eps and interval.xmin < end - eps
    ]


def overlap_duration(interval: Interval, start: float, end: float) -> float:
    return max(0.0, min(interval.xmax, end) - max(interval.xmin, start))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load_audio(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        return [(sample - 128) / 128.0 for sample in frames[::channels]], sr

    if sample_width in {2, 4}:
        type_code = "h" if sample_width == 2 else "i"
        scale = float(2 ** (8 * sample_width - 1))
        samples = array.array(type_code)
        samples.frombytes(frames)
        return [samples[index] / scale for index in range(0, len(samples), channels)], sr

    if sample_width == 3:
        values = []
        frame_size = sample_width * channels
        scale = float(2 ** 23)
        for offset in range(0, len(frames), frame_size):
            sample_bytes = frames[offset:offset + 3]
            sample = int.from_bytes(sample_bytes, "little", signed=False)
            if sample >= 2 ** 23:
                sample -= 2 ** 24
            values.append(sample / scale)
        return values, sr

    raise ValueError(f"Unsupported wav sample width: {sample_width}")


def frame_rms(audio: list[float], frame_size: int, hop_size: int) -> list[float]:
    if len(audio) < frame_size:
        return []
    values = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[start:start + frame_size]
        values.append(math.sqrt(sum(sample * sample for sample in frame) / len(frame)) + 1e-12)
    return values


def find_speech_region_in_silence(
    audio: list[float],
    sr: int,
    silence_start: float,
    silence_end: float,
    search_sec: float,
    frame_ms: float,
    hop_ms: float,
    threshold_ratio: float,
    min_region_sec: float,
    min_trailing_silence_sec: float,
) -> tuple[float, float] | None:
    search_end = min(silence_end, silence_start + search_sec)
    start_sample = max(0, int(silence_start * sr))
    end_sample = min(len(audio), int(search_end * sr))
    if end_sample <= start_sample:
        return None

    segment = audio[start_sample:end_sample]
    frame_size = max(1, int(frame_ms / 1000.0 * sr))
    hop_size = max(1, int(hop_ms / 1000.0 * sr))
    rms = frame_rms(segment, frame_size, hop_size)
    if not rms:
        return None

    tail = rms[max(0, int(len(rms) * 0.6)):]
    noise_floor = median(tail) if tail else median(rms)
    peak = max(rms)
    threshold = max(noise_floor * threshold_ratio, peak * 0.15)
    active = [value > threshold for value in rms]
    min_frames = max(1, int(min_region_sec / (hop_ms / 1000.0)))
    trailing_frames = max(1, int(min_trailing_silence_sec / (hop_ms / 1000.0)))

    first = None
    for index in range(len(active)):
        if sum(active[index:index + min_frames]) >= min_frames:
            first = index
            break
    if first is None:
        return None

    last = None
    index = first
    while index < len(active):
        if not active[index] and sum(active[index:index + trailing_frames]) == 0:
            last = index
            break
        index += 1
    if last is None:
        last = max(index for index, value in enumerate(active) if value) + 1

    speech_start = silence_start + first * hop_ms / 1000.0
    speech_end = silence_start + last * hop_ms / 1000.0 + frame_ms / 1000.0
    speech_end = min(speech_end, silence_end)
    if speech_end - speech_start < min_region_sec:
        return None
    if speech_start - silence_start > 0.35:
        return None
    return speech_start, speech_end


def candidate_short_multi_indices(words: Tier, short_word_sec: float, min_silence_sec: float) -> list[int]:
    candidates = []
    for index, interval in enumerate(words.intervals[:-1]):
        text = interval.text.strip()
        next_interval = words.intervals[index + 1]
        if (
            len(text) >= 3
            and ("\u3063" in text or "\u30c3" in text)
            and interval.duration < short_word_sec
            and is_silence_mark(next_interval.text)
            and next_interval.duration >= min_silence_sec
        ):
            candidates.append(index)
    return candidates


def fix_short_multi_unit_before_silence(
    textgrid: TextGrid,
    wav_path: Path | None,
    args,
) -> tuple[TextGrid, list[dict], list[str]]:
    warnings = []
    fixes = []
    if wav_path is None:
        return textgrid, fixes, ["short_multi_unit fix skipped: missing wav"]

    words = tier_by_name(textgrid, "words")
    romaji = tier_by_name(textgrid, "romaji")
    phones = tier_by_name(textgrid, "phones")
    if words is None or phones is None:
        return textgrid, fixes, ["short_multi_unit fix skipped: missing words/phones"]

    candidates = candidate_short_multi_indices(
        words,
        short_word_sec=args.fix_short_word_sec,
        min_silence_sec=args.fix_min_silence_sec,
    )
    if not candidates:
        return textgrid, fixes, warnings

    audio, sr = load_audio(wav_path)
    for word_index in candidates:
        word = words.intervals[word_index]
        silence = words.intervals[word_index + 1]
        phone_indices = overlapping_indices(phones, word.xmin, word.xmax)
        non_sil_phone_indices = [
            index for index in phone_indices
            if not is_silence_mark(phones.intervals[index].text)
        ]
        if len(non_sil_phone_indices) < 2:
            continue

        p1_index = non_sil_phone_indices[-2]
        p2_index = non_sil_phone_indices[-1]
        following_phone_index = p2_index + 1
        if p2_index != p1_index + 1 or following_phone_index >= len(phones.intervals):
            continue

        following_phone = phones.intervals[following_phone_index]
        if not is_silence_mark(following_phone.text):
            continue

        p1 = phones.intervals[p1_index]
        p2 = phones.intervals[p2_index]
        if (normalize_phone(p1.text), normalize_phone(p2.text)) not in TARGET_PHONE_PAIRS:
            continue

        region = find_speech_region_in_silence(
            audio=audio,
            sr=sr,
            silence_start=silence.xmin,
            silence_end=silence.xmax,
            search_sec=args.fix_search_sec,
            frame_ms=args.fix_frame_ms,
            hop_ms=args.fix_hop_ms,
            threshold_ratio=args.fix_threshold_ratio,
            min_region_sec=args.fix_min_region_sec,
            min_trailing_silence_sec=args.fix_min_trailing_silence_sec,
        )
        if region is None:
            continue

        speech_start, speech_end = region
        t_end = speech_start + (speech_end - speech_start) * args.fix_t_split_ratio
        old_p1_xmin = p1.xmin
        old_following_phone_xmax = following_phone.xmax
        if not (old_p1_xmin < speech_start < t_end < speech_end <= old_following_phone_xmax):
            continue
        if speech_end <= word.xmax or speech_end >= silence.xmax:
            continue

        old_word_xmax = word.xmax
        word.xmax = speech_end
        silence.xmin = speech_end
        if romaji is not None and len(romaji.intervals) == len(words.intervals):
            romaji.intervals[word_index].xmax = speech_end
            romaji.intervals[word_index + 1].xmin = speech_end

        p1.xmin = old_p1_xmin
        p1.xmax = t_end
        p2.xmin = t_end
        p2.xmax = speech_end
        replacement = [p1, p2]
        if old_following_phone_xmax - speech_end > 1e-5:
            replacement.append(Interval(speech_end, old_following_phone_xmax, following_phone.text))
        phones.intervals = (
            phones.intervals[:p1_index]
            + replacement
            + phones.intervals[following_phone_index + 1:]
        )
        fixes.append(
            {
                "rule": "short_multi_unit_word_before_long_silence",
                "word_index": word_index + 1,
                "word": word.text,
                "old_word_xmax": round(old_word_xmax, 6),
                "new_word_xmax": round(speech_end, 6),
                "t_start": round(old_p1_xmin, 6),
                "t_end": round(t_end, 6),
                "speech_start": round(speech_start, 6),
                "speech_end": round(speech_end, 6),
                "phones": [p1.text, p2.text],
            }
        )

    return textgrid, fixes, warnings


def add_filter_issue(issues: list[dict], rule: str, interval: Interval, **extra) -> None:
    item = {
        "rule": rule,
        "start": round(interval.xmin, 6),
        "end": round(interval.xmax, 6),
        "duration": round(interval.duration, 6),
        "text": interval.text,
    }
    item.update(extra)
    issues.append(item)


def detect_alignment_filter_issues(
    textgrid: TextGrid,
    long_phone_sec: float,
    long_vowel_sec: float,
    long_word_sec: float,
    flank_silence_sec: float,
    remaining_long_consonant_sec: float,
    remaining_long_vowel_sec: float,
    remaining_short_phone_sec: float,
    remaining_min_word_sec: float,
    remaining_min_phone_coverage: float,
    remaining_edge_gap_sec: float,
) -> list[dict]:
    issues = []
    phone_evidence = []
    words = tier_by_name(textgrid, "words")
    phones = tier_by_name(textgrid, "phones")
    if words is None or phones is None:
        return [{"rule": "missing_words_or_phones_tier"}]

    for interval in phones.intervals:
        text = interval.text.strip()
        if not text or is_silence_mark(text) or text in QUESTION_MARKS:
            continue
        if is_vowel_phone(text) and interval.duration > long_vowel_sec:
            add_filter_issue(phone_evidence, "long_vowel_phone", interval)
        elif not is_vowel_phone(text) and interval.duration > long_phone_sec:
            add_filter_issue(phone_evidence, "long_consonant_phone", interval)

    for index, interval in enumerate(words.intervals):
        text = interval.text.strip()
        if not text or is_silence_mark(text) or text in QUESTION_MARKS:
            continue
        phone_hits = [
            phone for phone in overlapping_intervals(phones, interval.xmin, interval.xmax)
            if not is_silence_mark(phone.text) and phone.text.strip() not in QUESTION_MARKS
        ]
        if not phone_hits:
            add_filter_issue(issues, "word_without_non_silence_phone", interval)
        else:
            coverage = sum(
                overlap_duration(phone, interval.xmin, interval.xmax)
                for phone in phone_hits
            ) / max(interval.duration, 1e-6)
            phone_span_start = min(phone.xmin for phone in phone_hits)
            phone_span_end = max(phone.xmax for phone in phone_hits)
            start_gap = max(0.0, phone_span_start - interval.xmin)
            end_gap = max(0.0, interval.xmax - phone_span_end)
            if interval.duration >= remaining_min_word_sec and coverage < remaining_min_phone_coverage:
                add_filter_issue(
                    issues,
                    "low_phone_coverage_in_word",
                    interval,
                    coverage=round(coverage, 3),
                )
            if start_gap > remaining_edge_gap_sec or end_gap > remaining_edge_gap_sec:
                add_filter_issue(
                    issues,
                    "large_word_phone_edge_gap",
                    interval,
                    start_gap=round(start_gap, 6),
                    end_gap=round(end_gap, 6),
                )

        if interval.duration > long_word_sec:
            related_phone_evidence = [
                item for item in phone_evidence
                if item["start"] >= interval.xmin - 1e-4 and item["end"] <= interval.xmax + 1e-4
            ]
            if related_phone_evidence:
                add_filter_issue(
                    issues,
                    "long_word_with_long_phone",
                    interval,
                    phone_evidence=related_phone_evidence[:5],
                )

        prev_interval = words.intervals[index - 1] if index > 0 else None
        next_interval = words.intervals[index + 1] if index + 1 < len(words.intervals) else None
        if (
            text in SHORT_WORDS
            and interval.duration < 0.12
            and prev_interval is not None
            and next_interval is not None
            and is_silence_mark(prev_interval.text)
            and is_silence_mark(next_interval.text)
            and prev_interval.duration >= flank_silence_sec
            and next_interval.duration >= flank_silence_sec
        ):
            add_filter_issue(
                issues,
                "short_word_between_long_silences",
                interval,
                prev_silence_duration=round(prev_interval.duration, 6),
                next_silence_duration=round(next_interval.duration, 6),
            )
    return issues


def detect_remaining_phone_filter_issues(
    textgrid: TextGrid,
    long_consonant_sec: float,
    long_vowel_sec: float,
    short_phone_sec: float,
) -> list[dict]:
    phones = tier_by_name(textgrid, "phones")
    if phones is None:
        return []
    issues = []
    for index, interval in enumerate(phones.intervals):
        text = interval.text.strip()
        if not text or is_silence_mark(text) or text in QUESTION_MARKS:
            continue
        if is_vowel_phone(text) and interval.duration > long_vowel_sec:
            add_filter_issue(
                issues,
                "very_long_vowel_phone",
                interval,
                phone_interval=index + 1,
            )
        elif not is_vowel_phone(text) and interval.duration > long_consonant_sec:
            add_filter_issue(
                issues,
                "very_long_consonant_phone",
                interval,
                phone_interval=index + 1,
            )
        elif interval.duration < short_phone_sec:
            add_filter_issue(
                issues,
                "extremely_short_phone",
                interval,
                phone_interval=index + 1,
            )
    return issues


def process_one(
    textgrid_path: Path,
    txt_dir: Path,
    raw_texts: dict[str, str],
    wav_paths: dict[str, Path],
    output_dir: Path,
    filtered_dir: Path,
    kakasi_converter,
    args,
) -> dict:
    stem = textgrid_path.stem
    report = {"stem": stem, "status": "ok", "warnings": []}
    if stem not in raw_texts:
        raise ValueError(f"No JSONL text for {stem}")
    txt_path = txt_dir / f"{stem}.txt"
    if not txt_path.exists():
        raise ValueError(f"Missing txt file: {txt_path}")

    textgrid = parse_textgrid(textgrid_path)
    if len(textgrid.tiers) < 2:
        raise ValueError(f"{textgrid_path} must have at least words and phones tiers")
    words_tier = textgrid.tiers[0]
    phones_tier = textgrid.tiers[1]
    raw_text = raw_texts[stem]
    txt_text = txt_path.read_text(encoding="utf-8").strip()
    txt_tokens = txt_text.split()
    punctuated_txt_tokens, txt_warnings = attach_question_marks(raw_text, txt_tokens)
    report["warnings"].extend(f"txt: {warning}" for warning in txt_warnings)

    mfa_words = non_silence_texts(words_tier)
    cleaned_raw = strip_punctuation(raw_text)
    cleaned_txt = "".join(strip_punctuation(token) for token in txt_tokens)
    if cleaned_raw != cleaned_txt:
        report["warnings"].append("raw text and txt tokens do not match after punctuation removal")
    cleaned_mfa_words = "".join(strip_punctuation(token) for token in mfa_words)
    if cleaned_raw != cleaned_mfa_words:
        report["warnings"].append("raw text and MFA words do not match after punctuation removal")

    words_with_questions, phones_with_questions, q_warnings, q_stats = add_question_intervals(
        raw_text, words_tier, phones_tier
    )
    report["warnings"].extend(q_warnings)
    report.update(q_stats)
    romaji_tier = make_romaji_tier(words_with_questions, kakasi_converter)

    raw_tier = Tier(
        "raw_text",
        textgrid.xmin,
        textgrid.xmax,
        [Interval(textgrid.xmin, textgrid.xmax, raw_text)],
    )
    tokenized_tier = Tier(
        "tokenized_text",
        textgrid.xmin,
        textgrid.xmax,
        [Interval(textgrid.xmin, textgrid.xmax, " ".join(punctuated_txt_tokens))],
    )
    new_textgrid = TextGrid(
        textgrid.xmin,
        textgrid.xmax,
        [
            raw_tier,
            tokenized_tier,
            words_with_questions,
            romaji_tier,
            phones_with_questions,
        ],
    )

    wav_path = wav_paths.get(stem)
    if args.wav_dir is not None:
        for suffix in [".wav"]:
            candidate = args.wav_dir / f"{stem}{suffix}"
            if candidate.exists():
                wav_path = candidate
                break
    if wav_path is not None and not wav_path.is_absolute():
        wav_path = Path.cwd() / wav_path
    if wav_path is not None and not wav_path.exists():
        wav_path = None

    if args.fix_short_multi_unit:
        new_textgrid, fixes, fix_warnings = fix_short_multi_unit_before_silence(
            new_textgrid,
            wav_path,
            args,
        )
        report["fixes"] = fixes
        report["fix_count"] = len(fixes)
        report["warnings"].extend(fix_warnings)

    alignment_filter_issues = []
    if args.filter_suspicious_alignment:
        alignment_filter_issues = detect_alignment_filter_issues(
            new_textgrid,
            long_phone_sec=args.filter_long_phone_sec,
            long_vowel_sec=args.filter_long_vowel_sec,
            long_word_sec=args.filter_long_word_sec,
            flank_silence_sec=args.filter_flank_silence_sec,
            remaining_long_consonant_sec=args.filter_remaining_long_consonant_sec,
            remaining_long_vowel_sec=args.filter_remaining_long_vowel_sec,
            remaining_short_phone_sec=args.filter_remaining_short_phone_sec,
            remaining_min_word_sec=args.filter_remaining_min_word_sec,
            remaining_min_phone_coverage=args.filter_remaining_min_phone_coverage,
            remaining_edge_gap_sec=args.filter_remaining_edge_gap_sec,
        )
        alignment_filter_issues.extend(
            detect_remaining_phone_filter_issues(
                new_textgrid,
                long_consonant_sec=args.filter_remaining_long_consonant_sec,
                long_vowel_sec=args.filter_remaining_long_vowel_sec,
                short_phone_sec=args.filter_remaining_short_phone_sec,
            )
        )

    new_textgrid = relabel_all_silences(new_textgrid)

    filter_reasons = get_filter_reasons(new_textgrid)
    if alignment_filter_issues:
        filter_reasons.append("suspicious_alignment")
    if filter_reasons:
        output_path = filtered_dir / textgrid_path.name
        stale_path = output_dir / textgrid_path.name
        report["status"] = "filtered_" + "_".join(filter_reasons)
        report["filter_reasons"] = filter_reasons
        if alignment_filter_issues:
            report["alignment_filter_issues"] = alignment_filter_issues
    else:
        output_path = output_dir / textgrid_path.name
        stale_path = filtered_dir / textgrid_path.name

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if stale_path.exists() and args.overwrite:
        stale_path.unlink()
    write_textgrid(new_textgrid, output_path)
    report["output"] = str(output_path)
    report["word_count"] = len([i for i in words_tier.intervals if i.text not in SILENCE_LABELS])
    report["txt_token_count"] = len(punctuated_txt_tokens)
    report["mfa_word_count"] = len(mfa_words)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add raw text, tokenized text, and romaji tiers to MFA TextGrid files."
    )
    parser.add_argument("--jsonl", type=Path, default=Path("data/wav/generated.jsonl"))
    parser.add_argument("--txt-dir", type=Path, default=Path("data/txt_c"))
    parser.add_argument("--textgrid-dir", type=Path, default=Path("data/aligned_c"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, default=None)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--wav-key", default="wav_file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fix-short-multi-unit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix-short-word-sec", type=float, default=0.25)
    parser.add_argument("--fix-min-silence-sec", type=float, default=0.8)
    parser.add_argument("--fix-search-sec", type=float, default=0.5)
    parser.add_argument("--fix-frame-ms", type=float, default=10.0)
    parser.add_argument("--fix-hop-ms", type=float, default=5.0)
    parser.add_argument("--fix-threshold-ratio", type=float, default=4.0)
    parser.add_argument("--fix-t-split-ratio", type=float, default=0.22)
    parser.add_argument("--fix-min-region-sec", type=float, default=0.08)
    parser.add_argument("--fix-min-trailing-silence-sec", type=float, default=0.08)
    parser.add_argument("--filter-suspicious-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-long-phone-sec", type=float, default=0.25)
    parser.add_argument("--filter-long-vowel-sec", type=float, default=0.5)
    parser.add_argument("--filter-long-word-sec", type=float, default=1.2)
    parser.add_argument("--filter-flank-silence-sec", type=float, default=0.5)
    parser.add_argument("--filter-remaining-long-consonant-sec", type=float, default=0.32)
    parser.add_argument("--filter-remaining-long-vowel-sec", type=float, default=0.42)
    parser.add_argument("--filter-remaining-short-phone-sec", type=float, default=0.018)
    parser.add_argument("--filter-remaining-min-word-sec", type=float, default=0.18)
    parser.add_argument("--filter-remaining-min-phone-coverage", type=float, default=0.35)
    parser.add_argument("--filter-remaining-edge-gap-sec", type=float, default=0.28)
    parser.add_argument(
        "--copy-errors",
        action="store_true",
        help="Copy files with processing errors to the filtered directory.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.filtered_dir.mkdir(parents=True, exist_ok=True)

    raw_texts = load_texts(args.jsonl, args.text_key, args.wav_key)
    wav_paths = load_wav_paths(args.jsonl, args.wav_key)
    kakasi_converter = kakasi()
    reports = []
    textgrid_paths = sorted(args.textgrid_dir.glob("*.TextGrid"))
    for textgrid_path in tqdm(textgrid_paths, desc="Postprocess TextGrids", unit="tg"):
        try:
            report = process_one(
                textgrid_path=textgrid_path,
                txt_dir=args.txt_dir,
                raw_texts=raw_texts,
                wav_paths=wav_paths,
                output_dir=args.output_dir,
                filtered_dir=args.filtered_dir,
                kakasi_converter=kakasi_converter,
                args=args,
            )
        except Exception as exc:
            report = {
                "stem": textgrid_path.stem,
                "status": "error",
                "error": str(exc),
                "warnings": [],
            }
            if args.copy_errors:
                shutil.copy2(textgrid_path, args.filtered_dir / textgrid_path.name)
        reports.append(report)

    report_path = args.output_dir / "postprocess_report.jsonl"
    with report_path.open("w", encoding="utf-8") as file:
        for report in reports:
            file.write(json.dumps(report, ensure_ascii=False) + "\n")

    counts = {}
    for report in reports:
        counts[report["status"]] = counts.get(report["status"], 0) + 1
    print(f"Done. {counts}. report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
