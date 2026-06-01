r"""
从 JSONL 元数据生成 MFA 需要的同名 TXT 转写文件。

这个脚本做什么：
- 读取 JSONL，每行一个 JSON。
- 使用 JSON 里的 `text` 字段作为文本，`wav_file` 字段作为输出文件名来源。
- 使用 SudachiPy 对日文文本分词，并删除标点符号。
- 保留日文长音、促音、拨音等假名字符。
- 为每个 WAV 生成一个同名 TXT，例如 `00001.wav -> 00001.txt`。
- 可选加载 MFA 字典，尽量减少 OOV：
  - 原 token 在字典中：保留原文。
  - 原 token 不在字典中，但 SudachiPy 读音假名在字典中：改写成假名。
  - 遇到 `千二十四` 这类汉字数字：转成日语数字读法假名，再尝试分词/查字典。
  - 连续多个 token 合并后在字典中：合并输出。
  - 连续多个 token 的读音假名合并后在字典中：合并并改写成假名。
- 使用字典时，会把最终仍然 OOV 的 token 统计到 TXT 输出目录外侧：
  `<output-dir-parent>/<output-dir-name>_final_oovs.txt`。

输入：
- `--jsonl`：包含文本和 wav 路径的 JSONL。

输出：
- `--output-dir`：MFA corpus 使用的 txt 文件夹，只包含 `.txt` 转写文件。
- `<output-dir-parent>/<output-dir-name>_final_oovs.txt`：最终 OOV 汇总，不放进 txt 文件夹，避免影响 MFA。

可选参数：
- `--jsonl`：输入 JSONL，默认 `data/wav/generated.jsonl`。
- `--output-dir`：输出 TXT 文件夹，必填。
- `--text-key`：JSON 中转写文本字段名，默认 `text`。
- `--wav-key`：JSON 中 wav 路径字段名，默认 `wav_file`。
- `--mode`：SudachiPy 分词模式，`A/B/C`，默认 `A`(粒度最细)。
- `--overwrite`：覆盖已存在的 TXT。
- `--dict-path`：MFA 字典路径；提供后启用 OOV 减少逻辑(一定要用)。
- `--max-merge-tokens`：尝试合并匹配字典的最大相邻 token 数，默认 `5`。
- `--oov-report`：自定义最终 OOV 汇总文件路径；不指定时默认写到 txt 文件夹外侧。

使用示例：
python scripts/generate_wav_txt.py --jsonl data/wav/generated.jsonl --output-dir data/txt_a_dict --overwrite --mode A --dict-path models\mfa\pretrained_models\dictionary\japanese_mfa.dict


"""

import argparse
from collections import Counter
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:
    from sudachipy import dictionary
    from sudachipy import tokenizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "SudachiPy is not installed. Install it with:\n"
        "  pip install SudachiPy sudachidict_core"
    ) from exc


PUNCT_LIKE_SYMBOLS = set("\u301c\uff5e")
KANJI_NUMERALS = set("零〇一二三四五六七八九十百千万萬億兆")
DIGIT_READINGS = {
    0: "\u308c\u3044",
    1: "\u3044\u3061",
    2: "\u306b",
    3: "\u3055\u3093",
    4: "\u3088\u3093",
    5: "\u3054",
    6: "\u308d\u304f",
    7: "\u306a\u306a",
    8: "\u306f\u3061",
    9: "\u304d\u3085\u3046",
}
KANJI_DIGIT_VALUES = {
    "\u96f6": 0,
    "\u3007": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
}
KANJI_UNIT_VALUES = {
    "\u5341": 10,
    "\u767e": 100,
    "\u5343": 1000,
}
KANJI_LARGE_UNIT_VALUES = {
    "\u4e07": 10_000,
    "\u842c": 10_000,
    "\u5104": 100_000_000,
    "\u5146": 1_000_000_000_000,
}


@dataclass(frozen=True)
class TokenInfo:
    surface: str
    reading_kata: str
    reading_hira: str


@dataclass(frozen=True)
class CorrectionOption:
    texts: list[str]
    span_len: int
    oov_count: int
    token_count: int
    priority: int
    action: str


@dataclass
class CorrectionStats:
    original_tokens: int = 0
    output_tokens: int = 0
    original_oovs: int = 0
    output_oovs: int = 0
    kana_rewrites: int = 0
    merged_tokens: int = 0


def is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P") or char in PUNCT_LIKE_SYMBOLS


def strip_punctuation(text: str) -> str:
    return "".join(char for char in text if not is_punctuation(char))


def katakana_to_hiragana(text: str) -> str:
    chars = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def hiragana_to_katakana(text: str) -> str:
    chars = []
    for char in text:
        code = ord(char)
        if 0x3041 <= code <= 0x3096:
            chars.append(chr(code + 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def parse_kanji_number(text: str) -> int | None:
    if not text or any(char not in KANJI_NUMERALS for char in text):
        return None

    total = 0
    section = 0
    number = 0
    saw_any = False

    for char in text:
        if char in KANJI_DIGIT_VALUES:
            number = KANJI_DIGIT_VALUES[char]
            saw_any = True
        elif char in KANJI_UNIT_VALUES:
            unit = KANJI_UNIT_VALUES[char]
            section += (number or 1) * unit
            number = 0
            saw_any = True
        elif char in KANJI_LARGE_UNIT_VALUES:
            unit = KANJI_LARGE_UNIT_VALUES[char]
            total += (section + number or 1) * unit
            section = 0
            number = 0
            saw_any = True
        else:
            return None

    if not saw_any:
        return None
    return total + section + number


def read_under_10000(value: int) -> str:
    if not 0 <= value < 10000:
        raise ValueError(f"value must be under 10000: {value}")
    if value == 0:
        return ""

    thousands = value // 1000
    hundreds = (value % 1000) // 100
    tens = (value % 100) // 10
    ones = value % 10
    parts = []

    if thousands:
        if thousands == 1:
            parts.append("\u305b\u3093")
        elif thousands == 3:
            parts.append("\u3055\u3093\u305c\u3093")
        elif thousands == 8:
            parts.append("\u306f\u3063\u305b\u3093")
        else:
            parts.append(DIGIT_READINGS[thousands] + "\u305b\u3093")

    if hundreds:
        if hundreds == 1:
            parts.append("\u3072\u3083\u304f")
        elif hundreds == 3:
            parts.append("\u3055\u3093\u3073\u3083\u304f")
        elif hundreds == 6:
            parts.append("\u308d\u3063\u3074\u3083\u304f")
        elif hundreds == 8:
            parts.append("\u306f\u3063\u3074\u3083\u304f")
        else:
            parts.append(DIGIT_READINGS[hundreds] + "\u3072\u3083\u304f")

    if tens:
        if tens == 1:
            parts.append("\u3058\u3085\u3046")
        else:
            parts.append(DIGIT_READINGS[tens] + "\u3058\u3085\u3046")

    if ones:
        parts.append(DIGIT_READINGS[ones])

    return "".join(parts)


def integer_to_japanese_reading(value: int) -> str | None:
    if value < 0:
        return None
    if value == 0:
        return DIGIT_READINGS[0]

    large_units = [
        (1_000_000_000_000, "\u3061\u3087\u3046"),
        (100_000_000, "\u304a\u304f"),
        (10_000, "\u307e\u3093"),
    ]
    parts = []
    remainder = value
    for unit_value, unit_reading in large_units:
        section = remainder // unit_value
        if section:
            section_reading = integer_to_japanese_reading(section)
            if section_reading is None:
                return None
            parts.append(section_reading + unit_reading)
            remainder %= unit_value
    if remainder:
        parts.append(read_under_10000(remainder))
    return "".join(parts)


def kanji_number_reading_candidates(text: str) -> list[tuple[str, str]]:
    value = parse_kanji_number(text)
    if value is None:
        return []
    reading_hira = integer_to_japanese_reading(value)
    if not reading_hira:
        return []
    reading_kata = hiragana_to_katakana(reading_hira)
    candidates = [("kanji_number_hira", reading_hira)]
    if reading_kata != reading_hira:
        candidates.append(("kanji_number_kata", reading_kata))
    return candidates


def split_text_by_dict(text: str, word_set: set[str], max_piece_chars: int = 8) -> list[str] | None:
    if not text:
        return None
    n = len(text)
    dp: list[tuple[tuple[int, int], list[str]] | None] = [None] * (n + 1)
    dp[n] = ((0, 0), [])

    for i in range(n - 1, -1, -1):
        best = None
        max_end = min(n, i + max_piece_chars)
        for end in range(i + 1, max_end + 1):
            piece = text[i:end]
            if piece not in word_set or dp[end] is None:
                continue
            next_score, next_pieces = dp[end]
            score = (1 + next_score[0], -(end - i) + next_score[1])
            result = (score, [piece] + next_pieces)
            if best is None or score < best[0]:
                best = result
        dp[i] = best

    if dp[0] is None:
        return None
    return dp[0][1]


def sudachi_segment_text(text: str, sudachi, mode) -> list[str]:
    pieces = []
    for morpheme in sudachi.tokenize(text, mode):
        surface = strip_punctuation(morpheme.surface()).strip()
        if surface:
            pieces.append(surface)
    return pieces


def tokenize_to_infos(text: str, sudachi, mode) -> list[TokenInfo]:
    tokens = []
    for morpheme in sudachi.tokenize(text, mode):
        surface = strip_punctuation(morpheme.surface()).strip()
        if not surface:
            continue

        reading = morpheme.reading_form()
        if not reading or reading == "*":
            reading = surface
        reading = strip_punctuation(reading).strip()
        tokens.append(
            TokenInfo(
                surface=surface,
                reading_kata=reading,
                reading_hira=katakana_to_hiragana(reading),
            )
        )
    return tokens


def load_mfa_dict(dict_path: Path) -> set[str]:
    words = set()
    with dict_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                word = line.split("\t", 1)[0].strip()
            else:
                word = line.split(maxsplit=1)[0].strip()
            if word:
                words.add(word)
    return words


def span_candidates(tokens: list[TokenInfo], start: int, end: int) -> list[tuple[str, str]]:
    span = tokens[start:end]
    surface = "".join(token.surface for token in span)
    reading_kata = "".join(token.reading_kata for token in span)
    reading_hira = "".join(token.reading_hira for token in span)

    candidates = [("surface", surface)]
    if reading_kata and reading_kata != surface:
        candidates.append(("reading_kata", reading_kata))
    if reading_hira and reading_hira not in {surface, reading_kata}:
        candidates.append(("reading_hira", reading_hira))
    for action, candidate in kanji_number_reading_candidates(surface):
        if candidate not in {value for _, value in candidates}:
            candidates.append((action, candidate))
    return candidates


def best_dict_correction(
    tokens: list[TokenInfo],
    word_set: set[str],
    max_merge_tokens: int,
    sudachi,
    mode,
) -> tuple[list[str], CorrectionStats]:
    n = len(tokens)
    stats = CorrectionStats(original_tokens=n)
    stats.original_oovs = sum(1 for token in tokens if token.surface not in word_set)

    # dp[i] = (score_tuple, output_tokens, options) for tokens[i:].
    dp: list[tuple[tuple[int, int, int, int], list[str], list[CorrectionOption]] | None] = [
        None
    ] * (n + 1)
    dp[n] = ((0, 0, 0, 0), [], [])

    for i in range(n - 1, -1, -1):
        best = None
        max_end = min(n, i + max_merge_tokens)

        for end in range(i + 1, max_end + 1):
            span_len = end - i
            span_surface = "".join(token.surface for token in tokens[i:end])
            is_kanji_number_span = parse_kanji_number(span_surface) is not None
            valid_options: list[CorrectionOption] = []

            for candidate_index, (action, candidate) in enumerate(span_candidates(tokens, i, end)):
                if candidate in word_set:
                    valid_options.append(
                        CorrectionOption(
                            texts=[candidate],
                            span_len=span_len,
                            oov_count=0,
                            token_count=1,
                            priority=candidate_index,
                            action=action,
                        )
                    )

                if action.startswith("kanji_number_") or (
                    is_kanji_number_span and action in {"reading_kata", "reading_hira"}
                ):
                    sudachi_pieces = sudachi_segment_text(candidate, sudachi, mode)
                    if (
                        len(sudachi_pieces) > 1
                        and all(piece in word_set for piece in sudachi_pieces)
                    ):
                        valid_options.append(
                            CorrectionOption(
                                texts=sudachi_pieces,
                                span_len=span_len,
                                oov_count=0,
                                token_count=len(sudachi_pieces),
                                priority=10 + candidate_index,
                                action=f"{action}_sudachi_split",
                            )
                        )

                    dict_pieces = split_text_by_dict(candidate, word_set)
                    if dict_pieces is not None and len(dict_pieces) > 1:
                        valid_options.append(
                            CorrectionOption(
                                texts=dict_pieces,
                                span_len=span_len,
                                oov_count=0,
                                token_count=len(dict_pieces),
                                priority=20 + candidate_index,
                                action=f"{action}_dict_split",
                            )
                        )

            # Fallback only keeps one original token. Do not merge unknown spans.
            if span_len == 1 and not valid_options:
                valid_options.append(
                    CorrectionOption(
                        texts=[tokens[i].surface],
                        span_len=1,
                        oov_count=1,
                        token_count=1,
                        priority=99,
                        action="fallback_oov",
                    )
                )

            for option in valid_options:
                next_dp = dp[end]
                if next_dp is None:
                    continue
                next_score, next_output, next_options = next_dp
                score = (
                    option.oov_count + next_score[0],
                    option.token_count + next_score[1],
                    option.priority + next_score[2],
                    -option.span_len + next_score[3],
                )
                candidate_result = (
                    score,
                    option.texts + next_output,
                    [option] + next_options,
                )
                if best is None or score < best[0]:
                    best = candidate_result

        if best is None:
            option = CorrectionOption(
                texts=[tokens[i].surface],
                span_len=1,
                oov_count=1,
                token_count=1,
                priority=99,
                action="fallback_oov",
            )
            next_score, next_output, next_options = dp[i + 1]
            best = (
                (
                    option.oov_count + next_score[0],
                    option.token_count + next_score[1],
                    option.priority + next_score[2],
                    -option.span_len + next_score[3],
                ),
                option.texts + next_output,
                [option] + next_options,
            )

        dp[i] = best

    _, output_tokens, options = dp[0]
    stats.output_tokens = len(output_tokens)
    stats.output_oovs = sum(1 for token in output_tokens if token not in word_set)
    stats.kana_rewrites = sum(
        1 for option in options if option.action in {"reading_kata", "reading_hira"}
    )
    stats.merged_tokens = sum(max(0, option.span_len - 1) for option in options)
    return output_tokens, stats


def wav_to_txt_name(wav_file: str) -> str:
    return f"{Path(wav_file).stem}.txt"


def process_jsonl(
    jsonl_path: Path,
    output_dir: Path,
    text_key: str,
    wav_key: str,
    mode,
    overwrite: bool,
    dict_path: Path | None,
    max_merge_tokens: int,
    oov_report: Path | None,
) -> tuple[int, int, CorrectionStats]:
    sudachi = dictionary.Dictionary().create()
    output_dir.mkdir(parents=True, exist_ok=True)

    dict_words = None
    final_oov_counts: Counter[str] = Counter()
    total_stats = CorrectionStats()
    if dict_path is not None:
        print(f"Loading MFA dictionary from {dict_path} ...", file=sys.stderr)
        dict_words = load_mfa_dict(dict_path)
        print(f"Loaded {len(dict_words)} dictionary entries.", file=sys.stderr)

    written = 0
    skipped = 0
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            text = item.get(text_key)
            wav_file = item.get(wav_key)
            if not isinstance(text, str) or not isinstance(wav_file, str):
                skipped += 1
                print(
                    f"Skip line {line_no}: missing string field "
                    f"{text_key!r} or {wav_key!r}",
                    file=sys.stderr,
                )
                continue

            output_path = output_dir / wav_to_txt_name(wav_file)
            if output_path.exists() and not overwrite:
                skipped += 1
                print(f"Skip existing file: {output_path}", file=sys.stderr)
                continue

            token_infos = tokenize_to_infos(text, sudachi, mode)
            if dict_words is None:
                output_tokens = [token.surface for token in token_infos]
            else:
                output_tokens, stats = best_dict_correction(
                    token_infos,
                    dict_words,
                    max_merge_tokens=max_merge_tokens,
                    sudachi=sudachi,
                    mode=mode,
                )
                total_stats.original_tokens += stats.original_tokens
                total_stats.output_tokens += stats.output_tokens
                total_stats.original_oovs += stats.original_oovs
                total_stats.output_oovs += stats.output_oovs
                total_stats.kana_rewrites += stats.kana_rewrites
                total_stats.merged_tokens += stats.merged_tokens
                final_oov_counts.update(token for token in output_tokens if token not in dict_words)

            output_path.write_text(" ".join(output_tokens) + "\n", encoding="utf-8")
            written += 1

    if oov_report is None and dict_words is not None:
        oov_report = output_dir.parent / f"{output_dir.name}_final_oovs.txt"

    if oov_report is not None:
        if dict_words is None:
            print(
                "Warning: --oov-report was specified but --dict-path was not; "
                "OOV report was not written.",
                file=sys.stderr,
            )
        else:
            oov_report.parent.mkdir(parents=True, exist_ok=True)
            with oov_report.open("w", encoding="utf-8") as file:
                for token, count in final_oov_counts.most_common():
                    file.write(f"{token}\t{count}\n")
            print(
                f"Wrote final OOV report: {oov_report} "
                f"({len(final_oov_counts)} unique OOVs)",
                file=sys.stderr,
            )

    return written, skipped, total_stats


def parse_mode(value: str):
    modes = {
        "A": tokenizer.Tokenizer.SplitMode.A,
        "B": tokenizer.Tokenizer.SplitMode.B,
        "C": tokenizer.Tokenizer.SplitMode.C,
    }
    try:
        return modes[value.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("mode must be A, B, or C") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate same-name txt files from wav/text records in a JSONL file. "
            "Text is segmented by SudachiPy and punctuation is removed."
        )
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/wav/generated.jsonl"),
        help="Input JSONL file. Default: data/wav/generated.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where txt files will be written.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="JSON field containing the transcript text. Default: text",
    )
    parser.add_argument(
        "--wav-key",
        default="wav_file",
        help="JSON field containing the wav file name/path. Default: wav_file",
    )
    parser.add_argument(
        "--mode",
        type=parse_mode,
        default=tokenizer.Tokenizer.SplitMode.A,
        help="SudachiPy split mode: A, B, or C. Default: A",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite txt files that already exist.",
    )
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=None,
        help="Optional MFA dictionary file for OOV reduction, for example japanese_mfa.dict.",
    )
    parser.add_argument(
        "--max-merge-tokens",
        type=int,
        default=5,
        help="Maximum adjacent SudachiPy tokens to merge when matching the MFA dictionary. Default: 5",
    )
    parser.add_argument(
        "--oov-report",
        type=Path,
        default=None,
        help=(
            "Optional path to write final OOV token counts after dictionary correction. "
            "Default with --dict-path: <output-dir-parent>/<output-dir-name>_final_oovs.txt"
        ),
    )
    args = parser.parse_args()

    written, skipped, stats = process_jsonl(
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        text_key=args.text_key,
        wav_key=args.wav_key,
        mode=args.mode,
        overwrite=args.overwrite,
        dict_path=args.dict_path,
        max_merge_tokens=args.max_merge_tokens,
        oov_report=args.oov_report,
    )
    print(f"Done. written={written}, skipped={skipped}")
    if args.dict_path is not None:
        print(
            "Dictionary correction stats: "
            f"original_tokens={stats.original_tokens}, "
            f"output_tokens={stats.output_tokens}, "
            f"original_oovs={stats.original_oovs}, "
            f"output_oovs={stats.output_oovs}, "
            f"kana_rewrites={stats.kana_rewrites}, "
            f"merged_tokens={stats.merged_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
