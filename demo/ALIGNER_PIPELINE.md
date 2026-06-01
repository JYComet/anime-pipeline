# Japanese MFA Aligner Pipeline

这个文档记录从原始音频到最终 TextGrid 的完整流程。推荐在 Windows CMD 中运行下面命令。

## 1. 环境配置

建议使用独立 conda 环境：

```bat
conda create -n mfa_japanese python=3.11 -y
conda activate mfa_japanese
conda install -c conda-forge montreal-forced-aligner -y
pip install -r requirements.txt
```

下载 MFA 日文模型和字典：

```bat
set "MFA_ROOT_DIR=models\mfa" && mfa model download acoustic japanese_mfa && mfa model download dictionary japanese_mfa
```

目录约定：

- `data/wav`：处理后的 wav，供 MFA 使用。
- `data/txt_a`：给 MFA 使用的同名 txt 转写文件。
- `data/txt_a_final_oovs.txt`：最终 OOV 汇总，不放进 txt 文件夹，避免影响 MFA。
- `data/aligned_a`：MFA 原始对齐输出。
- `data/aligned_a_post`：后处理后可用的 TextGrid。
- `data/aligned_a_filtered`：后处理中过滤掉的 TextGrid。
- `models/mfa`：MFA 模型缓存目录。
- `models/temp`：MFA 临时目录。

## 2. 裁剪音频静音

脚本：[scripts/trim_silence_batch.py](scripts/trim_silence_batch.py)

作用：

- 递归读取输入 wav。
- 裁剪过长内部静音。
- 可选把开头和结尾静音统一到目标长度。
- 输出处理后的 wav，并保留相对路径结构。

输入：

- 原始 wav 文件夹，例如 `data/raw_wav` 或 `data/haoyu/wav`。

输出：

- 处理后的 wav 文件夹，例如 `data/wav` 或 `data/haoyu/wav_post`。

推荐命令：

```bat
cmd /d /c python scripts\trim_silence_batch.py ^
  --input-dir data\raw_wav ^
  --output-dir data\wav ^
  --max-silence-sec 1.0 ^
  --normalize-edges ^
  --target-edge-silence-sec 0.5 ^
  --edge-silence-threshold 0.001 ^
  --edge-frame-length 1024 ^
  --workers 8
```

常用参数：

- `--input-dir`：原始 wav 根目录。
- `--output-dir`：处理后 wav 输出目录。
- `--max-silence-sec`：内部静音最长保留秒数。
- `--sil-vol-threshold`：内部静音能量阈值。
- `--sil-len-threshold`：至少多长的连续静音才算静音段。
- `--normalize-edges`：启用开头/结尾静音规范化。
- `--target-edge-silence-sec`：开头/结尾目标静音长度。
- `--edge-silence-threshold`：开头/结尾静音检测阈值。
- `--edge-frame-length`：开头/结尾静音检测帧长。
- `--workers`：并行线程数。

## 3. 生成 MFA TXT

脚本：[scripts/generate_wav_txt.py](scripts/generate_wav_txt.py)

作用：

- 从 JSONL 读取 `text` 和 `wav_file`。
- 使用 SudachiPy 分词。
- 删除标点，但保留长音、促音、拨音。
- 为每个 wav 生成同名 txt。
- 使用 MFA 字典减少 OOV，包括假名读音替换、相邻 token 合并、汉字数字读音转换。
- 自动把最终 OOV 统计到 txt 文件夹外侧。

输入：

- JSONL，例如 `data/wav/generated.jsonl`。
- MFA 字典，例如 `models/mfa/pretrained_models/dictionary/japanese_mfa.dict`。

输出：

- `data/txt_a/*.txt`：给 MFA align 使用的转写。
- `data/txt_a_final_oovs.txt`：最终 OOV 汇总。

推荐命令：

```bat
cmd /d /c python scripts\generate_wav_txt.py ^
  --jsonl data\wav\generated.jsonl ^
  --output-dir data\txt_a ^
  --overwrite ^
  --mode A ^
  --dict-path models\mfa\pretrained_models\dictionary\japanese_mfa.dict
```

常用参数：

- `--jsonl`：输入 JSONL。
- `--output-dir`：txt 输出目录。
- `--text-key`：JSONL 文本字段名，默认 `text`。
- `--wav-key`：JSONL wav 路径字段名，默认 `wav_file`。
- `--mode`：SudachiPy 分词模式，推荐 `A`。
- `--overwrite`：覆盖已存在 txt。
- `--dict-path`：MFA 字典路径；提供后启用 OOV 减少。
- `--max-merge-tokens`：最多尝试合并多少个相邻 token 匹配字典。
- `--oov-report`：自定义 OOV 汇总输出路径；不指定时自动写到 txt 文件夹外侧。

## 4. MFA Align

命令文档：[scripts/MFA_COMMANDS.md](scripts/MFA_COMMANDS.md)

align 生成原始 TextGrid：

```bat
set "MFA_ROOT_DIR=models\mfa" && mfa align ^
  "D:\code\FA\MFA\japanese\demo\data\txt_a" ^
  japanese_mfa ^
  japanese_mfa ^
  "D:\code\FA\MFA\japanese\demo\data\aligned_a" ^
  --audio_directory "D:\code\FA\MFA\japanese\demo\data\wav_post" ^
  --temporary_directory "D:\code\FA\MFA\japanese\models\temp" ^
  --output_format long_textgrid ^
  --num_jobs 8 ^
  --clean ^
  --overwrite ^
  --no_tokenization ^
  --no_textgrid_cleanup
```

关键点：

- txt 文件名 stem 必须和 wav 一致，例如 `00001.txt` 对应 `00001.wav`。
- `--audio_directory` 用于 txt 和 wav 分目录的情况。
- `--no_tokenization` 表示 MFA 不再重新分词，使用 txt 里的 token。
- `--no_textgrid_cleanup` 尽量保留静音 interval，方便后处理。

## 5. 后处理 TextGrid

脚本：[scripts/postprocess_textgrids.py](scripts/postprocess_textgrids.py)

作用：

- 把 MFA 的 `words/phones` 扩展成 `raw_text/tokenized_text/words/romaji/phones` 五层。
- 问号尽量作为独立 interval 插入。
- 使用 pykakasi 生成罗马音层。
- 自动修复 `っ + て` 这类短词后长静音错位问题。
- 把所有静默统一为 `<sp0>` 到 `<sp3>`。
- 过滤明显异常样本到 filtered 目录。

输入：

- `data/wav/generated.jsonl`：原始文本。
- `data/txt_a`：生成的 txt。
- `data/aligned_a`：MFA 原始 TextGrid。
- `data/wav`：处理后的 wav，用于能量辅助修复。

输出：

- `data/aligned_a_post`：最终可用 TextGrid。
- `data/aligned_a_filtered`：过滤掉的 TextGrid。
- `data/aligned_a_post/postprocess_report.jsonl`：处理报告。

推荐命令：

```bat
cmd /d /c python scripts\postprocess_textgrids.py ^
  --jsonl data\wav\generated.jsonl ^
  --txt-dir data\txt_a ^
  --textgrid-dir data\aligned_a ^
  --output-dir data\aligned_a_post ^
  --filtered-dir data\aligned_a_filtered ^
  --wav-dir data\wav ^
  --overwrite
```

常用参数：

- `--jsonl`：原始 JSONL。
- `--txt-dir`：txt 文件夹。
- `--textgrid-dir`：MFA 原始 TextGrid 文件夹。
- `--output-dir`：最终保留的 TextGrid 输出目录。
- `--filtered-dir`：过滤样本输出目录。
- `--wav-dir`：wav 文件夹，用于能量辅助修复。
- `--overwrite`：覆盖并清理 post/filtered 两侧同名旧文件。
- `--no-fix-short-multi-unit`：关闭短词错位修复。
- `--no-filter-suspicious-alignment`：关闭可疑对齐过滤。
- `--copy-errors`：处理失败时把原始 TextGrid 复制到 filtered。

## 6. 推荐运行顺序

完整顺序：

```text
原始 wav
  -> trim_silence_batch.py
  -> data/wav
  -> generate_wav_txt.py
  -> data/txt_a
  -> mfa validate / mfa align
  -> data/aligned_a
  -> postprocess_textgrids.py
  -> data/aligned_a_post + data/aligned_a_filtered
```

如果发现 OOV 很多：

1. 查看 `data/txt_a_final_oovs.txt`。
2. 补充 MFA 字典。
3. 重新生成 txt。
4. 重新运行 validate / align / postprocess。


