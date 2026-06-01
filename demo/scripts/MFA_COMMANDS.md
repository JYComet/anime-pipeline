# MFA Japanese Commands

这个文档记录本项目常用的 MFA 命令：下载日文模型、validate 检查、align 对齐、OOV 检查，以及增大 beam 的对齐变体。示例按 Windows CMD 写法编写，可以直接复制运行。

下面示例适用于 CMD，不需要单独先执行 `$env:MFA_ROOT_DIR = ...`。
如果在linux上运行，请修改命令的形式做适配

当前示例路径：

- wav: `D:\code\FA\MFA\japanese\data\wav`
- txt: `D:\code\FA\MFA\japanese\data\txt_c`
- align output: `D:\code\FA\MFA\japanese\data\aligned_c`
- validate output: `D:\code\FA\MFA\japanese\data\validate_c`
- MFA root: `D:\code\FA\MFA\japanese\models\mfa`
- MFA temp: `D:\code\FA\MFA\japanese\models\temp`

如果要换成 `txt_a` 或 `txt_b`，把命令里的 `data\txt_c`、`validate_c`、`aligned_c` 对应改掉即可。

## 1. Download Japanese Models

下载日文声学模型、发音字典到指定 MFA root 目录：

```bat
set "MFA_ROOT_DIR=D:\code\FA\MFA\japanese\models\mfa" && mfa model download acoustic japanese_mfa && mfa model download dictionary japanese_mfa
```

可选：下载日文 G2P 模型，用于处理 OOV：

```bat
set "MFA_ROOT_DIR=D:\code\FA\MFA\japanese\models\mfa" && mfa model download g2p japanese_katakana_mfa
```

## 2. Validate

txt 和 wav 分目录，检查数据是否能用于对齐：

```bat
set "MFA_ROOT_DIR=D:\code\FA\MFA\japanese\models\mfa" && mfa validate ^
  "D:\code\FA\MFA\japanese\data\txt_c" ^
  japanese_mfa ^
  --acoustic_model_path japanese_mfa ^
  --audio_directory "D:\code\FA\MFA\japanese\data\wav" ^
  --output_directory "D:\code\FA\MFA\japanese\data\validate_c" ^
  --temporary_directory "D:\code\FA\MFA\japanese\models\temp" ^
  --num_jobs 8 ^
  --clean ^
  --overwrite
```

要求：`00001.wav` 对应 `00001.txt`，文件名 stem 必须一致。

## 3. Align With sil Labels

并行执行 MFA align，txt 和 wav 分目录，输出 TextGrid，并尽量保留 `sil` 标签：

```bat
set "MFA_ROOT_DIR=D:\code\FA\MFA\japanese\models\mfa" && mfa align ^
  "D:\code\FA\MFA\japanese\data\txt_c" ^
  japanese_mfa ^
  japanese_mfa ^
  "D:\code\FA\MFA\japanese\data\aligned_c" ^
  --audio_directory "D:\code\FA\MFA\japanese\data\wav" ^
  --temporary_directory "D:\code\FA\MFA\japanese\models\temp" ^
  --output_format long_textgrid ^
  --num_jobs 8 ^
  --clean ^
  --overwrite ^
  --no_tokenization ^
  --no_textgrid_cleanup
```

说明：

- `--num_jobs 8` 是并行数量，可以按 CPU 核心数调整。
- `--audio_directory` 用于 txt 和 wav 分开放的情况。
- `--no_textgrid_cleanup` 尽量保留 phone tier 里的静音段，便于看到 `sil`。


