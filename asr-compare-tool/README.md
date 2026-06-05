# ASR Compare Tool

比较 FireRed ASR2（本地模型）与 Qwen3-ASR（阿里云百炼 API）的语音识别结果，自动删除匹配率低于阈值的音频。

## 功能

- 批量处理音频文件（支持 `.wav` `.mp3` `.m4a` `.flac` `.aac` `.ogg` `.opus` `.wma`）
- VAD 语音检测 + 分段切割（9-16 秒/段）
- FireRed ASR2 批量 GPU 推理（2-4x 吞吐提升）
- 可选 TensorRT 加速编码器（额外 2-3x 提升）
- Qwen3-ASR API 并行转写
- 逐段文本对比：Levenshtein 编辑距离 + LCS 差分
- 中文文本深度标准化（繁简转换、异体字统一、数字规范化等）
- 自动删除匹配率低于阈值的音频及其所有输出文件
- 双层 tqdm 进度条 + 完整日志记录

## 硬件要求

- **NVIDIA GPU + CUDA**（必需，不支持 CPU）
- 推荐显存 >= 8GB（FireRed ASR2-AED 约 4.5GB + 其他子模型约 5GB）
- TensorRT 模式需要额外 ~1GB 用于 engine

## 目录结构

```
asr-compare-tool/
├── asr_compare.py              # 主入口脚本
├── deploy.py                   # 一键部署脚本（自动检测OS + 安装依赖）
├── config.yaml                 # 配置文件
├── requirements.txt            # Python 依赖清单
├── export_firered_trt.py       # TensorRT 引擎导出脚本（可选）
├── run.bat                     # Windows 一键启动
├── run.sh                      # Linux / macOS 一键启动
├── README.md                   # 本文档
├── asr_compare_lib/
│   ├── __init__.py
│   ├── pipeline.py             # 核心流程编排（含进度条 + batch 调度）
│   ├── models.py               # ASR 模型（含 FireRed-TRT 包装器）
│   ├── comparison.py           # 文本对比算法
│   ├── normalization.py        # 文本标准化
│   └── audio_utils.py          # VAD、音频格式转换
├── models/                     # 本地模型文件
│   ├── silero_vad/silero_vad.onnx
│   └── firered_asr2/           # FireRed 预训练模型
│       ├── FireRedVAD/VAD/
│       ├── FireRedLID/
│       ├── FireRedASR2-AED/
│       ├── FireRedPunc/
│       └── FireRedASR2-AED-TRT/   # TensorRT engine（可选）
├── FireRedASR2S/               # FireRed 源码
├── audio_input/                # 输入音频目录（自行创建）
└── output/                     # 输出目录（自动创建）
    └── {音频名}/
        ├── {音频名}.wav
        ├── {音频名}_firered.srt / _qwen3-api.srt
        ├── {音频名}_firered.txt / _qwen3-api.txt
        └── segments/
            ├── {音频名}_seg001.wav ...
            └── txt/
                ├── {音频名}_seg001_firered.txt
                └── {音频名}_seg001_qwen3-api.txt
```

## 资源目录

| 资源 | 网络路径 |
|---|---|
| ASR 对比输出数据（segments） | `\\RS3621\CompanyShare-Confidential\Persons\jiangyichen\ASR_Compare\segments` |
| 人声分离后数据 | `\\RS3621\CompanyShare-Confidential\Persons\jiangyichen\Enhanced` |

## 快速开始（一键部署）

### Windows

双击 `run.bat`，脚本会自动：

1. 检测 Python 环境、CUDA、ffmpeg、模型文件
2. 安装所需 Python 依赖
3. 启动 ASR 对比任务（带进度条）

### Linux / macOS

```bash
bash run.sh
```

### 部署与运行分开执行

```bash
python deploy.py              # 检测环境 + 安装依赖
python deploy.py --check      # 仅检查环境不安装
python asr_compare.py         # 运行对比
```

## 手动部署

### 1. 环境要求

- **Python 3.10+**
- **NVIDIA GPU + CUDA**（必需，不支持 CPU）
- **ffmpeg** / **ffprobe**（在 PATH 中可用）

### 2. 安装依赖

```bash
# CUDA 11.8（推荐）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.x
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3. 配置

编辑 `config.yaml`：

```yaml
# ========== API 配置（必填）==========
api:
  dashscope_api_key: "sk-xxxxxxxxxxxxxxxx"
  dashscope_api_base: "https://dashscope.aliyuncs.com"
  api_model: "qwen3-asr-flash"
  request_timeout: 120
  max_retries: 3

# ========== 路径配置 ==========
paths:
  audio_input_dir: "./audio_input"
  output_dir: "./output"
  firered_source_path: "./FireRedASR2S"
  firered_models_dir: "./models/firered_asr2"
  firered_trt_engine_dir: "./models/firered_asr2/FireRedASR2-AED-TRT"  # 可选

# ========== ASR 配置 ==========
asr:
  language: "zh"
  device: "cuda"
  hotwords: ""

# ========== 对比配置 ==========
compare:
  match_threshold: 90
  delete_below_threshold: true
  filter_english: true
```

可使用 `config.local.yaml` 覆盖默认配置（不会被 git 跟踪）。

### 4. 运行

```bash
python asr_compare.py                           # 默认配置
python asr_compare.py --audio-dir ./my_audio    # 覆盖输入目录
python asr_compare.py --no-delete               # 只标记不删除
python asr_compare.py --match-threshold 85      # 调整阈值
python asr_compare.py --language ja             # 日语
python asr_compare.py --hotwords "角色名,地名"   # 热词
```

## FireRed 优化方案

本项目对 FireRed ASR2 进行了 4 项优化（与原项目同步）：

| # | 方案 | 原理 | 收益 |
|---|---|---|---|
| 1 | 跳过内部 VAD+LID | pipeline 已做 VAD，直接调用 `model.asr.transcribe()` | 省去冗余推理 |
| 2 | `beam_size=1` 贪心解码 | 单条最优路径，跳过 beam search | 解码 ~3x |
| 3 | 批量 GPU 推理 | 所有段一次传入 Conformer encoder | 吞吐 2-4x |
| 4 | TensorRT encoder | Conformer 导出为 TRT engine | encoder 2-3x |

运行时自动检测 `firered_trt_engine_dir` 下是否有 `encoder.plan`，有则启用 TRT。

### 构建 TensorRT 引擎（可选）

```bash
pip install tensorrt_cu12_bindings tensorrt_cu12_libs tensorrt onnx \
    --extra-index-url https://pypi.nvidia.com --only-binary :all:

python export_firered_trt.py \
    --model-dir models/firered_asr2/FireRedASR2-AED \
    --output-dir models/firered_asr2/FireRedASR2-AED-TRT
```

构建需要 10-30 分钟（取决于 GPU），之后 `encoder.plan` 会被自动使用。

## 对比算法说明

与 ComicCut 原项目完全一致的流程：

1. **VAD 语音检测**：Silero VAD（阈值 0.5，最小语音段 250ms，最小静音 100ms）
2. **分段**：9-16 秒/段，短段合并，长段均分
3. **批量转写**：FireRed 所有段一次 GPU 调用 + Qwen3 API 逐段调用
4. **文本标准化（MFA 风格）**：
   - 去除标点符号与空白字符
   - 第三人称代词统一（他/她/它/祂/牠 → 他）
   - 中文数字转阿拉伯数字（三百五十六 → 356）
   - 全角数字转半角（０１２ → 012）
5. **Levenshtein 编辑距离**：计算归一化文本差异
6. **匹配率**：`match_rate = 100% - (编辑距离 / 最大文本长度 × 100%)`
7. **英语检测**：输出含 3+ 连续 ASCII 字母 → 匹配率强制为 0%（视为幻觉）
8. **整体匹配率**：各段按文本长度加权平均

## 日志

- 控制台输出：双层 tqdm 进度条 + 实时段匹配率
- 文件日志：`asr_compare.log`，包含完整 DEBUG 级别信息
- 格式：`[2026-06-05 14:30:00] INFO    处理中...`

## API 配置（Qwen3-ASR）

本工具使用阿里云百炼 DashScope API 调用 Qwen3-ASR 模型。

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `dashscope_api_key` | 阿里云百炼 API Key，以 `sk-` 开头 | 已预填 |
| `dashscope_api_base` | API 服务地址 | `https://dashscope.aliyuncs.com` |
| `api_model` | 模型名，Flash 版更快更省 | `qwen3-asr-flash` |
| `request_timeout` | 单次请求超时秒数 | `120` |
| `max_retries` | 网络错误自动重试次数 | `3` |

### 获取 API Key

1. 登录 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 开通"语音识别"服务
3. 左侧菜单 → **API Key 管理** → 创建 Key
4. 填入 `config.yaml` 的 `dashscope_api_key` 字段

## FireRedASR2 模型准备

项目已自带源码（`FireRedASR2S/`）和模型（`models/firered_asr2/`）。

如需手动获取：

```bash
# 源码
git clone https://github.com/FireRedTeam/FireRedASR2S.git

# 模型从官方下载，放入 models/firered_asr2/
models/firered_asr2/
├── FireRedVAD/VAD/          # VAD 模型
├── FireRedLID/              # 语言识别模型
├── FireRedASR2-AED/         # AED 语音识别模型（核心）
└── FireRedPunc/             # 标点恢复模型
```
