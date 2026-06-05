# Video Pipeline — 视频管线处理工具

从 Anime Pipeline 原项目中提取的独立视频管线处理脚本，用于批量处理视频/音频文件。

## 功能

| 步骤 | 说明 |
|------|------|
| **时长切分** | 将长视频/音频按固定时长（默认 10 分钟）切分为片段，自动去除片头片尾 |
| **WAV 转换** | 统一转换为 32kHz 单声道 16-bit PCM WAV 格式 |
| **BGM 分离** | 使用 Meta Demucs (htdemucs_ft) 模型分离人声，去除背景音乐、鼓、贝斯 |

## 目录结构

```
video-pipeline/
├── config.yaml              # 主配置文件
├── main.py                  # 主入口脚本
├── scripts/
│   ├── config_loader.py     # 配置加载器
│   ├── splitter.py          # 时长切分模块
│   ├── converter.py         # WAV 转换模块
│   └── bgm_separator.py     # BGM 分离模块 (Demucs)
├── checkpoints/
│   └── hub/checkpoints/     # Demucs 模型权重 (~400 MB, torch hub 缓存)
├── data/
│   ├── input/               # 输入文件目录
│   ├── output/              # 输出文件目录
│   ├── temp/                # 临时文件目录
│   └── logs/                # 日志文件目录
├── requirements.txt
├── setup.bat / setup.sh     # 一键配置脚本
├── start.bat / start.sh     # 一键启动脚本
└── README.md
```

## 环境要求

### 硬件
- **GPU**: NVIDIA 显卡，建议 8GB+ 显存（BGM 分离使用 CUDA）
- **内存**: 16GB+
- **磁盘**: 预留足够空间存放模型 (~400MB) 和处理临时文件

### 软件
- **Python**: 3.10+
- **ffmpeg**: 用于音视频处理（必须在系统 PATH 中或配置完整路径）
- **CUDA**: 12.x（推荐 12.6）

## 快速开始

### Windows

```batch
# 1. 双击运行 setup.bat（首次使用，自动安装所有依赖）
setup.bat

# 2. 将待处理文件放入 data\input\ 目录

# 3. 双击运行 start.bat
start.bat
```

### Linux / macOS

```bash
# 1. 运行配置脚本（首次使用）
bash setup.sh

# 2. 将待处理文件放入 data/input/ 目录

# 3. 启动处理
bash start.sh
```

### 命令行使用

```bash
# 激活虚拟环境
# Windows: venv\Scripts\activate
# Linux:   source venv/bin/activate

# 处理默认输入目录 (config.yaml 中配置的 input_dir)
python main.py

# 处理指定目录
python main.py -i /path/to/videos

# 处理单个文件
python main.py -i /path/to/video.mp4

# 使用自定义配置文件
python main.py -c custom_config.yaml

# 预览模式（只显示文件列表，不处理）
python main.py --dry-run
```

## 配置说明

编辑 `config.yaml` 自定义设置：

```yaml
# 路径设置
paths:
  input_dir: "./data/input"       # 输入文件夹
  output_dir: "./data/output"     # 输出文件夹
  temp_dir: "./data/temp"         # 临时文件夹
  log_dir: "./data/logs"          # 日志文件夹

# 外部工具 (如果 ffmpeg 不在 PATH 中，填写完整路径)
tools:
  ffmpeg: "ffmpeg"

# 处理设置
processing:
  segment_duration: 600           # 切分时长 (秒)，默认 600 = 10分钟
  keep_ends: false                # 是否保留首尾分段
  sample_rate: 32000              # 输出采样率 (Hz)
  channels: 1                     # 声道数 (1=单声道, 2=立体声)

# 模型实例池
model_pool:
  demucs_instances: 2             # Demucs 实例数: 2 或 4
                                  # 2 实例 ≈ 3 GB VRAM (适合 8GB 显卡)
                                  # 4 实例 ≈ 6 GB VRAM (适合 12GB+ 显卡)

# 运行设备
device: "cuda"                    # cuda = GPU, cpu = CPU (不推荐)
```

### 实例池说明

BGM 分离步骤使用 Demucs 模型池来控制并发。`demucs_instances` 配置项同时控制：
- 加载的模型实例数量（每个约 1.5 GB VRAM）
- 最大并行处理的片段数

- **2 实例** (默认): 适合 8GB 显存显卡 (RTX 3060/4060)，约占用 3GB VRAM
- **4 实例**: 适合 12GB+ 显存显卡 (RTX 3080/4070/4080/4090)，约占用 6GB VRAM，处理速度翻倍

## 支持的格式

| 类型 | 格式 |
|------|------|
| 视频 | `.mp4` `.mkv` `.avi` `.mov` `.wmv` `.webm` `.flv` |
| 音频 | `.wav` `.mp3` `.flac` `.aac` `.ogg` `.m4a` |

## 输出结构

处理后，每个输入文件会在输出目录下创建同名子文件夹：

```
data/output/
└── 文件名/
    ├── 文件名_seg000_vocals.wav    # 第1段人声
    ├── 文件名_seg001_vocals.wav    # 第2段人声
    └── ...
```

## 日志

- **控制台**: 实时显示处理状态（INFO 级别）
- **文件**: `data/logs/pipeline_YYYYMMDD_HHMMSS.log`（DEBUG 级别，记录完整详细信息）
- **错误日志**: `data/logs/bgm_separate_errors.log`（BGM 分离错误的详细堆栈）

## 依赖列表

### 基础依赖 (requirements.txt)
| 包名 | 版本 | 用途 |
|------|------|------|
| pyyaml | >=6.0 | 配置文件解析 |
| numpy | >=1.24.0 | 数值计算 |
| soundfile | >=0.12.0 | 音频文件读写 |
| torch | >=2.0.0 | 深度学习框架 |
| torchaudio | >=2.0.0 | 音频处理 |

### 模型部署依赖
| 包名 | 版本 | 用途 |
|------|------|------|
| demucs | >=4.0 | BGM/人声分离模型 (Meta) |

### 系统工具
| 工具 | 用途 |
|------|------|
| ffmpeg | 音视频转换与切分 |
| ffprobe | 媒体文件信息读取 |

## 常见问题

### Q: 启动时报 "CUDA GPU 不可用"
A: 确认已安装 NVIDIA 驱动和 CUDA 版 PyTorch。运行：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
如果输出 `False`，重新安装 CUDA 版 PyTorch：
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Q: ffmpeg 找不到
A: 将 ffmpeg 添加到系统 PATH，或在 `config.yaml` 中设置绝对路径：
```yaml
tools:
  ffmpeg: "C:/tools/ffmpeg/bin/ffmpeg.exe"
```

### Q: 显存不足 (CUDA out of memory)
A: 将 `demucs_instances` 从 4 降为 2，或设置 `device: "cpu"`（会非常慢）。

### Q: 如何修改切分时长
A: 编辑 `config.yaml` 中 `processing.segment_duration` 的值（秒）。
