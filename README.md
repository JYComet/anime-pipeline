# ComicCut

动漫音频数据管线 — 从动漫番剧到高质量日语音频训练数据的一体化工具链。

## 功能概览

- **资源搜索** — 对接 animes.garden API，聚合 dmhy / moe / ani 资源站，支持关键字、字幕组、类型过滤
- **P2P 下载** — 基于 aria2c 后台下载，支持 DHT / PEX / 多 Tracker，下载完成自动触发后续流程
- **字幕提取** — 智能识别 MKV 内嵌中文字幕轨道，支持 ASS / SSA / SRT 格式
- **视频切割** — 按字幕时间轴精确切割片段，支持 NVENC / AMF / QSV 硬件加速，失败自动回退 CPU 编码
- **人工审核** — Web 界面在线试听/观看片段，一键批准/跳过，状态持久化
- **AI 降噪** — 三阶段 ClearVoice (MossFormer2) 处理：语音增强 → 语音分离 + 多说话人检测 → 超分辨率
- **质量过滤** — 混响检测 / 静音比例 / VAD 语音活动检测 / PAD 首尾规范化

## 目录结构

```
ComicCut/
├── anime-pipeline/               # 主项目
│   ├── scripts/                  # Python 后端
│   │   ├── server.py             # FastAPI 服务器入口
│   │   ├── config.py             # 全局配置（路径、工具、参数）
│   │   ├── pipeline.py           # 流水线任务编排
│   │   ├── downloader.py         # 资源搜索
│   │   ├── aria2_rpc.py          # aria2c JSON-RPC 客户端
│   │   ├── qbittorrent_client.py # qBittorrent 下载后端
│   │   ├── file_watcher.py       # 文件监视器（自动检测新视频）
│   │   ├── extract_subs.py       # 字幕提取
│   │   ├── split_video.py        # 按字幕时间轴切割视频
│   │   ├── convert_audio.py      # MP4 → WAV 音频转换
│   │   ├── audio_pipeline.py     # 音频质量筛选
│   │   └── denoise_audio.py      # AI 降噪管线
│   ├── frontend/index.html       # Web 前端（SPA）
│   ├── data/                     # 运行时数据目录
│   └── tools/                    # 本地工具（aria2c 等）
├── QuickCut/                     # ffmpeg / ffprobe 工具链
├── mkvtoolnix/                   # MKV 工具链
├── ClearerVoice-Studio-main/     # AI 语音增强模型
│   └── clearvoice/               # MossFormer2 系列模型
└── checkpoints/                  # 模型权重文件
```

## 快速开始

### 环境要求

- Windows 10+
- Python 3.8+
- GPU（推荐，用于 AI 降噪加速）
- 依赖：`fastapi`, `uvicorn`, `requests`, `librosa`, `torch`, `torchaudio`, `numpy`, `soundfile`

### 启动

```bash
# 方式一：双击启动
start.bat

# 方式二：命令行
cd anime-pipeline/scripts
python server.py
```

启动后访问 **http://localhost:5800**

### 使用流程

1. **搜索下载** — 在搜索页输入动漫名称（支持中文/日文/罗马音），点击下载
2. **自动处理** — 下载完成后自动提取字幕并切割视频片段
3. **人工审核** — 在审核页试听/观看片段，通过的自动转为 WAV
4. **批量降噪** — 在降噪页选中已批准的音频，一键执行 ClearVoice 三阶段降噪
5. **最终输出** — 高质量日语单说话人音频片段，`data/cleaned/` 目录下 `.norm.wav` 结尾

也支持**本地文件模式**：将 MKV/MP4 放入 `video/` 目录，文件监视器自动检测并触发处理。

## 核心工作流

```
搜索资源 → aria2c 后台下载 → 文件监视器检测完成
    → 提取中文字幕 → 按字幕时间轴切割视频片段
    → 过滤过短视频(< 1 s) → 人工审核（试听/批准/跳过）
    → MP4 → WAV → ClearVoice 三阶段降噪
    → 音频质量筛选（混响/静音/VAD/PAD） → 最终输出
```

## API 接口

| 模块 | 路径 | 说明 |
|------|------|------|
| 搜索 | `GET /api/search` | 搜索动漫资源 |
| 搜索 | `GET /api/teams` | 字幕组列表 |
| 下载 | `GET /api/magnet/info` | 解析 magnet 链接 |
| 本地 | `GET /api/local/videos` | 列出本地视频 |
| 任务 | `POST /api/jobs/process-download` | 提交下载任务 |
| 任务 | `GET /api/jobs` | 列出所有任务 |
| 任务 | `POST /api/jobs/{id}/cancel` | 取消任务 |
| 审核 | `GET /api/review/clips` | 列出待审核片段 |
| 审核 | `POST /api/review/approve` | 批准片段 |
| 结果 | `GET /api/results/videos` | 查看处理结果 |
| 降噪 | `POST /api/denoise/batch` | 批量降噪 |

## 依赖工具

| 工具 | 用途 |
|------|------|
| ffmpeg / ffprobe | 视频编解码、音频转换、元数据查询 |
| mkvextract / mkvmerge | MKV 轨道提取与分析 |
| aria2c | P2P 下载引擎 |
| ClearVoice (MossFormer2) | AI 语音增强 / 分离 / 超分辨率 |

## License

MIT
