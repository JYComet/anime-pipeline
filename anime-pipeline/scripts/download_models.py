#!/usr/bin/env python3
"""
模型下载脚本 —— 一键下载 Anime Pipeline 所需的全部预训练模型。

用法: python download_models.py [--force]

模型清单（仅保留 Qwen3-ASR，其余 ASR 模型已移除）：
  Qwen3-ASR-1.7B          ASR 语音识别       modelscope     ~2.0 GB
  FSMN-VAD                 语音活动检测       modelscope     ~50  MB
  htdemucs_ft              BGM/音乐分离       torch hub      ~330 MB
  MossFormer2_SE_48K       语音增强           ClearVoice     ~300 MB
  MossFormer2_SR_48K       语音超分辨率       ClearVoice     ~300 MB
  Gender-Detection-ONNX    性别检测           huggingface    ~90  MB
  japanese_mfa             MFA 声学+字典      MFA CLI         ~40  MB

总计约 3.1 GB（全部自动下载到系统缓存目录）
"""

import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download_models")

# 确保 scripts 目录在 path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ============================================================
# 1. Qwen3-ASR-1.7B   (ASR 语音识别)
# ============================================================
def download_qwen3_asr():
    """下载 Qwen3-ASR 模型到 modelscope 缓存 (~2.0 GB)"""
    log.info("=" * 60)
    log.info("下载 Qwen3-ASR-1.7B（ASR 语音识别）~2.0 GB")
    log.info("=" * 60)
    try:
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
        log.info("正在从 modelscope 下载 Qwen/Qwen3-ASR-1.7B ...")
        model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B")
        log.info("Qwen3-ASR 下载完成，缓存于 ~/.cache/modelscope/hub/models/Qwen/")
    except ImportError:
        log.error("qwen_asr 未安装，请先运行: pip install qwen-asr")
        raise
    except Exception as e:
        log.error(f"Qwen3-ASR 下载失败: {e}")
        raise


# ============================================================
# 2. FSMN-VAD   (语音活动检测)
# ============================================================
def download_fsmn_vad():
    """下载 FSMN-VAD 模型 (~50 MB)"""
    log.info("=" * 60)
    log.info("下载 FSMN-VAD（语音活动检测）~50 MB")
    log.info("=" * 60)
    try:
        from funasr import AutoModel
        log.info("正在从 modelscope 下载 fsmn-vad ...")
        model = AutoModel(model="fsmn-vad")
        log.info("FSMN-VAD 下载完成")
    except ImportError:
        log.error("funasr 未安装，请先运行: pip install funasr")
        raise
    except Exception as e:
        log.error(f"FSMN-VAD 下载失败: {e}")
        raise


# ============================================================
# 3. htdemucs_ft   (BGM / 音乐分离)
# ============================================================
def download_demucs():
    """下载 Meta Demucs HT 模型 (~330 MB)"""
    log.info("=" * 60)
    log.info("下载 Demucs htdemucs_ft（BGM/音乐分离）~330 MB")
    log.info("=" * 60)
    try:
        from demucs import pretrained
        log.info("正在从 Facebook Torch Hub 下载 htdemucs_ft ...")
        model = pretrained.get_model("htdemucs_ft")
        log.info("Demucs 下载完成，缓存于 ~/.cache/torch/hub/checkpoints/")
    except ImportError:
        log.error("demucs 未安装，请先运行: pip install demucs")
        raise
    except Exception as e:
        log.error(f"Demucs 下载失败: {e}")
        raise


# ============================================================
# 4 & 5. ClearVoice (MossFormer2)   (语音增强 + 超分辨率)
# ============================================================
def download_clearvoice():
    """下载 ClearVoice MossFormer2 模型 (~600 MB 两个模型)"""
    log.info("=" * 60)
    log.info("下载 ClearVoice MossFormer2（语音增强 + 超分辨率）~600 MB")
    log.info("=" * 60)

    # 查找 ClearVoice 安装路径
    cv_paths = []
    for p in [
        os.path.join(os.path.dirname(SCRIPT_DIR), "..", "ClearerVoice-Studio-main", "ClearerVoice-Studio-main"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "ClearerVoice-Studio-main"),
    ]:
        real = os.path.realpath(p)
        if os.path.isdir(real):
            cv_paths.append(real)

    if cv_paths:
        cv_path = cv_paths[0]
        log.info(f"找到 ClearVoice 路径: {cv_path}")
        if cv_path not in sys.path:
            sys.path.insert(0, cv_path)

    try:
        from clearvoice import ClearVoice

        log.info("下载 MossFormer2_SE_48K（语音增强）...")
        cv_se = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        log.info("MossFormer2_SE_48K 下载完成")

        log.info("下载 MossFormer2_SR_48K（语音超分辨率）...")
        cv_sr = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
        log.info("MossFormer2_SR_48K 下载完成")

    except ImportError:
        log.error(
            "ClearVoice 未安装。请从以下地址安装：\n"
            "  git clone https://github.com/modelscope/ClearerVoice-Studio.git\n"
            "  然后将 clearvoice/ 放到 ComiCut/ClearerVoice-Studio-main/ 下\n"
            "  或安装 pip install ClearerVoice-Studio"
        )
        raise
    except Exception as e:
        log.error(f"ClearVoice 下载失败: {e}")
        raise


# ============================================================
# 6. Gender Detection ONNX   (性别检测)
# ============================================================
def download_gender_detection():
    """下载性别检测 ONNX 模型 (~90 MB)"""
    log.info("=" * 60)
    log.info("下载 Gender Detection ONNX（性别检测）~90 MB")
    log.info("=" * 60)
    try:
        from huggingface_hub import hf_hub_download
        log.info("正在从 HuggingFace 下载 prithivMLmods/Common-Voice-Gender-Detection-ONNX ...")
        path = hf_hub_download(
            repo_id="prithivMLmods/Common-Voice-Gender-Detection-ONNX",
            filename="onnx/model.onnx",
        )
        log.info(f"Gender Detection 下载完成: {path}")
    except ImportError:
        log.error("huggingface_hub 未安装，请先运行: pip install huggingface_hub")
        raise
    except Exception as e:
        log.error(f"Gender Detection 下载失败: {e}")
        raise


# ============================================================
# 7. MFA 模型   (日语强制对齐)
# ============================================================
def download_mfa_models():
    """下载 MFA 日语声学模型和字典 (~40 MB)"""
    log.info("=" * 60)
    log.info("下载 MFA japanese_mfa（日语强制对齐）~40 MB")
    log.info("=" * 60)

    try:
        # 检查 mfa 是否可用
        result = subprocess.run(
            ["mfa", "version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            log.error("MFA CLI 未安装。请运行: conda install -c conda-forge montreal-forced-aligner")
            return False

        # 下载日语声学模型
        log.info("下载 japanese_mfa 声学模型...")
        subprocess.run(
            ["mfa", "model", "download", "acoustic", "japanese_mfa"],
            check=True,
        )
        log.info("japanese_mfa 声学模型下载完成")

        # 下载日语字典
        log.info("下载 japanese_mfa 字典...")
        subprocess.run(
            ["mfa", "model", "download", "dictionary", "japanese_mfa"],
            check=True,
        )
        log.info("japanese_mfa 字典下载完成")

    except FileNotFoundError:
        log.error("MFA CLI 未找到。请先安装 Montreal Forced Aligner")
        raise
    except subprocess.CalledProcessError as e:
        log.error(f"MFA 模型下载失败: {e}")
        raise


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="下载 Anime Pipeline 全部模型")
    parser.add_argument(
        "--skip-asr", action="store_true", help="跳过 Qwen3-ASR"
    )
    parser.add_argument(
        "--skip-vad", action="store_true", help="跳过 FSMN-VAD"
    )
    parser.add_argument(
        "--skip-demucs", action="store_true", help="跳过 Demucs BGM分离"
    )
    parser.add_argument(
        "--skip-clearvoice", action="store_true",
        help="跳过 ClearVoice 语音增强/超分辨率"
    )
    parser.add_argument(
        "--skip-gender", action="store_true", help="跳过性别检测"
    )
    parser.add_argument(
        "--skip-mfa", action="store_true", help="跳过 MFA"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新下载（即使已缓存）"
    )

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Anime Pipeline — 模型下载脚本")
    log.info("  总计约 3.1 GB，请确保网络畅通和磁盘空间充足")
    log.info("=" * 60)

    steps = [
        ("Qwen3-ASR-1.7B",        download_qwen3_asr,         args.skip_asr),
        ("FSMN-VAD",               download_fsmn_vad,          args.skip_vad),
        ("Demucs htdemucs_ft",     download_demucs,           args.skip_demucs),
        ("ClearVoice MossFormer2", download_clearvoice,       args.skip_clearvoice),
        ("Gender Detection ONNX",  download_gender_detection, args.skip_gender),
        ("MFA japanese_mfa",       download_mfa_models,       args.skip_mfa),
    ]

    failed = []

    for name, func, skip in steps:
        if skip:
            log.info(f"跳过: {name}")
            continue
        try:
            func()
        except Exception as e:
            log.error(f"{name} 失败: {e}")
            failed.append(name)

    log.info("=" * 60)
    if failed:
        log.error(f"以下模型下载失败: {', '.join(failed)}")
        log.error("请检查网络连接后重试")
        sys.exit(1)
    else:
        log.info("全部模型下载完成！")


if __name__ == "__main__":
    main()
