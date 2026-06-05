"""
BGM / music separation using Meta Demucs (htdemucs_ft model).
Separates audio into 4 stems: vocals, drums, bass, other.
Keeps only the vocals track.

Modified from the original Anime Pipeline to support configurable pool size
(2 or 4 Demucs instances) via config.yaml.
"""
import os
import datetime
import threading
import queue

import numpy as np
import soundfile as sf
import torch

from .config_loader import get_model_pool, get_device

# Sources order from htdemucs_ft
SOURCES = ["drums", "bass", "other", "vocals"]

# Pool state
_pool_size = None
_model_pool = None
_pool_lock = threading.Lock()
_pool_loaded = False


def _init_pool():
    """Initialize the Demucs model pool based on config settings."""
    global _pool_size, _model_pool, _pool_loaded

    pool_cfg = get_model_pool()
    size = pool_cfg.get("demucs_instances", 2)
    if size not in (2, 4):
        print(f"[BGM] 警告: demucs_instances 值 {size} 无效，使用默认值 2")
        size = 2

    if _pool_size == size and _pool_loaded:
        return

    with _pool_lock:
        if _pool_loaded:
            return

        _pool_size = size
        _model_pool = queue.Queue(maxsize=_pool_size)
        device = get_device()

        from demucs import pretrained

        print(f"[BGM] 加载 Demucs htdemucs_ft 模型 x{_pool_size}...")
        for i in range(_pool_size):
            model = pretrained.get_model("htdemucs_ft")
            if device != "cpu" and torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            _model_pool.put(model)
            print(f"[BGM]   实例 {i + 1}/{_pool_size} 就绪")

        _pool_loaded = True
        print(f"[BGM] 模型池初始化完成 ({_pool_size} 实例, 设备: {device})")


def _get_model():
    """Get a Demucs model from the pool (blocks until one is free)."""
    if not _pool_loaded:
        _init_pool()
    return _model_pool.get()


def _return_model(model):
    """Return a model instance to the pool."""
    _model_pool.put(model)


def get_pool_size():
    """Return the configured pool size (0 if not initialized)."""
    return _pool_size or get_model_pool().get("demucs_instances", 2)


def separate_vocals(input_path, output_dir, logger=None):
    """Separate vocals from audio and save only the vocals track.

    Args:
        input_path: Path to input WAV file.
        output_dir: Directory for output. The vocals file will be named
                    ``<basename>_vocals.wav``.
        logger: Optional logger instance.

    Returns:
        Path to the vocals WAV file, or empty string on failure.

    After successful separation the drums/bass/other tracks are deleted
    from disk (only vocals is kept to save space).
    """
    device = get_device()

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 不可用，BGM 分离需要 GPU (Demucs)。请在配置中将 device 设为 cpu 或安装 CUDA。")

    base = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(output_dir, f"{base}_vocals.wav")

    if os.path.exists(vocals_path) and os.path.getsize(vocals_path) > 0:
        if logger:
            logger.info(f"  [BGM] 跳过，已存在: {os.path.basename(vocals_path)}")
        return vocals_path

    log = logger.info if logger else print
    warn = logger.warning if logger else print

    model = None
    try:
        import demucs.apply

        model = _get_model()

        # Load audio (Demucs expects float32 tensor, stereo)
        audio, sr = sf.read(input_path, dtype="float32")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)  # mono -> stereo
        else:
            audio = audio.T  # (samples, channels) -> (channels, samples)

        mix = torch.from_numpy(audio).float()
        mix = mix.unsqueeze(0)  # add batch dim -> (1, channels, samples)
        if device != "cpu" and torch.cuda.is_available():
            mix = mix.cuda()

        # Separate
        with torch.no_grad():
            sources = demucs.apply.apply_model(
                model, mix, split=True, overlap=0.25, progress=False,
                device=device, shifts=1,
            )

        source_names = getattr(model, "sources", SOURCES)
        try:
            vocals_idx = list(source_names).index("vocals")
        except ValueError:
            warn(f"  [BGM] 模型 sources 列表中未找到 'vocals': {source_names}")
            return ""

        # Extract vocals: shape (batch=1, channels, samples)
        vocals = sources[0, vocals_idx].cpu().numpy()  # (channels, samples)
        vocals = vocals.T  # -> (samples, channels)

        # Collapse to mono if stereo channels are identical (anime source)
        if vocals.shape[1] == 2 and np.allclose(vocals[:, 0], vocals[:, 1], atol=1e-5):
            vocals = vocals[:, 0]
        elif vocals.shape[1] == 2 and np.allclose(vocals[:, 0], vocals[:, 1], rtol=0.01):
            vocals = vocals[:, 0]

        os.makedirs(output_dir, exist_ok=True)
        sf.write(vocals_path, vocals, sr)

        # Free GPU memory
        del sources, mix, audio
        torch.cuda.empty_cache()

        if os.path.exists(vocals_path) and os.path.getsize(vocals_path) > 0:
            log(f"  [BGM] 完成: {os.path.basename(vocals_path)}")
            return vocals_path
        return ""

    except Exception:
        import traceback

        err_msg = f"[{datetime.datetime.now()}] {input_path}\n{traceback.format_exc()}\n"
        if logger:
            logger.error(f"  [BGM] 失败: {os.path.basename(input_path)}\n{traceback.format_exc()}")
        else:
            print(err_msg, flush=True)

        # Write to dedicated error log
        log_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "bgm_separate_errors.log")
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(err_msg + "\n")
        except Exception:
            pass
        return ""
    finally:
        if model is not None:
            _return_model(model)


def shutdown_pool():
    """Release all Demucs model instances and free GPU memory."""
    global _model_pool, _pool_loaded, _pool_size
    if not _pool_loaded:
        return
    with _pool_lock:
        if not _pool_loaded:
            return
        print("[BGM] 释放模型池...")
        while not _model_pool.empty():
            try:
                _model_pool.get_nowait()
            except queue.Empty:
                break
        _model_pool = None
        _pool_loaded = False
        _pool_size = None
        torch.cuda.empty_cache()
        print("[BGM] 模型池已释放")
