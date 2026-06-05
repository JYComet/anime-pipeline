"""
BGM / music separation using Meta Demucs (htdemucs_ft model).
Separates audio into 4 stems: vocals, drums, bass, other.
Keeps only the vocals track; deletes instrumental tracks automatically.
"""
import os
import numpy as np
import soundfile as sf
import torch
import threading
import queue

# Model pool (2 instances for ~3 GB VRAM total on 24 GB GPU)
_POOL_SIZE = 2
_model_pool = queue.Queue(maxsize=_POOL_SIZE)
_pool_lock = threading.Lock()
_pool_loaded = False

# htDemucs outputs 4 sources
SOURCES = ["drums", "bass", "other", "vocals"]


def _get_model(device="cuda"):
    """Get a Demucs model from the pool (blocks until one is free)."""
    global _pool_loaded
    with _pool_lock:
        if not _pool_loaded:
            from demucs import pretrained
            for _ in range(_POOL_SIZE):
                model = pretrained.get_model("htdemucs_ft")
                if device != "cpu" and torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                _model_pool.put(model)
            _pool_loaded = True
    return _model_pool.get()


def _return_model(model):
    """Return a model instance to the pool."""
    _model_pool.put(model)


def separate_vocals(input_path: str, output_dir: str, device="cuda") -> str:
    """Separate vocals from audio and save only the vocals track.

    Args:
        input_path: Path to input WAV file.
        output_dir: Directory for output. The vocals file will be named
                    ``<basename>_vocals.wav``.
        device: "cuda" or "cpu". Default is "cuda" — CPU fallback is disabled.

    Returns:
        Path to the vocals WAV file, or empty string on failure.

    After successful separation the drums/bass/other tracks are deleted
    from disk (only vocals is kept to save space).
    """
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for music separation (Demucs). CPU fallback is disabled.")

    base = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(output_dir, f"{base}_vocals.wav")

    if os.path.exists(vocals_path) and os.path.getsize(vocals_path) > 0:
        return vocals_path

    model = None
    try:
        import demucs.apply
        model = _get_model(device=device)

        # Load audio (Demucs expects float32 tensor, stereo)
        audio, sr = sf.read(input_path, dtype="float32")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)  # mono → stereo
        else:
            audio = audio.T  # (samples, channels) → (channels, samples)

        mix = torch.from_numpy(audio).float()  # (channels, samples)
        mix = mix.unsqueeze(0)  # add batch dim → (1, channels, samples)
        if device != "cpu" and torch.cuda.is_available():
            mix = mix.cuda()

        # Separate
        with torch.no_grad():
            sources = demucs.apply.apply_model(
                model, mix, split=True, overlap=0.25, progress=False,
                device=device, shifts=1,
            )

        # Demucs v4 returns tensor: (batch, num_sources, channels, samples)
        # Source order is model.sources, typically: ['drums', 'bass', 'other', 'vocals']
        source_names = getattr(model, 'sources', SOURCES)
        try:
            vocals_idx = list(source_names).index("vocals")
        except ValueError:
            return ""

        # Extract vocals: shape (batch=1, channels, samples)
        vocals = sources[0, vocals_idx].cpu().numpy()  # (channels, samples)
        vocals = vocals.T  # → (samples, channels) for soundfile

        # Collapse to mono if stereo channels are identical (anime source)
        if vocals.shape[1] == 2 and np.allclose(vocals[:, 0], vocals[:, 1], atol=1e-5):
            vocals = vocals[:, 0]
        elif vocals.shape[1] == 2 and np.allclose(vocals[:, 0], vocals[:, 1], rtol=0.01):
            vocals = vocals[:, 0]  # nearly-identical stereo → mono

        os.makedirs(output_dir, exist_ok=True)
        sf.write(vocals_path, vocals, sr)

        # Free GPU memory
        del sources, mix, audio
        torch.cuda.empty_cache()

        if os.path.exists(vocals_path) and os.path.getsize(vocals_path) > 0:
            return vocals_path
        return ""

    except Exception as e:
        import traceback
        import datetime
        err_msg = f"[{datetime.datetime.now()}] {input_path}\n{traceback.format_exc()}\n"
        print(err_msg, flush=True)
        log_path = os.path.join(os.path.dirname(output_dir), "music_separate_errors.log")
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(err_msg + "\n")
        except Exception:
            pass
        return ""
    finally:
        if model is not None:
            _return_model(model)
