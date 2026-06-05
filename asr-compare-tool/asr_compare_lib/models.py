"""
ASR model handling: Qwen3-ASR API (DashScope) and FireRedASR2 (local + TensorRT).
"""
import os
import sys
import re
import base64
import logging
import threading

import numpy as np
import torch
import soundfile as sf

from .audio_utils import convert_to_pcm_wav

logger = logging.getLogger(__name__)

# --- Model singletons ---
_firered_model = None
_firered_model_key = None  # "firered" or "firered-trt" to track which is loaded
_model_lock = threading.Lock()


# =============================================================================
# Qwen3-ASR via DashScope API (synchronous, base64-encoded data URI)
# =============================================================================
def transcribe_qwen3_api(audio_path: str, api_key: str, api_base: str,
                         language: str = "zh", hotwords: str = "",
                         api_model: str = "qwen3-asr-flash") -> list:
    """Transcribe audio via DashScope Qwen3-ASR-Flash API (sync mode).

    Returns list of segment dicts [{text, start_ms, end_ms}, ...].
    """
    try:
        import dashscope
    except ImportError:
        raise RuntimeError(
            "使用API模型需要安装 dashscope SDK: pip install dashscope"
        ) from None

    dashscope.api_key = api_key
    dashscope.base_http_api_url = f"{api_base.rstrip('/')}/api/v1"

    ext = os.path.splitext(audio_path)[1].lower()
    _mime_map = {
        '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4',
        '.flac': 'audio/flac', '.aac': 'audio/aac', '.ogg': 'audio/ogg',
    }
    mime_type = _mime_map.get(ext, 'audio/wav')
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")
    data_uri = f"data:{mime_type};base64,{audio_b64}"

    messages = [
        {"role": "system", "content": [{"text": ""}]},
        {"role": "user", "content": [{"audio": data_uri}]},
    ]

    _kwargs: dict = {}
    if language and language != "auto":
        _kwargs["asr_options"] = {"language": language}
    if hotwords:
        _kwargs.setdefault("asr_options", {})["context"] = hotwords

    try:
        import asyncio
        import inspect
        from dashscope import AioMultiModalConversation
        _coro = AioMultiModalConversation.call(
            model=api_model, messages=messages, **_kwargs,
        )
        result = asyncio.run(_coro) if inspect.iscoroutine(_coro) else _coro
    except Exception as e:
        raise RuntimeError(f"API请求失败: {e}") from e

    output = result.output if hasattr(result, 'output') else result
    text = ""
    if isinstance(output, dict):
        choices = output.get("choices", [])
        if choices:
            content_list = choices[0].get("message", {}).get("content", [])
            text = "".join(
                c.get("text", "") for c in content_list if isinstance(c, dict)
            ).strip()
        if not text:
            text = (output.get("text", "") or "").strip()
    elif isinstance(result, dict):
        text = (result.get("text", "") or "").strip()

    if not text:
        return []

    try:
        info = sf.info(audio_path)
        duration_ms = int((info.frames / info.samplerate) * 1000) if info.samplerate > 0 else 0
    except Exception:
        duration_ms = 0

    return [{"text": text, "start_ms": 0, "end_ms": duration_ms}]


# =============================================================================
# FireRedASR2-TRT: TensorRT-accelerated Conformer encoder
# =============================================================================
class _FireRedAsr2TRT:
    """FireRed ASR with TensorRT-accelerated Conformer encoder.

    The Conformer encoder accounts for ~70% of inference time. Replacing it
    with a TensorRT engine gives 2-3x encoder speedup, ~1.5-2x end-to-end.

    Requires: pip install tensorrt (matching CUDA version)
    Created by: python export_firered_trt.py
    """

    def __init__(self, trt_engine_dir: str, tokenizer, decoder, feat_extractor,
                 device_id: int = 0, use_half: bool = True):
        import tensorrt as trt

        self.device_id = device_id
        self.use_half = use_half
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer
        self.decoder = decoder  # PyTorch AED decoder

        # Load TensorRT encoder engine
        engine_path = os.path.join(trt_engine_dir, "encoder.plan")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}. "
                "Run: python export_firered_trt.py"
            )
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()

        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._trt_runtime = trt.Runtime(self._trt_logger)
        self._trt_engine = self._trt_runtime.deserialize_cuda_engine(engine_buffer)
        self._trt_context = self._trt_engine.create_execution_context()
        self._stream = torch.cuda.current_stream(device_id)

        # Cache I/O binding info
        self._num_io = self._trt_engine.num_io_tensors
        self._io_names = [self._trt_engine.get_tensor_name(i) for i in range(self._num_io)]
        self._io_modes = [self._trt_engine.get_tensor_mode(name) for name in self._io_names]
        logger.info("FireRedASR2-TRT encoder loaded (%s)", engine_path)

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav):
        feats, lengths, durs, batch_wav, batch_uttid = \
            self.feat_extractor(batch_wav, batch_uttid)
        if feats is None:
            return [{"uttid": uttid, "text": ""} for uttid in batch_uttid]
        total_dur = sum(durs)
        feats = feats.cuda(self.device_id)
        lengths = lengths.cuda(self.device_id)
        if self.use_half:
            feats = feats.half()

        # TRT encoder forward
        enc_out, enc_len, src_mask = self._trt_encoder_forward(feats, lengths)

        # PyTorch AED decoder forward
        hyps = self.decoder.decode(
            enc_out, enc_len, src_mask,
            beam_size=1, nbest=1, decode_max_len=0,
        )

        results = []
        for uttid, hyp, dur in zip(batch_uttid, hyps, durs):
            hyp = hyp[0]
            hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
            text = self.tokenizer.detokenize(hyp_ids)
            text = re.sub(r"(<blank>)|(<sil>)", "", text)
            results.append({
                "uttid": uttid, "text": text.lower(),
                "confidence": round(hyp["confidence"].cpu().item(), 3),
                "dur_s": round(dur, 3),
            })
        return results

    def _trt_encoder_forward(self, padded_input, input_lengths):
        """Run TensorRT encoder inference with dynamic shapes."""
        import tensorrt as trt

        batch_size = padded_input.size(0)
        seq_len = padded_input.size(1)

        # Set dynamic input shapes
        self._trt_context.set_input_shape("padded_input",
            (batch_size, seq_len, padded_input.size(2)))
        self._trt_context.set_input_shape("input_lengths", (batch_size,))

        # Allocate output buffers
        out_shapes = {}
        for i in range(self._num_io):
            name = self._io_names[i]
            mode = self._io_modes[i]
            if mode == trt.TensorIOMode.OUTPUT:
                shape = self._trt_context.get_tensor_shape(name)
                out_shapes[name] = shape

        inputs = {
            "padded_input": padded_input.contiguous(),
            "input_lengths": input_lengths.contiguous().to(torch.int32),
        }
        outputs = {}
        for name, shape in out_shapes.items():
            dtype = self._trt_engine.get_tensor_dtype(name)
            tch_dtype = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
                trt.int64: torch.int64,
            }.get(dtype, torch.float32)
            outputs[name] = torch.empty(
                tuple(shape), dtype=tch_dtype, device=f"cuda:{self.device_id}")

        for name, tensor in {**inputs, **outputs}.items():
            self._trt_context.set_tensor_address(name, tensor.data_ptr())

        self._trt_context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        return outputs["enc_output"], outputs["output_lengths"], outputs["src_mask"]


# =============================================================================
# FireRedASR2 model loading (standard + TensorRT)
# =============================================================================
def _ensure_firered_path(firered_source_path: str):
    """Ensure FireRed source is on sys.path."""
    if not os.path.isdir(firered_source_path):
        raise FileNotFoundError(
            f"FireRedASR2 source path not found: {firered_source_path}"
        )
    if firered_source_path not in sys.path:
        sys.path.insert(0, firered_source_path)


def load_firered_model(firered_source_path: str, firered_models_dir: str,
                       device: str = "cuda") -> object:
    """Load the standard FireRedASR2-AED model (thread-safe, singleton cached).

    Optimized: beam_size=1 (greedy decode, 3x faster), skip VAD/LID at
    inference time since caller provides pre-segmented audio.
    """
    global _firered_model, _firered_model_key

    if _firered_model is not None and _firered_model_key == "firered":
        return _firered_model

    with _model_lock:
        if _firered_model is not None and _firered_model_key == "firered":
            return _firered_model

        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "FireRed ASR2 requires CUDA GPU. "
                f"device={device}, cuda_available={torch.cuda.is_available()}"
            )
        _ensure_firered_path(firered_source_path)

        from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
        from fireredasr2s.fireredasr2 import FireRedAsr2Config
        from fireredasr2s.fireredvad import FireRedVadConfig
        from fireredasr2s.fireredlid import FireRedLidConfig
        from fireredasr2s.fireredpunc import FireRedPuncConfig

        use_half = True
        if not torch.cuda.is_bf16_supported():
            try:
                torch.cuda.empty_cache()
                torch.tensor([1.0], device="cuda", dtype=torch.float16)
            except Exception:
                use_half = False

        vad_cfg = FireRedVadConfig(use_gpu=True)
        lid_cfg = FireRedLidConfig(use_gpu=True)
        asr_cfg = FireRedAsr2Config(
            use_gpu=True,
            use_half=use_half,
            return_timestamp=False,
            beam_size=1,          # greedy decode, ~3x faster
            decode_max_len=0,     # no length limit
        )
        punc_cfg = FireRedPuncConfig(use_gpu=True)

        sys_cfg = FireRedAsr2SystemConfig(
            os.path.join(firered_models_dir, "FireRedVAD", "VAD"),
            os.path.join(firered_models_dir, "FireRedLID"),
            "aed",
            os.path.join(firered_models_dir, "FireRedASR2-AED"),
            os.path.join(firered_models_dir, "FireRedPunc"),
            vad_cfg, lid_cfg, asr_cfg, punc_cfg,
            enable_vad=1, enable_lid=1, enable_punc=1,
        )
        _firered_model = FireRedAsr2System(sys_cfg)
        _firered_model_key = "firered"
        logger.info("FireRedASR2 model loaded (beam_size=1, greedy decode)")
        return _firered_model


def load_firered_trt_model(firered_source_path: str, firered_models_dir: str,
                           trt_engine_dir: str, device: str = "cuda") -> object:
    """Load FireRedASR2 with TensorRT-accelerated encoder.

    Requires a pre-built TensorRT engine in trt_engine_dir/encoder.plan.
    Run: python export_firered_trt.py first.
    """
    global _firered_model, _firered_model_key

    if _firered_model is not None and _firered_model_key == "firered-trt":
        return _firered_model

    with _model_lock:
        if _firered_model is not None and _firered_model_key == "firered-trt":
            return _firered_model

        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FireRed ASR2-TRT requires CUDA GPU.")
        if not trt_engine_dir or not os.path.isdir(trt_engine_dir):
            raise RuntimeError(
                "TRT engine directory not configured or not found. "
                "Run: python export_firered_trt.py first."
            )
        _ensure_firered_path(firered_source_path)

        from fireredasr2s.fireredasr2.asr import FireRedAsr2Config, load_fireredasr_aed_model

        aed_model_dir = os.path.join(firered_models_dir, "FireRedASR2-AED")
        cmvn_path = os.path.join(aed_model_dir, "cmvn.ark")
        dict_path = os.path.join(aed_model_dir, "dict.txt")
        spm_model = os.path.join(aed_model_dir, "train_bpe1000.model")
        model_path = os.path.join(aed_model_dir, "model.pth.tar")

        from fireredasr2s.fireredasr2.data.asr_feat import ASRFeatExtractor
        from fireredasr2s.fireredasr2.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

        feat_extractor = ASRFeatExtractor(cmvn_path)
        tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        full_model = load_fireredasr_aed_model(model_path)
        decoder = full_model.decoder
        decoder.eval()

        use_half = True
        if not torch.cuda.is_bf16_supported():
            try:
                torch.tensor([1.0], device="cuda", dtype=torch.float16)
            except Exception:
                use_half = False

        if use_half:
            decoder.half().cuda()
        else:
            decoder.cuda()

        _firered_model = _FireRedAsr2TRT(
            trt_engine_dir, tokenizer, decoder, feat_extractor,
            device_id=0, use_half=use_half,
        )
        _firered_model_key = "firered-trt"
        logger.info("FireRedASR2-TRT model loaded")
        return _firered_model


# =============================================================================
# FireRed transcription (optimized: direct asr.transcribe, skip VAD+LID+Punc)
# =============================================================================
def _read_audio_int16(audio_path: str) -> tuple:
    """Read audio as int16, convert to mono 16kHz if needed. Returns (samples, sr)."""
    work_path = audio_path
    temp_wav = None
    try:
        info = sf.info(audio_path)
        needs_convert = info.samplerate != 16000 or info.channels != 1
    except Exception:
        needs_convert = True

    if needs_convert:
        import uuid as _uuid
        import tempfile as _tempfile
        os.makedirs(_tempfile.gettempdir(), exist_ok=True)
        temp_wav = os.path.join(_tempfile.gettempdir(), f"firered_{_uuid.uuid4().hex[:8]}.wav")
        convert_to_pcm_wav(audio_path, temp_wav)
        work_path = temp_wav

    audio_np, sr = sf.read(work_path, dtype="int16")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1).astype("int16")

    if temp_wav and os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass
    return audio_np, sr


def transcribe_firered(audio_path: str, firered_source_path: str,
                       firered_models_dir: str, device: str = "cuda") -> str:
    """Transcribe a single audio segment with FireRed ASR.

    Uses direct model.asr.transcribe() — skips VAD+LID+Punc since
    segments are already VAD-segmented and language is known.

    Returns plain text string.
    """
    model = load_firered_model(firered_source_path, firered_models_dir, device)
    audio_np, sr = _read_audio_int16(audio_path)
    results = model.asr.transcribe([os.path.basename(audio_path)], [(sr, audio_np)])
    return " ".join(r.get("text", "").strip() for r in results if r.get("text", "").strip())


def transcribe_firered_batch(segments: list, firered_source_path: str,
                              firered_models_dir: str, device: str = "cuda") -> dict:
    """Batch transcribe all audio segments with FireRed ASR in a single GPU call.

    2-4x faster than per-segment calls — Conformer encoder processes padded
    batches, fully utilizing GPU parallelism.

    Args:
        segments: list of dicts with 'wav_path', 'name', 'index' keys
        firered_source_path: path to FireRedASR2S source
        firered_models_dir: path to pretrained models
        device: device string

    Returns:
        dict mapping seg_index -> text string
    """
    model = load_firered_model(firered_source_path, firered_models_dir, device)

    batch_uttid = []
    batch_wav = []
    seg_indices = []
    for seg in segments:
        try:
            audio_np, sr = sf.read(seg["wav_path"], dtype="int16")
        except Exception:
            audio_np = np.zeros(160, dtype=np.int16)
            sr = 16000
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1).astype("int16")
        batch_uttid.append(seg.get("name", str(seg["index"])))
        batch_wav.append((sr, audio_np))
        seg_indices.append(seg["index"])

    results = model.asr.transcribe(batch_uttid, batch_wav)
    text_map = {}
    for idx, r in zip(seg_indices, results):
        t = r.get("text", "").strip()
        text_map[idx] = t.lower() if t else ""
    return text_map


def transcribe_firered_trt_batch(segments: list, firered_source_path: str,
                                  firered_models_dir: str,
                                  trt_engine_dir: str, device: str = "cuda") -> dict:
    """Batch transcribe with FireRed TRT. Returns dict[seg_index -> text]."""
    model = load_firered_trt_model(firered_source_path, firered_models_dir,
                                   trt_engine_dir, device)

    batch_uttid = []
    batch_wav = []
    seg_indices = []
    for seg in segments:
        try:
            audio_np, sr = sf.read(seg["wav_path"], dtype="int16")
        except Exception:
            audio_np = np.zeros(160, dtype=np.int16)
            sr = 16000
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1).astype("int16")
        batch_uttid.append(seg.get("name", str(seg["index"])))
        batch_wav.append((sr, audio_np))
        seg_indices.append(seg["index"])

    results = model.transcribe(batch_uttid, batch_wav)
    text_map = {}
    for idx, r in zip(seg_indices, results):
        t = r.get("text", "").strip()
        text_map[idx] = t.lower() if t else ""
    return text_map


# =============================================================================
# GPU warm-up
# =============================================================================
def warmup_firered(firered_source_path: str, firered_models_dir: str,
                   device: str = "cuda", use_trt: bool = False,
                   trt_engine_dir: str = None):
    """Warm up FireRed ASR model by running dummy inferences.

    Critical for first-use initialization: FireRed lazy-loads sub-models
    on first call. The sine sweep triggers VAD, and the direct ASR call
    ensures encoder/decoder CUDA kernels are compiled.
    """
    import tempfile as _tempfile

    # Sine sweep to trigger VAD + full pipeline init
    _dummy_len = 32000
    _dummy = (np.sin(2 * np.pi * np.linspace(300, 3000, _dummy_len) / 16000
                      * np.arange(_dummy_len)) * 0.05).astype(np.float32)

    logger.info("Warming up FireRed ASR...")
    try:
        with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, _dummy, 16000, subtype="PCM_16")
            audio_np, sr = _read_audio_int16(tf.name)
        os.unlink(tf.name)

        if use_trt:
            model = load_firered_trt_model(firered_source_path, firered_models_dir,
                                           trt_engine_dir, device)
            model.transcribe(["warmup"], [(sr, audio_np)])
        else:
            model = load_firered_model(firered_source_path, firered_models_dir, device)
            model.asr.transcribe(["warmup"], [(sr, audio_np)])

        # Direct ASR warm-up for encoder CUDA kernel compilation
        _dummy_asr = np.random.randn(16000).astype(np.float32) * 0.001
        _dummy_asr_int16 = (_dummy_asr * 32767).astype(np.int16)
        if not use_trt:
            model.asr.transcribe(["warmup_asr"], [(16000, _dummy_asr_int16)])

        logger.info("FireRed ASR warm-up complete")
    except Exception as e:
        logger.warning("FireRed ASR warm-up failed (non-fatal): %s", e)
