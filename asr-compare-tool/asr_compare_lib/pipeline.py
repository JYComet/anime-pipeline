"""
Core ASR comparison pipeline.
Orchestrates: audio scanning, VAD, segment extraction, parallel ASR transcription,
text comparison, output file writing, and auto-deletion of low-match audio.
"""
import os
import sys
import shutil
import time
import logging
import threading
import concurrent.futures

import numpy as np
import soundfile as sf

from .audio_utils import (
    get_audio_duration, ensure_pcm_wav, run_vad_silero,
    chunk_vad_segments, segments_to_srt,
)
from .models import (
    transcribe_qwen3_api, transcribe_firered, transcribe_firered_batch,
    transcribe_firered_trt_batch, warmup_firered,
)
from .comparison import compare_segment_texts

logger = logging.getLogger(__name__)

# Audio extensions to process
_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.opus', '.wma'}

# Per-segment ASR timeout (seconds)
_SEGMENT_ASR_TIMEOUT = 120


class _CancelPipeline(Exception):
    """Raised when the pipeline is cancelled mid-processing."""
    pass


class ASRComparePipeline:
    """Main pipeline for comparing FireRed ASR vs Qwen3-API ASR."""

    def __init__(self, config: dict):
        """
        Args:
            config: Parsed configuration dict (from config.yaml).
        """
        # Paths
        self.audio_input_dir = config["paths"]["audio_input_dir"]
        self.output_dir = config["paths"]["output_dir"]
        self.firered_source_path = config["paths"]["firered_source_path"]
        self.firered_models_dir = config["paths"]["firered_models_dir"]
        self.firered_trt_engine_dir = config["paths"].get("firered_trt_engine_dir", "")

        # API
        self.api_key = config["api"]["dashscope_api_key"]
        self.api_base = config["api"]["dashscope_api_base"]
        self.api_model = config["api"]["api_model"]

        # ASR
        self.language = config["asr"]["language"]
        self.device = config["asr"]["device"]
        self.hotwords = config["asr"]["hotwords"]

        # VAD
        self.vad_engine = config["vad"]["engine"]
        self.segment_min_s = config["vad"]["segment_min_s"]
        self.segment_max_s = config["vad"]["segment_max_s"]

        # Compare
        self.match_threshold = config["compare"]["match_threshold"]
        self.filter_english = config["compare"]["filter_english"]
        self.delete_below_threshold = config["compare"]["delete_below_threshold"]
        self.keep_segments = config["compare"]["keep_segments"]

        # State
        self._cancel = threading.Event()

        # Validate
        if not self.api_key:
            raise ValueError("API key is required. Set dashscope_api_key in config.yaml")

    def cancel(self):
        """Signal cancellation."""
        self._cancel.set()

    def is_cancelled(self):
        return self._cancel.is_set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, use_progress: bool = True) -> dict:
        """Run the full ASR comparison pipeline.

        Args:
            use_progress: If True, display tqdm progress bars.

        Returns:
            Summary dict with keys:
              total, processed, kept, deleted, failed, results
        """
        _tqdm = _get_tqdm(use_progress)
        _tprint = _tqdm.write if _tqdm else _print_log

        # Scan audio files
        audio_files = self._scan_audio_files()
        if not audio_files:
            logger.warning("No audio files found in: %s", self.audio_input_dir)
            return {"total": 0, "processed": 0, "kept": 0, "deleted": 0,
                    "failed": 0, "results": []}

        _tprint(f"Found {len(audio_files)} audio file(s) to process")

        # Determine FireRed mode
        _use_trt = bool(self.firered_trt_engine_dir and os.path.isdir(self.firered_trt_engine_dir))
        _mode_label = "FireRed ASR2-TRT" if _use_trt else "FireRed ASR2"

        # Warm up FireRed model
        _tprint(f"Loading {_mode_label} model (this may take a few moments)...")
        if self.firered_source_path and self.firered_models_dir:
            warmup_firered(self.firered_source_path, self.firered_models_dir,
                          self.device, use_trt=_use_trt,
                          trt_engine_dir=self.firered_trt_engine_dir if _use_trt else None)
            _tprint(f"{_mode_label} model ready")
        else:
            _tprint("WARNING: FireRed paths not configured — skipping FireRed model")

        results = []
        kept = 0
        deleted = 0
        failed = 0

        # File-level progress bar
        pbar_files = None
        if _tqdm:
            pbar_files = _tqdm(
                total=len(audio_files),
                desc="Files",
                unit="file",
                position=0,
                bar_format="{desc:>12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

        try:
            for idx, audio_path in enumerate(audio_files):
                if self.is_cancelled():
                    _tprint("Pipeline cancelled by user")
                    break

                fname = os.path.basename(audio_path)
                if pbar_files:
                    pbar_files.set_description(f"Files [{fname[:40]}]")

                try:
                    result = self._process_one_audio(audio_path, use_progress, _tprint)
                    results.append(result)

                    match_rate = result.get("overall_match_rate", 0)
                    if match_rate < self.match_threshold:
                        if self.delete_below_threshold:
                            self._delete_audio_and_output(result)
                            deleted += 1
                            _tprint(f"DELETED (match {match_rate:.1f}% < {self.match_threshold}%): {fname}")
                        else:
                            kept += 1
                            _tprint(f"FLAGGED (match {match_rate:.1f}% < {self.match_threshold}%) — kept: {fname}")
                    else:
                        kept += 1
                        _tprint(f"KEPT (match {match_rate:.1f}% >= {self.match_threshold}%): {fname}")

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    failed += 1
                    logger.error("Failed to process %s: %s", fname, e, exc_info=True)
                    _tprint(f"FAILED: {fname} — {e}")
                    results.append({
                        "audio_path": audio_path,
                        "audio_name": os.path.splitext(fname)[0],
                        "error": str(e),
                    })

                if pbar_files:
                    pbar_files.update(1)
        except KeyboardInterrupt:
            _tprint("Pipeline interrupted by user — returning partial results")

        if pbar_files:
            pbar_files.close()

        summary = {
            "total": len(audio_files),
            "processed": len(results),
            "kept": kept,
            "deleted": deleted,
            "failed": failed,
            "results": results,
        }
        _tprint("=" * 60)
        _tprint("Pipeline Complete")
        _tprint(f"Total: {summary['total']} | Processed: {summary['processed']} | "
               f"Kept: {summary['kept']} | Deleted: {summary['deleted']} | "
               f"Failed: {summary['failed']}")
        return summary

    # ------------------------------------------------------------------
    # Audio file scanning
    # ------------------------------------------------------------------
    def _scan_audio_files(self) -> list:
        """Scan input directory for audio files. Returns sorted list of paths."""
        audio_dir = self.audio_input_dir
        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(f"Audio input directory not found: {audio_dir}")

        files = []
        for fname in sorted(os.listdir(audio_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in _AUDIO_EXTS:
                files.append(os.path.join(audio_dir, fname))

        logger.info("Scanned %s: found %d audio file(s)", audio_dir, len(files))
        return files

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unique_path(path: str) -> str:
        """Return a unique path by appending a counter if the file already exists."""
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------
    def _process_one_audio(self, audio_path: str, use_progress: bool,
                           _tprint) -> dict:
        """Process a single audio file: VAD -> segment -> ASR compare -> save.

        Follows the exact same flow as segment_and_compare_pipeline() in the original.
        """
        _tqdm = _get_tqdm(use_progress)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        t_start = time.time()

        # Ensure 16kHz mono WAV
        work_path = ensure_pcm_wav(audio_path)
        duration = get_audio_duration(work_path)
        duration_ms = int(duration * 1000)
        _tprint(f"  {base_name} — duration: {duration:.1f}s")

        # Output directory: output/{audio_name}/
        audio_out_dir = os.path.join(self.output_dir, base_name)
        os.makedirs(audio_out_dir, exist_ok=True)

        # Copy original audio to output
        out_audio_path = os.path.join(audio_out_dir, f"{base_name}.wav")
        if work_path != out_audio_path:
            shutil.copy2(work_path, out_audio_path)

        # ---- Step 1: VAD ----
        seg_out_dir = os.path.join(audio_out_dir, "segments")
        os.makedirs(seg_out_dir, exist_ok=True)

        vad_segments = []
        if self.vad_engine == "silero":
            vad_segments = run_vad_silero(work_path)

        if not vad_segments:
            if duration <= self.segment_max_s * 2:
                vad_segments = [(0, duration_ms)]
            else:
                chunk_ms = int((self.segment_min_s + self.segment_max_s) / 2 * 1000)
                vad_segments = [(i * chunk_ms, min((i + 1) * chunk_ms, duration_ms))
                               for i in range((duration_ms + chunk_ms - 1) // chunk_ms)]

        # ---- Step 2: Chunk ----
        chunks = chunk_vad_segments(vad_segments, self.segment_min_s, self.segment_max_s)
        if not chunks:
            chunks = [(0, duration_ms)]

        # ---- Step 3: Extract segment WAVs ----
        audio, sr = sf.read(work_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        segment_files = []
        for i, (start_ms, end_ms) in enumerate(chunks):
            start_samp = max(0, int(start_ms * sr / 1000))
            end_samp = min(len(audio), int(end_ms * sr / 1000))
            if end_samp <= start_samp:
                continue
            chunk = audio[start_samp:end_samp]
            seg_name = f"{base_name}_seg{i + 1:03d}.wav"
            seg_path = os.path.join(seg_out_dir, seg_name)
            sf.write(seg_path, chunk, sr, subtype="PCM_16")
            segment_files.append({
                "index": i + 1,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_s": round((end_ms - start_ms) / 1000, 1),
                "wav_path": seg_path,
                "name": seg_name,
            })

        # ---- Step 4: Batch FireRed transcription (2-4x GPU throughput) ----
        _use_trt = bool(self.firered_trt_engine_dir and os.path.isdir(self.firered_trt_engine_dir))
        _firered_texts: dict = {}  # seg_index -> text
        if segment_files:
            try:
                if _tqdm:
                    _tqdm.write("  Batching segments for FireRed ASR...")
                if _use_trt:
                    _firered_texts = transcribe_firered_trt_batch(
                        segment_files, self.firered_source_path,
                        self.firered_models_dir, self.firered_trt_engine_dir, self.device)
                else:
                    _firered_texts = transcribe_firered_batch(
                        segment_files, self.firered_source_path,
                        self.firered_models_dir, self.device)
                if _tqdm:
                    _tqdm.write(f"  FireRed batch: {len(_firered_texts)}/{len(segment_files)} segments transcribed")
            except Exception as _e:
                if _tqdm:
                    _tqdm.write(f"  WARNING: FireRed batch failed, will retry per-segment: {_e}")
                _firered_texts = {}

        # ---- Step 5: Per-segment Qwen3 API + Compare ----
        total_segs = len(segment_files)
        segment_results = []
        all_firered_segs = []
        all_qwen3_segs = []

        pbar_segs = None
        if _tqdm:
            pbar_segs = _tqdm(
                total=total_segs,
                desc=f"  Segments [{base_name[:30]}]",
                unit="seg",
                position=1,
                leave=False,
                bar_format="{desc:>12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

        for idx_seg, seg in enumerate(segment_files):
            if self.is_cancelled():
                raise _CancelPipeline("Pipeline cancelled")

            # Use pre-computed FireRed batch result when available
            text_firered = _firered_texts.get(seg["index"], "")
            error_firered = None

            # Fallback: per-segment transcription if batch failed or missed this segment
            if not text_firered and not error_firered and seg["index"] not in _firered_texts:
                try:
                    text_firered = transcribe_firered(
                        seg["wav_path"], self.firered_source_path,
                        self.firered_models_dir, self.device)
                except Exception as e:
                    error_firered = str(e)

            text_qwen3 = ""
            error_qwen3 = None

            def _run_qwen3():
                nonlocal text_qwen3, error_qwen3
                try:
                    segs = transcribe_qwen3_api(
                        seg["wav_path"], self.api_key, self.api_base,
                        self.language, self.hotwords, self.api_model,
                    )
                    text_qwen3 = " ".join(
                        s.get("text", "").strip() for s in segs if s.get("text", "").strip()
                    )
                except Exception as e:
                    error_qwen3 = str(e)

            # Run Qwen3 API in thread (FireRed already done via batch)
            t_b = threading.Thread(target=_run_qwen3, daemon=True)
            t_b.start()
            t_b.join(_SEGMENT_ASR_TIMEOUT)
            if t_b.is_alive():
                error_qwen3 = f"Qwen3 timed out after {_SEGMENT_ASR_TIMEOUT}s"

            # Save TXT
            txt_dir = os.path.join(seg_out_dir, "txt")
            os.makedirs(txt_dir, exist_ok=True)
            seg_base = os.path.splitext(seg["name"])[0]

            txt_path_fr = os.path.join(txt_dir, f"{seg_base}_firered.txt")
            txt_path_q3 = os.path.join(txt_dir, f"{seg_base}_qwen3-api.txt")

            with open(txt_path_fr, "w", encoding="utf-8") as f:
                f.write((text_firered or "").strip())
            with open(txt_path_q3, "w", encoding="utf-8") as f:
                f.write((text_qwen3 or "").strip())

            # Compare
            cmp = compare_segment_texts(
                text_firered, text_qwen3,
                self.match_threshold, self.filter_english,
            )

            seg_result = {
                "seg_index": seg["index"],
                "seg_name": seg["name"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "duration_s": seg["duration_s"],
                "wav_path": seg["wav_path"],
                "text_firered": text_firered,
                "text_qwen3": text_qwen3,
                "txt_path_firered": txt_path_fr,
                "txt_path_qwen3": txt_path_q3,
                "error_firered": error_firered,
                "error_qwen3": error_qwen3,
                "diff_percent": cmp["diff_percent"],
                "match_rate": cmp["match_rate"],
                "flagged": cmp["flagged"],
                "diff_chunks": cmp["diff_chunks"],
                "user_action": None,
            }
            segment_results.append(seg_result)

            if text_firered:
                all_firered_segs.append({"text": text_firered,
                    "start_ms": seg["start_ms"], "end_ms": seg["end_ms"]})
            if text_qwen3:
                all_qwen3_segs.append({"text": text_qwen3,
                    "start_ms": seg["start_ms"], "end_ms": seg["end_ms"]})

            if error_firered or error_qwen3:
                errs = []
                if error_firered:
                    errs.append(f"FireRed: {error_firered[:60]}")
                if error_qwen3:
                    errs.append(f"Qwen3: {error_qwen3[:60]}")
                logger.warning("  seg%03d errors: %s", seg["index"], " | ".join(errs))

            # Update segment progress bar
            if pbar_segs:
                pbar_segs.set_postfix_str(f"match={cmp['match_rate']:.1f}%")
                pbar_segs.update(1)

        if pbar_segs:
            pbar_segs.close()

        # ---- Step 6: Generate per-model SRT + TXT for whole audio ----
        srt_path_fr = self._unique_path(os.path.join(audio_out_dir, f"{base_name}_firered.srt"))
        srt_path_q3 = self._unique_path(os.path.join(audio_out_dir, f"{base_name}_qwen3-api.srt"))
        txt_path_fr = self._unique_path(os.path.join(audio_out_dir, f"{base_name}_firered.txt"))
        txt_path_q3 = self._unique_path(os.path.join(audio_out_dir, f"{base_name}_qwen3-api.txt"))

        if all_firered_segs:
            segments_to_srt(all_firered_segs, srt_path_fr)
        else:
            with open(srt_path_fr, "w", encoding="utf-8") as f:
                f.write("")
        if all_qwen3_segs:
            segments_to_srt(all_qwen3_segs, srt_path_q3)
        else:
            with open(srt_path_q3, "w", encoding="utf-8") as f:
                f.write("")

        full_text_fr = "\n".join(s.get("text", "").strip() for s in all_firered_segs)
        full_text_q3 = "\n".join(s.get("text", "").strip() for s in all_qwen3_segs)
        with open(txt_path_fr, "w", encoding="utf-8") as f:
            f.write(full_text_fr)
        with open(txt_path_q3, "w", encoding="utf-8") as f:
            f.write(full_text_q3)

        # ---- Step 7: Compute overall match rate ----
        total_weight = 0
        weighted_match = 0.0
        for sr_item in segment_results:
            w = max(len(sr_item["text_firered"]), len(sr_item["text_qwen3"]))
            weighted_match += sr_item["match_rate"] * w
            total_weight += w

        overall_match_rate = round(weighted_match / total_weight, 1) if total_weight > 0 else 0.0
        overall_flagged = overall_match_rate < self.match_threshold

        # Log segment details
        for sr_item in segment_results:
            flag = " [FLAGGED]" if sr_item["flagged"] else ""
            logger.info("  seg%03d: match=%.1f%% (diff=%.1f%%)%s  FR:%s  Q3:%s",
                       sr_item["seg_index"], sr_item["match_rate"],
                       sr_item["diff_percent"], flag,
                       (sr_item["text_firered"] or "(empty)")[:60],
                       (sr_item["text_qwen3"] or "(empty)")[:60])

        elapsed = time.time() - t_start
        logger.info("  Overall match: %.1f%% | Flagged: %s | Time: %.1fs",
                   overall_match_rate, overall_flagged, elapsed)

        # Clean up temp
        if work_path != audio_path and work_path != out_audio_path:
            try:
                os.remove(work_path)
            except OSError:
                pass

        return {
            "audio_path": audio_path,
            "audio_name": base_name,
            "output_dir": audio_out_dir,
            "duration_sec": round(duration, 1),
            "segment_count": len(segment_results),
            "overall_match_rate": overall_match_rate,
            "overall_flagged": overall_flagged,
            "srt_path_firered": srt_path_fr,
            "srt_path_qwen3": srt_path_q3,
            "txt_path_firered": txt_path_fr,
            "txt_path_qwen3": txt_path_q3,
            "segments_dir": seg_out_dir,
            "segments": segment_results,
            "elapsed_sec": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Auto-deletion
    # ------------------------------------------------------------------
    def _delete_audio_and_output(self, result: dict):
        """Delete the output directory for an audio with match rate below threshold."""
        output_dir = result.get("output_dir", "")
        if output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
            logger.info("  Deleted output directory: %s", output_dir)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _get_tqdm(use_progress: bool):
    """Return tqdm class if progress is enabled, else None."""
    if not use_progress:
        return None
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return None


def _print_log(msg: str):
    """Fallback print when tqdm is disabled."""
    logger.info(msg)
