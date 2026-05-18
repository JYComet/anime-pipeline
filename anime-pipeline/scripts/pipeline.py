"""
Pipeline orchestration module.
Coordinates the full workflow: download -> extract subtitles -> split video.
"""
import os
import json
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable

from config import DOWNLOAD_DIR, SUBTITLE_DIR, CLIPS_DIR, TEMP_DIR


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    step: str
    status: StepStatus
    message: str = ""
    data: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class PipelineJob:
    """Represents a single pipeline job processing one anime."""
    job_id: str
    title: str = ""
    magnet: str = ""
    gid: str = ""
    mkv_path: str = ""
    subtitle_paths: list[str] = field(default_factory=list)
    clip_dir: str = ""
    clip_paths: list[str] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    current_step: str = ""
    progress: float = 0.0  # 0-100
    status: str = "pending"  # pending, running, completed, failed, cancelled
    cancelled: bool = False

    def __post_init__(self):
        self.cancel_event = threading.Event()
        self._process = None  # current subprocess.Popen, killed on cancel

    def kill_process(self):
        """Kill the running subprocess if any."""
        self.cancel_event.set()
        proc = self._process
        if proc and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "title": self.title,
            "magnet": self.magnet,
            "gid": self.gid,
            "mkv_path": self.mkv_path,
            "subtitle_paths": self.subtitle_paths,
            "clip_dir": self.clip_dir,
            "clip_count": len(self.clip_paths),
            "steps": [
                {"step": s.step, "status": s.status.value, "message": s.message,
                 "duration": s.duration_seconds}
                for s in self.steps
            ],
            "current_step": self.current_step,
            "progress": self.progress,
            "status": self.status,
        }


class Pipeline:
    """Orchestrates the full anime processing pipeline."""

    def __init__(self):
        self.jobs: dict[str, PipelineJob] = {}
        self._lock = threading.Lock()
        self._jobs_file = os.path.join(os.path.dirname(DOWNLOAD_DIR), "jobs.json")
        self._load_jobs()

    def _save_jobs(self):
        """Persist all jobs to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.jobs.items()}
            tmp = self._jobs_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._jobs_file)
        except Exception as e:
            print(f"[pipeline] Failed to save jobs: {e}")

    def _load_jobs(self):
        """Restore jobs from disk on startup."""
        if not os.path.exists(self._jobs_file):
            return
        try:
            with open(self._jobs_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for jid, jd in data.items():
                # Only restore non-download jobs (download ones come from aria2c)
                st = jd.get("status", "")
                if st in ("download_submitted",):
                    continue  # these get restored from aria2c
                job = PipelineJob(
                    job_id=jid,
                    title=jd.get("title", ""),
                    magnet=jd.get("magnet", ""),
                )
                job.mkv_path = jd.get("mkv_path", "")
                job.subtitle_paths = jd.get("subtitle_paths", [])
                job.clip_dir = jd.get("clip_dir", "")
                job.clip_paths = []
                job.status = st
                job.progress = jd.get("progress", 0)
                job.current_step = jd.get("current_step", "")
                for s in jd.get("steps", []):
                    job.steps.append(StepResult(
                        step=s["step"],
                        status=StepStatus(s["status"]),
                        message=s.get("message", ""),
                        duration_seconds=s.get("duration", 0),
                    ))
                self.jobs[jid] = job
            print(f"[pipeline] Restored {len(self.jobs)} jobs from disk")
        except Exception as e:
            print(f"[pipeline] Failed to load jobs: {e}")

    def create_job(self, job_id: str, title: str = "", magnet: str = "") -> PipelineJob:
        job = PipelineJob(job_id=job_id, title=title, magnet=magnet)
        with self._lock:
            self.jobs[job_id] = job
        self._save_jobs()
        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self.jobs.values()]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job — kills subprocess, stops all work, marks as failed."""
        job = self.get_job(job_id)
        if not job:
            return False
        # Kill running subprocess (ffmpeg extraction/splitting)
        job.kill_process()
        job.cancelled = True
        job.status = "failed"
        job.current_step = "cancelled"
        job.progress = 0
        job.steps.append(StepResult(
            step="cancel",
            status=StepStatus.FAILED,
            message="手动取消 — 所有处理已中止",
        ))
        # Clear partial outputs
        job.subtitle_paths = []
        job.clip_paths = []
        job.clip_dir = ""
        self._save_jobs()
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from history."""
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self._save_jobs()
                return True
        return False

    def run_full_pipeline(
        self,
        job_id: str,
        mkv_path: str = "",
        magnet: str = "",
        title: str = "",
        hw_accel: str = "auto",
        on_step: Optional[Callable] = None,
    ) -> PipelineJob:
        """Run the complete pipeline: download (if magnet) -> extract subs -> split.

        If mkv_path is provided, skip the download step and use the local file directly.
        """
        job = self.create_job(job_id, title=title, magnet=magnet)
        job.mkv_path = mkv_path
        job.status = "running"

        steps = []

        # --- Step 1: Submit to aria2c daemon (non-blocking) ---
        if magnet and not mkv_path:
            job.current_step = "download"
            job.progress = 5
            t0 = time.time()
            try:
                from aria2_rpc import add_magnet, tell_status

                gid = add_magnet(magnet)
                job.gid = gid
                status = tell_status(gid)
                name = status.get("bittorrent", {}).get("info", {}).get("name", "")
                total_mb = int(status.get("totalLength", 0)) / 1024 / 1024

                job.progress = 10
                steps.append(StepResult(
                    step="download",
                    status=StepStatus.COMPLETED,
                    message=(
                        f"已提交到 aria2c 后台下载"
                        + (f" ({name[:40]})" if name else "")
                        + (f" | 大小: {total_mb:.0f}MB" if total_mb > 0 else "")
                        + "\n下载完成后文件监控器会自动检测并触发处理。"
                    ),
                    data={"gid": gid, "name": name},
                    duration_seconds=time.time() - t0,
                ))
            except RuntimeError as e:
                msg = str(e)
                # Handle duplicate: link to existing download
                if "already registered" in msg.lower() or "duplicate" in msg.lower():
                    steps.append(StepResult(
                        step="download",
                        status=StepStatus.COMPLETED,
                        message=f"该资源已在 aria2c 下载队列中，无需重复添加。\n下载完成后将自动处理。",
                        duration_seconds=time.time() - t0,
                    ))
                else:
                    steps.append(StepResult(
                        step="download",
                        status=StepStatus.FAILED,
                        message=f"aria2 添加失败: {msg}",
                        duration_seconds=time.time() - t0,
                    ))
            except Exception as e:
                steps.append(StepResult(
                    step="download",
                    status=StepStatus.FAILED,
                    message=f"aria2 异常: {e}",
                    duration_seconds=time.time() - t0,
                ))
            job.steps = steps
            # Don't proceed to extract — file watcher handles completion
            job.status = "download_submitted"
            with self._lock:
                self.jobs[job_id] = job
            self._save_jobs()
            return job

        # --- Step 2: Extract subtitles ---
        if job.mkv_path and os.path.exists(job.mkv_path):
            job.current_step = "extract_subtitles"
            job.progress = 35
            t0 = time.time()
            try:
                from extract_subs import extract_all_chinese_subs
                subs = extract_all_chinese_subs(job.mkv_path, cancel_event=job.cancel_event)
                job.subtitle_paths = subs
                job.progress = 65
                steps.append(StepResult(
                    step="extract_subtitles",
                    status=StepStatus.COMPLETED if subs else StepStatus.FAILED,
                    message=f"Extracted {len(subs)} subtitle track(s)",
                    data={"subtitle_paths": subs},
                    duration_seconds=time.time() - t0,
                ))
            except Exception as e:
                steps.append(StepResult(
                    step="extract_subtitles",
                    status=StepStatus.FAILED,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))
            job.steps = steps
            if on_step:
                on_step(steps[-1])

        # --- Step 3: Split video by subtitles ---
        if job.subtitle_paths and job.mkv_path:
            job.current_step = "split_video"
            job.progress = 70
            t0 = time.time()
            try:
                from split_video import split_video_by_subtitle
                all_clips = []

                for sub_path in job.subtitle_paths:
                    def split_progress(current, total, text):
                        pct = 70 + (current / max(total, 1)) * 25
                        job.progress = min(pct, 95)

                    clips = split_video_by_subtitle(
                        job.mkv_path,
                        sub_path,
                        hw_accel=hw_accel,
                        on_progress=split_progress,
                        cancel_event=job.cancel_event,
                    )
                    all_clips.extend(clips)

                if all_clips:
                    job.clip_dir = os.path.dirname(all_clips[0])
                job.clip_paths = all_clips
                job.progress = 97
                steps.append(StepResult(
                    step="split_video",
                    status=StepStatus.COMPLETED if all_clips else StepStatus.FAILED,
                    message=f"Created {len(all_clips)} video clips",
                    data={"clip_dir": job.clip_dir, "clip_count": len(all_clips)},
                    duration_seconds=time.time() - t0,
                ))

                # --- Step 4: Auto-filter short clips (default 1.0s) ---
                job.current_step = "filter_clips"
                job.progress = 98
                t_filter = time.time()
                try:
                    filter_result = self.filter_clips(job_id, min_duration=2.0, clip_dir=job.clip_dir)
                    job.clip_paths = []  # will be refreshed
                    job.progress = 100
                    steps.append(StepResult(
                        step="filter_clips",
                        status=StepStatus.COMPLETED,
                        message=filter_result.get("message", ""),
                        data=filter_result,
                        duration_seconds=time.time() - t_filter,
                    ))
                except Exception as fe:
                    steps.append(StepResult(
                        step="filter_clips",
                        status=StepStatus.FAILED,
                        message=str(fe),
                        duration_seconds=time.time() - t_filter,
                    ))

                # --- Step 5: Auto-filter high-silence clips ---
                job.current_step = "filter_silence"
                job.progress = 99
                t_silence = time.time()
                try:
                    silence_result = self.filter_silence(job_id, max_silence_ratio=0.6, clip_dir=job.clip_dir)
                    job.clip_paths = []
                    job.progress = 100
                    steps.append(StepResult(
                        step="filter_silence",
                        status=StepStatus.COMPLETED,
                        message=silence_result.get("message", ""),
                        data=silence_result,
                        duration_seconds=time.time() - t_silence,
                    ))
                except Exception as fe:
                    steps.append(StepResult(
                        step="filter_silence",
                        status=StepStatus.FAILED,
                        message=str(fe),
                        duration_seconds=time.time() - t_silence,
                    ))
            except Exception as e:
                steps.append(StepResult(
                    step="split_video",
                    status=StepStatus.FAILED,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))
            job.steps = steps
            if on_step:
                on_step(steps[-1])

        job.status = "completed"
        if any(s.status == StepStatus.FAILED for s in steps):
            # Check if we have at least some results
            if job.clip_paths:
                job.status = "completed"  # partial success is OK
            else:
                job.status = "failed"

        with self._lock:
            self.jobs[job_id] = job
            self._save_jobs()

        return job

    def filter_clips(self, job_id: str, min_duration: float = 1.0, clip_dir: str = "") -> dict:
        """Filter out clips shorter than min_duration seconds.

        Uses ffprobe to get each clip's duration and deletes short ones.
        If clip_dir is provided, uses that directly (robust against restarts).
        Returns {deleted, freed_mb, remaining, message}.
        """
        from config import FFPROBE
        import subprocess

        # Resolve clip_dir from job if not explicitly provided
        if not clip_dir:
            job = self.get_job(job_id)
            if job and job.clip_dir and os.path.exists(job.clip_dir):
                clip_dir = job.clip_dir
            elif job and job.clip_paths:
                clip_dir = os.path.dirname(job.clip_paths[0])
            else:
                # Fallback: search clips directory for this video's folder
                from config import CLIPS_DIR as clips_root
                if os.path.exists(clips_root):
                    for entry in os.listdir(clips_root):
                        full = os.path.join(clips_root, entry)
                        if os.path.isdir(full) and any(f.endswith('.mp4') for f in os.listdir(full)[:1]):
                            clip_dir = full
                            break
                if not clip_dir:
                    return {"deleted": 0, "freed_mb": 0, "remaining": 0, "message": "No clip directory found"}

        if not os.path.exists(clip_dir):
            return {"deleted": 0, "freed_mb": 0, "remaining": 0, "message": "Clip directory not found: " + clip_dir}

        # Gather all MP4 files in the clip directory
        all_clips = []
        for f in sorted(os.listdir(clip_dir)):
            if f.endswith('.mp4'):
                all_clips.append(os.path.join(clip_dir, f))

        deleted = 0
        freed_bytes = 0
        remaining = []

        for clip_path in all_clips:
            # Get duration via ffprobe
            try:
                result = subprocess.run(
                    [FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", clip_path],
                    capture_output=True, text=True, encoding='utf-8', errors='replace',
                    timeout=10,
                )
                duration = float(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                duration = 0

            if duration > 0 and duration < min_duration:
                size = os.path.getsize(clip_path)
                os.remove(clip_path)
                deleted += 1
                freed_bytes += size
            else:
                remaining.append(clip_path)

        freed_mb = freed_bytes / (1024 * 1024)

        # Update job if it exists
        job = self.get_job(job_id)
        if job:
            job.clip_paths = remaining
            with self._lock:
                self.jobs[job_id] = job

        return {
            "deleted": deleted,
            "freed_mb": round(freed_mb, 1),
            "remaining": len(remaining),
            "message": f"Removed {deleted} clips shorter than {min_duration}s, freed {freed_mb:.1f} MB, {len(remaining)} remaining",
        }

    def filter_silence(self, job_id: str, max_silence_ratio: float = 0.6, clip_dir: str = "") -> dict:
        """Filter out clips with too much silence.

        Extracts audio from each MP4, runs silence ratio check,
        and deletes clips exceeding max_silence_ratio.
        Returns {deleted, freed_mb, remaining, message}.
        """
        import subprocess
        import tempfile
        from config import FFMPEG, TEMP_DIR

        # Resolve clip_dir
        if not clip_dir:
            job = self.get_job(job_id)
            if job and job.clip_dir and os.path.exists(job.clip_dir):
                clip_dir = job.clip_dir
            elif job and job.clip_paths:
                clip_dir = os.path.dirname(job.clip_paths[0])
            else:
                return {"deleted": 0, "freed_mb": 0, "remaining": 0, "message": "No clip directory found"}

        if not os.path.exists(clip_dir):
            return {"deleted": 0, "freed_mb": 0, "remaining": 0, "message": "Clip directory not found"}

        # Collect all MP4 files
        all_clips = []
        for f in sorted(os.listdir(clip_dir)):
            if f.endswith('.mp4'):
                all_clips.append(os.path.join(clip_dir, f))

        if not all_clips:
            return {"deleted": 0, "freed_mb": 0, "remaining": 0, "message": "No clips to filter"}

        os.makedirs(TEMP_DIR, exist_ok=True)
        deleted = 0
        freed_bytes = 0
        remaining = []

        for clip_path in all_clips:
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            temp_wav = os.path.join(TEMP_DIR, f"_silence_check_{job_id}_{clip_name}.wav")

            try:
                # Extract audio to temp WAV
                result = subprocess.run(
                    [FFMPEG, "-y", "-i", clip_path, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "16000", "-ac", "1", temp_wav],
                    capture_output=True, text=True, encoding='utf-8', errors='replace',
                    timeout=30,
                )
                if result.returncode != 0 or not os.path.exists(temp_wav):
                    remaining.append(clip_path)
                    continue

                # Run silence ratio check
                from audio_pipeline import check_silence_ratio
                is_bad, info = check_silence_ratio(temp_wav)
                ratio = info.get("silence_ratio", 0)

                if is_bad or ratio > max_silence_ratio:
                    size = os.path.getsize(clip_path)
                    os.remove(clip_path)
                    deleted += 1
                    freed_bytes += size
                else:
                    remaining.append(clip_path)

            except Exception:
                remaining.append(clip_path)
            finally:
                # Clean up temp WAV
                try:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                except Exception:
                    pass

        freed_mb = freed_bytes / (1024 * 1024)

        # Update job
        job = self.get_job(job_id)
        if job:
            job.clip_paths = remaining
            with self._lock:
                self.jobs[job_id] = job

        return {
            "deleted": deleted,
            "freed_mb": round(freed_mb, 1),
            "remaining": len(remaining),
            "message": f"Removed {deleted} clips with silence ratio > {max_silence_ratio}, freed {freed_mb:.1f} MB, {len(remaining)} remaining",
        }


    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job and clean up its data."""
        import shutil
        job = self.get_job(job_id)
        if not job:
            return {"status": "error", "message": "Job not found"}
        if job.status not in ("running", "pending"):
            return {"status": "error", "message": f"Job is {job.status}, cannot cancel"}

        job.cancelled = True
        job.status = "cancelled"
        cleaned = []

        # Clean clip directory
        if job.clip_dir and os.path.exists(job.clip_dir):
            try:
                shutil.rmtree(job.clip_dir)
                cleaned.append(f"clips: {os.path.basename(job.clip_dir)}")
            except Exception:
                pass

        # Clean extracted subtitle files
        for sp in job.subtitle_paths:
            if os.path.exists(sp):
                try:
                    os.remove(sp)
                    cleaned.append(f"subtitle: {os.path.basename(sp)}")
                except Exception:
                    pass

        # Clean download if it came from magnet
        if job.magnet and job.mkv_path and os.path.exists(job.mkv_path):
            download_dir = os.path.dirname(job.mkv_path)
            # Only remove if it's under our DOWNLOAD_DIR
            if download_dir.startswith(DOWNLOAD_DIR):
                try:
                    # Remove the MKV file and any associated files in the same folder
                    for f in os.listdir(download_dir):
                        fpath = os.path.join(download_dir, f)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                    cleaned.append(f"download: {os.path.basename(job.mkv_path)}")
                except Exception:
                    pass

        with self._lock:
            self.jobs[job_id] = job

        return {"status": "ok", "message": f"Cancelled, cleaned: {', '.join(cleaned) if cleaned else 'nothing to clean'}"}


    def run_extract_and_split(
        self,
        job_id: str,
        mkv_path: str,
        title: str = "",
        hw_accel: str = "auto",
        on_step: Optional[Callable] = None,
    ) -> PipelineJob:
        """Run only the extract+split steps (for local files, no download)."""
        return self.run_full_pipeline(
            job_id=job_id,
            mkv_path=mkv_path,
            title=title,
            hw_accel=hw_accel,
            on_step=on_step,
        )


# Global pipeline instance
pipeline = Pipeline()
