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

from config import DOWNLOAD_DIR, SUBTITLE_DIR, CLIPS_DIR, TEMP_DIR, DATA_DIR


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
                 "duration": s.duration_seconds, "data": s.data}
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
        self._jobs_file = os.path.join(DATA_DIR, "jobs.json")
        self._load_jobs()

    def _save_jobs(self):
        """Persist all jobs to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.jobs.items()}
            tmp = self._jobs_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Keep .bak as recovery fallback, but don't backup empty state
            if os.path.exists(self._jobs_file) and os.path.getsize(self._jobs_file) > 2:
                try:
                    os.replace(self._jobs_file, self._jobs_file + ".bak")
                except Exception:
                    pass
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
        except Exception as e:
            print(f"[pipeline] Failed to read jobs file: {e}")
            return

        restored = 0
        for jid, jd in data.items():
            try:
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
                # Running/pending jobs were interrupted by server restart
                if st in ("running", "pending"):
                    job.status = "interrupted"
                    job.current_step = "服务器重启中断，可继续执行或丢弃"
                    job.steps.append(StepResult(
                        step="interrupted",
                        status=StepStatus.FAILED,
                        message="服务器重启导致任务中断，可选择继续执行或丢弃",
                    ))
                for s in jd.get("steps", []):
                    job.steps.append(StepResult(
                        step=s["step"],
                        status=StepStatus(s["status"]),
                        message=s.get("message", ""),
                        duration_seconds=s.get("duration", 0),
                        data=s.get("data", {}),
                    ))
                self.jobs[jid] = job
                restored += 1
            except Exception as e:
                print(f"[pipeline] Failed to restore job {jid}: {e}")
        print(f"[pipeline] Restored {restored} jobs from disk")

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

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job — kills subprocess, cleans up files, marks as cancelled."""
        import shutil
        job = self.get_job(job_id)
        if not job:
            return {"status": "error", "message": "Job not found"}
        if job.status not in ("running", "pending"):
            return {"status": "error", "message": f"Job is {job.status}, cannot cancel"}
        # Kill running subprocess (ffmpeg extraction/splitting)
        job.kill_process()
        job.cancelled = True
        job.status = "cancelled"
        job.current_step = "cancelled"
        job.progress = 0
        job.steps.append(StepResult(
            step="cancel",
            status=StepStatus.FAILED,
            message="手动取消 — 所有处理已中止",
        ))
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
            if download_dir.startswith(DOWNLOAD_DIR):
                try:
                    for f in os.listdir(download_dir):
                        fpath = os.path.join(download_dir, f)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                    cleaned.append(f"download: {os.path.basename(job.mkv_path)}")
                except Exception:
                    pass

        job.subtitle_paths = []
        job.clip_paths = []
        job.clip_dir = ""
        self._save_jobs()
        return {"status": "ok", "message": f"Cancelled, cleaned: {', '.join(cleaned) if cleaned else 'nothing to clean'}"}

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from history."""
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self._save_jobs()
                return True
        return False

    def discard_interrupted(self, job_id: str) -> dict:
        """Discard an interrupted job — mark as cancelled."""
        job = self.get_job(job_id)
        if not job:
            return {"status": "error", "message": "Job not found"}
        if job.status != "interrupted":
            return {"status": "error", "message": f"Job is {job.status}, not interrupted"}
        job.status = "cancelled"
        job.current_step = "cancelled"
        self._save_jobs()
        return {"status": "ok", "message": "已丢弃中断的任务"}

    def resume_job(self, job_id: str) -> dict:
        """Mark an interrupted job as ready to resume. The caller re-submits it."""
        job = self.get_job(job_id)
        if not job:
            return {"status": "error", "message": "Job not found"}
        if job.status != "interrupted":
            return {"status": "error", "message": f"Job is {job.status}, not interrupted"}
        job.status = "pending"
        job.current_step = ""
        job.steps = [s for s in job.steps if s.step != "interrupted"]
        self._save_jobs()
        return {"status": "ok", "message": "任务已标记为待恢复", "job": job.to_dict()}

    def run_full_pipeline(
        self,
        job_id: str,
        mkv_path: str = "",
        magnet: str = "",
        title: str = "",
        hw_accel: str = "auto",
        on_step: Optional[Callable] = None,
        download_method: str = "aria2c",
    ) -> PipelineJob:
        """Run the complete pipeline: download (if magnet) -> extract subs -> split.

        If the job already exists (e.g. after a server restart and resume), reuse
        it and skip steps that are already marked COMPLETED. Intermediate state
        is saved after every step so progress survives unexpected restarts.

        Args:
            download_method: One of 'aria2c', 'qbittorrent', 'bitcomet'.
                             aria2c = headless daemon, others = GUI apps.

        If mkv_path is provided, skip the download step and use the local file directly.
        """
        # Reuse existing job (e.g. after resume) or create a fresh one
        existing = self.get_job(job_id)
        if existing:
            job = existing
            job.status = "running"
        else:
            job = self.create_job(job_id, title=title, magnet=magnet)
        if mkv_path:
            job.mkv_path = mkv_path

        def _step_done(step_name):
            return any(
                s.step == step_name and s.status == StepStatus.COMPLETED
                for s in job.steps
            )

        steps = list(job.steps)

        # --- Step 1: Submit to download backend (non-blocking) ---
        if magnet and not mkv_path:
            if _step_done("download"):
                job.current_step = "download"
                job.progress = 10
            else:
                job.current_step = "download"
                job.progress = 5
                t0 = time.time()

                method = download_method.lower()
                backend_label = download_method
                try:
                    if method == "qbittorrent":
                        from qbittorrent_client import add_magnet as _add_mag
                        _add_mag(magnet, save_path=DOWNLOAD_DIR)
                        steps.append(StepResult(
                            step="download",
                            status=StepStatus.COMPLETED,
                            message=f"已提交到 qBittorrent 下载\n下载完成后将自动检测并触发处理。",
                            data={"method": "qbittorrent"},
                            duration_seconds=time.time() - t0,
                        ))
                    elif method == "bitcomet":
                        from bitcomet_client import add_magnet as _add_mag
                        _add_mag(magnet, save_path=DOWNLOAD_DIR)
                        steps.append(StepResult(
                            step="download",
                            status=StepStatus.COMPLETED,
                            message=f"已提交到 BitComet 下载\n下载完成后将自动检测并触发处理。",
                            data={"method": "bitcomet"},
                            duration_seconds=time.time() - t0,
                        ))
                    else:
                        # Default: aria2c
                        from aria2_rpc import add_magnet, tell_status

                        gid = add_magnet(magnet)
                        job.gid = gid
                        status = tell_status(gid)
                        name = status.get("bittorrent", {}).get("info", {}).get("name", "")
                        total_mb = int(status.get("totalLength", 0)) / 1024 / 1024

                        steps.append(StepResult(
                            step="download",
                            status=StepStatus.COMPLETED,
                            message=(
                                f"已提交到 aria2c 后台下载"
                                + (f" ({name[:40]})" if name else "")
                                + (f" | 大小: {total_mb:.0f}MB" if total_mb > 0 else "")
                                + "\n下载完成后文件监控器会自动检测并触发处理。"
                            ),
                            data={"gid": gid, "name": name, "method": "aria2c"},
                            duration_seconds=time.time() - t0,
                        ))
                    job.progress = 10
                except RuntimeError as e:
                    msg = str(e)
                    if "already registered" in msg.lower() or "duplicate" in msg.lower():
                        steps.append(StepResult(
                            step="download",
                            status=StepStatus.COMPLETED,
                            message=f"该资源已在下载队列中，无需重复添加。\n下载完成后将自动处理。",
                            duration_seconds=time.time() - t0,
                        ))
                    else:
                        steps.append(StepResult(
                            step="download",
                            status=StepStatus.FAILED,
                            message=f"{backend_label} 添加失败: {msg}",
                            duration_seconds=time.time() - t0,
                        ))
                except Exception as e:
                    steps.append(StepResult(
                        step="download",
                        status=StepStatus.FAILED,
                        message=f"{backend_label} 异常: {e}",
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
            if _step_done("extract_subtitles"):
                # Restore subtitle paths from step data or filesystem
                subs = []
                for s in job.steps:
                    if s.step == "extract_subtitles" and s.status == StepStatus.COMPLETED:
                        subs = (s.data or {}).get("subtitle_paths", [])
                        break
                if subs and all(os.path.exists(p) for p in subs):
                    job.subtitle_paths = subs
                else:
                    # Fallback: find .ass/.srt files next to MKV or in SUBTITLE_DIR
                    import glob as _glob
                    base = os.path.splitext(os.path.basename(job.mkv_path))[0]
                    found = []
                    sub_dir = os.path.join(SUBTITLE_DIR, base)
                    if os.path.isdir(sub_dir):
                        found = sorted(_glob.glob(os.path.join(sub_dir, "*.ass"))) + \
                                sorted(_glob.glob(os.path.join(sub_dir, "*.srt")))
                    if not found:
                        mkv_dir = os.path.dirname(job.mkv_path)
                        found = sorted(_glob.glob(os.path.join(mkv_dir, base + "*.ass"))) + \
                                sorted(_glob.glob(os.path.join(mkv_dir, base + "*.srt")))
                    job.subtitle_paths = found
                job.progress = 65
            else:
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
            with self._lock:
                self.jobs[job_id] = job
            self._save_jobs()

        # --- Step 3: Split video by subtitles ---
        if job.subtitle_paths and job.mkv_path:
            if _step_done("split_video"):
                # Restore clip_dir from step data or filesystem
                clip_dir = job.clip_dir
                if not clip_dir or not os.path.isdir(clip_dir):
                    for s in job.steps:
                        if s.step == "split_video" and s.status == StepStatus.COMPLETED:
                            clip_dir = (s.data or {}).get("clip_dir", "")
                            break
                if not clip_dir or not os.path.isdir(clip_dir):
                    # Fallback: find clips dir by MKV name
                    base = os.path.splitext(os.path.basename(job.mkv_path))[0]
                    potential = os.path.join(CLIPS_DIR, base)
                    if os.path.isdir(potential):
                        clip_dir = potential
                job.clip_dir = clip_dir
                job.progress = 97
            else:
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
            with self._lock:
                self.jobs[job_id] = job
            self._save_jobs()

        # --- Step 4 & 5: Filter clips (fast, always run if there are clips) ---
        if job.clip_dir and os.path.isdir(job.clip_dir):
            if not _step_done("filter_clips"):
                job.current_step = "filter_clips"
                job.progress = 98
                t_filter = time.time()
                try:
                    filter_result = self.filter_clips(job_id, min_duration=2.0, clip_dir=job.clip_dir)
                    job.clip_paths = []
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
                job.steps = steps
                with self._lock:
                    self.jobs[job_id] = job
                self._save_jobs()

            if not _step_done("filter_silence"):
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
            job.steps = steps
            with self._lock:
                self.jobs[job_id] = job
            self._save_jobs()

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
