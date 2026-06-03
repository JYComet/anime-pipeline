"""
FastAPI backend server for the Anime Pipeline.
Provides REST API endpoints for search, download, extraction, and splitting.
"""
import os
import json
import uuid
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, Query, Body, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import (
    PROJECT_ROOT, COMICUT_ROOT, VIDEO_DIR, DATA_DIR, DOWNLOAD_DIR, SUBTITLE_DIR, CLIPS_DIR,
    APPROVED_DIR, CLEANED_DIR, CLEANED_UNREVIEWED_DIR, DENOISED_APPROVED_DIR, STITCHED_DIR,
    PIPELINE_VIDEO_DIR, TEMP_DIR,
    ASR_DIR, ASR_AUDIO_DIR, ASR_SUBTITLE_DIR,
    ASR_COMPARE_DIR, ASR_COMPARE_SUBTITLE_DIR, ASR_COMPARE_AUDIO_DIR,
    ASR_COMPARE_OUTPUT_DIR, ASR_COMPARE_DISCARD_DIR,
    ASR_COMPARE_SEGMENTS_DIR, ASR_COMPARE_KEPT_DIR,
    HF_CACHE_DIR, MS_CACHE_DIR, MODELS_DIR, ASR_MODELS_DIR,
    EMOTION_DIR, EMOTION_DENOISE_DIR,
    MFA_SCRIPTS_DIR, MFA_MODELS_DIR, MFA_TEMP_DIR, MFA_DICT_PATH, MFA_DICT_PATH_ZH,
    MFA_RAW_WAV_DIR, MFA_WAV_DIR, MFA_TXT_DIR,
    MFA_ALIGNED_DIR, MFA_JSONL_DIR, MFA_POST_DIR, MFA_FILTERED_DIR, MFA_VALIDATE_DIR,
    SPLIT_DIR,
    FFPROBE, FFMPEG
)
from pipeline import pipeline, PipelineJob, StepStatus

# --- App setup ---
app = FastAPI(title="Anime Pipeline", version="1.0.0")

# Limit concurrent pipeline processing to 3
_pipeline_semaphore = threading.BoundedSemaphore(3)

# ============================================================
# Denoise job tracking
# ============================================================
import time
from dataclasses import dataclass, field

@dataclass
class DenoiseFileItem:
    name: str
    input_path: str
    status: str = "pending"   # pending/running/completed/discarded/error
    steps: list = field(default_factory=list)
    output_path: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "input_path": self.input_path,
            "status": self.status,
            "steps": self.steps,
            "output_path": self.output_path,
        }

@dataclass
class DenoiseJob:
    job_id: str
    video_name: str
    files: list  # list[DenoiseFileItem]
    status: str = "pending"   # pending/running/completed
    progress: float = 0.0
    current_file: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "video_name": self.video_name,
            "status": self.status,
            "progress": self.progress,
            "current_file": self.current_file,
            "files": [f.to_dict() for f in self.files],
            "created_at": self.created_at,
        }

_denoise_jobs: dict[str, DenoiseJob] = {}
_denoise_lock = threading.RLock()

_DENOISE_JOBS_FILE = os.path.join(DATA_DIR, "denoise_jobs.json")


def _save_denoise_jobs():
    """Persist denoise jobs to disk."""
    try:
        with _denoise_lock:
            data = {k: v.to_dict() for k, v in _denoise_jobs.items()}
        tmp = _DENOISE_JOBS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        if os.path.exists(_DENOISE_JOBS_FILE) and os.path.getsize(_DENOISE_JOBS_FILE) > 2:
            try:
                os.replace(_DENOISE_JOBS_FILE, _DENOISE_JOBS_FILE + ".bak")
            except Exception:
                pass
        os.replace(tmp, _DENOISE_JOBS_FILE)
    except Exception as e:
        print(f"[denoise] Failed to save jobs: {e}")


def _load_denoise_jobs():
    """Restore denoise jobs from disk on startup."""
    if not os.path.exists(_DENOISE_JOBS_FILE):
        return
    try:
        with open(_DENOISE_JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[startup] Failed to read denoise jobs file: {e}")
        return

    restored = 0
    for job_id, jd in data.items():
        try:
            files = []
            for fd in jd.get("files", []):
                files.append(DenoiseFileItem(
                    name=fd["name"],
                    input_path=fd["input_path"],
                    status=fd.get("status", "completed"),
                    steps=fd.get("steps", []),
                    output_path=fd.get("output_path", ""),
                ))
            job = DenoiseJob(
                job_id=jd["job_id"],
                video_name=jd.get("video_name", ""),
                files=files,
                status=jd.get("status", "completed"),
                progress=jd.get("progress", 100),
                current_file=jd.get("current_file", ""),
                created_at=jd.get("created_at", time.time()),
            )
            if job.status in ("running", "pending"):
                job.status = "interrupted"
                job.current_file = "服务器重启中断"
            _denoise_jobs[job_id] = job
            restored += 1
        except Exception as e:
            print(f"[startup] Failed to restore denoise job {job_id}: {e}")
    print(f"[startup] Restored {restored} denoise jobs from disk")


# ============================================================
# ASR job persistence
# ============================================================

_ASR_JOBS_FILE = os.path.join(DATA_DIR, "asr_jobs.json")


def _save_asr_jobs():
    """Persist ASR jobs to disk."""
    try:
        with _asr_lock:
            data = {}
            for k, v in _asr_jobs.items():
                d = dict(v)
                if "result" in d and d["result"] is not None and hasattr(d["result"], "copy"):
                    d["result"] = dict(d["result"])
                data[k] = d
        tmp = _ASR_JOBS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        if os.path.exists(_ASR_JOBS_FILE) and os.path.getsize(_ASR_JOBS_FILE) > 2:
            try:
                os.replace(_ASR_JOBS_FILE, _ASR_JOBS_FILE + ".bak")
            except Exception:
                pass
        os.replace(tmp, _ASR_JOBS_FILE)
    except Exception as e:
        print(f"[asr] Failed to save jobs: {e}")


def _load_asr_jobs():
    """Restore ASR jobs from disk on startup. Mark running/pending jobs as interrupted."""
    if not os.path.exists(_ASR_JOBS_FILE):
        return
    try:
        with open(_ASR_JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[startup] Failed to read ASR jobs file: {e}")
        return

    restored = 0
    for job_id, jd in data.items():
        try:
            if jd.get("status") in ("running", "pending"):
                jd["status"] = "interrupted"
                jd["current_step"] = "服务器重启中断"
            _asr_jobs[job_id] = jd
            restored += 1
        except Exception as e:
            print(f"[startup] Failed to restore ASR job {job_id}: {e}")
    print(f"[startup] Restored {restored} ASR jobs from disk")


# ============================================================
# Pipeline Video job tracking
# ============================================================

@dataclass
class PipelineVideoFileItem:
    name: str
    input_path: str
    wav_path: str = ""
    status: str = "pending"   # pending/converting/running/completed/error
    current_step: str = ""    # convert/enhance/super_resolve/asr/cut
    progress: float = 0.0
    steps: list = field(default_factory=list)       # [{step, status, message}]
    output_clips: list = field(default_factory=list)
    error: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "input_path": self.input_path,
            "wav_path": self.wav_path,
            "status": self.status,
            "current_step": self.current_step,
            "progress": self.progress,
            "steps": self.steps,
            "output_clips": self.output_clips,
            "error": self.error,
        }

@dataclass
class PipelineVideoJob:
    job_id: str
    folder_name: str
    files: list  # list[PipelineVideoFileItem]
    status: str = "pending"   # pending/running/completed/cancelled
    progress: float = 0.0
    cancelled: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "folder_name": self.folder_name,
            "status": self.status,
            "progress": self.progress,
            "files": [f.to_dict() for f in self.files],
            "created_at": self.created_at,
        }

_pipeline_video_jobs: dict[str, PipelineVideoJob] = {}
_pipeline_video_lock = threading.RLock()

_PV_JOBS_FILE = os.path.join(DATA_DIR, "pipeline_video_jobs.json")


def _save_pv_jobs():
    """Persist pipeline video jobs to disk."""
    try:
        with _pipeline_video_lock:
            data = {k: v.to_dict() for k, v in _pipeline_video_jobs.items()}
        tmp = _PV_JOBS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # Atomic replace: keep .bak as recovery fallback
        if os.path.exists(_PV_JOBS_FILE) and os.path.getsize(_PV_JOBS_FILE) > 2:
            try:
                os.replace(_PV_JOBS_FILE, _PV_JOBS_FILE + ".bak")
            except Exception:
                pass
        os.replace(tmp, _PV_JOBS_FILE)
    except Exception as e:
        print(f"[pv] Failed to save jobs: {e}")


def _load_pv_jobs():
    """Restore pipeline video jobs from disk on startup."""
    if not os.path.exists(_PV_JOBS_FILE):
        return
    try:
        with open(_PV_JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[startup] Failed to read pipeline video jobs file: {e}")
        return

    restored = 0
    for job_id, jd in data.items():
        try:
            files = []
            for fd in jd.get("files", []):
                files.append(PipelineVideoFileItem(
                    name=fd["name"],
                    input_path=fd["input_path"],
                    wav_path=fd.get("wav_path", ""),
                    status=fd.get("status", "completed"),
                    current_step=fd.get("current_step", ""),
                    progress=fd.get("progress", 100),
                    steps=fd.get("steps", []),
                    output_clips=fd.get("output_clips", []),
                    error=fd.get("error", ""),
                ))
            job = PipelineVideoJob(
                job_id=jd["job_id"],
                folder_name=jd.get("folder_name", ""),
                files=files,
                status=jd.get("status", "completed"),
                progress=jd.get("progress", 100),
                cancelled=jd.get("cancelled", False),
                created_at=jd.get("created_at", time.time()),
            )
            # Only keep running/pending jobs; discard completed/cancelled/error
            if job.status in ("running", "pending"):
                job.status = "interrupted"
                job.current_step = "interrupted (服务器重启中断)"
                _pipeline_video_jobs[job_id] = job
                restored += 1
            # completed/cancelled/error are discarded
        except Exception as e:
            print(f"[startup] Failed to restore pipeline video job {job_id}: {e}")
    _save_pv_jobs()  # persist the cleaned state
    print(f"[startup] Restored {restored} pipeline video jobs from disk")


def _update_pipeline_job_progress(job: PipelineVideoJob):
    """Compute overall job progress across all files.

    Uses f.progress (set per-segment by the phase-batch loop) as the primary
    indicator, falling back to step-weight heuristics only when f.progress is 0.
    """
    files = job.files
    if not files:
        job.progress = 0
        return
    total_pct = 0.0
    for f in files:
        if f.status == "completed":
            total_pct += 100
        elif f.status == "error":
            total_pct += max(f.progress, 0)
        elif f.progress > 0:
            # Use the per-segment progress set by the pipeline steps
            total_pct += f.progress
        else:
            # Fallback: estimate from current_step position
            step_order = ["duration_split", "convert", "music_separate",
                          "enhance", "super_resolve", "asr", "cut"]
            try:
                idx = step_order.index(f.current_step) if f.current_step in step_order else -1
                if idx >= 0:
                    total_pct += (idx + 1) * (100 // len(step_order))
            except ValueError:
                pass
    job.progress = round(total_pct / len(files))


def _auto_process_new_mkv(mkv_path: str):
    """Callback for file watcher: record download completion, no auto-split.

    Downloaded files appear in the "本地文件" tab for manual processing.
    """
    import uuid
    from datetime import datetime
    job_id = uuid.uuid4().hex[:12]
    title = os.path.splitext(os.path.basename(mkv_path))[0]
    ext = os.path.splitext(mkv_path)[1].lower()
    size_mb = os.path.getsize(mkv_path) / 1024 / 1024
    dl_time = datetime.fromtimestamp(os.path.getmtime(mkv_path)).strftime("%m-%d %H:%M")

    job = pipeline.create_job(job_id, title=title)
    job.mkv_path = mkv_path
    job.status = "completed"
    job.progress = 100
    from pipeline import StepResult, StepStatus
    job.steps = [StepResult(
        step="download",
        status=StepStatus.COMPLETED,
        message=f"{ext.upper()} 文件已下载完成（{size_mb:.0f}MB）\n下载时间: {dl_time}\n请前往「本地文件」页面手动处理。",
    )]
    print(f"[auto-process] Download complete: {title} ({size_mb:.0f}MB) at {dl_time}")


@app.on_event("startup")
def startup_services():
    """Start aria2c daemon and file watcher on server startup."""
    # Load user settings overrides
    from config import load_settings
    load_settings()

    def start_aria2():
        try:
            from aria2_rpc import ensure_running
            ok = ensure_running()
            print(f"[startup] aria2c daemon: {'running' if ok else 'failed'}")
        except Exception as e:
            print(f"[startup] aria2c error: {e}")
    threading.Thread(target=start_aria2, daemon=True).start()

    from file_watcher import start_watcher
    start_watcher(_auto_process_new_mkv, interval=5)

    # Restore denoise, pipeline video, and ASR jobs from disk
    _load_denoise_jobs()
    _load_pv_jobs()
    _load_asr_jobs()
    _load_asr_compare_jobs()
    _load_mfa_jobs()

    # Preload ASR models in background so first request is fast
    def _preload_asr():
        import time
        time.sleep(2)  # let server finish startup first
        try:
            from asr_pipeline import _get_asr_model
            print("[startup] Preloading Qwen3-ASR model...")
            _get_asr_model("qwen3-asr", device="cuda")
            print("[startup] Qwen3-ASR model loaded to GPU")
        except Exception as e:
            print(f"[startup] ASR preload failed (will load on first use): {e}")

    threading.Thread(target=_preload_asr, daemon=True).start()

    # Restore download jobs from aria2c state, with fallback to jobs.json
    def restore_jobs():
        import time
        time.sleep(3)
        restored_magnets = set()
        try:
            from aria2_rpc import list_all
            from pipeline import StepResult, StepStatus
            for item in list_all():
                name = item.get("bittorrent", {}).get("info", {}).get("name", "")
                if not name:
                    continue
                magnet_uri = item.get("magnetUri", "")
                total = int(item.get("totalLength", 0))
                completed = int(item.get("completedLength", 0))
                pct = (completed / total * 100) if total > 0 else 0
                gid = item.get("gid", "")[:12]
                job = pipeline.create_job(
                    f"aria2_{gid}",
                    title=name,
                    magnet=magnet_uri,
                )
                job.status = "download_submitted"
                job.progress = 5 + pct * 0.25
                job.current_step = f"download ({name[:25]}... {pct:.0f}%)" if pct > 0 else "download"
                job.steps = [StepResult(
                    step="download",
                    status=StepStatus.COMPLETED,
                    message=f"aria2c 后台下载中 ({completed/1024/1024:.0f}/{total/1024/1024:.0f}MB, {pct:.0f}%)",
                )]
                if magnet_uri:
                    restored_magnets.add(magnet_uri)
                print(f"[startup] Restored job {job.job_id}: {name[:40]} ({pct:.0f}%)")
        except Exception as e:
            print(f"[startup] Failed to restore jobs from aria2c: {e}")

        # Fallback: restore "download_submitted" jobs from jobs.json that weren't
        # matched by aria2c (e.g. aria2c session file was lost or corrupted).
        # These are marked as interrupted so the user can retry or discard them.
        try:
            import json
            jobs_file = pipeline._jobs_file
            if os.path.exists(jobs_file):
                with open(jobs_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                restored_fallback = 0
                for jid, jd in raw.items():
                    st = jd.get("status", "")
                    if st != "download_submitted":
                        continue
                    magnet = jd.get("magnet", "")
                    # Skip if already restored from aria2c
                    if magnet and magnet in restored_magnets:
                        continue
                    job = pipeline.create_job(
                        jid,
                        title=jd.get("title", ""),
                        magnet=magnet,
                    )
                    job.status = "interrupted"
                    job.current_step = "interrupted (下载中断：服务器重启且aria2c会话丢失)"
                    job.gid = jd.get("gid", "")
                    job.steps = []
                    for s in jd.get("steps", []):
                        job.steps.append(StepResult(
                            step=s["step"],
                            status=StepStatus(s["status"]),
                            message=s.get("message", ""),
                            duration_seconds=s.get("duration", 0),
                        ))
                    job.steps.append(StepResult(
                        step="interrupted",
                        status=StepStatus.FAILED,
                        message="服务器重启导致下载中断，aria2c会话已丢失。可选择重新下载或丢弃。",
                    ))
                    restored_fallback += 1
                    print(f"[startup] Fallback restored job {jid}: {jd.get('title', '')[:40]}")
                if restored_fallback:
                    print(f"[startup] Fallback restored {restored_fallback} download jobs from jobs.json")
        except Exception as e:
            print(f"[startup] Failed to restore jobs from jobs.json fallback: {e}")

    threading.Thread(target=restore_jobs, daemon=True).start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(VIDEO_DIR, exist_ok=True)


# ============================================================
# Search endpoints
# ============================================================

@app.get("/api/search")
def search_anime(
    query: str = Query("", description="Search keywords"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    provider: str = Query("", description="Provider: dmhy, moe, ani"),
    fansub: str = Query("", description="Fansub group name"),
    resource_type: str = Query("", description="Resource type filter"),
):
    """Search anime resources from animes.garden."""
    try:
        from downloader import search_resources
        result = search_resources(
            query=query,
            page=page,
            page_size=page_size,
            provider=provider,
            fansub=fansub,
            resource_type=resource_type,
        )
        return {
            "resources": [
                {
                    "id": r.id,
                    "provider": r.provider,
                    "provider_id": r.provider_id,
                    "title": r.title,
                    "href": r.href,
                    "type": r.type,
                    "magnet": r.magnet,
                    "size": r.size,
                    "size_mb": round(r.size_mb, 1),
                    "file_format": r.file_format,
                    "is_mkv": r.is_mkv,
                    "created_at": r.created_at,
                    "publisher": r.publisher.get("name", ""),
                    "fansub": r.fansub.get("name", ""),
                }
                for r in result.resources
            ],
            "page": result.page,
            "page_size": result.page_size,
            "complete": result.complete,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/resource/{provider}/{provider_id}")
def get_resource_detail(provider: str, provider_id: str):
    """Get detailed info for a single resource."""
    try:
        from downloader import get_resource_detail
        return get_resource_detail(provider, provider_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/teams")
def get_teams():
    """Fetch all fansub teams."""
    try:
        from downloader import fetch_teams
        teams = fetch_teams()
        return {"teams": teams}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/magnet/info")
def magnet_info(magnet: str = Query(..., description="Magnet URI to parse")):
    """Parse a magnet link and return its components."""
    try:
        from downloader import parse_magnet
        info = parse_magnet(magnet)
        # Also try fetching file list from DHT (short timeout, fire-and-forget style)
        info["files"] = []
        info["files_fetched"] = False
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/magnet/files")
def magnet_files(magnet: str = Query(..., description="Magnet URI to fetch file list for")):
    """Fetch torrent file list via DHT metadata retrieval."""
    try:
        from downloader import fetch_magnet_files
        files = fetch_magnet_files(magnet, timeout=60)
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Local files endpoints
# ============================================================

@app.get("/api/local/videos")
def list_local_videos():
    """List MKV/MP4 files available for processing (video folder + downloads).

    Sorted by download time (newest first).
    """
    from datetime import datetime
    video_extensions = {".mkv", ".mp4", ".avi", ".mov", ".wmv"}
    files = []

    for d in [VIDEO_DIR, DOWNLOAD_DIR]:
        if not os.path.exists(d):
            continue
        for root, dirs, filenames in os.walk(d):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in video_extensions:
                    full_path = os.path.join(root, f)
                    mtime = os.path.getmtime(full_path)
                    size_mb = os.path.getsize(full_path) / 1024 / 1024
                    files.append({
                        "name": f,
                        "path": full_path,
                        "source": "video" if d == VIDEO_DIR else "downloads",
                        "size_mb": round(size_mb, 1),
                        "mtime": mtime,
                        "time_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                    })

    # Sort by mtime descending (newest first)
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return {"videos": files}


# ============================================================
# Track inspection endpoint
# ============================================================

@app.get("/api/local/tracks")
def get_video_tracks(path: str = Query(..., description="Full path to video file")):
    """Get track information for a local video file."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        from extract_subs import list_all_tracks
        tracks = list_all_tracks(path)
        return {"tracks": tracks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/local/subtitles")
def list_local_subtitles():
    """List available subtitle files (SRT/ASS) from video folder and subtitles dir."""
    sub_extensions = {".srt", ".ass", ".ssa", ".vtt"}
    files = []
    for d in [VIDEO_DIR, SUBTITLE_DIR]:
        if not os.path.exists(d):
            continue
        for f in sorted(os.listdir(d)):
            ext = os.path.splitext(f)[1].lower()
            if ext in sub_extensions:
                full = os.path.join(d, f)
                files.append({
                    "name": f,
                    "path": full,
                    "size_kb": round(os.path.getsize(full) / 1024, 1),
                })
    return {"subtitles": files}


@app.post("/api/local/upload")
async def upload_local_video(file: UploadFile = File(...)):
    """Upload a video file to the video directory."""
    import shutil
    safe_name = file.filename.replace("\\", "/").split("/")[-1]
    dest = os.path.join(VIDEO_DIR, safe_name)

    if os.path.exists(dest):
        return {"status": "error", "detail": f"文件已存在: {safe_name}"}

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"status": "ok", "path": dest, "name": safe_name,
                "size_mb": round(os.path.getsize(dest) / 1024 / 1024, 1)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# Local files browser endpoints
# ============================================================

@app.get("/api/local-files/browse")
def browse_local_files(dir: str = Query("", description="Subdirectory relative to data dir")):
    """List folders and files in a directory under data/.

    Returns folders first, then files. Each entry has name, path, type, size, mtime.
    """
    from datetime import datetime
    import shutil
    base = os.path.normpath(DATA_DIR)
    # Resolve target dir, prevent path traversal
    target = os.path.normpath(os.path.join(base, dir))
    if not target.startswith(base):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="Directory not found")

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv"}
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}
    sub_exts = {".srt", ".ass", ".ssa", ".vtt"}

    def get_file_type(ext):
        ext = ext.lower()
        if ext in audio_exts: return "audio"
        if ext in video_exts: return "video"
        if ext in image_exts: return "image"
        if ext in sub_exts: return "subtitle"
        return "other"

    folders = []
    files = []
    try:
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            if os.path.isdir(full):
                folders.append({"name": name, "path": full, "type": "folder"})
            else:
                ext = os.path.splitext(name)[1]
                size = os.path.getsize(full)
                mtime = os.path.getmtime(full)
                files.append({
                    "name": name,
                    "path": full,
                    "type": get_file_type(ext),
                    "ext": ext,
                    "size": size,
                    "size_str": _format_file_size(size),
                    "mtime": mtime,
                    "time_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    # Breadcrumb: build path segments from data dir to current dir
    breadcrumbs = [{"name": "data", "dir": ""}]
    if dir:
        parts = dir.replace("\\", "/").split("/")
        accum = ""
        for p in parts:
            if p:
                accum = os.path.join(accum, p).replace("\\", "/") if accum else p
                breadcrumbs.append({"name": p, "dir": accum})

    return {
        "dir": dir,
        "full_path": target,
        "breadcrumbs": breadcrumbs,
        "folders": folders,
        "files": files,
    }


@app.delete("/api/local-files/delete")
def delete_local_file(path: str = Query(..., description="Full path to file or folder to delete")):
    """Delete a file or folder under data/ or video/."""
    import shutil
    allowed_bases = [os.path.normpath(DATA_DIR), os.path.normpath(VIDEO_DIR)]
    target = os.path.normpath(path)
    if not any(target.startswith(b) for b in allowed_bases):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Not found")

    try:
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/local-files/delete-batch")
def delete_local_files_batch(paths: list[str] = Body(..., embed=True)):
    """Delete multiple files/folders under data/."""
    import shutil
    base = os.path.normpath(DATA_DIR)
    deleted = []
    errors = []
    for path in paths:
        target = os.path.normpath(path)
        if not target.startswith(base):
            errors.append({"path": path, "error": "Access denied"})
            continue
        if not os.path.exists(target):
            errors.append({"path": path, "error": "Not found"})
            continue
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
            deleted.append(path)
        except Exception as e:
            errors.append({"path": path, "error": str(e)})
    return {"status": "ok", "deleted": len(deleted), "errors": errors}


@app.get("/api/local-files/stream")
def stream_local_file(path: str = Query(..., description="Full path to audio file")):
    """Stream an audio file for playback."""
    base = os.path.normpath(DATA_DIR)
    target = os.path.normpath(path)
    if not target.startswith(base):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="File not found")

    ext = os.path.splitext(target)[1].lower()
    media_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".opus": "audio/opus",
        ".wma": "audio/x-ms-wma",
    }
    media_type = media_map.get(ext, "application/octet-stream")
    return FileResponse(target, media_type=media_type)


def _format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


# ============================================================
# Pipeline job endpoints
# ============================================================

@app.post("/api/jobs/process-local")
def start_process_local(
    video_path: str = Query(..., description="Full path to local video file"),
    subtitle_path: str = Query("", description="Optional external subtitle file path"),
    hw_accel: str = Query("auto", description="HW acceleration: nvenc, amf, qsv, libx264, auto"),
):
    """Start processing a local video: extract subtitles + split by subtitle.

    If subtitle_path is provided, skip extraction and use that file directly.
    """
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    job_id = uuid.uuid4().hex[:12]
    title = os.path.splitext(os.path.basename(video_path))[0]

    def run():
        if subtitle_path and os.path.exists(subtitle_path):
            # Skip extraction, use provided subtitle for splitting only
            job = pipeline.create_job(job_id, title=title)
            job.mkv_path = video_path
            job.subtitle_paths = [subtitle_path]
            job.status = "running"
            job.current_step = "split_video"
            job.progress = 60

            from split_video import split_video_by_subtitle
            from pipeline import StepResult, StepStatus
            try:
                clips = split_video_by_subtitle(
                    video_path, subtitle_path, hw_accel=hw_accel,
                    on_progress=lambda c, t, txt: setattr(job, 'progress', 60 + (c / max(t, 1)) * 35),
                )
                job.clip_paths = clips
                if clips:
                    job.clip_dir = os.path.dirname(clips[0])
                job.progress = 100
                job.status = "completed"
                job.steps = [
                    StepResult("extract_subtitles", StepStatus.SKIPPED, "Used external subtitle"),
                    StepResult("split_video", StepStatus.COMPLETED, f"Created {len(clips)} clips"),
                ]
            except Exception as e:
                job.status = "failed"
                job.steps = [
                    StepResult("extract_subtitles", StepStatus.SKIPPED, "Used external subtitle"),
                    StepResult("split_video", StepStatus.FAILED, str(e)),
                ]
            finally:
                # Persist final state so restarted server can see completed/failed status
                pipeline._save_jobs()
        else:
            pipeline.run_extract_and_split(
                job_id=job_id,
                mkv_path=video_path,
                title=title,
                hw_accel=hw_accel,
            )

    threading.Thread(target=run, daemon=True).start()

    return {"job_id": job_id, "title": title, "status": "started"}


@app.post("/api/jobs/process-download")
def start_process_download(
    magnet: str = Query(..., description="Magnet link to download and process"),
    title: str = Query("", description="Anime title"),
    hw_accel: str = Query("auto"),
    method: str = Query("aria2c", description="Download method: aria2c, qbittorrent, bitcomet"),
):
    """Download via selected method, fall back to file watcher if P2P fails.

    The file watcher monitors video/ and downloads/ for new MKV files regardless.
    """
    job_id = uuid.uuid4().hex[:12]

    def run():
        try:
            pipeline.run_full_pipeline(
                job_id=job_id,
                magnet=magnet,
                title=title,
                hw_accel=hw_accel,
                download_method=method,
            )
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()

    from file_watcher import is_watching

    method_labels = {"aria2c": "aria2c", "qbittorrent": "qBittorrent", "bitcomet": "BitComet"}
    label = method_labels.get(method.lower(), method)

    return {
        "job_id": job_id,
        "title": title or magnet[:60],
        "status": "started",
        "magnet": magnet,
        "method": method,
        "watching": is_watching(),
        "message": (
            f"已提交到 {label} 下载。如果 P2P 下载失败（无可用节点），"
            "可使用磁力链接通过其他下载工具下载，"
            "将 MKV 放入 video/ 目录后系统会自动检测并处理。"
        ),
    }


@app.get("/api/backend/status")
def get_backend_status():
    """Check which download backends are available."""
    status = {
        "aria2c": False,
        "qbittorrent": False,
        "bitcomet": False,
    }
    try:
        from aria2_rpc import ensure_running
        status["aria2c"] = ensure_running()
    except Exception:
        pass
    try:
        from qbittorrent_client import ensure_running
        status["qbittorrent"] = ensure_running()
    except Exception:
        pass
    try:
        from bitcomet_client import ensure_running
        status["bitcomet"] = ensure_running()
    except Exception:
        pass
    return status


@app.post("/api/jobs/{job_id}/filter")
def filter_job_clips(
    job_id: str,
    min_duration: float = Query(2.0, description="Minimum clip duration in seconds"),
    clip_dir: str = Query("", description="Optional: direct path to clip directory"),
):
    """Filter out clips shorter than min_duration seconds."""
    result = pipeline.filter_clips(job_id, min_duration, clip_dir=clip_dir)
    return result


@app.post("/api/jobs/{job_id}/filter-silence")
def filter_job_silence(
    job_id: str,
    max_silence_ratio: float = Query(0.6, description="Maximum silence ratio (0-1)"),
    clip_dir: str = Query("", description="Optional: direct path to clip directory"),
):
    """Filter out clips with silence ratio above max_silence_ratio."""
    result = pipeline.filter_silence(job_id, max_silence_ratio, clip_dir=clip_dir)
    return result


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get the current status of a pipeline job."""
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running job and clean up its data."""
    result = pipeline.cancel_job(job_id)
    return result


@app.get("/api/jobs")
def list_jobs():
    """List all pipeline jobs. Updates download_submitted jobs from aria2c status."""
    jobs = pipeline.get_all_jobs()

    # Check aria2c status for download_submitted jobs
    for j in jobs:
        if j.get("status") != "download_submitted":
            continue
        job_gid = j.get("gid", "")
        try:
            from aria2_rpc import tell_status
            if job_gid:
                a = tell_status(job_gid)
                if not a:
                    continue
                a_status = a.get("status", "")
                a_completed = int(a.get("completedLength", 0))
                a_total = int(a.get("totalLength", 0))
                a_speed = int(a.get("downloadSpeed", 0))
                a_seeders = int(a.get("numSeeders", 0))
                a_conn = int(a.get("connections", 0))
                a_error = a.get("errorMessage", "")

                j["download_speed"] = a_speed
                j["total_mb"] = a_total / 1048576
                j["seeders"] = a_seeders
                j["connections"] = a_conn
                if a_status in ("complete", "removed"):
                    j["status"] = "completed"
                    j["progress"] = 100
                elif a_status == "error":
                    if "already registered" in a_error.lower():
                        # Duplicate — find the active download with same infohash
                        j["status"] = "download_submitted"
                        j["current_step"] = "已加入下载队列（重复提交）"
                        j["progress"] = 0
                    else:
                        j["status"] = "failed"
                        j["progress"] = 0
                        j["current_step"] = f"下载失败: {a_error[:50]}"
                elif a_total > 0:
                    j["progress"] = (a_completed / a_total) * 100
                    j["downloaded_mb"] = a_completed / 1048576
                    if a_speed > 0:
                        j["current_step"] = f"下载中 {a_speed//1024}KB/s 种子:{a_seeders}"
                    elif a_conn > 0:
                        j["current_step"] = f"连接中 种子:{a_seeders}"
                    else:
                        j["current_step"] = "等待节点中..."
                else:
                    j["current_step"] = "获取文件信息中..."
        except Exception:
            pass

    return {"jobs": jobs}


@app.delete("/api/jobs/clear")
def clear_jobs_by_status(status: str = Query("completed", description="Status to clear: completed or failed")):
    """Delete all jobs with a given status."""
    deleted = 0
    with pipeline._lock:
        to_delete = [jid for jid, j in pipeline.jobs.items() if j.status == status]
        for jid in to_delete:
            del pipeline.jobs[jid]
            deleted += 1
        pipeline._save_jobs()
    return {"status": "ok", "deleted": deleted}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job from history."""
    ok = pipeline.delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "ok"}


@app.post("/api/jobs/{job_id}/discard")
def discard_pipeline_job(job_id: str):
    """Discard an interrupted pipeline job."""
    result = pipeline.discard_interrupted(job_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/jobs/{job_id}/resume")
def resume_pipeline_job(job_id: str):
    """Resume an interrupted pipeline job. Re-submits the job for processing."""
    result = pipeline.resume_job(job_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    jd = result.get("job", {})
    # Re-submit pipeline based on what the job had
    if jd.get("magnet"):
        threading.Thread(target=_resubmit_pipeline, args=(job_id,), daemon=True).start()
        return {"status": "ok", "message": "任务已恢复，正在重新提交下载"}
    elif jd.get("mkv_path"):
        threading.Thread(target=_resubmit_split, args=(job_id,), daemon=True).start()
        return {"status": "ok", "message": "任务已恢复，正在重新分割"}
    return {"status": "ok", "message": "任务已标记为待恢复，需手动重新提交"}


def _resubmit_pipeline(job_id: str):
    """Re-run full pipeline for a resumed job."""
    j = pipeline.get_job(job_id)
    if not j:
        return
    pipeline.run_full_pipeline(
        job_id=job_id, magnet=j.magnet, title=j.title,
        hw_accel="auto", download_method="aria2c",
    )


def _resubmit_split(job_id: str):
    """Re-run extract+split for a resumed job."""
    j = pipeline.get_job(job_id)
    if not j:
        return
    pipeline.run_extract_and_split(
        job_id=job_id, mkv_path=j.mkv_path, title=j.title,
        hw_accel="auto",
    )


# ============================================================
# Results endpoints
# ============================================================

def _get_audio_info(filepath: str) -> dict:
    """Get detailed audio/video info using ffprobe."""
    from config import FFPROBE
    import subprocess
    info = {"duration_s": 0, "codec": "", "bitrate": "", "sample_rate": ""}
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", filepath],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            info["duration_s"] = round(float(fmt.get("duration", 0)), 1)
            info["bitrate"] = f"{int(fmt.get('bit_rate', 0)) // 1000}kbps" if fmt.get("bit_rate") else ""
            for s in data.get("streams", []):
                if s.get("codec_type") == "audio":
                    info["codec"] = s.get("codec_name", "")
                    info["sample_rate"] = s.get("sample_rate", "")
                    break
                elif s.get("codec_type") == "video":
                    info["codec"] = "video: " + s.get("codec_name", "")
    except Exception:
        pass
    return info


def _get_video_names_from_dirs(base_dir: str) -> set[str]:
    """Get video base names from directory entries (strip suffix)."""
    names = set()
    if os.path.exists(base_dir):
        for entry in os.listdir(base_dir):
            full = os.path.join(base_dir, entry)
            if os.path.isdir(full):
                names.add(entry)
    return names


@app.get("/api/results/videos")
def list_result_videos():
    """List all processed video names with denoise status."""
    clip_names = _get_video_names_from_dirs(CLIPS_DIR)
    approved_names = _get_video_names_from_dirs(APPROVED_DIR)
    all_names = sorted(clip_names | approved_names)

    videos = []
    for name in all_names:
        clip_dir = os.path.join(CLIPS_DIR, name)
        approved_dir = os.path.join(APPROVED_DIR, name)
        cleaned_dir = os.path.join(CLEANED_DIR, name)
        total_clips = len([f for f in os.listdir(clip_dir) if f.endswith(".mp4")]) if os.path.isdir(clip_dir) else 0
        approved = len([f for f in os.listdir(approved_dir) if f.endswith(".mp4")]) if os.path.isdir(approved_dir) else 0
        cleaned = len([f for f in os.listdir(cleaned_dir) if _is_denoised_wav(f)]) if os.path.isdir(cleaned_dir) else 0
        denoised_all = cleaned > 0 and cleaned >= approved
        videos.append({
            "name": name,
            "total_clips": total_clips,
            "approved": approved,
            "cleaned": cleaned,
            "denoised_all": denoised_all,
        })
    return {"videos": videos}


@app.get("/api/results/video/{video_name}/subtitles")
def get_video_subtitles(video_name: str):
    """Get subtitle files matching a video name."""
    files = []
    if os.path.exists(SUBTITLE_DIR):
        for f in sorted(os.listdir(SUBTITLE_DIR)):
            if video_name in f:
                full = os.path.join(SUBTITLE_DIR, f)
                files.append({
                    "name": f,
                    "path": full,
                    "size_kb": round(os.path.getsize(full) / 1024, 1),
                })
    return {"subtitles": files}


@app.get("/api/results/video/{video_name}/approved")
def get_video_approved(video_name: str):
    """Get approved clips for a video - returns WAV audio files with detailed info."""
    items = []
    approved_dir = os.path.join(APPROVED_DIR, video_name)
    if os.path.exists(approved_dir):
        for f in sorted(os.listdir(approved_dir)):
            if not f.lower().endswith(".wav"):
                continue
            full = os.path.join(approved_dir, f)
            info = _get_audio_info(full)
            items.append({
                "name": f,
                "path": full,
                "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                "duration_s": info["duration_s"],
                "codec": info["codec"],
                "bitrate": info["bitrate"],
                "sample_rate": info["sample_rate"],
            })
    return {"approved": items, "count": len(items)}


@app.get("/api/results/video/{video_name}/clips")
def get_video_clips(video_name: str):
    """Get all clips (pending + skipped) for a video."""
    items = []
    clip_dir = os.path.join(CLIPS_DIR, video_name)
    if os.path.exists(clip_dir):
        for f in sorted(os.listdir(clip_dir)):
            if not f.endswith(".mp4"):
                continue
            full = os.path.join(clip_dir, f)
            items.append({
                "name": f,
                "path": full,
                "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
            })
    return {"clips": items, "count": len(items)}


@app.get("/api/results/video/{video_name}/cleaned")
def get_video_cleaned(video_name: str):
    """Get denoised audio files for a video with detailed info."""
    items = []
    cleaned_dir = os.path.join(CLEANED_DIR, video_name)
    if os.path.exists(cleaned_dir):
        for f in sorted(os.listdir(cleaned_dir)):
            if not f.lower().endswith(".wav"):
                continue
            full = os.path.join(cleaned_dir, f)
            info = _get_audio_info(full)
            items.append({
                "name": f,
                "path": full,
                "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                "duration_s": info["duration_s"],
                "codec": info["codec"],
                "bitrate": info["bitrate"],
                "sample_rate": info["sample_rate"],
            })
    return {"cleaned": items, "count": len(items)}


@app.post("/api/results/delete")
def delete_result_files(
    paths: list[str] = Body([], embed=True, description="List of file paths to delete"),
):
    """Delete selected result files."""
    deleted = []
    failed = []
    for p in paths:
        if not p:
            continue
        if not os.path.exists(p):
            failed.append({"path": p, "reason": "Not found"})
            continue
        try:
            os.remove(p)
            deleted.append(p)
        except Exception as e:
            failed.append({"path": p, "reason": str(e)})
    return {"deleted": len(deleted), "failed": failed}


@app.get("/api/results/stream")
def stream_result(path: str = Query(..., description="Full path to file")):
    """Stream a result file for playback."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = path.lower()
    if ext.endswith(".wav"):
        mt = "audio/wav"
    elif ext.endswith(".aac"):
        mt = "audio/aac"
    elif ext.endswith(".mp3"):
        mt = "audio/mpeg"
    else:
        mt = "video/mp4"
    return FileResponse(path, media_type=mt)


# ============================================================
# Audio Review endpoints
# ============================================================

def _get_review_state(clip_dir: str) -> dict:
    """Load review state from a JSON file in the clip directory."""
    state_path = os.path.join(clip_dir, ".review_state.json")
    if os.path.exists(state_path):
        import json
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_review_state(clip_dir: str, state: dict):
    """Save review state to a JSON file in the clip directory."""
    state_path = os.path.join(clip_dir, ".review_state.json")
    import json
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def _find_clip_dirs() -> list[str]:
    """Find all clip directories with MP4 files."""
    dirs = []
    if os.path.exists(CLIPS_DIR):
        for entry in sorted(os.listdir(CLIPS_DIR)):
            full = os.path.join(CLIPS_DIR, entry)
            if os.path.isdir(full):
                if any(f.endswith(".mp4") for f in os.listdir(full)):
                    dirs.append(full)
    return dirs


@app.get("/api/review/clips")
def list_review_clips(
    clip_dir: str = Query("", description="Path to clip directory"),
):
    """List clips available for review, with review status."""
    if not clip_dir:
        dirs = _find_clip_dirs()
        if not dirs:
            return {"clip_dir": "", "clips": [], "stats": {"total": 0, "approved": 0, "pending": 0}}
        clip_dir = dirs[-1]  # most recent

    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Clip directory not found")

    state = _get_review_state(clip_dir)
    approved = state.get("approved", [])
    skipped = state.get("skipped", [])

    clips = []
    for f in sorted(os.listdir(clip_dir)):
        if not f.endswith(".mp4"):
            continue
        if f.startswith("."):
            continue
        full = os.path.join(clip_dir, f)
        if os.path.getsize(full) == 0:
            continue
        status = "approved" if f in approved else ("skipped" if f in skipped else "pending")
        clips.append({
            "name": f,
            "path": full,
            "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
            "status": status,
        })

    # Find first pending clip index
    pending_idx = next((i for i, c in enumerate(clips) if c["status"] == "pending"), -1)

    return {
        "clip_dir": clip_dir,
        "clips": clips,
        "pending_index": pending_idx,
        "stats": {
            "total": len(clips),
            "approved": len(approved),
            "skipped": len(skipped),
            "pending": len(clips) - len(approved) - len(skipped),
        },
    }


@app.post("/api/review/approve")
def approve_clip(
    clip_dir: str = Query(..., description="Path to clip directory"),
    clip_name: str = Query(..., description="Clip filename to approve"),
):
    """Approve a clip: copy it to the approved folder."""
    src = os.path.join(clip_dir, clip_name)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Clip not found")

    # Create approved subfolder mirroring the source directory name
    dir_name = os.path.basename(clip_dir)
    dst_dir = os.path.join(APPROVED_DIR, dir_name)
    os.makedirs(dst_dir, exist_ok=True)

    dst = os.path.join(dst_dir, clip_name)
    # If already exists in approved, skip
    if os.path.exists(dst):
        # Still mark as approved in state
        state = _get_review_state(clip_dir)
        state.setdefault("approved", []).append(clip_name)
        _save_review_state(clip_dir, state)
        return {"status": "ok", "action": "already_approved"}

    import shutil
    shutil.copy2(src, dst)

    # Update review state immediately
    state = _get_review_state(clip_dir)
    state.setdefault("approved", []).append(clip_name)
    if clip_name in state.get("skipped", []):
        state["skipped"].remove(clip_name)
    _save_review_state(clip_dir, state)

    # Convert to WAV audio (no auto-denoise — user triggers that manually)
    def _convert():
        try:
            from convert_audio import mp4_to_wav
            mp4_to_wav(dst)
        except Exception as e:
            print(f"[approve] WAV conversion failed: {e}")

    threading.Thread(target=_convert, daemon=True).start()

    return {"status": "ok", "action": "approved", "copied_to": dst}


@app.post("/api/review/skip")
def skip_clip(
    clip_dir: str = Query(..., description="Path to clip directory"),
    clip_name: str = Query(..., description="Clip filename to skip"),
):
    """Skip a clip: mark as skipped in review state."""
    src = os.path.join(clip_dir, clip_name)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Clip not found")

    state = _get_review_state(clip_dir)
    state.setdefault("skipped", []).append(clip_name)
    # Remove from approved if it was there
    if clip_name in state.get("approved", []):
        state["approved"].remove(clip_name)
    _save_review_state(clip_dir, state)

    return {"status": "ok", "action": "skipped"}


@app.post("/api/review/classify-emotion")
def classify_emotion(
    clip_dir: str = Query(..., description="Path to clip directory"),
    clip_name: str = Query(..., description="Clip filename to classify"),
    emotion: str = Query(..., description="Emotion category name"),
):
    """Copy the current clip to the corresponding emotion folder.
    If the clip already exists in another emotion folder, remove it from there first."""
    src = os.path.join(clip_dir, clip_name)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Clip not found")

    import shutil

    # Remove clip from any other emotion folder (single-choice)
    for e in EMOTIONS:
        if e == emotion:
            continue
        existing = os.path.join(EMOTION_DIR, e, clip_name)
        if os.path.exists(existing):
            os.remove(existing)

    dst_dir = os.path.join(EMOTION_DIR, emotion)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, clip_name)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

    return {"status": "ok", "emotion": emotion, "copied_to": dst}


EMOTIONS = ["中性", "激动", "自信", "好奇", "嫌弃", "生气", "温柔", "高兴", "傲娇", "无奈", "疲惫", "感动", "焦虑", "慌张", "俏皮", "委屈", "难过", "期待", "认真", "害羞", "严肃", "惊讶", "疑惑"]


def _get_clip_emotions(clip_name: str, base_dir: str) -> list:
    """Return list of emotion folders that contain a file named clip_name."""
    found = []
    for e in EMOTIONS:
        if os.path.exists(os.path.join(base_dir, e, clip_name)):
            found.append(e)
    return found


def _get_emotion_counts(base_dir: str) -> dict:
    """Return file counts for each emotion folder."""
    counts = {}
    for e in EMOTIONS:
        d = os.path.join(base_dir, e)
        counts[e] = len([f for f in os.listdir(d) if not f.startswith(".")]) if os.path.exists(d) else 0
    return counts


@app.get("/api/review/emotion-counts")
def get_emotion_counts():
    """Return file counts for each emotion folder (review)."""
    return {"counts": _get_emotion_counts(EMOTION_DIR)}


@app.get("/api/denoise-review/emotion-counts")
def get_denoise_emotion_counts():
    """Return file counts for each emotion folder (denoise review)."""
    return {"counts": _get_emotion_counts(EMOTION_DENOISE_DIR)}


@app.get("/api/review/clip-emotions")
def get_clip_emotions(
    clip_dir: str = Query(..., description="Path to clip directory"),
    clip_name: str = Query(..., description="Clip filename"),
):
    """Check which emotion folders already contain this clip."""
    return {"emotions": _get_clip_emotions(clip_name, EMOTION_DIR)}


@app.post("/api/review/clear-short")
def clear_short_clips(
    clip_dir: str = Query(..., description="Path to clip directory"),
    min_duration: float = Query(2.0, description="Delete clips with duration <= this value (seconds)"),
):
    """Delete all clips in the directory with duration <= min_duration seconds.

    Runs in background to avoid timeout from ffprobe scanning.
    """
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Clip directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from split_video import get_video_duration

        deleted = []
        errors = []
        mp4_files = [f for f in sorted(os.listdir(clip_dir)) if f.endswith(".mp4")]
        total = len(mp4_files)

        job = pipeline.create_job(job_id, title="clear-short: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "scanning"
        job.progress = 0

        for i, f in enumerate(mp4_files):
            full = os.path.join(clip_dir, f)
            try:
                dur = get_video_duration(full)
                if dur > 0 and dur <= min_duration:
                    os.remove(full)
                    deleted.append(f)
            except OSError as e:
                errors.append({"file": f, "reason": str(e)})
            job.progress = ((i + 1) / total) * 100

        # Update review state
        state = _get_review_state(clip_dir)
        for name in deleted:
            if name in state.get("approved", []):
                state["approved"].remove(name)
            if name in state.get("skipped", []):
                state["skipped"].remove(name)
        _save_review_state(clip_dir, state)

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="clear_short",
            status=StepStatus.COMPLETED,
            message=f"删除了 {len(deleted)} 个时长<={min_duration}s 的片段",
            data={"deleted": deleted},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "started",
        "job_id": job_id,
    }


@app.post("/api/review/detect-bgm")
def detect_bgm_clips(
    clip_dir: str = Query(..., description="Path to clip directory"),
):
    """Analyze all clips for BGM/reverb characteristics. Runs in background.

    Returns a job_id for polling.
    """
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Clip directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from audio_pipeline import analyze_clip_bgm

        mp4_files = sorted([f for f in os.listdir(clip_dir) if f.endswith(".mp4")])
        total = len(mp4_files)
        results = []

        job = pipeline.create_job(job_id, title="BGM检测: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        for i, f in enumerate(mp4_files):
            full = os.path.join(clip_dir, f)
            try:
                # Extract audio to temp WAV
                import tempfile
                wav_tmp = os.path.join(tempfile.gettempdir(), "bgm_detect_" + uuid.uuid4().hex[:8] + ".wav")
                subprocess.run(
                    [FFMPEG, "-y", "-i", full, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "16000", "-ac", "1", "-t", "120", wav_tmp],
                    capture_output=True, timeout=30,
                )
                if os.path.exists(wav_tmp) and os.path.getsize(wav_tmp) > 0:
                    analysis = analyze_clip_bgm(wav_tmp)
                    if analysis["has_bgm"] or analysis["has_reverb"]:
                        results.append({
                            "name": f,
                            "path": full,
                            "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                            **analysis,
                        })
                    try:
                        os.remove(wav_tmp)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[detect-bgm] Error on {f}: {e}")

            job.progress = ((i + 1) / total) * 100

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="detect_bgm",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {len(mp4_files)} 个片段中 {len(results)} 个含有BGM/混响",
            data={"results": results},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.post("/api/review/detect-male")
def detect_male_voices(
    clip_dir: str = Query(..., description="Path to clip directory"),
):
    """Analyze all clips for male voices using pitch detection. Runs in background.

    Uses ffmpeg pipe + ThreadPoolExecutor for ~4x speedup over sequential temp files.
    """
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Clip directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        import numpy as np
        from audio_pipeline import analyze_clip_gender
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_audio_pipe(video_path: str):
            """Extract first 30s of audio as float32 numpy array via ffmpeg pipe."""
            proc = subprocess.Popen(
                [FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "-t", "30", "-f", "wav", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            raw = proc.stdout.read()
            proc.wait(timeout=10)
            if len(raw) < 44:
                return None
            audio = np.frombuffer(raw[44:], dtype=np.int16).astype(np.float32) / 32768.0
            return audio

        mp4_files = sorted([f for f in os.listdir(clip_dir) if f.endswith(".mp4")])
        total = len(mp4_files)
        male_results = []

        job = pipeline.create_job(job_id, title="男声检测: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        # Build work list: (index, name, path)
        work = [(i, f, os.path.join(clip_dir, f)) for i, f in enumerate(mp4_files)]

        completed = 0

        def _process_one(item):
            idx, name, path = item
            try:
                audio = _extract_audio_pipe(path)
                if audio is not None and len(audio) > 0:
                    analysis = analyze_clip_gender(audio_array=audio)
                    if analysis["is_male"]:
                        return {
                            "name": name,
                            "path": path,
                            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
                            **analysis,
                        }
            except Exception as e:
                print(f"[detect-male] Error on {name}: {e}")
            return None

        # Process in parallel (4 workers — I/O bound, not CPU bound)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_process_one, w): w for w in work}
            for future in as_completed(futures):
                completed += 1
                job.progress = (completed / total) * 100
                r = future.result()
                if r is not None:
                    male_results.append(r)

        # Sort by original index order
        male_results.sort(key=lambda x: x["name"])

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="detect_male",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {total} 个片段中 {len(male_results)} 个含有男声",
            data={"results": male_results},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.post("/api/review/detect-multi-voice")
def detect_multi_voice_clips(
    clip_dir: str = Query(..., description="Path to clip directory"),
):
    """Analyze all clips for multiple human voices. Runs in background.

    Uses pitch segmentation analysis to detect clips where different
    speech segments have distinctly different voice ranges.
    """
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Clip directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        import numpy as np
        from audio_pipeline import analyze_clip_multi_voice
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_audio_pipe(video_path: str):
            proc = subprocess.Popen(
                [FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "-t", "60", "-f", "wav", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            raw = proc.stdout.read()
            proc.wait(timeout=10)
            if len(raw) < 44:
                return None
            audio = np.frombuffer(raw[44:], dtype=np.int16).astype(np.float32) / 32768.0
            return audio

        mp4_files = sorted([f for f in os.listdir(clip_dir) if f.endswith(".mp4")])
        total = len(mp4_files)
        multi_voice_results = []

        job = pipeline.create_job(job_id, title="多人声检测: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        work = [(i, f, os.path.join(clip_dir, f)) for i, f in enumerate(mp4_files)]
        completed = 0

        def _process_one(item):
            idx, name, path = item
            try:
                audio = _extract_audio_pipe(path)
                if audio is not None and len(audio) > 0:
                    analysis = analyze_clip_multi_voice(audio_array=audio)
                    if analysis["has_multi_voice"]:
                        return {
                            "name": name,
                            "path": path,
                            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
                            **analysis,
                        }
            except Exception as e:
                print(f"[detect-multi-voice] Error on {name}: {e}")
            return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_process_one, w): w for w in work}
            for future in as_completed(futures):
                completed += 1
                job.progress = (completed / total) * 100
                r = future.result()
                if r is not None:
                    multi_voice_results.append(r)

        multi_voice_results.sort(key=lambda x: x["name"])

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="detect_multi_voice",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {total} 个片段中 {len(multi_voice_results)} 个含多人声",
            data={"results": multi_voice_results},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.post("/api/review/detect-low-volume")
def detect_low_volume_clips(
    clip_dir: str = Query(..., description="Path to clip directory"),
):
    """Detect clips with long segments of low-volume-but-audible content (e.g. background voices).
    Runs in background. Returns flagged clips, does NOT auto-delete."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Clip directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        import numpy as np
        from audio_pipeline import detect_low_volume_segments
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_audio_pipe(video_path: str):
            proc = subprocess.Popen(
                [FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "-t", "120", "-f", "wav", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            raw = proc.stdout.read()
            proc.wait(timeout=10)
            if len(raw) < 44:
                return None
            audio = np.frombuffer(raw[44:], dtype=np.int16).astype(np.float32) / 32768.0
            return audio

        mp4_files = sorted([f for f in os.listdir(clip_dir) if f.endswith(".mp4")])
        total = len(mp4_files)
        results = []

        job = pipeline.create_job(job_id, title="低音量检测: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        work = [(i, f, os.path.join(clip_dir, f)) for i, f in enumerate(mp4_files)]
        completed = 0

        def _process_one(item):
            idx, name, path = item
            try:
                audio = _extract_audio_pipe(path)
                if audio is not None and len(audio) > 0:
                    analysis = detect_low_volume_segments(audio_array=audio)
                    if analysis["has_low_volume"]:
                        return {
                            "name": name,
                            "path": path,
                            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
                            **analysis,
                        }
            except Exception as e:
                print(f"[detect-low-volume] Error on {name}: {e}")
            return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_process_one, w): w for w in work}
            for future in as_completed(futures):
                completed += 1
                job.progress = (completed / total) * 100
                r = future.result()
                if r is not None:
                    results.append(r)

        results.sort(key=lambda x: x["name"])

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="detect_low_volume",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {total} 个片段中 {len(results)} 个含长段低音量",
            data={"results": results},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.get("/api/review/stream")
def stream_clip(path: str = Query(..., description="Full path to clip file")):
    """Stream a clip file for playback in the browser."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/review/dirs")
def list_review_dirs():
    """List all clip directories available for review."""
    dirs = _find_clip_dirs()
    return {
        "dirs": [{"name": os.path.basename(d), "path": d} for d in dirs],
    }


# ============================================================
# Denoise endpoints
# ============================================================

# Temp upload directory for manually selected files
DENOISE_UPLOAD_DIR = os.path.join(DATA_DIR, "temp_uploads")
os.makedirs(DENOISE_UPLOAD_DIR, exist_ok=True)


@app.post("/api/denoise/upload")
async def denoise_upload_files(files: list[UploadFile] = File(...)):
    """Upload WAV files manually selected by the user. Returns server-side paths."""
    saved = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".wav"):
            continue
        # Sanitize filename
        safe_name = f.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not safe_name:
            safe_name = f.filename
        # Avoid overwrites
        dest = os.path.join(DENOISE_UPLOAD_DIR, safe_name)
        base, ext = os.path.splitext(safe_name)
        counter = 1
        while os.path.exists(dest):
            dest = os.path.join(DENOISE_UPLOAD_DIR, f"{base}_{counter}{ext}")
            counter += 1
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        info = _get_audio_info(dest)
        saved.append({
            "name": os.path.basename(dest),
            "path": dest.replace("\\", "/"),
            "size_mb": round(os.path.getsize(dest) / 1024 / 1024, 1),
            "duration_s": info["duration_s"],
        })
    return {"status": "ok", "files": saved, "count": len(saved)}


@app.post("/api/denoise/batch")
def denoise_batch(payload: dict = Body(...)):
    """Start a batch denoise job. Processes files sequentially in background.

    Body: {video_name: str, paths: [str, ...], steps: [str, ...]?}

    steps is an optional ordered list of step keys to execute.
    Default: ["enhance", "super_resolve", "reverb", "silence", "vad", "pad"]
    """
    video_name = payload.get("video_name", "")
    paths = payload.get("paths", [])
    steps = payload.get("steps")  # optional, list of step keys in order

    if not paths:
        raise HTTPException(status_code=400, detail="No paths provided")

    # Filter: only existing WAV files
    valid_paths = [p for p in paths if os.path.exists(p) and p.lower().endswith(".wav")]
    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid WAV files found")

    job_id = uuid.uuid4().hex[:12]
    files = [DenoiseFileItem(
        name=os.path.basename(p),
        input_path=p,
    ) for p in valid_paths]

    job = DenoiseJob(job_id=job_id, video_name=video_name, files=files)
    with _denoise_lock:
        _denoise_jobs[job_id] = job
        _save_denoise_jobs()

    # Run in background thread
    def _run():
        from denoise_audio import run_full_denoise

        job.status = "running"
        total = len(files)
        completed_count = 0

        for f in job.files:
            f.status = "running"
            job.current_file = f.name
            job.progress = (completed_count / total) * 100

            video_dir = video_name or os.path.basename(os.path.dirname(f.input_path))
            output_dir = os.path.join(CLEANED_DIR, video_dir)
            os.makedirs(output_dir, exist_ok=True)

            def on_step(step_key, status, message):
                f.steps.append({"step": step_key, "status": status, "message": message})

            result = run_full_denoise(f.input_path, output_dir, on_step=on_step, steps=steps)

            if result["success"]:
                f.status = "completed"
                f.output_path = result["output_path"]
            elif result.get("discard_reason"):
                f.status = "discarded"
            else:
                f.status = "error"

            completed_count += 1
            job.progress = (completed_count / total) * 100

        job.current_file = ""
        job.status = "completed"
        job.progress = 100
        _save_denoise_jobs()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {"status": "ok", "job_id": job_id, "file_count": len(valid_paths)}


@app.get("/api/denoise/job/{job_id}")
def get_denoise_job(job_id: str):
    """Poll denoise job status. Returns full job with per-file step details."""
    with _denoise_lock:
        job = _denoise_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/api/denoise/jobs")
def list_denoise_jobs():
    """List all denoise jobs (latest 20)."""
    with _denoise_lock:
        jobs = sorted(_denoise_jobs.values(), key=lambda j: j.created_at, reverse=True)[:20]
    return {"jobs": [j.to_dict() for j in jobs], "count": len(jobs)}


@app.post("/api/denoise/job/{job_id}/discard")
def denoise_discard_job(job_id: str):
    """Discard an interrupted denoise job."""
    with _denoise_lock:
        job = _denoise_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("interrupted", "pending"):
        raise HTTPException(status_code=400, detail=f"Job is {job.status}, not interrupted/pending")
    job.status = "cancelled"
    _save_denoise_jobs()
    return {"status": "ok", "message": "已丢弃中断的任务"}


@app.post("/api/denoise/job/{job_id}/resume")
def denoise_resume_job(job_id: str):
    """Resume an interrupted or pending denoise job, skipping completed files."""
    with _denoise_lock:
        job = _denoise_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("interrupted", "pending"):
        raise HTTPException(status_code=400, detail=f"Job is {job.status}, not interrupted/pending")
    job.status = "running"
    _save_denoise_jobs()
    threading.Thread(target=_run_denoise_resume, args=(job_id,), daemon=True).start()
    return {"status": "ok", "message": "已恢复执行"}


def _run_denoise_resume(job_id: str):
    """Resume an interrupted denoise job, skipping completed files."""
    from denoise_audio import run_full_denoise

    with _denoise_lock:
        job = _denoise_jobs.get(job_id)
    if not job:
        return

    completed = [f for f in job.files if f.status == "completed"]
    pending = [f for f in job.files if f.status != "completed"]

    if not pending:
        job.status = "completed"
        job.progress = 100
        _save_denoise_jobs()
        return

    total = len(job.files)
    already_done = len(completed)
    completed_count = already_done

    job.status = "running"
    for f in pending:
        f.status = "running"
        job.current_file = f.name
        job.progress = (completed_count / total) * 100

        video_dir = job.video_name or os.path.basename(os.path.dirname(f.input_path))
        output_dir = os.path.join(CLEANED_DIR, video_dir)
        os.makedirs(output_dir, exist_ok=True)

        def on_step(step_key, status, message):
            f.steps.append({"step": step_key, "status": status, "message": message})

        # Check if processing WAV or MP4
        is_mp4 = f.input_path.lower().endswith(".mp4")
        wav_input = f.input_path
        wav_tmp = ""

        if is_mp4:
            import tempfile
            wav_tmp = os.path.join(tempfile.gettempdir(), "resume_denoise_" + uuid.uuid4().hex[:8] + ".wav")
            try:
                conv = subprocess.run(
                    [FFMPEG, "-y", "-i", f.input_path, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "48000", "-ac", "1", wav_tmp],
                    capture_output=True, timeout=60,
                )
                if conv.returncode != 0 or not os.path.exists(wav_tmp) or os.path.getsize(wav_tmp) == 0:
                    f.status = "error"
                    f.steps.append({"step": "convert", "status": "error", "message": "音频提取失败"})
                    completed_count += 1
                    continue
                wav_input = wav_tmp
            except Exception as e:
                f.status = "error"
                f.steps.append({"step": "convert", "status": "error", "message": str(e)[:100]})
                completed_count += 1
                continue

        result = run_full_denoise(wav_input, output_dir, on_step=on_step)

        if wav_tmp:
            try:
                os.remove(wav_tmp)
            except Exception:
                pass

        if result["success"]:
            f.status = "completed"
            f.output_path = result["output_path"]
        elif result.get("discard_reason"):
            f.status = "discarded"
        else:
            f.status = "error"

        completed_count += 1

    job.current_file = ""
    job.status = "completed"
    job.progress = 100
    _save_denoise_jobs()


@app.get("/api/denoise/sources")
def list_denoise_sources():
    """List approved WAV files available for denoise processing."""
    items = []
    if os.path.exists(APPROVED_DIR):
        for root, dirs, files in os.walk(APPROVED_DIR):
            for f in sorted(files):
                if not f.lower().endswith(".wav"):
                    continue
                full = os.path.join(root, f)
                size_mb = os.path.getsize(full) / 1024 / 1024
                video_name = os.path.basename(os.path.dirname(full))
                items.append({
                    "name": f,
                    "path": full,
                    "video": video_name,
                    "size_mb": round(size_mb, 1),
                })
    return {"sources": items, "count": len(items)}


# ============================================================
# Denoise — Unreviewed clips
# ============================================================

def _is_denoised_wav(filename: str) -> bool:
    """Check if a file is a denoised WAV output (contains _enhanced or _norm, ends with .wav)."""
    if not filename.endswith(".wav"):
        return False
    return "_enhanced" in filename or "_norm" in filename


def _is_media_file(filename: str) -> bool:
    """Check if a file is a reviewable media file (WAV, MP4, MKV, etc.)."""
    return _is_denoised_wav(filename) or filename.lower().endswith(('.mp4', '.mkv', '.mp3', '.aac', '.flac', '.webm'))


@app.get("/api/denoise/unreviewed-dirs")
def list_unreviewed_denoise_dirs():
    """List clip directories with unreviewed (pending) clips."""
    dirs = []
    for entry in sorted(os.listdir(CLIPS_DIR)):
        full = os.path.join(CLIPS_DIR, entry)
        if not os.path.isdir(full):
            continue
        # Load review state
        state = _get_review_state(full)
        approved = set(state.get("approved", []))
        skipped = set(state.get("skipped", []))
        mp4_files = [f for f in os.listdir(full) if f.endswith(".mp4")]
        pending = [f for f in mp4_files if f not in approved and f not in skipped]
        # Check how many have been denoised already
        cleaned_dir = os.path.join(CLEANED_UNREVIEWED_DIR, entry)
        denoised = 0
        if os.path.isdir(cleaned_dir):
            denoised = len([f for f in os.listdir(cleaned_dir) if _is_denoised_wav(f)])
        if pending:
            dirs.append({
                "name": entry,
                "path": full,
                "total": len(mp4_files),
                "pending": len(pending),
                "denoised": denoised,
                "denoised_all": denoised > 0 and denoised >= len(pending),
            })
    return {"dirs": dirs}


@app.get("/api/denoise/unreviewed-clips")
def list_unreviewed_clips(
    clip_dir: str = Query(..., description="Path to clip directory"),
):
    """List unreviewed (pending) clips in a directory."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    state = _get_review_state(clip_dir)
    approved = set(state.get("approved", []))
    skipped = set(state.get("skipped", []))

    clips = []
    for f in sorted(os.listdir(clip_dir)):
        if not f.endswith(".mp4"):
            continue
        if f in approved or f in skipped:
            continue
        full = os.path.join(clip_dir, f)
        clips.append({
            "name": f,
            "path": full,
            "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
        })
    return {"clips": clips, "count": len(clips), "clip_dir": clip_dir}


@app.post("/api/denoise/unreviewed-batch")
def denoise_unreviewed_batch(payload: dict = Body(...)):
    """Batch denoise unreviewed clips. Saves to CLEANED_UNREVIEWED_DIR.

    Body: {video_name: str, paths: [str, ...], steps: [str, ...]?}
    """
    video_name = payload.get("video_name", "")
    paths = payload.get("paths", [])
    steps = payload.get("steps")

    valid_paths = [p for p in paths if os.path.exists(p) and p.lower().endswith(".mp4")]
    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid MP4 files found")

    job_id = uuid.uuid4().hex[:12]
    files = [DenoiseFileItem(
        name=os.path.basename(p),
        input_path=p,
    ) for p in valid_paths]

    job = DenoiseJob(job_id=job_id, video_name=video_name, files=files)
    with _denoise_lock:
        _denoise_jobs[job_id] = job

    def _run():
        import tempfile
        from denoise_audio import run_full_denoise

        job.status = "running"
        total = len(files)
        completed_count = 0

        for f in job.files:
            f.status = "running"
            job.current_file = f.name
            job.progress = (completed_count / total) * 100

            video_dir = video_name or os.path.basename(os.path.dirname(f.input_path))
            output_dir = os.path.join(CLEANED_UNREVIEWED_DIR, video_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Convert MP4 to temp WAV (denoise pipeline expects WAV input)
            wav_tmp = os.path.join(tempfile.gettempdir(), "urev_denoise_" + uuid.uuid4().hex[:8] + ".wav")
            try:
                conv = subprocess.run(
                    [FFMPEG, "-y", "-i", f.input_path, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "48000", "-ac", "1", wav_tmp],
                    capture_output=True, timeout=60,
                )
                if conv.returncode != 0 or not os.path.exists(wav_tmp) or os.path.getsize(wav_tmp) == 0:
                    f.status = "error"
                    f.steps.append({"step": "convert", "status": "error", "message": "音频提取失败"})
                    completed_count += 1
                    continue
            except Exception as e:
                f.status = "error"
                f.steps.append({"step": "convert", "status": "error", "message": str(e)[:100]})
                completed_count += 1
                continue

            def on_step(step_key, status, message):
                f.steps.append({"step": step_key, "status": status, "message": message})

            result = run_full_denoise(wav_tmp, output_dir, on_step=on_step, steps=steps)

            # Clean up temp WAV
            try:
                os.remove(wav_tmp)
            except Exception:
                pass

            if result["success"]:
                f.status = "completed"
                f.output_path = result["output_path"]
            elif result.get("discard_reason"):
                f.status = "discarded"
            else:
                f.status = "error"

            completed_count += 1
            job.progress = (completed_count / total) * 100

        job.current_file = ""
        job.status = "completed"
        job.progress = 100

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "ok", "job_id": job_id, "file_count": len(valid_paths)}


@app.get("/api/denoise/unreviewed-results")
def list_unreviewed_denoise_results(
    video_name: str = Query("", description="Filter by video name"),
):
    """List denoised unreviewed WAV files."""
    items = []
    if os.path.exists(CLEANED_UNREVIEWED_DIR):
        for root, dirs, files in os.walk(CLEANED_UNREVIEWED_DIR):
            for f in sorted(files):
                if not _is_denoised_wav(f):
                    continue
                full = os.path.join(root, f)
                vname = os.path.basename(os.path.dirname(full))
                if video_name and vname != video_name:
                    continue
                items.append({
                    "name": f,
                    "path": full.replace("\\", "/"),
                    "video": vname,
                    "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                    "duration_s": 0,
                    "codec": "",
                    "bitrate": "",
                    "sample_rate": "",
                })
    return {"results": items, "count": len(items)}


# ============================================================
# Denoise Review endpoints
# ============================================================

def _find_denoised_dirs(bases: list[str] | None = None) -> list[str]:
    """Find all directories with media files under the given base directories."""
    if bases is None:
        bases = [CLEANED_DIR, CLEANED_UNREVIEWED_DIR]
    dirs = []
    seen = set()
    for base in bases:
        if not os.path.exists(base):
            continue
        for root, subdirs, files in os.walk(base):
            if any(_is_media_file(f) for f in files):
                if root not in seen:
                    dirs.append(root)
                    seen.add(root)
    return dirs


def _find_denoise_base_dirs() -> list[dict]:
    """Find data/ subdirectories that contain reviewable media (as potential base folders for denoise review)."""
    bases = []
    if not os.path.exists(DATA_DIR):
        return bases
    for name in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, name)
        if not os.path.isdir(p) or name.startswith("."):
            continue
        has_media = False
        for f in os.listdir(p):
            fp = os.path.join(p, f)
            if os.path.isfile(fp) and (_is_denoised_wav(f) or f.lower().endswith(('.mp4', '.mkv', '.mp3', '.aac', '.flac'))):
                has_media = True
                break
            if os.path.isdir(fp) and not f.startswith("."):
                try:
                    if any(_is_denoised_wav(sf) or sf.lower().endswith(('.mp4', '.mkv', '.wav', '.mp3', '.aac', '.flac')) for sf in os.listdir(fp)):
                        has_media = True
                        break
                except OSError:
                    pass
        if has_media:
            bases.append({"name": name, "path": p})
    # Ensure these key directories are always present
    existing = {b["path"] for b in bases}
    for d in [CLEANED_UNREVIEWED_DIR, CLEANED_DIR, PIPELINE_VIDEO_DIR]:
        if d not in existing and os.path.exists(d):
            bases.append({"name": os.path.basename(d), "path": d})
    return bases


@app.get("/api/denoise-review/base-dirs")
def list_denoise_review_base_dirs():
    """List available data/ subdirectories that can serve as base folders for denoise review."""
    return {"bases": _find_denoise_base_dirs()}


def _get_denoise_review_state(clip_dir: str) -> dict:
    """Load denoise review state from a JSON file in the directory."""
    state_path = os.path.join(clip_dir, ".denoise_review_state.json")
    if os.path.exists(state_path):
        import json
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_denoise_review_state(clip_dir: str, state: dict):
    """Save denoise review state to a JSON file in the directory."""
    state_path = os.path.join(clip_dir, ".denoise_review_state.json")
    import json
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


@app.get("/api/denoise-review/dirs")
def list_denoise_review_dirs(
    base: str = Query("", description="Base directory to scan. Defaults to cleaned_unreviewed + cleaned."),
):
    """List all directories with denoised audio available for review."""
    if base and os.path.isdir(base):
        bases = [base]
    else:
        bases = [CLEANED_UNREVIEWED_DIR, CLEANED_DIR]
    dirs = _find_denoised_dirs(bases)
    result = []
    for d in dirs:
        if d.startswith(CLEANED_UNREVIEWED_DIR):
            source = "未审核降噪"
            rel = os.path.relpath(d, CLEANED_UNREVIEWED_DIR)
        elif d.startswith(CLEANED_DIR):
            source = "已审核降噪"
            rel = os.path.relpath(d, CLEANED_DIR)
        else:
            # Generic: show path relative to the base dir
            for b in bases:
                if d.startswith(b):
                    source = os.path.basename(b)
                    rel = os.path.relpath(d, b) if d != b else "."
                    break
            else:
                source = "未知"
                rel = os.path.basename(d)
        name = rel.replace("\\", "/") if rel != "." else os.path.basename(d)
        result.append({"name": name, "path": d.replace("\\", "/"), "source": source})
    return {"dirs": result, "base": base if base else "<default>"}


@app.get("/api/denoise-review/clips")
def list_denoise_review_clips(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
):
    """List media files available for denoise review, with review status."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    state = _get_denoise_review_state(clip_dir)
    approved = state.get("approved", [])
    skipped = state.get("skipped", [])

    clips = []
    for f in sorted(os.listdir(clip_dir)):
        if not _is_media_file(f):
            continue
        if f.startswith("."):
            continue
        full = os.path.join(clip_dir, f)
        if not os.path.isfile(full) or os.path.getsize(full) == 0:
            continue
        status = "approved" if f in approved else ("skipped" if f in skipped else "pending")
        clips.append({
            "name": f,
            "path": full.replace("\\", "/"),
            "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
            "duration_s": 0,
            "status": status,
        })

    pending_idx = next((i for i, c in enumerate(clips) if c["status"] == "pending"), -1)

    return {
        "clip_dir": clip_dir.replace("\\", "/"),
        "clips": clips,
        "pending_index": pending_idx,
        "stats": {
            "total": len(clips),
            "approved": len(approved),
            "skipped": len(skipped),
            "pending": len(clips) - len(approved) - len(skipped),
        },
    }


@app.post("/api/denoise-review/approve")
def denoise_review_approve(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
    clip_name: str = Query(..., description="Audio filename to approve"),
):
    """Approve a denoised audio file: copy to DENOISED_APPROVED_DIR."""
    src = os.path.join(clip_dir, clip_name)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="File not found")

    # Create approved subfolder mirroring the source relative path
    if clip_dir.startswith(CLEANED_UNREVIEWED_DIR):
        dir_name = os.path.relpath(clip_dir, CLEANED_UNREVIEWED_DIR).replace("\\", "/")
    elif clip_dir.startswith(CLEANED_DIR):
        dir_name = os.path.relpath(clip_dir, CLEANED_DIR).replace("\\", "/")
    else:
        dir_name = os.path.basename(clip_dir)
    dst_dir = os.path.join(DENOISED_APPROVED_DIR, dir_name)
    os.makedirs(dst_dir, exist_ok=True)

    dst = os.path.join(dst_dir, clip_name)
    if not os.path.exists(dst):
        import shutil
        shutil.copy2(src, dst)

    state = _get_denoise_review_state(clip_dir)
    state.setdefault("approved", []).append(clip_name)
    if clip_name in state.get("skipped", []):
        state["skipped"].remove(clip_name)
    _save_denoise_review_state(clip_dir, state)

    return {"status": "ok", "action": "approved", "saved_to": dst}


@app.post("/api/denoise-review/skip")
def denoise_review_skip(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
    clip_name: str = Query(..., description="Audio filename to skip"),
):
    """Skip a denoised audio file."""
    src = os.path.join(clip_dir, clip_name)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="File not found")

    state = _get_denoise_review_state(clip_dir)
    state.setdefault("skipped", []).append(clip_name)
    if clip_name in state.get("approved", []):
        state["approved"].remove(clip_name)
    _save_denoise_review_state(clip_dir, state)

    return {"status": "ok", "action": "skipped"}


@app.post("/api/denoise-review/classify-emotion")
def denoise_review_classify_emotion(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
    clip_name: str = Query(..., description="Audio filename to classify"),
    emotion: str = Query(..., description="Emotion category name"),
):
    """Copy the current denoised audio to the corresponding emotion-denoise folder.
    If the audio already exists in another emotion folder, remove it from there first."""
    src = os.path.join(clip_dir, clip_name)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="File not found")

    import shutil

    # Remove clip from any other emotion folder (single-choice)
    for e in EMOTIONS:
        if e == emotion:
            continue
        existing = os.path.join(EMOTION_DENOISE_DIR, e, clip_name)
        if os.path.exists(existing):
            os.remove(existing)

    dst_dir = os.path.join(EMOTION_DENOISE_DIR, emotion)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, clip_name)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

    return {"status": "ok", "emotion": emotion, "copied_to": dst}


@app.get("/api/denoise-review/clip-emotions")
def get_denoise_clip_emotions(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
    clip_name: str = Query(..., description="Audio filename"),
):
    """Check which emotion-denoise folders already contain this audio."""
    return {"emotions": _get_clip_emotions(clip_name, EMOTION_DENOISE_DIR)}


@app.post("/api/denoise-review/remove-short")
def denoise_review_remove_short(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
    min_duration: float = Query(2.0, description="Remove audio with duration <= this value (seconds)"),
):
    """Remove all denoised WAV files with duration <= min_duration seconds. Runs in background."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from split_video import get_video_duration

        deleted = []
        wav_files = [f for f in sorted(os.listdir(clip_dir)) if _is_denoised_wav(f)]
        total = len(wav_files)

        job = pipeline.create_job(job_id, title="降噪审核-清除短音频: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "scanning"
        job.progress = 0

        for i, f in enumerate(wav_files):
            full = os.path.join(clip_dir, f)
            try:
                dur = _get_audio_info(full)["duration_s"]
                if dur > 0 and dur <= min_duration:
                    os.remove(full)
                    deleted.append(f)
            except OSError:
                pass
            job.progress = ((i + 1) / total) * 100

        state = _get_denoise_review_state(clip_dir)
        for name in deleted:
            if name in state.get("approved", []):
                state["approved"].remove(name)
            if name in state.get("skipped", []):
                state["skipped"].remove(name)
        _save_denoise_review_state(clip_dir, state)

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="remove_short",
            status=StepStatus.COMPLETED,
            message=f"删除了 {len(deleted)} 个时长<={min_duration}s 的降噪音频",
            data={"deleted": deleted},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.post("/api/denoise-review/remove-reverb")
def denoise_review_remove_reverb(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
):
    """Detect and remove denoised audio files with reverb. Runs in background."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from audio_pipeline import analyze_clip_bgm

        wav_files = sorted([f for f in os.listdir(clip_dir) if f.endswith("_norm.wav")])
        total = len(wav_files)
        removed = []

        job = pipeline.create_job(job_id, title="降噪审核-去除混响: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        for i, f in enumerate(wav_files):
            full = os.path.join(clip_dir, f)
            try:
                analysis = analyze_clip_bgm(full)
                if analysis["has_reverb"]:
                    os.remove(full)
                    removed.append(f)
            except Exception as e:
                print(f"[denoise-review-remove-reverb] Error on {f}: {e}")

            job.progress = ((i + 1) / total) * 100

        state = _get_denoise_review_state(clip_dir)
        for name in removed:
            if name in state.get("approved", []):
                state["approved"].remove(name)
            if name in state.get("skipped", []):
                state["skipped"].remove(name)
        _save_denoise_review_state(clip_dir, state)

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="remove_reverb",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {total} 个降噪音频中删除了 {len(removed)} 个含混响的片段",
            data={"deleted": removed},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.post("/api/denoise-review/remove-male")
def denoise_review_remove_male(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
):
    """Detect and remove denoised audio files with male voices. Runs in background."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from audio_pipeline import analyze_clip_gender

        wav_files = sorted([f for f in os.listdir(clip_dir) if f.endswith("_norm.wav")])
        total = len(wav_files)
        removed = []

        job = pipeline.create_job(job_id, title="降噪审核-去除男声: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        for i, f in enumerate(wav_files):
            full = os.path.join(clip_dir, f)
            try:
                analysis = analyze_clip_gender(audio_path=full)
                if analysis.get("is_male"):
                    os.remove(full)
                    removed.append(f)
            except Exception as e:
                print(f"[denoise-review-remove-male] Error on {f}: {e}")

            job.progress = ((i + 1) / total) * 100

        state = _get_denoise_review_state(clip_dir)
        for name in removed:
            if name in state.get("approved", []):
                state["approved"].remove(name)
            if name in state.get("skipped", []):
                state["skipped"].remove(name)
        _save_denoise_review_state(clip_dir, state)

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="remove_male",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {total} 个降噪音频中删除了 {len(removed)} 个男声片段",
            data={"deleted": removed},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.post("/api/denoise-review/remove-multi-voice")
def denoise_review_remove_multi_voice(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
):
    """Detect and remove denoised audio files with multiple human voices. Runs in background."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from audio_pipeline import analyze_clip_multi_voice

        wav_files = sorted([f for f in os.listdir(clip_dir) if f.endswith("_norm.wav")])
        total = len(wav_files)
        removed = []

        job = pipeline.create_job(job_id, title="降噪审核-去除多人声: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        for i, f in enumerate(wav_files):
            full = os.path.join(clip_dir, f)
            try:
                analysis = analyze_clip_multi_voice(audio_path=full)
                if analysis["has_multi_voice"]:
                    os.remove(full)
                    removed.append(f)
            except Exception as e:
                print(f"[denoise-review-remove-multi-voice] Error on {f}: {e}")

            job.progress = ((i + 1) / total) * 100

        state = _get_denoise_review_state(clip_dir)
        for name in removed:
            if name in state.get("approved", []):
                state["approved"].remove(name)
            if name in state.get("skipped", []):
                state["skipped"].remove(name)
        _save_denoise_review_state(clip_dir, state)

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="remove_multi_voice",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {total} 个降噪音频中删除了 {len(removed)} 个多人声片段",
            data={"deleted": removed},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.post("/api/denoise-review/remove-low-volume")
def denoise_review_remove_low_volume(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
):
    """Detect and remove denoised audio files with long low-volume segments. Runs in background."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from audio_pipeline import detect_low_volume_segments

        wav_files = sorted([f for f in os.listdir(clip_dir) if f.endswith("_norm.wav")])
        total = len(wav_files)
        removed = []

        job = pipeline.create_job(job_id, title="降噪审核-去除低音量: " + os.path.basename(clip_dir))
        job.status = "running"
        job.current_step = "analyzing"
        job.progress = 0

        for i, f in enumerate(wav_files):
            full = os.path.join(clip_dir, f)
            try:
                analysis = detect_low_volume_segments(audio_path=full)
                if analysis["has_low_volume"]:
                    os.remove(full)
                    removed.append(f)
            except Exception as e:
                print(f"[denoise-review-remove-low-volume] Error on {f}: {e}")

            job.progress = ((i + 1) / total) * 100

        state = _get_denoise_review_state(clip_dir)
        for name in removed:
            if name in state.get("approved", []):
                state["approved"].remove(name)
            if name in state.get("skipped", []):
                state["skipped"].remove(name)
        _save_denoise_review_state(clip_dir, state)

        job.status = "completed"
        job.progress = 100
        job.current_step = "done"
        from pipeline import StepResult, StepStatus
        job.steps = [StepResult(
            step="remove_low_volume",
            status=StepStatus.COMPLETED,
            message=f"检测完成: {total} 个降噪音频中删除了 {len(removed)} 个低音量片段",
            data={"deleted": removed},
        )]
        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "job_id": job_id}


@app.get("/api/denoise-review/stream")
def denoise_review_stream(path: str = Query(..., description="Full path to media file")):
    """Stream a media file for playback in the browser."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    # Auto-detect media type from extension
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".aac": "audio/aac", ".flac": "audio/flac",
        ".mp4": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm",
    }
    return FileResponse(path, media_type=mime_map.get(ext, "application/octet-stream"))


def _display_available():
    """True if the server can show GUI dialogs (needs desktop environment)."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@app.get("/api/list-dirs")
def list_dirs(path: str = Query("", description="Directory path to list"),
              root: str = Query("", description="Root path for navigation")):
    """List subdirectories for browser-based folder picker."""
    import platform as _platform
    if not path or not os.path.isdir(path):
        path = os.path.expanduser("~") if _platform.system() != "Windows" else "C:\\"
    roots = []
    if _platform.system() == "Windows":
        import string as _string
        for letter in _string.ascii_uppercase:
            p = f"{letter}:\\"
            if os.path.isdir(p):
                roots.append({"name": f"{letter}:", "path": p})
    else:
        roots = [{"name": "/", "path": "/"}]
    try:
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full) and not name.startswith("."):
                entries.append({"name": name, "path": full, "type": "dir"})
        parent = os.path.dirname(path) if path not in ["/", "C:\\"] else None
        return {
            "path": path,
            "parent": parent if parent and os.path.isdir(parent) else None,
            "entries": entries,
            "roots": roots,
        }
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")


@app.post("/api/open-folder")
def open_folder_in_explorer(payload: dict = Body(...)):
    """Open a folder in the system file explorer.

    Accepts either a direct path, or a clip_dir + type to deduce the folder.
    Types: "approved" (review tab), "denoised-approved" (denoise review tab).
    """
    path = payload.get("path", "")
    if not path:
        clip_dir = payload.get("clip_dir", "")
        folder_type = payload.get("type", "approved")
        if clip_dir and os.path.isdir(clip_dir):
            dir_name = os.path.basename(clip_dir)
            if folder_type == "denoised-approved":
                if clip_dir.startswith(CLEANED_UNREVIEWED_DIR):
                    rel = os.path.relpath(clip_dir, CLEANED_UNREVIEWED_DIR)
                elif clip_dir.startswith(CLEANED_DIR):
                    rel = os.path.relpath(clip_dir, CLEANED_DIR)
                else:
                    rel = dir_name
                path = os.path.join(DENOISED_APPROVED_DIR, rel.replace("\\", "/"))
            else:
                path = os.path.join(APPROVED_DIR, dir_name)
            os.makedirs(path, exist_ok=True)
    if not path or not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="文件夹不存在")
    try:
        import platform
        if platform.system() == "Windows":
            os.startfile(path)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings")
def get_settings():
    """Return current path settings and default step configs."""
    import config
    paths = {}
    for k in config._PATH_VARS:
        paths[k] = getattr(config, k, '')
    steps = getattr(config, 'DENOISE_DEFAULT_STEPS', None)
    pv_steps = getattr(config, 'PV_DEFAULT_STEPS', None)
    config_vals = {k: getattr(config, k, config._CONFIG_KEYS[k]) for k in config._CONFIG_KEYS}
    return {"paths": paths, "denoise_default_steps": steps, "pv_default_steps": pv_steps, "config": config_vals}


@app.post("/api/settings")
def save_settings(payload: dict = Body(...)):
    """Save path settings and/or default step configs."""
    from config import save_settings
    paths = payload.get("paths", {})
    steps = payload.get("denoise_default_steps", None)
    pv_steps = payload.get("pv_default_steps", None)
    config_vals = payload.get("config", None)
    save_settings(paths, steps, pv_steps, config_vals)
    return {"status": "ok"}


# ============================================================
# Hotword Configurations API
# ============================================================

from config import HOTWORDS_DIR


def _list_hotword_configs():
    """List hotword config files. Returns [{name, hotwords}, ...]."""
    configs = []
    os.makedirs(HOTWORDS_DIR, exist_ok=True)
    for fname in sorted(os.listdir(HOTWORDS_DIR)):
        if fname.endswith('.txt'):
            name = fname[:-4]
            path = os.path.join(HOTWORDS_DIR, fname)
            with open(path, 'r', encoding='utf-8') as f:
                configs.append({"name": name, "hotwords": f.read().strip()})
    return configs


@app.get("/api/hotwords/configs")
def list_hotwords_configs():
    """List all hotword configurations."""
    return {"configs": _list_hotword_configs()}


@app.post("/api/hotwords/configs")
def save_hotwords_config(payload: dict = Body(...)):
    """Create or update a hotword configuration. Uses name as filename."""
    name = (payload.get("name") or "").strip()
    hotwords = (payload.get("hotwords") or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="\u540d\u79f0\u4e0d\u80fd\u4e3a\u7a7a")
    if not hotwords:
        raise HTTPException(status_code=400, detail="\u70ed\u8bcd\u4e0d\u80fd\u4e3a\u7a7a")

    os.makedirs(HOTWORDS_DIR, exist_ok=True)
    path = os.path.join(HOTWORDS_DIR, name + ".txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(hotwords)
    return {"status": "ok"}


@app.delete("/api/hotwords/configs/{name}")
def delete_hotwords_config(name: str):
    """Delete a hotword configuration file by name."""
    path = os.path.join(HOTWORDS_DIR, name + ".txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="\u914d\u7f6e\u4e0d\u5b58\u5728")
    os.remove(path)
    return {"status": "ok"}

@app.get("/api/browse-folder")
def browse_folder_generic(title: str = "选择文件夹", initialdir: str = ""):
    """Open a native folder picker and return the selected path."""
    import subprocess, tempfile
    if not _display_available():
        return {"path": "", "error": "Server has no display — use manual path input instead."}

    if not initialdir or not os.path.isdir(initialdir):
        initialdir = DATA_DIR if os.path.isdir(DATA_DIR) else os.path.expanduser("~")

    result_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    result_path = result_file.name
    result_file.close()

    picker_code = f'''
import tkinter as tk
from tkinter import filedialog
import sys, os, ctypes
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.lift()
root.focus_force()
root.update()
hwnd = root.winfo_id() if root.winfo_exists() else 0
if hwnd:
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.BringWindowToTop(hwnd)
try:
    path = filedialog.askdirectory(title=r"{title}", initialdir=r"{initialdir}")
    if path:
        with open(r"{result_path}", "w", encoding="utf-8") as f:
            f.write(path)
except Exception:
    pass
try:
    root.destroy()
except Exception:
    pass
'''

    picker_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    picker_file.write(picker_code)
    picker_path = picker_file.name
    picker_file.close()

    try:
        subprocess.run([sys.executable, picker_path], timeout=300)
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                path = f.read().strip()
            return {"path": path}
        return {"path": ""}
    except subprocess.TimeoutExpired:
        return {"path": "", "error": "操作超时"}
    except Exception as e:
        return {"path": "", "error": str(e)}
    finally:
        for f in (picker_path, result_path):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


@app.get("/api/results/video/{video_name}/cleaned")
def get_cleaned_for_video(video_name: str):
    """Get cleaned/denoised audio files for a specific video."""
    items = []
    # Check per-video subdirectory first
    video_cleaned_dir = os.path.join(CLEANED_DIR, video_name)
    search_dirs = [video_cleaned_dir, CLEANED_DIR]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith("_norm.wav"):
                full = os.path.join(d, f)
                info = _get_audio_info(full)
                items.append({
                    "name": f,
                    "path": full,
                    "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                    "duration_s": info["duration_s"],
                    "codec": info["codec"],
                    "bitrate": info["bitrate"],
                    "sample_rate": info["sample_rate"],
                })
    return {"cleaned": items, "count": len(items)}


# ============================================================
# MKV Subtitle Extraction endpoints
# ============================================================

@app.get("/api/extract/mkv-files")
def list_mkv_files():
    """List all MKV files available for extraction from downloads directory."""
    mkv_files = []
    if os.path.exists(DOWNLOAD_DIR):
        for root, dirs, filenames in os.walk(DOWNLOAD_DIR):
            for f in filenames:
                if f.lower().endswith(".mkv"):
                    full = os.path.join(root, f)
                    size_mb = os.path.getsize(full) / 1024 / 1024
                    mtime = os.path.getmtime(full)
                    from datetime import datetime
                    mkv_files.append({
                        "name": f,
                        "path": full,
                        "size_mb": round(size_mb, 1),
                        "mtime": mtime,
                        "time_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                    })
    mkv_files.sort(key=lambda x: x["mtime"], reverse=True)
    return {"files": mkv_files}


@app.get("/api/extract/tracks")
def get_extract_tracks(path: str = Query(..., description="Full path to MKV file")):
    """Get all tracks from an MKV file for extraction selection."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        from extract_subs import get_mkv_tracks
        tracks = get_mkv_tracks(path)
        return {"tracks": tracks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract/run")
def run_extraction(
    path: str = Query(..., description="Full path to MKV file"),
    track_ids: str = Query("", description="Comma-separated track IDs to extract. Empty = all subtitle tracks"),
):
    """Extract subtitle tracks from an MKV file to the subtitles directory.

    If track_ids is empty, extracts all subtitle tracks.
    Returns the list of extracted file paths.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    import threading

    job_id = uuid.uuid4().hex[:12]

    def _run():
        from extract_subs import get_mkv_tracks, extract_subtitle_track
        import time as _time

        # Create a job record for tracking
        job = pipeline.create_job(job_id, title=os.path.basename(path))
        job.mkv_path = path
        job.status = "running"
        job.progress = 10
        job.current_step = "inspecting"

        from pipeline import StepResult, StepStatus

        try:
            all_tracks = get_mkv_tracks(path)

            if track_ids:
                requested_ids = set(int(t.strip()) for t in track_ids.split(",") if t.strip())
                sub_tracks = [t for t in all_tracks if t.get("id") in requested_ids and t.get("type") == "subtitles"]
            else:
                # Extract all subtitle tracks
                sub_tracks = [t for t in all_tracks if t.get("type") == "subtitles"]

            if not sub_tracks:
                job.status = "completed"
                job.progress = 100
                job.current_step = "done"
                job.steps = [StepResult(
                    step="extract",
                    status=StepStatus.COMPLETED,
                    message="没有找到可提取的字幕轨道",
                )]
                pipeline._save_jobs()
                return

            extracted = []
            total = len(sub_tracks)
            for i, track in enumerate(sub_tracks):
                track_id = track.get("id", 0)
                codec = track.get("codec", "")
                codec_id = track.get("codec_id", "")
                if "ass" in codec.lower() + codec_id.lower() or "substation" in codec.lower() + codec_id.lower():
                    ext = ".ass"
                elif "subrip" in (codec.lower() + codec_id.lower()):
                    ext = ".srt"
                else:
                    ext = ".ass"

                base = os.path.splitext(os.path.basename(path))[0]
                lang = track.get("language", "unknown")
                output = os.path.join(SUBTITLE_DIR, f"{base}_track{track_id}_{lang}{ext}")

                job.current_step = f"extracting track {track_id}"
                job.progress = 10 + (i / total) * 80

                result = extract_subtitle_track(path, track_id, output)
                if result:
                    extracted.append(result)

            job.progress = 100
            job.status = "completed"
            job.current_step = "done"
            msg = f"提取完成: {len(extracted)}/{total} 个字幕轨道"
            if not extracted:
                msg = "提取失败: 没有成功提取任何字幕轨道"
            job.steps = [StepResult(
                step="extract",
                status=StepStatus.COMPLETED,
                message=msg,
            )]

        except Exception as e:
            job.status = "failed"
            job.progress = 0
            job.current_step = "error"
            job.steps = [StepResult(
                step="extract",
                status=StepStatus.FAILED,
                message=str(e),
            )]

        pipeline._save_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"job_id": job_id, "status": "started"}


@app.get("/api/extract/subtitle-files")
def list_extracted_subtitles():
    """List all extracted subtitle files."""
    files = []
    if os.path.exists(SUBTITLE_DIR):
        for f in sorted(os.listdir(SUBTITLE_DIR)):
            ext = os.path.splitext(f)[1].lower()
            if ext in {".srt", ".ass", ".ssa", ".vtt"}:
                full = os.path.join(SUBTITLE_DIR, f)
                files.append({
                    "name": f,
                    "path": full,
                    "size_kb": round(os.path.getsize(full) / 1024, 1),
                })
    return {"subtitles": files}


# ============================================================
# Audio Stitching endpoints
# ============================================================

def _find_subtitle_for_video(video_name: str) -> str:
    """Find the subtitle file in SUBTITLE_DIR that best matches a video name."""
    if not os.path.exists(SUBTITLE_DIR):
        return ""
    for f in sorted(os.listdir(SUBTITLE_DIR)):
        if f.endswith(('.ass', '.srt', '.ssa')) and video_name in f:
            return os.path.join(SUBTITLE_DIR, f)
    return ""


def _extract_index_from_clip_name(clip_name: str) -> int:
    """Extract subtitle index from clip filename like 'video_S003_text.wav'."""
    import re
    m = re.search(r'_S(\d+)_', clip_name)
    if m:
        return int(m.group(1)) - 1  # 1-based in filename, 0-based in entries
    return -1


@app.get("/api/stitch/videos")
def list_stitchable_videos():
    """List videos that have denoised WAV files available for stitching."""
    videos = []
    if os.path.exists(CLEANED_DIR):
        for entry in sorted(os.listdir(CLEANED_DIR)):
            full = os.path.join(CLEANED_DIR, entry)
            if os.path.isdir(full):
                wavs = [f for f in os.listdir(full) if f.lower().endswith('.wav')]
                if wavs:
                    videos.append({
                        "name": entry,
                        "path": full,
                        "wav_count": len(wavs),
                    })
    return {"videos": videos}


@app.get("/api/stitch/clips")
def list_stitch_clips(video: str = Query(..., description="Video name")):
    """List denoised WAV clips for a video with associated subtitle text."""
    cleaned_dir = os.path.join(CLEANED_DIR, video)
    if not os.path.isdir(cleaned_dir):
        raise HTTPException(status_code=404, detail="Video not found")

    # Parse subtitle file to get full text per entry
    sub_texts = {}  # index -> text
    sub_path = _find_subtitle_for_video(video)
    if sub_path:
        try:
            from split_video import parse_subtitle_file
            sf = parse_subtitle_file(sub_path)
            if sf:
                for entry in sf.entries:
                    sub_texts[entry.index] = entry.text
        except Exception:
            pass

    clips = []
    for f in sorted(os.listdir(cleaned_dir)):
        if not f.lower().endswith('.wav'):
            continue
        full = os.path.join(cleaned_dir, f)
        idx = _extract_index_from_clip_name(f)
        text = sub_texts.get(idx, '')
        size_mb = round(os.path.getsize(full) / 1024 / 1024, 1)

        # Get duration via ffprobe
        dur = 0
        try:
            import subprocess as sp
            r = sp.run(
                [FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", full],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
            )
            if r.returncode == 0:
                dur = round(float(r.stdout.strip()), 1)
        except Exception:
            pass

        clips.append({
            "name": f,
            "path": full,
            "index": idx,
            "size_mb": size_mb,
            "duration_s": dur,
            "subtitle_text": text,
        })

    return {"clips": clips, "video": video, "subtitle_path": sub_path or ""}


@app.post("/api/stitch/concat")
def concat_clips(payload: dict = Body(...)):
    """Concatenate selected WAV clips into one audio file, merge subtitles.

    Body: {video: str, paths: [str, ...]}

    Uses ffmpeg concat demuxer for lossless WAV concatenation.
    Returns the output path and merged subtitle text.
    """
    video = payload.get("video", "unknown")
    paths = payload.get("paths", [])

    if not paths:
        raise HTTPException(status_code=400, detail="No clips selected")
    if len(paths) == 1:
        # Single clip — just return it directly with its subtitle
        idx = _extract_index_from_clip_name(os.path.basename(paths[0]))
        sub_texts = {}
        sub_path = _find_subtitle_for_video(video)
        if sub_path:
            try:
                from split_video import parse_subtitle_file
                sf = parse_subtitle_file(sub_path)
                if sf:
                    for e in sf.entries:
                        sub_texts[e.index] = e.text
            except Exception:
                pass
        merged_text = sub_texts.get(idx, os.path.splitext(os.path.basename(paths[0]))[0])
        return {
            "status": "ok",
            "output_path": paths[0],
            "output_name": os.path.basename(paths[0]),
            "clip_count": 1,
            "merged_subtitle": merged_text,
        }

    # Validate all files exist
    for p in paths:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"File not found: {p}")

    # Get subtitle text for each clip
    sub_texts = {}
    sub_path = _find_subtitle_for_video(video)
    if sub_path:
        try:
            from split_video import parse_subtitle_file
            sf = parse_subtitle_file(sub_path)
            if sf:
                for e in sf.entries:
                    sub_texts[e.index] = e.text
        except Exception:
            pass

    # Build merged subtitle: each clip's text on its own line
    merged_lines = []
    for p in paths:
        idx = _extract_index_from_clip_name(os.path.basename(p))
        text = sub_texts.get(idx, '')
        if not text:
            text = os.path.splitext(os.path.basename(p))[0]
            # Try to extract text from filename: video_S003_text.wav -> text
            parts = text.split('_', 2)
            if len(parts) >= 3:
                text = parts[2]
        merged_lines.append(text)

    # Write concat file list for ffmpeg
    import time as _time
    ts = str(int(_time.time()))
    safe_video = video.replace('/', '_').replace('\\', '_')
    output_name = f"{safe_video}_stitched_{ts}.wav"
    output_path = os.path.join(STITCHED_DIR, output_name)

    concat_list_path = os.path.join(STITCHED_DIR, f"_concat_{ts}.txt")
    try:
        with open(concat_list_path, "w", encoding="utf-8") as clf:
            for p in paths:
                # ffmpeg concat uses 'file <path>' lines
                safe_p = p.replace("'", "'\\''")
                clf.write(f"file '{safe_p}'\n")

        # Use ffmpeg concat demuxer (lossless for WAV)
        result = subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
             "-c", "copy", output_path],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
        )

        if result.returncode != 0 or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail=f"Concat failed: {result.stderr[:200]}")
    finally:
        # Clean up temp concat list
        if os.path.exists(concat_list_path):
            try:
                os.remove(concat_list_path)
            except OSError:
                pass

    return {
        "status": "ok",
        "output_path": output_path,
        "output_name": output_name,
        "clip_count": len(paths),
        "merged_subtitle": "\n".join(merged_lines),
    }


# ============================================================
# Video Split / Cut endpoints
# ============================================================

@app.get("/api/split/clip-dirs")
def list_split_clip_dirs():
    """List all clip directories under data/clips/ and data/split/ with stats."""
    dirs = []
    seen = set()
    for base in [CLIPS_DIR, SPLIT_DIR]:
        if os.path.exists(base):
            for entry in sorted(os.listdir(base)):
                full = os.path.join(base, entry)
                if os.path.isdir(full) and entry not in seen:
                    seen.add(entry)
                    media_exts = {".mp4", ".mkv", ".wav"}
                    media_count = len([f for f in os.listdir(full)
                        if os.path.splitext(f)[1].lower() in media_exts])
                    dirs.append({
                        "name": entry,
                        "path": full,
                        "clip_count": media_count,
                        "total_size_mb": round(
                            sum(os.path.getsize(os.path.join(full, f))
                                for f in os.listdir(full)
                                if os.path.splitext(f)[1].lower() in media_exts
                                and os.path.isfile(os.path.join(full, f))
                            ) / 1024 / 1024, 1,
                        ),
                        "mtime": os.path.getmtime(full),
                    })
    dirs.sort(key=lambda d: d["mtime"], reverse=True)
    return {"dirs": dirs, "count": len(dirs)}


@app.get("/api/split/clips/{video_name}")
def list_split_clips(video_name: str):
    """List all clips in data/clips/{video_name}/ or data/split/{video_name}/."""
    clips_dir = None
    for base in [CLIPS_DIR, SPLIT_DIR]:
        candidate = os.path.join(base, video_name)
        if os.path.exists(candidate):
            clips_dir = candidate
            break
    if not clips_dir:
        return {"clips": [], "count": 0, "video_name": video_name}

    clips = []
    for f in sorted(os.listdir(clips_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".mp4", ".mkv", ".wav"):
            continue
        full = os.path.join(clips_dir, f)
        clips.append({
            "name": f,
            "path": full,
            "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
        })
    return {"clips": clips, "count": len(clips), "video_name": video_name}


@app.get("/api/split/dir-browse")
def browse_split_directory(path: str = Query(""), scan_files: bool = Query(False)):
    """Browse a directory for sub-folders, video files, and WAV audio files.

    Set scan_files=true to also scan for media files (slower, use only when entering a folder).
    When false, only subdirectories are returned — this is fast.
    """
    if not path:
        path = SPLIT_DIR
    real_path = os.path.realpath(path)
    real_data = os.path.realpath(DATA_DIR)
    if not real_path.startswith(real_data + os.sep) and real_path != real_data:
        raise HTTPException(status_code=403, detail="Access denied: path outside data directory")
    if not os.path.isdir(real_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    parent_path = os.path.dirname(real_path)
    if not parent_path.startswith(real_data + os.sep) and parent_path != real_data:
        parent_path = real_data if real_path != real_data else ""

    entries = sorted(os.scandir(real_path), key=lambda e: e.name)
    subdirs = []
    video_files = []
    audio_files = []
    subtitle_files = []
    video_exts = {".mkv", ".mp4", ".avi", ".mov", ".wmv"}
    audio_exts = {".wav"}
    sub_exts = {".srt", ".ass", ".ssa", ".vtt"}

    for entry in entries:
        if entry.is_dir() and not entry.name.startswith('.'):
            subdirs.append(entry.name)
        elif scan_files and entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in video_exts:
                size = entry.stat().st_size
                video_files.append({
                    "name": entry.name,
                    "ext": ext,
                    "size_mb": round(size / 1024 / 1024, 1),
                    "full_path": os.path.join(real_path, entry.name),
                })
            elif ext in audio_exts:
                size = entry.stat().st_size
                audio_files.append({
                    "name": entry.name,
                    "ext": ext,
                    "size_mb": round(size / 1024 / 1024, 1),
                    "full_path": os.path.join(real_path, entry.name),
                })
            elif ext in sub_exts:
                size = entry.stat().st_size
                subtitle_files.append({
                    "name": entry.name,
                    "ext": ext,
                    "size_kb": round(size / 1024, 1),
                    "full_path": os.path.join(real_path, entry.name),
                })

    return {
        "current_path": real_path,
        "parent_path": parent_path,
        "subdirs": subdirs,
        "video_files": video_files,
        "audio_files": audio_files,
        "subtitle_files": subtitle_files,
        "has_videos": len(video_files) > 0,
        "has_audio": len(audio_files) > 0,
    }


@app.post("/api/split/run")
def run_split(payload: dict = Body(...)):
    """Run a video split job in any of 3 modes.

    Body:
        mode: "subtitle" | "duration" | "size" | "trim"
        video_path: str (required)
        subtitle_path: str (for subtitle mode)
        padding: float (subtitle mode, default 0.1)
        group_count: int (subtitle mode, default 1)
        segment_duration: float (duration mode, seconds)
        target_size_mb: float (size mode, MB)
        start_time: float (trim mode required; duration/size optional)
        end_time: float (trim mode required; duration/size optional)
        hw_accel: str (default "auto")
        output_ext: str (default ".mp4")
        start_time: float (duration/size modes, default 0)
        end_time: float (duration/size modes, default 0 = full)
    """
    mode = payload.get("mode", "subtitle")
    video_path = payload.get("video_path", "")
    hw_accel = payload.get("hw_accel", "auto")

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    job_id = uuid.uuid4().hex[:12]
    title = os.path.splitext(os.path.basename(video_path))[0]

    # Create job synchronously so it's immediately visible in the task list
    job = pipeline.create_job(job_id, title=title)
    job.mkv_path = video_path
    job.status = "running"
    job.current_step = mode
    job.progress = 5

    def run():

        try:
            is_audio = video_path.lower().endswith('.wav')

            if mode == "subtitle":
                subtitle_path = payload.get("subtitle_path", "")
                if not subtitle_path or not os.path.exists(subtitle_path):
                    job.status = "failed"
                    job.current_step = "error"
                    job.progress = 0
                    from pipeline import StepResult, StepStatus
                    job.steps = [StepResult(
                        step="split_video", status=StepStatus.FAILED,
                        message="请选择一个字幕文件",
                    )]
                    pipeline._save_jobs()
                    return

                padding = float(payload.get("padding", 0.1))
                group_count = int(payload.get("group_count", 1))
                deduplicate = bool(payload.get("deduplicate", True))
                output_dir = payload.get("output_dir", "")

                from split_video import split_video_by_subtitle
                job.current_step = "subtitle_split"
                job.progress = 10

                clips = split_video_by_subtitle(
                    video_path, subtitle_path,
                    output_dir=output_dir,
                    hw_accel=hw_accel, padding=padding,
                    deduplicate=deduplicate,
                    on_progress=lambda c, t, txt: setattr(
                        job, 'progress', 10 + (c / max(t, 1)) * 80),
                    cancel_event=job.cancel_event,
                )
                job.clip_paths = clips
                if clips:
                    job.clip_dir = os.path.dirname(clips[0])

            elif mode == "duration":
                segment_duration = float(payload.get("segment_duration", 60))
                start_time = float(payload.get("start_time", 0))
                end_time = float(payload.get("end_time", 0))
                output_ext = payload.get("output_ext", ".mp4")

                from split_video import split_video_by_duration
                job.current_step = "duration_split"
                job.progress = 10

                clips = split_video_by_duration(
                    video_path, segment_duration,
                    hw_accel=hw_accel, output_ext=output_ext,
                    start_offset=start_time, end_time=end_time,
                    on_progress=lambda c, t: setattr(
                        job, 'progress', 10 + (c / max(t, 1)) * 80),
                    cancel_event=job.cancel_event,
                )
                job.clip_paths = clips
                if clips:
                    job.clip_dir = os.path.dirname(clips[0])

            elif mode == "size":
                target_size_mb = float(payload.get("target_size_mb", 100))
                output_ext = payload.get("output_ext", ".mp4")

                from split_video import split_video_by_size
                job.current_step = "size_split"
                job.progress = 10

                clips = split_video_by_size(
                    video_path, target_size_mb,
                    hw_accel=hw_accel, output_ext=output_ext,
                    on_progress=lambda c, t: setattr(
                        job, 'progress', 10 + (c / max(t, 1)) * 80),
                    cancel_event=job.cancel_event,
                )
                job.clip_paths = clips
                if clips:
                    job.clip_dir = os.path.dirname(clips[0])

            elif mode == "trim":
                start_time = float(payload.get("start_time", 0))
                end_time = float(payload.get("end_time", 0))
                if end_time <= start_time:
                    job.status = "failed"
                    job.progress = 0
                    from pipeline import StepResult, StepStatus
                    job.steps = [StepResult(
                        step="split_video", status=StepStatus.FAILED,
                        message="结束时间必须大于开始时间",
                    )]
                    pipeline._save_jobs()
                    return

                from split_video import trim_video
                job.current_step = "trim_video"
                job.progress = 10

                clips = trim_video(
                    video_path, start_time, end_time,
                    hw_accel=hw_accel,
                )
                job.clip_paths = clips
                if clips:
                    job.clip_dir = os.path.dirname(clips[0])

            else:
                job.status = "failed"
                job.progress = 0
                from pipeline import StepResult, StepStatus
                job.steps = [StepResult(
                    step="split_video", status=StepStatus.FAILED,
                    message=f"Unknown mode: {mode}",
                )]
                pipeline._save_jobs()
                return

            job.progress = 100
            job.status = "completed"
            job.current_step = "done"
            from pipeline import StepResult, StepStatus
            msg = f"切割完成: 生成 {len(job.clip_paths)} 个片段"
            job.steps = [StepResult(
                step="split_video", status=StepStatus.COMPLETED, message=msg,
            )]

        except Exception as e:
            job.status = "failed"
            job.progress = 0
            job.current_step = "error"
            from pipeline import StepResult, StepStatus
            job.steps = [StepResult(
                step="split_video", status=StepStatus.FAILED, message=str(e),
            )]

        pipeline._save_jobs()

    threading.Thread(target=run, daemon=True).start()

    return {
        "job_id": job_id,
        "title": title,
        "status": "started",
        "mode": mode,
    }


# ============================================================
# ASR (Speech Recognition) endpoints
# ============================================================

# ASR job tracking
_asr_jobs: dict[str, dict] = {}
_asr_lock = threading.RLock()


def _find_videos_for_asr() -> list:
    """Find all video files (MKV, MP4) in downloads and video directories."""
    videos = []
    search_dirs = [DOWNLOAD_DIR]

    # Also check a video/ directory at repo root level
    video_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "video")
    if os.path.isdir(video_dir):
        search_dirs.append(video_dir)

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, _dirs, filenames in os.walk(search_dir):
            for f in filenames:
                ext = f.lower()
                if ext.endswith(".mp4"):
                    full = os.path.join(root, f)
                    size_mb = os.path.getsize(full) / 1024 / 1024
                    mtime = os.path.getmtime(full)
                    from datetime import datetime
                    videos.append({
                        "name": f,
                        "path": full,
                        "size_mb": round(size_mb, 1),
                        "mtime": mtime,
                        "time_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                    })

    videos.sort(key=lambda x: x["mtime"], reverse=True)
    return videos


@app.get("/api/asr/videos")
def list_asr_videos():
    """List video files available for ASR extraction."""
    return {"videos": _find_videos_for_asr()}


@app.get("/api/asr/models")
def list_asr_models():
    """List available ASR models and their configurations."""
    from asr_pipeline import ASR_MODELS
    models = []
    for key, info in ASR_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "languages": info["languages"],
        })
    return {"models": models}


@app.post("/api/asr/run")
def run_asr_extraction(
    path: str = Query(..., description="Full path to video file"),
    model: str = Query("qwen3-asr", description="ASR model key"),
    language: str = Query("zh", description="Language code or 'auto'"),
    device: str = Query("cuda", description="Device: cuda or cpu"),
    hotwords: str = Query("", description="Context/ proper nouns for recognition"),
):
    """Start an ASR extraction job on a video file.

    Returns a job_id for polling via GET /api/asr/job/{job_id}.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video file not found")

    job_id = uuid.uuid4().hex[:12]
    video_name = os.path.basename(path)

    with _asr_lock:
        _asr_jobs[job_id] = {
            "job_id": job_id,
            "video_name": video_name,
            "video_path": path,
            "model": model,
            "language": language,
            "hotwords": hotwords,
            "status": "pending",
            "progress": 0,
            "current_step": "",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
        _save_asr_jobs()

    def _run():
        from asr_pipeline import run_asr_pipeline, ASR_MODELS

        with _asr_lock:
            job = _asr_jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            _save_asr_jobs()

        try:
            model_info = ASR_MODELS.get(model, ASR_MODELS.get("qwen3-asr", {}))

            def on_progress(step, pct):
                with _asr_lock:
                    j = _asr_jobs.get(job_id)
                    if j:
                        j["progress"] = pct
                        step_labels = {
                            "extracting_audio": "正在从视频提取音频...",
                            "running_asr": f"正在使用 {model_info.get('name', model)} 进行语音识别...",
                            "generating_srt": "正在生成字幕文件...",
                            "completed": "处理完成",
                        }
                        j["current_step"] = step_labels.get(step, step)

            result = run_asr_pipeline(
                path, model_key=model, language=language, device=device,
                progress_callback=on_progress, hotwords=hotwords,
            )

            with _asr_lock:
                j = _asr_jobs.get(job_id)
                if j:
                    j["status"] = "completed"
                    j["progress"] = 100
                    j["current_step"] = "处理完成"
                    j["result"] = result
                    _save_asr_jobs()

        except Exception as e:
            with _asr_lock:
                j = _asr_jobs.get(job_id)
                if j:
                    j["status"] = "failed"
                    j["error"] = str(e)
                    j["current_step"] = f"错误: {e}"
                    _save_asr_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {"job_id": job_id, "status": "started"}


@app.get("/api/asr/job/{job_id}")
def get_asr_job(job_id: str):
    """Get the status and result of an ASR job."""
    with _asr_lock:
        job = _asr_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/asr/jobs")
def list_asr_jobs():
    """List all ASR jobs (most recent first)."""
    with _asr_lock:
        jobs = list(_asr_jobs.values())
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return {"jobs": jobs}


@app.delete("/api/asr/job/{job_id}")
def cancel_asr_job(job_id: str):
    """Cancel a running ASR job."""
    with _asr_lock:
        job = _asr_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") not in ("running", "pending"):
            raise HTTPException(status_code=400, detail="Job is not running")
        job["status"] = "cancelled"
        job["current_step"] = "已取消"
        _save_asr_jobs()
    return {"status": "cancelled"}


@app.delete("/api/asr/jobs")
def clear_asr_jobs():
    """Clear completed, failed, and cancelled ASR jobs."""
    with _asr_lock:
        to_remove = [
            jid for jid, j in _asr_jobs.items()
            if j.get("status") in ("completed", "failed", "cancelled")
        ]
        for jid in to_remove:
            del _asr_jobs[jid]
        if to_remove:
            _save_asr_jobs()
    return {"cleared": len(to_remove)}


@app.post("/api/asr/job/{job_id}/discard")
def asr_discard_job(job_id: str):
    """Discard an interrupted ASR job."""
    with _asr_lock:
        job = _asr_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job["status"] = "cancelled"
        job["current_step"] = "已丢弃"
        _save_asr_jobs()
    return {"status": "discarded"}


@app.post("/api/asr/job/{job_id}/resume")
def asr_resume_job(job_id: str):
    """Resume an interrupted ASR job."""
    with _asr_lock:
        job = _asr_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") not in ("interrupted",):
            raise HTTPException(status_code=400, detail="Job is not in interrupted state")
        job_type = job.get("type", "video")

    if job_type == "folder":
        threading.Thread(target=_run_asr_folder_resume, args=(job_id,), daemon=True).start()
    else:
        threading.Thread(target=_run_asr_resume, args=(job_id,), daemon=True).start()
    return {"status": "resuming"}


def _run_asr_resume(job_id: str):
    """Re-run an interrupted single-video ASR job."""
    from asr_pipeline import run_asr_pipeline, ASR_MODELS

    with _asr_lock:
        job = _asr_jobs.get(job_id)
        if not job:
            return
        path = job.get("video_path", "")
        model = job.get("model", "qwen3-asr")
        language = job.get("language", "zh")
        device = job.get("device", "cuda")
        hotwords = job.get("hotwords", "")
        job["status"] = "running"
        job["error"] = None
        _save_asr_jobs()

    try:
        model_info = ASR_MODELS.get(model, ASR_MODELS.get("qwen3-asr", {}))

        def on_progress(step, pct):
            with _asr_lock:
                j = _asr_jobs.get(job_id)
                if j:
                    j["progress"] = pct
                    step_labels = {
                        "extracting_audio": "正在从视频提取音频...",
                        "running_asr": f"正在使用 {model_info.get('name', model)} 进行语音识别...",
                        "generating_srt": "正在生成字幕文件...",
                        "completed": "处理完成",
                    }
                    j["current_step"] = step_labels.get(step, step)

        result = run_asr_pipeline(
            path, model_key=model, language=language, device=device,
            progress_callback=on_progress, hotwords=hotwords,
        )

        with _asr_lock:
            j = _asr_jobs.get(job_id)
            if j:
                j["status"] = "completed"
                j["progress"] = 100
                j["current_step"] = "处理完成"
                j["result"] = result
                _save_asr_jobs()

    except Exception as e:
        with _asr_lock:
            j = _asr_jobs.get(job_id)
            if j:
                j["status"] = "failed"
                j["error"] = str(e)
                j["current_step"] = f"错误: {e}"
                _save_asr_jobs()


def _run_asr_folder_resume(job_id: str):
    """Re-run an interrupted folder ASR job, skipping already completed files."""
    from asr_pipeline import run_asr_on_audio

    with _asr_lock:
        job = _asr_jobs.get(job_id)
        if not job:
            return
        folder_path = job.get("folder_path", "")
        model_key = job.get("model", "qwen3-asr")
        language = job.get("language", "zh")
        device = job.get("device", "cuda")
        hotwords = job.get("hotwords", "")
        output_dir = job.get("output_dir", "")
        all_files = list(job.get("files", []))
        completed_names = {r.get("audio_name", "") for r in job.get("results", [])}

    # Only process files that haven't been completed yet
    pending_files = [af for af in all_files if af.get("name", "") not in completed_names]
    if not pending_files:
        with _asr_lock:
            j = _asr_jobs.get(job_id)
            if j:
                j["status"] = "completed"
                j["progress"] = 100
                j["current_step"] = "处理完成"
                _save_asr_jobs()
        return

    with _asr_lock:
        job = _asr_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["current_step"] = "加载模型中..."
        _save_asr_jobs()

    for idx, af in enumerate(pending_files):
        with _asr_lock:
            j = _asr_jobs.get(job_id)
            if not j or j.get("status") == "cancelled":
                return
            j["current_file"] = af["name"]
            j["progress"] = round((idx / len(pending_files)) * 100)

        def on_progress(step, msg):
            with _asr_lock:
                jj = _asr_jobs.get(job_id)
                if jj and jj.get("status") != "cancelled":
                    jj["current_step"] = str(msg) if msg else str(step)

        try:
            result = run_asr_on_audio(
                af["path"], output_dir,
                model_key=model_key, language=language, device=device,
                progress_callback=on_progress, hotwords=hotwords,
            )
            with _asr_lock:
                jj = _asr_jobs.get(job_id)
                if jj:
                    jj["results"].append(result)
                    jj["completed"] = len(jj["results"])
        except Exception as e:
            with _asr_lock:
                jj = _asr_jobs.get(job_id)
                if jj:
                    jj["errors"].append({"file": af["name"], "error": str(e)})

    with _asr_lock:
        j = _asr_jobs.get(job_id)
        if j and j.get("status") != "cancelled":
            j["status"] = "completed"
            j["progress"] = 100
            j["current_file"] = ""
            j["current_step"] = "处理完成"
            _save_asr_jobs()


@app.get("/api/asr/preview-audio")
def preview_asr_audio(path: str = Query(..., description="Path to video file")):
    """Stream audio from a video file (or cached WAV) for preview."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    base = os.path.splitext(os.path.basename(path))[0]
    wav_path = os.path.join(ASR_AUDIO_DIR, f"{base}.wav")

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return FileResponse(wav_path, media_type="audio/wav")

    # Pipe audio directly from video via ffmpeg
    def _generate():
        proc = subprocess.Popen(
            [FFMPEG, "-y", "-i", path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.stdout.close()
            proc.wait()

    return StreamingResponse(_generate(), media_type="audio/wav")


@app.get("/api/asr/results")
def list_asr_results(video_name: str = Query("", description="Optional filter by video name")):
    """List ASR-generated subtitles and audio files."""
    results = []
    if os.path.isdir(ASR_SUBTITLE_DIR):
        for f in sorted(os.listdir(ASR_SUBTITLE_DIR), reverse=True):
            if not f.endswith(".srt"):
                continue
            full = os.path.join(ASR_SUBTITLE_DIR, f)
            if video_name and video_name not in f:
                continue
            base = os.path.splitext(f)[0]
            # Find matching audio file
            audio_path = os.path.join(ASR_AUDIO_DIR, f"{base}.wav")
            if not os.path.exists(audio_path):
                # Try without model suffix
                for candidate in sorted(os.listdir(ASR_AUDIO_DIR)):
                    if candidate.startswith(base.split("_")[0]) and candidate.endswith(".wav"):
                        audio_path = os.path.join(ASR_AUDIO_DIR, candidate)
                        break

            # Try to find the source video path from completed jobs
            video_path = ""
            with _asr_lock:
                for job in _asr_jobs.values():
                    if job.get("result", {}).get("srt_path") == full:
                        video_path = job.get("video_path", "")
                        break

            results.append({
                "name": f,
                "path": full,
                "size_kb": round(os.path.getsize(full) / 1024, 1),
                "audio_path": audio_path if os.path.exists(audio_path) else "",
                "audio_name": os.path.basename(audio_path) if os.path.exists(audio_path) else "",
                "video_path": video_path,
                "mtime": os.path.getmtime(full),
            })

    return {"results": results}


@app.get("/api/asr/stream")
def stream_asr_file(path: str = Query(..., description="Full path to SRT or audio file")):
    """Stream an ASR result file (SRT subtitle or WAV audio)."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "audio/wav" if path.lower().endswith(".wav") else "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media_type)


@app.delete("/api/asr/delete")
def delete_asr_result(path: str = Query(..., description="Path to SRT file to delete")):
    """Delete an ASR result (SRT and optionally its audio)."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    deleted = []
    try:
        os.remove(path)
        deleted.append(os.path.basename(path))
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Try to delete matching audio
    base = os.path.splitext(path)[0]
    for ext in (".wav",):
        audio_path = base + ext
        if os.path.exists(audio_path) and audio_path != path:
            try:
                os.remove(audio_path)
                deleted.append(os.path.basename(audio_path))
            except OSError:
                pass

    return {"deleted": deleted}


@app.get("/api/asr/dir-browse")
def browse_asr_directory(path: str = Query("")):
    """Browse a directory for sub-folders and audio files. Used by the in-page file browser."""
    if not path:
        path = DATA_DIR
    # Security: resolve real path and ensure it's within DATA_DIR
    real_path = os.path.realpath(path)
    real_data = os.path.realpath(DATA_DIR)
    if not real_path.startswith(real_data + os.sep) and real_path != real_data:
        raise HTTPException(status_code=403, detail="Access denied: path outside data directory")
    if not os.path.isdir(real_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    parent_path = os.path.dirname(real_path)
    # Don't allow going above DATA_DIR
    if not parent_path.startswith(real_data + os.sep) and parent_path != real_data:
        parent_path = real_data if real_path != real_data else ""

    entries = sorted(os.scandir(real_path), key=lambda e: e.name)
    subdirs = []
    audio_files = []
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus", ".wv"}

    for entry in entries:
        if entry.is_dir() and not entry.name.startswith('.'):
            subdirs.append(entry.name)
        elif entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in audio_exts:
                size = entry.stat().st_size
                audio_files.append({
                    "name": entry.name,
                    "ext": ext,
                    "size_mb": round(size / 1024 / 1024, 1),
                })

    return {
        "current_path": real_path,
        "parent_path": parent_path,
        "subdirs": subdirs,
        "audio_files": audio_files,
        "has_audio": len(audio_files) > 0,
    }


@app.post("/api/asr/folder-detect")
def detect_asr_folder_files(payload: dict = Body(...)):
    """Scan a folder for audio files and return the list. Uses POST to avoid URL encoding issues with paths."""
    path = payload.get("path", "").strip()
    debug_info = {"path_received": path, "exists": os.path.exists(path) if path else False,
                  "isdir": os.path.isdir(path) if path else False}

    if not path or not os.path.isdir(path):
        debug_info["error"] = "路径不存在或不是文件夹"
        return {"files": [], "total_size_mb": 0, "count": 0, "_debug": debug_info}

    try:
        raw_entries = sorted(os.listdir(path))
    except PermissionError:
        debug_info["error"] = "没有访问权限"
        return {"files": [], "total_size_mb": 0, "count": 0, "_debug": debug_info}

    debug_info["raw_entry_count"] = len(raw_entries)
    debug_info["raw_sample"] = raw_entries[:10]

    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus", ".wv"}
    files = []
    total_bytes = 0
    for f in raw_entries:
        ext = os.path.splitext(f)[1].lower()
        if ext in audio_exts:
            full = os.path.join(path, f)
            if os.path.isfile(full):
                size = os.path.getsize(full)
                total_bytes += size
                files.append({
                    "name": f,
                    "ext": ext,
                    "size_mb": round(size / 1024 / 1024, 1),
                })

    return {
        "files": files,
        "total_size_mb": round(total_bytes / 1024 / 1024, 1),
        "count": len(files),
        "_debug": debug_info,
    }


@app.get("/api/asr/browse-folder")
def browse_folder():
    """Open a native folder picker dialog and return the selected path.

    Launches a separate Python process with tkinter to show the dialog,
    because the background server thread cannot reliably display Windows UI."""
    if not _display_available():
        return {"path": "", "error": "Server has no display — use manual path input instead."}
    import subprocess, tempfile, os, sys

    # Write result path to a temp file (dialog runs in a separate process)
    result_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    result_path = result_file.name
    result_file.close()

    default_dir = DATA_DIR if os.path.isdir(DATA_DIR) else os.path.expanduser("~")
    picker_code = f'''
import tkinter as tk
from tkinter import filedialog
import sys, os, ctypes
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.lift()
root.focus_force()
root.update()
# Force the tkinter window to the foreground via Windows API
hwnd = root.winfo_id() if root.winfo_exists() else 0
if hwnd:
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.BringWindowToTop(hwnd)
try:
    path = filedialog.askdirectory(title="选择包含音频文件的文件夹", initialdir=r"{default_dir}")
    if path:
        with open(r"{result_path}", "w", encoding="utf-8") as f:
            f.write(path)
except Exception:
    pass
try:
    root.destroy()
except Exception:
    pass
'''

    picker_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    picker_file.write(picker_code)
    picker_path = picker_file.name
    picker_file.close()

    try:
        subprocess.run(
            [sys.executable, picker_path],
            timeout=300,
        )
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                path = f.read().strip()
            return {"path": path}
        return {"path": ""}
    except subprocess.TimeoutExpired:
        return {"path": "", "error": "操作超时"}
    except Exception as e:
        return {"path": "", "error": str(e)}
    finally:
        for f in (picker_path, result_path):
            if os.path.exists(f):
                os.unlink(f)


@app.post("/api/asr/folder")
def run_asr_folder(payload: dict = Body(...)):
    """Process all audio files in a folder with ASR.

    Body: {folder_path, model?, language?, device?, output_dir?, selected_files?}
    Saves SRT files to output_dir/<folder_name>/ named after each audio file.
    """
    folder_path = payload.get("folder_path", "").strip()
    model_key = payload.get("model", "qwen3-asr")
    language = payload.get("language", "zh")
    device = payload.get("device", "cuda")
    output_base = payload.get("output_dir", "").strip()
    selected_files = payload.get("selected_files", None)  # optional list of file names
    hotwords = payload.get("hotwords", "")

    if not folder_path or not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="无效的文件夹路径")

    # Find audio files in the folder
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus", ".wv"}
    audio_files = []
    for f in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(f)[1].lower()
        if ext in audio_exts:
            full = os.path.join(folder_path, f)
            if os.path.isfile(full):
                audio_files.append({
                    "name": f,
                    "path": full,
                    "ext": ext,
                    "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                })

    # Filter to selected files only
    if selected_files is not None and len(selected_files) > 0:
        sel_set = set(selected_files)
        audio_files = [af for af in audio_files if af["name"] in sel_set]

    if not audio_files:
        raise HTTPException(status_code=400, detail="文件夹中没有找到音频文件")

    # Determine output directory
    folder_name = os.path.basename(folder_path.rstrip("/\\"))
    if not output_base:
        output_base = os.path.join(ASR_DIR, "folder_output")
    output_dir = os.path.join(output_base, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    job_id = uuid.uuid4().hex[:12]

    job = {
        "job_id": job_id,
        "status": "pending",
        "type": "folder",
        "folder_path": folder_path,
        "folder_name": folder_name,
        "output_dir": output_dir,
        "files": audio_files,
        "total": len(audio_files),
        "completed": 0,
        "progress": 0,
        "current_file": "",
        "current_step": "",
        "model": model_key,
        "language": language,
        "device": device,
        "hotwords": hotwords,
        "results": [],
        "errors": [],
    }

    with _asr_lock:
        _asr_jobs[job_id] = job
        _save_asr_jobs()

    def _run():
        with _asr_lock:
            j = _asr_jobs.get(job_id)
            if not j:
                return
            j["status"] = "running"
            j["current_step"] = "加载模型中..."
            _save_asr_jobs()

        from asr_pipeline import run_asr_on_audio

        for idx, af in enumerate(audio_files):
            with _asr_lock:
                j = _asr_jobs.get(job_id)
                if not j or j.get("status") == "cancelled":
                    return
                j["current_file"] = af["name"]
                j["progress"] = round((idx / len(audio_files)) * 100)

            def on_progress(step, msg):
                with _asr_lock:
                    jj = _asr_jobs.get(job_id)
                    if jj and jj.get("status") != "cancelled":
                        jj["current_step"] = str(msg) if msg else str(step)

            try:
                result = run_asr_on_audio(
                    af["path"], output_dir,
                    model_key=model_key, language=language, device=device,
                    progress_callback=on_progress, hotwords=hotwords,
                )
                with _asr_lock:
                    jj = _asr_jobs.get(job_id)
                    if jj:
                        jj["results"].append(result)
                        jj["completed"] = idx + 1
            except Exception as e:
                with _asr_lock:
                    jj = _asr_jobs.get(job_id)
                    if jj:
                        jj["errors"].append({"file": af["name"], "error": str(e)})

        with _asr_lock:
            j = _asr_jobs.get(job_id)
            if j and j.get("status") != "cancelled":
                j["status"] = "completed"
                j["progress"] = 100
                j["current_file"] = ""
                j["current_step"] = "处理完成"
                _save_asr_jobs()

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "ok",
        "job_id": job_id,
        "file_count": len(audio_files),
        "output_dir": output_dir,
    }


@app.get("/api/asr/folder-output-dir")
def get_asr_folder_output_dir():
    """Return the default output directory for folder ASR results."""
    output_base = os.path.join(ASR_DIR, "folder_output")
    return {"path": output_base}


# ============================================================
# ASR Comparison endpoints
# ============================================================

_asr_compare_jobs: dict[str, dict] = {}
_asr_compare_lock = threading.RLock()
_asr_compare_results: dict[str, dict] = {}  # keyed by audio path

_ASR_COMPARE_JOBS_FILE = os.path.join(DATA_DIR, "asr_compare_jobs.json")


def _save_asr_compare_jobs():
    """Persist ASR compare jobs to disk."""
    try:
        with _asr_compare_lock:
            data = dict(_asr_compare_jobs)
        tmp = _ASR_COMPARE_JOBS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, _ASR_COMPARE_JOBS_FILE)
    except Exception as e:
        print(f"[asr-compare] Failed to save jobs: {e}")


def _sync_user_action_to_jobs(audio_path: str):
    """Copy user_action/flagged from _asr_compare_results into all job results and persist."""
    with _asr_compare_lock:
        result = _asr_compare_results.get(audio_path, {})
        user_action = result.get("user_action")
        flagged = result.get("flagged", False)
        segments = result.get("segments")
        for job in _asr_compare_jobs.values():
            job_results = job.get("results", {})
            if audio_path in job_results:
                job_results[audio_path]["user_action"] = user_action
                job_results[audio_path]["flagged"] = flagged
                if segments is not None:
                    job_results[audio_path]["segments"] = segments
        _save_asr_compare_jobs()


def _load_asr_compare_jobs():
    """Restore ASR compare jobs from disk. Mark running/loading jobs as interrupted."""
    if not os.path.exists(_ASR_COMPARE_JOBS_FILE):
        return
    try:
        with open(_ASR_COMPARE_JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for job_id, jd in data.items():
            status = jd.get("status", "")
            if status in ("running", "loading", "pending"):
                jd["status"] = "interrupted"
                jd["current_step"] = "服务器重启中断"
            _asr_compare_jobs[job_id] = jd
        print(f"[startup] Restored {len(_asr_compare_jobs)} ASR compare jobs from disk")
    except Exception as e:
        print(f"[startup] Failed to load ASR compare jobs: {e}")


def _find_unreviewed_wav_files() -> list:
    """Find all denoised WAV files in cleaned_unreviewed/, grouped by source dir."""
    dirs = []
    base = CLEANED_UNREVIEWED_DIR
    if not os.path.exists(base):
        return dirs

    for entry in sorted(os.listdir(base)):
        dir_path = os.path.join(base, entry)
        if not os.path.isdir(dir_path):
            continue
        wav_files = []
        for f in sorted(os.listdir(dir_path)):
            if f.endswith(".wav"):
                full = os.path.join(dir_path, f)
                wav_files.append({
                    "name": f,
                    "path": full.replace("\\", "/"),
                    "size_kb": round(os.path.getsize(full) / 1024, 1),
                })
        if wav_files:
            dirs.append({
                "name": entry,
                "path": dir_path.replace("\\", "/"),
                "files": wav_files,
                "file_count": len(wav_files),
            })
    return dirs


@app.get("/api/asr-compare/audio-files")
def list_asr_compare_files():
    """List all WAV files from cleaned_unreviewed/, grouped by source directory."""
    return {"dirs": _find_unreviewed_wav_files()}


@app.get("/api/asr-compare/dir-browse")
def browse_asr_compare_directory(path: str = Query("")):
    """Browse a directory for sub-folders and audio files.

    Defaults to the configured ASR_COMPARE_DEFAULT_PATH, or CLEANED_UNREVIEWED_DIR.
    Supports both local paths and network shares (SMB/NFS mounts).
    Falls back to CLEANED_UNREVIEWED_DIR if the default path does not exist.
    """
    from config import ASR_COMPARE_DEFAULT_PATH
    if not path:
        path = ASR_COMPARE_DEFAULT_PATH or CLEANED_UNREVIEWED_DIR

    # Normalize path: convert Windows backslash to forward slash
    path = path.replace("\\", "/")

    # Try to resolve the real path (may fail for unmounted network paths)
    try:
        real_path = os.path.realpath(path)
    except Exception:
        real_path = path

    # If the requested path does not exist and we were using the default,
    # try falling back to CLEANED_UNREVIEWED_DIR as a local alternative
    if not os.path.isdir(real_path):
        if (not path or path == ASR_COMPARE_DEFAULT_PATH) and os.path.isdir(CLEANED_UNREVIEWED_DIR):
            real_path = os.path.realpath(CLEANED_UNREVIEWED_DIR)
        else:
            raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    parent_path = os.path.dirname(real_path)

    try:
        entries = sorted(os.scandir(real_path), key=lambda e: e.name)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {real_path}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Cannot read directory: {real_path} ({e.strerror})")

    subdirs = []
    audio_files = []
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus", ".wv"}

    for entry in entries:
        if entry.is_dir() and not entry.name.startswith('.'):
            subdirs.append(entry.name)
        elif entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in audio_exts:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                audio_files.append({
                    "name": entry.name,
                    "ext": ext,
                    "size_mb": round(size / 1024 / 1024, 1),
                })

    return {
        "current_path": real_path,
        "parent_path": parent_path,
        "subdirs": subdirs,
        "audio_files": audio_files,
        "has_audio": len(audio_files) > 0,
    }


@app.get("/api/asr-compare/models")
def list_asr_compare_models():
    """Return the two comparison models with their language support."""
    from asr_pipeline import ASR_MODELS, COMPARE_MODELS
    models = []
    for key in COMPARE_MODELS:
        info = ASR_MODELS.get(key)
        if info:
            models.append({
                "key": key,
                "name": info["name"],
                "description": info["description"],
                "languages": info["languages"],
                "abbr": info.get("abbr", key),
            })
    return {"models": models}


@app.post("/api/asr-compare/run-all")
def run_asr_compare_all(
    request: Request,
    language: str = Query("zh"),
    device: str = Query("cuda"),
    model_a: str = Query("qwen3-asr"),
    model_b: str = Query("qwen3-asr"),
    hotwords: str = Query("", description="Context/ proper nouns for recognition"),
    filter_english: str = Query("1", description="1=force 0% match if English detected"),
    payload: dict = Body(default={}),
):
    """Queue audio files from a browsed folder and process sequentially.

    Accepts JSON body with optional folder_path and selected_files.
    Falls back to query-param dirs for backward compatibility.
    """
    from asr_pipeline import compare_asr_pipeline, ASR_MODELS
    _filter_english = filter_english == "1"

    all_files = []
    # Priority 1: JSON body with folder_path + selected_files
    folder_path = payload.get("folder_path", "").strip() if payload else ""
    selected_files = payload.get("selected_files", []) if payload else []
    if folder_path and selected_files:
        # Normalize path: convert Windows backslash to forward slash
        folder_path = folder_path.replace("\\", "/")
        try:
            real_folder = os.path.realpath(folder_path)
        except Exception:
            real_folder = folder_path
        if not os.path.isdir(real_folder):
            raise HTTPException(status_code=404, detail=f"Directory not found: {folder_path}")
        source_dir = os.path.basename(real_folder) or os.path.basename(os.path.dirname(real_folder))
        for fname in selected_files:
            fp = os.path.join(real_folder, fname)
            if os.path.isfile(fp):
                all_files.append({
                    "name": fname,
                    "path": fp.replace("\\", "/"),
                    "source_dir": source_dir,
                })
    # Priority 2: fallback to old dirs query param
    if not all_files:
        dirs: list[str] = request.query_params.getlist("dirs")
        all_dirs = _find_unreviewed_wav_files()
        dir_filter = set(dirs) if dirs else None
        for d in all_dirs:
            if dir_filter and d["name"] not in dir_filter:
                continue
            for f in d["files"]:
                all_files.append({
                    "name": f["name"],
                    "path": f["path"],
                    "source_dir": d["name"],
                })

    if not all_files:
        raise HTTPException(status_code=404, detail="No audio files found")

    # Add per-file tracking fields
    for _f in all_files:
        _f.setdefault("status", "pending")
        _f.setdefault("progress", 0)
        _f.setdefault("current_step", "")

    job_id = uuid.uuid4().hex[:12]

    with _asr_compare_lock:
        _asr_compare_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "current_step": "",
            "total": len(all_files),
            "completed": 0,
            "flagged_count": 0,
            "files": all_files,
            "results": {},
            "error": None,
            "created_at": time.time(),
            "_model_a": model_a,
            "_model_b": model_b,
            "_language": language,
            "_device": device,
            "_hotwords": hotwords,
        }
        _save_asr_compare_jobs()

    def _run():
        from asr_pipeline import compare_asr_pipeline, _get_asr_model, _CancelPipeline
        from asr_pipeline import ASR_MODELS, COMPARE_MODELS

        def _cancel_check():
            with _asr_compare_lock:
                j = _asr_compare_jobs.get(job_id)
                return bool(j and j.get("_cancelled"))

        with _asr_compare_lock:
            job = _asr_compare_jobs.get(job_id)
            if not job:
                return
            job["status"] = "loading"
            job["current_step"] = "正在加载模型..."
            _save_asr_compare_jobs()

        # Preload both ASR models before starting processing.
        # This is critical because FunASR / ModelScope downloads can behave
        # unpredictably in threads. We run with daemon=False to mitigate this.
        try:
            for _mk in (model_a, model_b):
                with _asr_compare_lock:
                    job = _asr_compare_jobs.get(job_id)
                    if job:
                        job["current_step"] = f"正在加载模型: {_mk}"
                        _save_asr_compare_jobs()
                _get_asr_model(_mk, device=device, use_compile=False)
        except Exception as e:
            with _asr_compare_lock:
                job = _asr_compare_jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["error"] = f"模型加载失败: {e}"
                    job["current_step"] = f"模型加载失败: {e}"
                    _save_asr_compare_jobs()
            return

        with _asr_compare_lock:
            job = _asr_compare_jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["current_step"] = "模型加载完成，开始处理..."
            _save_asr_compare_jobs()

        files = all_files
        total = len(files)

        for idx, f in enumerate(files):
            with _asr_compare_lock:
                job = _asr_compare_jobs.get(job_id)
                if not job or job.get("_cancelled"):
                    if job:
                        job["status"] = "cancelled"
                        job["current_step"] = "已中止"
                        _save_asr_compare_jobs()
                    return
                f["status"] = "running"
                f["progress"] = 0
                f["current_step"] = "开始处理..."

            audio_path = f["path"]
            audio_name = f["name"]

            def on_progress(step, pct):
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["current_step"] = f"[{idx + 1}/{total}] {audio_name} — {step}"
                        f["progress"] = int(pct)
                        f["current_step"] = step

            try:
                result = compare_asr_pipeline(
                    audio_path,
                    language=language,
                    device=device,
                    progress_callback=on_progress,
                    source_dir=f["source_dir"],
                    model_a=model_a,
                    model_b=model_b,
                    hotwords=hotwords,
                    cancel_check=_cancel_check,
                    filter_english=_filter_english,
                )
                result["source_dir"] = f["source_dir"]

                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["completed"] = idx + 1
                        j["progress"] = int(((idx + 1) / total) * 100)
                        j["results"][audio_path] = result
                        if result.get("flagged"):
                            j["flagged_count"] += 1
                    f["status"] = "completed"
                    f["progress"] = 100

            except _CancelPipeline:
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["status"] = "cancelled"
                        j["current_step"] = "已中止"
                        _save_asr_compare_jobs()
                return
            except Exception as e:
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["completed"] = idx + 1
                        j["progress"] = int(((idx + 1) / total) * 100)
                        j["results"][audio_path] = {
                            "audio_path": audio_path,
                            "audio_name": audio_name,
                            "source_dir": f["source_dir"],
                            "error": str(e),
                            "flagged": True,
                        }
                        j["flagged_count"] += 1
                    f["status"] = "error"
                    f["progress"] = 0
                    f["error"] = str(e)

        with _asr_compare_lock:
            j = _asr_compare_jobs.get(job_id)
            if j:
                j["status"] = "completed"
                j["progress"] = 100
                j["current_step"] = f"处理完成 — {total} 个文件, {j['flagged_count']} 个异常"
                _save_asr_compare_jobs()

        # Persist results for later queries
        with _asr_compare_lock:
            j = _asr_compare_jobs.get(job_id)
            if j:
                for path, result in j["results"].items():
                    _asr_compare_results[path] = result

    # Return response immediately so the frontend doesn't hang on "启动中...".
    # Model preloading and processing happen in a background daemon thread.
    # FunASR/ModelScope model downloads can hang in daemon threads, so we use
    # a regular (non-daemon) thread and preload models as the first step.
    threading.Thread(target=_run, daemon=False).start()

    return {"job_id": job_id, "status": "started", "total": len(all_files)}


@app.get("/api/asr-compare/job/{job_id}")
def get_asr_compare_job(job_id: str):
    """Poll comparison job status."""
    with _asr_compare_lock:
        job = _asr_compare_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/asr-compare/jobs")
def list_asr_compare_jobs():
    """List all ASR compare jobs (most recent first)."""
    with _asr_compare_lock:
        jobs = list(_asr_compare_jobs.values())
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return {"jobs": jobs}


@app.post("/api/asr-compare/cancel")
def cancel_asr_compare_job(payload: dict = Body(...)):
    """Cancel a running comparison job."""
    job_id = payload.get("job_id", "")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")

    with _asr_compare_lock:
        job = _asr_compare_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") not in ("running", "pending", "loading"):
            return {"status": "error", "detail": f"Job is {job.get('status')}, cannot cancel"}
        job["_cancelled"] = True
        job["current_step"] = "正在中止..."
        _save_asr_compare_jobs()
    return {"status": "ok", "detail": "Cancelling..."}


@app.post("/api/asr-compare/job/{job_id}/discard")
def asr_compare_discard_job(job_id: str):
    """Discard an interrupted ASR compare job."""
    with _asr_compare_lock:
        job = _asr_compare_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job["status"] = "cancelled"
        job["current_step"] = "已丢弃"
        _save_asr_compare_jobs()
    return {"status": "discarded"}


@app.post("/api/asr-compare/job/{job_id}/resume")
def asr_compare_resume_job(job_id: str):
    """Resume an interrupted ASR compare job, skipping already-completed files.

    Mirrors the pipeline-video resume pattern: resets pending files, preloads
    models with status updates, and runs with daemon=False.
    """
    with _asr_compare_lock:
        job = _asr_compare_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") not in ("interrupted",):
            raise HTTPException(status_code=400, detail="Job is not in interrupted state")
        files = list(job.get("files", []))
        model_a = job.get("_model_a", "qwen3-asr")
        model_b = job.get("_model_b", "qwen3-asr")
        language = job.get("_language", "ja")
        device = job.get("_device", "cuda")
        hotwords = job.get("_hotwords", "")
        completed_paths = set(job.get("results", {}).keys())
        is_segmented = job.get("is_segmented", False)

    def _run_resume():
        from asr_pipeline import compare_asr_pipeline, segment_and_compare_pipeline
        from asr_pipeline import _get_asr_model, _CancelPipeline, ASR_MODELS

        def _cancel_check():
            with _asr_compare_lock:
                j = _asr_compare_jobs.get(job_id)
                return bool(j and j.get("_cancelled"))

        with _asr_compare_lock:
            j = _asr_compare_jobs.get(job_id)
            if not j:
                return
            j["status"] = "loading"
            j["current_step"] = "正在加载模型..."
            _save_asr_compare_jobs()

        # Preload both ASR models before processing
        try:
            for _mk in (model_a, model_b):
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["current_step"] = f"正在加载模型: {_mk}"
                        _save_asr_compare_jobs()
                _get_asr_model(_mk, device=device, use_compile=False)
        except Exception as e:
            with _asr_compare_lock:
                j = _asr_compare_jobs.get(job_id)
                if j:
                    j["status"] = "failed"
                    j["error"] = f"模型加载失败: {e}"
                    j["current_step"] = f"模型加载失败: {e}"
                    _save_asr_compare_jobs()
            return

        with _asr_compare_lock:
            j = _asr_compare_jobs.get(job_id)
            if not j:
                return
            j["status"] = "running"
            j["current_step"] = "模型加载完成，开始处理..."
            _save_asr_compare_jobs()

        total = len(files)
        # Count already-completed files for accurate progress tracking
        already_done = len(completed_paths)

        # Reset per-file status for non-completed files (they may be stuck in "running")
        for _f in files:
            if _f["path"] not in completed_paths:
                _f["status"] = "pending"
                _f["progress"] = 0
                _f["current_step"] = ""

        for idx, f in enumerate(files):
            audio_path = f["path"]
            if audio_path in completed_paths:
                continue

            audio_name = f["name"]

            with _asr_compare_lock:
                j = _asr_compare_jobs.get(job_id)
                if not j or j.get("_cancelled") or j.get("status") == "cancelled":
                    if j:
                        j["status"] = "cancelled"
                        j["current_step"] = "已中止"
                        _save_asr_compare_jobs()
                    return
                j["current_file"] = audio_name
                f["status"] = "running"
                f["progress"] = 0
                f["current_step"] = "开始处理..."

            def on_progress(step, pct):
                with _asr_compare_lock:
                    jj = _asr_compare_jobs.get(job_id)
                    if jj and jj.get("status") != "cancelled":
                        done = jj.get("completed", 0)
                        jj["current_step"] = f"[{done + 1}/{total}] {audio_name} — {step}"
                    f["progress"] = int(pct)
                    f["current_step"] = step

            try:
                if is_segmented:
                    seg_min = j.get("_segment_min_s", 9.0)
                    seg_max = j.get("_segment_max_s", 16.0)
                    result = segment_and_compare_pipeline(
                        audio_path, language=language, device=device,
                        model_a=model_a, model_b=model_b, hotwords=hotwords,
                        segment_min_s=seg_min, segment_max_s=seg_max,
                        progress_callback=on_progress, source_dir=f["source_dir"],
                        cancel_check=_cancel_check,
                        filter_english=_filter_english,
                    )
                else:
                    result = compare_asr_pipeline(
                        audio_path, language=language, device=device,
                        progress_callback=on_progress, source_dir=f["source_dir"],
                        model_a=model_a, model_b=model_b, hotwords=hotwords,
                        cancel_check=_cancel_check,
                        filter_english=_filter_english,
                    )
                result["source_dir"] = f["source_dir"]

                with _asr_compare_lock:
                    jj = _asr_compare_jobs.get(job_id)
                    if jj:
                        jj["completed"] = jj.get("completed", 0) + 1
                        jj["progress"] = int((jj["completed"] / total) * 100)
                        jj["results"][audio_path] = result
                        if result.get("flagged"):
                            jj["flagged_count"] = jj.get("flagged_count", 0) + 1
                f["status"] = "completed"
                f["progress"] = 100

            except _CancelPipeline:
                with _asr_compare_lock:
                    jj = _asr_compare_jobs.get(job_id)
                    if jj:
                        jj["status"] = "cancelled"
                        jj["current_step"] = "已中止"
                        _save_asr_compare_jobs()
                return
            except Exception as e:
                with _asr_compare_lock:
                    jj = _asr_compare_jobs.get(job_id)
                    if jj:
                        jj["completed"] = jj.get("completed", 0) + 1
                        jj["progress"] = int((jj["completed"] / total) * 100)
                        jj["results"][audio_path] = {
                            "audio_path": audio_path,
                            "audio_name": audio_name,
                            "source_dir": f["source_dir"],
                            "error": str(e),
                            "flagged": True,
                        }
                        jj["flagged_count"] = jj.get("flagged_count", 0) + 1
                f["status"] = "error"
                f["progress"] = 0
                f["error"] = str(e)

        with _asr_compare_lock:
            jj = _asr_compare_jobs.get(job_id)
            if jj and jj.get("status") != "cancelled":
                jj["status"] = "completed"
                jj["progress"] = 100
                jj["current_step"] = f"处理完成 — {total} 个文件, {jj['flagged_count']} 个异常"
                _save_asr_compare_jobs()
                for path, result in jj["results"].items():
                    _asr_compare_results[path] = result

    threading.Thread(target=_run_resume, daemon=False).start()
    return {"status": "resuming"}


@app.get("/api/asr-compare/results")
def list_asr_compare_results():
    """Get all comparison results."""
    with _asr_compare_lock:
        results = list(_asr_compare_results.values())

    # Also pull from latest completed job (skip deleted/discarded)
    with _asr_compare_lock:
        for j in _asr_compare_jobs.values():
            if j.get("status") == "completed":
                for path, result in j.get("results", {}).items():
                    if path not in _asr_compare_results:
                        _asr_compare_results[path] = result

    results = [r for r in _asr_compare_results.values()
               if r.get("user_action") not in ("deleted", "discarded")]
    results.sort(key=lambda r: r.get("source_dir", "") + r.get("audio_name", ""))
    return {"results": results}


@app.post("/api/asr-compare/keep")
def keep_asr_compare_audio(payload: dict = Body(...)):
    """Mark a flagged audio as kept."""
    audio_path = payload.get("path", "")
    if not audio_path:
        raise HTTPException(status_code=400, detail="Missing path")

    with _asr_compare_lock:
        if audio_path in _asr_compare_results:
            _asr_compare_results[audio_path]["flagged"] = False
            _asr_compare_results[audio_path]["user_action"] = "kept"

    _sync_user_action_to_jobs(audio_path)
    return {"status": "ok", "action": "kept"}


@app.post("/api/asr-compare/discard")
def discard_asr_compare_audio(payload: dict = Body(...)):
    """Move a flagged audio to the discard folder."""
    import shutil as _shutil
    audio_path = payload.get("path", "")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="Missing or invalid path")

    os.makedirs(ASR_COMPARE_DISCARD_DIR, exist_ok=True)
    dest = os.path.join(ASR_COMPARE_DISCARD_DIR, os.path.basename(audio_path))
    _shutil.move(audio_path, dest)

    # Also move associated SRT files
    with _asr_compare_lock:
        result = _asr_compare_results.get(audio_path, {})
    srt_paths = result.get("srt_paths", {})
    for model_key, srt_path in srt_paths.items():
        if os.path.exists(srt_path):
            srt_dest = os.path.join(ASR_COMPARE_DISCARD_DIR, os.path.basename(srt_path))
            _shutil.move(srt_path, srt_dest)

    with _asr_compare_lock:
        if audio_path in _asr_compare_results:
            _asr_compare_results[audio_path]["flagged"] = False
            _asr_compare_results[audio_path]["user_action"] = "discarded"

    _sync_user_action_to_jobs(audio_path)
    return {"status": "ok", "action": "discarded"}


@app.post("/api/asr-compare/delete")
def delete_asr_compare_audio(payload: dict = Body(...)):
    """Permanently delete an audio file and its associated SRT files from disk."""
    import shutil as _shutil
    audio_path = payload.get("path", "")
    if not audio_path:
        raise HTTPException(status_code=400, detail="Missing path")

    # Security: only allow deletion within DATA_DIR
    real_path = os.path.realpath(audio_path)
    real_data = os.path.realpath(DATA_DIR)
    if not real_path.startswith(real_data + os.sep) and real_path != real_data:
        raise HTTPException(status_code=403, detail="Access denied: path outside data directory")

    deleted_files = []
    if os.path.exists(audio_path):
        os.remove(audio_path)
        deleted_files.append(audio_path)

    # Also delete associated SRT files
    with _asr_compare_lock:
        result = _asr_compare_results.get(audio_path, {})
    srt_paths = result.get("srt_paths", {})
    for _model_key, srt_path in srt_paths.items():
        if os.path.exists(srt_path):
            os.remove(srt_path)
            deleted_files.append(srt_path)

    with _asr_compare_lock:
        if audio_path in _asr_compare_results:
            _asr_compare_results[audio_path]["flagged"] = False
            _asr_compare_results[audio_path]["user_action"] = "deleted"

    _sync_user_action_to_jobs(audio_path)
    return {"status": "ok", "action": "deleted", "files": deleted_files}


@app.post("/api/asr-compare/batch-delete")
def batch_delete_asr_compare_audios(payload: dict = Body(...)):
    """Permanently delete multiple audio files and their SRTs from disk."""
    import shutil as _shutil
    paths = payload.get("paths", [])
    if not paths:
        raise HTTPException(status_code=400, detail="No paths provided")

    real_data = os.path.realpath(DATA_DIR)
    results = []
    for audio_path in paths:
        real_path = os.path.realpath(audio_path)
        if not real_path.startswith(real_data + os.sep) and real_path != real_data:
            results.append({"path": audio_path, "status": "skipped", "reason": "access denied"})
            continue

        deleted = []
        if os.path.exists(audio_path):
            os.remove(audio_path)
            deleted.append(audio_path)

        with _asr_compare_lock:
            result = _asr_compare_results.get(audio_path, {})
        srt_paths = result.get("srt_paths", {})
        for _model_key, srt_path in srt_paths.items():
            if os.path.exists(srt_path):
                os.remove(srt_path)
                deleted.append(srt_path)

        with _asr_compare_lock:
            if audio_path in _asr_compare_results:
                _asr_compare_results[audio_path]["flagged"] = False
                _asr_compare_results[audio_path]["user_action"] = "deleted"

        _sync_user_action_to_jobs(audio_path)
        results.append({"path": audio_path, "status": "deleted", "files": deleted})

    return {"status": "ok", "total": len(paths), "results": results}


# --- Segmented ASR Compare ---

@app.post("/api/asr-compare/run-segmented")
def run_asr_compare_segmented(
    language: str = Query("zh"),
    device: str = Query("cuda"),
    model_a: str = Query("qwen3-asr"),
    model_b: str = Query("cohere-transcribe"),
    hotwords: str = Query("", description="Context/ proper nouns for recognition"),
    filter_english: str = Query("1", description="1=force 0% match if English detected"),
    segment_min_s: float = Query(9.0),
    segment_max_s: float = Query(16.0),
    payload: dict = Body(default={}),
):
    """VAD-split each audio into 10-15s segments, run two models on each, compare."""
    _filter_english = filter_english == "1"
    from asr_pipeline import segment_and_compare_pipeline, ASR_MODELS

    folder_path = payload.get("folder_path", "").strip() if payload else ""
    selected_files = payload.get("selected_files", []) if payload else []

    all_files = []
    if folder_path and selected_files:
        folder_path = folder_path.replace("\\", "/")
        try:
            real_folder = os.path.realpath(folder_path)
        except Exception:
            real_folder = folder_path
        if not os.path.isdir(real_folder):
            raise HTTPException(status_code=404, detail=f"Directory not found: {folder_path}")
        source_dir = os.path.basename(real_folder) or os.path.basename(os.path.dirname(real_folder))
        for fname in selected_files:
            fp = os.path.join(real_folder, fname)
            if os.path.isfile(fp):
                all_files.append({"name": fname, "path": fp, "source_dir": source_dir})

    if not all_files:
        raise HTTPException(status_code=404, detail="No audio files found")

    # Add per-file tracking fields
    for _f in all_files:
        _f.setdefault("status", "pending")
        _f.setdefault("progress", 0)
        _f.setdefault("current_step", "")

    job_id = uuid.uuid4().hex[:12]
    with _asr_compare_lock:
        _asr_compare_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "current_step": "",
            "total": len(all_files),
            "completed": 0,
            "flagged_count": 0,
            "files": all_files,
            "results": {},
            "is_segmented": True,
            "error": None,
            "created_at": time.time(),
            "_model_a": model_a,
            "_model_b": model_b,
            "_language": language,
            "_device": device,
            "_hotwords": hotwords,
            "_segment_min_s": segment_min_s,
            "_segment_max_s": segment_max_s,
        }
        _save_asr_compare_jobs()

    def _run():
        from asr_pipeline import segment_and_compare_pipeline, _get_asr_model, _CancelPipeline
        from asr_pipeline import ASR_MODELS

        def _cancel_check():
            with _asr_compare_lock:
                j = _asr_compare_jobs.get(job_id)
                return bool(j and j.get("_cancelled"))

        with _asr_compare_lock:
            job = _asr_compare_jobs.get(job_id)
            if not job:
                return
            job["status"] = "loading"
            job["current_step"] = "正在加载模型..."
            _save_asr_compare_jobs()

        # Preload both ASR models before starting processing.
        # This is critical because FunASR / ModelScope downloads can behave
        # unpredictably in threads. We run with daemon=False to mitigate this.
        try:
            for _mk in (model_a, model_b):
                with _asr_compare_lock:
                    job = _asr_compare_jobs.get(job_id)
                    if job:
                        job["current_step"] = f"正在加载模型: {_mk}"
                        _save_asr_compare_jobs()
                _get_asr_model(_mk, device=device, use_compile=False)
        except Exception as e:
            with _asr_compare_lock:
                job = _asr_compare_jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["error"] = f"模型加载失败: {e}"
                    job["current_step"] = f"模型加载失败: {e}"
                    _save_asr_compare_jobs()
            return

        with _asr_compare_lock:
            job = _asr_compare_jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["current_step"] = "模型加载完成，开始处理..."
            _save_asr_compare_jobs()

        files = all_files
        total = len(files)

        for idx, f in enumerate(files):
            with _asr_compare_lock:
                job = _asr_compare_jobs.get(job_id)
                if not job or job.get("_cancelled"):
                    if job:
                        job["status"] = "cancelled"
                        job["current_step"] = "已中止"
                        _save_asr_compare_jobs()
                    return
                f["status"] = "running"
                f["progress"] = 0
                f["current_step"] = "开始处理..."

            audio_path = f["path"]
            audio_name = f["name"]

            def on_progress(step, pct):
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["current_step"] = f"[{idx + 1}/{total}] {audio_name} — {step}"
                        j["progress"] = int(((idx + pct / 100) / total) * 100)
                    f["progress"] = int(pct)
                    f["current_step"] = step

            try:
                result = segment_and_compare_pipeline(
                    audio_path,
                    language=language,
                    device=device,
                    model_a=model_a,
                    model_b=model_b,
                    hotwords=hotwords,
                    segment_min_s=segment_min_s,
                    segment_max_s=segment_max_s,
                    progress_callback=on_progress,
                    source_dir=f["source_dir"],
                    cancel_check=_cancel_check,
                )

                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["completed"] = idx + 1
                        j["progress"] = int(((idx + 1) / total) * 100)
                        j["results"][audio_path] = result
                        flagged = sum(1 for s in result.get("segments", []) if s.get("flagged"))
                        j["flagged_count"] += flagged
                    f["status"] = "completed"
                    f["progress"] = 100

            except _CancelPipeline:
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["status"] = "cancelled"
                        j["current_step"] = "已中止"
                        _save_asr_compare_jobs()
                return
            except Exception as e:
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["completed"] = idx + 1
                        j["progress"] = int(((idx + 1) / total) * 100)
                        j["results"][audio_path] = {
                            "audio_path": audio_path,
                            "audio_name": audio_name,
                            "source_dir": f["source_dir"],
                            "error": str(e),
                            "segment_count": 0,
                            "segments": [],
                        }
                        j["flagged_count"] = j.get("flagged_count", 0) + 1
                    f["status"] = "error"
                    f["progress"] = 0
                    f["error"] = str(e)

        with _asr_compare_lock:
            j = _asr_compare_jobs.get(job_id)
            if j and j["status"] != "cancelled":
                j["status"] = "completed"
                j["progress"] = 100
                j["current_step"] = f"处理完成 — {total} 个文件, {j['flagged_count']} 个异常分段"
                _save_asr_compare_jobs()
                for path, result in j["results"].items():
                    _asr_compare_results[path] = result

    # Return response immediately so the frontend doesn't hang on "启动中...".
    # Model preloading and processing happen in a background thread.
    threading.Thread(target=_run, daemon=False).start()

    return {"job_id": job_id, "status": "started", "total": len(all_files)}


@app.post("/api/asr-compare/keep-segments")
def keep_asr_compare_segments(payload: dict = Body(...)):
    """Copy selected segments (audio + SRTs) to ASR_COMPARE_KEPT_DIR."""
    import shutil as _shutil
    wav_paths = payload.get("paths", [])
    if not wav_paths:
        raise HTTPException(status_code=400, detail="No paths provided")

    os.makedirs(ASR_COMPARE_KEPT_DIR, exist_ok=True)
    kept = []
    for wav_path in wav_paths:
        if not os.path.exists(wav_path):
            continue
        real_path = os.path.realpath(wav_path)
        real_data = os.path.realpath(DATA_DIR)
        if not real_path.startswith(real_data + os.sep) and real_path != real_data:
            continue

        # Copy segment WAV
        base = os.path.basename(wav_path)
        # Keep folder structure: KEPT_DIR/audio_name/seg_name.wav
        parent_dir = os.path.basename(os.path.dirname(wav_path))
        dest_dir = os.path.join(ASR_COMPARE_KEPT_DIR, parent_dir)
        os.makedirs(dest_dir, exist_ok=True)
        dest_wav = os.path.join(dest_dir, base)
        _shutil.copy2(wav_path, dest_wav)

        # Also copy associated SRT files from compare subtitle dirs
        seg_stem = os.path.splitext(base)[0]
        kept.append({"path": wav_path, "dest": dest_wav})

        # Look for SRT files matching this segment
        srt_base_dir = ASR_COMPARE_SUBTITLE_DIR
        if os.path.isdir(srt_base_dir):
            for root, dirs, files in os.walk(srt_base_dir):
                for fn in files:
                    if fn.startswith(seg_stem) and fn.endswith(".srt"):
                        src_srt = os.path.join(root, fn)
                        dest_srt = os.path.join(dest_dir, fn)
                        _shutil.copy2(src_srt, dest_srt)
                        kept.append({"path": src_srt, "dest": dest_srt})

    return {"status": "ok", "kept": len(kept), "dest_dir": ASR_COMPARE_KEPT_DIR}


@app.post("/api/asr-compare/delete-segments")
def delete_asr_compare_segments(payload: dict = Body(...)):
    """Permanently delete segment WAV files from the segments folder."""
    wav_paths = payload.get("paths", [])
    if not wav_paths:
        raise HTTPException(status_code=400, detail="No paths provided")

    real_data = os.path.normcase(os.path.realpath(DATA_DIR))
    deleted = []
    deleted_txt = []
    skipped_security = []
    skipped_missing = []
    affected_parents = set()
    for wav_path in wav_paths:
        real_path = os.path.normcase(os.path.realpath(wav_path))
        print(f"[delete-segments] wav_path={wav_path!r}")
        print(f"[delete-segments] real_data={real_data!r} real_path={real_path!r}")
        print(f"[delete-segments] exists={os.path.exists(wav_path)}")
        if not real_path.startswith(real_data + os.sep) and real_path != real_data:
            skipped_security.append({"path": wav_path, "real_path": real_path, "real_data": real_data})
            print(f"[delete-segments] SKIPPED: security check failed")
            continue

        # Find the segment in results to get associated TXT paths
        found = False
        for parent_audio_path, result in _asr_compare_results.items():
            segs = result.get("segments") or []
            for seg in segs:
                if seg.get("wav_path") == wav_path:
                    found = True
                    affected_parents.add(parent_audio_path)
                    # Delete TXT files
                    for txt_key in ("txt_path_a", "txt_path_b"):
                        txt_path = seg.get(txt_key)
                        if txt_path and os.path.exists(txt_path):
                            os.remove(txt_path)
                            deleted_txt.append(txt_path)
                    # Clear in-memory text data
                    seg["text_a"] = ""
                    seg["text_b"] = ""
                    seg["txt_path_a"] = None
                    seg["txt_path_b"] = None
                    seg["diff_chunks"] = []
                    break
            else:
                continue
            break
        print(f"[delete-segments] found_in_results={found}")

        if os.path.exists(wav_path):
            os.remove(wav_path)
            deleted.append(wav_path)
        else:
            skipped_missing.append(wav_path)

    for parent_path in affected_parents:
        _sync_user_action_to_jobs(parent_path)
    print(f"[delete-segments] deleted={deleted} skipped_security={skipped_security} skipped_missing={skipped_missing}")
    return {"status": "ok", "deleted": len(deleted), "paths": deleted, "_debug": {"skipped_security": skipped_security, "skipped_missing": skipped_missing}}


@app.post("/api/asr-compare/package")
def package_asr_compare_results(payload: dict = Body(...)):
    """Copy kept audio + SRTs to organized output folder grouped by source dir.

    Expects JSON body with optional 'results' list; if omitted, packages all kept results.
    """
    import shutil as _shutil

    results_to_package = payload.get("results")
    if results_to_package:
        results = results_to_package
    else:
        with _asr_compare_lock:
            results = list(_asr_compare_results.values())

    packaged = 0
    for r in results:
        if r.get("user_action") == "discarded":
            continue
        if r.get("error"):
            continue

        source_dir = r.get("source_dir", "unknown")
        audio_name = r.get("audio_name", "unknown")
        audio_path = r.get("audio_path", "")

        out_dir = os.path.join(ASR_COMPARE_OUTPUT_DIR, source_dir)
        out_sub_dir = os.path.join(out_dir, "subtitles")
        out_audio_dir = os.path.join(out_dir, "audio")
        os.makedirs(out_sub_dir, exist_ok=True)
        os.makedirs(out_audio_dir, exist_ok=True)

        # Copy audio
        if audio_path and os.path.exists(audio_path):
            _shutil.copy2(audio_path, os.path.join(out_audio_dir, os.path.basename(audio_path)))

        # Copy SRTs
        srt_paths = r.get("srt_paths", {})
        for model_key, srt_path in srt_paths.items():
            if os.path.exists(srt_path):
                _shutil.copy2(srt_path, os.path.join(out_sub_dir, os.path.basename(srt_path)))

        packaged += 1

    return {"status": "ok", "packaged": packaged, "output_dir": ASR_COMPARE_OUTPUT_DIR.replace("\\", "/")}


@app.get("/api/asr-compare/stream")
def stream_asr_compare_file(path: str = Query(..., description="Full path to SRT or audio file")):
    """Stream a comparison result file."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "audio/wav" if path.lower().endswith(".wav") else "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media_type)


# ============================================================
# Pipeline Video — 视频管线  (import, process, progress, stream)
# ============================================================

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}


def _get_media_info(file_path: str) -> dict:
    """Get duration and other info for a media file via ffprobe."""
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", file_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            return {
                "duration_s": float(fmt.get("duration", 0)),
                "size_mb": os.path.getsize(file_path) / 1024 / 1024,
            }
    except Exception:
        pass
    return {"duration_s": 0, "size_mb": os.path.getsize(file_path) / 1024 / 1024}


@app.get("/api/pipeline-video/scan")
def pv_scan_folder(folder_path: str = Query("", description="Full path to folder")):
    """Scan a folder for video and audio files."""
    if not folder_path or not os.path.isdir(folder_path):
        return {"error": "文件夹不存在", "files": [], "count": 0}

    return _scan_single_folder(folder_path)


@app.post("/api/pipeline-video/scan-folders")
def pv_scan_folders(payload: dict = Body(...)):
    """Scan multiple folders for video and audio files.

    Body: {folder_paths: [str, ...]}
    """
    folder_paths = payload.get("folder_paths", [])
    if not folder_paths:
        raise HTTPException(status_code=400, detail="No folder paths provided")

    all_files = []
    for fp in folder_paths:
        if os.path.isdir(fp):
            result = _scan_single_folder(fp)
            if result.get("files"):
                all_files.extend(result["files"])

    return {"files": all_files, "count": len(all_files), "folders_scanned": len(folder_paths)}


def _scan_single_folder(folder_path: str) -> dict:
    """Scan a single folder for video and audio files (parallel ffprobe)."""
    try:
        entries = sorted(os.listdir(folder_path))
    except PermissionError:
        return {"error": "没有权限访问该文件夹", "files": [], "count": 0}

    # Collect matching files first (no ffprobe yet)
    candidates = []
    for name in entries:
        full = os.path.join(folder_path, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in VIDEO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
            continue
        candidates.append({"name": name, "path": full, "ext": ext, "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1)})

    # Parallel ffprobe for duration (up to 4 concurrent)
    import concurrent.futures as _futures

    def _probe_duration(c):
        info = _get_media_info(c["path"])
        c["duration_s"] = round(info.get("duration_s", 0), 1)
        return c

    with _futures.ThreadPoolExecutor(max_workers=4) as _pool:
        files = list(_pool.map(_probe_duration, candidates))

    for f in files:
        f["folder"] = folder_path
    return {"folder_path": folder_path, "files": files, "count": len(files)}


@app.get("/api/pipeline-video/browse-folder")
def pv_browse_folder():
    """Open a native folder picker dialog and return the selected path."""
    if not _display_available():
        return {"path": "", "error": "Server has no display — use manual path input instead."}
    import subprocess as _subprocess
    import tempfile as _tempfile
    import sys as _sys

    result_file = _tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    result_path = result_file.name
    result_file.close()

    if os.path.isdir(PIPELINE_VIDEO_DIR):
        default_dir = PIPELINE_VIDEO_DIR
    elif os.path.isdir(DATA_DIR):
        default_dir = DATA_DIR
    else:
        default_dir = os.path.expanduser("~")

    picker_code = f'''
import tkinter as tk
from tkinter import filedialog
import sys, os, ctypes
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.lift()
root.focus_force()
root.update()
hwnd = root.winfo_id() if root.winfo_exists() else 0
if hwnd:
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.BringWindowToTop(hwnd)
try:
    path = filedialog.askdirectory(title="选择包含视频/音频文件的文件夹", initialdir=r"{default_dir}")
    if path:
        with open(r"{result_path}", "w", encoding="utf-8") as f:
            f.write(path)
except Exception:
    pass
try:
    root.destroy()
except Exception:
    pass
'''

    picker_file = _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    picker_file.write(picker_code)
    picker_path = picker_file.name
    picker_file.close()

    try:
        _subprocess.run([_sys.executable, picker_path], timeout=300)
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                path = f.read().strip()
            return {"path": path}
        return {"path": ""}
    except _subprocess.TimeoutExpired:
        return {"path": "", "error": "操作超时"}
    except Exception as e:
        return {"path": "", "error": str(e)}
    finally:
        for f in (picker_path, result_path):
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass


@app.post("/api/pipeline-video/start")
def pv_start_pipeline(payload: dict = Body(...)):
    """Start a video pipeline job.

    Body: {folder_path: str, file_paths: [str, ...]}
    """
    folder_path = payload.get("folder_path", "")
    file_paths = payload.get("file_paths", [])

    if not file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided")

    valid_paths = [p for p in file_paths if os.path.exists(p)]
    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid files found")

    folder_name = os.path.basename(folder_path) or "unknown"
    job_id = uuid.uuid4().hex[:12]

    files = [PipelineVideoFileItem(
        name=os.path.basename(p),
        input_path=p,
    ) for p in valid_paths]

    job = PipelineVideoJob(job_id=job_id, folder_name=folder_name, files=files)
    job.steps_config = payload.get("steps", None)
    job.output_dir = payload.get("output_dir", "") or PIPELINE_VIDEO_DIR
    with _pipeline_video_lock:
        _pipeline_video_jobs[job_id] = job
        _save_pv_jobs()

    threading.Thread(target=_run_pipeline_video_safe, args=(job_id,), daemon=True).start()

    return {"status": "ok", "job_id": job_id, "file_count": len(valid_paths)}


@app.get("/api/pipeline-video/job/{job_id}")
def pv_get_job(job_id: str):
    """Poll pipeline video job status."""
    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/api/pipeline-video/jobs")
def pv_list_jobs():
    """List recent pipeline video jobs (latest 20)."""
    with _pipeline_video_lock:
        jobs = sorted(_pipeline_video_jobs.values(), key=lambda j: j.created_at, reverse=True)[:20]
    return {"jobs": [j.to_dict() for j in jobs], "count": len(jobs)}


@app.get("/api/pipeline-video/stream")
def pv_stream_file(path: str = Query(..., description="Full path to media file")):
    """Stream a video or audio file for HTML5 playback."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    ext = path.lower().split(".")[-1] if "." in path else ""
    mime_map = {
        "mp4": "video/mp4", "mkv": "video/x-matroska", "webm": "video/webm",
        "avi": "video/x-msvideo", "mov": "video/quicktime", "wmv": "video/x-ms-wmv",
        "flv": "video/x-flv", "wav": "audio/wav", "mp3": "audio/mpeg",
        "flac": "audio/flac", "aac": "audio/aac", "ogg": "audio/ogg", "m4a": "audio/mp4",
    }
    media_type = mime_map.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type)


@app.post("/api/pipeline-video/job/{job_id}/cancel")
def pv_cancel_job(job_id: str):
    """Cancel a running pipeline video job."""
    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancelled = True
    job.status = "cancelled"
    _update_pipeline_job_progress(job)
    _save_pv_jobs()
    return {"status": "ok", "message": "已取消任务"}


@app.post("/api/pipeline-video/job/{job_id}/discard")
def pv_discard_job(job_id: str):
    """Discard an interrupted pipeline video job."""
    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "interrupted":
        raise HTTPException(status_code=400, detail=f"Job is {job.status}, not interrupted")
    job.status = "cancelled"
    _update_pipeline_job_progress(job)
    _save_pv_jobs()
    return {"status": "ok", "message": "已丢弃中断的任务"}


@app.post("/api/pipeline-video/job/{job_id}/resume")
def pv_resume_job(job_id: str):
    """Resume an interrupted or pending pipeline video job, skipping completed files."""
    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("interrupted", "pending"):
        raise HTTPException(status_code=400, detail=f"Job is {job.status}, not interrupted/pending")
    job.status = "running"
    job.cancelled = False
    _update_pipeline_job_progress(job)
    _save_pv_jobs()
    threading.Thread(target=_run_pipeline_video_resume_safe, args=(job_id,), daemon=True).start()
    return {"status": "ok", "message": "已恢复执行"}


@app.delete("/api/pipeline-video/job/{job_id}")
def pv_delete_job(job_id: str):
    """Delete a pipeline video job. Cancels if running, then removes data."""
    import shutil
    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Cancel if running
    if job.status == "running":
        job.cancelled = True
        job.status = "cancelled"
    # Remove from dict and persist
    with _pipeline_video_lock:
        _pipeline_video_jobs.pop(job_id, None)
        _save_pv_jobs()
    # Remove associated temp directory
    temp_dir = os.path.join(TEMP_DIR, "pipeline_video", job_id)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    return {"status": "ok", "message": "任务已删除"}


# ============================================================
# Pipeline Video — background processing
# ============================================================

def _convert_to_wav_ffmpeg(input_path: str, sample_rate: int = 32000, channels: int = 1) -> str:
    """Convert an audio file (AAC/MP3/FLAC/OGG/M4A etc.) to WAV format using ffmpeg.

    Args:
        input_path: Path to source audio file.
        sample_rate: Output sample rate in Hz.
        channels: Number of output channels.
    """
    wav_path = os.path.splitext(input_path)[0] + ".wav"
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path
    try:
        result = subprocess.run(
            [FFMPEG, "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", str(sample_rate), "-ac", str(channels), wav_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            return wav_path
        if result.stderr:
            _log.warning("ffmpeg 转换异常 %s: %s", os.path.basename(input_path), result.stderr[:200])
    except Exception as e:
        _log.error("ffmpeg 转换失败 %s: %s", os.path.basename(input_path), e)
    return ""


def _get_optimal_wav_params(enabled_steps: list) -> tuple:
    """Choose WAV sample rate and channels. All steps use 32kHz mono."""
    return 32000, 1


def _run_pipeline_video_resume(job_id: str):
    """Resume an interrupted pipeline video job — skip completed files."""
    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        return

    completed = [f for f in job.files if f.status == "completed"]
    pending = [f for f in job.files if f.status != "completed"]

    if not pending:
        job.status = "completed"
        job.progress = 100
        _update_pipeline_job_progress(job)
        _save_pv_jobs()
        return

    for f in pending:
        f.status = "pending"
        f.progress = 0
        f.current_step = ""
        f.steps = []
        f.output_clips = []
        f.wav_path = ""
        f.error = ""

    job.files = pending
    job.status = "running"
    _save_pv_jobs()

    _run_pipeline_video_safe(job_id)

    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
        if job and job.status != "error":
            job.files = completed + job.files
            _update_pipeline_job_progress(job)
            _save_pv_jobs()


def _run_pipeline_video_resume_safe(job_id: str):
    """Thread-safe wrapper for resume that catches any unhandled exception."""
    try:
        _run_pipeline_video_resume(job_id)
    except Exception:
        import logging as _logging, sys as _sys
        _log = _logging.getLogger("pipeline-video")
        if not _log.handlers:
            _h = _logging.StreamHandler(_sys.stdout)
            _h.setFormatter(_logging.Formatter(
                "[pipeline-video] %(asctime)s %(message)s", datefmt="%H:%M:%S"
            ))
            _log.addHandler(_h)
        _log.exception("管线恢复异常崩溃 — 后台线程")
        with _pipeline_video_lock:
            job = _pipeline_video_jobs.get(job_id)
            if job and job.status not in ("completed", "cancelled", "error"):
                job.status = "error"
                for f in job.files:
                    if f.status not in ("completed", "skipped", "error"):
                        f.status = "error"
                        if not f.error:
                            f.error = "管线内部异常"
                        f.steps.append({"step": "fatal", "status": "error",
                                        "message": "管线恢复线程异常崩溃，请查看服务器日志"})
                _update_pipeline_job_progress(job)
                _save_pv_jobs()


def _run_pipeline_video(job_id: str):
    """Run the video pipeline in background.

    Steps are configurable via job.steps_config.
    Each file flows through all enabled steps independently.
    Per-step BoundedSemaphore(2) controls model concurrency.
    """
    import logging as _logging, sys as _sys
    _log = _logging.getLogger("pipeline-video")
    _log.setLevel(_logging.INFO)
    if not _log.handlers:
        _h = _logging.StreamHandler(_sys.stdout)
        _h.setFormatter(_logging.Formatter(
            "[pipeline-video] %(asctime)s %(message)s", datefmt="%H:%M:%S"
        ))
        _log.addHandler(_h)

    t0 = time.time()

    import concurrent.futures
    import shutil as _shutil

    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        return

    job.status = "running"
    files = job.files
    folder = job.folder_name

    _log.info("=== 任务开始: %s | 文件夹: %s | 文件数: %d ===", job_id[:8], folder, len(files))

    # Parse enabled steps from config (or default)
    steps_config = getattr(job, 'steps_config', None)
    if not steps_config:
        steps_config = [
            {"key": "duration_split", "enabled": True},
            {"key": "convert", "enabled": True},
            {"key": "music_separate", "enabled": True},
            {"key": "enhance", "enabled": True},
            {"key": "super_resolve", "enabled": False},
            {"key": "asr", "enabled": False},
            {"key": "cut", "enabled": False},
        ]
    enabled_steps = [s["key"] for s in steps_config if s.get("enabled", True)]
    step_cfg_map = {s["key"]: s for s in steps_config}

    _log.info("步骤: %s", " → ".join(enabled_steps))

    # GPU sanity check — fail fast if any GPU step is enabled but CUDA is unavailable
    _GPU_STEPS = {"music_separate", "enhance", "super_resolve", "asr"}
    if any(s in _GPU_STEPS for s in enabled_steps):
        import torch
        if not torch.cuda.is_available():
            gpu_steps = [s for s in enabled_steps if s in _GPU_STEPS]
            for f in files:
                f.status = "error"
                f.error = f"GPU步骤 ({', '.join(gpu_steps)}) 需要CUDA GPU，但当前环境不可用"
                f.steps.append({"step": "init", "status": "error",
                                "message": f"CUDA不可用，无法执行GPU步骤: {', '.join(gpu_steps)}"})
            _update_pipeline_job_progress(job)
            job.status = "error"
            _save_pv_jobs()
            return

    # Job-specific temp directory for intermediate files
    temp_dir = os.path.join(TEMP_DIR, "pipeline_video", job_id)
    os.makedirs(temp_dir, exist_ok=True)
    _log.info("临时目录: %s", temp_dir)

    # Final output dir — created eagerly so per-file flush works from the start
    output_base = getattr(job, 'output_dir', PIPELINE_VIDEO_DIR) or PIPELINE_VIDEO_DIR
    final_dir = os.path.join(output_base, folder)
    os.makedirs(final_dir, exist_ok=True)

    def _publish_completed_file(f, state=None):
        """Copy outputs to per-file folder structure, then clean this file's temp data.

        Only runs when ALL segments of this file have completed.
        Regex ^{base}(_seg\\d{{3}})?_.+$ matches only this exact file's temp
        artifacts — it won't match "歌 2" when cleaning "歌".

        Output layout under final_dir/{base_name}/:
          enhanced/   ← final enhanced WAV
          subtitles/  ← ASR SRT
          clips/      ← cut audio segments
        """
        if f.status not in ("completed", "skipped"):
            return

        base_name = os.path.splitext(f.name)[0]
        file_out = os.path.join(final_dir, base_name)
        enhanced_dir = os.path.join(file_out, "enhanced")
        subtitles_dir = os.path.join(file_out, "subtitles")
        clips_dir = os.path.join(file_out, "clips")
        _log.info("[%s] 发布到: %s", f.name, file_out)

        def _safe_copy(src, dst_dir, dst_name):
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, dst_name)
            try:
                if os.path.abspath(src) != os.path.abspath(dst):
                    _shutil.copy2(src, dst)
            except Exception:
                pass
            return dst

        # 1. Enhanced WAV — skip if already published by _step_asr
        enhanced_dst = os.path.join(enhanced_dir, base_name + ".wav")
        if not os.path.exists(enhanced_dst):
            wav_src = (state or {}).get("wav", "")
            if wav_src and os.path.exists(wav_src):
                _safe_copy(wav_src, enhanced_dir, base_name + ".wav")
                _log.info("[%s] 增强音频 → enhanced/", f.name)

        # 2. SRT subtitles — skip if already published by _step_asr
        srt_dst = os.path.join(subtitles_dir, base_name + ".srt")
        if not os.path.exists(srt_dst):
            srt_src = (state or {}).get("srt", "")
            if srt_src and os.path.exists(srt_src):
                _safe_copy(srt_src, subtitles_dir, base_name + ".srt")
                _log.info("[%s] 字幕 → subtitles/", f.name)

        # 3. Collect all output paths for frontend reference
        # (enhanced WAV and SRT already published by _step_asr,
        #  cut clips already placed by _step_cut into clips_dir)
        if os.path.isdir(clips_dir):
            for cf in os.listdir(clips_dir):
                cp = os.path.join(clips_dir, cf)
                if cp not in f.output_clips:
                    f.output_clips.append(cp)
        if os.path.exists(enhanced_dst) and enhanced_dst not in f.output_clips:
            f.output_clips.append(enhanced_dst)
        if os.path.exists(srt_dst) and srt_dst not in f.output_clips:
            f.output_clips.append(srt_dst)

        # --- Clean this file's temp data ---
        # Only clean if the file actually produced outputs. If processing was
        # incomplete (e.g. all segments timed out), preserve temp files.
        has_output = os.path.exists(enhanced_dst) or os.path.exists(srt_dst) or \
                     (os.path.isdir(clips_dir) and len(os.listdir(clips_dir)) > 0)
        if has_output:
            import re as _re
            base_rgx_str = _re.escape(base_name)
            _file_rgx = _re.compile(r'^' + base_rgx_str + r'(_seg\d{3})?_.+$')
            seg_dir = "segments_" + base_name
            sep_dir = "separated_" + base_name
            cleaned = []
            if os.path.isdir(temp_dir):
                for fn in os.listdir(temp_dir):
                    fp = os.path.join(temp_dir, fn)
                    if _file_rgx.match(fn) or fn in (seg_dir, sep_dir):
                        try:
                            if os.path.isfile(fp):
                                os.remove(fp)
                                cleaned.append(fn)
                            elif os.path.isdir(fp):
                                _shutil.rmtree(fp, ignore_errors=True)
                                cleaned.append(fn + "/")
                        except OSError:
                            pass
            if cleaned:
                _log.info("[%s] 清理临时文件: %d 个", f.name, len(cleaned))

    # Per-step semaphores — tuned for RTX 4090 (24 GB).
    #
    # GPU budget (23.5 GB total, ~2 GB reserved by desktop/CUDA context):
    #   Demucs HT        ≈ 0.5 GB   (model) + 1.0 GB (working set)
    #   ClearVoice SE     ≈ 0.3 GB   (model) + 0.5 GB (working set)
    #   ClearVoice SR     ≈ 0.3 GB   (model) + 0.5 GB (working set)
    #   Qwen3-ASR-1.7B   ≈ 3.4 GB   (model: bf16) + 2.0 GB (working set)
    #   Concurrent peak:  (0.5+1.0) + (0.3+0.5) + (3.4+2.0) = 7.7 GB
    #   Headroom:         23.5 - 7.7 = 15.8 GB (safe)
    #
    # "gpu" semaphore limits TOTAL concurrent GPU operations across all steps.
    # Non-GPU steps (split / convert / cut) do NOT acquire the gpu sem.
    #
    # Pipeline strategy: each GPU step uses a DIFFERENT singleton model, so they
    # can overlap safely.  File A can run music_separate while File B runs enhance
    # and File C runs asr — 3x GPU utilization vs the old 1-at-a-time config.
    #
    # Per-step semaphores are set to 1 because each model is a singleton (not
    # reentrant-safe for concurrent inference).  Increasing "gpu" to 3 enables
    # cross-file pipelining: as File A advances from music_separate → enhance,
    # File B takes over the music_separate slot, keeping all models busy.
    _sem = {
        "gpu":      threading.BoundedSemaphore(4),  # global GPU cap: 2 MS + 1 EN + 1 ASR
        "ffmpeg":   threading.BoundedSemaphore(8),  # global ffmpeg cap
        "convert":  threading.BoundedSemaphore(6),
        "music_separate": threading.BoundedSemaphore(2),  # 2x Demucs instances (~3 GB model + working mem)
        "enhance":  threading.BoundedSemaphore(1),  # singleton ClearVoice SE
        "super_resolve": threading.BoundedSemaphore(1),  # singleton ClearVoice SR
        "asr":      threading.BoundedSemaphore(1),  # singleton Qwen3-ASR (3.4 GB)
        "duration_split": threading.BoundedSemaphore(6),  # ffmpeg segment split (I/O only)
        "cut":      threading.BoundedSemaphore(3),
    }

    file_state = {}

    def _temp_path(f, step_key, ext=".wav", state=None):
        base = os.path.splitext(f.name)[0]
        suffix = state.get("seg_suffix", "") if state else ""
        return os.path.join(temp_dir, f"{base}{suffix}_{step_key}{ext}")

    def _step_convert(f, job, state):
        from convert_audio import mp4_to_wav
        f.current_step = "convert"
        f.status = "converting"
        f.progress = 0
        _update_pipeline_job_progress(job)

        out = _temp_path(f, "convert")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            state["wav"] = out
            f.status = "running"
            f.progress = 100
            f.steps.append({"step": "convert", "status": "completed", "message": "WAV已存在"})
            _update_pipeline_job_progress(job)
            return True

        # Pick optimal WAV params based on downstream steps
        opt_sr, opt_ch = _get_optimal_wav_params(enabled_steps)

        ext = os.path.splitext(f.input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            wav_path = mp4_to_wav(f.input_path, out, sample_rate=opt_sr, channels=opt_ch)
            if wav_path and os.path.exists(wav_path):
                state["wav"] = wav_path
                f.status = "running"
                f.progress = 100
                f.steps.append({"step": "convert", "status": "completed",
                                "message": f"视频转WAV完成 ({opt_sr}Hz)"})
                _update_pipeline_job_progress(job)
                return True
        elif ext == ".wav":
            # Resample to target rate/channels even if already WAV
            result = subprocess.run(
                [FFMPEG, "-y", "-i", f.input_path, "-vn",
                 "-acodec", "pcm_s16le",
                 "-ar", str(opt_sr), "-ac", str(opt_ch), out],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=300,
            )
            if os.path.exists(out) and os.path.getsize(out) > 0:
                state["wav"] = out
                f.status = "running"
                f.progress = 100
                f.steps.append({"step": "convert", "status": "completed",
                                "message": f"WAV统一 ({opt_sr}Hz/{'单声道' if opt_ch == 1 else '立体声'})"})
                _update_pipeline_job_progress(job)
                return True
        else:
            wav_path = _convert_to_wav_ffmpeg(f.input_path, sample_rate=opt_sr, channels=opt_ch)
            if wav_path:
                if wav_path != out:
                    _shutil.copy2(wav_path, out)
                state["wav"] = out
                f.progress = 100
                f.steps.append({"step": "convert", "status": "completed",
                                "message": f"音频转WAV完成 ({opt_sr}Hz)"})
                _update_pipeline_job_progress(job)
                return True

        f.status = "error"
        f.error = "WAV转换失败"
        f.steps.append({"step": "convert", "status": "error", "message": "转换失败"})
        _update_pipeline_job_progress(job)
        return False

    def _step_music_separate(f, job, state):
        """BGM separation: Demucs HT → keep vocals only, delete instrumentals."""
        from music_separate import separate_vocals
        f.current_step = "music_separate"

        src = state.get("wav", f.wav_path)
        out = _temp_path(f, "separated", ".wav", state)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            state["wav"] = out
            f.steps.append({"step": "music_separate", "status": "completed",
                            "message": "人声分离已完成 (复用缓存)"})
            _update_pipeline_job_progress(job)
            return True

        f.steps.append({"step": "music_separate", "status": "running", "message": "加载 Demucs 模型..."})
        _update_pipeline_job_progress(job)
        seg_dir = os.path.join(temp_dir, f"separated_{os.path.splitext(f.name)[0]}")
        os.makedirs(seg_dir, exist_ok=True)

        try:
            vocals_path = separate_vocals(src, seg_dir)
            if vocals_path and os.path.exists(vocals_path):
                if vocals_path != out:
                    _shutil.copy2(vocals_path, out)
                state["wav"] = out
                f.steps.append({"step": "music_separate", "status": "completed",
                                "message": "人声分离完成 (BGM/鼓/贝斯已删除)"})
            else:
                if src != out and os.path.exists(src):
                    _shutil.copy2(src, out)
                state["wav"] = out
                f.steps.append({"step": "music_separate", "status": "passed",
                                "message": "分离未产出人声，使用原始音频"})
        except Exception as e:
            f.steps.append({"step": "music_separate", "status": "error", "message": str(e)[:100]})
        _update_pipeline_job_progress(job)
        return True

    def _step_enhance(f, job, state):
        from denoise_audio import run_full_denoise
        f.current_step = "enhance"

        src = state.get("wav", f.wav_path)
        out = _temp_path(f, "enhance", ".wav", state)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            state["wav"] = out
            f.steps.append({"step": "enhance", "status": "completed",
                            "message": "语音增强已完成 (复用缓存)"})
            _update_pipeline_job_progress(job)
            return True

        f.steps.append({"step": "enhance", "status": "running", "message": "加载模型中..."})
        _update_pipeline_job_progress(job)

        def on_step(step_key, status, message):
            f.steps.append({"step": step_key, "status": status, "message": message})

        try:
            result = run_full_denoise(src, temp_dir, on_step=on_step, steps=["enhance"])
            if result.get("success") and result.get("output_path") and os.path.exists(result["output_path"]):
                out_path = result["output_path"]
                if out_path != out:
                    _shutil.copy2(out_path, out)
                state["wav"] = out
                f.steps.append({"step": "enhance", "status": "completed", "message": "语音增强完成"})
            else:
                reason = result.get("discard_reason", "")
                if src != out:
                    _shutil.copy2(src, out)
                state["wav"] = out
                f.steps.append({"step": "enhance", "status": "passed", "message": f"增强跳过: {reason}"})
        except Exception as e:
            f.steps.append({"step": "enhance", "status": "error", "message": str(e)[:100]})
        _update_pipeline_job_progress(job)
        return True

    def _step_super_resolve(f, job, state):
        from denoise_audio import run_full_denoise
        f.current_step = "super_resolve"

        src = state.get("wav", f.wav_path)
        out = _temp_path(f, "super_resolve", ".wav", state)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            state["wav"] = out
            f.progress = 100
            f.steps.append({"step": "super_resolve", "status": "completed",
                            "message": "超分辨率已完成 (复用缓存)"})
            _update_pipeline_job_progress(job)
            return True

        f.progress = 5
        f.steps.append({"step": "super_resolve", "status": "running", "message": "加载超分模型..."})
        _update_pipeline_job_progress(job)

        def on_step(step_key, status, message):
            f.steps.append({"step": step_key, "status": status, "message": message})

        try:
            f.progress = 20
            _update_pipeline_job_progress(job)
            result = run_full_denoise(src, temp_dir, on_step=on_step, steps=["super_resolve"])
            if result.get("success") and result.get("output_path") and os.path.exists(result["output_path"]):
                out_path = result["output_path"]
                if out_path != out:
                    _shutil.copy2(out_path, out)
                state["wav"] = out
                f.progress = 100
                f.steps.append({"step": "super_resolve", "status": "completed", "message": "超分辨率完成"})
            else:
                if src != out:
                    _shutil.copy2(src, out)
                state["wav"] = out
                f.steps.append({"step": "super_resolve", "status": "passed", "message": "超分跳过"})
                f.progress = 100
        except Exception as e:
            f.steps.append({"step": "super_resolve", "status": "error", "message": str(e)[:100]})
            f.progress = 100
        _update_pipeline_job_progress(job)
        return True

    def _step_asr(f, job, state):
        from asr_pipeline import run_asr_on_audio
        f.current_step = "asr"

        base_name = os.path.splitext(f.name)[0]
        file_out = os.path.join(final_dir, base_name)
        enhanced_dir = os.path.join(file_out, "enhanced")
        subtitles_dir = os.path.join(file_out, "subtitles")
        os.makedirs(enhanced_dir, exist_ok=True)
        os.makedirs(subtitles_dir, exist_ok=True)

        enhanced_wav = os.path.join(enhanced_dir, base_name + ".wav")
        srt_final = os.path.join(subtitles_dir, base_name + ".srt")

        # Cache check: if SRT already in final dir, reuse
        if os.path.exists(srt_final) and os.path.getsize(srt_final) > 0:
            state["srt"] = srt_final
            f.progress = 100
            f.steps.append({"step": "asr", "status": "completed",
                            "message": "ASR已完成 (复用缓存)"})
            _update_pipeline_job_progress(job)
            return True

        f.progress = 0
        _update_pipeline_job_progress(job)

        # Publish enhanced WAV to pipelinevideo for ASR input
        src = state.get("wav", f.wav_path)
        if src and os.path.exists(src) and not os.path.exists(enhanced_wav):
            try:
                _shutil.copy2(src, enhanced_wav)
            except Exception:
                pass
        asr_input = enhanced_wav if os.path.exists(enhanced_wav) else src

        def on_progress(phase, msg):
            if isinstance(msg, str):
                f.steps.append({"step": f"asr_{phase}", "status": "running", "message": str(msg)[:80]})
            else:
                f.progress = float(msg) if isinstance(msg, (int, float)) else f.progress

        try:
            # Read ASR configuration from step config
            asr_cfg = step_cfg_map.get("asr", {})
            asr_opts = asr_cfg.get("config", {}) or {}
            model_key = asr_opts.get("model", "qwen3-asr")
            language = asr_opts.get("language", "zh")

            # Resolve hotwords
            hw_cfg_name = asr_opts.get("hotword_config", "")
            _hotwords = ""
            if hw_cfg_name:
                hw_path = os.path.join(HOTWORDS_DIR, hw_cfg_name + ".txt")
                if os.path.exists(hw_path):
                    with open(hw_path, "r", encoding="utf-8") as _hf:
                        _hotwords = _hf.read().strip()
            if not _hotwords:
                from config import ASR_DEFAULT_HOTWORDS as _hw_default
                _hotwords = _hw_default
            result = run_asr_on_audio(
                asr_input, output_dir=subtitles_dir, model_key=model_key, language=language,
                progress_callback=on_progress, hotwords=_hotwords,
            )
            srt = result.get("srt_path", "")
            if srt and os.path.exists(srt):
                # Move to final subtitles path if ASR put it elsewhere
                if os.path.abspath(srt) != os.path.abspath(srt_final):
                    try:
                        _shutil.move(srt, srt_final)
                    except Exception:
                        try:
                            _shutil.copy2(srt, srt_final)
                        except Exception:
                            srt_final = srt
                state["srt"] = srt_final
                f.progress = 100
                f.steps.append({"step": "asr", "status": "completed",
                                "message": f"ASR完成: {result.get('segments_count', 0)}条字幕"})
            else:
                f.steps.append({"step": "asr", "status": "error", "message": "ASR未生成字幕"})
        except Exception as e:
            f.steps.append({"step": "asr", "status": "error", "message": str(e)[:100]})
        _update_pipeline_job_progress(job)
        return True

    def _step_duration_split(f, job, state):
        """Split source file directly into N-minute segments.

        Strategy (optimized to skip full-file WAV conversion):
        - WAV source: split with ``-c copy`` (instant, no re-encode).
        - Audio source (AAC/MP3/etc.): split with ``-c copy`` into original-codec
          segments, which are then converted to WAV in parallel by the caller.
        - Video source (MP4/MKV): extract audio + convert to WAV + split in one
          ffmpeg pass, writing WAV segments directly.

        First and last segments are always discarded (opening/closing noise).
        Results stored in ``state['duration_segments']``.
        """
        f.current_step = "duration_split"
        f.progress = 0
        _update_pipeline_job_progress(job)

        # Use the ORIGINAL source file, not a pre-converted WAV
        src = f.input_path
        if not src or not os.path.exists(src):
            f.steps.append({"step": "duration_split", "status": "error", "message": "源文件不存在"})
            _update_pipeline_job_progress(job)
            return True

        cfg = step_cfg_map.get("duration_split", {}).get("config", {})
        segment_dur = float(cfg.get("segment_duration", 600))
        keep_ends = cfg.get("keep_ends", False)

        base = os.path.splitext(f.name)[0]
        seg_dir = os.path.join(temp_dir, f"segments_{base}")
        os.makedirs(seg_dir, exist_ok=True)

        # Resume: check if segments already exist from a previous run
        existing_segs = sorted([
            os.path.join(seg_dir, x) for x in os.listdir(seg_dir)
            if x.endswith(('.wav', '.aac', '.mp3', '.flac', '.ogg'))
            and os.path.getsize(os.path.join(seg_dir, x)) > 1024
        ])
        if existing_segs and len(existing_segs) >= 2:
            # Already split — check if first/last need trimming (only if > 2 segs and previously trimmed)
            state["duration_segments"] = existing_segs
            state["seg_dir"] = seg_dir
            # Determine ext from existing files
            state["seg_ext"] = os.path.splitext(existing_segs[0])[1] or ".wav"
            state["opt_sr"], state["opt_ch"] = _get_optimal_wav_params(enabled_steps)
            f.progress = 100
            f.steps.append({"step": "duration_split", "status": "completed",
                            "message": f"时长切分已完成 (复用缓存): {len(existing_segs)}段"})
            _update_pipeline_job_progress(job)
            return True

        ext = os.path.splitext(src)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS

        # Pick optimal WAV params for downstream steps
        opt_sr, opt_ch = _get_optimal_wav_params(enabled_steps)

        if ext == ".wav":
            # Already WAV — split and resample to unified format
            seg_pattern = os.path.join(seg_dir, f"{base}_%03d.wav")
            cmd = [
                FFMPEG, "-y", "-i", src,
                "-f", "segment", "-segment_time", str(segment_dur),
                "-acodec", "pcm_s16le",
                "-ar", str(opt_sr), "-ac", str(opt_ch),
                seg_pattern,
            ]
            seg_ext = ".wav"
        elif is_video:
            # Video: extract audio + convert to WAV + split in one pass
            seg_pattern = os.path.join(seg_dir, f"{base}_%03d.wav")
            cmd = [
                FFMPEG, "-y", "-i", src, "-vn",
                "-f", "segment", "-segment_time", str(segment_dur),
                "-acodec", "pcm_s16le",
                "-ar", str(opt_sr), "-ac", str(opt_ch),
                seg_pattern,
            ]
            seg_ext = ".wav"
        else:
            # Audio (AAC/MP3/FLAC etc.): split with -c copy (no re-encode),
            # keeping original codec. Segments converted to WAV later in parallel.
            seg_ext = ext if ext else ".aac"
            seg_pattern = os.path.join(seg_dir, f"{base}_%03d{seg_ext}")
            cmd = [
                FFMPEG, "-y", "-i", src,
                "-f", "segment", "-segment_time", str(segment_dur),
                "-c", "copy", seg_pattern,
            ]

        try:
            subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
            # Collect generated segment files
            segments = sorted([
                os.path.join(seg_dir, x) for x in os.listdir(seg_dir)
                if x.endswith(seg_ext) and os.path.getsize(os.path.join(seg_dir, x)) > 0
            ])
            if len(segments) >= 3 and not keep_ends:
                removed_first = segments.pop(0)
                removed_last = segments.pop(-1)
                os.remove(removed_first)
                os.remove(removed_last)
                _log.info("[%s] 去除首尾分段: 原始%d段 → %d段",
                          f.name, len(segments) + 2, len(segments))
            if segments:
                state["duration_segments"] = segments
                state["seg_dir"] = seg_dir
                state["seg_ext"] = seg_ext  # track whether conversion needed
                state["opt_sr"] = opt_sr
                state["opt_ch"] = opt_ch
                f.progress = 100
                _msg = f"时长切分: {len(segments)}段 (每段{segment_dur}秒"
                _msg += "，保留首尾)" if keep_ends else "，已去除首尾)"
                f.steps.append({"step": "duration_split", "status": "completed",
                                "message": _msg})
            else:
                f.steps.append({"step": "duration_split", "status": "error", "message": "切分未产生有效片段"})
        except Exception as e:
            f.steps.append({"step": "duration_split", "status": "error", "message": str(e)[:100]})
        _update_pipeline_job_progress(job)
        return True

    def _step_cut(f, job, state):
        f.current_step = "cut"
        f.progress = 0
        _update_pipeline_job_progress(job)

        srt = state.get("srt", "")
        if not srt:
            f.steps.append({"step": "cut", "status": "skipped", "message": "无字幕，跳过切割"})
            _update_pipeline_job_progress(job)
            return True

        base_name = os.path.splitext(f.name)[0]
        file_out = os.path.join(final_dir, base_name)
        enhanced_dir = os.path.join(file_out, "enhanced")
        clips_dir = os.path.join(file_out, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        # Read enhanced WAV from pipelinevideo if available, fall back to temp
        enhanced_wav = os.path.join(enhanced_dir, base_name + ".wav")
        src = enhanced_wav if os.path.exists(enhanced_wav) else state.get("wav", f.wav_path)

        # Resume: check if cut clips already exist in final dir
        existing_cuts = [os.path.join(clips_dir, x) for x in os.listdir(clips_dir)
                        if x.endswith('.wav') and os.path.getsize(os.path.join(clips_dir, x)) > 0]
        if existing_cuts:
            f.output_clips.extend(existing_cuts)
            f.progress = 100
            f.steps.append({"step": "cut", "status": "completed",
                            "message": f"切割已完成 (复用缓存): {len(existing_cuts)}个片段"})
            _update_pipeline_job_progress(job)
            return True

        def on_progress(cur, total):
            if total > 0:
                f.progress = (cur / total) * 100

        try:
            clips = cut_audio_by_subtitle(
                audio_path=src, srt_path=srt, output_dir=clips_dir,
                base_name=base_name,
                on_progress=on_progress,
            )
            f.output_clips.extend(clips)
            f.progress = 100
            f.steps.append({"step": "cut", "status": "completed",
                            "message": f"切割完成: {len(clips)}个片段"})
        except Exception as e:
            f.steps.append({"step": "cut", "status": "error", "message": str(e)[:100]})
        _update_pipeline_job_progress(job)
        return True

    STEP_HANDLERS = {
        "music_separate": _step_music_separate,
        "enhance": _step_enhance,
        "super_resolve": _step_super_resolve,
        "asr": _step_asr,
        "duration_split": _step_duration_split,
        "cut": _step_cut,
    }
    # Steps that require GPU — automatically acquire _sem["gpu"] during execution
    _GPU_STEPS = {"music_separate", "enhance", "super_resolve", "asr"}

    # Order: steps before duration_split run once on full WAV,
    # then duration_split splits, then remaining steps run per-segment.
    def _get_step_index(key):
        try:
            return enabled_steps.index(key)
        except ValueError:
            return -1

    def _check_already_done(f):
        """Check if this file already has all expected outputs in final_dir.
        Returns True (and marks skipped) if every expected output exists."""
        base_name = os.path.splitext(f.name)[0]
        file_out = os.path.join(final_dir, base_name)
        has_enhanced = os.path.isfile(os.path.join(file_out, "enhanced", base_name + ".wav"))
        has_srt = os.path.isfile(os.path.join(file_out, "subtitles", base_name + ".srt"))
        has_clips = os.path.isdir(os.path.join(file_out, "clips")) and \
                    len(os.listdir(os.path.join(file_out, "clips"))) > 0
        # Build expected set based on enabled steps
        need_srt = "asr" in enabled_steps
        need_clips = "cut" in enabled_steps
        all_ok = has_enhanced
        if need_srt: all_ok = all_ok and has_srt
        if need_clips: all_ok = all_ok and has_clips
        if all_ok:
            f.status = "skipped"
            f.progress = 100
            f.steps.append({"step": "init", "status": "skipped",
                            "message": "输出已存在，跳过处理"})
            # Still collect existing output paths for frontend
            _publish_completed_file(f)
            return True
        return False

    def _process_one_file(f):
        try:
            state = file_state.setdefault(f.input_path, {})
            f.status = "running"
            _t0 = time.time()
            _log.info("[%s] 开始处理", f.name)

            # Pre-check: skip if outputs already exist in final_dir
            if _check_already_done(f):
                _log.info("[%s] 已有输出，跳过 (%.1fs)", f.name, time.time() - _t0)
                _update_pipeline_job_progress(job)
                return

            if job.cancelled:
                f.status = "error"; f.error = "任务已取消"
                _update_pipeline_job_progress(job); return

            ds_idx = _get_step_index("duration_split")
            pre_steps = enabled_steps[:ds_idx+1] if ds_idx >= 0 else enabled_steps
            post_steps = enabled_steps[ds_idx+1:] if ds_idx >= 0 else []

            # --- Fast path: duration_split is first, no pre-split steps ---
            # Split source directly → convert segments in parallel → process segments.
            # Avoids converting the entire file to WAV before splitting (saves ~1.2GB I/O).
            if ds_idx == 0:
                # Step A: Split source directly (no pre-convert needed)
                with _sem["duration_split"]:
                    _step_duration_split(f, job, state)

                segments = state.get("duration_segments", [])
                segments_needs_convert = state.get("seg_ext", ".wav") != ".wav"

                if segments and segments_needs_convert:
                    # Step B: Convert each segment to WAV in parallel with optimal params
                    opt_sr = state.get("opt_sr", 32000)
                    opt_ch = state.get("opt_ch", 1)

                    def _convert_segment(idx, seg_path):
                        if job.cancelled:
                            return
                        base = os.path.splitext(f.name)[0]
                        wav_out = os.path.join(state["seg_dir"], f"{base}_seg{idx:03d}.wav")
                        if os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
                            return wav_out
                        try:
                            with _sem["ffmpeg"]:
                                subprocess.run(
                                    [FFMPEG, "-y", "-i", seg_path, "-vn",
                                     "-acodec", "pcm_s16le",
                                     "-ar", str(opt_sr), "-ac", str(opt_ch),
                                     wav_out],
                                    capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=120,
                                )
                            if os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
                                os.remove(seg_path)  # clean up original-codec segment
                                return wav_out
                            return ""
                        except Exception:
                            return ""

                    f.steps.append({"step": "convert", "status": "running",
                                    "message": f"并行转换 {len(segments)} 段 ({opt_sr}Hz)..."})
                    _update_pipeline_job_progress(job)

                    # Phase B uses its own executor to avoid deadlock with the
                    # outer executor. Limit workers to prevent thread explosion
                    # when processing many files simultaneously.
                    convert_workers = min(len(segments), 8)
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=convert_workers
                    ) as convert_exec:
                        convert_futures = [convert_exec.submit(_convert_segment, i, p)
                                          for i, p in enumerate(segments)]
                        concurrent.futures.wait(convert_futures)

                    # Replace segment list with converted WAV paths
                    new_segments = []
                    failed = 0
                    for i, fut in enumerate(convert_futures):
                        result_path = fut.result()
                        if result_path and os.path.exists(result_path):
                            new_segments.append(result_path)
                        else:
                            failed += 1

                    if new_segments:
                        state["duration_segments"] = new_segments
                        segments = new_segments
                        state["seg_ext"] = ".wav"
                        f.steps.append({"step": "convert", "status": "completed",
                                        "message": f"并行转换完成: {len(new_segments)} 段 "
                                                   f"({opt_sr}Hz/{'单声道' if opt_ch == 1 else '立体声'})"
                                                   + (f", {failed}段失败" if failed else "")})
                    elif segments:
                        f.steps.append({"step": "convert", "status": "error",
                                        "message": f"所有 {len(segments)} 段转换失败"})
                        segments = []
                    _update_pipeline_job_progress(job)

                if segments:
                    # Process each segment through all remaining steps in parallel.
                    # If any segment fails any step → mark file error, stop immediately.
                    import threading as _threading2
                    seg_lock = _threading2.Lock()
                    seg_count = len(segments)
                    seg_states = [{"wav": p, "seg_suffix": f"_seg{i:03d}"} for i, p in enumerate(segments)]

                    for step_key in post_steps:
                        if job.cancelled:
                            f.status = "error"; f.error = "任务已取消"
                            _update_pipeline_job_progress(job); return

                        if step_key == "convert":
                            continue

                        handler = STEP_HANDLERS.get(step_key)
                        if not handler:
                            continue
                        step_sem = _sem.get(step_key, _threading2.BoundedSemaphore(2))

                        step_error = [False]
                        step_done = [0]
                        step_count_before = len(f.steps)
                        f.progress = 5
                        _update_pipeline_job_progress(job)

                        def _run_one_seg(_idx, _ss):
                            if job.cancelled or step_error[0]:
                                return
                            with seg_lock:
                                f.steps.append({"step": step_key, "status": "running",
                                                "message": f"片段 {_idx+1}/{seg_count}"})
                                _update_pipeline_job_progress(job)
                            try:
                                is_gpu = step_key in _GPU_STEPS
                                if is_gpu:
                                    with step_sem:
                                        with _sem["gpu"]:
                                            handler(f, job, _ss)
                                else:
                                    with step_sem:
                                        handler(f, job, _ss)
                            except Exception as e:
                                step_error[0] = True
                                with seg_lock:
                                    f.steps.append({"step": step_key, "status": "error",
                                                    "message": str(e)[:100]})
                                    _update_pipeline_job_progress(job)
                                return
                            with seg_lock:
                                step_done[0] += 1
                                f.progress = 5 + int(step_done[0] / seg_count * 80)
                                _update_pipeline_job_progress(job)
                            if "cut" not in enabled_steps and step_key == post_steps[-1]:
                                final_wav = _ss.get("wav", "")
                                if final_wav and os.path.exists(final_wav):
                                    with seg_lock:
                                        f.output_clips.append(final_wav)

                        step_workers = min(seg_count, 4)
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=step_workers
                        ) as step_exec:
                            futures = [step_exec.submit(_run_one_seg, idx, ss)
                                      for idx, ss in enumerate(seg_states)]
                            # Poll with timeout so cancellation is responsive
                            while futures:
                                done, futures = concurrent.futures.wait(
                                    futures, timeout=5,
                                    return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                if job.cancelled:
                                    for fut in futures:
                                        fut.cancel()
                                    break

                        # Check for handler-recorded errors
                        handler_errors = any(
                            s.get("step") == step_key and s.get("status") == "error"
                            for s in f.steps[step_count_before:]
                        )
                        if step_error[0] or handler_errors:
                            f.status = "error"
                            f.error = f"步骤 {step_key} 失败"
                            _update_pipeline_job_progress(job)
                            return

                    f.progress = 100
                    _update_pipeline_job_progress(job)
                    if "cut" in enabled_steps:
                        for ss in seg_states:
                            for clip in ss.get("output_clips", []):
                                if clip and os.path.exists(clip):
                                    f.output_clips.append(clip)
                else:
                    # No segments — file was fully discarded
                    pass
            else:
                # --- Standard path: steps may exist before duration_split, or no split at all ---
                # Convert full file to WAV first
                with _sem["convert"]:
                    if not _step_convert(f, job, state):
                        return

                for step_key in pre_steps:
                    if job.cancelled:
                        f.status = "error"; f.error = "任务已取消"
                        _update_pipeline_job_progress(job); return
                    handler = STEP_HANDLERS.get(step_key)
                    if not handler:
                        continue
                    sem = _sem.get(step_key, threading.BoundedSemaphore(2))
                    is_gpu = step_key in _GPU_STEPS
                    try:
                        if is_gpu:
                            with sem:
                                with _sem["gpu"]:
                                    handler(f, job, state)
                        else:
                            with sem:
                                handler(f, job, state)
                    except Exception as e:
                        f.steps.append({"step": step_key, "status": "error",
                                        "message": str(e)[:100]})
                        _update_pipeline_job_progress(job)

                # If duration_split produced segments, process each through remaining steps
                segments = state.get("duration_segments", [])
                if segments:
                    import threading as _threading2
                    seg_lock = _threading2.Lock()

                    def _process_segment(seg_idx, seg_path):
                        if job.cancelled: return
                        seg_suffix = f"_seg{seg_idx:03d}"
                        seg_state = {"wav": seg_path, "seg_suffix": seg_suffix}
                        with seg_lock:
                            f.steps.append({"step": "duration_split", "status": "running",
                                            "message": f"处理片段 {seg_idx+1}/{len(segments)}"})
                            _update_pipeline_job_progress(job)

                        for step_key in post_steps:
                            if job.cancelled: return
                            handler = STEP_HANDLERS.get(step_key)
                            if not handler: continue
                            sem = _sem.get(step_key, _threading2.BoundedSemaphore(2))
                            is_gpu = step_key in _GPU_STEPS
                            try:
                                if is_gpu:
                                    with sem:
                                        with _sem["gpu"]:
                                            handler(f, job, seg_state)
                                else:
                                    with sem:
                                        handler(f, job, seg_state)
                            except Exception as e:
                                with seg_lock:
                                    f.steps.append({"step": step_key, "status": "error",
                                                    "message": str(e)[:100]})
                                    _update_pipeline_job_progress(job)
                        if "cut" not in enabled_steps:
                            final_wav = seg_state.get("wav", "")
                            if final_wav and os.path.exists(final_wav):
                                with seg_lock:
                                    f.output_clips.append(final_wav)

                    # Use fresh executor to avoid deadlock with outer executor
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(len(segments), 4)
                    ) as seg_exec:
                        seg_futures = [seg_exec.submit(_process_segment, idx, p)
                                      for idx, p in enumerate(segments)]
                        while seg_futures:
                            done, seg_futures = concurrent.futures.wait(
                                seg_futures, timeout=5,
                                return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            if job.cancelled:
                                for fut in seg_futures:
                                    fut.cancel()
                                break
                else:
                    # No duration splitting, run remaining steps on full WAV
                    for step_key in post_steps:
                        if job.cancelled:
                            f.status = "error"; f.error = "任务已取消"
                            _update_pipeline_job_progress(job); return
                        handler = STEP_HANDLERS.get(step_key)
                        if not handler:
                            continue
                        sem = _sem.get(step_key, threading.BoundedSemaphore(2))
                        is_gpu = step_key in _GPU_STEPS
                        try:
                            if is_gpu:
                                with sem:
                                    with _sem["gpu"]:
                                        handler(f, job, state)
                            else:
                                with sem:
                                    handler(f, job, state)
                        except Exception as e:
                            f.steps.append({"step": step_key, "status": "error",
                                            "message": str(e)[:100]})
                            _update_pipeline_job_progress(job)
                    if "cut" not in enabled_steps:
                        final_wav = state.get("wav", "")
                        if final_wav and os.path.exists(final_wav):
                            f.output_clips.append(final_wav)

            f.status = "completed"
            _publish_completed_file(f, state)
            _log.info("[%s] 处理完成 (%.1fs) | 输出: %d 个文件",
                      f.name, time.time() - _t0, len(f.output_clips))
        except Exception as e:
            f.status = "error"
            f.error = str(e)[:200]
            _log.error("[%s] 处理失败 (%.1fs): %s", f.name, time.time() - _t0, str(e)[:100])
        _update_pipeline_job_progress(job)

    max_workers = min(len(files), 12)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_one_file, f) for f in files]
        # Poll with timeout so cancellation is responsive
        while futures:
            done, futures = concurrent.futures.wait(
                futures, timeout=5,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            if job.cancelled:
                for fut in futures:
                    fut.cancel()
                break

    # Only clean temp dir if all files produced outputs. Preserve temp data
    # for incomplete processing (e.g. segments timed out).
    all_have_outputs = all(
        f.status == "skipped" or len(f.output_clips) > 0
        for f in files
    )
    had_errors = any(f.status == "error" for f in files)
    if not had_errors and not job.cancelled and all_have_outputs:
        try:
            _shutil.rmtree(temp_dir)
            _log.info("清理临时目录: %s", temp_dir)
        except Exception as e:
            _log.warning("清理临时目录失败: %s", e)

    # Preserve real status: don't overwrite running/pending to completed
    for f in files:
        if f.status not in ("error", "skipped", "running", "pending"):
            f.status = "completed"
    if had_errors:
        job.status = "error"
    else:
        job.status = "completed"
    job.progress = 100
    _update_pipeline_job_progress(job)
    _save_pv_jobs()

    # Summary
    completed = sum(1 for f in files if f.status == "completed")
    skipped = sum(1 for f in files if f.status == "skipped")
    errors = sum(1 for f in files if f.status == "error")
    total_clips = sum(len(f.output_clips) for f in files)
    _log.info("=== 任务完成: %s | 耗时: %.1fs | 完成: %d | 跳过: %d | 异常: %d | 输出: %d 个文件 ===",
              job_id[:8], time.time() - t0, completed, skipped, errors, total_clips)


def _run_pipeline_video_safe(job_id: str):
    """Thread-safe wrapper that catches any unhandled exception."""
    try:
        _run_pipeline_video(job_id)
    except Exception:
        import logging as _logging, sys as _sys
        _log = _logging.getLogger("pipeline-video")
        if not _log.handlers:
            _h = _logging.StreamHandler(_sys.stdout)
            _h.setFormatter(_logging.Formatter(
                "[pipeline-video] %(asctime)s %(message)s", datefmt="%H:%M:%S"
            ))
            _log.addHandler(_h)
        _log.exception("管线异常崩溃 — 后台线程")
        with _pipeline_video_lock:
            job = _pipeline_video_jobs.get(job_id)
            if job and job.status not in ("completed", "cancelled", "error"):
                job.status = "error"
                for f in job.files:
                    if f.status not in ("completed", "skipped", "error"):
                        f.status = "error"
                        if not f.error:
                            f.error = "管线内部异常"
                        f.steps.append({"step": "fatal", "status": "error",
                                        "message": "管线处理线程异常崩溃，请查看服务器日志"})
                _update_pipeline_job_progress(job)
                _save_pv_jobs()

# ============================================================
# cut_audio_by_subtitle — helper to slice audio by SRT
# ============================================================

def cut_audio_by_subtitle(
    audio_path: str,
    srt_path: str,
    output_dir: str,
    base_name: str = "",
    padding: float = 0.1,
    on_progress=None,
) -> list:
    """Cut audio into WAV segments based on SRT subtitle timestamps.

    Each subtitle entry produces: {base_name}_{sanitized_text}.wav
    """
    import re
    import json as _json

    if not base_name:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    # Get audio duration
    audio_duration = 0.0
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        if result.returncode == 0:
            data = _json.loads(result.stdout)
            audio_duration = float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass

    # Parse SRT entries
    entries = _parse_srt_entries(srt_path)
    total = len(entries)
    clips = []

    for i, entry in enumerate(entries):
        start = max(0, entry["start"] - padding)
        end = min(audio_duration or 999999, entry["end"] + padding)
        duration = end - start

        if duration < 0.3:
            continue

        # Sanitize text for filename
        safe_text = re.sub(r'[\\/*?:"<>|]', '', entry["text"])[:30].strip()
        if not safe_text:
            safe_text = f"seg_{i + 1:03d}"
        safe_text = re.sub(r'\s+', '_', safe_text)
        if len(safe_text) > 50:
            safe_text = safe_text[:50]

        output_path = os.path.join(output_dir, f"{base_name}_{safe_text}.wav")

        counter = 1
        base_out = output_path
        while os.path.exists(output_path):
            stem, ext = os.path.splitext(base_out)
            output_path = f"{stem}_{counter}{ext}"
            counter += 1

        cmd = [
            FFMPEG, "-y",
            "-ss", str(start),
            "-i", audio_path,
            "-t", str(duration),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                clips.append(output_path)
        except Exception as e:
            print(f"[cut_audio] Error on segment {i + 1}: {e}")

        if on_progress:
            on_progress(i + 1, total)

    print(f"[cut_audio] Created {len(clips)} clips in {output_dir}")
    return clips


def _parse_srt_entries(srt_path: str) -> list:
    """Parse SRT file into list of {index, start, end, text}."""
    import re
    entries = []
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return entries

    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        # Skip index line, find timestamp line
        time_line_idx = None
        for li, line in enumerate(lines):
            if re.match(r'\d+:\d+:\d+[,\.]\d+\s*-->\s*\d+:\d+:\d+[,\.]\d+', line.strip()):
                time_line_idx = li
                break
        if time_line_idx is None:
            continue

        time_match = re.match(
            r'(\d+:\d+:\d+[,\.]\d+)\s*-->\s*(\d+:\d+:\d+[,\.]\d+)',
            lines[time_line_idx].strip()
        )
        if not time_match:
            continue

        start = _srt_time_to_sec(time_match.group(1))
        end = _srt_time_to_sec(time_match.group(2))
        text = " ".join(l.strip() for l in lines[time_line_idx + 1:] if l.strip())
        text = re.sub(r'<[^>]+>', '', text)  # strip HTML tags

        entries.append({"index": len(entries), "start": start, "end": end, "text": text})

    return entries


def _srt_time_to_sec(t: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    t = t.replace(",", ".")
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


# ============================================================
# MFA (Montreal Forced Aligner) endpoints
# ============================================================

_MFA_JOBS_FILE = os.path.join(DATA_DIR, "mfa_jobs.json")
_mfa_jobs: dict[str, dict] = {}
_mfa_lock = threading.RLock()


def _save_mfa_jobs():
    """Persist MFA jobs to disk."""
    try:
        with _mfa_lock:
            data = {k: dict(v) for k, v in _mfa_jobs.items()}
        tmp = _MFA_JOBS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, _MFA_JOBS_FILE)
    except Exception as e:
        print(f"[mfa] Failed to save jobs: {e}")


def _load_mfa_jobs():
    """Restore MFA jobs from disk on startup. Mark running jobs as interrupted."""
    if not os.path.exists(_MFA_JOBS_FILE):
        return
    try:
        with open(_MFA_JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for job_id, jd in data.items():
            if jd.get("status") == "running":
                jd["status"] = "interrupted"
                jd["current_step"] = "服务器重启中断"
            _mfa_jobs[job_id] = jd
        print(f"[startup] Restored {len(_mfa_jobs)} MFA jobs from disk")
    except Exception as e:
        print(f"[startup] Failed to load MFA jobs: {e}")


def _run_mfa_subprocess(job_id: str, cmd: list[str], env_extra: dict = None,
                         step_label: str = "", progress_total_pat: str = ""):
    """Run a subprocess for an MFA job, capturing stdout and updating progress.

    progress_total_pat: if set, regex pattern to extract progress like "Found N wav files"
      or "[N/total]" to compute percentage.
    """
    import re as _mfa_re
    with _mfa_lock:
        job = _mfa_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["current_step"] = step_label or "处理中..."
        _save_mfa_jobs()

    try:
        process_env = os.environ.copy()
        # Ensure conda bin is in PATH (for mfa, fstcompile, etc.)
        for _p in [os.path.expanduser("~/miniconda3/bin"), os.path.expanduser("~/anaconda3/bin")]:
            if os.path.isdir(_p) and _p not in process_env.get("PATH", ""):
                process_env["PATH"] = _p + os.pathsep + process_env.get("PATH", "")
        if env_extra:
            process_env.update(env_extra)

        # Log the command for debugging
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        stdout_lines = [
            "=" * 60 + "\n",
            f"MFA 任务: {job.get('type', 'unknown')}\n",
            f"步骤: {step_label}\n",
            f"命令: {cmd_str}\n",
            "=" * 60 + "\n\n",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=process_env,
            text=True,
            encoding='utf-8',
            errors='replace',
        )

        # Progress parsing patterns
        re_progress = _mfa_re.compile(r'\[(\d+)/(\d+)\]')  # [N/total]
        re_found = _mfa_re.compile(r'Found (\d+) wav files')
        total = 0

        for line in iter(process.stdout.readline, ''):
            stdout_lines.append(line)
            with _mfa_lock:
                j = _mfa_jobs.get(job_id)
                if j:
                    j["stdout"] = "".join(stdout_lines[-300:])
                    # Try to parse progress
                    m = re_progress.search(line)
                    if m:
                        current = int(m.group(1))
                        total = int(m.group(2))
                        if total > 0:
                            j["progress"] = round(current / total * 100)
                    elif re_found.search(line):
                        total_match = re_found.search(line)
                        if total_match:
                            total = int(total_match.group(1))
                    elif "Done." in line and "written" in line:
                        j["progress"] = 100
                    _save_mfa_jobs()

            with _mfa_lock:
                j = _mfa_jobs.get(job_id)
                if j and j.get("cancelled"):
                    process.kill()
                    j["status"] = "cancelled"
                    j["current_step"] = "已取消"
                    j["stdout"] = "".join(stdout_lines)
                    _save_mfa_jobs()
                    return

        process.wait()

        with _mfa_lock:
            j = _mfa_jobs.get(job_id)
            if j and not j.get("cancelled"):
                j["stdout"] = "".join(stdout_lines)
                if process.returncode == 0:
                    j["status"] = "completed"
                    j["progress"] = 100
                    j["current_step"] = "完成"
                else:
                    j["status"] = "failed"
                    j["error"] = "".join(stdout_lines[-10:]) if stdout_lines else f"Exit code: {process.returncode}"
                    j["current_step"] = f"失败 (exit {process.returncode})"
                _save_mfa_jobs()
    except Exception as e:
        with _mfa_lock:
            j = _mfa_jobs.get(job_id)
            if j:
                j["status"] = "failed"
                j["error"] = str(e)
                j["current_step"] = "异常退出"
                _save_mfa_jobs()


def _get_mfa_config(key: str, default=None):
    """Get an MFA config value from module globals (loaded from settings)."""
    val = globals().get(key)
    if val is not None:
        return val
    from config import _CONFIG_KEYS
    cfg = _CONFIG_KEYS
    return cfg.get(key, default)


# Step 1: Trim Silence
@app.post("/api/mfa/trim-silence")
def mfa_trim_silence(
    input_dir: str = Query(..., description="Raw WAV input directory"),
    output_dir: str = Query(..., description="Trimmed WAV output directory"),
    max_silence_sec: float = Query(1.0),
    sil_vol_threshold: float = Query(0.001),
    sil_len_threshold: float = Query(0.08),
    normalize_edges: bool = Query(True),
    target_edge_silence_sec: float = Query(0.5),
    edge_silence_threshold: float = Query(0.001),
    edge_frame_length: int = Query(1024),
    workers: int = Query(8),
):
    """Step 1: Batch trim silence from WAV files."""
    job_id = uuid.uuid4().hex[:12]
    python_exe = _get_mfa_config("MFA_PYTHON", "python")
    script = os.path.join(MFA_SCRIPTS_DIR, "trim_silence_batch.py")

    cmd = [
        python_exe, script,
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--max-silence-sec", str(max_silence_sec),
        "--sil-vol-threshold", str(sil_vol_threshold),
        "--sil-len-threshold", str(sil_len_threshold),
        "--workers", str(workers),
    ]
    if normalize_edges:
        cmd.extend(["--normalize-edges",
                     "--target-edge-silence-sec", str(target_edge_silence_sec),
                     "--edge-silence-threshold", str(edge_silence_threshold),
                     "--edge-frame-length", str(edge_frame_length)])

    with _mfa_lock:
        _mfa_jobs[job_id] = {
            "job_id": job_id, "type": "trim-silence", "status": "pending",
            "progress": 0, "current_step": "等待开始", "stdout": "", "error": None,
            "created_at": time.time(),
            "params": {"input_dir": input_dir, "output_dir": output_dir},
        }
        _save_mfa_jobs()

    threading.Thread(target=_run_mfa_subprocess,
                     args=(job_id, cmd),
                     kwargs={"step_label": "裁剪静音中...", "env_extra": None},
                     daemon=True).start()
    return {"status": "ok", "job_id": job_id}


def _tokenize_chinese_to_pinyin(text: str) -> str:
    """Convert Chinese text to space-separated pinyin syllables with tone numbers.

    "大家好" → "da4 jia1 hao3"
    Non-CJK characters (Latin, digits, punctuation) are kept as-is.

    Tries pypinyin in-process first; falls back to a subprocess using MFA_PYTHON
    so the conversion works even when the server's own Python lacks pypinyin.
    """
    try:
        from pypinyin import lazy_pinyin, Style
        tokens = []
        for ch in text:
            if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
                py = lazy_pinyin(ch, style=Style.TONE3, errors='ignore')
                tokens.append(py[0] if py else ch)
            elif ch.strip():
                tokens.append(ch)
        result = " ".join(tokens)
        if result.strip():
            return result
    except ImportError:
        pass

    # In-process pypinyin not available — try subprocess with MFA_PYTHON
    import sys as _sys
    python_exe = _get_mfa_config("MFA_PYTHON", _sys.executable)
    script = (
        "import sys, json\n"
        "text = sys.stdin.read()\n"
        "from pypinyin import lazy_pinyin, Style\n"
        "tokens = []\n"
        "for ch in text:\n"
        "    if '\\u4e00' <= ch <= '\\u9fff' or '\\u3400' <= ch <= '\\u4dbf':\n"
        "        py = lazy_pinyin(ch, style=Style.TONE3, errors='ignore')\n"
        "        tokens.append(py[0] if py else ch)\n"
        "    elif ch.strip():\n"
        "        tokens.append(ch)\n"
        "print(' '.join(tokens))\n"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", script],
            input=text, capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Last resort: character-level split (won't match pronunciation dict well)
    return " ".join(ch for ch in text if ch.strip())


# Step 2: Generate TXT
def _zh_tokenize_script(jsonl_path, output_dir, text_key, wav_key, dict_path, overwrite, wav_link_dir=None):
    """Return a Python one-liner script for Chinese tokenization + TXT generation.

    Converts Chinese text to space-separated pinyin syllables with tone numbers
    (e.g. "你好世界" → "ni3 hao3 shi4 jie4") so MFA can look them up in the
    fullpinyin_enword pronunciation dictionary.

    If wav_link_dir is provided, each source WAV is hardlinked (or copied) into
    that directory so MFA align can find it by stem without requiring the user
    to manually copy files around.
    """
    wav_link_repr = repr(str(wav_link_dir)) if wav_link_dir else ''
    return (
        "import json, os, sys, re, shutil\n"
        "os.makedirs(" + repr(str(output_dir)) + ", exist_ok=True)\n"
        "_wav_link_dir = " + (repr(str(wav_link_dir)) if wav_link_dir else 'None') + "\n"
        "if _wav_link_dir: os.makedirs(_wav_link_dir, exist_ok=True)\n"
        "# --- pypinyin: Chinese char → pinyin with tone number (ni3, hao3) ---\n"
        "try:\n"
        "  from pypinyin import lazy_pinyin, Style\n"
        "except ImportError:\n"
        "  lazy_pinyin = None\n"
        "# --- dict_words: load dictionary keys for OOV checking ---\n"
        "dict_words = set()\n"
        "if " + repr(bool(dict_path)) + ":\n"
        "  with open(" + repr(str(dict_path)) + ", 'r', encoding='utf-8-sig') as f:\n"
        "    for line in f:\n"
        "      line = line.strip()\n"
        "      if line and not line.startswith('#'):\n"
        "        w = line.split()[0].split('\\t')[0].strip()\n"
        "        if w: dict_words.add(w)\n"
        "oovs = {}\n"
        "written = 0\n"
        "linked = 0\n"
        "with open(" + repr(str(jsonl_path)) + ", 'r', encoding='utf-8') as f:\n"
        "  for line in f:\n"
        "    line = line.strip()\n"
        "    if not line: continue\n"
        "    item = json.loads(line)\n"
        "    text = item.get(" + repr(text_key) + ", '')\n"
        "    wav = item.get(" + repr(wav_key) + ", '')\n"
        "    if not text or not wav: continue\n"
        "    stem = os.path.splitext(os.path.basename(wav))[0]\n"
        "    out = os.path.join(" + repr(str(output_dir)) + ", stem + '.txt')\n"
        "    # Link WAV into wav_link_dir so MFA finds it without manual copy\n"
        "    if _wav_link_dir:\n"
        "      _wav_src = wav if os.path.isabs(wav) else os.path.join(os.getcwd(), wav)\n"
        "      _wav_dst = os.path.join(_wav_link_dir, stem + '.wav')\n"
        "      if os.path.exists(_wav_src) and not os.path.exists(_wav_dst):\n"
        "        try:\n"
        "          os.link(_wav_src, _wav_dst)\n"  # hardlink, fast & no space cost
        "          linked += 1\n"
        "        except OSError:\n"
        "          shutil.copy2(_wav_src, _wav_dst)\n"
        "          linked += 1\n"
        "    if os.path.exists(out) and not " + repr(overwrite) + ": continue\n"
        "    if lazy_pinyin:\n"
        "      # Convert each CJK char to pinyin tone3; keep non-CJK tokens as-is.\n"
        "      tokens = []\n"
        "      for ch in text:\n"
        "        if '\\u4e00' <= ch <= '\\u9fff' or '\\u3400' <= ch <= '\\u4dbf':\n"
        "          # CJK character → pinyin syllable\n"
        "          py = lazy_pinyin(ch, style=Style.TONE3, errors='ignore')\n"
        "          tokens.append(py[0] if py else ch)\n"
        "        elif ch.strip():\n"
        "          # Non-CJK (Latin, digits, punctuation) — keep as-is, skip spaces\n"
        "          tokens.append(ch)\n"
        "    else:\n"
        "      # Fallback: split on whitespace / char-level (won't match dict)\n"
        "      try:\n"
        "        import jieba\n"
        "        tokens = [w.strip() for w in jieba.cut(text) if w.strip()]\n"
        "      except ImportError:\n"
        "        tokens = list(text)\n"
        "    # Check OOV against dictionary\n"
        "    if dict_words:\n"
        "      for t in tokens:\n"
        "        if t not in dict_words:\n"
        "          oovs[t] = oovs.get(t, 0) + 1\n"
        "    with open(out, 'w', encoding='utf-8') as of:\n"
        "      of.write(' '.join(tokens) + '\\n')\n"
        "    written += 1\n"
        "print(f'Done. written={written} linked={linked}')\n"
        "if oovs:\n"
        "  oov_path = os.path.join(os.path.dirname(" + repr(str(output_dir)) + "), os.path.basename(" + repr(str(output_dir)) + ") + '_final_oovs.txt')\n"
        "  with open(oov_path, 'w', encoding='utf-8') as of:\n"
        "    for w, c in sorted(oovs.items(), key=lambda x: -x[1]):\n"
        "      of.write(f'{w}\\t{c}\\n')\n"
        "  print(f'OOV report: {len(oovs)} unique → {oov_path}')\n"
    )


@app.post("/api/mfa/generate-txt")
def mfa_generate_txt(
    jsonl_path: str = Query(..., description="JSONL file path"),
    output_dir: str = Query(..., description="TXT output directory"),
    text_key: str = Query("text"),
    wav_key: str = Query("wav_file"),
    mode: str = Query("A"),
    overwrite: bool = Query(True),
    dict_path: str = Query("", description="Optional MFA dictionary path"),
    max_merge_tokens: int = Query(5),
    oov_report: str = Query("", description="Optional custom OOV report path"),
    language: str = Query("zh", description="Language: jp or zh"),
    wav_link_dir: str = Query("", description="Auto-link source WAVs into this dir so MFA finds them without manual copy"),
):
    """Step 2: Generate MFA TXT transcript files from JSONL."""
    job_id = uuid.uuid4().hex[:12]
    python_exe = _get_mfa_config("MFA_PYTHON", "python")

    _wav_link = wav_link_dir or MFA_WAV_DIR

    if language == "jp":
        script = os.path.join(MFA_SCRIPTS_DIR, "generate_wav_txt.py")
        cmd = [python_exe, script, "--jsonl", jsonl_path, "--output-dir", output_dir,
               "--text-key", text_key, "--wav-key", wav_key, "--mode", mode.upper(),
               "--max-merge-tokens", str(max_merge_tokens)]
        if overwrite: cmd.append("--overwrite")
        if dict_path: cmd.extend(["--dict-path", dict_path])
        if oov_report: cmd.extend(["--oov-report", oov_report])
        if _wav_link: cmd.extend(["--wav-link-dir", _wav_link])
        step_label = "SudachiPy 分词中..."
    else:
        # Chinese: use internal tokenizer (pypinyin)
        step_label = "pypinyin 中文分词+拼音转换中..."
        cmd = [python_exe, "-c", _zh_tokenize_script(jsonl_path, output_dir, text_key, wav_key, dict_path, overwrite, _wav_link)]

    with _mfa_lock:
        _mfa_jobs[job_id] = {
            "job_id": job_id, "type": "generate-txt", "status": "pending",
            "progress": 0, "current_step": "等待开始", "stdout": "", "error": None,
            "created_at": time.time(),
            "params": {"jsonl_path": jsonl_path, "output_dir": output_dir, "mode": mode, "language": language},
        }
        _save_mfa_jobs()

    threading.Thread(target=_run_mfa_subprocess,
                     args=(job_id, cmd),
                     kwargs={"step_label": step_label},
                     daemon=True).start()
    return {"status": "ok", "job_id": job_id}


# Step 3a: MFA Validate
@app.post("/api/mfa/validate")
def mfa_validate(
    txt_dir: str = Query(..., description="TXT corpus directory"),
    wav_dir: str = Query("", description="WAV audio directory (if separate from txt)"),
    output_dir: str = Query(..., description="Validate output directory"),
    acoustic_model: str = Query("japanese_mfa"),
    dictionary: str = Query("japanese_mfa"),
    temp_dir: str = Query(""),
    num_jobs: int = Query(8),
    clean: bool = Query(True),
    overwrite: bool = Query(True),
):
    """Step 3a: Run MFA validate to check data readiness."""
    import tempfile as _tmp
    job_id = uuid.uuid4().hex[:12]
    models_root = _get_mfa_config("MFA_MODELS_DIR", MFA_MODELS_DIR)
    temp = _tmp.mkdtemp(prefix="mfa_val_")
    mfa_exe = _get_mfa_config("MFA_EXECUTABLE", "mfa")

    dict_path = os.path.join(models_root, "pretrained_models", "dictionary", dictionary + ".dict")
    acoustic_path = os.path.join(models_root, "pretrained_models", "acoustic", acoustic_model, acoustic_model)
    if not os.path.isdir(acoustic_path):
        acoustic_path = os.path.join(models_root, "pretrained_models", "acoustic", acoustic_model)

    cmd = [
        mfa_exe, "validate",
        txt_dir,
        dict_path,
        "--acoustic_model_path", acoustic_path,
        "--output_directory", output_dir,
        "--temporary_directory", temp,
        "--num_jobs", str(num_jobs),
    ]
    if wav_dir:
        cmd.extend(["--audio_directory", wav_dir])
    if clean:
        cmd.append("--clean")
    if overwrite:
        cmd.append("--overwrite")

    with _mfa_lock:
        _mfa_jobs[job_id] = {
            "job_id": job_id, "type": "mfa-validate", "status": "pending",
            "progress": 0, "current_step": "等待开始", "stdout": "", "error": None,
            "created_at": time.time(),
            "params": {"txt_dir": txt_dir, "output_dir": output_dir, "wav_dir": wav_dir},
        }
        _save_mfa_jobs()

    threading.Thread(target=_run_mfa_subprocess,
                     args=(job_id, cmd),
                     kwargs={"step_label": "MFA Validate 中...",
                             "env_extra": {"MFA_ROOT_DIR": models_root}},
                     daemon=True).start()
    return {"status": "ok", "job_id": job_id}


# Step 3b: MFA Align
@app.post("/api/mfa/align")
def mfa_align(
    txt_dir: str = Query(..., description="TXT corpus directory"),
    dictionary: str = Query("japanese_mfa"),
    acoustic_model: str = Query("japanese_mfa"),
    output_dir: str = Query(..., description="Aligned TextGrid output directory"),
    wav_dir: str = Query("", description="WAV audio directory (if separate from txt)"),
    temp_dir: str = Query(""),
    num_jobs: int = Query(8),
    clean: bool = Query(True),
    overwrite: bool = Query(True),
    output_format: str = Query("long_textgrid"),
    no_tokenization: bool = Query(True),
    no_textgrid_cleanup: bool = Query(True),
    check_audio_stem: str = Query("", description="Audio stem to check if TextGrid already exists"),
):
    """Step 3b: Run MFA align to produce TextGrid files."""
    import tempfile as _tmp, shutil as _shutil
    job_id = uuid.uuid4().hex[:12]
    models_root = _get_mfa_config("MFA_MODELS_DIR", MFA_MODELS_DIR)
    temp = _tmp.mkdtemp(prefix="mfa_align_")
    mfa_exe = _get_mfa_config("MFA_EXECUTABLE", "mfa")

    # Resolve relative paths to absolute (MFA needs absolute paths)
    if not os.path.isabs(txt_dir):
        txt_dir = os.path.normpath(os.path.join(PROJECT_ROOT, txt_dir))
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(PROJECT_ROOT, output_dir))
    if wav_dir and not os.path.isabs(wav_dir):
        wav_dir = os.path.normpath(os.path.join(PROJECT_ROOT, wav_dir))

    # Check if TextGrid already exists for the given audio stem
    if check_audio_stem:
        for ext in (".TextGrid", ".textgrid"):
            tg_path = os.path.join(output_dir, check_audio_stem + ext)
            if os.path.exists(tg_path):
                return {"status": "ok", "already_done": True, "message": f"TextGrid 已存在，无需重复生成: {check_audio_stem}{ext}"}

    # Resolve model names to full paths
    dict_path = os.path.join(models_root, "pretrained_models", "dictionary", dictionary + ".dict")
    if not os.path.exists(dict_path):
        dict_path = _get_mfa_config("MFA_DICT_PATH_ZH" if "pinyin" in dictionary or "enword" in dictionary else "MFA_DICT_PATH", dictionary)
    acoustic_path = os.path.join(models_root, "pretrained_models", "acoustic", acoustic_model, acoustic_model)
    if not os.path.isdir(acoustic_path):
        acoustic_path = os.path.join(models_root, "pretrained_models", "acoustic", acoustic_model)

    # Normalize TXT filenames: strip _qwen3/_firered suffixes so stems match WAV
    import re as _mfa_re
    corpus_dir = txt_dir
    if os.path.isdir(txt_dir):
        needs_fix = False
        for _fn in os.listdir(txt_dir):
            if _fn.lower().endswith(".txt") and _mfa_re.search(r'_(qwen3|firered|sensevoice)', _fn.lower()):
                needs_fix = True; break
        if needs_fix:
            norm_dir = _tmp.mkdtemp(prefix="mfa_corpus_")
            for _fn in os.listdir(txt_dir):
                if _fn.lower().endswith(".txt"):
                    _new = _mfa_re.sub(r'_(qwen3|firered|sensevoice)', '', _fn, flags=_mfa_re.IGNORECASE)
                    _shutil.copy2(os.path.join(txt_dir, _fn), os.path.join(norm_dir, _new))
            corpus_dir = norm_dir

    cmd = [
        mfa_exe, "align",
        corpus_dir,
        dict_path,
        acoustic_path,
        output_dir,
        "--output_format", output_format,
        "--temporary_directory", temp,
        "--num_jobs", str(num_jobs),
    ]
    if wav_dir:
        cmd.extend(["--audio_directory", wav_dir])
    if clean:
        cmd.append("--clean")
    if overwrite:
        cmd.append("--overwrite")
    if no_tokenization:
        cmd.append("--no_tokenization")
    if no_textgrid_cleanup:
        cmd.append("--no_textgrid_cleanup")

    with _mfa_lock:
        _mfa_jobs[job_id] = {
            "job_id": job_id, "type": "mfa-align", "status": "pending",
            "progress": 0, "current_step": "等待开始", "stdout": "", "error": None,
            "created_at": time.time(),
            "params": {"txt_dir": txt_dir, "output_dir": output_dir, "wav_dir": wav_dir},
        }
        _save_mfa_jobs()

    threading.Thread(target=_run_mfa_subprocess,
                     args=(job_id, cmd),
                     kwargs={"step_label": "MFA Align 中...",
                             "env_extra": {"MFA_ROOT_DIR": models_root}},
                     daemon=True).start()
    return {"status": "ok", "job_id": job_id}


# Step 4: Post-process TextGrid
@app.post("/api/mfa/postprocess")
def mfa_postprocess(
    jsonl_path: str = Query(..., description="JSONL file path"),
    txt_dir: str = Query(..., description="Generated TXT directory"),
    textgrid_dir: str = Query(..., description="MFA aligned TextGrid input directory"),
    output_dir: str = Query(..., description="Post-processed TextGrid output directory"),
    filtered_dir: str = Query(..., description="Filtered (bad) TextGrid output directory"),
    wav_dir: str = Query("", description="WAV directory for energy-based fix"),
    text_key: str = Query("text"),
    wav_key: str = Query("wav_file"),
    overwrite: bool = Query(True),
    fix_short_multi_unit: bool = Query(True),
    filter_suspicious_alignment: bool = Query(True),
    copy_errors: bool = Query(False),
    language: str = Query("jp", description="Language: zh (Chinese pinyin) or jp (Japanese romaji)"),
):
    """Step 4: Post-process MFA TextGrid output (add tiers, fix alignment, filter)."""
    job_id = uuid.uuid4().hex[:12]
    python_exe = _get_mfa_config("MFA_PYTHON", "python")
    script = os.path.join(MFA_SCRIPTS_DIR, "postprocess_textgrids.py")

    cmd = [
        python_exe, script,
        "--jsonl", jsonl_path,
        "--txt-dir", txt_dir,
        "--textgrid-dir", textgrid_dir,
        "--output-dir", output_dir,
        "--filtered-dir", filtered_dir,
        "--text-key", text_key,
        "--wav-key", wav_key,
    ]
    if wav_dir:
        cmd.extend(["--wav-dir", wav_dir])
    if overwrite:
        cmd.append("--overwrite")
    if fix_short_multi_unit:
        cmd.append("--fix-short-multi-unit")
    else:
        cmd.append("--no-fix-short-multi-unit")
    if filter_suspicious_alignment:
        cmd.append("--filter-suspicious-alignment")
    else:
        cmd.append("--no-filter-suspicious-alignment")
    if copy_errors:
        cmd.append("--copy-errors")
    cmd.extend(["--language", language])

    with _mfa_lock:
        _mfa_jobs[job_id] = {
            "job_id": job_id, "type": "mfa-postprocess", "status": "pending",
            "progress": 0, "current_step": "等待开始", "stdout": "", "error": None,
            "created_at": time.time(),
            "params": {"txt_dir": txt_dir, "textgrid_dir": textgrid_dir, "output_dir": output_dir, "wav_dir": wav_dir},
        }
        _save_mfa_jobs()

    threading.Thread(target=_run_mfa_subprocess,
                     args=(job_id, cmd),
                     kwargs={"step_label": "后处理 TextGrid 中..."},
                     daemon=True).start()
    return {"status": "ok", "job_id": job_id}


# Run all 4 steps sequentially
@app.post("/api/mfa/run-all")
def mfa_run_all(payload: dict = Body(...)):
    """Run all 4 MFA pipeline steps sequentially.

    Body keys match the individual step params, prefixed by the step number:
    trim_*  for step 1, gen_* for step 2, align_* for step 3, post_* for step 4.
    """
    job_id = uuid.uuid4().hex[:12]

    with _mfa_lock:
        _mfa_jobs[job_id] = {
            "job_id": job_id, "type": "mfa-run-all", "status": "pending",
            "progress": 0, "current_step": "等待开始", "stdout": "", "error": None,
            "created_at": time.time(), "cancelled": False,
            "params": payload,
        }
        _save_mfa_jobs()

    def _run_all():
        python_exe = _get_mfa_config("MFA_PYTHON", "python")
        model_root = _get_mfa_config("MFA_MODELS_DIR", MFA_MODELS_DIR)
        all_stdout = []
        skip_step2 = payload.get("skip_step2", False)

        steps = [
            {
                "name": "trim",
                "label": "步骤 1/4: 裁剪静音",
                "script": os.path.join(MFA_SCRIPTS_DIR, "trim_silence_batch.py"),
                "build_cmd": lambda: _build_trim_cmd(python_exe, payload),
            },
            {
                "name": "generate",
                "label": "步骤 2/4: 生成TXT",
                "script": os.path.join(MFA_SCRIPTS_DIR, "generate_wav_txt.py"),
                "build_cmd": lambda: _build_generate_cmd(python_exe, payload),
            },
            {
                "name": "align",
                "label": "步骤 3/4: MFA对齐",
                "build_cmd": lambda: _build_align_cmd(payload),
                "env": {"MFA_ROOT_DIR": model_root},
            },
            {
                "name": "postprocess",
                "label": "步骤 4/4: 后处理",
                "script": os.path.join(MFA_SCRIPTS_DIR, "postprocess_textgrids.py"),
                "build_cmd": lambda: _build_postprocess_cmd(python_exe, payload),
            },
        ]

        total_steps = 3 if skip_step2 else 4
        step_idx = 0

        for i, step_info in enumerate(steps):
            # Skip step 2 when external TXT file is provided
            if skip_step2 and step_info["name"] == "generate":
                step_label = "步骤 2/4: 已跳过 (使用外部 TXT)"
                with _mfa_lock:
                    j = _mfa_jobs.get(job_id)
                    if j and not j.get("cancelled"):
                        j["current_step"] = step_label
                        j["progress"] = round((1 / total_steps) * 100)
                        j["stdout"] = "".join(all_stdout) + "\n[跳过] 步骤2: 使用外部 TXT 文件作为对齐语料\n"
                        all_stdout.append("[跳过] 步骤2: 使用外部 TXT 文件作为对齐语料\n")
                        _save_mfa_jobs()
                continue
            with _mfa_lock:
                j = _mfa_jobs.get(job_id)
                if not j or j.get("cancelled"):
                    return
                j["current_step"] = step_info["label"]
                j["progress"] = round((step_idx / total_steps) * 100)
                _save_mfa_jobs()

            try:
                cmd = step_info["build_cmd"]()
                env = step_info.get("env", None)
                process_env = os.environ.copy()
                for _p in [os.path.expanduser("~/miniconda3/bin"), os.path.expanduser("~/anaconda3/bin")]:
                    if os.path.isdir(_p) and _p not in process_env.get("PATH", ""):
                        process_env["PATH"] = _p + os.pathsep + process_env.get("PATH", "")
                if env:
                    process_env.update(env)

                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT, env=process_env,
                    text=True, encoding='utf-8', errors='replace',
                )
                step_stdout = []
                for line in iter(process.stdout.readline, ''):
                    step_stdout.append(line)
                    all_stdout.append(line)
                    with _mfa_lock:
                        j = _mfa_jobs.get(job_id)
                        if j:
                            j["stdout"] = "".join(all_stdout[-500:])
                            j["progress"] = round((step_idx + 0.5) / total_steps * 100)
                            _save_mfa_jobs()
                    # Check cancellation
                    with _mfa_lock:
                        j = _mfa_jobs.get(job_id)
                        if j and j.get("cancelled"):
                            process.kill()
                            j["status"] = "cancelled"
                            j["current_step"] = "已取消"
                            _save_mfa_jobs()
                            return

                process.wait()
                if process.returncode != 0:
                    with _mfa_lock:
                        j = _mfa_jobs.get(job_id)
                        if j and not j.get("cancelled"):
                            j["status"] = "failed"
                            j["error"] = f"步骤 {step_info['name']} 失败: exit code {process.returncode}\n" + "".join(step_stdout[-20:])
                            j["current_step"] = f"步骤 {step_info['name']} 失败"
                            j["stdout"] = "".join(all_stdout)
                            _save_mfa_jobs()
                    return
                step_idx += 1
            except Exception as e:
                with _mfa_lock:
                    j = _mfa_jobs.get(job_id)
                    if j and not j.get("cancelled"):
                        j["status"] = "failed"
                        j["error"] = f"步骤 {i+1} 异常: {e}"
                        j["current_step"] = f"步骤 {i+1} 异常"
                        j["stdout"] = "".join(all_stdout)
                        _save_mfa_jobs()
                return

        # All steps completed
        with _mfa_lock:
            j = _mfa_jobs.get(job_id)
            if j and not j.get("cancelled"):
                j["status"] = "completed"
                j["progress"] = 100
                j["current_step"] = "全部完成"
                j["stdout"] = "".join(all_stdout)
                _save_mfa_jobs()

    threading.Thread(target=_run_all, daemon=True).start()
    return {"status": "ok", "job_id": job_id}


def _build_trim_cmd(python_exe, payload):
    script = os.path.join(MFA_SCRIPTS_DIR, "trim_silence_batch.py")
    cmd = [
        python_exe, script,
        "--input-dir", payload.get("trim_input_dir", ""),
        "--output-dir", payload.get("trim_output_dir", ""),
        "--max-silence-sec", str(payload.get("trim_max_silence_sec", 1.0)),
        "--sil-vol-threshold", str(payload.get("trim_sil_vol_threshold", 0.001)),
        "--sil-len-threshold", str(payload.get("trim_sil_len_threshold", 0.08)),
        "--workers", str(payload.get("trim_workers", 8)),
    ]
    if payload.get("trim_normalize_edges", True):
        cmd.extend([
            "--normalize-edges",
            "--target-edge-silence-sec", str(payload.get("trim_target_edge_silence_sec", 0.5)),
            "--edge-silence-threshold", str(payload.get("trim_edge_silence_threshold", 0.001)),
            "--edge-frame-length", str(payload.get("trim_edge_frame_length", 1024)),
        ])
    return cmd


def _build_generate_cmd(python_exe, payload):
    lang = payload.get("language", "zh")
    _wav_link = payload.get("gen_wav_link_dir", "") or MFA_WAV_DIR
    if lang == "jp":
        script = os.path.join(MFA_SCRIPTS_DIR, "generate_wav_txt.py")
        cmd = [python_exe, script, "--jsonl", payload.get("gen_jsonl_path", ""),
               "--output-dir", payload.get("gen_output_dir", ""),
               "--text-key", payload.get("gen_text_key", "text"),
               "--wav-key", payload.get("gen_wav_key", "wav_file"),
               "--mode", payload.get("gen_mode", "A"),
               "--max-merge-tokens", str(payload.get("gen_max_merge_tokens", 5))]
        if payload.get("gen_overwrite", True): cmd.append("--overwrite")
        if payload.get("gen_dict_path"): cmd.extend(["--dict-path", payload["gen_dict_path"]])
        if payload.get("gen_oov_report"): cmd.extend(["--oov-report", payload["gen_oov_report"]])
        cmd.extend(["--wav-link-dir", _wav_link])
    else:
        cmd = [python_exe, "-c", _zh_tokenize_script(
            payload.get("gen_jsonl_path", ""), payload.get("gen_output_dir", ""),
            payload.get("gen_text_key", "text"), payload.get("gen_wav_key", "wav_file"),
            payload.get("gen_dict_path", ""), payload.get("gen_overwrite", True),
            _wav_link)]
    return cmd


def _build_align_cmd(payload):
    wav_dir = payload.get("align_wav_dir", "")
    output_format = payload.get("align_output_format", "long_textgrid")
    num_jobs = int(payload.get("align_num_jobs", 8))
    clean = payload.get("align_clean", True)
    overwrite = payload.get("align_overwrite", True)
    no_tokenization = payload.get("align_no_tokenization", True)
    no_textgrid_cleanup = payload.get("align_no_textgrid_cleanup", True)
    mfa_exe = _get_mfa_config("MFA_EXECUTABLE", "mfa")
    model_root = _get_mfa_config("MFA_MODELS_DIR", MFA_MODELS_DIR)
    dictionary = payload.get("align_dictionary", "fullpinyin_enword")
    acoustic = payload.get("align_acoustic_model", "corp4EPL_sat2")

    dict_path = os.path.join(model_root, "pretrained_models", "dictionary", dictionary + ".dict")
    acoustic_path = os.path.join(model_root, "pretrained_models", "acoustic", acoustic, acoustic)
    if not os.path.isdir(acoustic_path):
        acoustic_path = os.path.join(model_root, "pretrained_models", "acoustic", acoustic)

    # Normalize TXT filenames: strip _qwen3/_firered suffixes
    import re as _mfa_re2, shutil as _shutil2, tempfile as _tmp2
    txt_dir = payload.get("align_txt_dir", "")
    corpus_dir = txt_dir
    if os.path.isdir(txt_dir):
        needs_fix = False
        for _fn in os.listdir(txt_dir):
            if _fn.lower().endswith(".txt") and _mfa_re2.search(r'_(qwen3|firered|sensevoice)', _fn.lower()):
                needs_fix = True; break
        if needs_fix:
            norm_dir = _tmp2.mkdtemp(prefix="mfa_alld_")
            for _fn in os.listdir(txt_dir):
                if _fn.lower().endswith(".txt"):
                    _new = _mfa_re2.sub(r'_(qwen3|firered|sensevoice)', '', _fn, flags=_mfa_re2.IGNORECASE)
                    _shutil2.copy2(os.path.join(txt_dir, _fn), os.path.join(norm_dir, _new))
            corpus_dir = norm_dir

    cmd = [
        mfa_exe, "align",
        corpus_dir,
        dict_path,
        acoustic_path,
        payload.get("align_output_dir", ""),
        "--output_format", output_format,
        "--temporary_directory", temp,
        "--num_jobs", str(num_jobs),
    ]
    if wav_dir:
        cmd.extend(["--audio_directory", wav_dir])
    if clean:
        cmd.append("--clean")
    if overwrite:
        cmd.append("--overwrite")
    if no_tokenization:
        cmd.append("--no_tokenization")
    if no_textgrid_cleanup:
        cmd.append("--no_textgrid_cleanup")
    return cmd


def _build_postprocess_cmd(python_exe, payload):
    script = os.path.join(MFA_SCRIPTS_DIR, "postprocess_textgrids.py")
    cmd = [
        python_exe, script,
        "--jsonl", payload.get("post_jsonl_path", ""),
        "--txt-dir", payload.get("post_txt_dir", ""),
        "--textgrid-dir", payload.get("post_textgrid_dir", ""),
        "--output-dir", payload.get("post_output_dir", ""),
        "--filtered-dir", payload.get("post_filtered_dir", ""),
        "--text-key", payload.get("post_text_key", "text"),
        "--wav-key", payload.get("post_wav_key", "wav_file"),
    ]
    wav_dir = payload.get("post_wav_dir", "")
    if wav_dir:
        cmd.extend(["--wav-dir", wav_dir])
    if payload.get("post_overwrite", True):
        cmd.append("--overwrite")
    if payload.get("post_fix_short_multi_unit", True):
        cmd.append("--fix-short-multi-unit")
    else:
        cmd.append("--no-fix-short-multi-unit")
    if payload.get("post_filter_suspicious_alignment", True):
        cmd.append("--filter-suspicious-alignment")
    else:
        cmd.append("--no-filter-suspicious-alignment")
    if payload.get("post_copy_errors", False):
        cmd.append("--copy-errors")
    return cmd


# ============================================================
# MFA TextGrid Viewer — parse + audio stream
# ============================================================

_RE_TG_UNQUOTE = __import__('re').compile(r'^"(.*)"$')


def _tg_unquote(v: str) -> str:
    v = v.strip()
    m = _RE_TG_UNQUOTE.match(v)
    if m:
        return m.group(1).replace('""', '"')
    return v


def _parse_textgrid_file(path: str) -> dict:
    """Parse a Praat TextGrid file into structured JSON.

    Returns {xmin, xmax, tiers: [{name, intervals: [{xmin, xmax, text, duration}]}]}.
    Works with both long_textgrid and short_textgrid formats.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    xmin = 0.0
    xmax = 0.0
    tiers = []
    current = None
    pending_xmin = None
    pending_xmax = None
    in_items = False
    in_interval = False

    for raw_line in lines:
        line = raw_line.strip()
        if line == "item []:" or line == "item[]:":
            in_items = True
            continue
        if not in_items:
            if line.startswith("xmin ="):
                xmin = float(line.split("=", 1)[1])
            elif line.startswith("xmax ="):
                xmax = float(line.split("=", 1)[1])
            continue
        if line.startswith("item [") or line.startswith("item["):
            if current is not None:
                tiers.append(current)
            current = {"name": "", "intervals": []}
            pending_xmin = None
            pending_xmax = None
            in_interval = False
        elif current is not None and line.startswith("name ="):
            current["name"] = _tg_unquote(line.split("=", 1)[1].strip())
        elif current is not None and line.startswith("xmin ="):
            v = float(line.split("=", 1)[1])
            if in_interval:
                pending_xmin = v
            else:
                current["_xmin"] = v
        elif current is not None and line.startswith("xmax ="):
            v = float(line.split("=", 1)[1])
            if in_interval:
                pending_xmax = v
            else:
                current["_xmax"] = v
        elif current is not None and (line.startswith("intervals [") or line.startswith("intervals[")):
            pending_xmin = None
            pending_xmax = None
            in_interval = True
        elif current is not None and line.startswith("text ="):
            text = _tg_unquote(line.split("=", 1)[1].strip())
            if pending_xmin is None or pending_xmax is None:
                continue
            dur = round(pending_xmax - pending_xmin, 6)
            current["intervals"].append({
                "xmin": round(pending_xmin, 6),
                "xmax": round(pending_xmax, 6),
                "text": text,
                "duration": dur,
            })
            pending_xmin = None
            pending_xmax = None
            in_interval = False

    if current is not None:
        tiers.append(current)

    for t in tiers:
        t.pop("_xmin", None)
        t.pop("_xmax", None)

    return {"xmin": round(xmin, 6), "xmax": round(xmax, 6), "tiers": tiers}


@app.get("/api/mfa/parse-textgrid")
def mfa_parse_textgrid(path: str = Query(..., description="Path to TextGrid file")):
    """Parse a Praat TextGrid and return interval data as JSON."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"TextGrid not found: {path}")
    if not path.lower().endswith(".textgrid"):
        raise HTTPException(status_code=400, detail="File must be a .TextGrid")
    try:
        data = _parse_textgrid_file(path)
        # Summarize: total intervals per tier
        for t in data["tiers"]:
            t["interval_count"] = len(t["intervals"])
        data["tier_count"] = len(data["tiers"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")


@app.get("/api/mfa/find-wav")
def mfa_find_wav(textgrid_path: str = Query(..., description="TextGrid file path, finds matching WAV by stem"),
                 wav_dir: str = Query("", description="Step 3 wav_dir — audio input directory")):
    """Find the WAV that was fed into step 3 (align) for this TextGrid.

    MFA align produces TextGrid with the same stem as the input WAV.
    So just look for {tg_stem}.wav in wav_dir, tg_dir, then MFA_WAV_DIR.
    """
    if not os.path.exists(textgrid_path):
        raise HTTPException(status_code=404, detail=f"TextGrid not found: {textgrid_path}")

    tg_stem = os.path.splitext(os.path.basename(textgrid_path))[0]
    tg_dir = os.path.dirname(textgrid_path)

    print(f"[find-wav] TextGrid: {textgrid_path}")
    print(f"[find-wav]   stem={tg_stem}, wav_dir={wav_dir or '(none)'}")

    # Simple sequential lookup — no fuzzy, no guesswork
    search_dirs = []
    if wav_dir and os.path.isdir(wav_dir):
        search_dirs.append(wav_dir)
    search_dirs.append(tg_dir)
    if os.path.isdir(MFA_WAV_DIR):
        search_dirs.append(MFA_WAV_DIR)

    for sd in search_dirs:
        wav_path = os.path.join(sd, tg_stem + ".wav")
        exists = os.path.exists(wav_path)
        print(f"[find-wav]   check: {wav_path} -> {'OK' if exists else 'NO'}")
        if exists:
            return {"wav_path": os.path.normpath(wav_path).replace("\\", "/"),
                    "stem": tg_stem, "source": "exact"}

    # Fuzzy fallback: strip _separated_* suffixes (MFA splits audio into separated tracks)
    import re as _fuzzy_re
    fuzzy_stems = []
    m = _fuzzy_re.match(r'^(.+?)_separated_([^_]+)$', tg_stem)
    if m:
        base = m.group(1)
        fuzzy_stems.append(base)
        nested = _fuzzy_re.match(r'^(.+?)_seg\d+$', base)
        if nested:
            fuzzy_stems.append(nested.group(1))
    m2 = _fuzzy_re.match(r'^(.+?)_seg\d+$', tg_stem)
    if m2:
        fuzzy_stems.append(m2.group(1))
    # Strip trailing _segNNN_separated_segNNN chain
    simplified = _fuzzy_re.sub(r'(_seg\d+)*(_separated_[^_]+)*$', '', tg_stem)
    if simplified and simplified != tg_stem:
        fuzzy_stems.append(simplified)

    # Deduplicate while preserving order
    seen = set()
    fuzzy_stems = [s for s in fuzzy_stems if not (s in seen or seen.add(s))]

    for fs in fuzzy_stems:
        for sd in search_dirs:
            wav_path = os.path.join(sd, fs + ".wav")
            exists = os.path.exists(wav_path)
            print(f"[find-wav]   fuzzy check: {wav_path} -> {'OK' if exists else 'NO'}")
            if exists:
                return {"wav_path": os.path.normpath(wav_path).replace("\\", "/"),
                        "stem": fs, "source": "fuzzy_stem", "original_stem": tg_stem}

    # Last resort: scan the directory for any WAV whose stem is a prefix of the tg_stem
    for sd in search_dirs:
        if not os.path.isdir(sd):
            continue
        try:
            for fn in sorted(os.listdir(sd)):
                if not fn.lower().endswith(".wav"):
                    continue
                wav_stem = fn[:-4]
                if tg_stem.startswith(wav_stem) or wav_stem.startswith(tg_stem):
                    wav_path = os.path.join(sd, fn)
                    print(f"[find-wav]   prefix match: {wav_path} -> OK")
                    return {"wav_path": os.path.normpath(wav_path).replace("\\", "/"),
                            "stem": wav_stem, "source": "prefix_match", "original_stem": tg_stem}
        except OSError:
            pass

    print(f"[find-wav]   NOT FOUND! stem={tg_stem}, search_dirs={search_dirs}")
    return {"wav_path": "", "stem": tg_stem, "source": "not_found",
            "search_dirs": [d.replace("\\", "/") for d in search_dirs],
            "hint": f"在以下目录未找到 {tg_stem}.wav, 也试了模糊匹配: {fuzzy_stems}"}


@app.get("/api/mfa/stream")
def mfa_stream(path: str = Query(..., description="Path to audio file (WAV/MP3)")):
    """Stream an audio file for playback in the browser."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = path.lower()
    mt = "audio/wav" if ext.endswith(".wav") else ("audio/mpeg" if ext.endswith(".mp3") else "audio/ogg")
    return FileResponse(path, media_type=mt)


@app.get("/api/mfa/waveform")
def mfa_waveform(path: str = Query(..., description="Path to WAV file"),
                 resolution: int = Query(1000, description="Number of peak data points")):
    """Return waveform peak data as JSON — much lighter than downloading the full WAV."""
    import numpy as np
    import soundfile as sf

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        info = sf.info(path)
        total_frames = info.frames
        sr = info.samplerate
        duration = total_frames / max(sr, 1)

        # Read only enough data for the requested resolution
        resolution = min(resolution, 2000)  # cap at 2000 points
        step = max(1, total_frames // resolution)

        # Read in chunks to keep memory low
        peaks = []
        pos = 0
        block_size = step * 4  # read 4 rows worth per chunk
        while pos < total_frames:
            end = min(pos + block_size, total_frames)
            data, _ = sf.read(path, start=pos, stop=end, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)  # mix down to mono
            n = len(data)
            for i in range(0, n, step):
                chunk_end = min(i + step, n)
                chunk = data[i:chunk_end]
                if len(chunk) > 0:
                    peaks.append(round(float(np.max(np.abs(chunk))), 6))
            pos = end

        return {
            "path": path,
            "duration": round(duration, 3),
            "sample_rate": sr,
            "resolution": len(peaks),
            "peaks": peaks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read waveform: {e}")


# Tokenize a TXT file directly (skip JSONL)
@app.post("/api/mfa/tokenize-txt")
def mfa_tokenize_txt(
    txt_path: str = Query(..., description="Path to the TXT file with raw text"),
    wav_file: str = Query(..., description="Matching WAV filename"),
    output_dir: str = Query(..., description="Output directory for tokenized TXT"),
    language: str = Query("zh", description="zh or jp"),
):
    """Read a TXT, tokenize with jieba (zh) or SudachiPy (jp), write result."""
    if not os.path.isfile(txt_path):
        raise HTTPException(status_code=404, detail="TXT file not found")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise HTTPException(status_code=400, detail="TXT file is empty")

    # Resolve to absolute path relative to PROJECT_ROOT
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(wav_file).stem
    out_path = out_dir / (stem + ".txt")

    if language == "jp":
        python_exe = _get_mfa_config("MFA_PYTHON", "python")
        import tempfile as _tmp3
        with _tmp3.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as tmp:
            json.dump({"text": text, "wav_file": Path(wav_file).name}, tmp, ensure_ascii=False)
            tmp.write("\n"); tmp_jsonl = tmp.name
        script = os.path.join(MFA_SCRIPTS_DIR, "generate_wav_txt.py")
        subprocess.run([python_exe, script, "--jsonl", tmp_jsonl, "--output-dir", str(out_dir),
                        "--mode", "A", "--overwrite"], capture_output=True, cwd=PROJECT_ROOT, timeout=120)
        try: os.unlink(tmp_jsonl)
        except: pass
    else:
        tokenized = _tokenize_chinese_to_pinyin(text)
        out_path.write_text(tokenized + "\n", encoding="utf-8")

    result_text = ""
    if out_path.exists():
        with open(str(out_path), "r", encoding="utf-8") as f:
            result_text = f.read()[:300]
    return {"status": "ok", "output_path": str(out_path), "output_dir": str(out_dir),
            "token_count": len(result_text.split()), "preview": result_text}


# Auto-find TXT transcript for an audio file
@app.get("/api/mfa/find-txt")
def mfa_find_txt(audio_path: str = Query(..., description="Path to audio file")):
    """Given an audio file path, find matching TXT transcript.

    Directory structure:
      segments/<name>/<name>_seg001.wav
      segments/<name>/txt/<name>_seg001_qwen3.txt
    Looks in <audio_dir>/txt/ for files starting with the audio stem,
    preferring *_qwen3.txt over *_firered.txt.
    """
    if not os.path.isfile(audio_path):
        return {"txt_path": "", "found": False, "hint": "Audio file not found"}

    audio_dir = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]

    # Priority: {audio_dir}/txt/{stem}_qwen3.txt > {stem}_firered.txt > {stem}_*.txt
    txt_dir = os.path.join(audio_dir, "txt")
    if os.path.isdir(txt_dir):
        matching = []
        try:
            for name in os.listdir(txt_dir):
                if name.startswith(stem) and name.lower().endswith(".txt"):
                    matching.append(name)
        except OSError:
            pass
        # Sort: qwen3 first, then firered, then others
        def _rank(n):
            low = n.lower()
            if "_qwen3" in low: return 0
            if "_firered" in low: return 1
            return 2
        matching.sort(key=_rank)
        if matching:
            best = matching[0]
            full = os.path.join(txt_dir, best)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    preview = f.read()[:300]
            except Exception:
                preview = ""
            return {"txt_path": os.path.normpath(full).replace("\\", "/"), "found": True,
                    "stem": stem, "all_matches": matching,
                    "preview": preview,
                    "txt_dir": txt_dir.replace("\\", "/")}

    # Shared ranking helper
    def _rank_mfa(n):
        low = n.lower()
        if "_qwen3" in low: return 0
        if "_firered" in low: return 1
        return 2

    # Fallback 1: look in <audio_dir>/<stem>/txt/ (audio and folder are siblings)
    fb_dir = os.path.join(audio_dir, stem, "txt")
    if os.path.isdir(fb_dir):
        matching = []
        try:
            for name in os.listdir(fb_dir):
                if name.startswith(stem) and name.lower().endswith(".txt"):
                    matching.append(name)
        except OSError: pass
        matching.sort(key=_rank_mfa)
        if matching:
            full = os.path.join(fb_dir, matching[0])
            return {"txt_path": os.path.normpath(full).replace("\\", "/"), "found": True,
                    "stem": stem, "all_matches": matching,
                    "txt_dir": fb_dir.replace("\\", "/")}

    # Fallback 2: search sibling dirs for a folder matching the stem base
    #   (strip _segNNN suffix), then check its txt/ subdir
    import re as _mfa_re
    base_m = _mfa_re.match(r'^(.+?)(_seg\d+)?$', stem)
    if base_m:
        base = base_m.group(1)
        try:
            for entry in os.listdir(audio_dir):
                entry_path = os.path.join(audio_dir, entry)
                if os.path.isdir(entry_path) and entry == base:
                    stxt = os.path.join(entry_path, "txt")
                    if os.path.isdir(stxt):
                        matching = []
                        for name in os.listdir(stxt):
                            if name.startswith(stem) and name.lower().endswith(".txt"):
                                matching.append(name)
                        matching.sort(key=_rank_mfa)
                        if matching:
                            full = os.path.join(stxt, matching[0])
                            return {"txt_path": os.path.normpath(full).replace("\\", "/"), "found": True,
                                    "stem": stem, "all_matches": matching,
                                    "txt_dir": stxt.replace("\\", "/")}
        except OSError: pass

    # Fallback 3: exact stem.txt in same dir or txt subdir
    for c in [os.path.join(audio_dir, stem + ".txt"),
              os.path.join(audio_dir, "txt", stem + ".txt")]:
        if os.path.isfile(c):
            return {"txt_path": os.path.normpath(c).replace("\\", "/"), "found": True,
                    "stem": stem, "all_matches": [],
                    "txt_dir": os.path.dirname(c).replace("\\", "/")}

    return {"txt_path": "", "found": False, "stem": stem,
            "hint": f"TXT not found, searched: {txt_dir}, {fb_dir}"}


# Simple TTL cache for browse-dir results (helps with NAS/large dirs)
_browse_cache: dict[str, tuple[float, dict]] = {}
_BROWSE_CACHE_TTL = 10.0  # seconds


# MFA directory browser — shows folders + SRT files
@app.get("/api/mfa/browse-dir")
def mfa_browse_dir(path: str = Query("", description="Absolute directory path"),
                   ext: str = Query("", description="File extensions to show, comma-separated. Default: .srt")):
    """List folders and files in a directory. Filter by extension if specified."""
    if not ext:
        ext = ".srt"
    exts = tuple(e.strip().lower() for e in ext.split(",") if e.strip())

    if not path or not os.path.isdir(path):
        # Choose default based on file type
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
        textgrid_exts = {".textgrid"}
        if set(exts) & audio_exts:
            path = PIPELINE_VIDEO_DIR
        elif set(exts) & textgrid_exts:
            # For TextGrid files, prefer MFA post dir, then aligned, then MFA root
            if os.path.isdir(MFA_POST_DIR):
                path = MFA_POST_DIR
            elif os.path.isdir(MFA_ALIGNED_DIR):
                path = MFA_ALIGNED_DIR
            else:
                path = MFA_WAV_DIR
        else:
            path = os.path.join(DATA_DIR, "asr", "folder_output")
        if not os.path.isdir(path):
            path = MFA_WAV_DIR
        if not os.path.isdir(path):
            path = DATA_DIR
        os.makedirs(path, exist_ok=True)

    path = os.path.normpath(path)

    # Check cache (keyed by path + sorted exts)
    cache_key = path + "|" + ",".join(sorted(exts))
    now = time.time()
    cached = _browse_cache.get(cache_key)
    if cached and now - cached[0] < _BROWSE_CACHE_TTL:
        return cached[1]

    folders = []
    files = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir() and not entry.name.startswith("."):
                    folders.append({"name": entry.name, "path": entry.path.replace("\\", "/"), "type": "dir"})
                elif entry.is_file() and entry.name.lower().endswith(exts):
                    ft = os.path.splitext(entry.name)[1].lower().lstrip(".")
                    try:
                        st = entry.stat()
                        mtime = st.st_mtime
                        size_kb = round(st.st_size / 1024, 1)
                    except OSError:
                        mtime = 0
                        size_kb = 0
                    files.append({"name": entry.name, "path": entry.path.replace("\\", "/"), "type": ft,
                                  "size_kb": size_kb, "mtime": mtime})

        folders.sort(key=lambda e: e["name"])
        files.sort(key=lambda e: e["name"])

        parent = os.path.dirname(path)
        if parent == path or not os.path.isdir(parent):
            parent = None

        result = {
            "path": path.replace("\\", "/"),
            "parent": parent.replace("\\", "/") if parent else None,
            "folders": folders,
            "files": files,
        }
        # Store in cache with TTL
        _browse_cache[cache_key] = (now, result)
        # Limit cache size
        if len(_browse_cache) > 50:
            oldest = min(_browse_cache, key=lambda k: _browse_cache[k][0])
            del _browse_cache[oldest]
        return result
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")


# MFA read text file — for viewing logs and text content
@app.get("/api/mfa/read-text")
def mfa_read_text(path: str = Query(..., description="Path to text/log file")):
    """Read a text file and return its content (truncated to 50KB for UI display)."""
    import mimetypes
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(path)[1].lower()
    # Allow text-like extensions and common log/vocab files
    allowed = {".txt", ".log", ".csv", ".json", ".out", ".dict", ".yaml", ".yml", ".textgrid", ".TextGrid"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(51200)  # 50KB limit
        return {"path": os.path.normpath(path).replace("\\", "/"), "content": content, "size": len(content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MFA text file matching — fuzzy search text files by audio stem fields
@app.get("/api/mfa/match-text")
def mfa_match_text(
    audio_stem: str = Query(..., description="Audio filename stem (without extension)"),
    text_dir: str = Query(..., description="Directory to search for matching text files"),
):
    """Fuzzy match text files whose names contain all fields from the audio stem.

    Splits audio_stem by underscore/hyphen into "fields", then returns files
    from text_dir whose stem contains every one of those fields (case-insensitive).
    """
    import re as _mfa_re2
    fields = _mfa_re2.split(r'[_\-]+', audio_stem.lower())
    # Keep fields with at least 2 chars (filter noise like single digits/letters
    # that would match too broadly)
    fields = [f for f in fields if len(f) >= 2]

    if not fields:
        return {"audio_stem": audio_stem, "fields": [], "matches": [],
                "hint": "Audio stem too short to extract meaningful fields"}

    matches = []
    if os.path.isdir(text_dir):
        for name in sorted(os.listdir(text_dir)):
            full = os.path.join(text_dir, name)
            if not os.path.isfile(full):
                continue
            stem_lower = os.path.splitext(name)[0].lower()
            if all(f in stem_lower for f in fields):
                ext = os.path.splitext(name)[1].lower()
                size_kb = round(os.path.getsize(full) / 1024, 1)
                matches.append({
                    "name": name,
                    "path": full.replace("\\", "/"),
                    "ext": ext,
                    "size_kb": size_kb,
                    "is_txt": ext == ".txt",
                })

    return {"audio_stem": audio_stem, "fields": fields, "matches": matches}


# Copy a matched text file to MFA TXT directory with audio stem naming
@app.post("/api/mfa/copy-text-for-align")
def mfa_copy_text_for_align(
    text_path: str = Query(..., description="Source text file path"),
    audio_stem: str = Query(..., description="Target audio stem for naming"),
    output_dir: str = Query(..., description="MFA TXT output directory"),
    language: str = Query("zh", description="Language: jp or zh"),
    dict_path: str = Query("", description="Optional MFA dictionary path for OOV checking"),
):
    """Tokenize an external text file and save to the MFA TXT directory.

    Unlike the old behaviour, this runs tokenization (jieba for zh, SudachiPy for jp)
    so the output is ready for MFA alignment without Step 2.
    """
    if not os.path.exists(text_path):
        raise HTTPException(status_code=404, detail=f"Text file not found: {text_path}")

    try:
        with open(text_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read text file: {e}")

    if not content.strip():
        raise HTTPException(status_code=400, detail="Text file is empty")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, audio_stem + ".txt")

    if language == "jp":
        # Japanese: tokenize via SudachiPy (inline subprocess to isolate deps)
        python_exe = _get_mfa_config("MFA_PYTHON", sys.executable)
        tokenize_script = (
            "import sys, os\n"
            "text = sys.stdin.read()\n"
            "try:\n"
            "    from sudachipy import dictionary, tokenizer\n"
            "    tk = dictionary.Dictionary().create()\n"
            "    mode = tokenizer.Tokenizer.SplitMode." + {"A": "A", "B": "B", "C": "C"}.get(
                _get_mfa_config("MFA_DEFAULT_TOKENIZE_MODE", "A"), "A") + "\n"
            "    tokens = [m.surface() for m in tk.tokenize(text, mode)]\n"
            "except ImportError:\n"
            "    # Fallback: just split by common delimiters\n"
            "    import re\n"
            "    tokens = [t for t in re.split(r'[\\s。、，！？…「」『』（）\\(\\)]+', text) if t]\n"
            "print(' '.join(tokens))\n"
        )
        try:
            result = subprocess.run(
                [python_exe, "-c", tokenize_script],
                input=content, capture_output=True, text=True, timeout=60,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or "SudachiPy tokenization failed")
            tokenized = result.stdout.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Japanese tokenization failed: {e}")
    else:
        # Chinese: convert to pinyin for fullpinyin_enword dictionary
        tokenized = _tokenize_chinese_to_pinyin(content)

    # Write tokenized result
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tokenized + "\n")

    preview = tokenized[:500] if len(tokenized) > 500 else tokenized
    return {
        "status": "ok",
        "output_path": out_path.replace("\\", "/"),
        "content_preview": preview,
        "content_length": len(tokenized),
        "language": language,
        "tokenized": True,
    }


# Read text file content for display in UI
@app.get("/api/mfa/read-text-content")
def mfa_read_text_content(path: str = Query(..., description="Path to text file")):
    """Read a text file's content for display in the post-processing section."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if not path.lower().endswith((".txt", ".lab", ".text", ".transcription")):
        raise HTTPException(status_code=400, detail="File must be a text file (.txt/.lab/.text)")

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    return {"path": path.replace("\\", "/"), "content": content, "size": len(content)}


# SRT to JSONL import helper
@app.post("/api/mfa/srt-to-jsonl")
def mfa_srt_to_jsonl(
    srt_path: str = Query(..., description="Path to SRT subtitle file"),
    wav_file: str = Query(..., description="Matching WAV filename (e.g. audio.wav)"),
    output_jsonl: str = Query("", description="JSONL output path. Defaults to <srt_dir>/generated.jsonl"),
    text_key: str = Query("text", description="JSON key for the transcript text"),
    wav_key: str = Query("wav_file", description="JSON key for the wav filename"),
):
    """Parse an SRT file, extract plain text, and append a JSONL entry.

    Reads the SRT, strips sequence numbers and timestamps, joins text lines
    into a single transcript string. The resulting JSON line is appended to
    the specified JSONL file (or a generated.jsonl next to the SRT if not given).
    """
    srt_path_obj = Path(srt_path)
    if not srt_path_obj.exists():
        raise HTTPException(status_code=404, detail=f"SRT file not found: {srt_path}")

    # Read and parse SRT
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    text_lines = []
    raw_lines = content.splitlines()
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip sequence numbers (pure digits, including full-width)
        if stripped.isdigit():
            continue
        # Skip timestamp lines
        if "-->" in stripped:
            continue
        text_lines.append(stripped)

    full_text = "".join(text_lines)

    if not full_text:
        raise HTTPException(status_code=400, detail="No text content found in SRT file")

    # Determine JSONL output path
    if output_jsonl:
        jsonl_path = Path(output_jsonl)
    else:
        # Default: data/mfa/jsonl/<srt_stem>.jsonl
        srt_stem = srt_path_obj.stem
        jsonl_path = Path(os.path.join(MFA_DIR, "jsonl", srt_stem + ".jsonl"))

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        text_key: full_text,
        wav_key: Path(wav_file).name,
    }

    # Append entry to JSONL
    with open(str(jsonl_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Count total entries in JSONL
    total = 0
    if jsonl_path.exists():
        with open(str(jsonl_path), "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

    return {
        "status": "ok",
        "srt_path": str(srt_path_obj),
        "srt_lines": len(raw_lines),
        "text_lines": len(text_lines),
        "extracted_text": full_text,
        "jsonl_path": str(jsonl_path),
        "jsonl_entry": entry,
        "jsonl_total_entries": total,
        "text_preview": full_text[:300] + ("..." if len(full_text) > 300 else ""),
    }


# SRT → TXT: parse SRT, tokenize, write TXT directly (skip JSONL)
@app.post("/api/mfa/srt-to-txt")
def mfa_srt_to_txt(
    srt_path: str = Query(..., description="Path to SRT subtitle file"),
    wav_file: str = Query(..., description="Matching WAV filename (e.g. audio.wav)"),
    output_dir: str = Query("", description="TXT output directory. Defaults to data/mfa/txt"),
    language: str = Query("zh", description="Language: zh or jp"),
):
    """Parse an SRT file, extract text, tokenize for MFA, and write a TXT file.

    Unlike srt-to-jsonl, this runs the full tokenization pipeline (SudachiPy for JP)
    and outputs a ready-to-use MFA TXT file.
    """
    srt_path_obj = Path(srt_path)
    if not srt_path_obj.exists():
        raise HTTPException(status_code=404, detail=f"SRT file not found: {srt_path}")

    # Read and parse SRT — extract plain text
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read()
    text_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        text_lines.append(stripped)
    full_text = "".join(text_lines)
    if not full_text:
        raise HTTPException(status_code=400, detail="No text content found in SRT file")

    # Determine output paths
    if output_dir:
        txt_dir = Path(output_dir)
    else:
        txt_dir = Path(MFA_TXT_DIR)
    txt_dir.mkdir(parents=True, exist_ok=True)
    wav_stem = Path(wav_file).stem
    txt_path = txt_dir / f"{wav_stem}.txt"

    if language == "jp":
        # Use SudachiPy via generate_wav_txt.py (temp JSONL approach)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as tmp:
            json.dump({"text": full_text, "wav_file": Path(wav_file).name}, tmp, ensure_ascii=False)
            tmp.write("\n")
            tmp_jsonl = tmp.name
        try:
            python_exe = _get_mfa_config("MFA_PYTHON", "python")
            script = os.path.join(MFA_SCRIPTS_DIR, "generate_wav_txt.py")
            dict_path = _get_mfa_config("MFA_DICT_PATH", MFA_DICT_PATH)
            cmd = [python_exe, script, "--jsonl", tmp_jsonl, "--output-dir", str(txt_dir),
                   "--mode", "A", "--overwrite"]
            if os.path.exists(dict_path):
                cmd.extend(["--dict-path", dict_path, "--max-merge-tokens", "5"])
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    cwd=PROJECT_ROOT, timeout=300)
            stdout = result.stdout
            if result.returncode != 0:
                stdout = (result.stdout + "\n" + result.stderr) if result.stderr else result.stdout
        finally:
            try: os.unlink(tmp_jsonl)
            except Exception: pass
    else:
        # Chinese: convert to pinyin for fullpinyin_enword dictionary
        tokenized = _tokenize_chinese_to_pinyin(full_text)
        txt_path.write_text(tokenized + "\n", encoding="utf-8")
        stdout = f"Chinese pinyin: {len(tokenized.split())} syllables → {txt_path}"

    txt_exists = txt_path.exists()
    txt_content = ""
    if txt_exists:
        with open(str(txt_path), "r", encoding="utf-8") as f:
            txt_content = f.read()[:500]

    return {
        "status": "ok",
        "srt_path": str(srt_path_obj),
        "extracted_text": full_text[:300],
        "txt_path": str(txt_path),
        "txt_content_preview": txt_content,
        "txt_exists": txt_exists,
        "language": language,
        "stdout": stdout[-500:] if stdout else "",
    }


# Job polling and management
@app.get("/api/mfa/job/{job_id}")
def get_mfa_job(job_id: str):
    """Poll MFA job status."""
    with _mfa_lock:
        job = _mfa_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/mfa/jobs")
def list_mfa_jobs():
    """List all MFA jobs (latest 20)."""
    with _mfa_lock:
        jobs = sorted(_mfa_jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)[:20]
    return {"jobs": jobs, "count": len(jobs)}


@app.delete("/api/mfa/job/{job_id}")
def cancel_mfa_job(job_id: str):
    """Cancel a running MFA job."""
    with _mfa_lock:
        job = _mfa_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not running")
    job["cancelled"] = True
    job["status"] = "cancelled"
    job["current_step"] = "已取消"
    _save_mfa_jobs()
    return {"status": "ok"}


@app.delete("/api/mfa/jobs")
def clear_mfa_jobs(status: str = Query("completed", description="Status filter: completed, failed, cancelled")):
    """Clear MFA jobs by status."""
    with _mfa_lock:
        to_delete = [jid for jid, j in _mfa_jobs.items() if j.get("status") == status]
        for jid in to_delete:
            del _mfa_jobs[jid]
        _save_mfa_jobs()
    return {"status": "ok", "deleted": len(to_delete)}

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Serve frontend static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"Frontend: {FRONTEND_DIR}")
    print(f"Data dir: {PROJECT_ROOT}/data")
    print(f"Video dir: {VIDEO_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=5800)
