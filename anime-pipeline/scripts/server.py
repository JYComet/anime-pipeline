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

from fastapi import FastAPI, Query, Body, HTTPException, UploadFile, File
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
    EMOTION_DIR, EMOTION_DENOISE_DIR,
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
_denoise_lock = threading.Lock()


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
_pipeline_video_lock = threading.Lock()


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

    # Restore download jobs from aria2c state
    def restore_jobs():
        import time
        time.sleep(3)
        try:
            from aria2_rpc import list_all
            for item in list_all():
                name = item.get("bittorrent", {}).get("info", {}).get("name", "")
                if not name:
                    continue
                total = int(item.get("totalLength", 0))
                completed = int(item.get("completedLength", 0))
                pct = (completed / total * 100) if total > 0 else 0
                gid = item.get("gid", "")[:12]
                job = pipeline.create_job(
                    f"aria2_{gid}",
                    title=name,
                    magnet=item.get("magnetUri", ""),
                )
                job.status = "download_submitted"
                job.progress = 5 + pct * 0.25
                job.current_step = f"download ({name[:25]}... {pct:.0f}%)" if pct > 0 else "download"
                from pipeline import StepResult, StepStatus
                job.steps = [StepResult(
                    step="download",
                    status=StepStatus.COMPLETED,
                    message=f"aria2c 后台下载中 ({completed/1024/1024:.0f}/{total/1024/1024:.0f}MB, {pct:.0f}%)",
                )]
                print(f"[startup] Restored job {job.job_id}: {name[:40]} ({pct:.0f}%)")
        except Exception as e:
            print(f"[startup] Failed to restore jobs: {e}")
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
                from pipeline import StepResult, StepStatus
                job.steps = [
                    StepResult("extract_subtitles", StepStatus.SKIPPED, "Used external subtitle"),
                    StepResult("split_video", StepStatus.COMPLETED, f"Created {len(clips)} clips"),
                ]
            except Exception as e:
                job.status = "failed"
                from pipeline import StepResult, StepStatus
                job.steps = [
                    StepResult("extract_subtitles", StepStatus.SKIPPED, "Used external subtitle"),
                    StepResult("split_video", StepStatus.FAILED, str(e)),
                ]
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
    return {"paths": paths, "denoise_default_steps": steps, "pv_default_steps": pv_steps}


@app.post("/api/settings")
def save_settings(payload: dict = Body(...)):
    """Save path settings and/or default step configs."""
    from config import save_settings
    paths = payload.get("paths", {})
    steps = payload.get("denoise_default_steps", None)
    pv_steps = payload.get("pv_default_steps", None)
    save_settings(paths, steps, pv_steps)
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
    """List all clip directories under data/clips/ with stats."""
    dirs = []
    if os.path.exists(CLIPS_DIR):
        for entry in sorted(os.listdir(CLIPS_DIR)):
            full = os.path.join(CLIPS_DIR, entry)
            if os.path.isdir(full):
                mp4_count = len([f for f in os.listdir(full) if f.endswith(".mp4")])
                mkvs = [f for f in os.listdir(full) if f.endswith(".mkv")]
                dirs.append({
                    "name": entry,
                    "path": full,
                    "clip_count": mp4_count,
                    "total_size_mb": round(
                        sum(os.path.getsize(os.path.join(full, f))
                            for f in os.listdir(full)
                            if f.endswith(".mp4") and os.path.isfile(os.path.join(full, f))
                        ) / 1024 / 1024, 1,
                    ),
                    "mtime": os.path.getmtime(full),
                })
    dirs.sort(key=lambda d: d["mtime"], reverse=True)
    return {"dirs": dirs, "count": len(dirs)}


@app.get("/api/split/clips/{video_name}")
def list_split_clips(video_name: str):
    """List all clips in data/clips/{video_name}/."""
    clips_dir = os.path.join(CLIPS_DIR, video_name)
    if not os.path.exists(clips_dir):
        return {"clips": [], "count": 0, "video_name": video_name}

    clips = []
    for f in sorted(os.listdir(clips_dir)):
        if not f.endswith(".mp4"):
            continue
        full = os.path.join(clips_dir, f)
        clips.append({
            "name": f,
            "path": full,
            "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
        })
    return {"clips": clips, "count": len(clips), "video_name": video_name}


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

                from split_video import split_video_by_subtitle
                job.current_step = "subtitle_split"
                job.progress = 10

                # If group_count > 1, we split normally then can post-process
                # For now, split one-per-sub
                clips = split_video_by_subtitle(
                    video_path, subtitle_path,
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
_asr_lock = threading.Lock()


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
    language: str = Query("ja", description="Language code or 'auto'"),
    device: str = Query("cuda", description="Device: cuda or cpu"),
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
            "status": "pending",
            "progress": 0,
            "current_step": "",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }

    def _run():
        from asr_pipeline import run_asr_pipeline, ASR_MODELS

        with _asr_lock:
            job = _asr_jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"

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
                progress_callback=on_progress,
            )

            with _asr_lock:
                j = _asr_jobs.get(job_id)
                if j:
                    j["status"] = "completed"
                    j["progress"] = 100
                    j["current_step"] = "处理完成"
                    j["result"] = result

        except Exception as e:
            with _asr_lock:
                j = _asr_jobs.get(job_id)
                if j:
                    j["status"] = "failed"
                    j["error"] = str(e)
                    j["current_step"] = f"错误: {e}"

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
    return {"cleared": len(to_remove)}


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
    language = payload.get("language", "ja")
    device = payload.get("device", "cuda")
    output_base = payload.get("output_dir", "").strip()
    selected_files = payload.get("selected_files", None)  # optional list of file names

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
        "results": [],
        "errors": [],
    }

    with _asr_lock:
        _asr_jobs[job_id] = job

    def _run():
        with _asr_lock:
            j = _asr_jobs.get(job_id)
            if not j:
                return
            j["status"] = "running"
            j["current_step"] = "加载模型中..."

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
                    progress_callback=on_progress,
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
_asr_compare_lock = threading.Lock()
_asr_compare_results: dict[str, dict] = {}  # keyed by audio path


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
    language: str = Query("ja"),
    device: str = Query("cuda"),
    dirs: list[str] = Query([], description="Optional list of source directory names to process"),
    model_a: str = Query("qwen3-asr"),
    model_b: str = Query("cohere-transcribe"),
):
    """Queue WAV files from cleaned_unreviewed/ and process sequentially.

    If dirs is provided, only files from those source directories are processed.
    """
    from asr_pipeline import compare_asr_pipeline, ASR_MODELS

    all_dirs = _find_unreviewed_wav_files()
    dir_filter = set(dirs) if dirs else None

    all_files = []
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
        msg = "No WAV files found"
        if dir_filter:
            msg += f" in selected directories: {', '.join(sorted(dir_filter))}"
        raise HTTPException(status_code=404, detail=msg)

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
        }

    def _run():
        from asr_pipeline import compare_asr_pipeline, ASR_MODELS, COMPARE_MODELS

        with _asr_compare_lock:
            job = _asr_compare_jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"

        files = all_files
        total = len(files)

        for idx, f in enumerate(files):
            with _asr_compare_lock:
                job = _asr_compare_jobs.get(job_id)
                if not job or job.get("_cancelled"):
                    if job:
                        job["status"] = "cancelled"
                        job["current_step"] = "已中止"
                    return

            audio_path = f["path"]
            audio_name = f["name"]

            def on_progress(step, pct):
                with _asr_compare_lock:
                    j = _asr_compare_jobs.get(job_id)
                    if j:
                        j["current_step"] = f"[{idx + 1}/{total}] {audio_name} — {step}"

            try:
                result = compare_asr_pipeline(
                    audio_path,
                    language=language,
                    device=device,
                    progress_callback=on_progress,
                    source_dir=f["source_dir"],
                    model_a=model_a,
                    model_b=model_b,
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

        with _asr_compare_lock:
            j = _asr_compare_jobs.get(job_id)
            if j:
                j["status"] = "completed"
                j["progress"] = 100
                j["current_step"] = f"处理完成 — {total} 个文件, {j['flagged_count']} 个异常"

        # Persist results for later queries
        with _asr_compare_lock:
            j = _asr_compare_jobs.get(job_id)
            if j:
                for path, result in j["results"].items():
                    _asr_compare_results[path] = result

    threading.Thread(target=_run, daemon=True).start()

    return {"job_id": job_id, "status": "started", "total": len(all_files)}


@app.get("/api/asr-compare/job/{job_id}")
def get_asr_compare_job(job_id: str):
    """Poll comparison job status."""
    with _asr_compare_lock:
        job = _asr_compare_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


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
        if job.get("status") not in ("running", "pending"):
            return {"status": "error", "detail": f"Job is {job.get('status')}, cannot cancel"}
        job["_cancelled"] = True
        job["current_step"] = "正在中止..."
    return {"status": "ok", "detail": "Cancelling..."}


@app.get("/api/asr-compare/results")
def list_asr_compare_results():
    """Get all comparison results."""
    with _asr_compare_lock:
        results = list(_asr_compare_results.values())

    # Also pull from latest completed job
    with _asr_compare_lock:
        for j in _asr_compare_jobs.values():
            if j.get("status") == "completed":
                for path, result in j.get("results", {}).items():
                    if path not in _asr_compare_results:
                        _asr_compare_results[path] = result

    results = list(_asr_compare_results.values())
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

    return {"status": "ok", "action": "discarded"}


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

    threading.Thread(target=_run_pipeline_video, args=(job_id,), daemon=True).start()

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
    return {"status": "ok", "message": "已取消任务"}


# ============================================================
# Pipeline Video — background processing
# ============================================================

def _convert_to_wav_ffmpeg(input_path: str, sample_rate: int = 44100, channels: int = 2) -> str:
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
            print(f"[pipeline-video] ffmpeg convert error for {input_path}: {result.stderr[:300]}")
    except Exception as e:
        print(f"[pipeline-video] ffmpeg convert exception for {input_path}: {e}")
    return ""


def _get_optimal_wav_params(enabled_steps: list) -> tuple:
    """Choose optimal WAV sample rate and channels based on enabled pipeline steps.

    ClearVoice models (enhance, super_resolve) work natively at 48kHz.
    Demucs BGM separation needs stereo for source panning.
    ASR models need 16kHz mono.
    Default: 44100 Hz stereo for general-purpose / unknown pipelines.
    """
    if "enhance" in enabled_steps or "super_resolve" in enabled_steps:
        return 48000, 1  # ClearVoice native rate, mono (anime is mono source)
    if "music_separate" in enabled_steps:
        return 44100, 1  # Demucs works on mono too (spectral separation)
    if "asr" in enabled_steps:
        return 16000, 1  # ASR native rate, mono
    return 44100, 2  # default stereo


def _run_pipeline_video(job_id: str):
    """Run the video pipeline in background.

    Steps are configurable via job.steps_config.
    Each file flows through all enabled steps independently.
    Per-step BoundedSemaphore(2) controls model concurrency.
    Intermediate files kept in a job-specific temp directory, cleaned on success.
    """
    import concurrent.futures
    import shutil as _shutil

    with _pipeline_video_lock:
        job = _pipeline_video_jobs.get(job_id)
    if not job:
        return

    job.status = "running"
    files = job.files
    folder = job.folder_name

    # Parse enabled steps from config (or default)
    steps_config = getattr(job, 'steps_config', None)
    if not steps_config:
        steps_config = [
            {"key": "duration_split", "enabled": True},
            {"key": "music_separate", "enabled": True},
            {"key": "enhance", "enabled": True},
            {"key": "super_resolve", "enabled": False},
            {"key": "asr", "enabled": True},
            {"key": "cut", "enabled": False},
        ]
    enabled_steps = [s["key"] for s in steps_config if s.get("enabled", True)]
    step_cfg_map = {s["key"]: s for s in steps_config}

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
            return

    # Job-specific temp directory for intermediate files
    temp_dir = os.path.join(TEMP_DIR, "pipeline_video", job_id)
    os.makedirs(temp_dir, exist_ok=True)

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
        "gpu":      threading.BoundedSemaphore(3),  # global GPU cap: up to 3 concurrent (diff models)
        "ffmpeg":   threading.BoundedSemaphore(6),  # global ffmpeg cap
        "convert":  threading.BoundedSemaphore(4),
        "music_separate": threading.BoundedSemaphore(1),  # singleton Demucs
        "enhance":  threading.BoundedSemaphore(1),  # singleton ClearVoice SE
        "super_resolve": threading.BoundedSemaphore(1),  # singleton ClearVoice SR
        "asr":      threading.BoundedSemaphore(1),  # singleton Qwen3-ASR (3.4 GB)
        "duration_split": threading.BoundedSemaphore(2),  # ffmpeg segment split
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
            if f.input_path != out:
                _shutil.copy2(f.input_path, out)
            state["wav"] = out
            f.status = "running"
            f.progress = 100
            f.steps.append({"step": "convert", "status": "completed", "message": "已是WAV格式"})
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
        f.steps.append({"step": "music_separate", "status": "running", "message": "加载 Demucs 模型..."})
        _update_pipeline_job_progress(job)

        src = state.get("wav", f.wav_path)
        out = _temp_path(f, "separated", ".wav", state)
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
        f.steps.append({"step": "enhance", "status": "running", "message": "加载模型中..."})
        _update_pipeline_job_progress(job)

        src = state.get("wav", f.wav_path)
        out = _temp_path(f, "enhance", ".wav", state)

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
        f.progress = 5
        f.steps.append({"step": "super_resolve", "status": "running", "message": "加载超分模型..."})
        _update_pipeline_job_progress(job)

        src = state.get("wav", f.wav_path)
        out = _temp_path(f, "super_resolve", ".wav", state)

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
        f.progress = 0
        _update_pipeline_job_progress(job)

        src = state.get("wav", f.wav_path)

        def on_progress(phase, msg):
            if isinstance(msg, str):
                f.steps.append({"step": f"asr_{phase}", "status": "running", "message": str(msg)[:80]})
            else:
                f.progress = float(msg) if isinstance(msg, (int, float)) else f.progress

        try:
            result = run_asr_on_audio(
                src, output_dir=temp_dir, model_key="qwen3-asr", language="zh",
                progress_callback=on_progress,
            )
            srt = result.get("srt_path", "")
            if srt and os.path.exists(srt):
                state["srt"] = srt
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

        base = os.path.splitext(f.name)[0]
        seg_dir = os.path.join(temp_dir, f"segments_{base}")
        os.makedirs(seg_dir, exist_ok=True)

        ext = os.path.splitext(src)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS

        # Pick optimal WAV params for downstream steps
        opt_sr, opt_ch = _get_optimal_wav_params(enabled_steps)

        if ext == ".wav":
            # Already WAV — split with -c copy, instant
            seg_pattern = os.path.join(seg_dir, f"{base}_%03d.wav")
            cmd = [
                FFMPEG, "-y", "-i", src,
                "-f", "segment", "-segment_time", str(segment_dur),
                "-c", "copy", seg_pattern,
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
            if len(segments) >= 3:
                # Remove first and last segment (opening/closing noise)
                removed_first = segments.pop(0)
                removed_last = segments.pop(-1)
                os.remove(removed_first)
                os.remove(removed_last)
                print(f"[pipeline-video] Removed first and last segment")
            if segments:
                state["duration_segments"] = segments
                state["seg_dir"] = seg_dir
                state["seg_ext"] = seg_ext  # track whether conversion needed
                state["opt_sr"] = opt_sr
                state["opt_ch"] = opt_ch
                f.progress = 100
                f.steps.append({"step": "duration_split", "status": "completed",
                                "message": f"时长切分: {len(segments)}段 (每段{segment_dur}秒, 已去除首尾)"})
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

        src = state.get("wav", f.wav_path)
        out_dir = os.path.join(temp_dir, f"cut_{os.path.splitext(f.name)[0]}")
        os.makedirs(out_dir, exist_ok=True)

        def on_progress(cur, total):
            if total > 0:
                f.progress = (cur / total) * 100

        try:
            clips = cut_audio_by_subtitle(
                audio_path=src, srt_path=srt, output_dir=out_dir,
                base_name=os.path.splitext(f.name)[0],
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

    def _process_one_file(f):
        try:
            state = file_state.setdefault(f.input_path, {})
            f.status = "running"
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
                    opt_sr = state.get("opt_sr", 44100)
                    opt_ch = state.get("opt_ch", 2)

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
                    convert_workers = min(len(segments), 4)
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
                    # Step C: Process segments through remaining steps
                    # Phase-batch: each step runs on all segments before moving to next.
                    import threading as _threading2
                    seg_lock = _threading2.Lock()
                    seg_count = len(segments)
                    seg_states = [{"wav": p, "seg_suffix": f"_seg{i:03d}"} for i, p in enumerate(segments)]
                    total_active_steps = sum(1 for s in post_steps if s != "convert")

                    active_idx = 0  # counts only non-convert steps
                    for step_key in post_steps:
                        if job.cancelled:
                            f.status = "error"; f.error = "任务已取消"
                            _update_pipeline_job_progress(job); return

                        # "convert" already done in Phase B (parallel segment conversion).
                        if step_key == "convert":
                            continue

                        handler = STEP_HANDLERS.get(step_key)
                        if not handler:
                            continue
                        step_sem = _sem.get(step_key, _threading2.BoundedSemaphore(2))

                        # Progress: split(0-5) + convert(5-15) + step_N(15+active*85/total .. )
                        step_base = 15 + active_idx * (85 // max(total_active_steps, 1))
                        step_range = 85 // max(total_active_steps, 1)
                        active_idx += 1
                        step_done = [0]  # mutable counter for threads
                        f.progress = step_base
                        _update_pipeline_job_progress(job)

                        step_workers = min(seg_count, 4)
                        step_futures = []
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=step_workers
                        ) as step_exec:
                            for idx, ss in enumerate(seg_states):
                                if job.cancelled:
                                    break

                                def _run_step(_idx, _ss, _step_key=step_key):
                                    if job.cancelled:
                                        return
                                    with seg_lock:
                                        f.steps.append({"step": _step_key, "status": "running",
                                                        "message": f"处理片段 {_idx+1}/{seg_count}"})
                                        _update_pipeline_job_progress(job)
                                    try:
                                        h = STEP_HANDLERS.get(_step_key)
                                        if h:
                                            is_gpu = _step_key in _GPU_STEPS
                                            if is_gpu:
                                                with _sem["gpu"]:
                                                    with step_sem:
                                                        h(f, job, _ss)
                                            else:
                                                with step_sem:
                                                    h(f, job, _ss)
                                    except Exception as e:
                                        with seg_lock:
                                            f.steps.append({"step": _step_key, "status": "error",
                                                            "message": str(e)[:100]})
                                    with seg_lock:
                                        step_done[0] += 1
                                        f.progress = step_base + int(step_done[0] / seg_count * step_range)
                                        _update_pipeline_job_progress(job)
                                    if "cut" not in enabled_steps and _step_key == post_steps[-1]:
                                        final_wav = _ss.get("wav", "")
                                        if final_wav and os.path.exists(final_wav):
                                            with seg_lock:
                                                f.output_clips.append(final_wav)

                                step_futures.append(step_exec.submit(_run_step, idx, ss))
                            concurrent.futures.wait(step_futures)

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
                            with _sem["gpu"]:
                                with sem:
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
                                    with _sem["gpu"]:
                                        with sem:
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
                        concurrent.futures.wait(seg_futures)
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
                                with _sem["gpu"]:
                                    with sem:
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
        except Exception as e:
            f.status = "error"
            f.error = str(e)[:200]
        _update_pipeline_job_progress(job)

    max_workers = min(len(files), 10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_one_file, f) for f in files]
        concurrent.futures.wait(futures)

    # Move final clips to output
    output_base = getattr(job, 'output_dir', PIPELINE_VIDEO_DIR) or PIPELINE_VIDEO_DIR
    final_dir = os.path.join(output_base, folder)
    os.makedirs(final_dir, exist_ok=True)
    for f in files:
        if f.output_clips:
            for clip in list(f.output_clips):
                if os.path.exists(clip):
                    dest = os.path.join(final_dir, os.path.basename(clip))
                    try:
                        _shutil.move(clip, dest)
                    except Exception:
                        try:
                            _shutil.copy2(clip, dest)
                        except Exception:
                            pass
            f.output_clips = [os.path.join(final_dir, os.path.basename(c)) for c in f.output_clips]

    had_errors = any(f.status == "error" for f in files)
    if not had_errors and not job.cancelled:
        try:
            _shutil.rmtree(temp_dir)
            print(f"[pipeline-video] Cleaned temp dir: {temp_dir}")
        except Exception as e:
            print(f"[pipeline-video] Failed to clean temp: {e}")

    for f in files:
        if f.status not in ("error",):
            f.status = "completed"
    job.status = "completed"
    job.progress = 100
    _update_pipeline_job_progress(job)

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
# Static frontend serving
# ============================================================

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
