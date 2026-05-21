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
    PROJECT_ROOT, DATA_DIR, DOWNLOAD_DIR, SUBTITLE_DIR, CLIPS_DIR, APPROVED_DIR, CLEANED_DIR,
    CLEANED_UNREVIEWED_DIR, DENOISED_APPROVED_DIR, STITCHED_DIR,
    ASR_DIR, ASR_AUDIO_DIR, ASR_SUBTITLE_DIR,
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

# --- Video directory for testing ---
COMICUT_ROOT = os.path.dirname(PROJECT_ROOT)
VIDEO_DIR = os.path.join(COMICUT_ROOT, "video")
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


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running job."""
    ok = pipeline.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    return {"status": "ok"}


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
    """Approve a clip: move it to the approved folder."""
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
    shutil.move(src, dst)

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

    return {"status": "ok", "action": "approved", "moved_to": dst}


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

def _find_denoised_dirs() -> list[str]:
    """Find all directories with denoised WAV files (from cleaned/ and cleaned_unreviewed/)."""
    dirs = []
    seen = set()
    for base in [CLEANED_DIR, CLEANED_UNREVIEWED_DIR]:
        if not os.path.exists(base):
            continue
        for root, subdirs, files in os.walk(base):
            if any(_is_denoised_wav(f) for f in files):
                if root not in seen:
                    dirs.append(root)
                    seen.add(root)
    return dirs


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
def list_denoise_review_dirs():
    """List all directories with denoised audio available for review."""
    dirs = _find_denoised_dirs()
    # Determine source type and display name
    result = []
    for d in dirs:
        if d.startswith(CLEANED_UNREVIEWED_DIR):
            source = "未审核降噪"
            rel = os.path.relpath(d, CLEANED_UNREVIEWED_DIR)
        elif d.startswith(CLEANED_DIR):
            source = "已审核降噪"
            rel = os.path.relpath(d, CLEANED_DIR)
        else:
            source = "未知"
            rel = os.path.basename(d)
        result.append({"name": rel.replace("\\", "/"), "path": d.replace("\\", "/"), "source": source})
    return {"dirs": result}


@app.get("/api/denoise-review/clips")
def list_denoise_review_clips(
    clip_dir: str = Query(..., description="Path to denoised audio directory"),
):
    """List denoised WAV files available for review, with review status."""
    if not os.path.exists(clip_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    state = _get_denoise_review_state(clip_dir)
    approved = state.get("approved", [])
    skipped = state.get("skipped", [])

    clips = []
    for f in sorted(os.listdir(clip_dir)):
        if not _is_denoised_wav(f):
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


@app.get("/api/denoise-review/stream")
def denoise_review_stream(path: str = Query(..., description="Full path to audio file")):
    """Stream a denoised audio file for playback in the browser."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav")


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
        os.startfile(path)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        mode: "subtitle" | "duration" | "size"
        video_path: str (required)
        subtitle_path: str (for subtitle mode)
        padding: float (subtitle mode, default 0.1)
        group_count: int (subtitle mode, default 1)
        segment_duration: float (duration mode, seconds)
        target_size_mb: float (size mode, MB)
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

    def run():
        job = pipeline.create_job(job_id, title=title)
        job.mkv_path = video_path
        job.status = "running"
        job.current_step = mode
        job.progress = 5

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
    model: str = Query("sensevoice", description="ASR model key"),
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


@app.delete("/api/asr/jobs")
def clear_asr_jobs():
    """Clear completed and failed ASR jobs."""
    with _asr_lock:
        to_remove = [
            jid for jid, j in _asr_jobs.items()
            if j.get("status") in ("completed", "failed")
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
