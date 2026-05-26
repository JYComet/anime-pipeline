"""
File watcher: monitors download/video directories for new MKV files
and auto-triggers the extract+split pipeline.
"""
import os
import time
import threading
from config import DOWNLOAD_DIR, COMICUT_ROOT, VIDEO_DIR

# Set of known files to avoid re-processing
_known_files: set[str] = set()
_watcher_running = False
_watcher_thread = None
_on_new_mkv = None  # callback(mkv_path)


VIDEO_EXTS = {".mkv", ".mp4"}


def _scan_directory(directory: str) -> list[str]:
    """Scan a directory recursively for video files not yet seen."""
    new_files = []
    if not os.path.isdir(directory):
        return new_files
    for root, dirs, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            full = os.path.join(root, f)
            if full in _known_files:
                continue
            # Skip files still being written (size changing or .aria2 temp)
            if f.endswith(".aria2"):
                continue
            try:
                size1 = os.path.getsize(full)
                if size1 == 0:
                    continue
                time.sleep(2)
                size2 = os.path.getsize(full)
                if size1 != size2:
                    continue
            except Exception:
                continue
            _known_files.add(full)
            new_files.append(full)
    return new_files


def _watcher_loop(interval: int = 5):
    """Background loop: scan directories every N seconds."""
    global _watcher_running
    # Pre-seed known files: mark ALL existing files as known so startup doesn't spam
    for d in [DOWNLOAD_DIR, VIDEO_DIR]:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for f in files:
                    if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                        full = os.path.join(root, f)
                        _known_files.add(full)

    while _watcher_running:
        try:
            for d in [VIDEO_DIR, DOWNLOAD_DIR]:
                new = _scan_directory(d)
                for path in new:
                    print(f"[watcher] New video detected: {path}")
                    if _on_new_mkv:
                        try:
                            _on_new_mkv(path)
                        except Exception as e:
                            print(f"[watcher] callback error: {e}")
        except Exception as e:
            print(f"[watcher] scan error: {e}")
        time.sleep(interval)


def start_watcher(on_new_mkv_callback, interval: int = 5) -> bool:
    """Start the file watcher in a background thread.

    Args:
        on_new_mkv_callback: Called with full path when a new MKV is detected.
        interval: Scan interval in seconds.

    Returns:
        True if started, False if already running.
    """
    global _watcher_running, _watcher_thread, _on_new_mkv
    if _watcher_running:
        return False

    _on_new_mkv = on_new_mkv_callback
    _watcher_running = True
    _watcher_thread = threading.Thread(target=_watcher_loop, args=(interval,), daemon=True)
    _watcher_thread.start()
    print("[watcher] Started monitoring downloads/ and video/ for new MKV files")
    return True


def stop_watcher():
    """Stop the file watcher."""
    global _watcher_running
    _watcher_running = False


def is_watching() -> bool:
    return _watcher_running
