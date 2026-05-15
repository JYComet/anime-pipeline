"""
qBittorrent launcher — opens qBittorrent GUI with magnet link,
file watcher handles subsequent processing.
"""
import os
import subprocess
import time
import requests
from config import DOWNLOAD_DIR, QBITTORRENT_EXE


def add_magnet(magnet: str, save_path: str = "") -> bool:
    """Launch qBittorrent with a magnet link.

    qBittorrent will open (or bring to front if already running)
    and add the magnet to the download queue.
    The file watcher monitors the download directory for completion.

    Returns True if qBittorrent launched successfully.
    """
    if not save_path:
        save_path = DOWNLOAD_DIR
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(QBITTORRENT_EXE):
        raise RuntimeError(f"qBittorrent 未找到: {QBITTORRENT_EXE}")

    try:
        subprocess.Popen(
            [QBITTORRENT_EXE, magnet],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        raise RuntimeError(f"无法启动 qBittorrent: {e}")


def ensure_running() -> bool:
    """Check if qBittorrent executable exists."""
    return os.path.exists(QBITTORRENT_EXE)


def list_torrents() -> list[dict]:
    """Not supported via CLI mode — returns empty list."""
    return []


def wait_for_completion(hash_or_magnet: str, timeout: int = 0, on_progress=None) -> list[str]:
    """Launch qBittorrent and return immediately.

    Actual download is handled by qBittorrent GUI.
    File watcher monitors the download directory and triggers processing.
    """
    magnet = hash_or_magnet if hash_or_magnet.startswith("magnet:") else ""
    if magnet:
        add_magnet(magnet)

    # Return empty — file watcher handles completion
    if on_progress:
        on_progress(0, 0, "等待 qBittorrent 下载...")
    return []


def shutdown():
    """No-op — qBittorrent GUI is managed by the user."""
    pass
