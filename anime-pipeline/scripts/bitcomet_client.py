"""
BitComet launcher — opens BitComet with magnet link.
Downloads go to the user's configured BitComet default directory,
or to data/downloads if configured via set_download_dir().
"""
import os
import subprocess
import time
import xml.etree.ElementTree as ET

from config import DOWNLOAD_DIR

BITCOMET_EXE = r"C:\Program Files\BitComet\BitComet.exe"
BITCOMETD_EXE = r"C:\Program Files\BitComet\bitcometd.exe"
BITCOMET_DIR = os.path.join(os.environ.get("APPDATA", ""), "BitComet")
BITCOMET_CONFIG = os.path.join(BITCOMET_DIR, "BitComet.xml")


def _get_default_download_dir() -> str:
    """Read BitComet's current default download directory from config."""
    if not os.path.exists(BITCOMET_CONFIG):
        return ""
    try:
        tree = ET.parse(BITCOMET_CONFIG)
        root = tree.getroot()
        settings = root.find("Settings")
        if settings is not None:
            elem = settings.find("DefaultDownloadPath")
            if elem is not None and elem.text:
                return elem.text
    except Exception:
        pass
    return ""


def set_download_dir(path: str = "") -> tuple[bool, str]:
    """Configure BitComet to download to a specific directory.

    Returns (success, old_path) so the caller can restore or notify.
    """
    if not path:
        path = DOWNLOAD_DIR

    os.makedirs(path, exist_ok=True)

    old_path = _get_default_download_dir()
    if old_path == path:
        return True, old_path

    if not os.path.exists(BITCOMET_CONFIG):
        return False, old_path

    # BitComet must not be running while we edit its config
    _ensure_not_running()

    try:
        tree = ET.parse(BITCOMET_CONFIG)
        root = tree.getroot()
        settings = root.find("Settings")
        if settings is None:
            settings = ET.SubElement(root, "Settings")

        elem = settings.find("DefaultDownloadPath")
        if elem is not None:
            elem.text = path
        else:
            elem = ET.SubElement(settings, "DefaultDownloadPath")
            elem.text = path

        tree.write(BITCOMET_CONFIG, encoding="utf-8", xml_declaration=True)
        print(f"[BitComet] Download dir: {old_path!r} -> {path!r}")
        return True, old_path
    except Exception as e:
        print(f"[BitComet] Failed to set download dir: {e}")
        return False, old_path


def restore_download_dir(path: str) -> bool:
    """Restore BitComet's default download directory to a previous value."""
    if not path or not os.path.exists(BITCOMET_CONFIG):
        return False
    try:
        tree = ET.parse(BITCOMET_CONFIG)
        root = tree.getroot()
        settings = root.find("Settings")
        if settings is not None:
            elem = settings.find("DefaultDownloadPath")
            if elem is not None:
                elem.text = path
                tree.write(BITCOMET_CONFIG, encoding="utf-8", xml_declaration=True)
                return True
    except Exception:
        pass
    return False


def _ensure_not_running():
    """Wait for BitComet GUI to close."""
    try:
        for _ in range(15):
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq BitComet.exe"],
                capture_output=True, text=True, timeout=5,
            )
            if "BitComet.exe" not in result.stdout:
                return True
            time.sleep(0.5)
    except Exception:
        pass
    return False


def ensure_running() -> bool:
    """Check if BitComet executable exists."""
    return os.path.exists(BITCOMET_EXE)


def add_magnet(magnet: str, save_path: str = "") -> bool:
    """Launch BitComet with a magnet link.

    BitComet will open (or bring to front if already running)
    and add the magnet to the download queue.
    The file watcher monitors the download directory for completion.

    Args:
        magnet: Magnet URI to add.
        save_path: Optional — if set, sets BitComet's default download
                   directory to this path before launching.

    Returns True if BitComet launched successfully.
    """
    if not os.path.exists(BITCOMET_EXE):
        raise RuntimeError(f"BitComet 未找到: {BITCOMET_EXE}")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        set_download_dir(save_path)

    try:
        subprocess.Popen(
            [BITCOMET_EXE, magnet],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        raise RuntimeError(f"无法启动 BitComet: {e}")


def list_torrents() -> list[dict]:
    """Read active torrents from BitComet's Downloads.xml."""
    downloads_xml = os.path.join(BITCOMET_DIR, "Downloads.xml")
    if not os.path.exists(downloads_xml):
        return []

    torrents = []
    try:
        tree = ET.parse(downloads_xml)
        root = tree.getroot()
        torrents_elem = root.find("Torrents")
        if torrents_elem is None:
            return []

        for t in torrents_elem.findall("Torrent"):
            size = int(t.get("Size", 0))
            downloaded = int(t.get("SelectedFileDownload", 0))
            left = int(t.get("Left", 0))
            name = t.get("ShowName", t.get("SaveName", ""))
            savedir = t.get("SaveDirectory", "")
            info_hash = t.get("InfoHashHex", "")

            status = "downloading"
            if left == 0 and size > 0:
                status = "complete"

            progress = 0
            if size > 0:
                progress = int((downloaded / size) * 100)

            torrents.append({
                "gid": info_hash[:12],
                "name": name,
                "status": status,
                "progress": progress,
                "size": size,
                "downloaded": downloaded,
                "save_dir": savedir,
                "info_hash": info_hash,
            })
    except Exception as e:
        print(f"[BitComet] Failed to read torrents: {e}")

    return torrents


def wait_for_completion(hash_or_magnet: str, timeout: int = 0, on_progress=None) -> list[str]:
    """Launch BitComet and return immediately.

    Actual download is handled by BitComet GUI.
    File watcher monitors the download directory and triggers processing.
    """
    magnet = hash_or_magnet if hash_or_magnet.startswith("magnet:") else ""
    if magnet:
        add_magnet(magnet)

    if on_progress:
        on_progress(0, 0, "等待 BitComet 下载...")
    return []


def shutdown():
    """No-op — BitComet GUI is managed by the user."""
    pass
