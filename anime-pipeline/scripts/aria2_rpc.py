"""
aria2c JSON-RPC client for headless downloads.
Communicates with aria2c daemon running on port 6800.
"""
import os
import time
import json
import subprocess
import requests
from config import DOWNLOAD_DIR, ARIA2C, DATA_DIR

RPC_URL = "http://127.0.0.1:6800/jsonrpc"

# Session file for persisting download state across restarts
_ARIA2_SESSION_FILE = os.path.join(DATA_DIR, "aria2.session")

_trackers = [
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://open.demonii.com:1337/announce",
    "udp://tracker.torrent.eu.org:451/announce",
    "udp://explodie.org:6969/announce",
    "udp://tracker.coppersurfer.tk:6969/announce",
    "udp://9.rarbg.to:2710/announce",
    "udp://tracker.internetwarriors.net:1337/announce",
    "http://tracker.opentrackr.org:1337/announce",
    "http://open.acgnxtracker.com:80/announce",
    "https://trakx.herokuapp.com:443/announce",
    "udp://tracker.openbittorrent.com:6969/announce",
    "udp://tracker.publicbt.com:6961/announce",
]


def _rpc(method: str, params: list = None) -> dict:
    """Make a JSON-RPC call to aria2c."""
    payload = {
        "jsonrpc": "2.0",
        "id": "anime_pipeline",
        "method": method,
        "params": params or [],
    }
    try:
        resp = requests.post(RPC_URL, json=payload, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def ensure_running() -> bool:
    """Start aria2c RPC daemon if not running, or return True if already up."""
    # Check if already running
    result = _rpc("aria2.getVersion")
    if "result" in result:
        return True

    # Start daemon
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    # Ensure session file exists (aria2c won't create it automatically)
    if not os.path.exists(_ARIA2_SESSION_FILE):
        open(_ARIA2_SESSION_FILE, "w").close()
    cmd = [
        ARIA2C,
        "--enable-rpc",
        "--rpc-listen-port=6800",
        "--rpc-allow-origin-all=true",
        "--rpc-listen-all=false",
        "--dir", DOWNLOAD_DIR,
        "--save-session", _ARIA2_SESSION_FILE,
        "--save-session-interval=30",
        "--input-file", _ARIA2_SESSION_FILE,
        "--seed-time=0",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--enable-dht=true",
        "--dht-listen-port=6881-6999",
        "--dht-entry-point=router.bittorrent.com:6881",
        "--dht-entry-point=dht.transmissionbt.com:6881",
        "--dht-entry-point=router.utorrent.com:6881",
        "--bt-enable-lpd=true",
        "--enable-peer-exchange=true",
        "--daemon=true",
    ]

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[aria2] Failed to start daemon: {e}")
        return False

    # Wait for it to come up
    for _ in range(15):
        time.sleep(1)
        result = _rpc("aria2.getVersion")
        if "result" in result:
            return True
    return False


def add_magnet(magnet: str, save_path: str = "") -> str:
    """Add a magnet link to aria2c. Returns GID (download ID)."""
    if not ensure_running():
        raise RuntimeError("aria2c RPC 服务未启动，无法添加下载。")

    if not save_path:
        save_path = DOWNLOAD_DIR
    os.makedirs(save_path, exist_ok=True)

    result = _rpc("aria2.addUri", [
        [magnet],
        {
            "dir": save_path,
            "bt-tracker": ",".join(_trackers),
        },
    ])

    if "result" in result:
        return result["result"]  # GID
    raise RuntimeError(f"aria2 添加任务失败: {result.get('error', 'unknown')}")


def tell_status(gid: str) -> dict:
    """Get download status for a given GID."""
    result = _rpc("aria2.tellStatus", [gid])
    return result.get("result", {})


def list_active() -> list[dict]:
    """List all active downloads."""
    result = _rpc("aria2.tellActive")
    return result.get("result", [])


def list_all() -> list[dict]:
    """List all downloads (active + waiting + stopped)."""
    all_items = []
    for method in ["aria2.tellActive", "aria2.tellWaiting", "aria2.tellStopped"]:
        result = _rpc(method, [0, 1000])
        items = result.get("result", [])
        if items:
            all_items.extend(items)
    return all_items


def get_downloaded_files(gid: str) -> list[str]:
    """Get list of downloaded file paths for a completed download."""
    status = tell_status(gid)
    files = status.get("files", [])
    dir_path = status.get("dir", DOWNLOAD_DIR)
    paths = []
    for f in files:
        path = f.get("path", "")
        # aria2 returns relative paths; make absolute
        if not os.path.isabs(path):
            path = os.path.join(dir_path, path)
        if os.path.exists(path) and path.lower().endswith(".mkv"):
            paths.append(path)
    return paths


def wait_for_completion(gid: str, timeout: int = 600, on_progress=None) -> list[str]:
    """Wait for a download to complete.

    Args:
        gid: aria2c download GID.
        timeout: Max wait time in seconds (default 10 min).
        on_progress: Optional callback(progress_pct, speed_bytes, name).

    Returns:
        List of downloaded MKV file paths. Empty if timed out / no peers.
    """
    t0 = time.time()
    stuck_since = 0  # track how long we've been at 0%
    while True:
        status = tell_status(gid)
        if not status:
            time.sleep(1)
            continue

        total = int(status.get("totalLength", 0))
        completed = int(status.get("completedLength", 0))
        speed = int(status.get("downloadSpeed", 0))
        name = status.get("bittorrent", {}).get("info", {}).get("name", "")
        seeders = int(status.get("numSeeders", 0))
        error = status.get("errorMessage", "")

        if total > 0 and on_progress:
            pct = completed / total * 100
            on_progress(pct, speed, name)

        st = status.get("status", "")
        if st in ("complete", "removed"):
            return get_downloaded_files(gid)
        if st == "error":
            raise RuntimeError(f"下载失败: {error or '未知错误'}")

        # Detect stalled download
        if speed == 0 and seeders == 0:
            stuck_since += 1
        else:
            stuck_since = 0

        # Never started: 2 min stuck = fail fast
        if completed == 0 and stuck_since > 120:
            raise RuntimeError(
                "无可用下载节点（无种子/用户做种）。\n"
                "该资源可能已失效，或网络环境限制了 P2P 连接。\n"
                "建议：尝试其他字幕组的资源，或使用代理/VPN。"
            )
        # Started but stalled: 5 min stuck = give up
        if completed > 0 and stuck_since > 300:
            raise RuntimeError(
                f"下载中断（已下载 {completed/1024/1024:.0f}MB，停滞超过 5 分钟）。\n"
                "节点连接不稳定，建议尝试其他资源。"
            )

        elapsed = time.time() - t0
        if timeout > 0 and elapsed > timeout:
            return []

        time.sleep(1)


def shutdown():
    """Shutdown aria2c daemon."""
    try:
        _rpc("aria2.shutdown")
    except Exception:
        pass
