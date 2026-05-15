"""
Anime resource search and download module.
Searches animes.garden API and downloads via magnet links.
"""
import os
import re
import time
import json
import subprocess
import requests
import urllib3
from typing import Optional
from dataclasses import dataclass, field

# api.animes.garden has SSL cert issues on some systems
urllib3.disable_warnings()
_VERIFY_SSL = False

# Create a shared session with retry logic
import requests as _requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SESSION = _requests.Session()
_retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry)
_SESSION.mount('https://', _adapter)
_SESSION.mount('http://', _adapter)

def _api_get(url: str, **kwargs) -> _requests.Response:
    """Make an API request with retry and timeout handling."""
    kwargs.setdefault('timeout', (5, 30))  # (connect timeout, read timeout)
    kwargs.setdefault('verify', _VERIFY_SSL)
    kwargs.setdefault('headers', HEADERS)
    try:
        return _SESSION.get(url, **kwargs)
    except _requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"无法连接到动漫资源网 (animes.garden)。\n"
            f"请检查网络连接或稍后重试。\n"
            f"详情: {e}"
        )
    except _requests.exceptions.ReadTimeout as e:
        raise RuntimeError(
            f"动漫资源网响应超时，可能服务器繁忙或网络不稳定。\n"
            f"请稍后重试。\n"
            f"详情: {e}"
        )

from config import (
    RESOURCES_ENDPOINT, ANIMEGARDEN_API, ANIMEGARDEN_RESOURCE_DETAIL, HEADERS, DOWNLOAD_DIR,
    ARIA2C, FFMPEG
)


@dataclass
class Resource:
    """A single anime resource from the API."""
    id: int
    provider: str
    provider_id: str
    title: str
    href: str
    type: str
    magnet: str
    size: int  # bytes
    created_at: str
    fetched_at: str
    publisher: dict = field(default_factory=dict)
    fansub: dict = field(default_factory=dict)
    subject_id: int = 0

    @property
    def size_mb(self) -> float:
        return self.size / 1024 / 1024

    @property
    def display_title(self) -> str:
        return self.title

    @property
    def magnet_hash(self) -> str:
        """Extract BT info hash from magnet link."""
        m = re.search(r'btih:([a-fA-F0-9]+)', self.magnet)
        return m.group(1) if m else ""

    @property
    def file_format(self) -> str:
        """Detect the file format from the title (e.g. MKV, MP4, etc.)."""
        # Look for video format keywords in the title
        match = re.search(r'\b(MKV|MP4|AVI|MOV|WMV|WEBM|FLV|TS|M2TS)\b',
                          self.title, re.IGNORECASE)
        return match.group(1).upper() if match else "UNKNOWN"

    @property
    def is_mkv(self) -> bool:
        return self.file_format == "MKV"


@dataclass
class SearchResult:
    """Search results from the API."""
    resources: list[Resource]
    total_hint: int  # approximate total
    page: int
    page_size: int
    complete: bool


def search_resources(
    query: str = "",
    page: int = 1,
    page_size: int = 20,
    provider: str = "",
    fansub: str = "",
    resource_type: str = "",
    after: str = "",
    before: str = "",
) -> SearchResult:
    """Search anime resources on animes.garden.

    Args:
        query: Full-text search keywords
        page: Page number (1-based)
        page_size: Items per page (max ~100)
        provider: Filter by provider ('dmhy', 'moe', 'ani')
        fansub: Filter by fansub group name
        resource_type: Filter by resource type
        after/before: ISO datetime range for publishedAt
    """
    params = {"page": page, "pageSize": page_size}
    if query:
        params["search"] = query
    if provider:
        params["provider"] = provider
    if fansub:
        params["fansub"] = fansub
    if resource_type:
        params["type"] = resource_type
    if after:
        params["after"] = after
    if before:
        params["before"] = before

    resp = _api_get(RESOURCES_ENDPOINT, params=params)
    resp.raise_for_status()
    data = resp.json()

    resources = []
    for r in data.get("resources", []):
        resources.append(Resource(
            id=r.get("id", 0),
            provider=r.get("provider", ""),
            provider_id=r.get("providerId", ""),
            title=r.get("title", ""),
            href=r.get("href", ""),
            type=r.get("type", ""),
            magnet=r.get("magnet", ""),
            size=r.get("size", 0),
            created_at=r.get("createdAt", ""),
            fetched_at=r.get("fetchedAt", ""),
            publisher=r.get("publisher", {}),
            fansub=r.get("fansub", {}),
            subject_id=r.get("subjectId", 0),
        ))

    pagination = data.get("pagination", {})
    return SearchResult(
        resources=resources,
        total_hint=len(resources),
        page=pagination.get("page", page),
        page_size=pagination.get("pageSize", page_size),
        complete=data.get("complete", False),
    )


def fetch_teams() -> list[dict]:
    """Fetch all fansub teams from the API."""
    url = f"{ANIMEGARDEN_API}/teams"
    resp = _api_get(url)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        return data.get("teams", [])
    if isinstance(data, list):
        return data
    return []


def get_resource_detail(provider: str, provider_id: str) -> dict:
    """Get detailed resource info including file list."""
    url = f"{ANIMEGARDEN_RESOURCE_DETAIL}/{provider}/{provider_id}"
    resp = _api_get(url)
    resp.raise_for_status()
    return resp.json()


def check_aria2c() -> bool:
    """Check if aria2c is available."""
    try:
        subprocess.run([ARIA2C, "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def download_magnet(
    magnet: str,
    output_dir: str = "",
    on_progress=None,
    timeout: int = 600,  # max seconds to wait for download to start
) -> Optional[str]:
    """Download a magnet link using aria2c.

    Args:
        magnet: Magnet URI
        output_dir: Where to save downloaded files
        on_progress: Optional callback(percent, speed, eta)
        timeout: Max seconds to wait before giving up (default 10 min)

    Returns:
        Path to the downloaded file, or None if failed.
    """
    if not output_dir:
        output_dir = DOWNLOAD_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not check_aria2c():
        raise RuntimeError("aria2c 未安装，下载功能不可用。")

    # Common trackers (UDP and HTTP)
    trackers = [
        "udp://tracker.opentrackr.org:1337/announce",
        "udp://open.demonii.com:1337/announce",
        "udp://tracker.torrent.eu.org:451/announce",
        "udp://explodie.org:6969/announce",
        "udp://tracker.coppersurfer.tk:6969/announce",
        "udp://tracker.leechers-paradise.org:6969/announce",
        "udp://9.rarbg.to:2710/announce",
        "udp://tracker.internetwarriors.net:1337/announce",
        "http://tracker.opentrackr.org:1337/announce",
        "http://open.acgnxtracker.com:80/announce",
        "https://trakx.herokuapp.com:443/announce",
    ]

    cmd = [
        ARIA2C,
        "--seed-time=0",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--dir", output_dir,
        # DHT configuration
        "--enable-dht=true",
        "--dht-listen-port=6881-6999",
        "--dht-entry-point=router.bittorrent.com:6881",
        "--dht-entry-point=router.utorrent.com:6881",
        "--dht-entry-point=dht.transmissionbt.com:6881",
        "--enable-dht6=false",
        # Peer discovery
        "--bt-enable-lpd=true",
        "--enable-peer-exchange=true",
        # Timeouts
        "--bt-stop-timeout=600",
        "--bt-tracker-connect-timeout=10",
        "--bt-tracker-timeout=10",
        "--connect-timeout=10",
        "--bt-request-peer-speed-limit=50K",
        # Trackers
        *[f"--bt-tracker={t}" for t in trackers],
        "--bt-detach-seed-only",
        magnet,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    import threading
    killed = [False]

    def kill_on_timeout():
        killed[0] = True
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(timeout, kill_on_timeout)
    timer.start()

    output_lines = []
    has_progress = False
    try:
        for line in proc.stdout:
            line = line.strip()
            output_lines.append(line)

            if on_progress:
                pct_match = re.search(r'\((\d+)%\)', line)
                if pct_match:
                    has_progress = True
                    speed_match = re.search(r'DL:(\S+)', line)
                    speed = speed_match.group(1) if speed_match else ""
                    on_progress(int(pct_match.group(1)), speed, "")
    except Exception:
        pass
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if killed[0]:
        return None  # timed out

    if not has_progress:
        # Never started downloading — check error output for hints
        last_lines = "\n".join(output_lines[-10:])
        if "no peers" in last_lines.lower() or "tracker" in last_lines.lower():
            raise RuntimeError(
                f"无法找到下载节点。磁力链接可能没有可用资源，或网络环境限制了 P2P 连接。\n"
                f"建议：复制磁力链接使用其他下载工具（如 qBittorrent、迅雷等）下载后，\n"
                f"将 MKV 文件放入 video/ 目录，然后从「本地文件」页面处理。"
            )
        raise RuntimeError(f"下载未能启动。aria2c 输出:\n{last_lines[:500]}")

    if proc.returncode != 0 and proc.returncode is not None:
        raise RuntimeError(f"下载失败 (aria2c exit code: {proc.returncode})")

    return output_dir


def download_via_ffmpeg(m3u8_url: str, output_path: str) -> bool:
    """Fallback: download a stream via ffmpeg (not for magnet)."""
    cmd = [
        FFMPEG, "-y",
        "-i", m3u8_url,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=7200)
    return result.returncode == 0


def parse_magnet(magnet: str) -> dict:
    """Parse a magnet URI into its components.

    Returns dict with keys: info_hash, name, trackers, exact_topic
    """
    result = {"info_hash": "", "name": "", "trackers": [], "exact_topic": ""}
    m = re.search(r'btih:([a-fA-F0-9]+)', magnet)
    if m:
        result["info_hash"] = m.group(1)
    m = re.search(r'dn=([^&]+)', magnet)
    if m:
        from urllib.parse import unquote
        result["name"] = unquote(m.group(1))
    m = re.search(r'xt=([^&]+)', magnet)
    if m:
        result["exact_topic"] = m.group(1)
    for tr in re.findall(r'tr=([^&]+)', magnet):
        from urllib.parse import unquote
        result["trackers"].append(unquote(tr))
    return result


def fetch_magnet_files(magnet: str, timeout: int = 60) -> list[dict]:
    """Fetch file list from a magnet link using aria2c metadata-only mode.

    Returns list of dicts: {name, size, size_mb}
    """
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="mg_meta_")

    cmd = [
        ARIA2C,
        "--bt-metadata-only=true",
        "--bt-save-metadata=true",
        "--dir", tmpdir,
        "--enable-dht=true",
        "--dht-entry-point=router.bittorrent.com:6881",
        "--dht-entry-point=dht.transmissionbt.com:6881",
        "--bt-tracker=udp://tracker.opentrackr.org:1337/announce",
        "--connect-timeout=10",
        "--bt-tracker-connect-timeout=10",
        "-q",
        magnet,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass

    # Find .torrent file
    torrent_files = []
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            if f.endswith(".torrent"):
                torrent_files.append(os.path.join(root, f))

    if not torrent_files:
        return []

    # Parse torrent file to get file list
    files = []
    try:
        import bencodepy
        with open(torrent_files[0], "rb") as fh:
            data = bencodepy.decode(fh.read())
        info = data.get(b"info", {})
        name = info.get(b"name", b"").decode("utf-8", errors="replace")
        flist = info.get(b"files", [])
        if flist:
            for entry in flist:
                fname = "/".join(
                    p.decode("utf-8", errors="replace") for p in entry.get(b"path", [])
                )
                fsize = entry.get(b"length", 0)
                files.append({
                    "name": f"{name}/{fname}",
                    "size_bytes": fsize,
                    "size_mb": round(fsize / 1024 / 1024, 1),
                })
        else:
            fsize = info.get(b"length", 0)
            files.append({
                "name": name,
                "size_bytes": fsize,
                "size_mb": round(fsize / 1024 / 1024, 1),
            })
    except ImportError:
        # Fallback: parse .torrent with basic bencode
        files = _parse_torrent_simple(torrent_files[0])
    except Exception:
        files = _parse_torrent_simple(torrent_files[0])

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    return files


def _parse_torrent_simple(torrent_path: str) -> list[dict]:
    """Basic torrent parser without external dependencies."""
    with open(torrent_path, "rb") as f:
        data = f.read()

    files = []
    # Try to find file names using regex patterns
    # Look for 4:nameX:... or 5:files patterns
    import re
    # Find file paths (like 4:nameX:....)
    path_pattern = re.compile(rb'(\d+):([^\d]{2,30})')
    # Find file sizes
    size_pattern = re.compile(rb'6:lengthi(\d+)e')

    matches_path = path_pattern.findall(data)
    matches_size = size_pattern.findall(data)

    # Try to reconstruct file list from .torrent structure
    # Simple approach: extract the info section and parse manually
    info_start = data.find(b'4:info')
    if info_start >= 0:
        info_data = data[info_start + 6:]
        # Try to decode as much as possible
        try:
            text = info_data.decode("utf-8", errors="replace")
            # Extract meaningful file names
            for m in re.finditer(r'(\d+):([^\d]{3,60})', text):
                s = m.group(2).strip()
                if len(s) > 3 and not s.startswith(":") and not s.startswith("i"):
                    files.append({
                        "name": s,
                        "size_bytes": 0,
                        "size_mb": 0,
                    })
        except Exception:
            pass

    # Deduplicate and clean
    seen = set()
    clean = []
    for f in files:
        if f["name"] not in seen and len(f["name"]) > 2:
            seen.add(f["name"])
            clean.append(f)

    return clean


def find_local_video_files(directory: str = "") -> list[str]:
    """Find all video files (MKV, MP4, etc.) in a directory."""
    if not directory:
        directory = DOWNLOAD_DIR
    exts = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".webm", ".flv", ".ts"}
    video_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                video_files.append(os.path.join(root, f))
    return video_files


def find_local_mkv_files(directory: str = "") -> list[str]:
    """Find all MKV files in a directory (for testing with pre-downloaded files)."""
    if not directory:
        directory = DOWNLOAD_DIR
    mkv_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.mkv'):
                mkv_files.append(os.path.join(root, f))
    return mkv_files
