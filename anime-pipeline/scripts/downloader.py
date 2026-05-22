"""
Anime resource search and download module.
Searches animes.garden API and downloads via magnet links.
"""
import os
import re
import subprocess
import urllib3
from dataclasses import dataclass, field

# api.animes.garden has SSL cert issues on some systems
urllib3.disable_warnings()
_VERIFY_SSL = False

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SESSION = requests.Session()
_retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry)
_SESSION.mount('https://', _adapter)
_SESSION.mount('http://', _adapter)

def _api_get(url: str, **kwargs) -> requests.Response:
    """Make an API request with retry and timeout handling."""
    kwargs.setdefault('timeout', (5, 30))  # (connect timeout, read timeout)
    kwargs.setdefault('verify', _VERIFY_SSL)
    kwargs.setdefault('headers', HEADERS)
    try:
        return _SESSION.get(url, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"无法连接到动漫资源网 (animes.garden)。\n"
            f"请检查网络连接或稍后重试。\n"
            f"详情: {e}"
        )
    except requests.exceptions.ReadTimeout as e:
        raise RuntimeError(
            f"动漫资源网响应超时，可能服务器繁忙或网络不稳定。\n"
            f"请稍后重试。\n"
            f"详情: {e}"
        )

from config import (
    RESOURCES_ENDPOINT, ANIMEGARDEN_API, ANIMEGARDEN_RESOURCE_DETAIL, HEADERS, DOWNLOAD_DIR,
    ARIA2C
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
        return []
    except Exception:
        return []

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    return files


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
