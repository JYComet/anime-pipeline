"""
Configuration loader — reads config.yaml and provides typed access.
Mirrors the settings structure of the original Anime Pipeline.
"""
import os
import sys
import shutil
import yaml

# --- Platform detection ---
IS_WIN = sys.platform == "win32"
EXE = ".exe" if IS_WIN else ""

# --- Project root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Default config path ---
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

# Cached config dict, loaded once
_config = None
_config_path = None


def _find_tool(name, preferred_dirs):
    """Find a tool binary, checking preferred_dirs first, then PATH."""
    fname = f"{name}{EXE}"
    for d in preferred_dirs:
        if not d:
            continue
        p = os.path.join(d, fname)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    found = shutil.which(name)
    if found:
        return found
    # On Windows, also try without .exe if the name already has it
    if IS_WIN and not name.endswith(EXE):
        found = shutil.which(fname)
        if found:
            return found
    return fname


def load_config(config_path=None):
    """Load and cache configuration from YAML file."""
    global _config, _config_path
    path = config_path or DEFAULT_CONFIG_PATH
    if _config is not None and _config_path == path:
        return _config
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    _config_path = path

    # Resolve relative paths to absolute (relative to PROJECT_ROOT)
    for key in ("input_dir", "output_dir", "temp_dir", "log_dir"):
        raw = _config.get("paths", {}).get(key, "")
        if raw and not os.path.isabs(raw):
            _config["paths"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, raw))

    return _config


def get_config():
    """Return cached config, loading defaults if not loaded yet."""
    global _config
    if _config is None:
        return load_config()
    return _config


def get_path(key):
    """Get a path from config, normalized to absolute."""
    cfg = get_config()
    raw = cfg.get("paths", {}).get(key, "")
    if not raw:
        return ""
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(PROJECT_ROOT, raw))


def get_tool(name):
    """Get a tool path (ffmpeg / ffprobe) from config or PATH."""
    cfg = get_config()
    configured = cfg.get("tools", {}).get(name, name)
    if configured == name:
        # Not configured explicitly — search PATH
        return _find_tool(name, [])
    if os.path.isabs(configured):
        return configured
    # Relative — search project and PATH
    return _find_tool(name, [os.path.join(PROJECT_ROOT, os.path.dirname(configured))])


def get_processing():
    """Return processing settings dict."""
    return get_config().get("processing", {})


def get_model_pool():
    """Return model pool settings dict."""
    return get_config().get("model_pool", {})


def get_concurrency():
    """Return concurrency settings dict."""
    return get_config().get("concurrency", {})


def get_device():
    """Return target device string."""
    return get_config().get("device", "cuda")


def get_logging():
    """Return logging settings dict."""
    return get_config().get("logging", {})
