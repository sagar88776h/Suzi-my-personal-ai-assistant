"""Well-known filesystem locations used by Suzy.

All paths are resolved relative to the application folder so Suzy works the
same whether it is run from source (``python main.py``) or as a PyInstaller
executable (``Suzy.exe``).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _app_dir() -> Path:
    """Return the folder that holds config.yaml, .env, logs/ and data/."""
    if getattr(sys, "frozen", False):  # running as a PyInstaller bundle
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_DIR: Path = _app_dir()
LOG_DIR: Path = APP_DIR / "logs"
DATA_DIR: Path = APP_DIR / "data"
CONFIG_PATH: Path = APP_DIR / "config.yaml"
ENV_PATH: Path = APP_DIR / ".env"
