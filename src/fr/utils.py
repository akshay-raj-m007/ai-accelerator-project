"""Small shared helpers: seeding, environment logging, git hash, JSON."""
from __future__ import annotations

import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def git_info() -> dict:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or "unknown",
                "dirty": bool(run("status", "--porcelain"))}
    except Exception:  # git missing / not a repo
        return {"commit": "unknown", "dirty": None}


def _pkg_version(name: str) -> str:
    """Installed version via package metadata; never raises (some builds lack __version__)."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        try:
            return str(getattr(__import__(name), "__version__", "unknown"))
        except Exception:
            return "unknown"


def env_info() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _pkg_version("torchvision"),
        "numpy": np.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "git": git_info(),
    }


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_Encoder, allow_nan=True)
