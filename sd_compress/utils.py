"""Shared utilities used by multiple pipeline stages."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("sd_compress")


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def select_device() -> str:
    """Return ``cuda``, ``mps`` or ``cpu`` depending on what is available."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a required dep at runtime
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float = 1e-7) -> float:
    """Vanilla cosine annealing schedule used by several stages."""
    if total_steps <= 0:
        return lr_max
    progress = step / total_steps
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def warmup_lr(step: int, warmup_steps: int, lr_max: float) -> float:
    """Linear warm-up to ``lr_max`` over ``warmup_steps`` steps."""
    if warmup_steps <= 0:
        return lr_max
    return lr_max * min(step / warmup_steps, 1.0)


def save_json(path: str, payload: Any) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def directory_size_mb(path: str) -> int:
    """Return the on-disk size of a directory in MiB via ``du`` if available."""
    if not Path(path).exists():
        return 0
    try:
        result = subprocess.run(
            ["du", "-sm", path], capture_output=True, text=True, check=True
        )
        return int(result.stdout.split()[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        # Fallback for systems without ``du`` (e.g. Windows)
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        return total // (1024 * 1024)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def free_cuda() -> None:
    """Best-effort CUDA cache release used between heavy stages."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def load_captions(path: str) -> list[dict[str, str]]:
    return load_json(path)


def stage_eval_dir(output_dir: str, stage: str) -> str:
    return str(Path(output_dir) / "eval" / stage)


def maybe_import_eval():
    """Import metric helpers from the top-level :mod:`evaluate` module.

    The metric helpers live in ``evaluate.py`` at the repository root to remain
    backwards compatible with the original pipeline. Importing lazily lets the
    rest of the package run even if optional metric dependencies are missing.
    """
    try:
        from evaluate import (  # type: ignore  # noqa: F401
            compute_clip_score,
            compute_lpips,
            compute_psnr,
            compute_ssim,
        )
        return True
    except Exception as exc:  # pragma: no cover - depends on optional deps
        LOGGER.warning("Evaluation metrics unavailable: %s", exc)
        return False


def report_torch_environment() -> dict[str, Any]:
    """Return a short description of the active PyTorch environment."""
    info: dict[str, Any] = {"device": "cpu"}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["device"] = select_device()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:  # pragma: no cover
        info["torch_version"] = "not-installed"
    return info
