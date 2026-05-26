"""Stable Diffusion 1.5 compression and optimization pipeline.

This package provides modular building blocks for distillation, structured
pruning, fine-tuning, quantization, deployment export and quality evaluation
of Stable Diffusion 1.5 checkpoints.

Each stage is exposed both as an importable function and through the
``python -m sd_compress`` command-line interface defined in :mod:`sd_compress.cli`.
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
