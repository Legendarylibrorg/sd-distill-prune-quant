"""FP16 and dynamic INT8 quantization."""

from __future__ import annotations

import os
from pathlib import Path

from .config import PipelineConfig
from .utils import LOGGER, free_cuda

# Dynamic INT8 state-dicts use torch quantized tensors, which safetensors cannot
# represent. They are stored as ``.pt`` and must be loaded with ``weights_only=True``.
INT8_FILENAME = "unet_int8.pt"


def _model_size_mb(model) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def load_int8_state_dict(path: str | os.PathLike[str]):
    """Load an INT8 UNet state-dict produced by :func:`quantize`.

    Always uses ``weights_only=True`` so untrusted ``.pt`` files cannot execute
    arbitrary pickle payloads. Prefer this helper over a bare ``torch.load``.
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"INT8 state-dict not found: {path}")
    LOGGER.info("Loading INT8 state-dict from %s (weights_only=True)", path)
    return torch.load(path, map_location="cpu", weights_only=True)


def quantize(config: PipelineConfig) -> None:
    """Save an FP16 pipeline directory and a side-car INT8 UNet state-dict."""
    import torch
    import torch.nn as nn
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel

    LOGGER.info("Loading model for quantisation from %s", config.finetune_dir)
    unet = UNet2DConditionModel.from_pretrained(config.finetune_dir, subfolder="unet")

    LOGGER.info("Applying dynamic INT8 quantisation (Linear + Conv2d)")
    quantized = torch.quantization.quantize_dynamic(
        unet, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

    original_mb = _model_size_mb(unet)
    quantized_mb = _model_size_mb(quantized)
    reduction = (1 - quantized_mb / original_mb) * 100 if original_mb else 0.0
    LOGGER.info(
        "Quantisation: %.1f MB -> %.1f MB (%.1f%% reduction)",
        original_mb,
        quantized_mb,
        reduction,
    )

    os.makedirs(config.quant_dir, exist_ok=True)

    LOGGER.info("Saving FP16 pipeline to %s", config.quant_dir)
    pipe = StableDiffusionPipeline.from_pretrained(config.finetune_dir, torch_dtype=torch.float16)
    pipe.save_pretrained(config.quant_dir, safe_serialization=True)

    int8_path = os.path.join(config.quant_dir, INT8_FILENAME)
    LOGGER.info(
        "Saving INT8 state-dict to %s "
        "(pickle/.pt required for torch dynamic quant; load via load_int8_state_dict)",
        int8_path,
    )
    torch.save(quantized.state_dict(), int8_path)

    free_cuda()


__all__ = ["INT8_FILENAME", "load_int8_state_dict", "quantize"]
