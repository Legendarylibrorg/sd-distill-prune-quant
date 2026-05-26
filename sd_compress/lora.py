"""Sanity-check LoRA compatibility with the compressed pipeline."""

from __future__ import annotations

import os

from .config import PipelineConfig
from .utils import LOGGER, save_json

_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def test_lora_compatibility(config: PipelineConfig) -> dict:
    """Inspect the UNet for shapes that LoRA adapters can plug into."""
    import torch
    from diffusers import StableDiffusionPipeline

    LOGGER.info("Checking LoRA compatibility of %s", config.quant_dir)
    pipe = StableDiffusionPipeline.from_pretrained(
        config.quant_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()

    issues = []
    compatible = 0
    total = 0

    for name, module in pipe.unet.named_modules():
        if not any(target in name for target in _LORA_TARGETS):
            continue
        total += 1
        if hasattr(module, "weight"):
            shape = tuple(module.weight.shape)
            if shape[0] >= 64 and shape[1] >= 64:
                compatible += 1
            else:
                issues.append(f"{name}: shape {shape} may be too small for typical LoRAs")

    summary = {
        "total_attention_layers": total,
        "compatible_layers": compatible,
        "issues": issues,
        "weight_shapes_compatible": not issues,
    }
    report = {"shape_compatibility": summary, "summary": summary}

    path = os.path.join(config.output_dir, "lora_compatibility_report.json")
    save_json(path, report)
    LOGGER.info(
        "LoRA compatibility: %d/%d layers OK (%d issues) -> %s",
        compatible,
        total,
        len(issues),
        path,
    )
    return report


__all__ = ["test_lora_compatibility"]
