"""ONNX export and safetensors sharding helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig
from .utils import LOGGER


def export_onnx(config: PipelineConfig) -> None:
    """Export UNet + VAE decoder to ONNX for TensorRT / ONNX Runtime backends."""
    import torch

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:  # pragma: no cover
        LOGGER.error("diffusers is required for ONNX export: %s", exc)
        return

    onnx_dir = Path(config.export_dir) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s for ONNX export", config.quant_dir)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(config.quant_dir, torch_dtype=torch.float32)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed to load quantised pipeline for export: %s", exc)
        return

    dummy_latent = torch.randn(1, 4, 64, 64)
    dummy_timestep = torch.tensor([1])
    dummy_encoder_hidden = torch.randn(1, 77, 768)

    unet_path = onnx_dir / "unet.onnx"
    LOGGER.info("Exporting UNet to %s", unet_path)
    try:
        torch.onnx.export(
            pipe.unet,
            (dummy_latent, dummy_timestep, dummy_encoder_hidden),
            str(unet_path),
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            dynamic_axes={
                "sample": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
            },
            opset_version=14,
        )
    except Exception as exc:  # pragma: no cover - depends on torch/onnx
        LOGGER.warning("UNet ONNX export failed: %s", exc)

    vae_path = onnx_dir / "vae_decoder.onnx"
    LOGGER.info("Exporting VAE decoder to %s", vae_path)
    try:
        torch.onnx.export(
            pipe.vae.decoder,
            dummy_latent,
            str(vae_path),
            input_names=["latent"],
            output_names=["image"],
            dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
            opset_version=14,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("VAE decoder ONNX export failed: %s", exc)

    LOGGER.info("ONNX export finished. TensorRT example: trtexec --onnx=%s --fp16", unet_path)


def shard_safetensors(config: PipelineConfig) -> None:
    """Split the quantised UNet state-dict into ~``config.shard_size_mb`` shards."""
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file

    LOGGER.info("Sharding UNet state-dict (max %d MB per shard)", config.shard_size_mb)

    sharded_dir = Path(config.export_dir) / "sharded"
    sharded_dir.mkdir(parents=True, exist_ok=True)

    try:
        unet = UNet2DConditionModel.from_pretrained(config.quant_dir, subfolder="unet")
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Could not load UNet for sharding: %s", exc)
        return

    state_dict = unet.state_dict()
    current: dict = {}
    current_size = 0.0
    shard_idx = 0
    shard_map: dict[str, str] = {}

    for name, tensor in state_dict.items():
        tensor_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        if current and current_size + tensor_mb > config.shard_size_mb:
            shard_file = f"unet_shard_{shard_idx:03d}.safetensors"
            save_file(current, str(sharded_dir / shard_file))
            LOGGER.info("  Saved %s (%.1f MB)", shard_file, current_size)
            shard_idx += 1
            current = {}
            current_size = 0.0

        current[name] = tensor
        current_size += tensor_mb
        shard_map[name] = f"unet_shard_{shard_idx:03d}.safetensors"

    if current:
        shard_file = f"unet_shard_{shard_idx:03d}.safetensors"
        save_file(current, str(sharded_dir / shard_file))
        LOGGER.info("  Saved %s (%.1f MB)", shard_file, current_size)

    index_path = sharded_dir / "shard_index.json"
    with index_path.open("w", encoding="utf-8") as fp:
        json.dump({"total_shards": shard_idx + 1, "shard_map": shard_map}, fp, indent=2)
    LOGGER.info("Shard index saved to %s", index_path)


__all__ = ["export_onnx", "shard_safetensors"]
