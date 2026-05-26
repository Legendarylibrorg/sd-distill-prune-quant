"""Structured pruning of UNet, VAE and the CLIP text encoder.

We use L1 importance over output channels (Conv2d) or output features (Linear),
which is the simplest and most reproducible structured pruning recipe. Skip
connections, projection layers and other shape-sensitive layers are deliberately
left untouched.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from .config import PipelineConfig
from .utils import LOGGER, free_cuda

_TEXT_ENCODER_TARGET = "mlp.fc1"
_VAE_SKIP = ("conv_in", "conv_out", "conv_shortcut", "quant_conv", "post_quant_conv")
_UNET_SKIP = ("proj_in", "proj_out", "conv_shortcut", "time_emb", "conv_in", "conv_out")


def _conv_importance(layer):
    import torch

    return torch.sum(torch.abs(layer.weight.data), dim=(1, 2, 3))


def _linear_importance(layer):
    import torch

    return torch.sum(torch.abs(layer.weight.data), dim=1)


def _replace_module(root, dotted_name: str, new_module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def prune_text_encoder(config: PipelineConfig) -> None:
    """Structured pruning of CLIP MLP ``fc1``/``fc2`` pairs."""
    import torch.nn as nn
    from transformers import CLIPTextModel, CLIPTokenizer

    LOGGER.info("Pruning text encoder ratio=%.2f", config.text_encoder_prune_ratio)

    text_encoder = CLIPTextModel.from_pretrained(f"{config.distill_dir}/text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(f"{config.distill_dir}/tokenizer")

    params_before = sum(p.numel() for p in text_encoder.parameters())
    pruned = 0

    for name, module in list(text_encoder.named_modules()):
        if not (isinstance(module, nn.Linear) and _TEXT_ENCODER_TARGET in name):
            continue
        out_features = module.out_features
        if out_features <= 256:
            continue

        importance = _linear_importance(module)
        num_keep = max(int(out_features * (1 - config.text_encoder_prune_ratio)), 128)
        _, keep_idx = importance.topk(num_keep)
        keep_idx = keep_idx.sort()[0]

        new_fc1 = nn.Linear(module.in_features, num_keep, bias=module.bias is not None)
        new_fc1.weight.data = module.weight.data[keep_idx].clone()
        if module.bias is not None:
            new_fc1.bias.data = module.bias.data[keep_idx].clone()
        _replace_module(text_encoder, name, new_fc1)

        fc2_name = name.replace("fc1", "fc2")
        fc2 = text_encoder
        for part in fc2_name.split("."):
            fc2 = getattr(fc2, part)
        new_fc2 = nn.Linear(num_keep, fc2.out_features, bias=fc2.bias is not None)
        new_fc2.weight.data = fc2.weight.data[:, keep_idx].clone()
        if fc2.bias is not None:
            new_fc2.bias.data = fc2.bias.data.clone()
        _replace_module(text_encoder, fc2_name, new_fc2)

        pruned += 1
        LOGGER.info("  %s: %d -> %d", name, out_features, num_keep)

    params_after = sum(p.numel() for p in text_encoder.parameters())
    LOGGER.info(
        "Text encoder pruning: layers=%d, params %d -> %d (%.1f%% reduction)",
        pruned,
        params_before,
        params_after,
        (1 - params_after / params_before) * 100 if params_before else 0.0,
    )

    os.makedirs(f"{config.prune_dir}/text_encoder", exist_ok=True)
    text_encoder.save_pretrained(f"{config.prune_dir}/text_encoder")
    tokenizer.save_pretrained(f"{config.prune_dir}/tokenizer")
    free_cuda()


def _prune_conv_layers(model, ratio: float, skip: Iterable[str]) -> tuple[int, int, int]:
    import torch.nn as nn

    modules_dict = dict(model.named_modules())
    layers_pruned = 0
    channels_before = 0
    channels_after = 0

    for name, module in list(model.named_modules()):
        if not (isinstance(module, nn.Conv2d) and module.groups == 1):
            continue
        if any(s in name for s in skip):
            continue
        if module.out_channels <= 64:
            continue

        out_channels = module.out_channels
        channels_before += out_channels

        importance = _conv_importance(module)
        num_keep = max(int(out_channels * (1 - ratio)), 32)
        num_keep = min(num_keep, out_channels)

        _, keep_idx = importance.topk(num_keep)
        keep_idx = keep_idx.sort()[0]
        channels_after += num_keep

        if num_keep >= out_channels:
            continue

        new_conv = nn.Conv2d(
            module.in_channels,
            num_keep,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
        )
        new_conv.weight.data = module.weight.data[keep_idx].clone()
        if module.bias is not None:
            new_conv.bias.data = module.bias.data[keep_idx].clone()

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        if parent_name:
            parent = modules_dict[parent_name]
            setattr(parent, child_name, new_conv)
        layers_pruned += 1
        LOGGER.info("  %s: %d -> %d channels", name, out_channels, num_keep)

    return layers_pruned, channels_before, channels_after


def prune_vae(config: PipelineConfig) -> None:
    """Conservative L1 channel pruning of the VAE."""
    from diffusers import AutoencoderKL

    LOGGER.info("Pruning VAE ratio=%.2f", config.vae_prune_ratio)
    vae = AutoencoderKL.from_pretrained(config.distill_dir, subfolder="vae")

    params_before = sum(p.numel() for p in vae.parameters())
    pruned, *_ = _prune_conv_layers(vae, config.vae_prune_ratio, _VAE_SKIP)
    params_after = sum(p.numel() for p in vae.parameters())

    LOGGER.info(
        "VAE pruning: layers=%d, params %d -> %d (%.1f%% reduction)",
        pruned,
        params_before,
        params_after,
        (1 - params_after / params_before) * 100 if params_before else 0.0,
    )
    os.makedirs(f"{config.prune_dir}/vae", exist_ok=True)
    vae.save_pretrained(f"{config.prune_dir}/vae")
    free_cuda()


def prune_unet(config: PipelineConfig) -> None:
    """L1 channel pruning of the UNet and bundle a complete pipeline directory."""
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel

    LOGGER.info("Pruning UNet ratio=%.2f", config.prune_ratio)
    unet = UNet2DConditionModel.from_pretrained(config.distill_dir, subfolder="unet")
    reference = UNet2DConditionModel.from_pretrained(config.distill_dir, subfolder="unet")

    params_before = sum(p.numel() for p in reference.parameters())
    layers_pruned, channels_before, channels_after = _prune_conv_layers(
        unet, config.prune_ratio, _UNET_SKIP
    )
    params_after = sum(p.numel() for p in unet.parameters())

    LOGGER.info(
        "UNet pruning: layers=%d, channels %d -> %d, params %d -> %d (%.1f%% reduction)",
        layers_pruned,
        channels_before,
        channels_after,
        params_before,
        params_after,
        (1 - params_after / params_before) * 100 if params_before else 0.0,
    )

    os.makedirs(f"{config.prune_dir}/unet", exist_ok=True)
    unet.save_pretrained(f"{config.prune_dir}/unet")

    pipe = StableDiffusionPipeline.from_pretrained(config.distill_dir)
    pipe.unet = unet
    pipe.save_pretrained(config.prune_dir)
    free_cuda()


def prune_all(config: PipelineConfig) -> None:
    """Run text-encoder, VAE and UNet pruning in sequence."""
    prune_text_encoder(config)
    prune_vae(config)
    prune_unet(config)


__all__ = [
    "prune_text_encoder",
    "prune_vae",
    "prune_unet",
    "prune_all",
]
