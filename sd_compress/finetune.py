"""Short fine-tuning pass to recover quality after pruning."""

from __future__ import annotations

import os

from .config import PipelineConfig
from .runtime import amp_context, amp_grad_scaler, maybe_channels_last, prepare_training
from .utils import LOGGER, free_cuda, load_captions


def finetune_after_pruning(config: PipelineConfig) -> None:
    """Run a short diffusion fine-tune on the pruned UNet using the standard ε-prediction objective."""
    import torch
    import torch.nn as nn
    from diffusers import (
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    device = prepare_training(config)
    LOGGER.info(
        "Fine-tuning pruned UNet for %d steps lr=%.2e (device=%s)",
        config.finetune_steps,
        config.finetune_lr,
        device,
    )

    unet = UNet2DConditionModel.from_pretrained(config.prune_dir, subfolder="unet").to(device)
    unet = maybe_channels_last(unet, config)
    tokenizer = CLIPTokenizer.from_pretrained(config.prune_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.prune_dir, subfolder="text_encoder").to(device)
    scheduler = DDPMScheduler.from_pretrained(config.prune_dir, subfolder="scheduler")

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    unet.train()

    captions = load_captions(config.data_path)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.finetune_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(config.finetune_steps, 1), eta_min=1e-7
    )
    scaler = amp_grad_scaler(config, device)

    for step in tqdm(range(config.finetune_steps), desc="finetune"):
        caption = captions[step % len(captions)]["text"]
        tokens = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            text_emb = text_encoder(tokens)[0]

        latents = torch.randn(1, 4, 64, 64, device=device)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noise = torch.randn_like(latents)
        noisy = scheduler.add_noise(latents, noise, timesteps)

        with amp_context(config, device):
            pred = unet(noisy, timesteps, encoder_hidden_states=text_emb).sample
            loss = nn.functional.mse_loss(pred.float(), noise.float())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        if step % 50 == 0:
            LOGGER.info("  step=%d loss=%.6f", step, loss.item())

    os.makedirs(config.finetune_dir, exist_ok=True)
    unet.save_pretrained(f"{config.finetune_dir}/unet")

    pipe = StableDiffusionPipeline.from_pretrained(config.prune_dir)
    pipe.unet = unet
    pipe.save_pretrained(config.finetune_dir)

    free_cuda()
    LOGGER.info("Fine-tuning complete -> %s", config.finetune_dir)


__all__ = ["finetune_after_pruning"]
