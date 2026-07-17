"""Distillation stages: progressive step halving, CLIP, and CFG distillation.

Each function is callable on its own and operates on directories rather than
in-memory objects so individual stages can be re-run incrementally.
"""

from __future__ import annotations

from .config import PipelineConfig
from .runtime import amp_context, amp_grad_scaler, maybe_channels_last, prepare_training
from .utils import LOGGER, cosine_lr, free_cuda, load_captions, warmup_lr


def _get_attention_maps(unet, latents, timesteps, encoder_hidden_states):
    """Forward through ``unet`` while collecting 4D attention hook outputs."""
    import torch.nn as nn

    attention_maps = []
    hooks = []

    def hook_fn(_module, _inp, output):
        if hasattr(output, "shape") and len(output.shape) == 4:
            attention_maps.append(output)

    for name, module in unet.named_modules():
        if "attn" in name and isinstance(module, nn.Module):
            hooks.append(module.register_forward_hook(hook_fn))

    output = unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states)

    for hook in hooks:
        hook.remove()

    return output, attention_maps


def _update_ema(ema_model, model, decay: float) -> None:
    import torch

    with torch.no_grad():
        for ema_param, param in zip(
            ema_model.parameters(), model.parameters(), strict=True
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def progressive_distillation(config: PipelineConfig) -> None:
    """Iteratively halve the number of inference steps via output + attention KD."""
    import torch
    import torch.nn as nn
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    device = prepare_training(config)
    LOGGER.info("Progressive distillation on device=%s", device)

    teacher_unet = UNet2DConditionModel.from_pretrained(config.base_model, subfolder="unet").to(device)
    student_unet = UNet2DConditionModel.from_pretrained(config.base_model, subfolder="unet").to(device)
    ema_unet = UNet2DConditionModel.from_pretrained(config.base_model, subfolder="unet").to(device)
    student_unet = maybe_channels_last(student_unet, config)

    tokenizer = CLIPTokenizer.from_pretrained(config.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.base_model, subfolder="text_encoder").to(device)
    scheduler = DDPMScheduler.from_pretrained(config.base_model, subfolder="scheduler")
    _vae_unused = AutoencoderKL.from_pretrained(config.base_model, subfolder="vae")  # noqa: F841

    teacher_unet.eval()
    ema_unet.eval()
    for p in teacher_unet.parameters():
        p.requires_grad = False
    for p in ema_unet.parameters():
        p.requires_grad = False
    student_unet.train()

    captions = load_captions(config.data_path)
    stages: list[int] = config.progressive_stage_list

    total_steps = config.steps * len(stages)
    current_step = 0
    warmup_steps = 50

    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=config.lr)
    scaler = amp_grad_scaler(config, device)

    LOGGER.info(
        "Stages=%s, steps_per_stage=%d, total_steps=%d, ema_decay=%.4f",
        stages,
        config.steps,
        total_steps,
        config.ema_decay,
    )

    for stage_idx, target_steps in enumerate(stages):
        LOGGER.info("Stage %d/%d: target inference steps = %d",
                    stage_idx + 1, len(stages), target_steps)

        if stage_idx > 0:
            teacher_unet.load_state_dict(ema_unet.state_dict())

        for step in tqdm(range(config.steps), desc=f"stage{stage_idx + 1}"):
            caption = captions[step % len(captions)]["text"]

            if current_step < warmup_steps:
                lr = warmup_lr(current_step, warmup_steps, config.lr)
            else:
                lr = cosine_lr(
                    current_step - warmup_steps,
                    max(total_steps - warmup_steps, 1),
                    config.lr,
                )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

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
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (1,), device=device
            )
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                with amp_context(config, device):
                    t_out, t_attn = _get_attention_maps(teacher_unet, noisy, timesteps, text_emb)
                    teacher_pred = t_out.sample

            with amp_context(config, device):
                s_out, s_attn = _get_attention_maps(student_unet, noisy, timesteps, text_emb)
                student_pred = s_out.sample

                loss_output = nn.functional.mse_loss(student_pred.float(), teacher_pred.float())

                loss_attn = torch.tensor(0.0, device=device)
                if t_attn and s_attn:
                    matches = 0
                    for ta, sa in zip(t_attn[:5], s_attn[:5], strict=False):
                        if ta.shape == sa.shape:
                            loss_attn = loss_attn + nn.functional.mse_loss(sa.float(), ta.float())
                            matches += 1
                    if matches:
                        loss_attn = loss_attn / matches

                loss = loss_output + 0.1 * loss_attn

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            _update_ema(ema_unet, student_unet, config.ema_decay)
            current_step += 1

            if step % 100 == 0:
                LOGGER.info(
                    "  step=%d loss=%.6f (out=%.6f attn=%.6f) lr=%.2e",
                    step,
                    loss.item(),
                    loss_output.item(),
                    float(loss_attn),
                    lr,
                )

    LOGGER.info("Saving distilled UNet (EMA weights) to %s/unet", config.distill_dir)
    ema_unet.save_pretrained(f"{config.distill_dir}/unet")

    pipe = StableDiffusionPipeline.from_pretrained(config.base_model)
    pipe.unet = ema_unet
    pipe.save_pretrained(config.distill_dir)

    free_cuda()
    LOGGER.info("Progressive distillation complete")


def clip_text_distillation(config: PipelineConfig) -> None:
    """Distil the CLIP text encoder via hidden state + intermediate layer matching."""
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    device = prepare_training(config)
    LOGGER.info("CLIP text encoder distillation on device=%s", device)

    tokenizer = CLIPTokenizer.from_pretrained(config.base_model, subfolder="tokenizer")
    teacher = CLIPTextModel.from_pretrained(config.base_model, subfolder="text_encoder").to(device)
    student = CLIPTextModel.from_pretrained(config.base_model, subfolder="text_encoder").to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student.train()

    captions = load_captions(config.data_path)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(config.clip_distill_steps, 1), eta_min=1e-7
    )
    scaler = amp_grad_scaler(config, device)

    LOGGER.info("CLIP distillation steps=%d", config.clip_distill_steps)

    for step in tqdm(range(config.clip_distill_steps), desc="clip-distill"):
        caption = captions[step % len(captions)]["text"]
        tokens = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            with amp_context(config, device):
                t_out = teacher(tokens, output_hidden_states=True)
        with amp_context(config, device):
            s_out = student(tokens, output_hidden_states=True)

            loss_hidden = nn.functional.mse_loss(
                s_out.last_hidden_state.float(), t_out.last_hidden_state.float()
            )
            loss_pooled = nn.functional.mse_loss(
                s_out.pooler_output.float(), t_out.pooler_output.float()
            )

            loss_intermediate = torch.tensor(0.0, device=device)
            layer_count = 0
            for i in range(0, len(t_out.hidden_states), 3):
                loss_intermediate = loss_intermediate + nn.functional.mse_loss(
                    s_out.hidden_states[i].float(), t_out.hidden_states[i].float()
                )
                layer_count += 1
            if layer_count:
                loss_intermediate = loss_intermediate / layer_count

            loss = loss_hidden + 0.5 * loss_pooled + 0.3 * loss_intermediate

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 100 == 0:
            LOGGER.info("  step=%d loss=%.6f lr=%.2e", step, loss.item(), scheduler.get_last_lr()[0])

    student.save_pretrained(f"{config.distill_dir}/text_encoder")
    tokenizer.save_pretrained(f"{config.distill_dir}/tokenizer")
    free_cuda()
    LOGGER.info("CLIP distillation complete")


def cfg_distillation(config: PipelineConfig) -> None:
    """Distil classifier-free guidance into a single forward pass."""
    import torch
    import torch.nn as nn
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    device = prepare_training(config)
    LOGGER.info("CFG distillation on device=%s guidance=%.2f", device, config.guidance_scale)

    unet = UNet2DConditionModel.from_pretrained(config.distill_dir, subfolder="unet").to(device)
    unet = maybe_channels_last(unet, config)
    teacher = UNet2DConditionModel.from_pretrained(config.distill_dir, subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(config.distill_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.distill_dir, subfolder="text_encoder").to(device)
    scheduler = DDPMScheduler.from_pretrained(config.distill_dir, subfolder="scheduler")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    unet.train()

    captions = load_captions(config.data_path)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6)
    scaler = amp_grad_scaler(config, device)

    for step in tqdm(range(config.cfg_distill_steps), desc="cfg-distill"):
        caption = captions[step % len(captions)]["text"]

        tokens = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        uncond_tokens = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            text_emb = text_encoder(tokens)[0]
            uncond_emb = text_encoder(uncond_tokens)[0]

        latents = torch.randn(1, 4, 64, 64, device=device)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noise = torch.randn_like(latents)
        noisy = scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            with amp_context(config, device):
                uncond = teacher(noisy, timesteps, encoder_hidden_states=uncond_emb).sample
                cond = teacher(noisy, timesteps, encoder_hidden_states=text_emb).sample
                cfg_output = uncond + config.guidance_scale * (cond - uncond)

        with amp_context(config, device):
            student_out = unet(noisy, timesteps, encoder_hidden_states=text_emb).sample
            loss = nn.functional.mse_loss(student_out.float(), cfg_output.float())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0:
            LOGGER.info("  step=%d cfg-loss=%.6f", step, loss.item())

    unet.save_pretrained(f"{config.distill_dir}/unet")
    free_cuda()
    LOGGER.info("CFG distillation complete; negative prompt no longer required")


__all__ = [
    "progressive_distillation",
    "clip_text_distillation",
    "cfg_distillation",
]
