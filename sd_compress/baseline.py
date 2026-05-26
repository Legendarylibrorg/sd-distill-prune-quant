"""Generate baseline reference images with the original SD 1.5 model.

The baseline images are used by every later stage as the ground truth for
LPIPS / PSNR / SSIM comparisons.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .config import PipelineConfig, ensure_captions
from .utils import (
    LOGGER,
    free_cuda,
    load_captions,
    mean,
    save_json,
    select_device,
    stage_eval_dir,
)


def generate_baseline(config: PipelineConfig, num_inference_steps: int = 50) -> dict:
    """Generate ``config.eval_samples`` reference images with the base model.

    Results are written to ``output/eval/baseline`` and returned as a metrics
    dictionary. If the metrics file already exists the existing copy is reused.
    """
    import torch
    from diffusers import StableDiffusionPipeline
    from tqdm.auto import tqdm

    ensure_captions(config.data_path)
    eval_dir = stage_eval_dir(config.output_dir, "baseline")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(eval_dir, "metrics.json")

    if os.path.exists(metrics_path):
        LOGGER.info("Baseline references already exist, skipping generation.")
        from .utils import load_json

        return load_json(metrics_path)

    device = select_device()
    LOGGER.info("Loading baseline model from %s (device=%s)", config.base_model, device)

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(config.base_model, torch_dtype=dtype)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    elif device == "mps":
        pipe = pipe.to("mps")

    captions = load_captions(config.data_path)
    prompts: list[str] = [c["text"] for c in captions[: config.eval_samples]]

    LOGGER.info("Generating %d baseline images at %d steps", len(prompts), num_inference_steps)

    metrics = {
        "stage": "baseline",
        "num_inference_steps": num_inference_steps,
        "prompts": prompts,
        "images": [],
        "clip_scores": [],
        "inference_times": [],
    }

    has_clip = False
    try:
        from evaluate import compute_clip_score  # type: ignore

        has_clip = True
    except Exception as exc:  # pragma: no cover - depends on optional deps
        LOGGER.warning("CLIP score unavailable for baseline: %s", exc)

    for i, prompt in enumerate(tqdm(prompts, desc="Baselines")):
        generator = torch.Generator(device=device).manual_seed(42 + i) if device != "mps" else None
        start = time.time()
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
        elapsed_ms = (time.time() - start) * 1000

        image_path = os.path.join(eval_dir, f"baseline_{i:03d}.png")
        image.save(image_path)

        metrics["images"].append(image_path)
        metrics["inference_times"].append(elapsed_ms)

        if has_clip:
            score = compute_clip_score(image, prompt, device)
            metrics["clip_scores"].append(score)
            LOGGER.info("  [%d] CLIP=%.4f Time=%.0fms", i, score, elapsed_ms)

    metrics["average_clip"] = mean(metrics["clip_scores"])
    metrics["average_time"] = mean(metrics["inference_times"])
    save_json(metrics_path, metrics)

    LOGGER.info(
        "Baseline complete: CLIP=%.4f, avg_time=%.0fms (saved to %s)",
        metrics["average_clip"],
        metrics["average_time"],
        eval_dir,
    )

    del pipe
    free_cuda()
    return metrics


__all__ = ["generate_baseline"]
