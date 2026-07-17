"""Stage-by-stage evaluation harness for the compression pipeline."""

from __future__ import annotations

import os
import time
from pathlib import Path

from .config import PipelineConfig
from .runtime import prepare_pipeline_for_inference
from .utils import (
    LOGGER,
    directory_size_mb,
    free_cuda,
    load_json,
    mean,
    save_json,
    select_device,
    stage_eval_dir,
)


def _baseline_metrics(config: PipelineConfig) -> dict | None:
    path = os.path.join(config.eval_dir, "baseline", "metrics.json")
    if not os.path.exists(path):
        LOGGER.warning("Baseline metrics missing at %s; run the baseline stage first", path)
        return None
    return load_json(path)


def evaluate_stage(
    config: PipelineConfig,
    stage: str,
    model_dir: str,
    num_inference_steps: int | None = None,
    *,
    compute_model_size: bool = False,
) -> dict:
    """Evaluate one stage against the baseline reference images.

    Parameters
    ----------
    stage:
        Logical stage name (``"distilled"``, ``"pruned"``, ``"finetuned"`` ...).
    model_dir:
        Path of the diffusers-format pipeline directory to evaluate.
    num_inference_steps:
        Step count for evaluation; defaults to ``config.eval_inference_steps``.
    compute_model_size:
        If True, include a directory size measurement in the report.
    """
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    from tqdm.auto import tqdm

    baseline = _baseline_metrics(config)
    if baseline is None:
        return {}

    device = select_device()
    eval_dir = stage_eval_dir(config.output_dir, stage)
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    steps = num_inference_steps or config.eval_inference_steps

    LOGGER.info("Evaluating stage=%s steps=%d device=%s", stage, steps, device)

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    # Eval skips compile/ToMe so stage timings are comparable without warmup skew.
    pipe = prepare_pipeline_for_inference(
        pipe,
        config,
        compile_unet=False,
        apply_tome=False,
        model_dir=model_dir,
    )["pipe"]

    has_metrics = False
    try:
        from evaluate import (  # type: ignore
            compute_clip_score,
            compute_lpips,
            compute_psnr,
            compute_ssim,
        )

        has_metrics = True
    except Exception as exc:  # pragma: no cover - depends on optional deps
        LOGGER.warning("Metric helpers unavailable: %s", exc)

    metrics = {
        "stage": stage,
        "num_inference_steps": steps,
        "clip_scores": [],
        "lpips_scores": [],
        "psnr_scores": [],
        "ssim_scores": [],
        "inference_times": [],
    }

    for i, prompt in enumerate(tqdm(baseline["prompts"], desc=f"eval:{stage}")):
        baseline_img = Image.open(baseline["images"][i])
        start = time.time()
        generator = torch.Generator(device=device).manual_seed(42 + i) if device != "mps" else None
        with torch.inference_mode():
            gen_image = pipe(
                prompt,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
        elapsed_ms = (time.time() - start) * 1000

        out_path = os.path.join(eval_dir, f"{stage}_{i:03d}.png")
        gen_image.save(out_path)
        metrics["inference_times"].append(elapsed_ms)

        if has_metrics:
            metrics["clip_scores"].append(compute_clip_score(gen_image, prompt, device))
            metrics["lpips_scores"].append(compute_lpips(gen_image, baseline_img, device))
            metrics["psnr_scores"].append(compute_psnr(gen_image, baseline_img))
            metrics["ssim_scores"].append(compute_ssim(gen_image, baseline_img))

    metrics["avg_clip"] = mean(metrics["clip_scores"])
    metrics["avg_lpips"] = mean(metrics["lpips_scores"])
    metrics["avg_psnr"] = mean(metrics["psnr_scores"])
    metrics["avg_ssim"] = mean(metrics["ssim_scores"])
    metrics["avg_time"] = mean(metrics["inference_times"])

    baseline_clip = baseline.get("average_clip", 0) or 0
    metrics["clip_retention"] = (
        metrics["avg_clip"] / baseline_clip if baseline_clip > 0 else 1.0
    )
    baseline_time = baseline.get("average_time", 0) or 0
    metrics["speedup"] = (
        baseline_time / metrics["avg_time"] if metrics["avg_time"] > 0 else 1.0
    )
    if compute_model_size:
        metrics["model_size_mb"] = directory_size_mb(model_dir)

    save_json(os.path.join(eval_dir, "metrics.json"), metrics)
    _log_summary(stage, metrics)

    del pipe
    free_cuda()
    return metrics


def _log_summary(stage: str, metrics: dict) -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage %s evaluation summary:", stage)
    LOGGER.info("  CLIP=%.4f retention=%.1f%%",
                metrics.get("avg_clip", 0.0), metrics.get("clip_retention", 0.0) * 100)
    LOGGER.info("  LPIPS=%.4f", metrics.get("avg_lpips", 0.0))
    LOGGER.info("  PSNR=%.2f dB", metrics.get("avg_psnr", 0.0))
    LOGGER.info("  SSIM=%.4f", metrics.get("avg_ssim", 0.0))
    LOGGER.info(
        "  Time=%.0fms speedup=%.2fx",
        metrics.get("avg_time", 0.0),
        metrics.get("speedup", 1.0),
    )
    LOGGER.info("=" * 60)


def write_full_report(config: PipelineConfig) -> dict:
    """Combine baseline + stage metrics into ``output/eval/full_report.json``."""
    baseline = _baseline_metrics(config) or {}
    stages = {}
    for stage in ("distilled", "pruned", "finetuned", "quantized"):
        metrics_path = os.path.join(config.eval_dir, stage, "metrics.json")
        if os.path.exists(metrics_path):
            stages[stage] = load_json(metrics_path)

    final = stages.get("quantized", {})
    summary = {
        "clip_retention": final.get("clip_retention", 1.0),
        "speedup": final.get("speedup", 1.0),
        "model_size_mb": final.get("model_size_mb", 0),
        "inference_steps": final.get("num_inference_steps", config.eval_inference_steps),
        "quality_target_met": final.get("clip_retention", 1.0) >= config.min_clip_retention,
    }
    report = {
        "baseline": {
            "clip": baseline.get("average_clip", 0),
            "time_ms": baseline.get("average_time", 0),
        },
        "stages": stages,
        "final": final,
        "summary": summary,
    }
    path = os.path.join(config.eval_dir, "full_report.json")
    save_json(path, report)
    LOGGER.info("Full report saved to %s", path)
    return report


__all__ = ["evaluate_stage", "write_full_report"]
