"""Centralised configuration for the compression pipeline.

All defaults live here and may be overridden by environment variables or CLI
flags. Keeping a single source of truth avoids drift between ``run.sh``,
``run.ps1`` and the individual Python modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PipelineConfig:
    """Configuration object shared by every pipeline stage."""

    # Model + data
    base_model: str = field(default_factory=lambda: _env("BASE", "runwayml/stable-diffusion-v1-5"))
    data_path: str = field(default_factory=lambda: _env("DATA", "./data/captions.json"))

    # Output roots
    output_dir: str = field(default_factory=lambda: _env("OUT", "./output"))
    distill_dir: str = field(default_factory=lambda: _env("DISTILL", "./output/distilled"))
    prune_dir: str = field(default_factory=lambda: _env("PRUNE", "./output/pruned"))
    finetune_dir: str = field(default_factory=lambda: _env("FINETUNE", "./output/finetuned"))
    quant_dir: str = field(default_factory=lambda: _env("QUANT", "./output/quant"))
    export_dir: str = field(default_factory=lambda: _env("EXPORT", "./output/export"))

    # Distillation
    steps: int = field(default_factory=lambda: _env_int("STEPS", 800))
    clip_distill_steps: int = field(default_factory=lambda: _env_int("CLIP_DISTILL_STEPS", 400))
    cfg_distill_steps: int = field(default_factory=lambda: _env_int("CFG_DISTILL_STEPS", 400))
    guidance_scale: float = field(default_factory=lambda: _env_float("GUIDANCE_SCALE", 7.5))
    lr: float = field(default_factory=lambda: _env_float("LR", 1e-5))
    ema_decay: float = field(default_factory=lambda: _env_float("EMA_DECAY", 0.9999))
    progressive_stages: str = field(default_factory=lambda: _env("PROGRESSIVE_STAGES", "50,25,12,6"))

    # Pruning
    prune_ratio: float = field(default_factory=lambda: _env_float("PRUNE_RATIO", 0.3))
    text_encoder_prune_ratio: float = field(
        default_factory=lambda: _env_float("TEXT_ENCODER_PRUNE_RATIO", 0.25)
    )
    vae_prune_ratio: float = field(default_factory=lambda: _env_float("VAE_PRUNE_RATIO", 0.2))

    # Fine-tuning after pruning
    finetune_steps: int = field(default_factory=lambda: _env_int("FINETUNE_STEPS", 200))
    finetune_lr: float = field(default_factory=lambda: _env_float("FINETUNE_LR", 5e-6))

    # Quantization
    int8_calibration_samples: int = field(
        default_factory=lambda: _env_int("INT8_CALIBRATION_SAMPLES", 100)
    )

    # Optimisation extras
    tome_ratio: float = field(default_factory=lambda: _env_float("TOME_RATIO", 0.5))
    shard_size_mb: int = field(default_factory=lambda: _env_int("SHARD_SIZE_MB", 500))

    # Linux-first CUDA runtime (safe no-ops on macOS/Windows CPU)
    enable_tf32: bool = field(default_factory=lambda: _env_bool("ENABLE_TF32", True))
    cudnn_benchmark: bool = field(default_factory=lambda: _env_bool("CUDNN_BENCHMARK", True))
    use_xformers: bool = field(default_factory=lambda: _env_bool("USE_XFORMERS", True))
    use_torch_compile: bool = field(default_factory=lambda: _env_bool("USE_TORCH_COMPILE", True))
    use_tome: bool = field(default_factory=lambda: _env_bool("USE_TOME", True))
    attention_slicing: bool = field(default_factory=lambda: _env_bool("ATTENTION_SLICING", True))
    vae_slicing: bool = field(default_factory=lambda: _env_bool("VAE_SLICING", True))
    channels_last: bool = field(default_factory=lambda: _env_bool("CHANNELS_LAST", True))
    use_amp: bool = field(default_factory=lambda: _env_bool("USE_AMP", True))
    # auto = bf16 when CUDA reports bf16 support (Ampere+), else fp16
    amp_dtype: str = field(default_factory=lambda: _env("AMP_DTYPE", "auto"))
    # auto = offload when VRAM < LOW_VRAM_GB; on/off force the behaviour
    cpu_offload: str = field(default_factory=lambda: _env("CPU_OFFLOAD", "auto"))
    low_vram_gb: float = field(default_factory=lambda: _env_float("LOW_VRAM_GB", 8.0))

    # Quality evaluation
    eval_samples: int = field(default_factory=lambda: _env_int("EVAL_SAMPLES", 4))
    eval_inference_steps: int = field(default_factory=lambda: _env_int("EVAL_INFERENCE_STEPS", 6))
    min_clip_retention: float = field(default_factory=lambda: _env_float("MIN_CLIP_RETENTION", 0.90))
    max_lpips_increase: float = field(default_factory=lambda: _env_float("MAX_LPIPS_INCREASE", 0.15))
    min_ssim_retention: float = field(default_factory=lambda: _env_float("MIN_SSIM_RETENTION", 0.85))

    # Server
    # Bind to loopback by default so `serve` is not reachable off-host. Set
    # SERVER_HOST=0.0.0.0 explicitly (behind auth / a reverse proxy) to expose it.
    server_host: str = field(default_factory=lambda: _env("SERVER_HOST", "127.0.0.1"))
    server_port: int = field(default_factory=lambda: _env_int("SERVER_PORT", 8080))
    # Server-side guard rails for the Gradio UI.
    max_batch_prompts: int = field(default_factory=lambda: _env_int("MAX_BATCH_PROMPTS", 8))
    max_inference_steps: int = field(default_factory=lambda: _env_int("MAX_INFERENCE_STEPS", 50))
    max_prompt_chars: int = field(default_factory=lambda: _env_int("MAX_PROMPT_CHARS", 1000))

    @property
    def progressive_stage_list(self) -> list[int]:
        return [int(s) for s in self.progressive_stages.split(",") if s.strip()]

    @property
    def eval_dir(self) -> str:
        return str(Path(self.output_dir) / "eval")

    def ensure_dirs(self) -> None:
        """Create every directory referenced by the configuration."""
        for path in [
            self.output_dir,
            self.distill_dir,
            self.prune_dir,
            self.finetune_dir,
            self.quant_dir,
            self.export_dir,
            self.eval_dir,
            os.path.dirname(self.data_path) or ".",
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str | None = None) -> str:
        """Serialise the configuration to JSON. Returns the path written."""
        target = path or os.path.join(self.output_dir, "pipeline_config.json")
        Path(os.path.dirname(target) or ".").mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)
        return target


DEFAULT_CAPTIONS: list[dict] = [
    {"text": "a photo of a cat sitting on a couch"},
    {"text": "a beautiful sunset over the ocean"},
    {"text": "a modern city skyline at night"},
    {"text": "a forest path in autumn with fallen leaves"},
    {"text": "a cup of coffee on a wooden table"},
    {"text": "a portrait of a person smiling"},
    {"text": "a mountain landscape with snow peaks"},
    {"text": "a bouquet of colorful flowers"},
    {"text": "an astronaut riding a horse on mars"},
    {"text": "a cyberpunk city with neon lights"},
    {"text": "a cozy cabin in snowy woods"},
    {"text": "abstract art with vibrant colors"},
    {"text": "a medieval castle on a cliff"},
    {"text": "a futuristic spaceship interior"},
    {"text": "a serene japanese garden"},
    {"text": "a steampunk mechanical owl"},
]


def ensure_captions(path: str) -> None:
    """Create a sample captions file at ``path`` if one does not already exist."""
    file_path = Path(path)
    if file_path.exists():
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(DEFAULT_CAPTIONS, fp, indent=2)
