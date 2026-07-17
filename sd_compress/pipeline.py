"""End-to-end orchestrator for the compression pipeline."""

from __future__ import annotations

from .baseline import generate_baseline
from .config import PipelineConfig, ensure_captions
from .distillation import cfg_distillation, clip_text_distillation, progressive_distillation
from .eval_runner import evaluate_stage, write_full_report
from .export import export_onnx, shard_safetensors
from .finetune import finetune_after_pruning
from .lora import test_lora_compatibility
from .optimization import install_runtime_helpers
from .pruning import prune_all
from .quantization import quantize
from .utils import LOGGER
from .runtime import report_runtime_environment


def run_full_pipeline(config: PipelineConfig, launch_server_after: bool = False) -> None:
    """Run every stage end-to-end. Each stage is safe to skip individually via the CLI."""
    config.ensure_dirs()
    ensure_captions(config.data_path)
    config.dump()

    env = report_runtime_environment(config)
    LOGGER.info("Environment: %s", env)

    LOGGER.info("=== Stage 0: baseline references ===")
    generate_baseline(config)

    LOGGER.info("=== Stage 1: progressive distillation ===")
    progressive_distillation(config)

    LOGGER.info("=== Eval: post-distillation ===")
    evaluate_stage(config, "distilled", config.distill_dir)

    LOGGER.info("=== Stage 2: CLIP text encoder distillation ===")
    clip_text_distillation(config)

    LOGGER.info("=== Stage 3: CFG distillation ===")
    cfg_distillation(config)

    LOGGER.info("=== Stage 4: structured pruning (text encoder, VAE, UNet) ===")
    prune_all(config)
    LOGGER.info("=== Eval: post-pruning ===")
    evaluate_stage(config, "pruned", config.prune_dir, compute_model_size=True)

    LOGGER.info("=== Stage 5: fine-tuning recovery ===")
    finetune_after_pruning(config)
    LOGGER.info("=== Eval: post-finetuning ===")
    evaluate_stage(config, "finetuned", config.finetune_dir)

    LOGGER.info("=== Stage 6: quantisation ===")
    quantize(config)
    LOGGER.info("=== Eval: final quantised model ===")
    evaluate_stage(config, "quantized", config.quant_dir, compute_model_size=True)

    LOGGER.info("=== Stage 7: runtime optimisations (ToMe + torch.compile helpers) ===")
    install_runtime_helpers(config)

    LOGGER.info("=== Stage 8: ONNX export & sharding ===")
    export_onnx(config)
    shard_safetensors(config)

    LOGGER.info("=== Stage 9: LoRA compatibility test ===")
    test_lora_compatibility(config)

    LOGGER.info("=== Final report ===")
    write_full_report(config)

    if launch_server_after:
        from .server import launch_server

        LOGGER.info("=== Stage 10: serving with Gradio ===")
        launch_server(config)


__all__ = ["run_full_pipeline"]
