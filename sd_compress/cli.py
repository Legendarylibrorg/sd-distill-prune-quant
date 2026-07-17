"""Command-line interface for the ``sd_compress`` package.

Examples
--------
.. code-block:: bash

    # Run the full pipeline
    python -m sd_compress run

    # Or run a single stage (lets you resume / iterate on failures)
    python -m sd_compress distill-progressive
    python -m sd_compress evaluate --stage distilled --model-dir ./output/distilled

    # Launch only the inference server against an existing quantised model
    python -m sd_compress serve
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import PipelineConfig, ensure_captions
from .utils import configure_logging


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    # Allow command-line overrides for the most common knobs.
    for attr in (
        "base_model",
        "data_path",
        "output_dir",
        "distill_dir",
        "prune_dir",
        "finetune_dir",
        "quant_dir",
        "export_dir",
    ):
        value = getattr(args, attr, None)
        if value:
            setattr(config, attr, value)
    return config


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", help="Hugging Face model id or local path")
    parser.add_argument("--data-path", help="Path to captions.json")
    parser.add_argument("--output-dir", help="Root output directory")
    parser.add_argument("--distill-dir", help="Override distilled model directory")
    parser.add_argument("--prune-dir", help="Override pruned model directory")
    parser.add_argument("--finetune-dir", help="Override fine-tuned model directory")
    parser.add_argument("--quant-dir", help="Override quantised model directory")
    parser.add_argument("--export-dir", help="Override export directory")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sd_compress",
        description="Stable Diffusion 1.5 compression pipeline",
    )
    parser.add_argument("--version", action="version", version=f"sd_compress {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        _add_common_args(sub)
        return sub

    add("info", "Print torch/CUDA environment summary")

    run_p = add("run", "Run the full compression pipeline end-to-end")
    run_p.add_argument(
        "--serve",
        action="store_true",
        help="Launch the Gradio server after the pipeline finishes",
    )

    add("baseline", "Generate baseline reference images")

    add("distill-progressive", "Progressive step-halving distillation")
    add("distill-clip", "Distil the CLIP text encoder")
    add("distill-cfg", "Distil classifier-free guidance into a single pass")

    prune_p = add("prune", "Run all structured pruning stages")
    prune_p.add_argument(
        "--component",
        choices=["all", "text-encoder", "vae", "unet"],
        default="all",
        help="Limit pruning to a single component",
    )

    add("finetune", "Short fine-tune to recover quality after pruning")
    add("quantize", "FP16 + INT8 quantisation")
    add("optimize", "Install ToMe + torch.compile runtime helpers")

    export_p = add("export", "Export ONNX and/or shard safetensors")
    export_p.add_argument(
        "--target",
        choices=["all", "onnx", "shard"],
        default="all",
        help="Which export artefact(s) to produce",
    )

    add("lora-test", "LoRA compatibility check on the quantised model")

    eval_p = add("evaluate", "Evaluate one stage against the baseline references")
    eval_p.add_argument("--stage", required=True, help="Logical stage name (e.g. distilled)")
    eval_p.add_argument(
        "--model-dir", required=True, help="Directory of the diffusers pipeline to evaluate"
    )
    eval_p.add_argument("--num-steps", type=int, help="Override inference steps for evaluation")

    add("report", "Aggregate stage metrics into output/eval/full_report.json")
    add("serve", "Launch the Gradio inference server (requires a quantised model)")

    return parser


def _execute(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = _config_from_args(args)
    config.ensure_dirs()
    ensure_captions(config.data_path)

    command = args.command

    if command == "info":
        from .runtime import report_runtime_environment
        from .utils import LOGGER

        LOGGER.info("sd_compress %s", __version__)
        LOGGER.info("Environment: %s", report_runtime_environment(config))
        LOGGER.info("Configuration: %s", config.to_dict())
        return 0

    if command == "run":
        from .pipeline import run_full_pipeline

        run_full_pipeline(config, launch_server_after=args.serve)
        return 0

    if command == "baseline":
        from .baseline import generate_baseline

        generate_baseline(config)
        return 0

    if command == "distill-progressive":
        from .distillation import progressive_distillation

        progressive_distillation(config)
        return 0

    if command == "distill-clip":
        from .distillation import clip_text_distillation

        clip_text_distillation(config)
        return 0

    if command == "distill-cfg":
        from .distillation import cfg_distillation

        cfg_distillation(config)
        return 0

    if command == "prune":
        from .pruning import prune_all, prune_text_encoder, prune_unet, prune_vae

        mapping = {
            "all": prune_all,
            "text-encoder": prune_text_encoder,
            "vae": prune_vae,
            "unet": prune_unet,
        }
        mapping[args.component](config)
        return 0

    if command == "finetune":
        from .finetune import finetune_after_pruning

        finetune_after_pruning(config)
        return 0

    if command == "quantize":
        from .quantization import quantize

        quantize(config)
        return 0

    if command == "optimize":
        from .optimization import install_runtime_helpers

        install_runtime_helpers(config)
        return 0

    if command == "export":
        from .export import export_onnx, shard_safetensors

        if args.target in {"all", "onnx"}:
            export_onnx(config)
        if args.target in {"all", "shard"}:
            shard_safetensors(config)
        return 0

    if command == "lora-test":
        from .lora import test_lora_compatibility

        test_lora_compatibility(config)
        return 0

    if command == "evaluate":
        from .eval_runner import evaluate_stage

        evaluate_stage(
            config,
            stage=args.stage,
            model_dir=args.model_dir,
            num_inference_steps=args.num_steps,
        )
        return 0

    if command == "report":
        from .eval_runner import write_full_report

        write_full_report(config)
        return 0

    if command == "serve":
        from .server import launch_server

        launch_server(config)
        return 0

    raise SystemExit(f"Unknown command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return _execute(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
