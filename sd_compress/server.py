"""Gradio server exposing the compressed pipeline with all optimisations on."""

from __future__ import annotations

import time

from .config import PipelineConfig
from .runtime import prepare_pipeline_for_inference
from .utils import LOGGER


def launch_server(config: PipelineConfig) -> None:
    """Load the quantised pipeline, apply runtime optimisations and serve via Gradio."""
    import gradio as gr
    import torch
    from diffusers import StableDiffusionPipeline

    LOGGER.info("Loading optimised pipeline from %s", config.quant_dir)

    pipe = StableDiffusionPipeline.from_pretrained(
        config.quant_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    runtime = prepare_pipeline_for_inference(
        pipe,
        config,
        compile_unet=True,
        apply_tome=True,
        model_dir=config.quant_dir,
    )
    pipe = runtime["pipe"]

    def generate(prompt: str, num_steps: int = 4, guidance: float = 7.5):
        if not prompt.strip():
            return None
        start = time.time()
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=int(num_steps),
                guidance_scale=guidance,
            ).images[0]
        LOGGER.info("generate(prompt=%r) -> %.2fs", prompt, time.time() - start)
        return image

    def generate_batch(text: str, num_steps: int = 4, guidance: float = 7.5):
        prompts = [p.strip() for p in text.strip().splitlines() if p.strip()]
        if not prompts:
            return []
        start = time.time()
        with torch.inference_mode():
            images = pipe(
                prompts,
                num_inference_steps=int(num_steps),
                guidance_scale=guidance,
            ).images
        LOGGER.info(
            "generate_batch(%d prompts) -> %.2fs (%.2fs/image)",
            len(prompts),
            time.time() - start,
            (time.time() - start) / len(prompts),
        )
        return images

    def _flag(key: str) -> str:
        value = runtime.get(key)
        if key == "tome_modules":
            return f"enabled ({value} modules)" if value else "disabled"
        if isinstance(value, bool):
            return "enabled" if value else "disabled"
        return str(value)

    with gr.Blocks(title="Optimised SD 1.5") as demo:
        gr.Markdown("# Optimised Stable Diffusion 1.5")
        gr.Markdown(
            "Distilled, pruned, fine-tuned and quantised. Use 4–8 steps for best speed/quality."
        )

        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    prompt_box = gr.Textbox(label="Prompt", placeholder="a photo of ...")
                    steps_slider = gr.Slider(1, 20, value=4, step=1, label="Inference Steps")
                    guidance_slider = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale")
                    generate_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Generated Image")
            generate_btn.click(
                generate,
                inputs=[prompt_box, steps_slider, guidance_slider],
                outputs=[output_image],
            )

        with gr.Tab("Batch"):
            with gr.Row():
                with gr.Column():
                    batch_prompts = gr.Textbox(
                        label="Prompts (one per line)",
                        placeholder="a cat\na dog\na bird",
                        lines=5,
                    )
                    batch_steps = gr.Slider(1, 20, value=4, step=1, label="Inference Steps")
                    batch_guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale")
                    batch_btn = gr.Button("Generate All", variant="primary")
                with gr.Column():
                    output_gallery = gr.Gallery(label="Generated Images")
            batch_btn.click(
                generate_batch,
                inputs=[batch_prompts, batch_steps, batch_guidance],
                outputs=[output_gallery],
            )

        with gr.Tab("Model Info"):
            gr.Markdown(
                f"""
                ### Active Optimisations

                | Feature | Status |
                |---|---|
                | Progressive distillation | enabled (50 -> 6 steps) |
                | CLIP / CFG distillation | enabled |
                | Structured pruning | enabled (UNet / VAE / Text Encoder) |
                | Fine-tuning recovery | enabled |
                | FP16 quantisation | enabled |
                | INT8 quantisation | available (`unet_int8.pt`) |
                | Device | `{runtime.get("device")}` |
                | TF32 (Linux CUDA) | {_flag("tf32")} |
                | xFormers | {_flag("xformers")} |
                | SDPA fallback | {_flag("sdpa")} |
                | CPU offload | {_flag("cpu_offload")} |
                | Full GPU residency | {_flag("full_gpu")} |
                | Token Merging | {_flag("tome_modules")} (ratio={config.tome_ratio}) |
                | torch.compile | {_flag("torch_compile")} |
                | Attention / VAE slicing | {_flag("attention_slicing")} / {_flag("vae_slicing")} |

                ### Paths
                - Quantised model: `{config.quant_dir}`
                - ONNX export: `{config.export_dir}/onnx`
                - Sharded weights: `{config.export_dir}/sharded`
                """
            )

    LOGGER.info("Starting Gradio server on http://%s:%d", config.server_host, config.server_port)
    demo.launch(server_name=config.server_host, server_port=config.server_port)


__all__ = ["launch_server"]
