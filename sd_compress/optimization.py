"""Generate runtime optimisation helper modules.

These helpers (token merging + ``torch.compile`` wrapper) are written into the
quantised output directory so they can be shipped alongside the model and used
without re-installing the full pipeline package.
"""

from __future__ import annotations

import os
import textwrap

from .config import PipelineConfig
from .utils import LOGGER

_TOME_TEMPLATE = textwrap.dedent(
    '''
    """Token Merging (ToMe) helpers for attention modules."""

    import torch
    import torch.nn.functional as F


    def bipartite_soft_matching(metric, r):
        """Compute bipartite soft matching indices for ToMe."""
        B, N, C = metric.shape
        if r <= 0:
            return None, None, None

        with torch.no_grad():
            metric = F.normalize(metric, dim=-1)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)

            unm_idx = edge_idx[..., r:]
            src_idx = edge_idx[..., :r]
            dst_idx = node_idx.gather(dim=-1, index=src_idx)

        return unm_idx, src_idx, dst_idx


    def merge_tokens(x, unm_idx, src_idx, dst_idx, mode="mean"):
        """Apply ToMe merging based on indices produced by ``bipartite_soft_matching``."""
        B, N, C = x.shape
        src = x.gather(dim=1, index=src_idx.unsqueeze(-1).expand(-1, -1, C))
        dst = x.gather(dim=1, index=(dst_idx * 2 + 1).unsqueeze(-1).expand(-1, -1, C))
        unm = x.gather(dim=1, index=(unm_idx * 2).unsqueeze(-1).expand(-1, -1, C))

        merged = (src + dst) / 2 if mode == "mean" else dst
        return torch.cat([unm, merged], dim=1)


    def apply_tome_to_attention(attn_module, ratio=0.5):
        """Wrap ``attn_module.forward`` with token merging."""
        original_forward = attn_module.forward

        def tome_forward(hidden_states, *args, **kwargs):
            B, N, C = hidden_states.shape
            r = int(N * ratio / 2)
            if r > 0 and N > 4:
                unm_idx, src_idx, dst_idx = bipartite_soft_matching(hidden_states, r)
                if unm_idx is not None:
                    hidden_states = merge_tokens(hidden_states, unm_idx, src_idx, dst_idx)
            return original_forward(hidden_states, *args, **kwargs)

        attn_module.forward = tome_forward
        return attn_module
    '''
).strip() + "\n"


_COMPILE_TEMPLATE = textwrap.dedent(
    '''
    """``torch.compile`` wrapper for the quantised pipeline (Linux CUDA first)."""

    import torch
    from diffusers import StableDiffusionPipeline


    def load_compiled_pipeline(model_path, compile_mode="reduce-overhead", full_gpu=True):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if torch.cuda.is_available():
            # Prefer TF32 + cuDNN autotune on Linux NVIDIA GPUs.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            if full_gpu:
                pipe = pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()

            if hasattr(torch, "compile"):
                pipe.unet = torch.compile(pipe.unet, mode=compile_mode)
                pipe.vae.decode = torch.compile(pipe.vae.decode, mode=compile_mode)
        return pipe


    def warmup_pipeline(pipe, prompt="warmup", steps=2):
        with torch.inference_mode():
            pipe(prompt, num_inference_steps=steps, output_type="latent")


    if __name__ == "__main__":
        import sys

        model_path = sys.argv[1] if len(sys.argv) > 1 else "./output/quant"
        warmup_pipeline(load_compiled_pipeline(model_path))
    '''
).strip() + "\n"


def install_runtime_helpers(config: PipelineConfig) -> None:
    """Write ToMe and torch.compile helper modules next to the quantised model."""
    os.makedirs(config.quant_dir, exist_ok=True)
    tome_path = os.path.join(config.quant_dir, "tome_utils.py")
    compile_path = os.path.join(config.quant_dir, "compile_utils.py")

    with open(tome_path, "w", encoding="utf-8") as fp:
        fp.write(_TOME_TEMPLATE)
    with open(compile_path, "w", encoding="utf-8") as fp:
        fp.write(_COMPILE_TEMPLATE)

    LOGGER.info("ToMe helpers written to %s", tome_path)
    LOGGER.info("torch.compile helper written to %s", compile_path)


__all__ = ["install_runtime_helpers"]
