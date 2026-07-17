"""Linux-first CUDA runtime helpers for inference and training.

Linux + NVIDIA is the primary target. Helpers here enable TF32, cuDNN
benchmarking, xFormers / SDPA attention, optional Token Merging,
``torch.compile``, and VRAM-aware CPU offload. Non-CUDA platforms get safe
no-ops so the same call sites work on macOS (MPS) and CPU.
"""

from __future__ import annotations

import platform
from typing import Any

from .config import PipelineConfig
from .utils import LOGGER, select_device


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def gpu_vram_gb(device_index: int = 0) -> float | None:
    """Return total VRAM in GiB for the given CUDA device, or ``None``."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory / (1024**3)
    except Exception:  # pragma: no cover - defensive
        return None


def configure_cuda_backends(config: PipelineConfig) -> dict[str, Any]:
    """Apply Linux/CUDA backend knobs (TF32, cuDNN benchmark, matmul precision).

    Returns a dict describing what was enabled for logging / ``info``.
    """
    status: dict[str, Any] = {
        "platform": platform.system().lower(),
        "linux_first": is_linux(),
        "tf32": False,
        "cudnn_benchmark": False,
        "matmul_precision": None,
    }
    try:
        import torch
    except ImportError:  # pragma: no cover
        return status

    if not torch.cuda.is_available():
        return status

    if config.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Prefer TF32 on Ampere+ while keeping reasonable precision.
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            status["matmul_precision"] = "high"
        status["tf32"] = True

    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        status["cudnn_benchmark"] = True

    if is_linux():
        LOGGER.info(
            "Linux CUDA backends: TF32=%s cudnn.benchmark=%s",
            status["tf32"],
            status["cudnn_benchmark"],
        )
    else:
        LOGGER.info(
            "CUDA backends: TF32=%s cudnn.benchmark=%s (platform=%s)",
            status["tf32"],
            status["cudnn_benchmark"],
            status["platform"],
        )
    return status


def resolve_cpu_offload(config: PipelineConfig, *, log: bool = True) -> bool:
    """Decide whether model CPU offload should be used on CUDA.

    ``auto`` (default) enables offload on GPUs with less than ``LOW_VRAM_GB``
    VRAM so larger Linux cards stay resident on GPU for lower latency.
    """
    mode = (config.cpu_offload or "auto").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False

    vram = gpu_vram_gb()
    if vram is None:
        return True
    use_offload = vram < config.low_vram_gb
    if log:
        LOGGER.info(
            "CPU offload auto: VRAM=%.1f GiB threshold=%.1f GiB -> %s",
            vram,
            config.low_vram_gb,
            "on" if use_offload else "off (full GPU)",
        )
    return use_offload


def _enable_sdpa(pipe: Any) -> bool:
    """Prefer PyTorch scaled-dot-product attention when xFormers is unavailable."""
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0

        pipe.unet.set_attn_processor(AttnProcessor2_0())
        return True
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("SDPA attention processor unavailable: %s", exc)
        return False


def _apply_tome(pipe: Any, ratio: float) -> int:
    """Apply Token Merging wrappers to UNet attention modules. Returns count."""
    applied = 0
    try:
        # Prefer the helper shipped next to a quantised model when present.
        from pathlib import Path

        tome_path = Path(getattr(pipe, "_sd_compress_model_dir", "") or "") / "tome_utils.py"
        apply_fn = None
        if tome_path.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("tome_utils", tome_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                apply_fn = getattr(module, "apply_tome_to_attention", None)

        if apply_fn is None:
            # Inline minimal ToMe wrap mirroring optimization._TOME_TEMPLATE.
            apply_fn = _inline_apply_tome

        for name, module in pipe.unet.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"attn1", "attn2"} and hasattr(module, "forward"):
                apply_fn(module, ratio=ratio)
                applied += 1
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("ToMe application failed: %s", exc)
        return 0
    return applied


def _inline_apply_tome(attn_module: Any, ratio: float = 0.5) -> Any:
    """Minimal ToMe forward wrap used when ``tome_utils.py`` is not on disk yet."""
    import torch
    import torch.nn.functional as F

    original_forward = attn_module.forward

    def bipartite_soft_matching(metric, r):
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

    def merge_tokens(x, unm_idx, src_idx, dst_idx):
        B, N, C = x.shape
        src = x.gather(dim=1, index=src_idx.unsqueeze(-1).expand(-1, -1, C))
        dst = x.gather(dim=1, index=(dst_idx * 2 + 1).unsqueeze(-1).expand(-1, -1, C))
        unm = x.gather(dim=1, index=(unm_idx * 2).unsqueeze(-1).expand(-1, -1, C))
        return torch.cat([unm, (src + dst) / 2], dim=1)

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


def prepare_pipeline_for_inference(
    pipe: Any,
    config: PipelineConfig,
    *,
    compile_unet: bool | None = None,
    apply_tome: bool | None = None,
    model_dir: str | None = None,
) -> dict[str, Any]:
    """Apply the Linux-first inference profile to a diffusers pipeline.

    Parameters
    ----------
    compile_unet / apply_tome:
        Override config defaults. Evaluation typically disables both to avoid
        compile warmup skewing timings; the Gradio server enables them.
    """
    import torch

    if model_dir:
        pipe._sd_compress_model_dir = model_dir

    status: dict[str, Any] = {
        "device": select_device(),
        "tf32": False,
        "attention_slicing": False,
        "vae_slicing": False,
        "xformers": False,
        "sdpa": False,
        "cpu_offload": False,
        "full_gpu": False,
        "tome_modules": 0,
        "torch_compile": False,
    }

    backends = configure_cuda_backends(config)
    status["tf32"] = backends.get("tf32", False)

    if config.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(slice_size="auto")
        status["attention_slicing"] = True

    if config.vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        status["vae_slicing"] = True

    if config.use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            status["xformers"] = True
            LOGGER.info("xFormers memory-efficient attention enabled")
        except Exception as exc:
            LOGGER.info("xFormers unavailable (%s); trying SDPA", exc)
            status["sdpa"] = _enable_sdpa(pipe)
            if status["sdpa"]:
                LOGGER.info("PyTorch SDPA attention enabled")
    else:
        status["sdpa"] = _enable_sdpa(pipe)

    use_compile = config.use_torch_compile if compile_unet is None else compile_unet
    use_tome = config.use_tome if apply_tome is None else apply_tome

    device = status["device"]
    if device == "cuda":
        if resolve_cpu_offload(config):
            pipe.enable_model_cpu_offload()
            status["cpu_offload"] = True
        else:
            pipe = pipe.to("cuda")
            status["full_gpu"] = True
            LOGGER.info("Pipeline resident on CUDA (no CPU offload)")

        if use_tome:
            status["tome_modules"] = _apply_tome(pipe, config.tome_ratio)
            if status["tome_modules"]:
                LOGGER.info(
                    "Token Merging applied to %d attention modules (ratio=%.2f)",
                    status["tome_modules"],
                    config.tome_ratio,
                )

        if use_compile and hasattr(torch, "compile"):
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
                status["torch_compile"] = True
                LOGGER.info("torch.compile enabled for UNet (reduce-overhead)")
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("torch.compile failed: %s", exc)
    elif device == "mps":
        pipe = pipe.to("mps")
        LOGGER.info("Apple MPS backend active")
    else:
        LOGGER.warning("Running on CPU — inference will be slow")

    status["pipe"] = pipe
    return status


def prepare_training(config: PipelineConfig) -> str:
    """Configure CUDA backends for training and return the selected device."""
    configure_cuda_backends(config)
    device = select_device()
    if device == "cuda" and config.use_amp:
        LOGGER.info("CUDA training: AMP enabled (USE_AMP=1)")
    if device == "cuda" and config.channels_last:
        LOGGER.info("CUDA training: channels_last preferred when modules support it")
    return device


def maybe_channels_last(module: Any, config: PipelineConfig) -> Any:
    """Convert a module to channels_last on CUDA when enabled."""
    if not config.channels_last:
        return module
    try:
        import torch

        if select_device() != "cuda":
            return module
        return module.to(memory_format=torch.channels_last)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("channels_last skipped: %s", exc)
        return module


def amp_context(config: PipelineConfig, device: str):
    """Return a ``torch.autocast`` context for CUDA AMP, or a nullcontext."""
    from contextlib import nullcontext

    if not config.use_amp or device != "cuda":
        return nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.float16)


def amp_grad_scaler(config: PipelineConfig, device: str):
    """Return a CUDA GradScaler when AMP is enabled, else a disabled scaler."""
    import torch

    enabled = bool(config.use_amp and device == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def report_runtime_environment(config: PipelineConfig | None = None) -> dict[str, Any]:
    """Extended environment report including Linux-first capability flags."""
    from .utils import report_torch_environment

    info = report_torch_environment()
    info["platform"] = platform.system().lower()
    info["linux_first"] = is_linux()
    info["python_version"] = platform.python_version()

    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0)
            )
            vram = gpu_vram_gb()
            if vram is not None:
                info["cuda_vram_gb"] = round(vram, 2)
            info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
            info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except ImportError:  # pragma: no cover
        pass

    try:
        import xformers  # type: ignore

        info["xformers"] = getattr(xformers, "__version__", "present")
    except ImportError:
        info["xformers"] = None

    if config is not None:
        info["runtime_profile"] = {
            "enable_tf32": config.enable_tf32,
            "cudnn_benchmark": config.cudnn_benchmark,
            "use_xformers": config.use_xformers,
            "use_torch_compile": config.use_torch_compile,
            "use_tome": config.use_tome,
            "cpu_offload": config.cpu_offload,
            "low_vram_gb": config.low_vram_gb,
            "use_amp": config.use_amp,
            "channels_last": config.channels_last,
        }
        if info.get("device") == "cuda":
            info["resolved_cpu_offload"] = resolve_cpu_offload(config, log=False)

    return info


__all__ = [
    "amp_context",
    "amp_grad_scaler",
    "configure_cuda_backends",
    "gpu_vram_gb",
    "is_linux",
    "maybe_channels_last",
    "prepare_pipeline_for_inference",
    "prepare_training",
    "report_runtime_environment",
    "resolve_cpu_offload",
]
