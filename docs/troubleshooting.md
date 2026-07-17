# Troubleshooting

## Installation

### `torch` install picks the CPU wheel even though I have a GPU
Reinstall PyTorch using the matching CUDA index URL **first**, then re-run
`pip install -r requirements.txt`. Example for CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

### `xformers` install fails
Skip it. The pipeline detects xformers at runtime and falls back to default
attention. xFormers is genuinely unsupported on macOS and is brittle on
Windows.

### `lpips` install is very slow
LPIPS depends on `scikit-image` which sometimes builds from source on macOS /
Windows. Force binary wheels:

```bash
pip install --prefer-binary lpips
```

### `CLIP install failed` warning during `run.sh`
The CLIP repo on GitHub may be unreachable; CLIP scoring is disabled but every
other stage still works. Retry with the **pinned** commit (do not track floating
``main``):

```bash
pip install "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
```

Override the pin via ``CLIP_GIT_REF=<sha>`` when using ``./run.sh`` / ``run.ps1``.
## Runtime

### `CUDA out of memory` during distillation
Try one or more of:

- Reduce `EVAL_SAMPLES` (default 4)
- Lower `STEPS` per stage
- Reduce `PROGRESSIVE_STAGES` (e.g. `"50,12,6"` instead of `"50,25,12,6"`)
- Export `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- Run stages individually with `python -m sd_compress <stage>` so old objects
  are garbage-collected between processes

### `RuntimeError: Sizes of tensors must match` after pruning
The pruning recipe is conservative but not perfect for every checkpoint; some
custom UNet variants have shape-coupling that is not captured by the skip
lists. Lower `PRUNE_RATIO` (or `VAE_PRUNE_RATIO` / `TEXT_ENCODER_PRUNE_RATIO`)
or restrict pruning to one component with `--component unet`.

### `torch.compile` errors with Triton
Either reinstall PyTorch (Triton ships with the GPU wheels) or disable compile
by exporting `TORCHINDUCTOR_DISABLE=1`. The pipeline functions correctly
without compile, just slightly slower.

### MPS (Apple Silicon) error: `Placeholder storage has not been allocated`
This is typically caused by mixing CPU- and MPS-side tensors. Reduce
parallelism by lowering `EVAL_SAMPLES` and avoid running multiple GPU-heavy
apps simultaneously. If the problem persists, fall back to CPU by exporting
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

### Hugging Face: `OSError: Can't load tokenizer` / `connection error`
The first run downloads the base model from Hugging Face Hub. If you do not
have internet access, mirror the checkpoint locally and pass it via
`--base-model /path/to/checkpoint`. To stop the cache from making network
calls thereafter, export `HF_HUB_OFFLINE=1`.

### Gradio server reports "Address already in use"
Either kill the existing server or pick another port:

```bash
SERVER_PORT=7860 python -m sd_compress serve
```

### Evaluation reports `CLIP=0.0`
CLIP could not be imported; reinstall it (see above). Other metrics (LPIPS /
PSNR / SSIM) still work without CLIP.

## Quality

### CLIP retention is below `MIN_CLIP_RETENTION`
The pipeline only warns — it does not abort. Options:

1. Increase `STEPS` (distillation) and `FINETUNE_STEPS`.
2. Lower `PRUNE_RATIO`.
3. Remove the CFG distillation stage if you need a variable guidance slider.
4. Run with smaller `PROGRESSIVE_STAGES` (e.g. stop at 12 instead of 6).

### LPIPS spikes after pruning but recovers after fine-tune
This is expected. The fine-tune stage exists specifically to recover the
quality lost to pruning. Inspect `output/eval/finetuned/metrics.json` to
confirm.

### Quantised model differs from FP16 baseline
Dynamic INT8 changes Linear / Conv2d numerics. If the divergence is severe,
keep the FP16 pipeline at `output/quant/` and ignore `unet_int8.pt`. INT8 is
intended for CPU x86 deployment where its memory advantage matters most.

### Loading `unet_int8.pt` safely
Dynamic INT8 uses torch quantized tensors, so the sidecar is ``.pt`` (not
safetensors). Always load with ``weights_only=True`` via the package helper —
never bare ``torch.load`` on an untrusted file:

```python
from sd_compress.quantization import load_int8_state_dict
state = load_int8_state_dict("./output/quant/unet_int8.pt")
```
