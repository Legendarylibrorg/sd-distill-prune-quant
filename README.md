# SD 1.5 Compression & Optimization Pipeline

A modular, end-to-end toolkit that takes a stock **Stable Diffusion 1.5** checkpoint
through progressive distillation, structured pruning, fine-tuning, quantisation,
runtime optimisation (Token Merging + `torch.compile`), ONNX export and a Gradio
inference server — with **per-stage quality evaluation** (CLIP / LPIPS / PSNR / SSIM)
baked into every step.

> **Result:** typically ~8–10× faster inference at ~3 GB VRAM with ≥90 % CLIP
> retention versus the 50-step baseline (see [Performance](#performance)).

---

## Contents

1. [Quick Start](#quick-start)
2. [Setup — Linux (recommended)](#setup--linux-recommended)
3. [Setup — macOS](#setup--macos)
4. [Setup — Windows](#setup--windows)
5. [Pipeline stages](#pipeline-stages)
6. [Configuration](#configuration)
7. [CLI reference](#cli-reference)
8. [Output layout](#output-layout)
9. [Quality metrics](#quality-metrics)
10. [Inference examples](#inference-examples)
11. [Performance](#performance)
12. [Project layout](#project-layout)
13. [Limitations](#limitations)
14. [References](#references)

---

## Quick Start

```bash
# Linux / macOS
git clone https://github.com/Legendarylibrorg/sd-distill-prune-quant.git
cd sd-distill-prune-quant
./run.sh
```

```powershell
# Windows (PowerShell)
git clone https://github.com/Legendarylibrorg/sd-distill-prune-quant.git
cd sd-distill-prune-quant
.\run.ps1
```

`run.sh` / `run.ps1` create a virtualenv, install dependencies, run the full
pipeline and start the Gradio UI on <http://localhost:8080>.
For finer control use the CLI directly: `python -m sd_compress --help`.

Hardware: an NVIDIA GPU with **≥6 GB VRAM** is strongly recommended. CPU or
Apple-Silicon (MPS) execution works but is much slower; use a small
`PROGRESSIVE_STAGES`/`STEPS` budget in that case.

---

## Setup — Linux (recommended)

Linux + NVIDIA + CUDA is the primary development target.

### 1. System prerequisites

The exact package names vary by distribution. On Debian / Ubuntu:

```bash
sudo apt update
sudo apt install -y \
    python3 python3-venv python3-pip \
    git build-essential
```

For NVIDIA acceleration, install the proprietary driver and verify CUDA is
visible:

```bash
nvidia-smi   # should list at least one GPU and a CUDA version
```

If you do not have a GPU you can still run the pipeline — see
[CPU-only / low-resource mode](#cpu-only--low-resource-mode) at the end of this
section.

### 2. Clone the repository

```bash
git clone https://github.com/Legendarylibrorg/sd-distill-prune-quant.git
cd sd-distill-prune-quant
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 4. Install PyTorch with CUDA

The `requirements.txt` does **not** pin a CUDA build of PyTorch (so the same
file works on every platform). Install the right CUDA wheel **first**, then
the rest of the requirements:

```bash
# CUDA 12.1 (most current Ampere/Ada/Hopper GPUs)
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision

# CUDA 11.8 (older GPUs / drivers)
# pip install --index-url https://download.pytorch.org/whl/cu118 \
#     torch torchvision
```

Pick the [official build matrix](https://pytorch.org/get-started/locally/) that
matches your driver.

### 5. Install the rest of the requirements

```bash
pip install -r requirements.txt

# CLIP score (optional but recommended)
pip install "git+https://github.com/openai/CLIP.git"

# xFormers memory-efficient attention (optional, big speedup when it works)
pip install xformers
```

### 6. Verify the installation

```bash
python -m sd_compress info
```

You should see a line like `Environment: {'device': 'cuda', ...}`.

### 7. Run the full pipeline

```bash
./run.sh                 # default: full pipeline + Gradio server
# or
./run.sh --no-serve      # full pipeline, no server
# or run individual stages:
./run.sh baseline
./run.sh distill-progressive
./run.sh prune --component unet
./run.sh evaluate --stage pruned --model-dir ./output/pruned
```

When the server starts, open <http://localhost:8080>.

#### CPU-only / low-resource mode

```bash
# Drastically reduce the training budget so it finishes in minutes, not hours
export STEPS=50
export CLIP_DISTILL_STEPS=25
export CFG_DISTILL_STEPS=25
export FINETUNE_STEPS=25
export INT8_CALIBRATION_SAMPLES=10
export EVAL_SAMPLES=2
export PROGRESSIVE_STAGES="50,25,6"
./run.sh
```

This is enough to validate the plumbing end-to-end; the quality numbers will
not match a full run.

---

## Setup — macOS

macOS runs on the **MPS (Metal Performance Shaders)** backend on Apple Silicon
or on the CPU on Intel Macs. CUDA-only features (FP16 INT8 wheels, `xformers`,
`torch.compile` GPU mode) are skipped automatically.

### 1. System prerequisites

Install the Apple command-line developer tools and a recent Python:

```bash
xcode-select --install     # if not already installed

# Recommended: install Python 3.11+ via Homebrew
brew install python git
```

### 2. Clone the repository

```bash
git clone https://github.com/Legendarylibrorg/sd-distill-prune-quant.git
cd sd-distill-prune-quant
```

### 3. Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 4. Install dependencies

Standard PyPI wheels work on macOS:

```bash
pip install -r requirements.txt
pip install "git+https://github.com/openai/CLIP.git"   # optional
```

Skip `xformers` — it does not ship for macOS.

### 5. Verify

```bash
python -m sd_compress info
# Expect: "device": "mps" on Apple Silicon, "cpu" on Intel.
```

### 6. Run the pipeline

```bash
# Reduce the training budget — MPS is slower than CUDA
export PROGRESSIVE_STAGES="50,12,6"
export STEPS=100
export CLIP_DISTILL_STEPS=50
export FINETUNE_STEPS=50

./run.sh
```

If you hit out-of-memory errors on MPS, lower `EVAL_SAMPLES` and use
`./run.sh --no-serve` so the Gradio server is not loaded in parallel.

> **Note:** dynamic INT8 quantisation in PyTorch targets CPU x86 backends. On
> Apple Silicon the FP16 pipeline is what you should ship; the saved
> `unet_int8.pt` still loads but only delivers speedups on x86 CPUs.

---

## Setup — Windows

Windows 10/11 with PowerShell 7+ is supported. CUDA acceleration requires the
NVIDIA driver and the matching PyTorch wheel.

### 1. System prerequisites

1. **Python 3.10+** from <https://www.python.org/downloads/windows/> (tick
   *"Add python.exe to PATH"* during install).
2. **Git for Windows** from <https://git-scm.com/download/win>.
3. (For GPU acceleration) The NVIDIA driver bundled with CUDA 11.8 or 12.x.
   Verify with:

   ```powershell
   nvidia-smi
   ```

4. (Recommended) **Windows Terminal** + **PowerShell 7**.

If your PowerShell refuses to run `.ps1` files, enable scripts once per user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 2. Clone the repository

```powershell
git clone https://github.com/Legendarylibrorg/sd-distill-prune-quant.git
cd sd-distill-prune-quant
```

### 3. Virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4. Install PyTorch with CUDA

```powershell
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 `
    torch torchvision

# CUDA 11.8 (older drivers)
# pip install --index-url https://download.pytorch.org/whl/cu118 `
#     torch torchvision
```

### 5. Install the remaining requirements

```powershell
pip install -r requirements.txt
pip install "git+https://github.com/openai/CLIP.git"

# Optional, often unavailable on Windows
# pip install xformers
```

### 6. Verify

```powershell
python -m sd_compress info
```

### 7. Run the pipeline

```powershell
.\run.ps1                                            # full pipeline + Gradio server
.\run.ps1 --no-serve                                 # pipeline only
.\run.ps1 distill-progressive                        # single stage
.\run.ps1 evaluate --stage pruned --model-dir .\output\pruned
```

#### Common Windows gotchas

- **Long path errors.** Enable long paths:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWord -Force`
- **`pip install xformers` fails.** Skip it — the pipeline detects it at runtime
  and falls back to the default attention implementation.
- **Anti-virus / Defender slowing torch.compile.** Add the repository folder to
  your Defender exclusions.
- **Use `python` not `python3`.** On Windows the launcher is `python`.

---

## Pipeline stages

```
0. Baseline       -- 50-step reference images with the original model
1. Distillation   -- progressive step halving (50 -> 25 -> 12 -> 6) + attention KD + EMA
   * CLIP         -- text encoder hidden-state / pooled / intermediate KD
   * CFG          -- merge classifier-free guidance into a single forward pass
2. Pruning        -- L1 structured pruning of UNet, VAE, text encoder
3. Fine-tuning    -- short ε-prediction recovery pass
4. Quantisation   -- FP16 pipeline + INT8 UNet state-dict
5. Optimisations  -- ToMe helper + torch.compile wrapper dropped into the model dir
6. Export         -- ONNX (UNet, VAE decoder) and safetensors shards
7. LoRA test      -- structural compatibility check of the compressed model
8. Server         -- Gradio UI with batch + model-info tabs
```

Quality evaluation runs **after every weight-changing stage** (distilled,
pruned, fine-tuned, quantised) using the prompts and seeds from stage 0, and
results are written to `output/eval/<stage>/metrics.json`. A final
`output/eval/full_report.json` aggregates everything.

---

## Configuration

Every knob is exposed via environment variables (read by `sd_compress.config.PipelineConfig`),
so the same defaults work for `run.sh`, `run.ps1` and `python -m sd_compress`.

| Variable | Default | Description |
| --- | --- | --- |
| **Model + paths** | | |
| `BASE` | `runwayml/stable-diffusion-v1-5` | Source checkpoint |
| `DATA` | `./data/captions.json` | Caption file (auto-generated if missing) |
| `OUT` | `./output` | Root output directory |
| `DISTILL` / `PRUNE` / `FINETUNE` / `QUANT` / `EXPORT` | `$OUT/<stage>` | Per-stage output |
| **Distillation** | | |
| `STEPS` | `800` | Steps per progressive stage |
| `CLIP_DISTILL_STEPS` | `400` | CLIP encoder distillation steps |
| `CFG_DISTILL_STEPS` | `400` | CFG distillation steps |
| `PROGRESSIVE_STAGES` | `50,25,12,6` | Step-halving schedule |
| `LR` | `1e-5` | Base learning rate |
| `EMA_DECAY` | `0.9999` | EMA decay for the student |
| `GUIDANCE_SCALE` | `7.5` | Guidance baked into the CFG-distilled model |
| **Pruning** | | |
| `PRUNE_RATIO` | `0.3` | UNet channel pruning ratio |
| `TEXT_ENCODER_PRUNE_RATIO` | `0.25` | Text encoder MLP pruning |
| `VAE_PRUNE_RATIO` | `0.2` | VAE channel pruning |
| **Fine-tuning** | | |
| `FINETUNE_STEPS` | `200` | Steps of recovery training |
| `FINETUNE_LR` | `5e-6` | Recovery learning rate |
| **Quantisation / optimisation** | | |
| `INT8_CALIBRATION_SAMPLES` | `100` | Samples used for INT8 calibration |
| `TOME_RATIO` | `0.5` | Token-merging ratio applied by the helper |
| `SHARD_SIZE_MB` | `500` | Target shard size for sharded safetensors |
| **Evaluation** | | |
| `EVAL_SAMPLES` | `4` | Number of evaluation prompts |
| `EVAL_INFERENCE_STEPS` | `6` | Step count used during evaluation |
| `MIN_CLIP_RETENTION` | `0.90` | Minimum acceptable CLIP retention |
| `MAX_LPIPS_INCREASE` | `0.15` | Maximum acceptable LPIPS increase |
| `MIN_SSIM_RETENTION` | `0.85` | Minimum acceptable SSIM retention |
| **Server** | | |
| `SERVER_HOST` | `0.0.0.0` | Gradio host |
| `SERVER_PORT` | `8080` | Gradio port |

Override any of them per-stage, e.g.:

```bash
PRUNE_RATIO=0.2 ./run.sh prune
```

---

## CLI reference

```text
python -m sd_compress --help

  run                   Full pipeline (add --serve to launch the UI after)
  baseline              Generate baseline reference images
  distill-progressive   Step-halving distillation
  distill-clip          CLIP text encoder distillation
  distill-cfg           CFG distillation
  prune [--component]   Structured pruning (all | text-encoder | vae | unet)
  finetune              Short recovery fine-tuning
  quantize              FP16 + INT8 quantisation
  optimize              Drop ToMe + torch.compile helpers next to the model
  export [--target]     ONNX export + sharded safetensors (all | onnx | shard)
  lora-test             LoRA structural compatibility report
  evaluate --stage --model-dir [--num-steps]
                        Score a model directory against the baseline references
  report                Aggregate stage metrics into full_report.json
  serve                 Launch the Gradio inference server
  info                  Print torch/CUDA environment + active configuration
```

All commands accept the same path overrides (`--base-model`, `--data-path`,
`--output-dir`, `--distill-dir`, …), so you can stitch together custom
pipelines without editing the source.

---

## Output layout

```
output/
├── eval/                       Quality metrics + generated images per stage
│   ├── baseline/
│   ├── distilled/
│   ├── pruned/
│   ├── finetuned/
│   ├── quantized/
│   └── full_report.json        Aggregated cross-stage report
├── pipeline_config.json        Snapshot of the configuration used
├── distilled/                  Progressive + CLIP + CFG distilled pipeline
├── pruned/                     All three components pruned
├── finetuned/                  Post-pruning fine-tuned pipeline
├── quant/                      FP16 pipeline + INT8 state-dict + ToMe / compile helpers
├── export/
│   ├── onnx/                   unet.onnx, vae_decoder.onnx
│   └── sharded/                unet_shard_NNN.safetensors + shard_index.json
└── lora_compatibility_report.json
```

---

## Quality metrics

| Metric | Description | Target |
| --- | --- | --- |
| **CLIP score** | Text-image alignment (higher is better) | ≥ 90 % of baseline |
| **LPIPS** | Perceptual distance to baseline (lower is better) | ≤ 0.15 increase |
| **PSNR** | Peak signal-to-noise ratio | > 15 dB |
| **SSIM** | Structural similarity | ≥ 85 % of baseline retention |

Thresholds are configurable (`MIN_CLIP_RETENTION`, `MAX_LPIPS_INCREASE`,
`MIN_SSIM_RETENTION`). The pipeline only prints warnings — it does not abort —
so you can always inspect the artefacts before deciding whether to ship.

Example aggregated report (truncated):

```text
======================================================================
FINAL COMPRESSION PIPELINE QUALITY REPORT
======================================================================

Stage           CLIP↑      LPIPS↓     PSNR↑      SSIM↑      Time(ms)
----------------------------------------------------------------------
Baseline        0.3245     -          -          -          3200
Distilled       0.3180     0.0823     22.45      0.8534     520
Pruned          0.2998     0.1156     19.87      0.7823     480
Finetuned       0.3102     0.0945     21.23      0.8245     485
QUANTIZED       0.3098     0.0952     21.18      0.8231     350
======================================================================
Quality Retention: 95.5%   Speedup: 9.1x   Model Size: 2100 MB
✅ SUCCESS: Quality target met (>=90% retention)
```

---

## Inference examples

### Basic (4 steps, FP16)

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "./output/quant",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = pipe("a cat astronaut on mars", num_inference_steps=4).images[0]
image.save("cat.png")
```

### With `torch.compile`

```python
from output.quant.compile_utils import load_compiled_pipeline, warmup_pipeline

pipe = load_compiled_pipeline("./output/quant")
warmup_pipeline(pipe)  # first call is slow because of compilation
image = pipe("a cat astronaut on mars", num_inference_steps=4).images[0]
```

### With Token Merging

```python
from output.quant.tome_utils import apply_tome_to_attention

for name, module in pipe.unet.named_modules():
    if "attn" in name:
        apply_tome_to_attention(module, ratio=0.5)
```

### INT8 (CPU)

```python
import torch
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("./output/quant", subfolder="unet")
unet = torch.quantization.quantize_dynamic(
    unet, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
unet.load_state_dict(torch.load("./output/quant/unet_int8.pt"))
```

### TensorRT (NVIDIA)

```bash
trtexec --onnx=./output/export/onnx/unet.onnx \
        --saveEngine=./output/export/unet.trt \
        --fp16
```

---

## Performance

Indicative numbers on an A100 80 GB, batch=1, 512×512. Your numbers will vary —
treat this as a relative ordering, not an absolute claim.

| Configuration | Steps | Time | VRAM | Quality |
| --- | --- | --- | --- | --- |
| Original SD 1.5 | 50 | 3.2 s | 8 GB | baseline |
| + Progressive distillation | 6 | 0.50 s | 8 GB | ~95 % |
| + Structured pruning (30 %) | 6 | 0.40 s | 6 GB | ~92 % |
| + Fine-tuning recovery | 6 | 0.40 s | 6 GB | ~94 % |
| + FP16 quantisation | 6 | 0.35 s | 3 GB | ~94 % |
| + `torch.compile` | 6 | 0.25 s | 3 GB | ~94 % |
| + Token Merging (ratio=0.5) | 6 | 0.18 s | 3 GB | ~92 % |

---

## Project layout

```
sd-distill-prune-quant/
├── README.md                  This file
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml             Installable as `pip install .`
├── run.sh                     Linux / macOS wrapper around the CLI
├── run.ps1                    Windows PowerShell wrapper
├── evaluate.py                Quality metrics (CLIP / LPIPS / PSNR / SSIM)
├── sd_compress/               Pipeline package
│   ├── __init__.py
│   ├── __main__.py            `python -m sd_compress`
│   ├── cli.py                 argparse CLI
│   ├── config.py              PipelineConfig (env-driven)
│   ├── utils.py               Shared helpers (device, logging, JSON, ...)
│   ├── baseline.py            Stage 0
│   ├── distillation.py        Progressive / CLIP / CFG distillation
│   ├── pruning.py             UNet / VAE / Text-encoder pruning
│   ├── finetune.py            Recovery training
│   ├── quantization.py        FP16 + INT8
│   ├── optimization.py        Drops ToMe + compile helpers into the quant dir
│   ├── export.py              ONNX + safetensors sharding
│   ├── lora.py                LoRA compatibility check
│   ├── eval_runner.py         Cross-stage evaluation harness
│   ├── server.py              Gradio UI
│   └── pipeline.py            Orchestrates every stage
├── data/                      Captions (auto-generated on first run)
├── output/                    All pipeline artefacts (gitignored)
└── docs/
    ├── setup-linux.md
    ├── setup-macos.md
    ├── setup-windows.md
    ├── pipeline.md
    ├── usage.md
    └── troubleshooting.md
```

---

## Limitations

1. **Distillation budgets.** The default `STEPS=800` per stage is *demonstration*
   scale. Production-grade distillation usually needs 10 k–100 k steps per stage.
2. **Aggressive pruning.** Going beyond ~40 % typically requires longer
   fine-tuning to recover.
3. **CFG distillation.** Bakes in a single guidance scale; if you need a slider,
   skip that stage or distil at multiple guidance values.
4. **INT8.** Best for CPU x86 deployments; on CUDA the FP16 path is usually
   faster and higher quality.
5. **LoRA.** Heavily pruned UNets may have mismatched shapes versus stock LoRAs;
   use the bundled compatibility report.
6. **Token Merging.** Aggressive ratios degrade fine details in complex scenes.

---

## References

- [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512)
- [Classifier-Free Guidance Distillation](https://arxiv.org/abs/2306.05284)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)
- [Token Merging (ToMe)](https://arxiv.org/abs/2210.09461)
- [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

---

## License

Released under the terms of the [MIT License](LICENSE).
