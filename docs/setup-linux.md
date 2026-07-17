# Linux setup (recommended)

This is the primary supported platform. Follow these steps top-to-bottom for a
fresh machine.

## 1. System packages

Tested on Ubuntu 22.04 / Debian 12; equivalent packages exist on every major
distribution.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential
```

### NVIDIA driver + CUDA toolkit

Install the proprietary driver via your distribution's package manager or the
NVIDIA installer, then verify:

```bash
nvidia-smi
```

You should see at least one GPU and a CUDA version. The pipeline supports CUDA
11.8 and 12.x; pick the matching PyTorch wheel in step 4.

## 2. Clone

```bash
git clone https://github.com/Legendarylibrorg/sd-distill-prune-quant.git
cd sd-distill-prune-quant
```

## 3. Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 4. PyTorch (CUDA build)

Install the CUDA-flavoured PyTorch wheel **before** the rest of the
requirements so pip does not pick up a CPU-only torch from PyPI.

```bash
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# CUDA 11.8 (older drivers)
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

Reference: <https://pytorch.org/get-started/locally/>.

## 5. Remaining requirements

```bash
pip install -r requirements.txt
pip install "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"   # optional
pip install xformers                                   # optional, often speeds attention
```

If `xformers` fails to install, simply skip it — the pipeline detects it at
runtime.

## 6. Verify

```bash
python -m sd_compress info
# expected fields include:
#   'device': 'cuda'
#   'linux_first': True
#   'runtime_profile': {...}
#   'resolved_cpu_offload': True/False   # auto based on VRAM
```

### Recommended Linux CUDA profile

Defaults already favour Linux + NVIDIA. On a ≥8 GB card you typically want the
model resident on GPU (faster than CPU offload):

```bash
export CPU_OFFLOAD=auto          # full GPU when VRAM >= LOW_VRAM_GB (8)
export ENABLE_TF32=1
export USE_XFORMERS=1
export USE_TORCH_COMPILE=1
export USE_TOME=1
export USE_AMP=1
export AMP_DTYPE=auto            # bf16 on Ampere+; fp16 fallback on older GPUs
```

On a 6 GB card force offload:

```bash
export CPU_OFFLOAD=on
# or lower the threshold:
export LOW_VRAM_GB=10
```

## 7. Run

```bash
./run.sh                            # full pipeline + Gradio server on :8080
./run.sh --no-serve                 # pipeline only
./run.sh baseline                   # single stage
./run.sh distill-progressive
./run.sh evaluate --stage distilled --model-dir ./output/distilled
```

`./run.sh` on Linux with a working `nvidia-smi` installs the CUDA PyTorch wheel
first (override the index with `TORCH_CUDA_INDEX` if needed).

## 8. (Optional) Install as a package

If you prefer the global CLI:

```bash
pip install -e .[onnx,dev]
sd-compress info
```

## Troubleshooting

- **CUDA out of memory.** Reduce `EVAL_SAMPLES`, `STEPS` or run individual
  stages instead of the full pipeline. Force offload with `CPU_OFFLOAD=on`, or
  set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`.
- **`No CUDA GPUs are available` despite a working `nvidia-smi`.** You probably
  installed a CPU-only PyTorch first; reinstall with the correct
  `--index-url` from step 4 (or delete `venv/` and re-run `./run.sh`).
- **`torch.compile` fails with a Triton error.** Triton is bundled with the
  GPU PyTorch wheels. Either reinstall PyTorch or disable compile with
  `USE_TORCH_COMPILE=0` (or `TORCHINDUCTOR_DISABLE=1`).
- **Cannot connect to Hugging Face.** Set `HF_HUB_OFFLINE=1` if you have a
  local mirror, or pass `--base-model /path/to/local/checkpoint`.
