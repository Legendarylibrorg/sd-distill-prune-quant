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
pip install "git+https://github.com/openai/CLIP.git"   # optional
pip install xformers                                   # optional, often speeds attention
```

If `xformers` fails to install, simply skip it — the pipeline detects it at
runtime.

## 6. Verify

```bash
python -m sd_compress info
# expected output contains:  'device': 'cuda'
```

## 7. Run

```bash
./run.sh                            # full pipeline + Gradio server on :8080
./run.sh --no-serve                 # pipeline only
./run.sh baseline                   # single stage
./run.sh distill-progressive
./run.sh evaluate --stage distilled --model-dir ./output/distilled
```

## 8. (Optional) Install as a package

If you prefer the global CLI:

```bash
pip install -e .[onnx,dev]
sd-compress info
```

## Troubleshooting

- **CUDA out of memory.** Reduce `EVAL_SAMPLES`, `STEPS` or run individual
  stages instead of the full pipeline. Enable CPU offload by setting
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`.
- **`No CUDA GPUs are available` despite a working `nvidia-smi`.** You probably
  installed a CPU-only PyTorch first; reinstall with the correct
  `--index-url` from step 4.
- **`torch.compile` fails with a Triton error.** Triton is bundled with the
  GPU PyTorch wheels. Either reinstall PyTorch or unset compile by exporting
  `TORCHINDUCTOR_DISABLE=1`.
- **Cannot connect to Hugging Face.** Set `HF_HUB_OFFLINE=1` if you have a
  local mirror, or pass `--base-model /path/to/local/checkpoint`.
