# macOS setup

The pipeline runs end-to-end on macOS using either the **Metal Performance
Shaders (MPS)** backend (Apple Silicon) or CPU (Intel). CUDA-only features are
auto-skipped.

> macOS is a useful platform for development and small-scale validation runs;
> for production-quality distillation use a Linux machine with a CUDA GPU.

## 1. System packages

```bash
xcode-select --install        # if not already installed
brew install python git       # Python 3.11+ recommended
```

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

## 4. Install dependencies

The standard PyPI PyTorch wheels include the MPS backend; no special index is
needed.

```bash
pip install -r requirements.txt
pip install "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"   # optional
```

Do **not** install `xformers` — it does not ship for macOS.

## 5. Verify

```bash
python -m sd_compress info
# expected: 'device': 'mps' on Apple Silicon, 'device': 'cpu' on Intel
```

## 6. Run

Reduce the training budget — MPS is several times slower than CUDA on
comparable silicon:

```bash
export PROGRESSIVE_STAGES="50,12,6"
export STEPS=100
export CLIP_DISTILL_STEPS=50
export CFG_DISTILL_STEPS=50
export FINETUNE_STEPS=50
export INT8_CALIBRATION_SAMPLES=20
export EVAL_SAMPLES=2

./run.sh --no-serve    # pipeline only
./run.sh serve         # later, launch the UI separately
```

## 7. Notes on quantisation

PyTorch's dynamic INT8 quantisation targets the FBGEMM / QNNPACK x86 backend.
On Apple Silicon the FP16 pipeline at `output/quant/` is the artefact you want
to ship; the saved `unet_int8.pt` still loads correctly but will not deliver
extra speed on an M-series chip.

## 8. Troubleshooting

- **`MPSNDArray... command buffer execution failed`.** Lower `EVAL_SAMPLES`,
  disable `enable_xformers_memory_efficient_attention` (already off on macOS),
  and avoid running other GPU-intensive apps in parallel.
- **`pip install lpips` is slow / fails.** Make sure you are on Python 3.10+
  and that wheels for SciPy can be installed. On older macOS versions you may
  need `brew install openblas`.
- **`torch.compile` is unavailable.** Apple Silicon's MPS backend does not yet
  support `torch.compile`; the helper at `output/quant/compile_utils.py` is
  written anyway, but it falls back to eager mode automatically.
