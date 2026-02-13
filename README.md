# SD1.5 Distillation, Pruning & Quantization Pipeline

Pipeline for creating a fast, low-VRAM Stable Diffusion 1.5 model through:
1. **Knowledge Distillation** - Train student to match teacher's noise predictions, enabling fewer inference steps
2. **Structured Pruning** - Remove entire channels based on L1 importance, actually reducing model size
3. **FP16 Quantization** - Reduce precision for 2x memory savings

## Quick Start

```bash
./run.sh
```

Or step-by-step:

```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create caption data (or provide your own)
mkdir -p data
# Add your captions.json with format: [{"text": "prompt"}, ...]

# Run pipeline
./run.sh
```

## Configuration

Edit `run.sh` to adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE` | `runwayml/stable-diffusion-v1-5` | Base model to distill |
| `STEPS` | `800` | Distillation training steps |
| `LR` | `1e-5` | Learning rate |
| `PRUNE_RATIO` | `0.3` | Fraction of channels to remove per layer |

## How It Works

### Distillation (Fixed Approach)

The original script tried to backpropagate through `pipe()` calls, which doesn't work because:
- Pipeline calls use `torch.no_grad()` internally
- The sampling loop isn't differentiable

**Correct approach:** Train at the UNet level:
1. Create random latents + noise
2. Get teacher's noise prediction
3. Get student's noise prediction
4. Minimize MSE between predictions
5. Student learns to denoise like teacher

### Structured Pruning

Channel-wise structured pruning that **actually reduces model size**:
- Computes L1 importance score for each output channel (sum of absolute weights)
- Removes the least important `PRUNE_RATIO` fraction of channels entirely
- Adjusts layer dimensions accordingly (smaller weight tensors)
- Skips critical layers (projections, shortcuts, small layers)
- Results in smaller file size and faster inference, not just sparse weights

### Quantization

FP16 conversion:
- Reduces memory by 2x
- Minimal quality loss for inference
- For further compression, consider INT8 quantization with calibration

## Output Structure

```
output/
├── distilled/     # Full pipeline with distilled UNet
├── pruned/        # Pipeline with pruned UNet
└── quant/         # FP16 quantized pipeline (final model)
```

## Low-VRAM Optimizations

The server uses:
- `enable_attention_slicing()` - Compute attention in chunks
- `enable_vae_slicing()` - Decode latents in chunks
- `enable_model_cpu_offload()` - Keep only active module on GPU

## Requirements

- CUDA GPU with 6GB+ VRAM (4GB possible with CPU offload)
- ~20GB disk space for models
- Python 3.10+

## Limitations

1. **Distillation quality**: 800 steps is minimal; production would need 10k-100k steps
2. **Structured pruning**: Aggressive pruning (>30%) may degrade quality; fine-tuning after pruning is recommended for production
3. **Few-step inference**: True few-step models (like LCM, SDXL-Turbo) use specialized training
