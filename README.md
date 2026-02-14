# SD1.5 Distillation, Pruning & Quantization Pipeline

Comprehensive pipeline for compressing Stable Diffusion 1.5 with full component optimization:

1. **UNet Distillation** - Train student UNet to match teacher's noise predictions
2. **CLIP Distillation** - Distill text encoder for faster prompt encoding
3. **Structured Pruning** - Remove entire channels from UNet, VAE, and text encoder
4. **FP16 Quantization** - Reduce precision for 2x memory savings
5. **LoRA Compatibility Testing** - Verify pruned model works with existing LoRAs

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
| `STEPS` | `800` | UNet distillation training steps |
| `CLIP_DISTILL_STEPS` | `400` | Text encoder distillation steps |
| `LR` | `1e-5` | Learning rate |
| `PRUNE_RATIO` | `0.3` | Fraction of UNet channels to remove |
| `TEXT_ENCODER_PRUNE_RATIO` | `0.25` | Fraction of text encoder neurons to remove |
| `VAE_PRUNE_RATIO` | `0.2` | Fraction of VAE channels to remove |

## How It Works

### UNet Distillation

Trains at the UNet level (not through pipeline calls):
1. Create random latents + noise
2. Get teacher's noise prediction
3. Get student's noise prediction
4. Minimize MSE between predictions
5. Student learns to match teacher's denoising quality

### CLIP Text Encoder Distillation

Distills the CLIP text encoder for faster prompt processing:
1. Feed same prompts to teacher and student encoders
2. Match both hidden states and pooled outputs
3. Combined loss: `MSE(hidden) + 0.5 * MSE(pooled)`
4. Results in a text encoder that produces similar embeddings faster

### Structured Pruning (UNet, VAE, Text Encoder)

Channel-wise structured pruning that **actually reduces model size**:

**UNet Pruning:**
- Computes L1 importance for each output channel
- Removes least important `PRUNE_RATIO` fraction entirely
- Skips critical layers (proj_in/out, shortcuts, conv_in/out)

**Text Encoder Pruning:**
- Targets MLP intermediate layers (fc1/fc2)
- Prunes neurons based on L1 importance
- Maintains fc2 input compatibility with pruned fc1

**VAE Pruning:**
- Prunes encoder and decoder conv layers
- Skips quant_conv/post_quant_conv and shortcuts
- More conservative ratio (default 20%) to preserve reconstruction

### Quantization

FP16 conversion:
- Reduces memory by 2x
- Minimal quality loss for inference
- For further compression, consider INT8 quantization with calibration

### LoRA Compatibility Testing

Automated test suite that verifies:
1. **Weight shape compatibility** - Checks attention layer dimensions
2. **LoRA API support** - Tests fuse/unfuse functionality
3. **Load/inference testing** - Tests actual LoRA files if available
4. **Generates report** - JSON report saved to `output/lora_compatibility_report.json`

## Output Structure

```
output/
├── distilled/                    # Full pipeline with distilled UNet + CLIP
│   ├── unet/
│   ├── text_encoder/
│   ├── vae/
│   └── ...
├── pruned/                       # All components pruned
│   ├── unet/                     # Structurally pruned UNet
│   ├── text_encoder/             # Pruned text encoder
│   ├── vae/                      # Pruned VAE
│   └── ...
├── quant/                        # FP16 quantized (final model)
├── lora_compatibility_report.json # LoRA test results
└── test_loras/                   # (optional) Place LoRAs here for testing
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

## Testing LoRA Compatibility

To test with your own LoRAs:

```bash
mkdir -p output/test_loras
cp /path/to/your/lora.safetensors output/test_loras/
./run.sh
```

The LoRA test will:
- Check if attention layer shapes are compatible
- Attempt to load and run inference with each LoRA
- Generate a detailed JSON report

## Limitations

1. **Distillation quality**: 800/400 steps is minimal; production would need 10k-100k steps
2. **Structured pruning**: Aggressive pruning (>30%) may degrade quality; fine-tuning after pruning recommended
3. **Few-step inference**: True few-step models (like LCM, SDXL-Turbo) use specialized training
4. **LoRA compatibility**: Heavily pruned models may have shape mismatches with some LoRAs
5. **VAE sensitivity**: VAE pruning can affect image quality more than UNet pruning
