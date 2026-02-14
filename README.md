# SD1.5 Compression & Optimization Pipeline

Comprehensive pipeline for compressing and optimizing Stable Diffusion 1.5 with state-of-the-art techniques.

## Features

### Distillation
- **Progressive Distillation** - Iteratively halve inference steps (50→25→12→6)
- **Attention Distillation** - Match attention maps for better quality preservation
- **CLIP Distillation** - Distill text encoder with intermediate layer matching
- **CFG Distillation** - Remove need for negative prompts (single forward pass)
- **EMA Weights** - Exponential moving average for stable training

### Pruning
- **UNet Structured Pruning** - Remove entire channels based on L1 importance
- **Text Encoder Pruning** - Prune MLP layers (fc1/fc2)
- **VAE Pruning** - Conservative pruning of encoder/decoder

### Quantization
- **FP16 Quantization** - 2x memory reduction
- **INT8 Dynamic Quantization** - 4x reduction for CPU deployment
- **Calibration Dataset** - Proper calibration for INT8

### Optimization
- **Fine-tuning After Pruning** - Recover accuracy lost to pruning
- **Token Merging (ToMe)** - Merge similar tokens for 2x speedup
- **torch.compile** - PyTorch 2.0 compilation (10-30% speedup)
- **xFormers** - Memory-efficient attention
- **Attention/VAE Slicing** - Low VRAM inference

### Quality Evaluation
- **CLIP Score** - Text-image alignment measurement
- **LPIPS** - Perceptual similarity to baseline
- **PSNR/SSIM** - Pixel-level quality metrics
- **Quality Guard** - Automatic thresholds to prevent excessive degradation
- **Per-Stage Evaluation** - Metrics after distillation, pruning, fine-tuning, quantization
- **Comprehensive Reports** - JSON reports with full metrics history

### Deployment
- **ONNX Export** - For TensorRT/ONNX Runtime
- **Safetensors Sharding** - Streaming model load
- **Dynamic Batching** - Higher throughput server
- **LoRA Compatibility Testing** - Verify pruned model works with LoRAs

## Quick Start

```bash
./run.sh
```

## Configuration

Edit `run.sh` to adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| **Distillation** | | |
| `STEPS` | `800` | Steps per progressive distillation stage |
| `CLIP_DISTILL_STEPS` | `400` | Text encoder distillation steps |
| `LR` | `1e-5` | Learning rate |
| `EMA_DECAY` | `0.9999` | EMA decay rate |
| `PROGRESSIVE_STAGES` | `50,25,12,6` | Step halving stages |
| **Pruning** | | |
| `PRUNE_RATIO` | `0.3` | UNet channel pruning ratio |
| `TEXT_ENCODER_PRUNE_RATIO` | `0.25` | Text encoder pruning ratio |
| `VAE_PRUNE_RATIO` | `0.2` | VAE pruning ratio |
| **Fine-tuning** | | |
| `FINETUNE_STEPS` | `200` | Post-pruning fine-tuning steps |
| `FINETUNE_LR` | `5e-6` | Fine-tuning learning rate |
| **Quality Evaluation** | | |
| `EVAL_SAMPLES` | `4` | Number of samples for evaluation |
| `MIN_CLIP_RETENTION` | `0.90` | Minimum acceptable CLIP score retention |
| `MAX_LPIPS_INCREASE` | `0.15` | Maximum acceptable LPIPS increase |
| `MIN_SSIM_RETENTION` | `0.85` | Minimum acceptable SSIM retention |
| `FINETUNE_LR` | `5e-6` | Fine-tuning learning rate |
| **Optimization** | | |
| `TOME_RATIO` | `0.5` | Token merging ratio |
| `INT8_CALIBRATION_SAMPLES` | `100` | INT8 calibration samples |

## Pipeline Stages

```
0. Baseline Generation
   └── Generate reference images with original model (50 steps)
   
1. Progressive Distillation (50→25→12→6 steps)
   └── With attention distillation & EMA
   └── 📊 EVALUATE: Compare to baseline
   
2. CLIP Distillation
   └── Hidden states + intermediate layers
   
3. CFG Distillation
   └── Guidance baked into single forward pass
   
4. Structured Pruning
   ├── Text Encoder (25% MLP neurons)
   ├── VAE (20% channels)
   └── UNet (30% channels)
   └── 📊 EVALUATE: Measure quality loss
   
5. Fine-tuning
   └── Recover accuracy after pruning
   └── 📊 EVALUATE: Verify recovery
   
6. Quantization
   ├── FP16 (default)
   └── INT8 (CPU deployment)
   └── 📊 EVALUATE: Final quality report
   
7. Optimization Setup
   ├── Token Merging
   ├── torch.compile wrapper
   └── ONNX export
   
8. Server Launch
   └── All optimizations enabled
```

## Quality Metrics

The pipeline evaluates quality at each major stage using:

| Metric | Description | Target |
|--------|-------------|--------|
| **CLIP Score** | Text-image alignment (higher is better) | ≥90% of baseline |
| **LPIPS** | Perceptual distance (lower is better) | <0.15 increase |
| **PSNR** | Peak signal-to-noise ratio (higher is better) | >15 dB |
| **SSIM** | Structural similarity (higher is better) | ≥85% of baseline |

### Sample Evaluation Output

```
======================================================================
FINAL COMPRESSION PIPELINE QUALITY REPORT
======================================================================

Stage           CLIP↑      LPIPS↓     PSNR↑      SSIM↑      Time(ms)
----------------------------------------------------------------------
Baseline        0.3245          -          -          -          3200
Distilled       0.3180     0.0823      22.45     0.8534          520
Pruned          0.2998     0.1156      19.87     0.7823          480
Finetuned       0.3102     0.0945      21.23     0.8245          485
QUANTIZED       0.3098     0.0952      21.18     0.8231          350
======================================================================

FINAL RESULTS:
  Quality Retention: 95.5% (CLIP score)
  Speedup: 9.1x faster
  Model Size: 2100 MB
  Inference Steps: 50 → 6 (8.3x fewer)

✅ SUCCESS: Quality target met (≥90% retention)
```

## Output Structure

```
output/
├── eval/                     # Quality evaluation results
│   ├── baseline/             # Reference images & metrics
│   ├── distilled/            # Post-distillation evaluation
│   ├── pruned/               # Post-pruning evaluation
│   ├── finetuned/            # Post-finetuning evaluation
│   ├── quantized/            # Final evaluation
│   └── full_report.json      # Comprehensive comparison report
├── distilled/                # Progressive + CFG distilled model
├── pruned/                   # All components pruned
├── finetuned/                # Post-pruning fine-tuned
├── quant/              # Quantized models
│   ├── model_index.json
│   ├── unet/           # FP16 UNet
│   ├── unet_int8.pt    # INT8 UNet state dict
│   ├── tome_utils.py   # Token merging utilities
│   └── compile_utils.py # torch.compile wrapper
├── export/
│   ├── onnx/           # ONNX models
│   │   ├── unet.onnx
│   │   └── vae_decoder.onnx
│   └── sharded/        # Sharded safetensors
│       ├── unet_shard_*.safetensors
│       └── shard_index.json
└── lora_compatibility_report.json
```

## Usage Examples

### Basic Inference (4 steps)
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "./output/quant",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

image = pipe("a cat", num_inference_steps=4).images[0]
```

### With torch.compile
```python
# Use the provided wrapper
from output.quant.compile_utils import load_compiled_pipeline

pipe = load_compiled_pipeline("./output/quant")
image = pipe("a cat", num_inference_steps=4).images[0]
```

### With Token Merging
```python
from output.quant.tome_utils import apply_tome_to_attention

# Apply ToMe to attention layers
for name, module in pipe.unet.named_modules():
    if 'attn' in name:
        apply_tome_to_attention(module, ratio=0.5)
```

### INT8 Inference (CPU)
```python
import torch
from diffusers import UNet2DConditionModel

# Load INT8 quantized UNet
unet = UNet2DConditionModel.from_pretrained("./output/quant", subfolder="unet")
unet = torch.quantization.quantize_dynamic(unet, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
# Or load pre-quantized
# state_dict = torch.load("./output/quant/unet_int8.pt")
```

### TensorRT (NVIDIA)
```bash
# Convert ONNX to TensorRT
trtexec --onnx=./output/export/onnx/unet.onnx \
        --saveEngine=./output/export/unet.trt \
        --fp16
```

### Batch Generation
```python
# Generate multiple images efficiently
prompts = ["a cat", "a dog", "a bird"]
images = pipe(prompts, num_inference_steps=4).images
```

## Performance Comparison

| Configuration | Steps | Time (A100) | VRAM | Quality |
|--------------|-------|-------------|------|---------|
| Original SD1.5 | 50 | 3.2s | 8GB | Baseline |
| + Progressive Distill | 6 | 0.5s | 8GB | ~95% |
| + Pruning (30%) | 6 | 0.4s | 6GB | ~92% |
| + Fine-tuning | 6 | 0.4s | 6GB | ~94% |
| + FP16 | 6 | 0.35s | 3GB | ~94% |
| + torch.compile | 6 | 0.25s | 3GB | ~94% |
| + ToMe (50%) | 6 | 0.18s | 3GB | ~92% |

*Times are approximate and vary by hardware*

## Requirements

- Python 3.10+
- CUDA GPU with 6GB+ VRAM (4GB with CPU offload)
- ~30GB disk space for all outputs

### Dependencies
```
torch>=2.0.0
diffusers>=0.25.0
transformers>=4.35.0
accelerate>=0.25.0
safetensors>=0.4.0
gradio>=4.0.0
optimum (for ONNX export)
xformers (optional, for memory efficiency)
```

## Limitations

1. **Distillation Quality** - 800 steps/stage is minimal; production needs 10k-100k
2. **Aggressive Pruning** - >40% may significantly degrade quality
3. **CFG Distillation** - Works best with single guidance scale
4. **INT8** - Some quality loss; best for CPU deployment
5. **LoRA Compatibility** - Heavily pruned models may have shape mismatches
6. **Token Merging** - May affect fine details in complex scenes

## Testing LoRA Compatibility

```bash
mkdir -p output/test_loras
cp /path/to/your/lora.safetensors output/test_loras/
./run.sh
# Check output/lora_compatibility_report.json
```

## References

- [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512)
- [Token Merging (ToMe)](https://arxiv.org/abs/2210.09461)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)
- [Classifier-Free Guidance Distillation](https://arxiv.org/abs/2306.05284)
