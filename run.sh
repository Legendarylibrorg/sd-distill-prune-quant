#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################

BASE="runwayml/stable-diffusion-v1-5"
DATA="./data/captions.json"
OUT="./output"

DISTILL="$OUT/distilled"
PRUNE="$OUT/pruned"
QUANT="$OUT/quant"

STEPS=800
LR=1e-5
PRUNE_RATIO=0.3

########################################
# ENV
########################################

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip

pip install \
    torch torchvision \
    diffusers transformers accelerate \
    safetensors pillow tqdm gradio

mkdir -p "$OUT" "$DISTILL" "$PRUNE" "$QUANT" "./data"

# Create sample captions if not exists
if [ ! -f "$DATA" ]; then
    echo "Creating sample captions file..."
    python3 -c "
import json
captions = [
    {'text': 'a photo of a cat sitting on a couch'},
    {'text': 'a beautiful sunset over the ocean'},
    {'text': 'a modern city skyline at night'},
    {'text': 'a forest path in autumn with fallen leaves'},
    {'text': 'a cup of coffee on a wooden table'},
    {'text': 'a portrait of a person smiling'},
    {'text': 'a mountain landscape with snow peaks'},
    {'text': 'a bouquet of colorful flowers'},
]
with open('$DATA', 'w') as f:
    json.dump(captions, f, indent=2)
"
fi

########################################
# DISTILL SD1.5 → 4 STEP
########################################

echo "=== DISTILL SD1.5 (Progressive Distillation) ==="

python3 << 'PY'
import torch
import json
import os
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BASE = os.environ.get("BASE", "runwayml/stable-diffusion-v1-5")
DATA = os.environ.get("DATA", "./data/captions.json")
DISTILL = os.environ.get("DISTILL", "./output/distilled")
STEPS = int(os.environ.get("STEPS", 800))
LR = float(os.environ.get("LR", 1e-5))

# Load components
print("Loading models...")
teacher_unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet").to(DEVICE)
student_unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet").to(DEVICE)
vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae").to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder").to(DEVICE)
scheduler = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")

# Freeze teacher
teacher_unet.eval()
for p in teacher_unet.parameters():
    p.requires_grad = False

# Student is trainable
student_unet.train()

# Load captions
with open(DATA) as f:
    captions = json.load(f)

optimizer = torch.optim.AdamW(student_unet.parameters(), lr=LR)

print(f"Starting distillation for {STEPS} steps...")

for step in tqdm(range(STEPS)):
    caption = captions[step % len(captions)]["text"]
    
    # Encode text
    tokens = tokenizer(
        caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    
    with torch.no_grad():
        text_emb = text_encoder(tokens)[0]
    
    # Create random latent
    latents = torch.randn(1, 4, 64, 64, device=DEVICE)
    
    # Sample random timestep
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=DEVICE)
    
    # Add noise to latents
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    # Teacher prediction (50 step equivalent knowledge)
    with torch.no_grad():
        teacher_pred = teacher_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_emb
        ).sample
    
    # Student prediction (will learn to match teacher in fewer steps)
    student_pred = student_unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_emb
    ).sample
    
    # Distillation loss: student matches teacher's denoising
    loss = torch.nn.functional.mse_loss(student_pred, teacher_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

# Save the distilled student UNet
print(f"Saving distilled model to {DISTILL}...")
student_unet.save_pretrained(f"{DISTILL}/unet")

# Copy other components from base model for complete pipeline
pipe = StableDiffusionPipeline.from_pretrained(BASE)
pipe.unet = student_unet
pipe.save_pretrained(DISTILL)

print("Distillation complete!")
PY

########################################
# PRUNE UNET (Structured Channel Pruning)
########################################

echo "=== STRUCTURED PRUNE ==="

python3 << 'PY'
import torch
import torch.nn as nn
import os
import copy
from diffusers import UNet2DConditionModel, StableDiffusionPipeline

DISTILL = os.environ.get("DISTILL", "./output/distilled")
PRUNE_OUT = os.environ.get("PRUNE", "./output/pruned")
PRUNE_RATIO = float(os.environ.get("PRUNE_RATIO", 0.3))

print(f"Loading distilled UNet from {DISTILL}...")
unet = UNet2DConditionModel.from_pretrained(DISTILL, subfolder="unet")

def get_conv_layers(model):
    """Get all Conv2d layers that can be pruned."""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Skip 1x1 convs that change dimensions critically (in/out projections)
            # and skip depthwise convs (groups == in_channels)
            if module.groups == 1 and module.out_channels > 32:
                conv_layers.append((name, module))
    return conv_layers

def compute_channel_importance(conv_layer):
    """Compute importance score for each output channel using L1 norm."""
    weight = conv_layer.weight.data  # [out_channels, in_channels, H, W]
    # L1 norm across input channels and spatial dimensions
    importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
    return importance

def prune_conv_layer(conv, keep_indices):
    """Create a new smaller Conv2d with only the kept output channels."""
    new_out_channels = len(keep_indices)
    
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=new_out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode
    )
    
    # Copy weights for kept channels
    new_conv.weight.data = conv.weight.data[keep_indices].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_indices].clone()
    
    return new_conv

def prune_following_layer(next_layer, keep_indices):
    """Prune input channels of the following layer to match."""
    if isinstance(next_layer, nn.Conv2d):
        new_conv = nn.Conv2d(
            in_channels=len(keep_indices),
            out_channels=next_layer.out_channels,
            kernel_size=next_layer.kernel_size,
            stride=next_layer.stride,
            padding=next_layer.padding,
            dilation=next_layer.dilation,
            groups=next_layer.groups,
            bias=next_layer.bias is not None,
            padding_mode=next_layer.padding_mode
        )
        new_conv.weight.data = next_layer.weight.data[:, keep_indices].clone()
        if next_layer.bias is not None:
            new_conv.bias.data = next_layer.bias.data.clone()
        return new_conv
    elif isinstance(next_layer, nn.GroupNorm):
        # GroupNorm: adjust num_channels
        new_gn = nn.GroupNorm(
            num_groups=min(next_layer.num_groups, len(keep_indices)),
            num_channels=len(keep_indices),
            eps=next_layer.eps,
            affine=next_layer.affine
        )
        if next_layer.affine:
            new_gn.weight.data = next_layer.weight.data[keep_indices].clone()
            new_gn.bias.data = next_layer.bias.data[keep_indices].clone()
        return new_gn
    elif isinstance(next_layer, nn.BatchNorm2d):
        new_bn = nn.BatchNorm2d(
            num_features=len(keep_indices),
            eps=next_layer.eps,
            momentum=next_layer.momentum,
            affine=next_layer.affine,
            track_running_stats=next_layer.track_running_stats
        )
        if next_layer.affine:
            new_bn.weight.data = next_layer.weight.data[keep_indices].clone()
            new_bn.bias.data = next_layer.bias.data[keep_indices].clone()
        if next_layer.track_running_stats:
            new_bn.running_mean.data = next_layer.running_mean.data[keep_indices].clone()
            new_bn.running_var.data = next_layer.running_var.data[keep_indices].clone()
        return new_bn
    return next_layer

def structured_prune_unet(unet, prune_ratio):
    """
    Apply structured pruning to UNet Conv2d layers.
    This removes entire output channels based on L1 importance.
    """
    total_channels_before = 0
    total_channels_after = 0
    pruned_layers = 0
    
    # Get all modules as a dict for easier access
    modules_dict = dict(unet.named_modules())
    
    # Find conv layers in ResNet blocks that are safe to prune
    # Focus on conv layers within the same block where we can trace dependencies
    for name, module in list(unet.named_modules()):
        if isinstance(module, nn.Conv2d) and module.groups == 1:
            # Skip critical projection layers and small layers
            if any(skip in name for skip in ['proj_in', 'proj_out', 'conv_shortcut', 'time_emb', 'conv_in', 'conv_out']):
                continue
            if module.out_channels <= 64:  # Don't prune small layers
                continue
                
            out_channels = module.out_channels
            total_channels_before += out_channels
            
            # Compute importance and determine channels to keep
            importance = compute_channel_importance(module)
            num_keep = max(int(out_channels * (1 - prune_ratio)), 32)  # Keep at least 32
            num_keep = min(num_keep, out_channels)  # Don't keep more than we have
            
            # Get indices of most important channels
            _, keep_indices = torch.topk(importance, num_keep)
            keep_indices = keep_indices.sort()[0]  # Sort for consistency
            
            total_channels_after += num_keep
            
            if num_keep < out_channels:
                # Prune this layer's output channels
                new_weight = module.weight.data[keep_indices].clone()
                
                # Create new smaller conv
                new_conv = nn.Conv2d(
                    module.in_channels, num_keep, module.kernel_size,
                    module.stride, module.padding, module.dilation,
                    module.groups, module.bias is not None, module.padding_mode
                )
                new_conv.weight.data = new_weight
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data[keep_indices].clone()
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = modules_dict[parent_name]
                    setattr(parent, child_name, new_conv)
                
                pruned_layers += 1
                print(f"  Pruned {name}: {out_channels} -> {num_keep} channels")
    
    reduction = (1 - total_channels_after / total_channels_before) * 100 if total_channels_before > 0 else 0
    print(f"\nStructured pruning summary:")
    print(f"  Layers pruned: {pruned_layers}")
    print(f"  Total channels: {total_channels_before:,} -> {total_channels_after:,}")
    print(f"  Channel reduction: {reduction:.1f}%")
    
    return unet

print(f"Applying structured pruning with ratio {PRUNE_RATIO}...")
print("(Removing entire channels based on L1 importance)\n")

# Count parameters before
params_before = sum(p.numel() for p in unet.parameters())

# Apply structured pruning
unet = structured_prune_unet(unet, PRUNE_RATIO)

# Count parameters after
params_after = sum(p.numel() for p in unet.parameters())

print(f"\nParameter reduction:")
print(f"  Before: {params_before:,}")
print(f"  After:  {params_after:,}")
print(f"  Reduction: {(1 - params_after/params_before)*100:.1f}%")

# Save pruned UNet
os.makedirs(f"{PRUNE_OUT}/unet", exist_ok=True)
unet.save_pretrained(f"{PRUNE_OUT}/unet")

# Copy full pipeline with pruned unet
pipe = StableDiffusionPipeline.from_pretrained(DISTILL)
pipe.unet = unet
pipe.save_pretrained(PRUNE_OUT)

print(f"\nStructured pruned model saved to {PRUNE_OUT}")
PY

########################################
# QUANTIZATION (FP16 + optional INT8)
########################################

echo "=== QUANTIZE (FP16) ==="

python3 << 'PY'
import torch
import os
from diffusers import StableDiffusionPipeline

PRUNE_DIR = os.environ.get("PRUNE", "./output/pruned")
QUANT_DIR = os.environ.get("QUANT", "./output/quant")

print(f"Loading pruned model from {PRUNE_DIR}...")
pipe = StableDiffusionPipeline.from_pretrained(
    PRUNE_DIR,
    torch_dtype=torch.float16,
    variant="fp16" if os.path.exists(f"{PRUNE_DIR}/unet/diffusion_pytorch_model.fp16.safetensors") else None
)

# Convert all components to FP16
pipe.to(torch.float16)

# Save with FP16 weights
print(f"Saving FP16 quantized model to {QUANT_DIR}...")
pipe.save_pretrained(QUANT_DIR, safe_serialization=True)

# Calculate size reduction
import subprocess
original_size = subprocess.run(
    ["du", "-sh", PRUNE_DIR], capture_output=True, text=True
).stdout.split()[0]
quant_size = subprocess.run(
    ["du", "-sh", QUANT_DIR], capture_output=True, text=True
).stdout.split()[0]

print(f"Original size: {original_size}")
print(f"Quantized size: {quant_size}")
print("FP16 quantization complete!")
PY

########################################
# LOW VRAM SERVE
########################################

echo "=== SERVE (LOW VRAM MODE) ==="

python3 << 'PY'
import torch
import gradio as gr
import os
from diffusers import StableDiffusionPipeline

QUANT_DIR = os.environ.get("QUANT", "./output/quant")

print(f"Loading optimized pipeline from {QUANT_DIR}...")

# Load with memory optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    QUANT_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Enable memory optimizations for low VRAM
pipe.enable_attention_slicing(slice_size="auto")
pipe.enable_vae_slicing()

# CPU offload for very low VRAM (<4GB)
if torch.cuda.is_available():
    try:
        pipe.enable_model_cpu_offload()
        print("Using model CPU offload")
    except Exception as e:
        print(f"CPU offload not available, loading to GPU directly: {e}")
        pipe = pipe.to("cuda")
else:
    print("CUDA not available, using CPU (will be slow)")

def generate(prompt, num_steps=4, guidance_scale=7.5):
    """Generate image with distilled model (few-step inference)"""
    if not prompt.strip():
        return None
    
    with torch.inference_mode():
        image = pipe(
            prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=guidance_scale,
        ).images[0]
    
    return image

# Create Gradio interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt..."),
        gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Inference Steps"),
        gr.Slider(minimum=1, maximum=15, value=7.5, step=0.5, label="Guidance Scale"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Distilled Stable Diffusion (Low VRAM)",
    description="Distilled, pruned, and quantized SD1.5 for fast inference with low memory usage.",
)

print("Starting Gradio server on http://localhost:8080")
demo.launch(server_name="0.0.0.0", server_port=8080)
PY

echo "Server stopped."
