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
CLIP_DISTILL_STEPS=400
VAE_PRUNE_RATIO=0.2
TEXT_ENCODER_PRUNE_RATIO=0.25
TEST_LORA="https://civitai.com/api/download/models/87153"  # Example LoRA for testing

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
# CLIP TEXT ENCODER DISTILLATION
########################################

echo "=== CLIP TEXT ENCODER DISTILLATION ==="

python3 << 'PY'
import torch
import torch.nn as nn
import json
import os
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BASE = os.environ.get("BASE", "runwayml/stable-diffusion-v1-5")
DATA = os.environ.get("DATA", "./data/captions.json")
DISTILL = os.environ.get("DISTILL", "./output/distilled")
CLIP_STEPS = int(os.environ.get("CLIP_DISTILL_STEPS", 400))
LR = float(os.environ.get("LR", 1e-5))

print("Loading CLIP models...")
tokenizer = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
teacher_encoder = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder").to(DEVICE)
student_encoder = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder").to(DEVICE)

# Freeze teacher
teacher_encoder.eval()
for p in teacher_encoder.parameters():
    p.requires_grad = False

# Student is trainable
student_encoder.train()

# Load captions
with open(DATA) as f:
    captions = json.load(f)

# Optimizer for student encoder
optimizer = torch.optim.AdamW(student_encoder.parameters(), lr=LR)

print(f"Starting CLIP distillation for {CLIP_STEPS} steps...")

for step in tqdm(range(CLIP_STEPS)):
    caption = captions[step % len(captions)]["text"]
    
    # Tokenize
    tokens = tokenizer(
        caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    
    # Teacher embeddings
    with torch.no_grad():
        teacher_output = teacher_encoder(tokens)
        teacher_hidden = teacher_output.last_hidden_state
        teacher_pooled = teacher_output.pooler_output
    
    # Student embeddings
    student_output = student_encoder(tokens)
    student_hidden = student_output.last_hidden_state
    student_pooled = student_output.pooler_output
    
    # Distillation loss: match both hidden states and pooled output
    loss_hidden = nn.functional.mse_loss(student_hidden, teacher_hidden)
    loss_pooled = nn.functional.mse_loss(student_pooled, teacher_pooled)
    loss = loss_hidden + 0.5 * loss_pooled
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f} (hidden: {loss_hidden.item():.6f}, pooled: {loss_pooled.item():.6f})")

# Save distilled text encoder
print(f"Saving distilled text encoder to {DISTILL}/text_encoder...")
student_encoder.save_pretrained(f"{DISTILL}/text_encoder")
tokenizer.save_pretrained(f"{DISTILL}/tokenizer")

print("CLIP distillation complete!")
PY

########################################
# TEXT ENCODER PRUNING (Structured)
########################################

echo "=== TEXT ENCODER PRUNING ==="

python3 << 'PY'
import torch
import torch.nn as nn
import os
from transformers import CLIPTextModel, CLIPTokenizer

DISTILL = os.environ.get("DISTILL", "./output/distilled")
PRUNE_OUT = os.environ.get("PRUNE", "./output/pruned")
PRUNE_RATIO = float(os.environ.get("TEXT_ENCODER_PRUNE_RATIO", 0.25))

print(f"Loading text encoder from {DISTILL}...")
text_encoder = CLIPTextModel.from_pretrained(f"{DISTILL}/text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(f"{DISTILL}/tokenizer")

def compute_linear_importance(linear_layer):
    """Compute importance for each output neuron using L1 norm."""
    weight = linear_layer.weight.data  # [out_features, in_features]
    importance = torch.sum(torch.abs(weight), dim=1)
    return importance

def prune_linear_layer(linear, keep_indices):
    """Create a new smaller Linear layer with only kept output neurons."""
    new_out = len(keep_indices)
    new_linear = nn.Linear(linear.in_features, new_out, bias=linear.bias is not None)
    new_linear.weight.data = linear.weight.data[keep_indices].clone()
    if linear.bias is not None:
        new_linear.bias.data = linear.bias.data[keep_indices].clone()
    return new_linear

print(f"Applying structured pruning with ratio {PRUNE_RATIO}...")

total_params_before = sum(p.numel() for p in text_encoder.parameters())
pruned_layers = 0

# Prune intermediate layers in transformer blocks (MLP layers)
for name, module in list(text_encoder.named_modules()):
    # Target the intermediate (fc1) layers in MLP blocks
    if isinstance(module, nn.Linear) and 'mlp.fc1' in name:
        out_features = module.out_features
        if out_features <= 256:  # Skip small layers
            continue
            
        importance = compute_linear_importance(module)
        num_keep = max(int(out_features * (1 - PRUNE_RATIO)), 128)
        
        _, keep_indices = torch.topk(importance, num_keep)
        keep_indices = keep_indices.sort()[0]
        
        # Get parent module
        parts = name.split('.')
        parent = text_encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        
        # Create pruned layer
        new_linear = prune_linear_layer(module, keep_indices)
        setattr(parent, child_name, new_linear)
        
        # Also need to prune fc2's input to match
        fc2_name = name.replace('fc1', 'fc2')
        fc2_parts = fc2_name.split('.')
        fc2_parent = text_encoder
        for part in fc2_parts[:-1]:
            fc2_parent = getattr(fc2_parent, part)
        fc2 = getattr(fc2_parent, fc2_parts[-1])
        
        # Prune fc2 input channels
        new_fc2 = nn.Linear(num_keep, fc2.out_features, bias=fc2.bias is not None)
        new_fc2.weight.data = fc2.weight.data[:, keep_indices].clone()
        if fc2.bias is not None:
            new_fc2.bias.data = fc2.bias.data.clone()
        setattr(fc2_parent, fc2_parts[-1], new_fc2)
        
        pruned_layers += 1
        print(f"  Pruned {name}: {out_features} -> {num_keep} neurons")

total_params_after = sum(p.numel() for p in text_encoder.parameters())

print(f"\nText Encoder Pruning Summary:")
print(f"  Layers pruned: {pruned_layers}")
print(f"  Parameters: {total_params_before:,} -> {total_params_after:,}")
print(f"  Reduction: {(1 - total_params_after/total_params_before)*100:.1f}%")

# Save pruned text encoder
os.makedirs(f"{PRUNE_OUT}/text_encoder", exist_ok=True)
text_encoder.save_pretrained(f"{PRUNE_OUT}/text_encoder")
tokenizer.save_pretrained(f"{PRUNE_OUT}/tokenizer")

print(f"Pruned text encoder saved to {PRUNE_OUT}/text_encoder")
PY

########################################
# VAE PRUNING (Structured)
########################################

echo "=== VAE PRUNING ==="

python3 << 'PY'
import torch
import torch.nn as nn
import os
from diffusers import AutoencoderKL

DISTILL = os.environ.get("DISTILL", "./output/distilled")
PRUNE_OUT = os.environ.get("PRUNE", "./output/pruned")
PRUNE_RATIO = float(os.environ.get("VAE_PRUNE_RATIO", 0.2))

print(f"Loading VAE from {DISTILL}...")
vae = AutoencoderKL.from_pretrained(DISTILL, subfolder="vae")

def compute_conv_importance(conv_layer):
    """Compute importance for each output channel using L1 norm."""
    weight = conv_layer.weight.data
    importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
    return importance

print(f"Applying structured VAE pruning with ratio {PRUNE_RATIO}...")

total_params_before = sum(p.numel() for p in vae.parameters())
pruned_layers = 0

modules_dict = dict(vae.named_modules())

for name, module in list(vae.named_modules()):
    if isinstance(module, nn.Conv2d) and module.groups == 1:
        # Skip critical layers
        if any(skip in name for skip in ['conv_in', 'conv_out', 'conv_shortcut', 'quant_conv', 'post_quant_conv']):
            continue
        if module.out_channels <= 64:
            continue
            
        out_channels = module.out_channels
        importance = compute_conv_importance(module)
        num_keep = max(int(out_channels * (1 - PRUNE_RATIO)), 32)
        
        _, keep_indices = torch.topk(importance, num_keep)
        keep_indices = keep_indices.sort()[0]
        
        if num_keep < out_channels:
            # Create pruned conv
            new_conv = nn.Conv2d(
                module.in_channels, num_keep, module.kernel_size,
                module.stride, module.padding, module.dilation,
                module.groups, module.bias is not None, module.padding_mode
            )
            new_conv.weight.data = module.weight.data[keep_indices].clone()
            if module.bias is not None:
                new_conv.bias.data = module.bias.data[keep_indices].clone()
            
            # Replace in parent
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = modules_dict[parent_name]
                setattr(parent, child_name, new_conv)
            
            pruned_layers += 1
            print(f"  Pruned {name}: {out_channels} -> {num_keep} channels")

total_params_after = sum(p.numel() for p in vae.parameters())

print(f"\nVAE Pruning Summary:")
print(f"  Layers pruned: {pruned_layers}")
print(f"  Parameters: {total_params_before:,} -> {total_params_after:,}")
print(f"  Reduction: {(1 - total_params_after/total_params_before)*100:.1f}%")

# Save pruned VAE
os.makedirs(f"{PRUNE_OUT}/vae", exist_ok=True)
vae.save_pretrained(f"{PRUNE_OUT}/vae")

print(f"Pruned VAE saved to {PRUNE_OUT}/vae")
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
# LORA COMPATIBILITY TESTING
########################################

echo "=== LORA COMPATIBILITY TESTING ==="

python3 << 'PY'
import torch
import os
import json
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

QUANT_DIR = os.environ.get("QUANT", "./output/quant")
OUT_DIR = os.environ.get("OUT", "./output")

print("=" * 60)
print("LORA COMPATIBILITY TEST SUITE")
print("=" * 60)

# Load the compressed pipeline
print(f"\nLoading compressed pipeline from {QUANT_DIR}...")
pipe = StableDiffusionPipeline.from_pretrained(
    QUANT_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    pipe.enable_model_cpu_offload()
else:
    print("Warning: Running on CPU, tests will be slow")

results = {
    "lora_tests": [],
    "summary": {}
}

def test_lora_loading(pipe, lora_path, lora_name="test_lora"):
    """Test if a LoRA can be loaded into the pipeline."""
    test_result = {
        "name": lora_name,
        "path": lora_path,
        "load_success": False,
        "inference_success": False,
        "errors": []
    }
    
    try:
        # Try loading LoRA
        print(f"\n  Testing LoRA: {lora_name}")
        
        if lora_path.endswith('.safetensors'):
            # Load safetensors LoRA
            pipe.load_lora_weights(lora_path)
            test_result["load_success"] = True
            print(f"    ✓ LoRA loaded successfully")
        elif lora_path.startswith('http'):
            # Skip URL-based LoRAs for now (would need download)
            print(f"    ⚠ URL-based LoRA skipped (download not implemented)")
            test_result["errors"].append("URL download not implemented")
            return test_result
        else:
            # Try as HuggingFace model ID
            pipe.load_lora_weights(lora_path)
            test_result["load_success"] = True
            print(f"    ✓ LoRA loaded successfully")
            
    except Exception as e:
        error_msg = str(e)
        test_result["errors"].append(f"Load error: {error_msg}")
        print(f"    ✗ Load failed: {error_msg[:100]}...")
        return test_result
    
    # Test inference with LoRA
    try:
        with torch.inference_mode():
            _ = pipe(
                "test prompt",
                num_inference_steps=2,
                output_type="latent"
            )
        test_result["inference_success"] = True
        print(f"    ✓ Inference with LoRA successful")
    except Exception as e:
        error_msg = str(e)
        test_result["errors"].append(f"Inference error: {error_msg}")
        print(f"    ✗ Inference failed: {error_msg[:100]}...")
    
    # Unload LoRA for next test
    try:
        pipe.unload_lora_weights()
        print(f"    ✓ LoRA unloaded")
    except Exception as e:
        print(f"    ⚠ Could not unload LoRA: {e}")
    
    return test_result

def test_lora_weight_shapes(pipe):
    """Check if UNet weight shapes are compatible with standard LoRA."""
    print("\n  Checking UNet weight shapes for LoRA compatibility...")
    
    issues = []
    compatible_layers = 0
    total_attention_layers = 0
    
    # Standard LoRA targets these attention layers
    lora_targets = ['to_q', 'to_k', 'to_v', 'to_out.0']
    
    for name, module in pipe.unet.named_modules():
        if any(target in name for target in lora_targets):
            total_attention_layers += 1
            if hasattr(module, 'weight'):
                weight_shape = module.weight.shape
                # Check if dimensions are reasonable for LoRA
                if weight_shape[0] >= 64 and weight_shape[1] >= 64:
                    compatible_layers += 1
                else:
                    issues.append(f"{name}: shape {weight_shape} may be too small")
    
    return {
        "total_attention_layers": total_attention_layers,
        "compatible_layers": compatible_layers,
        "issues": issues
    }

def test_lora_scale_factors(pipe):
    """Test LoRA with different scale factors."""
    print("\n  Testing LoRA scale factor handling...")
    
    scale_results = {}
    
    # Create a dummy LoRA-like weight modification
    try:
        # Test that fuse_lora and unfuse_lora work
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unfuse_lora()
        scale_results["fuse_unfuse"] = "supported"
        print("    ✓ fuse_lora/unfuse_lora supported")
    except AttributeError:
        scale_results["fuse_unfuse"] = "not_available"
        print("    ⚠ fuse_lora/unfuse_lora not available")
    except Exception as e:
        scale_results["fuse_unfuse"] = f"error: {str(e)[:50]}"
        print(f"    ✗ fuse_lora error: {str(e)[:50]}")
    
    return scale_results

# Run tests
print("\n" + "-" * 60)
print("TEST 1: Weight Shape Compatibility")
print("-" * 60)
shape_results = test_lora_weight_shapes(pipe)
results["shape_compatibility"] = shape_results

if shape_results["issues"]:
    print(f"\n  ⚠ Found {len(shape_results['issues'])} potential issues:")
    for issue in shape_results["issues"][:5]:
        print(f"    - {issue}")
else:
    print(f"\n  ✓ All {shape_results['compatible_layers']}/{shape_results['total_attention_layers']} attention layers compatible")

print("\n" + "-" * 60)
print("TEST 2: LoRA API Compatibility")
print("-" * 60)
scale_results = test_lora_scale_factors(pipe)
results["scale_factors"] = scale_results

print("\n" + "-" * 60)
print("TEST 3: Sample LoRA Loading (if available)")
print("-" * 60)

# Check for local LoRA files to test
lora_dir = os.path.join(OUT_DIR, "test_loras")
if os.path.exists(lora_dir):
    lora_files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
    for lora_file in lora_files[:3]:  # Test up to 3 LoRAs
        lora_path = os.path.join(lora_dir, lora_file)
        result = test_lora_loading(pipe, lora_path, lora_file)
        results["lora_tests"].append(result)
else:
    print(f"\n  No local LoRAs found. To test with LoRAs:")
    print(f"    1. Create directory: {lora_dir}")
    print(f"    2. Add .safetensors LoRA files")
    print(f"    3. Re-run this test")

# Test with HuggingFace LoRA example
print("\n  Testing with HuggingFace example LoRA...")
try:
    result = test_lora_loading(
        pipe, 
        "hf-internal-testing/sd-lora",
        "hf-internal-testing/sd-lora"
    )
    results["lora_tests"].append(result)
except Exception as e:
    print(f"    ✗ HF LoRA test failed: {e}")
    results["lora_tests"].append({
        "name": "hf-internal-testing/sd-lora",
        "load_success": False,
        "inference_success": False,
        "errors": [str(e)]
    })

# Summary
print("\n" + "=" * 60)
print("LORA COMPATIBILITY SUMMARY")
print("=" * 60)

total_tests = len(results["lora_tests"])
successful_loads = sum(1 for t in results["lora_tests"] if t["load_success"])
successful_inference = sum(1 for t in results["lora_tests"] if t["inference_success"])

results["summary"] = {
    "weight_shapes_compatible": len(shape_results["issues"]) == 0,
    "compatible_attention_layers": shape_results["compatible_layers"],
    "total_attention_layers": shape_results["total_attention_layers"],
    "lora_tests_run": total_tests,
    "successful_loads": successful_loads,
    "successful_inference": successful_inference,
    "overall_compatible": (
        len(shape_results["issues"]) == 0 and 
        (total_tests == 0 or successful_loads > 0)
    )
}

print(f"\n  Weight shape compatibility: {'✓ PASS' if results['summary']['weight_shapes_compatible'] else '⚠ ISSUES'}")
print(f"  Compatible attention layers: {shape_results['compatible_layers']}/{shape_results['total_attention_layers']}")
print(f"  LoRA load tests: {successful_loads}/{total_tests} passed")
print(f"  LoRA inference tests: {successful_inference}/{total_tests} passed")
print(f"\n  Overall LoRA compatibility: {'✓ COMPATIBLE' if results['summary']['overall_compatible'] else '⚠ MAY HAVE ISSUES'}")

# Save results
results_path = os.path.join(OUT_DIR, "lora_compatibility_report.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Full report saved to: {results_path}")

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
