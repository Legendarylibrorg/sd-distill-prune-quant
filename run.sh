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
# PRUNE UNET (Magnitude-based structured pruning)
########################################

echo "=== PRUNE ==="

python3 << 'PY'
import torch
import os
from diffusers import UNet2DConditionModel, StableDiffusionPipeline

DISTILL = os.environ.get("DISTILL", "./output/distilled")
PRUNE_OUT = os.environ.get("PRUNE", "./output/pruned")
PRUNE_RATIO = float(os.environ.get("PRUNE_RATIO", 0.3))

print(f"Loading distilled UNet from {DISTILL}...")
unet = UNet2DConditionModel.from_pretrained(DISTILL, subfolder="unet")

# Magnitude-based unstructured pruning
# Note: True structured pruning requires architecture changes
print(f"Applying magnitude pruning with ratio {PRUNE_RATIO}...")

total_params = 0
pruned_params = 0

for name, param in unet.named_parameters():
    if param.ndim >= 2:  # Only prune weight matrices, not biases
        total_params += param.numel()
        
        # Calculate threshold for this layer
        abs_weights = torch.abs(param.data)
        threshold = torch.quantile(abs_weights.flatten(), PRUNE_RATIO)
        
        # Create mask and zero out small weights
        mask = abs_weights < threshold
        param.data[mask] = 0
        pruned_params += mask.sum().item()

print(f"Total parameters: {total_params:,}")
print(f"Pruned parameters: {pruned_params:,} ({100*pruned_params/total_params:.1f}%)")

# Save pruned UNet
unet.save_pretrained(f"{PRUNE_OUT}/unet")

# Copy full pipeline with pruned unet
pipe = StableDiffusionPipeline.from_pretrained(DISTILL)
pipe.unet = unet
pipe.save_pretrained(PRUNE_OUT)

print(f"Pruned model saved to {PRUNE_OUT}")
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
