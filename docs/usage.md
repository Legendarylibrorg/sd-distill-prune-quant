# Usage cookbook

Recipes for the most common workflows. All examples assume the virtualenv is
activated and you're in the repository root.

## Quickest possible smoke test

Use a tiny budget so the whole pipeline finishes in minutes on CPU:

```bash
export STEPS=20
export CLIP_DISTILL_STEPS=10
export CFG_DISTILL_STEPS=10
export FINETUNE_STEPS=10
export INT8_CALIBRATION_SAMPLES=5
export EVAL_SAMPLES=2
export PROGRESSIVE_STAGES="50,6"

python -m sd_compress run --no-serve  # then `serve` separately
```

## Run a single stage

```bash
python -m sd_compress baseline
python -m sd_compress distill-progressive
python -m sd_compress distill-clip
python -m sd_compress distill-cfg
python -m sd_compress prune                    # all components
python -m sd_compress prune --component unet   # UNet only
python -m sd_compress finetune
python -m sd_compress quantize
python -m sd_compress export --target onnx
python -m sd_compress export --target shard
python -m sd_compress lora-test
python -m sd_compress evaluate --stage distilled --model-dir ./output/distilled
python -m sd_compress report
python -m sd_compress serve
```

## Use a custom base model

```bash
python -m sd_compress run \
    --base-model stabilityai/stable-diffusion-2-base \
    --output-dir ./output_sd2
```

> **Compatibility note.** The pruning ratios and attention-distillation layer
> indices were tuned for SD 1.5. Other checkpoints may need lower ratios.

## Provide your own captions

The pipeline expects a JSON file shaped like `[{"text": "..."}, ...]`. Drop one
in `./data/captions.json` (or point `DATA` at any other path) before running:

```bash
DATA=./my_captions.json python -m sd_compress run
```

## Override the inference step budget for evaluation

By default every evaluation uses 6 inference steps. To compare apples-to-apples
with a different student step count:

```bash
EVAL_INFERENCE_STEPS=8 python -m sd_compress evaluate \
    --stage distilled --model-dir ./output/distilled
```

## Skip the Gradio server

```bash
./run.sh --no-serve
# or
python -m sd_compress run
```

## Train on a subset of the captions

The training loops simply cycle through the caption list. Truncate it to
the captions you care about and adjust `STEPS` accordingly:

```bash
jq '.[:8]' data/captions.json > data/captions_subset.json
DATA=./data/captions_subset.json STEPS=200 \
    python -m sd_compress distill-progressive
```

## Use the package programmatically

```python
from sd_compress.config import PipelineConfig
from sd_compress.distillation import progressive_distillation
from sd_compress.eval_runner import evaluate_stage

config = PipelineConfig(steps=100, progressive_stages="50,25,6")
config.ensure_dirs()
progressive_distillation(config)
metrics = evaluate_stage(config, "distilled", config.distill_dir)
print(metrics["clip_retention"])
```

## Inference from the quantised model

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "./output/quant",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = pipe("a cat astronaut on mars", num_inference_steps=4).images[0]
image.save("astronaut.png")
```

## Test LoRA compatibility with your own LoRA

```bash
mkdir -p output/test_loras
cp /path/to/your/lora.safetensors output/test_loras/
python -m sd_compress lora-test
cat output/lora_compatibility_report.json
```

The bundled report only checks *shape* compatibility. To validate behavioural
compatibility, load the LoRA in your inference code and inspect the generated
images.
