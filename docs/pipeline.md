# Pipeline reference

The compression pipeline is a sequence of independent stages. Each stage reads
its inputs from a directory on disk and writes its outputs to a new directory,
so any stage can be resumed, replayed or swapped out without touching the
others.

```
base model
   |
   v
[stage 0] baseline references         -> output/eval/baseline
   |
   v
[stage 1] progressive distillation    -> output/distilled
   |              (also: CLIP, CFG distillation)
   |---> [eval]                       -> output/eval/distilled
   v
[stage 2] structured pruning          -> output/pruned
   |---> [eval]                       -> output/eval/pruned
   v
[stage 3] fine-tuning recovery        -> output/finetuned
   |---> [eval]                       -> output/eval/finetuned
   v
[stage 4] quantisation                -> output/quant
   |---> [eval]                       -> output/eval/quantized
   v
[stage 5] runtime helpers             -> output/quant/{tome_utils,compile_utils}.py
[stage 6] ONNX export + sharding      -> output/export/{onnx,sharded}
[stage 7] LoRA compatibility check    -> output/lora_compatibility_report.json
[stage 8] Gradio server               -> http://localhost:8080
```

## Stage details

### 0. Baseline
`python -m sd_compress baseline`

Generates the reference images used by every downstream evaluation, at 50
inference steps with deterministic seeds. The metrics file written to
`output/eval/baseline/metrics.json` records the prompts, generated paths,
CLIP scores and inference times.

### 1. Progressive distillation
`python -m sd_compress distill-progressive`

Iteratively halves the inference budget (defaults: 50 -> 25 -> 12 -> 6) while
matching:

- the student's noise prediction to the (previous) teacher's,
- the first five attention-map activations.

An EMA copy of the student is maintained throughout (`EMA_DECAY=0.9999`); the
EMA weights are what get saved at the end of training.

### 1a. CLIP text encoder distillation
`python -m sd_compress distill-clip`

Distils the text encoder by matching the last hidden state, the pooled
embedding, and every third intermediate hidden state. Output: a new
`text_encoder/` and `tokenizer/` inside `output/distilled/`.

### 1b. CFG distillation
`python -m sd_compress distill-cfg`

Bakes classifier-free guidance into a single forward pass. The teacher
computes the standard 2-pass CFG output; the student is trained to predict that
output from a single conditional pass. The result is a model that does not
require negative prompts to behave like a CFG-guided model.

### 2. Structured pruning
`python -m sd_compress prune` (or `--component {text-encoder,vae,unet}`)

L1 channel/feature pruning with per-component skip lists for shape-sensitive
layers (e.g. `conv_in`/`conv_out`, time-embedding projections, residual
shortcuts). Defaults:

- UNet: 30 % channels
- VAE: 20 % channels
- Text encoder: 25 % of MLP `fc1`/`fc2` features

### 3. Fine-tuning recovery
`python -m sd_compress finetune`

A short ε-prediction fine-tune (default 200 steps) to recover the quality lost
to pruning. The text encoder is frozen during this stage; gradients flow only
through the UNet.

### 4. Quantisation
`python -m sd_compress quantize`

Two artefacts are produced:

- `output/quant/` — a complete diffusers pipeline saved in **FP16**.
- `output/quant/unet_int8.pt` — a dynamic INT8 quantised UNet state-dict (best
  suited for CPU x86 deployment).

### 5. Runtime optimisations
`python -m sd_compress optimize`

Drops two helper modules next to the quantised model so downstream users do
not need to depend on the full pipeline package:

- `tome_utils.py` — bipartite soft matching + token merging helpers.
- `compile_utils.py` — `torch.compile` wrapper + warm-up routine.

### 6. ONNX export + sharding
`python -m sd_compress export` (`--target {all,onnx,shard}`)

Exports the UNet and VAE decoder to ONNX (opset 14) for use with TensorRT or
ONNX Runtime, and splits the UNet state-dict into safetensors shards of
configurable size (`SHARD_SIZE_MB`, default 500).

### 7. LoRA compatibility
`python -m sd_compress lora-test`

Walks every `to_q`/`to_k`/`to_v`/`to_out.0` attention projection in the
compressed UNet and reports which shapes are large enough to host typical LoRA
adapters.

### 8. Server
`python -m sd_compress serve`

Loads the quantised pipeline through the Linux-first runtime profile
(`sd_compress.runtime.prepare_pipeline_for_inference`): TF32 + cuDNN benchmark,
attention/VAE slicing, xFormers (SDPA fallback), VRAM-aware CPU offload vs full
GPU residency, Token Merging, and `torch.compile`. Launches a Gradio UI with
single-image, batch and model-info tabs that report which optimisations are
active.

## Evaluation

`python -m sd_compress evaluate --stage <name> --model-dir <path>` runs after
every weight-changing stage. It re-generates the same prompts at
`EVAL_INFERENCE_STEPS` (default 6) and computes CLIP / LPIPS / PSNR / SSIM
against the baseline references. Results land in
`output/eval/<stage>/metrics.json`.

`python -m sd_compress report` aggregates every stage's metrics into
`output/eval/full_report.json`.
