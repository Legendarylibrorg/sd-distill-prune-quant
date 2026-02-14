#!/usr/bin/env python3
"""
Quality Evaluation Module for SD Compression Pipeline

Metrics:
- CLIP Score: Text-image alignment
- LPIPS: Perceptual similarity to reference
- PSNR/SSIM: Pixel-level similarity
- FID: Distribution similarity (requires many samples)
- Inference Speed: Time per image
- Memory Usage: VRAM consumption
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Lazy imports for optional dependencies
_clip_model = None
_clip_preprocess = None
_lpips_model = None


@dataclass
class QualityMetrics:
    """Container for quality metrics."""
    clip_score: float = 0.0
    lpips: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    mse: float = 0.0
    inference_time_ms: float = 0.0
    vram_mb: float = 0.0
    model_size_mb: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"CLIP Score: {self.clip_score:.4f} | "
            f"LPIPS: {self.lpips:.4f} | "
            f"PSNR: {self.psnr:.2f} dB | "
            f"SSIM: {self.ssim:.4f} | "
            f"Time: {self.inference_time_ms:.1f}ms | "
            f"VRAM: {self.vram_mb:.0f}MB"
        )


def get_clip_model(device="cuda"):
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import clip
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
            _clip_model.eval()
        except ImportError:
            print("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            return None, None
    return _clip_model, _clip_preprocess


def get_lpips_model(device="cuda"):
    """Lazy load LPIPS model."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net='alex').to(device)
            _lpips_model.eval()
        except ImportError:
            print("LPIPS not installed. Install with: pip install lpips")
            return None
    return _lpips_model


def compute_clip_score(image: Image.Image, prompt: str, device="cuda") -> float:
    """Compute CLIP score between image and text prompt."""
    clip_model, clip_preprocess = get_clip_model(device)
    if clip_model is None:
        return 0.0
    
    try:
        import clip
        
        # Preprocess image
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize text
        text_input = clip.tokenize([prompt], truncate=True).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (image_features @ text_features.T).item()
        
        return similarity
    except Exception as e:
        print(f"CLIP score computation failed: {e}")
        return 0.0


def compute_lpips(image1: Image.Image, image2: Image.Image, device="cuda") -> float:
    """Compute LPIPS perceptual distance between two images."""
    lpips_model = get_lpips_model(device)
    if lpips_model is None:
        return 0.0
    
    try:
        # Convert to tensors
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img1_tensor = transform(image1).unsqueeze(0).to(device)
        img2_tensor = transform(image2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            distance = lpips_model(img1_tensor, img2_tensor).item()
        
        return distance
    except Exception as e:
        print(f"LPIPS computation failed: {e}")
        return 0.0


def compute_psnr(image1: Image.Image, image2: Image.Image) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    img1 = np.array(image1.resize((512, 512))).astype(np.float32)
    img2 = np.array(image2.resize((512, 512))).astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(image1: Image.Image, image2: Image.Image) -> float:
    """Compute Structural Similarity Index."""
    img1 = np.array(image1.resize((512, 512)).convert('L')).astype(np.float32)
    img2 = np.array(image2.resize((512, 512)).convert('L')).astype(np.float32)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim


def compute_mse_latent(latent1: torch.Tensor, latent2: torch.Tensor) -> float:
    """Compute MSE between latent representations."""
    return F.mse_loss(latent1, latent2).item()


def get_vram_usage() -> float:
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_model_size(model_path: str) -> float:
    """Get model size in MB."""
    total_size = 0
    path = Path(model_path)
    
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    
    for f in path.rglob("*"):
        if f.is_file() and f.suffix in ['.safetensors', '.bin', '.pt', '.pth']:
            total_size += f.stat().st_size
    
    return total_size / 1024 / 1024


class QualityEvaluator:
    """Comprehensive quality evaluator for diffusion models."""
    
    def __init__(
        self,
        reference_pipe=None,
        device: str = "cuda",
        eval_prompts: Optional[List[str]] = None,
        num_eval_samples: int = 8,
        seed: int = 42,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.reference_pipe = reference_pipe
        self.num_eval_samples = num_eval_samples
        self.seed = seed
        
        # Default evaluation prompts covering different scenarios
        self.eval_prompts = eval_prompts or [
            "a photo of a cat sitting on a couch, high quality",
            "a beautiful sunset over the ocean, vibrant colors",
            "portrait of a person smiling, professional photography",
            "a modern city skyline at night, cinematic",
            "a forest path in autumn with fallen leaves",
            "an astronaut riding a horse on mars, digital art",
            "a cozy cabin in snowy woods, warm lighting",
            "a bouquet of colorful flowers, studio photography",
        ]
        
        self.results_history = []
    
    def generate_reference_images(
        self,
        pipe,
        output_dir: str,
        num_inference_steps: int = 50,
    ) -> List[Tuple[str, Image.Image]]:
        """Generate reference images with the baseline model."""
        os.makedirs(output_dir, exist_ok=True)
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        references = []
        for i, prompt in enumerate(tqdm(self.eval_prompts[:self.num_eval_samples], desc="Generating references")):
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
            
            image_path = os.path.join(output_dir, f"reference_{i:03d}.png")
            image.save(image_path)
            references.append((prompt, image))
        
        return references
    
    def evaluate_model(
        self,
        pipe,
        model_name: str,
        model_path: str,
        references: List[Tuple[str, Image.Image]],
        num_inference_steps: int = 4,
        output_dir: Optional[str] = None,
    ) -> Dict[str, QualityMetrics]:
        """
        Evaluate a model against reference images.
        
        Returns:
            Dict with per-sample metrics and aggregated metrics
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        all_metrics = []
        
        # Clear VRAM cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        for i, (prompt, ref_image) in enumerate(tqdm(references, desc=f"Evaluating {model_name}")):
            # Measure inference time
            start_time = time.time()
            
            with torch.inference_mode():
                gen_image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(device=self.device).manual_seed(self.seed + i),
                ).images[0]
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Save generated image
            if output_dir:
                gen_image.save(os.path.join(output_dir, f"generated_{i:03d}.png"))
            
            # Compute metrics
            metrics = QualityMetrics(
                clip_score=compute_clip_score(gen_image, prompt, self.device),
                lpips=compute_lpips(gen_image, ref_image, self.device),
                psnr=compute_psnr(gen_image, ref_image),
                ssim=compute_ssim(gen_image, ref_image),
                inference_time_ms=inference_time,
                vram_mb=get_vram_usage(),
            )
            
            all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = QualityMetrics(
            clip_score=np.mean([m.clip_score for m in all_metrics]),
            lpips=np.mean([m.lpips for m in all_metrics]),
            psnr=np.mean([m.psnr for m in all_metrics]),
            ssim=np.mean([m.ssim for m in all_metrics]),
            inference_time_ms=np.mean([m.inference_time_ms for m in all_metrics]),
            vram_mb=max([m.vram_mb for m in all_metrics]),
            model_size_mb=get_model_size(model_path),
        )
        
        result = {
            "model_name": model_name,
            "model_path": model_path,
            "num_inference_steps": num_inference_steps,
            "num_samples": len(references),
            "average": avg_metrics.to_dict(),
            "per_sample": [m.to_dict() for m in all_metrics],
        }
        
        self.results_history.append(result)
        
        return result
    
    def compare_models(
        self,
        models: List[Tuple[str, any, str, int]],  # (name, pipe, path, steps)
        references: List[Tuple[str, Image.Image]],
        output_dir: str = "./output/eval",
    ) -> Dict:
        """Compare multiple models against references."""
        os.makedirs(output_dir, exist_ok=True)
        
        comparisons = []
        
        for model_name, pipe, model_path, num_steps in models:
            model_output_dir = os.path.join(output_dir, model_name.replace(" ", "_").lower())
            result = self.evaluate_model(
                pipe, model_name, model_path, references, num_steps, model_output_dir
            )
            comparisons.append(result)
            
            # Print summary
            avg = result["average"]
            print(f"\n{model_name} ({num_steps} steps):")
            print(f"  CLIP: {avg['clip_score']:.4f} | LPIPS: {avg['lpips']:.4f} | "
                  f"PSNR: {avg['psnr']:.2f} | SSIM: {avg['ssim']:.4f}")
            print(f"  Time: {avg['inference_time_ms']:.1f}ms | "
                  f"VRAM: {avg['vram_mb']:.0f}MB | Size: {avg['model_size_mb']:.0f}MB")
        
        # Save comparison report
        report = {
            "comparison": comparisons,
            "quality_retention": self._compute_quality_retention(comparisons),
        }
        
        with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate comparison table
        self._print_comparison_table(comparisons)
        
        return report
    
    def _compute_quality_retention(self, comparisons: List[Dict]) -> Dict:
        """Compute quality retention relative to first (baseline) model."""
        if len(comparisons) < 2:
            return {}
        
        baseline = comparisons[0]["average"]
        retention = {}
        
        for comp in comparisons[1:]:
            name = comp["model_name"]
            avg = comp["average"]
            
            retention[name] = {
                "clip_score_retention": (avg["clip_score"] / baseline["clip_score"] * 100) if baseline["clip_score"] > 0 else 0,
                "lpips_change": avg["lpips"] - baseline["lpips"],  # Lower is better
                "psnr_change": avg["psnr"] - baseline["psnr"],  # Higher is better
                "ssim_retention": (avg["ssim"] / baseline["ssim"] * 100) if baseline["ssim"] > 0 else 0,
                "speedup": baseline["inference_time_ms"] / avg["inference_time_ms"] if avg["inference_time_ms"] > 0 else 0,
                "size_reduction": (1 - avg["model_size_mb"] / baseline["model_size_mb"]) * 100 if baseline["model_size_mb"] > 0 else 0,
            }
        
        return retention
    
    def _print_comparison_table(self, comparisons: List[Dict]):
        """Print a formatted comparison table."""
        print("\n" + "=" * 100)
        print("QUALITY COMPARISON TABLE")
        print("=" * 100)
        
        headers = ["Model", "Steps", "CLIP↑", "LPIPS↓", "PSNR↑", "SSIM↑", "Time(ms)", "VRAM(MB)", "Size(MB)"]
        row_format = "{:<20} {:>6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}"
        
        print(row_format.format(*headers))
        print("-" * 100)
        
        for comp in comparisons:
            avg = comp["average"]
            row = [
                comp["model_name"][:20],
                comp["num_inference_steps"],
                f"{avg['clip_score']:.4f}",
                f"{avg['lpips']:.4f}",
                f"{avg['psnr']:.2f}",
                f"{avg['ssim']:.4f}",
                f"{avg['inference_time_ms']:.1f}",
                f"{avg['vram_mb']:.0f}",
                f"{avg['model_size_mb']:.0f}",
            ]
            print(row_format.format(*row))
        
        print("=" * 100)
        
        # Quality retention
        if len(comparisons) >= 2:
            baseline_name = comparisons[0]["model_name"]
            print(f"\nQuality retention relative to {baseline_name}:")
            
            for comp in comparisons[1:]:
                baseline = comparisons[0]["average"]
                avg = comp["average"]
                
                clip_ret = (avg["clip_score"] / baseline["clip_score"] * 100) if baseline["clip_score"] > 0 else 0
                ssim_ret = (avg["ssim"] / baseline["ssim"] * 100) if baseline["ssim"] > 0 else 0
                speedup = baseline["inference_time_ms"] / avg["inference_time_ms"] if avg["inference_time_ms"] > 0 else 0
                
                print(f"  {comp['model_name']}: "
                      f"CLIP {clip_ret:.1f}% | SSIM {ssim_ret:.1f}% | "
                      f"{speedup:.1f}x faster")
    
    def save_history(self, output_path: str):
        """Save evaluation history to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.results_history, f, indent=2)


class QualityGuard:
    """
    Quality guard to ensure compression doesn't degrade quality too much.
    Stops training or raises warning if quality drops below threshold.
    """
    
    def __init__(
        self,
        min_clip_retention: float = 0.90,  # 90% of baseline CLIP score
        max_lpips_increase: float = 0.15,   # Max LPIPS increase from baseline
        min_ssim_retention: float = 0.85,   # 85% of baseline SSIM
        min_psnr: float = 15.0,             # Minimum PSNR in dB
    ):
        self.min_clip_retention = min_clip_retention
        self.max_lpips_increase = max_lpips_increase
        self.min_ssim_retention = min_ssim_retention
        self.min_psnr = min_psnr
        
        self.baseline_metrics = None
        self.violations = []
    
    def set_baseline(self, metrics: QualityMetrics):
        """Set baseline metrics from reference model."""
        self.baseline_metrics = metrics
        print(f"Quality baseline set: {metrics}")
    
    def check(self, metrics: QualityMetrics, stage_name: str) -> Tuple[bool, List[str]]:
        """
        Check if quality metrics are within acceptable bounds.
        
        Returns:
            (passed, list of violations)
        """
        violations = []
        
        if self.baseline_metrics is None:
            return True, []
        
        # CLIP score check
        clip_retention = metrics.clip_score / self.baseline_metrics.clip_score if self.baseline_metrics.clip_score > 0 else 0
        if clip_retention < self.min_clip_retention:
            violations.append(
                f"CLIP retention {clip_retention:.1%} below threshold {self.min_clip_retention:.1%}"
            )
        
        # LPIPS check
        lpips_increase = metrics.lpips - self.baseline_metrics.lpips
        if lpips_increase > self.max_lpips_increase:
            violations.append(
                f"LPIPS increased by {lpips_increase:.4f}, max allowed {self.max_lpips_increase:.4f}"
            )
        
        # SSIM check
        ssim_retention = metrics.ssim / self.baseline_metrics.ssim if self.baseline_metrics.ssim > 0 else 0
        if ssim_retention < self.min_ssim_retention:
            violations.append(
                f"SSIM retention {ssim_retention:.1%} below threshold {self.min_ssim_retention:.1%}"
            )
        
        # PSNR check
        if metrics.psnr < self.min_psnr:
            violations.append(
                f"PSNR {metrics.psnr:.2f} dB below minimum {self.min_psnr:.2f} dB"
            )
        
        passed = len(violations) == 0
        
        if not passed:
            self.violations.append({
                "stage": stage_name,
                "metrics": metrics.to_dict(),
                "violations": violations,
            })
            print(f"\n⚠️  QUALITY WARNING at {stage_name}:")
            for v in violations:
                print(f"   - {v}")
        else:
            print(f"\n✓ Quality check passed at {stage_name}")
            print(f"  CLIP: {clip_retention:.1%} | LPIPS: +{lpips_increase:.4f} | SSIM: {ssim_retention:.1%}")
        
        return passed, violations
    
    def get_report(self) -> Dict:
        """Get quality guard report."""
        return {
            "baseline": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "thresholds": {
                "min_clip_retention": self.min_clip_retention,
                "max_lpips_increase": self.max_lpips_increase,
                "min_ssim_retention": self.min_ssim_retention,
                "min_psnr": self.min_psnr,
            },
            "violations": self.violations,
            "all_passed": len(self.violations) == 0,
        }


def quick_eval(
    pipe,
    prompts: List[str],
    reference_images: Optional[List[Image.Image]] = None,
    num_inference_steps: int = 4,
    device: str = "cuda",
) -> QualityMetrics:
    """
    Quick evaluation for use during training.
    
    Args:
        pipe: Diffusion pipeline
        prompts: List of prompts to evaluate
        reference_images: Optional reference images for comparison
        num_inference_steps: Number of inference steps
        device: Device to use
    
    Returns:
        Aggregated QualityMetrics
    """
    clip_scores = []
    lpips_scores = []
    psnr_scores = []
    ssim_scores = []
    times = []
    
    generator = torch.Generator(device=device).manual_seed(42)
    
    for i, prompt in enumerate(prompts):
        start = time.time()
        
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(42 + i),
            ).images[0]
        
        times.append((time.time() - start) * 1000)
        clip_scores.append(compute_clip_score(image, prompt, device))
        
        if reference_images and i < len(reference_images):
            lpips_scores.append(compute_lpips(image, reference_images[i], device))
            psnr_scores.append(compute_psnr(image, reference_images[i]))
            ssim_scores.append(compute_ssim(image, reference_images[i]))
    
    return QualityMetrics(
        clip_score=np.mean(clip_scores) if clip_scores else 0,
        lpips=np.mean(lpips_scores) if lpips_scores else 0,
        psnr=np.mean(psnr_scores) if psnr_scores else 0,
        ssim=np.mean(ssim_scores) if ssim_scores else 0,
        inference_time_ms=np.mean(times) if times else 0,
        vram_mb=get_vram_usage(),
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SD model quality")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--baseline", type=str, help="Path to baseline model")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--output", type=str, default="./output/eval", help="Output directory")
    
    args = parser.parse_args()
    
    from diffusers import StableDiffusionPipeline
    
    # Load models
    print(f"Loading model from {args.model}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    
    evaluator = QualityEvaluator(num_eval_samples=4)
    
    if args.baseline:
        print(f"Loading baseline from {args.baseline}...")
        baseline_pipe = StableDiffusionPipeline.from_pretrained(
            args.baseline,
            torch_dtype=torch.float16,
        )
        baseline_pipe.enable_model_cpu_offload()
        
        # Generate references
        refs = evaluator.generate_reference_images(
            baseline_pipe,
            os.path.join(args.output, "references"),
            num_inference_steps=50,
        )
        
        # Compare
        evaluator.compare_models(
            [
                ("Baseline (50 steps)", baseline_pipe, args.baseline, 50),
                ("Compressed", pipe, args.model, args.steps),
            ],
            refs,
            args.output,
        )
    else:
        # Just evaluate single model
        refs = evaluator.generate_reference_images(
            pipe,
            os.path.join(args.output, "references"),
            num_inference_steps=50,
        )
        
        result = evaluator.evaluate_model(
            pipe, "Model", args.model, refs, args.steps, args.output
        )
        print(f"\nResults: {result['average']}")
