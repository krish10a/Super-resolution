"""
Comprehensive Benchmark Evaluator across all 8 trained experiments and Bicubic baselines.
Evaluates on Set5, Set14, BSD100.
Computes: PSNR (Y), SSIM (Y), Inference Latency (ms), and Parameter Count.
Outputs results to results/benchmark_results.json and prints markdown summary table.
"""

import os
import json
import time
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN
from metrics.psnr_ssim import compute_psnr_ssim, rgb_to_y_cbcr

DATASETS = ["Set5", "Set14", "BSD100"]
SCALES = [2, 4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("results")
CHECKPOINT_DIR = Path("outputs/checkpoints")

EXPERIMENTS = [
    {"name": "E1_SRCNN_2x_MSE", "model": "srcnn", "scale": 2, "loss": "mse", "ckpt": "srcnn_2x_mse.pt"},
    {"name": "E2_SRCNN_2x_Perceptual", "model": "srcnn", "scale": 2, "loss": "perceptual", "ckpt": "srcnn_2x_mse_perceptual.pt"},
    {"name": "E3_SRCNN_4x_MSE", "model": "srcnn", "scale": 4, "loss": "mse", "ckpt": "srcnn_4x_mse.pt"},
    {"name": "E4_SRCNN_4x_Perceptual", "model": "srcnn", "scale": 4, "loss": "perceptual", "ckpt": "srcnn_4x_mse_perceptual.pt"},
    {"name": "E5_ESPCN_2x_MSE", "model": "espcn", "scale": 2, "loss": "mse", "ckpt": "espcn_2x_mse.pt"},
    {"name": "E6_ESPCN_2x_Perceptual", "model": "espcn", "scale": 2, "loss": "perceptual", "ckpt": "espcn_2x_mse_perceptual.pt"},
    {"name": "E7_ESPCN_4x_MSE", "model": "espcn", "scale": 4, "loss": "mse", "ckpt": "espcn_4x_mse.pt"},
    {"name": "E8_ESPCN_4x_Perceptual", "model": "espcn", "scale": 4, "loss": "perceptual", "ckpt": "espcn_4x_mse_perceptual.pt"},
]

def load_pairs(dataset: str, scale: int):
    lr_dir = Path("data/pairs") / f"{dataset}_LR_x{scale}"
    hr_dir = Path("data/pairs") / f"{dataset}_HR"
    if not lr_dir.exists() or not hr_dir.exists():
        return []
    lr_files = sorted(list(lr_dir.glob("*.png")))
    pairs = []
    for lr_p in lr_files:
        hr_p = hr_dir / lr_p.name
        if hr_p.exists():
            pairs.append((str(lr_p), str(hr_p)))
    return pairs

def evaluate_bicubic(dataset: str, scale: int):
    pairs = load_pairs(dataset, scale)
    if not pairs:
        return 0.0, 0.0, 0.0
    
    psnrs, ssims, times = [], [], []
    for lr_path, hr_path in pairs:
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        
        t0 = time.perf_counter()
        bic_img = lr_img.resize(hr_img.size, Image.BICUBIC)
        times.append((time.perf_counter() - t0) * 1000.0)
        
        pred_t = torch.from_numpy(np.array(bic_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        hr_t = torch.from_numpy(np.array(hr_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        p, s = compute_psnr_ssim(pred_t, hr_t, scale=scale, y_channel=True)
        psnrs.append(p)
        ssims.append(s)
        
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(times))

def evaluate_model(model, exp_info: dict, dataset: str):
    scale = exp_info["scale"]
    model_type = exp_info["model"]
    pairs = load_pairs(dataset, scale)
    if not pairs:
        return 0.0, 0.0, 0.0
    
    psnrs, ssims, times = [], [], []
    model.eval()
    
    with torch.no_grad():
        for lr_path, hr_path in pairs:
            lr_img = Image.open(lr_path).convert("RGB")
            hr_img = Image.open(hr_path).convert("RGB")
            
            t0 = time.perf_counter()
            if model_type == "srcnn":
                inp_img = lr_img.resize(hr_img.size, Image.BICUBIC)
                inp_arr = np.array(inp_img, dtype=np.float32) / 255.0
            else:
                inp_arr = np.array(lr_img, dtype=np.float32) / 255.0
                
            inp_t = torch.from_numpy(inp_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            out_t = model(inp_t)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latency = (time.perf_counter() - t0) * 1000.0
            times.append(latency)
            
            out_t = out_t.squeeze(0).cpu()
            hr_t = torch.from_numpy(np.array(hr_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            
            # Match spatial dimensions if modulo rounding caused slight difference
            min_h = min(out_t.shape[1], hr_t.shape[1])
            min_w = min(out_t.shape[2], hr_t.shape[2])
            out_t = out_t[:, :min_h, :min_w]
            hr_t = hr_t[:, :min_h, :min_w]
            
            p, s = compute_psnr_ssim(out_t, hr_t, scale=scale, y_channel=True)
            psnrs.append(p)
            ssims.append(s)
            
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(times))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "bicubic": {},
        "experiments": {}
    }
    
    print(f"Running benchmarks on Device: {DEVICE}")
    print("=" * 80)
    
    # 1. Bicubic Baselines
    for scale in SCALES:
        results["bicubic"][f"{scale}x"] = {}
        for ds in DATASETS:
            p, s, t = evaluate_bicubic(ds, scale)
            results["bicubic"][f"{scale}x"][ds] = {"psnr": p, "ssim": s, "latency_ms": t}
            print(f"Bicubic {scale}x on {ds:6s} -> PSNR: {p:.2f} dB, SSIM: {s:.4f}, Latency: {t:.2f} ms")
            
    print("=" * 80)
    
    # 2. Model Experiments
    for exp in EXPERIMENTS:
        ckpt_path = CHECKPOINT_DIR / exp["ckpt"]
        if not ckpt_path.exists():
            print(f"Checkpoint not found for {exp['name']}: {ckpt_path}")
            continue
            
        scale = exp["scale"]
        if exp["model"] == "srcnn":
            model = SRCNN().to(DEVICE)
        else:
            model = ESPCN(scale=scale).to(DEVICE)
            
        state = torch.load(ckpt_path, map_location=DEVICE)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
            
        params = count_parameters(model)
        exp_res = {
            "params": params,
            "scale": scale,
            "model": exp["model"],
            "loss": exp["loss"],
            "metrics": {}
        }
        
        for ds in DATASETS:
            p, s, t = evaluate_model(model, exp, ds)
            exp_res["metrics"][ds] = {"psnr": p, "ssim": s, "latency_ms": t}
            print(f"{exp['name']:25s} on {ds:6s} -> PSNR: {p:.2f} dB, SSIM: {s:.4f}, Latency: {t:.2f} ms (Params: {params:,})")
            
        results["experiments"][exp["name"]] = exp_res
        
    out_file = OUTPUT_DIR / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    main()
