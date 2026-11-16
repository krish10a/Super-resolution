"""
Surveillance Degradation Robustness Benchmark.
Tests SISR models and Bicubic baseline under realistic surveillance degradation conditions:
1. Clean (Standard Bicubic downsampling)
2. Gaussian Noise (Low illumination sensor noise, sigma=15)
3. Motion/Gaussian Blur (Defocus/camera shake, ksize=5, sigma=1.5)
4. Heavy JPEG Compression (Surveillance NVR bandwidth throttling, Q=30)
5. Compound Surveillance Degradation (Blur + Noise + JPEG + Downscale)

Evaluates on Set14 (standard SISR surveillance benchmark surrogate).
Outputs results to results/surveillance_robustness.json and prints comparative summary table.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN
from metrics.psnr_ssim import compute_psnr_ssim
from data.degradation import gaussian_blur, gaussian_noise, jpeg_compress

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("results")
CHECKPOINT_DIR = Path("outputs/checkpoints")

CONDITIONS = [
    {"name": "Clean", "blur": False, "noise": 0.0, "jpeg": None},
    {"name": "Sensor Noise (sigma=15)", "blur": False, "noise": 15.0, "jpeg": None},
    {"name": "Defocus Blur (k=5, s=1.5)", "blur": True, "noise": 0.0, "jpeg": None},
    {"name": "NVR Compression (JPEG Q=30)", "blur": False, "noise": 0.0, "jpeg": 30},
    {"name": "Compound Surveillance", "blur": True, "noise": 10.0, "jpeg": 40},
]

def apply_condition(lr_np, cond):
    out = lr_np.copy()
    if cond["blur"]:
        out = gaussian_blur(out, ksize=5, sigma=1.5)
    if cond["noise"] > 0:
        out = gaussian_noise(out, sigma=cond["noise"])
    if cond["jpeg"] is not None:
        out = jpeg_compress(out, quality=cond["jpeg"])
    return out

def load_set14(scale=4):
    lr_dir = Path("data/pairs") / f"Set14_LR_x{scale}"
    hr_dir = Path("data/pairs") / "Set14_HR"
    pairs = []
    for lr_p in sorted(lr_dir.glob("*.png")):
        hr_p = hr_dir / lr_p.name
        if hr_p.exists():
            pairs.append((str(lr_p), str(hr_p)))
    return pairs

def load_ckpt(model, path):
    st = torch.load(path, map_location=DEVICE)
    if isinstance(st, dict) and "model_state" in st:
        model.load_state_dict(st["model_state"])
    elif isinstance(st, dict) and "state_dict" in st:
        model.load_state_dict(st["state_dict"])
    else:
        model.load_state_dict(st)
    return model

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scale = 4
    pairs = load_set14(scale=scale)
    
    # Load 4x models
    srcnn_mse = load_ckpt(SRCNN().to(DEVICE), CHECKPOINT_DIR / "srcnn_4x_mse.pt").eval()
    srcnn_vgg = load_ckpt(SRCNN().to(DEVICE), CHECKPOINT_DIR / "srcnn_4x_mse_perceptual.pt").eval()
    espcn_mse = load_ckpt(ESPCN(scale=scale).to(DEVICE), CHECKPOINT_DIR / "espcn_4x_mse.pt").eval()
    espcn_vgg = load_ckpt(ESPCN(scale=scale).to(DEVICE), CHECKPOINT_DIR / "espcn_4x_mse_perceptual.pt").eval()

    results = {}

    print(f"\n================ SURVEILLANCE ROBUSTNESS BENCHMARK (Set14 4x) ================")
    print(f"{'Condition':30s} | {'Bicubic (dB/SSIM)':18s} | {'SRCNN-MSE':15s} | {'SRCNN-VGG':15s} | {'ESPCN-MSE':15s} | {'ESPCN-VGG':15s}")
    print("-" * 115)

    for cond in CONDITIONS:
        cname = cond["name"]
        m_bic = {"psnr": [], "ssim": []}
        m_s_mse = {"psnr": [], "ssim": []}
        m_s_vgg = {"psnr": [], "ssim": []}
        m_e_mse = {"psnr": [], "ssim": []}
        m_e_vgg = {"psnr": [], "ssim": []}

        with torch.no_grad():
            for lr_path, hr_path in pairs:
                hr_img = Image.open(hr_path).convert("RGB")
                lr_img = Image.open(lr_path).convert("RGB")
                lr_np = np.array(lr_img)
                
                # Apply degradation condition to LR image
                deg_lr_np = apply_condition(lr_np, cond)
                deg_lr_img = Image.fromarray(deg_lr_np)

                # HR target tensor
                hr_t = torch.from_numpy(np.array(hr_img, dtype=np.float32) / 255.0).permute(2, 0, 1)

                # 1. Bicubic
                bic_img = deg_lr_img.resize(hr_img.size, Image.BICUBIC)
                bic_t = torch.from_numpy(np.array(bic_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                p, s = compute_psnr_ssim(bic_t, hr_t, scale=scale, y_channel=True)
                m_bic["psnr"].append(p)
                m_bic["ssim"].append(s)

                # 2. SRCNN MSE & VGG
                inp_srcnn = deg_lr_img.resize(hr_img.size, Image.BICUBIC)
                inp_s_t = torch.from_numpy(np.array(inp_srcnn, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                
                out_s_mse = srcnn_mse(inp_s_t).squeeze(0).cpu()
                p, s = compute_psnr_ssim(out_s_mse, hr_t, scale=scale, y_channel=True)
                m_s_mse["psnr"].append(p)
                m_s_mse["ssim"].append(s)

                out_s_vgg = srcnn_vgg(inp_s_t).squeeze(0).cpu()
                p, s = compute_psnr_ssim(out_s_vgg, hr_t, scale=scale, y_channel=True)
                m_s_vgg["psnr"].append(p)
                m_s_vgg["ssim"].append(s)

                # 3. ESPCN MSE & VGG
                inp_e_t = torch.from_numpy(deg_lr_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                
                out_e_mse = espcn_mse(inp_e_t).squeeze(0).cpu()
                min_h = min(out_e_mse.shape[1], hr_t.shape[1])
                min_w = min(out_e_mse.shape[2], hr_t.shape[2])
                p, s = compute_psnr_ssim(out_e_mse[:, :min_h, :min_w], hr_t[:, :min_h, :min_w], scale=scale, y_channel=True)
                m_e_mse["psnr"].append(p)
                m_e_mse["ssim"].append(s)

                out_e_vgg = espcn_vgg(inp_e_t).squeeze(0).cpu()
                min_h = min(out_e_vgg.shape[1], hr_t.shape[1])
                min_w = min(out_e_vgg.shape[2], hr_t.shape[2])
                p, s = compute_psnr_ssim(out_e_vgg[:, :min_h, :min_w], hr_t[:, :min_h, :min_w], scale=scale, y_channel=True)
                m_e_vgg["psnr"].append(p)
                m_e_vgg["ssim"].append(s)

        res_c = {
            "Bicubic": {"psnr": float(np.mean(m_bic["psnr"])), "ssim": float(np.mean(m_bic["ssim"]))},
            "SRCNN_MSE": {"psnr": float(np.mean(m_s_mse["psnr"])), "ssim": float(np.mean(m_s_mse["ssim"]))},
            "SRCNN_VGG": {"psnr": float(np.mean(m_s_vgg["psnr"])), "ssim": float(np.mean(m_s_vgg["ssim"]))},
            "ESPCN_MSE": {"psnr": float(np.mean(m_e_mse["psnr"])), "ssim": float(np.mean(m_e_mse["ssim"]))},
            "ESPCN_VGG": {"psnr": float(np.mean(m_e_vgg["psnr"])), "ssim": float(np.mean(m_e_vgg["ssim"]))},
        }
        results[cname] = res_c

        print(f"{cname:30s} | {res_c['Bicubic']['psnr']:.2f}/{res_c['Bicubic']['ssim']:.3f}     | "
              f"{res_c['SRCNN_MSE']['psnr']:.2f}/{res_c['SRCNN_MSE']['ssim']:.3f} | "
              f"{res_c['SRCNN_VGG']['psnr']:.2f}/{res_c['SRCNN_VGG']['ssim']:.3f} | "
              f"{res_c['ESPCN_MSE']['psnr']:.2f}/{res_c['ESPCN_MSE']['ssim']:.3f} | "
              f"{res_c['ESPCN_VGG']['psnr']:.2f}/{res_c['ESPCN_VGG']['ssim']:.3f}")

    out_file = OUTPUT_DIR / "surveillance_robustness.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRobustness benchmark results saved to {out_file}")

if __name__ == "__main__":
    main()
