"""
Surveillance Degradation Robustness Benchmark.

Evaluates SISR models on Set14 under realistic CCTV/surveillance degradations:
1. Clean (standard bicubic)
2. Gaussian Sensor Noise (sigma=10)
3. Linear Motion Blur (k=5, angle=45)
4. Heavy JPEG Compression (Q=40)
5. Combined Surveillance Pipeline (Blur + Noise + JPEG)

Outputs full results to results/surveillance_robustness_results.json and CSV.
"""

import sys
import json
import time
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.espcn import ESPCN
from models.fsrcnn import FSRCNN
from models.edsr import EDSR
from metrics.psnr_ssim import compute_psnr_ssim
from data.degradation import (
    downscale_bicubic,
    gaussian_noise,
    motion_blur,
    jpeg_compress,
    degrade
)

MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def modcrop_np(img_np: np.ndarray, scale: int) -> np.ndarray:
    h, w = img_np.shape[:2]
    return img_np[:h - (h % scale), :w - (w % scale)]


def load_model(model_name: str, scale: int, ckpt_path: Path):
    if model_name == "espcn":
        model = ESPCN(scale=scale, mid_channels=32).to(DEVICE)
    elif model_name == "fsrcnn":
        model = FSRCNN(scale=scale).to(DEVICE)
    elif model_name == "edsr":
        model = EDSR(scale=scale, n_feats=64, n_resblocks=16).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    sd = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def run_eval(lr_np: np.ndarray, hr_np: np.ndarray, model, model_name: str, scale: int):
    hr_t = torch.from_numpy(hr_np).permute(2, 0, 1).float() / 255.0

    if model_name == "bicubic":
        h_hr, w_hr = hr_np.shape[:2]
        bic = cv2.resize(lr_np, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        pred_t = torch.from_numpy(bic).permute(2, 0, 1).float() / 255.0
    else:
        inp_t = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        with torch.no_grad():
            out_t = model(inp_t).squeeze(0).cpu().clamp(0.0, 1.0)
        min_h = min(out_t.shape[1], hr_t.shape[1])
        min_w = min(out_t.shape[2], hr_t.shape[2])
        pred_t = out_t[:, :min_h, :min_w]
        hr_t = hr_t[:, :min_h, :min_w]

    p, s = compute_psnr_ssim(pred_t, hr_t, scale=scale, y_channel=True)
    return p, s


def main():
    print("=" * 80)
    print("SURVEILLANCE DEGRADATION ROBUSTNESS BENCHMARK (Set14)")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    set14_entries = [e for e in manifest["entries"] if e["dataset"] == "Set14"]

    models_config = [
        {"id": "Bicubic_2x", "name": "bicubic", "scale": 2, "ckpt": None},
        {"id": "E9_ESPCN_2x", "name": "espcn", "scale": 2, "ckpt": PROJECT_ROOT / "outputs/checkpoints/phase3/E9_espcn_2x_best.pt"},
        {"id": "E10_FSRCNN_2x", "name": "fsrcnn", "scale": 2, "ckpt": PROJECT_ROOT / "outputs/checkpoints/phase3/E10_fsrcnn_2x_best.pt"},
        {"id": "E11_EDSR_2x", "name": "edsr", "scale": 2, "ckpt": PROJECT_ROOT / "outputs/checkpoints/phase3/E11_edsr_2x_best.pt"},
        {"id": "Bicubic_4x", "name": "bicubic", "scale": 4, "ckpt": None},
        {"id": "E12_ESPCN_4x", "name": "espcn", "scale": 4, "ckpt": PROJECT_ROOT / "outputs/checkpoints/phase3/E12_espcn_4x_best.pt"},
        {"id": "E13_FSRCNN_4x", "name": "fsrcnn", "scale": 4, "ckpt": PROJECT_ROOT / "outputs/checkpoints/phase3/E13_fsrcnn_4x_best.pt"},
        {"id": "E14_EDSR_4x", "name": "edsr", "scale": 4, "ckpt": PROJECT_ROOT / "outputs/checkpoints/phase3/E14_edsr_4x_best.pt"},
    ]

    loaded_models = {}
    for m in models_config:
        if m["ckpt"] is not None:
            loaded_models[m["id"]] = load_model(m["name"], m["scale"], m["ckpt"])

    degradation_regimes = {
        "Clean": lambda hr, s: downscale_bicubic(hr, s),
        "Sensor_Noise": lambda hr, s: gaussian_noise(downscale_bicubic(hr, s), sigma=10.0, seed=42),
        "Motion_Blur": lambda hr, s: downscale_bicubic(motion_blur(hr, size=5, angle=45.0), s),
        "JPEG_Q40": lambda hr, s: jpeg_compress(downscale_bicubic(hr, s), quality=40),
        "Combined_CCTV": lambda hr, s: degrade(hr, scale=s, blur_type="motion", blur_ksize=5, motion_angle=30.0,
                                               noise_sigma=8.0, jpeg_quality=50, seed=42)
    }

    all_results = {}

    for regime_name, deg_fn in degradation_regimes.items():
        print(f"\n--- Degradation Regime: {regime_name} ---")
        all_results[regime_name] = {}

        for m in models_config:
            m_id = m["id"]
            m_name = m["name"]
            scale = m["scale"]
            model_obj = loaded_models.get(m_id)

            psnrs, ssims = [], []
            for e in set14_entries:
                hr_img = Image.open(PROJECT_ROOT / e["hr_path"]).convert("RGB")
                hr_np = modcrop_np(np.array(hr_img), scale)
                lr_np = deg_fn(hr_np, scale)

                p, s = run_eval(lr_np, hr_np, model_obj, m_name, scale)
                psnrs.append(p)
                ssims.append(s)

            mean_p = float(np.mean(psnrs))
            mean_s = float(np.mean(ssims))
            all_results[regime_name][m_id] = {
                "psnr": round(mean_p, 2),
                "ssim": round(mean_s, 4),
                "scale": scale
            }
            print(f"  {m_id:15s} ({scale}x) -> PSNR: {mean_p:6.2f} dB | SSIM: {mean_s:6.4f}")

    # Export results
    out_json = PROJECT_ROOT / "results" / "surveillance_robustness_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    out_csv = PROJECT_ROOT / "results" / "surveillance_robustness_summary.csv"
    with open(out_csv, "w") as f:
        f.write("Regime,Model,Scale,PSNR,SSIM\n")
        for reg, models in all_results.items():
            for m_id, vals in models.items():
                f.write(f"{reg},{m_id},{vals['scale']},{vals['psnr']},{vals['ssim']}\n")

    print("\n" + "=" * 80)
    print(f"[OK] Saved surveillance robustness results to {out_json} and {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
