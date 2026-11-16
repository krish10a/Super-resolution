"""
Authoritative benchmark evaluator across canonical SISR datasets and models.

Evaluates against the canonical dataset manifest (data/dataset_manifest.json):
- Set5 (5 images)
- Set14 (14 images)
- BSD100 (100 images)

Supported baselines & models:
- Bicubic (MATLAB-compatible reference)
- E1-E8 legacy checkpoints from outputs/checkpoints/
- E9-E14 Phase 3 DIV2K trained checkpoints from outputs/checkpoints/phase3/

Metrics:
- PSNR (BT.601 Y channel, border shaved by scale)
- SSIM (BT.601 Y channel, border shaved by scale)
- Latency (ms, with CUDA synchronization when on GPU)
- Parameter count
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.srcnn import SRCNN
from models.espcn import ESPCN
from models.fsrcnn import FSRCNN
from models.edsr import EDSR
from metrics.psnr_ssim import compute_psnr_ssim

MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"

MODEL_REGISTRY = {
    # Legacy Phase 1/2 models
    "E1": {"name": "E1_SRCNN_2x_MSE", "model": "srcnn", "scale": 2, "ckpt": "outputs/checkpoints/srcnn_2x_mse.pt", "loss": "mse", "group": "legacy"},
    "E2": {"name": "E2_SRCNN_2x_Perceptual", "model": "srcnn", "scale": 2, "ckpt": "outputs/checkpoints/srcnn_2x_mse_perceptual.pt", "loss": "mse_perceptual", "group": "legacy"},
    "E3": {"name": "E3_SRCNN_4x_MSE", "model": "srcnn", "scale": 4, "ckpt": "outputs/checkpoints/srcnn_4x_mse.pt", "loss": "mse", "group": "legacy"},
    "E4": {"name": "E4_SRCNN_4x_Perceptual", "model": "srcnn", "scale": 4, "ckpt": "outputs/checkpoints/srcnn_4x_mse_perceptual.pt", "loss": "mse_perceptual", "group": "legacy"},
    "E5": {"name": "E5_ESPCN_2x_MSE", "model": "espcn", "scale": 2, "ckpt": "outputs/checkpoints/espcn_2x_mse.pt", "loss": "mse", "group": "legacy", "mid_channels": 64},
    "E6": {"name": "E6_ESPCN_2x_Perceptual", "model": "espcn", "scale": 2, "ckpt": "outputs/checkpoints/espcn_2x_mse_perceptual.pt", "loss": "mse_perceptual", "group": "legacy", "mid_channels": 64},
    "E7": {"name": "E7_ESPCN_4x_MSE", "model": "espcn", "scale": 4, "ckpt": "outputs/checkpoints/espcn_4x_mse.pt", "loss": "mse", "group": "legacy", "mid_channels": 64},
    "E8": {"name": "E8_ESPCN_4x_Perceptual", "model": "espcn", "scale": 4, "ckpt": "outputs/checkpoints/espcn_4x_mse_perceptual.pt", "loss": "mse_perceptual", "group": "legacy", "mid_channels": 64},

    # Phase 3 DIV2K models
    "E9": {"name": "E9_ESPCN_2x_Charbonnier", "model": "espcn", "scale": 2, "ckpt": "outputs/checkpoints/phase3/E9_espcn_2x_best.pt", "loss": "charbonnier", "group": "phase3", "mid_channels": 32},
    "E10": {"name": "E10_FSRCNN_2x_Charbonnier", "model": "fsrcnn", "scale": 2, "ckpt": "outputs/checkpoints/phase3/E10_fsrcnn_2x_best.pt", "loss": "charbonnier", "group": "phase3"},
    "E11": {"name": "E11_EDSR_2x_L1", "model": "edsr", "scale": 2, "ckpt": "outputs/checkpoints/phase3/E11_edsr_2x_best.pt", "loss": "l1", "group": "phase3", "n_feats": 64, "n_resblocks": 16},
    "E12": {"name": "E12_ESPCN_4x_Charbonnier", "model": "espcn", "scale": 4, "ckpt": "outputs/checkpoints/phase3/E12_espcn_4x_best.pt", "loss": "charbonnier", "group": "phase3", "mid_channels": 32},
    "E13": {"name": "E13_FSRCNN_4x_Charbonnier", "model": "fsrcnn", "scale": 4, "ckpt": "outputs/checkpoints/phase3/E13_fsrcnn_4x_best.pt", "loss": "charbonnier", "group": "phase3"},
    "E14": {"name": "E14_EDSR_4x_L1", "model": "edsr", "scale": 4, "ckpt": "outputs/checkpoints/phase3/E14_edsr_4x_best.pt", "loss": "l1", "group": "phase3", "n_feats": 64, "n_resblocks": 16},
}


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}. Run scripts/prepare_canonical_pairs.py first.")
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def evaluate_bicubic_image(lr_img: Image.Image, hr_img: Image.Image, scale: int):
    t0 = time.perf_counter()
    bic_img = lr_img.resize(hr_img.size, Image.BICUBIC)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_t = torch.from_numpy(np.array(bic_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    hr_t = torch.from_numpy(np.array(hr_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    p, s = compute_psnr_ssim(pred_t, hr_t, scale=scale, y_channel=True)
    return p, s, latency_ms


def evaluate_model_image(model, model_type: str, lr_img: Image.Image, hr_img: Image.Image, scale: int, device: torch.device):
    t0 = time.perf_counter()
    if model_type == "srcnn":
        inp_img = lr_img.resize(hr_img.size, Image.BICUBIC)
        inp_arr = np.array(inp_img, dtype=np.float32) / 255.0
    else:
        inp_arr = np.array(lr_img, dtype=np.float32) / 255.0

    inp_t = torch.from_numpy(inp_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_t = model(inp_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        out_t = out_t.squeeze(0).cpu().clamp(0.0, 1.0)
        hr_t = torch.from_numpy(np.array(hr_img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        min_h = min(out_t.shape[1], hr_t.shape[1])
        min_w = min(out_t.shape[2], hr_t.shape[2])
        out_t = out_t[:, :min_h, :min_w]
        hr_t = hr_t[:, :min_h, :min_w]

        p, s = compute_psnr_ssim(out_t, hr_t, scale=scale, y_channel=True)
    return p, s, latency_ms


def instantiate_model(info: dict, scale: int, device: torch.device):
    model_type = info["model"]
    if model_type == "srcnn":
        model = SRCNN().to(device)
    elif model_type == "espcn":
        mid_ch = info.get("mid_channels", 32 if info.get("group") == "phase3" else 64)
        model = ESPCN(scale=scale, mid_channels=mid_ch).to(device)
    elif model_type == "fsrcnn":
        model = FSRCNN(scale=scale).to(device)
    elif model_type == "edsr":
        n_feats = info.get("n_feats", 64)
        n_resblocks = info.get("n_resblocks", 16)
        model = EDSR(scale=scale, n_feats=n_feats, n_resblocks=n_resblocks).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt_path = PROJECT_ROOT / info["ckpt"]
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Canonical SISR Benchmark Evaluator")
    parser.add_argument("--dataset", choices=["Set5", "Set14", "BSD100", "all"], default="all")
    parser.add_argument("--scale", type=int, choices=[2, 4, 0], default=0, help="0 for all scales (2 and 4)")
    parser.add_argument("--model", "--models", dest="model", type=str, default="phase3",
                        help="'bicubic', 'all', 'legacy', 'phase3', or specific ID like 'E1', 'E9'")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export-json", type=str, default="results/canonical_benchmark_results.json")
    parser.add_argument("--export-csv", type=str, default="results/canonical_summary.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = load_manifest()

    datasets = ["Set5", "Set14", "BSD100"] if args.dataset == "all" else [args.dataset]
    scales = [2, 4] if args.scale == 0 else [args.scale]

    print("=" * 80)
    print(f"CANONICAL SISR BENCHMARK EVALUATOR (Device: {device})")
    print(f"Target: {args.model.upper()} | Datasets: {datasets} | Scales: {scales}")
    print("=" * 80)

    # Load existing results if json exists, to append/merge
    export_json = PROJECT_ROOT / args.export_json
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "manifest_version": manifest.get("version"),
            "protocol": manifest.get("protocol"),
        },
        "baselines": {},
        "models": {},
    }
    if export_json.exists():
        try:
            with open(export_json, "r") as f:
                prev_data = json.load(f)
                results["baselines"] = prev_data.get("baselines", {})
                results["models"] = prev_data.get("models", {})
        except Exception:
            pass

    # 1. Evaluate Bicubic Baselines if needed
    for scale in scales:
        if f"{scale}x" not in results["baselines"] or args.model in ["all", "bicubic"]:
            results["baselines"][f"{scale}x"] = {}
            for ds in datasets:
                entries = [e for e in manifest["entries"] if e["dataset"] == ds]
                psnrs, ssims, lats = [], [], []
                for e in entries:
                    hr_img = Image.open(PROJECT_ROOT / e["hr_path"]).convert("RGB")
                    lr_p = e[f"lr_x{scale}_path"]
                    lr_img = Image.open(PROJECT_ROOT / lr_p).convert("RGB")
                    p, s, lat = evaluate_bicubic_image(lr_img, hr_img, scale)
                    psnrs.append(p); ssims.append(s); lats.append(lat)

                mean_p, std_p = float(np.mean(psnrs)), float(np.std(psnrs))
                mean_s, std_s = float(np.mean(ssims)), float(np.std(ssims))
                mean_lat = float(np.mean(lats))

                results["baselines"][f"{scale}x"][ds] = {
                    "psnr_mean": mean_p, "psnr_std": std_p,
                    "ssim_mean": mean_s, "ssim_std": std_s,
                    "latency_ms": mean_lat, "n_images": len(entries),
                }
                print(f"[Bicubic {scale}x] on {ds:6s} ({len(entries):3d} img) -> PSNR: {mean_p:.4f} ± {std_p:.2f} dB | SSIM: {mean_s:.4f} ± {std_s:.4f} | Latency: {mean_lat:.2f} ms")

    if args.model == "bicubic":
        return

    # 2. Select Models
    target = args.model.lower()
    selected_models = []
    if target == "all":
        selected_models = list(MODEL_REGISTRY.keys())
    elif target in ["phase3", "p3"]:
        selected_models = [k for k, v in MODEL_REGISTRY.items() if v.get("group") == "phase3"]
    elif target in ["legacy", "p1", "p2"]:
        selected_models = [k for k, v in MODEL_REGISTRY.items() if v.get("group") == "legacy"]
    elif args.model.upper() in MODEL_REGISTRY:
        selected_models = [args.model.upper()]
    else:
        print(f"Unknown model filter: {args.model}")
        print(f"Available: all, legacy, phase3, or IDs ({', '.join(MODEL_REGISTRY.keys())})")
        return

    # 3. Evaluate Selected Models
    for exp_id in selected_models:
        info = MODEL_REGISTRY[exp_id]
        scale = info["scale"]
        if scale not in scales:
            continue

        ckpt_path = PROJECT_ROOT / info["ckpt"]
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        try:
            model = instantiate_model(info, scale, device)
        except Exception as err:
            print(f"[ERROR] Failed to instantiate {exp_id}: {err}")
            continue

        param_count = sum(p.numel() for p in model.parameters())
        results["models"][exp_id] = {
            "name": info["name"],
            "model_type": info["model"],
            "scale": scale,
            "loss": info["loss"],
            "group": info.get("group", "custom"),
            "params": param_count,
            "checkpoint": info["ckpt"],
            "metrics": {},
        }

        print(f"\nEvaluating {exp_id}: {info['name']} ({scale}x, {param_count:,} params)...")
        for ds in datasets:
            entries = [e for e in manifest["entries"] if e["dataset"] == ds]
            psnrs, ssims, lats = [], [], []
            for e in entries:
                hr_img = Image.open(PROJECT_ROOT / e["hr_path"]).convert("RGB")
                lr_p = e[f"lr_x{scale}_path"]
                lr_img = Image.open(PROJECT_ROOT / lr_p).convert("RGB")
                p, s, lat = evaluate_model_image(model, info["model"], lr_img, hr_img, scale, device)
                psnrs.append(p); ssims.append(s); lats.append(lat)

            mean_p, std_p = float(np.mean(psnrs)), float(np.std(psnrs))
            mean_s, std_s = float(np.mean(ssims)), float(np.std(ssims))
            mean_lat = float(np.mean(lats))

            base_p = results["baselines"].get(f"{scale}x", {}).get(ds, {}).get("psnr_mean", 0.0)
            delta_p = mean_p - base_p if base_p > 0 else 0.0

            results["models"][exp_id]["metrics"][ds] = {
                "psnr_mean": mean_p, "psnr_std": std_p,
                "ssim_mean": mean_s, "ssim_std": std_s,
                "latency_ms": mean_lat, "delta_psnr_vs_bicubic": delta_p,
                "n_images": len(entries),
            }
            print(f"  [{ds:6s} | {len(entries):3d} img] PSNR: {mean_p:.4f} dB ({delta_p:+.2f} dB vs Bicubic) | SSIM: {mean_s:.4f} | Latency: {mean_lat:.2f} ms")

    # Export outputs
    export_json.parent.mkdir(parents=True, exist_ok=True)
    with open(export_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Exported canonical results to {export_json}")

    export_csv = PROJECT_ROOT / args.export_csv
    export_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(export_csv, "w") as f:
        f.write("Exp_ID,Model,Scale,Loss,Group,Dataset,N,PSNR,PSNR_Std,SSIM,SSIM_Std,Latency_ms,Bicubic_PSNR,Delta_PSNR\n")
        for exp_id, data in results["models"].items():
            for ds, m in data["metrics"].items():
                scale = data["scale"]
                base_p = results["baselines"].get(f"{scale}x", {}).get(ds, {}).get("psnr_mean", 0.0)
                f.write(f"{exp_id},{data['model_type']},{scale},{data['loss']},{data.get('group', 'custom')},{ds},{m['n_images']},"
                        f"{m['psnr_mean']:.4f},{m['psnr_std']:.4f},{m['ssim_mean']:.4f},{m['ssim_std']:.4f},"
                        f"{m['latency_ms']:.2f},{base_p:.4f},{m['delta_psnr_vs_bicubic']:+.4f}\n")
    print(f"[OK] Exported canonical summary CSV to {export_csv}")


if __name__ == "__main__":
    main()
