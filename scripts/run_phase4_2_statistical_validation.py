"""
Phase 4.2: Statistical Robustness Validation Script.

Evaluates SISR models on Set14 (14 images) and BSD100 (100 images) under 5 degradation regimes:
1. Clean (canonical antialiased bicubic pairs)
2. Sensor Noise (Gaussian sigma=10.0)
3. Motion Blur (ksize=5, angle=45 deg)
4. Heavy JPEG (Quality=40)
5. Combined CCTV (Motion blur ksize=5, angle=30 deg + downscale + noise sigma=8.0 + JPEG Q=50)

Models evaluated:
- 2x: Bicubic, E9 (ESPCN), E10 (FSRCNN), E11 (EDSR), and legacy E1, E2, E5, E6
- 4x: Bicubic, E12 (ESPCN), E13 (FSRCNN), E14 (EDSR), and legacy E3, E4, E7, E8

Statistical rigor:
- Deterministic per-image seeds via SHA-256 (not Python hash)
- Single degraded LR generation per image/scale/condition across all models
- Paired bootstrap 95% confidence intervals (10,000 resamples, seed=42)
- Explicit win/tie/loss rate against Bicubic baseline with numerical tolerance
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
from PIL import Image
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.srcnn import SRCNN
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
RESULTS_DIR = PROJECT_ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
PSNR_TIE_TOL = 1e-4
SSIM_TIE_TOL = 1e-5


def derive_deterministic_seed(img_id: str, scale: int, condition: str) -> int:
    key = f"{img_id}_x{scale}_{condition}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def paired_bootstrap_ci(deltas: np.ndarray, num_resamples: int = BOOTSTRAP_RESAMPLES,
                        seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> Tuple[float, float]:
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(num_resamples, n))
    resampled_means = np.mean(deltas[indices], axis=1)
    low = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    high = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return low, high


def load_model_instance(model_id: str, model_type: str, scale: int, ckpt_rel_path: str, extra_kwargs: dict = None):
    extra = extra_kwargs or {}
    if model_type == "srcnn":
        model = SRCNN()
    elif model_type == "espcn":
        mid_ch = extra.get("mid_channels", 32)
        model = ESPCN(scale=scale, mid_channels=mid_ch)
    elif model_type == "fsrcnn":
        model = FSRCNN(scale=scale)
    elif model_type == "edsr":
        n_feats = extra.get("n_feats", 64)
        n_resblocks = extra.get("n_resblocks", 16)
        model = EDSR(scale=scale, n_feats=n_feats, n_resblocks=n_resblocks)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt_path = PROJECT_ROOT / ckpt_rel_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {model_id}: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_prediction(pred_t: torch.Tensor, hr_t: torch.Tensor, scale: int) -> Tuple[float, float]:
    min_h = min(pred_t.shape[1], hr_t.shape[1])
    min_w = min(pred_t.shape[2], hr_t.shape[2])
    pred_t = pred_t[:, :min_h, :min_w]
    hr_t = hr_t[:, :min_h, :min_w]

    psnr_val, ssim_val = compute_psnr_ssim(pred_t, hr_t, scale=scale, y_channel=True)
    return psnr_val, ssim_val


def run_model_inference(model, model_type: str, lr_np: np.ndarray, hr_shape: Tuple[int, int], scale: int) -> Tuple[torch.Tensor, float]:
    h_hr, w_hr = hr_shape[:2]
    t0 = time.perf_counter()

    if model_type == "bicubic":
        bic_img = Image.fromarray(lr_np).resize((w_hr, h_hr), Image.BICUBIC)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred_t = torch.from_numpy(np.array(bic_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return pred_t, latency_ms

    if model_type == "srcnn":
        bic_img = Image.fromarray(lr_np).resize((w_hr, h_hr), Image.BICUBIC)
        inp_arr = np.array(bic_img, dtype=np.float32) / 255.0
    else:
        inp_arr = lr_np.astype(np.float32) / 255.0

    inp_t = torch.from_numpy(inp_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_t = model(inp_t)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_t = out_t.squeeze(0).cpu().clamp(0.0, 1.0)

    return pred_t, latency_ms


def main():
    print("=" * 80)
    print("PHASE 4.2: STATISTICAL ROBUSTNESS VALIDATION")
    print(f"Device: {DEVICE}")
    print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES} | Seed: {BOOTSTRAP_SEED}")
    print("=" * 80)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Validate dataset image counts
    all_entries = manifest["entries"]
    set14_entries = [e for e in all_entries if e["dataset"] == "Set14"]
    bsd100_entries = [e for e in all_entries if e["dataset"] == "BSD100"]

    if len(set14_entries) != 14:
        raise ValueError(f"Expected 14 Set14 images, found {len(set14_entries)}")
    if len(bsd100_entries) != 100:
        raise ValueError(f"Expected 100 BSD100 images, found {len(bsd100_entries)}")

    print(f"Datasets verified: Set14 ({len(set14_entries)} imgs), BSD100 ({len(bsd100_entries)} imgs)")

    # Model configuration
    model_definitions = {
        2: [
            {"id": "Bicubic", "name": "bicubic", "type": "bicubic", "group": "baseline", "ckpt": None},
            {"id": "E9", "name": "E9_ESPCN_2x", "type": "espcn", "group": "phase3", "ckpt": "outputs/checkpoints/phase3/E9_espcn_2x_best.pt", "extra": {"mid_channels": 32}},
            {"id": "E10", "name": "E10_FSRCNN_2x", "type": "fsrcnn", "group": "phase3", "ckpt": "outputs/checkpoints/phase3/E10_fsrcnn_2x_best.pt"},
            {"id": "E11", "name": "E11_EDSR_2x", "type": "edsr", "group": "phase3", "ckpt": "outputs/checkpoints/phase3/E11_edsr_2x_best.pt", "extra": {"n_feats": 64, "n_resblocks": 16}},
            {"id": "E1", "name": "E1_SRCNN_2x_MSE", "type": "srcnn", "group": "legacy", "ckpt": "outputs/checkpoints/srcnn_2x_mse.pt"},
            {"id": "E2", "name": "E2_SRCNN_2x_Perceptual", "type": "srcnn", "group": "legacy", "ckpt": "outputs/checkpoints/srcnn_2x_mse_perceptual.pt"},
            {"id": "E5", "name": "E5_ESPCN_2x_MSE", "type": "espcn", "group": "legacy", "ckpt": "outputs/checkpoints/espcn_2x_mse.pt", "extra": {"mid_channels": 64}},
            {"id": "E6", "name": "E6_ESPCN_2x_Perceptual", "type": "espcn", "group": "legacy", "ckpt": "outputs/checkpoints/espcn_2x_mse_perceptual.pt", "extra": {"mid_channels": 64}},
        ],
        4: [
            {"id": "Bicubic", "name": "bicubic", "type": "bicubic", "group": "baseline", "ckpt": None},
            {"id": "E12", "name": "E12_ESPCN_4x", "type": "espcn", "group": "phase3", "ckpt": "outputs/checkpoints/phase3/E12_espcn_4x_best.pt", "extra": {"mid_channels": 32}},
            {"id": "E13", "name": "E13_FSRCNN_4x", "type": "fsrcnn", "group": "phase3", "ckpt": "outputs/checkpoints/phase3/E13_fsrcnn_4x_best.pt"},
            {"id": "E14", "name": "E14_EDSR_4x", "type": "edsr", "group": "phase3", "ckpt": "outputs/checkpoints/phase3/E14_edsr_4x_best.pt", "extra": {"n_feats": 64, "n_resblocks": 16}},
            {"id": "E3", "name": "E3_SRCNN_4x_MSE", "type": "srcnn", "group": "legacy", "ckpt": "outputs/checkpoints/srcnn_4x_mse.pt"},
            {"id": "E4", "name": "E4_SRCNN_4x_Perceptual", "type": "srcnn", "group": "legacy", "ckpt": "outputs/checkpoints/srcnn_4x_mse_perceptual.pt"},
            {"id": "E7", "name": "E7_ESPCN_4x_MSE", "type": "espcn", "group": "legacy", "ckpt": "outputs/checkpoints/espcn_4x_mse.pt", "extra": {"mid_channels": 64}},
            {"id": "E8", "name": "E8_ESPCN_4x_Perceptual", "type": "espcn", "group": "legacy", "ckpt": "outputs/checkpoints/espcn_4x_mse_perceptual.pt", "extra": {"mid_channels": 64}},
        ]
    }

    # Load all models into memory
    print("\n--- Loading Models ---")
    loaded_models = {}
    for scale, m_list in model_definitions.items():
        for m_info in m_list:
            m_id = m_info["id"]
            if m_info["ckpt"] is not None:
                key = f"{m_id}_{scale}x"
                try:
                    loaded_models[key] = load_model_instance(
                        m_id, m_info["type"], scale, m_info["ckpt"], m_info.get("extra")
                    )
                    print(f"Loaded {key}: {m_info['name']} from {m_info['ckpt']}")
                except Exception as e:
                    print(f"FAILED to load {key}: {e}")
                    raise

    # Degradation regimes
    conditions = ["Clean", "Sensor_Noise", "Motion_Blur", "JPEG_Q40", "Combined_CCTV"]

    def generate_lr(hr_np: np.ndarray, scale: int, condition: str, seed: int, canonical_lr_path: Path) -> np.ndarray:
        if condition == "Clean":
            if canonical_lr_path.exists():
                return np.array(Image.open(canonical_lr_path).convert("RGB"))
            return downscale_bicubic(hr_np, scale)
        elif condition == "Sensor_Noise":
            return gaussian_noise(downscale_bicubic(hr_np, scale), sigma=10.0, seed=seed)
        elif condition == "Motion_Blur":
            return downscale_bicubic(motion_blur(hr_np, size=5, angle=45.0), scale)
        elif condition == "JPEG_Q40":
            return jpeg_compress(downscale_bicubic(hr_np, scale), quality=40)
        elif condition == "Combined_CCTV":
            return degrade(hr_np, scale=scale, blur_type="motion", blur_ksize=5, motion_angle=30.0,
                           noise_sigma=8.0, jpeg_quality=50, seed=seed)
        else:
            raise ValueError(f"Unknown condition: {condition}")

    per_image_rows = []
    datasets_to_eval = [("Set14", set14_entries), ("BSD100", bsd100_entries)]

    total_eval_start = time.time()

    for ds_name, entries in datasets_to_eval:
        print(f"\n==================================================")
        print(f"Starting Evaluation on {ds_name} ({len(entries)} images)")
        print(f"==================================================")

        for scale in [2, 4]:
            models_for_scale = model_definitions[scale]

            for cond in conditions:
                cond_start = time.time()
                print(f"Processing: {ds_name} | Scale: {scale}x | Condition: {cond}...", flush=True)

                for entry in entries:
                    img_id = entry["image_id"]
                    hr_path = PROJECT_ROOT / entry["hr_path"]
                    lr_canonical_path = PROJECT_ROOT / entry[f"lr_x{scale}_path"]

                    hr_img = Image.open(hr_path).convert("RGB")
                    hr_np = np.array(hr_img)
                    hr_t = torch.from_numpy(hr_np.astype(np.float32) / 255.0).permute(2, 0, 1)

                    seed = derive_deterministic_seed(img_id, scale, cond)
                    # Generate LR once for all models
                    lr_np = generate_lr(hr_np, scale, cond, seed, lr_canonical_path)

                    # 1. Run Bicubic baseline first to establish reference
                    bic_pred, bic_lat = run_model_inference(None, "bicubic", lr_np, hr_np.shape, scale)
                    bic_psnr, bic_ssim = evaluate_prediction(bic_pred, hr_t, scale)

                    # Store bicubic row
                    per_image_rows.append({
                        "dataset": ds_name,
                        "image_id": img_id,
                        "scale": scale,
                        "condition": cond,
                        "model_id": "Bicubic",
                        "model_name": f"Bicubic_{scale}x",
                        "group": "baseline",
                        "psnr": bic_psnr,
                        "ssim": bic_ssim,
                        "bicubic_psnr": bic_psnr,
                        "bicubic_ssim": bic_ssim,
                        "delta_psnr": 0.0,
                        "delta_ssim": 0.0,
                        "latency_ms": bic_lat
                    })

                    # 2. Run all other models
                    for m_info in models_for_scale:
                        m_id = m_info["id"]
                        if m_id == "Bicubic":
                            continue

                        model_key = f"{m_id}_{scale}x"
                        model_obj = loaded_models[model_key]

                        pred_t, lat_ms = run_model_inference(
                            model_obj, m_info["type"], lr_np, hr_np.shape, scale
                        )
                        psnr_val, ssim_val = evaluate_prediction(pred_t, hr_t, scale)

                        delta_p = psnr_val - bic_psnr
                        delta_s = ssim_val - bic_ssim

                        per_image_rows.append({
                            "dataset": ds_name,
                            "image_id": img_id,
                            "scale": scale,
                            "condition": cond,
                            "model_id": m_id,
                            "model_name": m_info["name"],
                            "group": m_info["group"],
                            "psnr": psnr_val,
                            "ssim": ssim_val,
                            "bicubic_psnr": bic_psnr,
                            "bicubic_ssim": bic_ssim,
                            "delta_psnr": delta_p,
                            "delta_ssim": delta_s,
                            "latency_ms": lat_ms
                        })

                elapsed_cond = time.time() - cond_start
                print(f"  Completed {cond} in {elapsed_cond:.2f}s", flush=True)

    print(f"\nTotal inference completed in {time.time() - total_eval_start:.2f}s")
    print(f"Total per-image records: {len(per_image_rows)}")

    # Compute Statistical Aggregation & Bootstrap 95% CIs
    print("\n--- Computing Statistical Aggregations & Bootstrap 95% CIs ---")
    summary_rows = []
    json_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(DEVICE),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "psnr_tie_tolerance": PSNR_TIE_TOL,
            "ssim_tie_tolerance": SSIM_TIE_TOL,
        },
        "results": {}
    }

    # Group by (dataset, scale, condition, model_id)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in per_image_rows:
        key = (r["dataset"], r["scale"], r["condition"], r["model_id"])
        groups[key].append(r)

    for (ds_name, scale, cond, m_id), rows in groups.items():
        sample_count = len(rows)
        psnrs = np.array([r["psnr"] for r in rows], dtype=np.float64)
        ssims = np.array([r["ssim"] for r in rows], dtype=np.float64)
        bic_psnrs = np.array([r["bicubic_psnr"] for r in rows], dtype=np.float64)
        bic_ssims = np.array([r["bicubic_ssim"] for r in rows], dtype=np.float64)
        delta_psnrs = np.array([r["delta_psnr"] for r in rows], dtype=np.float64)
        delta_ssims = np.array([r["delta_ssim"] for r in rows], dtype=np.float64)
        latencies = np.array([r["latency_ms"] for r in rows], dtype=np.float64)

        m_name = rows[0]["model_name"]
        m_group = rows[0]["group"]

        # Means, medians, stds
        mean_p, med_p, std_p = float(np.mean(psnrs)), float(np.median(psnrs)), float(np.std(psnrs, ddof=1) if sample_count > 1 else 0.0)
        mean_s, med_s, std_s = float(np.mean(ssims)), float(np.median(ssims)), float(np.std(ssims, ddof=1) if sample_count > 1 else 0.0)

        mean_bic_p, mean_bic_s = float(np.mean(bic_psnrs)), float(np.mean(bic_ssims))

        mean_dp, med_dp, std_dp = float(np.mean(delta_psnrs)), float(np.median(delta_psnrs)), float(np.std(delta_psnrs, ddof=1) if sample_count > 1 else 0.0)
        mean_ds, med_ds, std_ds = float(np.mean(delta_ssims)), float(np.median(delta_ssims)), float(np.std(delta_ssims, ddof=1) if sample_count > 1 else 0.0)

        # Bootstrap CIs for improvement over Bicubic
        ci_p_low, ci_p_high = paired_bootstrap_ci(delta_psnrs)
        ci_s_low, ci_s_high = paired_bootstrap_ci(delta_ssims)

        # Statistical significance: 0 not in [low, high]
        is_p_sig = bool((ci_p_low > 0) or (ci_p_high < 0))
        is_s_sig = bool((ci_s_low > 0) or (ci_s_high < 0))

        # Win / Tie / Loss rates
        if m_id == "Bicubic":
            win_p = tie_p = loss_p = 0.0
            tie_p = 100.0
            win_s = tie_s = loss_s = 0.0
            tie_s = 100.0
        else:
            win_p = float(np.sum(delta_psnrs > PSNR_TIE_TOL) / sample_count * 100.0)
            tie_p = float(np.sum(np.abs(delta_psnrs) <= PSNR_TIE_TOL) / sample_count * 100.0)
            loss_p = float(np.sum(delta_psnrs < -PSNR_TIE_TOL) / sample_count * 100.0)

            win_s = float(np.sum(delta_ssims > SSIM_TIE_TOL) / sample_count * 100.0)
            tie_s = float(np.sum(np.abs(delta_ssims) <= SSIM_TIE_TOL) / sample_count * 100.0)
            loss_s = float(np.sum(delta_ssims < -SSIM_TIE_TOL) / sample_count * 100.0)

        mean_lat = float(np.mean(latencies))

        summary_rows.append({
            "dataset": ds_name,
            "scale": scale,
            "condition": cond,
            "model_id": m_id,
            "model_name": m_name,
            "group": m_group,
            "sample_count": sample_count,
            "mean_psnr": round(mean_p, 4),
            "median_psnr": round(med_p, 4),
            "std_psnr": round(std_p, 4),
            "mean_ssim": round(mean_s, 4),
            "median_ssim": round(med_s, 4),
            "std_ssim": round(std_s, 4),
            "bicubic_mean_psnr": round(mean_bic_p, 4),
            "bicubic_mean_ssim": round(mean_bic_s, 4),
            "mean_delta_psnr": round(mean_dp, 4),
            "median_delta_psnr": round(med_dp, 4),
            "std_delta_psnr": round(std_dp, 4),
            "ci95_delta_psnr_low": round(ci_p_low, 4),
            "ci95_delta_psnr_high": round(ci_p_high, 4),
            "is_psnr_significant": is_p_sig,
            "mean_delta_ssim": round(mean_ds, 4),
            "median_delta_ssim": round(med_ds, 4),
            "std_delta_ssim": round(std_ds, 4),
            "ci95_delta_ssim_low": round(ci_s_low, 4),
            "ci95_delta_ssim_high": round(ci_s_high, 4),
            "is_ssim_significant": is_s_sig,
            "win_rate_psnr": round(win_p, 2),
            "tie_rate_psnr": round(tie_p, 2),
            "loss_rate_psnr": round(loss_p, 2),
            "win_rate_ssim": round(win_s, 2),
            "tie_rate_ssim": round(tie_s, 2),
            "loss_rate_ssim": round(loss_s, 2),
            "mean_latency_ms": round(mean_lat, 2)
        })

    # Sort summary rows systematically
    cond_order = {c: i for i, c in enumerate(conditions)}
    model_order = {"Bicubic": 0, "E9": 1, "E10": 2, "E11": 3, "E12": 4, "E13": 5, "E14": 6,
                   "E1": 7, "E2": 8, "E3": 9, "E4": 10, "E5": 11, "E6": 12, "E7": 13, "E8": 14}
    summary_rows.sort(key=lambda x: (
        x["dataset"], x["scale"], cond_order.get(x["condition"], 99), model_order.get(x["model_id"], 99)
    ))

    # Save to CSV and JSON
    import csv

    per_image_csv_path = RESULTS_DIR / "phase4_2_per_image_metrics.csv"
    with open(per_image_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "image_id", "scale", "condition", "model_id", "model_name", "group",
            "psnr", "ssim", "bicubic_psnr", "bicubic_ssim", "delta_psnr", "delta_ssim", "latency_ms"
        ])
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"Saved: {per_image_csv_path}")

    summary_csv_path = RESULTS_DIR / "phase4_2_statistical_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_csv_path}")

    # Build structured JSON
    for s in summary_rows:
        ds = s["dataset"]
        scale_str = f"{s['scale']}x"
        cond = s["condition"]
        m_id = s["model_id"]

        if ds not in json_data["results"]:
            json_data["results"][ds] = {}
        if scale_str not in json_data["results"][ds]:
            json_data["results"][ds][scale_str] = {}
        if cond not in json_data["results"][ds][scale_str]:
            json_data["results"][ds][scale_str][cond] = {}

        json_data["results"][ds][scale_str][cond][m_id] = s

    json_path = RESULTS_DIR / "phase4_2_statistical_robustness.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n==================================================")
    print("PHASE 4.2 EXECUTION COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    main()
