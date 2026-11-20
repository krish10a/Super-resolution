"""
Phase 4.3: Denoising Ablation Study Script.

Evaluates the impact of classical preprocessing denoisers (NLM and Bilateral)
prior to Super-Resolution under sensor noise and combined CCTV degradations.

Ablation variants:
- A0: No denoising (direct SR on degraded LR)
- A1: Classical Non-Local Means (cv2.fastNlMeansDenoisingColored, h=3, hColor=3, template=7, search=21)
- A2: Bilateral Filter (cv2.bilateralFilter, d=5, sigmaColor=25, sigmaSpace=25)

Models evaluated:
- 2x: Bicubic, E9 (ESPCN), E11 (EDSR)
- 4x: Bicubic, E12 (ESPCN), E14 (EDSR)

Conditions:
- Sensor_Noise: Gaussian noise sigma=10.0
- Combined_CCTV: Motion blur (k=5, 30 deg) + downscale + noise sigma=8.0 + JPEG Q=50

Datasets:
- Set14 (14 images)
- BSD100 (100 images)

Metrics & Statistics:
- PSNR & SSIM (Y-channel, border shaved by scale)
- Denoiser latency vs Model SR inference latency measured separately
- Paired differences vs A0 (delta_vs_a0) and vs Bicubic A0 (delta_vs_bicubic)
- 10,000 paired bootstrap 95% confidence intervals (seed=42)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import cv2
from PIL import Image
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.espcn import ESPCN
from models.edsr import EDSR
from metrics.psnr_ssim import compute_psnr_ssim
from data.degradation import (
    downscale_bicubic,
    gaussian_noise,
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


def apply_denoiser(lr_np: np.ndarray, denoiser: str) -> Tuple[np.ndarray, float]:
    """
    Apply denoiser to uint8 RGB LR image.
    Returns (denoised_lr_np, denoiser_latency_ms).
    """
    if denoiser == "A0":
        return lr_np, 0.0

    t0 = time.perf_counter()
    if denoiser == "A1":
        # Non-Local Means (convert to BGR for proper Lab internal conversion in OpenCV)
        lr_bgr = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(
            lr_bgr, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
        )
        denoised_np = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    elif denoiser == "A2":
        # Bilateral filter
        denoised_np = cv2.bilateralFilter(lr_np, d=5, sigmaColor=25, sigmaSpace=25)
    else:
        raise ValueError(f"Unknown denoiser variant: {denoiser}")

    lat_ms = (time.perf_counter() - t0) * 1000.0
    return denoised_np, lat_ms


def load_model_instance(model_id: str, model_type: str, scale: int, ckpt_rel_path: str, extra_kwargs: dict = None):
    extra = extra_kwargs or {}
    if model_type == "espcn":
        mid_ch = extra.get("mid_channels", 32)
        model = ESPCN(scale=scale, mid_channels=mid_ch)
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
    print("PHASE 4.3: DENOISING ABLATION STUDY")
    print(f"Device: {DEVICE}")
    print(f"Denoisers: A0 (None), A1 (NLM), A2 (Bilateral)")
    print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES} | Seed: {BOOTSTRAP_SEED}")
    print("=" * 80)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    all_entries = manifest["entries"]
    set14_entries = [e for e in all_entries if e["dataset"] == "Set14"]
    bsd100_entries = [e for e in all_entries if e["dataset"] == "BSD100"]

    if len(set14_entries) != 14:
        raise ValueError(f"Expected 14 Set14 images, found {len(set14_entries)}")
    if len(bsd100_entries) != 100:
        raise ValueError(f"Expected 100 BSD100 images, found {len(bsd100_entries)}")

    # Model configuration for Phase 4.3
    model_definitions = {
        2: [
            {"id": "Bicubic", "name": "bicubic", "type": "bicubic", "ckpt": None},
            {"id": "E9", "name": "E9_ESPCN_2x", "type": "espcn", "ckpt": "outputs/checkpoints/phase3/E9_espcn_2x_best.pt", "extra": {"mid_channels": 32}},
            {"id": "E11", "name": "E11_EDSR_2x", "type": "edsr", "ckpt": "outputs/checkpoints/phase3/E11_edsr_2x_best.pt", "extra": {"n_feats": 64, "n_resblocks": 16}},
        ],
        4: [
            {"id": "Bicubic", "name": "bicubic", "type": "bicubic", "ckpt": None},
            {"id": "E12", "name": "E12_ESPCN_4x", "type": "espcn", "ckpt": "outputs/checkpoints/phase3/E12_espcn_4x_best.pt", "extra": {"mid_channels": 32}},
            {"id": "E14", "name": "E14_EDSR_4x", "type": "edsr", "ckpt": "outputs/checkpoints/phase3/E14_edsr_4x_best.pt", "extra": {"n_feats": 64, "n_resblocks": 16}},
        ]
    }

    # Load models
    print("\n--- Loading Models ---")
    loaded_models = {}
    for scale, m_list in model_definitions.items():
        for m_info in m_list:
            m_id = m_info["id"]
            if m_info["ckpt"] is not None:
                key = f"{m_id}_{scale}x"
                loaded_models[key] = load_model_instance(
                    m_id, m_info["type"], scale, m_info["ckpt"], m_info.get("extra")
                )
                print(f"Loaded {key}: {m_info['name']}")

    conditions = ["Sensor_Noise", "Combined_CCTV"]
    denoisers = ["A0", "A1", "A2"]

    def generate_raw_lr(hr_np: np.ndarray, scale: int, condition: str, seed: int) -> np.ndarray:
        if condition == "Sensor_Noise":
            return gaussian_noise(downscale_bicubic(hr_np, scale), sigma=10.0, seed=seed)
        elif condition == "Combined_CCTV":
            return degrade(hr_np, scale=scale, blur_type="motion", blur_ksize=5, motion_angle=30.0,
                           noise_sigma=8.0, jpeg_quality=50, seed=seed)
        else:
            raise ValueError(f"Unknown condition: {condition}")

    # Data collection
    # We will record per-image:
    # (dataset, image_id, scale, condition, denoiser, model_id) -> record
    raw_eval_records = []
    datasets_to_eval = [("Set14", set14_entries), ("BSD100", bsd100_entries)]

    total_start = time.time()

    for ds_name, entries in datasets_to_eval:
        print(f"\n==================================================")
        print(f"Starting Phase 4.3 Evaluation on {ds_name} ({len(entries)} images)")
        print(f"==================================================")

        for scale in [2, 4]:
            models_for_scale = model_definitions[scale]

            for cond in conditions:
                cond_start = time.time()
                print(f"Processing: {ds_name} | Scale: {scale}x | Condition: {cond}...", flush=True)

                for entry in entries:
                    img_id = entry["image_id"]
                    hr_path = PROJECT_ROOT / entry["hr_path"]
                    hr_img = Image.open(hr_path).convert("RGB")
                    hr_np = np.array(hr_img)
                    hr_t = torch.from_numpy(hr_np.astype(np.float32) / 255.0).permute(2, 0, 1)

                    seed = derive_deterministic_seed(img_id, scale, cond)
                    raw_lr_np = generate_raw_lr(hr_np, scale, cond, seed)

                    # For each denoiser variant:
                    # 1. Denoise the LR once
                    # 2. Feed identical denoised LR to all models
                    for d_name in denoisers:
                        denoised_lr_np, denoiser_lat = apply_denoiser(raw_lr_np, d_name)

                        for m_info in models_for_scale:
                            m_id = m_info["id"]
                            model_key = f"{m_id}_{scale}x"
                            model_obj = loaded_models.get(model_key)

                            pred_t, model_lat = run_model_inference(
                                model_obj, m_info["type"], denoised_lr_np, hr_np.shape, scale
                            )
                            p_val, s_val = evaluate_prediction(pred_t, hr_t, scale)

                            raw_eval_records.append({
                                "dataset": ds_name,
                                "image_id": img_id,
                                "scale": scale,
                                "condition": cond,
                                "denoiser": d_name,
                                "model_id": m_id,
                                "model_name": m_info["name"],
                                "psnr": p_val,
                                "ssim": s_val,
                                "denoiser_latency_ms": denoiser_lat,
                                "model_latency_ms": model_lat
                            })

                print(f"  Completed {cond} in {time.time() - cond_start:.2f}s", flush=True)

    print(f"\nAll inference completed in {time.time() - total_start:.2f}s")
    print(f"Total evaluation instances: {len(raw_eval_records)}")

    # Index records for paired delta calculations:
    # 1. Lookup for A0 baseline of same (dataset, image_id, scale, condition, model_id)
    # 2. Lookup for Bicubic A0 baseline of same (dataset, image_id, scale, condition)
    a0_lookup = {}
    bicubic_a0_lookup = {}
    for r in raw_eval_records:
        if r["denoiser"] == "A0":
            key_m = (r["dataset"], r["image_id"], r["scale"], r["condition"], r["model_id"])
            a0_lookup[key_m] = (r["psnr"], r["ssim"])
            if r["model_id"] == "Bicubic":
                key_bic = (r["dataset"], r["image_id"], r["scale"], r["condition"])
                bicubic_a0_lookup[key_bic] = (r["psnr"], r["ssim"])

    # Compute paired deltas
    per_image_rows = []
    for r in raw_eval_records:
        key_m = (r["dataset"], r["image_id"], r["scale"], r["condition"], r["model_id"])
        key_bic = (r["dataset"], r["image_id"], r["scale"], r["condition"])

        a0_p, a0_s = a0_lookup[key_m]
        bic_a0_p, bic_a0_s = bicubic_a0_lookup[key_bic]

        delta_p_vs_a0 = r["psnr"] - a0_p
        delta_s_vs_a0 = r["ssim"] - a0_s

        delta_p_vs_bic = r["psnr"] - bic_a0_p
        delta_s_vs_bic = r["ssim"] - bic_a0_s

        per_image_rows.append({
            "dataset": r["dataset"],
            "image_id": r["image_id"],
            "scale": r["scale"],
            "condition": r["condition"],
            "denoiser": r["denoiser"],
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "psnr": r["psnr"],
            "ssim": r["ssim"],
            "a0_psnr": a0_p,
            "a0_ssim": a0_s,
            "delta_psnr_vs_a0": delta_p_vs_a0,
            "delta_ssim_vs_a0": delta_s_vs_a0,
            "bicubic_a0_psnr": bic_a0_p,
            "bicubic_a0_ssim": bic_a0_s,
            "delta_psnr_vs_bicubic": delta_p_vs_bic,
            "delta_ssim_vs_bicubic": delta_s_vs_bic,
            "denoiser_latency_ms": r["denoiser_latency_ms"],
            "model_latency_ms": r["model_latency_ms"]
        })

    # Group and calculate summary statistics & bootstrap CIs
    print("\n--- Computing Denoising Statistical Summary & Bootstrap CIs ---")
    groups = defaultdict(list)
    for r in per_image_rows:
        key = (r["dataset"], r["scale"], r["condition"], r["denoiser"], r["model_id"])
        groups[key].append(r)

    summary_rows = []
    json_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(DEVICE),
            "denoisers": {
                "A0": "Direct SR (no denoiser)",
                "A1": "Non-Local Means (cv2.fastNlMeansDenoisingColored h=3, hColor=3, win=7, search=21)",
                "A2": "Bilateral Filter (cv2.bilateralFilter d=5, sigmaColor=25, sigmaSpace=25)"
            },
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "psnr_tie_tolerance": PSNR_TIE_TOL,
            "ssim_tie_tolerance": SSIM_TIE_TOL,
        },
        "results": {}
    }

    for (ds_name, scale, cond, d_name, m_id), rows in groups.items():
        sample_count = len(rows)
        psnrs = np.array([r["psnr"] for r in rows], dtype=np.float64)
        ssims = np.array([r["ssim"] for r in rows], dtype=np.float64)
        dp_vs_a0 = np.array([r["delta_psnr_vs_a0"] for r in rows], dtype=np.float64)
        ds_vs_a0 = np.array([r["delta_ssim_vs_a0"] for r in rows], dtype=np.float64)
        dp_vs_bic = np.array([r["delta_psnr_vs_bicubic"] for r in rows], dtype=np.float64)
        ds_vs_bic = np.array([r["delta_ssim_vs_bicubic"] for r in rows], dtype=np.float64)
        den_lats = np.array([r["denoiser_latency_ms"] for r in rows], dtype=np.float64)
        mod_lats = np.array([r["model_latency_ms"] for r in rows], dtype=np.float64)

        m_name = rows[0]["model_name"]

        mean_p, med_p, std_p = float(np.mean(psnrs)), float(np.median(psnrs)), float(np.std(psnrs, ddof=1) if sample_count > 1 else 0.0)
        mean_s, med_s, std_s = float(np.mean(ssims)), float(np.median(ssims)), float(np.std(ssims, ddof=1) if sample_count > 1 else 0.0)

        mean_dp_a0 = float(np.mean(dp_vs_a0))
        mean_ds_a0 = float(np.mean(ds_vs_a0))
        ci_dp_a0_low, ci_dp_a0_high = paired_bootstrap_ci(dp_vs_a0)
        ci_ds_a0_low, ci_ds_a0_high = paired_bootstrap_ci(ds_vs_a0)

        mean_dp_bic = float(np.mean(dp_vs_bic))
        mean_ds_bic = float(np.mean(ds_vs_bic))
        ci_dp_bic_low, ci_dp_bic_high = paired_bootstrap_ci(dp_vs_bic)
        ci_ds_bic_low, ci_ds_bic_high = paired_bootstrap_ci(ds_vs_bic)

        mean_den_lat = float(np.mean(den_lats))
        mean_mod_lat = float(np.mean(mod_lats))

        # Check significance of denoiser impact vs A0
        is_denoiser_p_sig = bool((ci_dp_a0_low > 0) or (ci_dp_a0_high < 0)) if d_name != "A0" else False
        is_denoiser_s_sig = bool((ci_ds_a0_low > 0) or (ci_ds_a0_high < 0)) if d_name != "A0" else False

        summary_rows.append({
            "dataset": ds_name,
            "scale": scale,
            "condition": cond,
            "denoiser": d_name,
            "model_id": m_id,
            "model_name": m_name,
            "sample_count": sample_count,
            "mean_psnr": round(mean_p, 4),
            "median_psnr": round(med_p, 4),
            "std_psnr": round(std_p, 4),
            "mean_ssim": round(mean_s, 4),
            "median_ssim": round(med_s, 4),
            "std_ssim": round(std_s, 4),
            "mean_delta_psnr_vs_a0": round(mean_dp_a0, 4),
            "ci95_delta_psnr_vs_a0_low": round(ci_dp_a0_low, 4),
            "ci95_delta_psnr_vs_a0_high": round(ci_dp_a0_high, 4),
            "is_denoiser_psnr_significant": is_denoiser_p_sig,
            "mean_delta_ssim_vs_a0": round(mean_ds_a0, 4),
            "ci95_delta_ssim_vs_a0_low": round(ci_ds_a0_low, 4),
            "ci95_delta_ssim_vs_a0_high": round(ci_ds_a0_high, 4),
            "is_denoiser_ssim_significant": is_denoiser_s_sig,
            "mean_delta_psnr_vs_bicubic": round(mean_dp_bic, 4),
            "ci95_delta_psnr_vs_bicubic_low": round(ci_dp_bic_low, 4),
            "ci95_delta_psnr_vs_bicubic_high": round(ci_dp_bic_high, 4),
            "mean_delta_ssim_vs_bicubic": round(mean_ds_bic, 4),
            "ci95_delta_ssim_vs_bicubic_low": round(ci_ds_bic_low, 4),
            "ci95_delta_ssim_vs_bicubic_high": round(ci_ds_bic_high, 4),
            "mean_denoiser_latency_ms": round(mean_den_lat, 2),
            "mean_model_latency_ms": round(mean_mod_lat, 2),
            "total_pipeline_latency_ms": round(mean_den_lat + mean_mod_lat, 2)
        })

    # Sort summary
    cond_order = {c: i for i, c in enumerate(conditions)}
    den_order = {"A0": 0, "A1": 1, "A2": 2}
    model_order = {"Bicubic": 0, "E9": 1, "E11": 2, "E12": 3, "E14": 4}
    summary_rows.sort(key=lambda x: (
        x["dataset"], x["scale"], cond_order.get(x["condition"], 99),
        den_order.get(x["denoiser"], 99), model_order.get(x["model_id"], 99)
    ))

    # Save CSVs
    import csv

    per_image_csv_path = RESULTS_DIR / "phase4_3_denoising_per_image_metrics.csv"
    with open(per_image_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"Saved: {per_image_csv_path}")

    summary_csv_path = RESULTS_DIR / "phase4_3_denoising_summary.csv"
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
        den = s["denoiser"]
        m_id = s["model_id"]

        if ds not in json_data["results"]:
            json_data["results"][ds] = {}
        if scale_str not in json_data["results"][ds]:
            json_data["results"][ds][scale_str] = {}
        if cond not in json_data["results"][ds][scale_str]:
            json_data["results"][ds][scale_str][cond] = {}
        if den not in json_data["results"][ds][scale_str][cond]:
            json_data["results"][ds][scale_str][cond][den] = {}

        json_data["results"][ds][scale_str][cond][den][m_id] = s

    json_path = RESULTS_DIR / "phase4_3_denoising_ablation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n==================================================")
    print("PHASE 4.3 EXECUTION COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    main()
