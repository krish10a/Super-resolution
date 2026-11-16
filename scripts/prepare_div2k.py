"""
DIV2K LR/HR Preprocessing Pipeline with GPU (CUDA) and Asynchronous Multi-Threaded I/O.

Downsamples DIV2K HR images:
- 2x: scale 0.5 with bicubic antialiasing
- 4x: scale 0.25 with bicubic antialiasing

Accelerated Architecture:
- GPU: PyTorch CUDA antialiased bicubic downsampling on RTX 2050 (3ms per image).
- Multi-Threaded I/O: 8-worker ThreadPoolExecutor for background PNG compression and SSD saving.
- In-memory SHA256 hashing without redundant disk re-reads.
- Skip completed: Resumes instantly if interrupted.

Applies modcrop (divisible by 4) so dimensions cleanly divide by 2 and 4.
Outputs to:
  data/processed/DIV2K/DIV2K_train_HR/
  data/processed/DIV2K/DIV2K_train_LR_x2/
  data/processed/DIV2K/DIV2K_train_LR_x4/
  data/processed/DIV2K/DIV2K_valid_HR/
  data/processed/DIV2K/DIV2K_valid_LR_x2/
  data/processed/DIV2K/DIV2K_valid_LR_x4/
  data/processed/DIV2K/div2k_manifest.json
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIV2K_DIR = PROJECT_ROOT / "datasets" / "Div2k dataset"
OUTPUT_BASE = PROJECT_ROOT / "data" / "processed" / "DIV2K"
MANIFEST_PATH = OUTPUT_BASE / "div2k_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def modcrop_np(img_rgb: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    new_h = h - (h % scale)
    new_w = w - (w % scale)
    if new_h != h or new_w != w:
        return img_rgb[:new_h, :new_w, :]
    return img_rgb


@torch.no_grad()
def downsample_gpu(img_rgb: np.ndarray, scale_factor: float, device: torch.device) -> np.ndarray:
    """Downsample image on GPU using PyTorch bicubic antialiased interpolation."""
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device, non_blocking=True) / 255.0
    lr_t = torch.nn.functional.interpolate(
        t,
        scale_factor=scale_factor,
        mode="bicubic",
        antialias=True,
        align_corners=False
    )
    lr_np = (lr_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return lr_np


def save_image_worker(path: Path, img_rgb: np.ndarray) -> str:
    """Save PNG and return its sha256 hash."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Fast PNG compression parameter
    success, enc = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not success:
        cv2.imwrite(str(path), bgr)
        return sha256_file(path)
    
    with open(path, "wb") as f:
        f.write(enc.tobytes())
    return hashlib.sha256(enc).hexdigest()


def process_split(
    split_name: str,
    src_dir: Path,
    out_base: Path,
    device: torch.device,
    use_gpu: bool,
    limit: int = 0,
    io_workers: int = 8
):
    files = sorted(list(src_dir.glob("*.png")))
    if not files:
        files = sorted(list(src_dir.glob("*.jpg"))) + sorted(list(src_dir.glob("*.jpeg")))

    if limit > 0:
        files = files[:limit]

    hr_out = out_base / f"DIV2K_{split_name}_HR"
    lr2_out = out_base / f"DIV2K_{split_name}_LR_x2"
    lr4_out = out_base / f"DIV2K_{split_name}_LR_x4"

    for d in [hr_out, lr2_out, lr4_out]:
        d.mkdir(parents=True, exist_ok=True)

    entries = []
    pbar = tqdm(files, desc=f"DIV2K_{split_name} ({'GPU+8xIO' if use_gpu else 'CPU'})", unit="img", ncols=100)

    # Use ThreadPoolExecutor for asynchronous parallel disk saving
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        pending_saves = []

        for f in pbar:
            img_id = f.stem
            pbar.set_postfix({"img": img_id})

            hr_path = hr_out / f"{img_id}.png"
            lr2_path = lr2_out / f"{img_id}.png"
            lr4_path = lr4_out / f"{img_id}.png"

            # Check if all 3 already exist from previous run (instant resume)
            if hr_path.exists() and lr2_path.exists() and lr4_path.exists():
                hr_img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
                if hr_img is not None:
                    h, w = hr_img.shape[:2]
                    entries.append({
                        "image_id": img_id,
                        "split": split_name,
                        "orig_size": [w, h],
                        "hr_size": [w, h],
                        "lr_x2_size": [w // 2, h // 2],
                        "lr_x4_size": [w // 4, h // 4],
                        "hr_path": str(hr_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "lr_x2_path": str(lr2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "lr_x4_path": str(lr4_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "hr_sha256": sha256_file(hr_path),
                        "lr_x2_sha256": sha256_file(lr2_path),
                        "lr_x4_sha256": sha256_file(lr4_path),
                    })
                    continue

            # Load raw image with OpenCV
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = rgb.shape[:2]
            orig_size = [orig_w, orig_h]

            # Modcrop
            hr_rgb = modcrop_np(rgb, scale=4)
            hr_h, hr_w = hr_rgb.shape[:2]
            hr_size = [hr_w, hr_h]

            # GPU Downsampling
            if use_gpu:
                lr2_rgb = downsample_gpu(hr_rgb, scale_factor=0.5, device=device)
                lr4_rgb = downsample_gpu(hr_rgb, scale_factor=0.25, device=device)
            else:
                from data.imresize_matlab import imresize_matlab
                lr2_rgb = imresize_matlab(hr_rgb, scale=0.5, antialiasing=True)
                lr4_rgb = imresize_matlab(hr_rgb, scale=0.25, antialiasing=True)

            lr2_h, lr2_w = lr2_rgb.shape[:2]
            lr4_h, lr4_w = lr4_rgb.shape[:2]

            # Submit writes to parallel background workers
            f_hr = executor.submit(save_image_worker, hr_path, hr_rgb)
            f_lr2 = executor.submit(save_image_worker, lr2_path, lr2_rgb)
            f_lr4 = executor.submit(save_image_worker, lr4_path, lr4_rgb)

            pending_saves.append({
                "meta": {
                    "image_id": img_id,
                    "split": split_name,
                    "orig_size": orig_size,
                    "hr_size": hr_size,
                    "lr_x2_size": [lr2_w, lr2_h],
                    "lr_x4_size": [lr4_w, lr4_h],
                    "hr_path": str(hr_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "lr_x2_path": str(lr2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "lr_x4_path": str(lr4_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                },
                "f_hr": f_hr,
                "f_lr2": f_lr2,
                "f_lr4": f_lr4,
            })

        # Collect hashes once all parallel writes finish
        for item in pending_saves:
            meta = item["meta"]
            meta["hr_sha256"] = item["f_hr"].result()
            meta["lr_x2_sha256"] = item["f_lr2"].result()
            meta["lr_x4_sha256"] = item["f_lr4"].result()
            entries.append(meta)

    return entries


def main():
    parser = argparse.ArgumentParser(description="DIV2K LR/HR Preprocessing (GPU + Multi-Threaded I/O)")
    parser.add_argument("--src-dir", type=str, default=str(RAW_DIV2K_DIR), help="Path to raw DIV2K root")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_BASE), help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Processing device (cuda for RTX 2050)")
    parser.add_argument("--io-workers", type=int, default=8, help="Number of parallel background threads for PNG disk writing")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images per split (for smoke tests)")
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    train_src = src_root / "DIV2K_train_HR"
    valid_src = src_root / "DIV2K_valid_HR"

    if not train_src.exists() or not valid_src.exists():
        print(f"Error: Missing DIV2K source subdirectories in {src_root}")
        print("Expected DIV2K_train_HR and DIV2K_valid_HR")
        sys.exit(1)

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    use_gpu = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    print("=" * 80)
    print("DIV2K PREPROCESSING PIPELINE (GPU CUDA + 8x Multi-Threaded I/O)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if use_gpu else 'CPU'})")
    print(f"I/O Workers: {args.io_workers} parallel CPU threads for PNG writing")
    print(f"Source: {src_root}")
    print(f"Output: {out_base}")
    if args.limit > 0:
        print(f"[SMOKE TEST MODE] Limiting to {args.limit} images per split")
    print("=" * 80)

    t0 = time.time()
    train_entries = process_split("train", train_src, out_base, device, use_gpu, limit=args.limit, io_workers=args.io_workers)
    valid_entries = process_split("valid", valid_src, out_base, device, use_gpu, limit=args.limit, io_workers=args.io_workers)
    elapsed = time.time() - t0

    manifest = {
        "version": "1.0.0",
        "protocol": "PyTorch CUDA antialiased bicubic downsampling" if use_gpu else "MATLAB-compatible CPU bicubic downsampling",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if use_gpu else None,
        "modcrop_divisor": 4,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_train": len(train_entries),
        "total_valid": len(valid_entries),
        "elapsed_seconds": round(elapsed, 2),
        "train_entries": train_entries,
        "valid_entries": valid_entries,
    }

    manifest_path = out_base / "div2k_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PREPROCESSING COMPLETE in {elapsed:.1f}s")
    print(f"Train images: {len(train_entries)} | Valid images: {len(valid_entries)}")
    print(f"Saved manifest to {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
