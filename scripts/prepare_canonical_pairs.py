"""
Prepare canonical SISR dataset pairs and manifest.

Processes only verified canonical raw datasets:
- Set5: data/raw/Set5_real (5 images)
- Set14: data/raw/Set14_png (14 images)
- BSD100: data/raw/BSD100_png (100 images)
- T91: data/raw/T91_png (91 images)

Applies modcrop (divisible by 4) and MATLAB-compatible bicubic downsampling (data/imresize_matlab.py).
Generates data/canonical_pairs/ and data/dataset_manifest.json.
"""

import os
import sys
import json
import hashlib
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.imresize_matlab import imresize_matlab as imresize

RAW_DATASETS = {
    "Set5": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "Set5_real",
        "split": "test",
    },
    "Set14": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "Set14_png",
        "split": "test",
    },
    "BSD100": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "BSD100_png",
        "split": "test",
    },
    "T91": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "T91_png",
        "split": "train_val",  # Deterministic split 80 train / 11 val
    },
}

CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical_pairs"
MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def modcrop(img: Image.Image, scale: int = 4) -> Image.Image:
    w, h = img.size
    new_w = w - (w % scale)
    new_h = h - (h % scale)
    return img.crop((0, 0, new_w, new_h))


def main():
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "2.0.0",
        "protocol": "MATLAB-compatible bicubic downsampling with antialiasing",
        "modcrop_divisor": 4,
        "seed": 42,
        "summary": {},
        "entries": [],
    }

    total_images = 0

    for ds_name, cfg in RAW_DATASETS.items():
        raw_dir = cfg["raw_dir"]
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory missing: {raw_dir}")

        files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        print(f"Processing {ds_name} from {raw_dir} ({len(files)} files)...")

        hr_dir = CANONICAL_DIR / f"{ds_name}_HR"
        lr2_dir = CANONICAL_DIR / f"{ds_name}_LR_x2"
        lr4_dir = CANONICAL_DIR / f"{ds_name}_LR_x4"
        for d in [hr_dir, lr2_dir, lr4_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Deterministic T91 train/val split (80 train / 11 val, seed=42)
        if cfg["split"] == "train_val":
            rng = random.Random(42)
            shuffled = list(files)
            rng.shuffle(shuffled)
            val_set = {f.name for f in shuffled[:11]}
        else:
            val_set = set()

        ds_entries = []

        for f in files:
            img = Image.open(f).convert("RGB")
            orig_size = list(img.size)

            # 1. Modcrop so divisible by 4
            hr_img = modcrop(img, scale=4)
            hr_path = hr_dir / f"{f.stem}.png"
            hr_img.save(hr_path, format="PNG")
            hr_hash = sha256_file(hr_path)
            hr_size = list(hr_img.size)

            # 2. LR x2 via MATLAB-compatible bicubic
            hr_np = np.array(hr_img)
            lr2_np = imresize(hr_np, scale=0.5, antialiasing=True)
            lr2_img = Image.fromarray(lr2_np, mode="RGB")
            lr2_path = lr2_dir / f"{f.stem}.png"
            lr2_img.save(lr2_path, format="PNG")
            lr2_hash = sha256_file(lr2_path)
            lr2_size = list(lr2_img.size)

            # 3. LR x4 via MATLAB-compatible bicubic
            lr4_np = imresize(hr_np, scale=0.25, antialiasing=True)
            lr4_img = Image.fromarray(lr4_np, mode="RGB")
            lr4_path = lr4_dir / f"{f.stem}.png"
            lr4_img.save(lr4_path, format="PNG")
            lr4_hash = sha256_file(lr4_path)
            lr4_size = list(lr4_img.size)

            if cfg["split"] == "train_val":
                split_name = "val" if f.name in val_set else "train"
            else:
                split_name = cfg["split"]

            entry = {
                "dataset": ds_name,
                "image_id": f.stem,
                "split": split_name,
                "orig_size": orig_size,
                "hr_size": hr_size,
                "lr_x2_size": lr2_size,
                "lr_x4_size": lr4_size,
                "hr_path": str(hr_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "lr_x2_path": str(lr2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "lr_x4_path": str(lr4_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "hr_sha256": hr_hash,
                "lr_x2_sha256": lr2_hash,
                "lr_x4_sha256": lr4_hash,
                "verified": True,
            }
            ds_entries.append(entry)
            total_images += 1

        manifest["summary"][ds_name] = {
            "count": len(ds_entries),
            "splits": {
                s: len([e for e in ds_entries if e["split"] == s])
                for s in sorted(set(e["split"] for e in ds_entries))
            },
        }
        manifest["entries"].extend(ds_entries)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Processed {total_images} images.")
    print("Manifest summary:")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
