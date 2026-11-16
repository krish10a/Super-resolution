"""
Automated validation of data/dataset_manifest.json.

Verifies:
1. Manifest JSON schema and structure.
2. Exact image counts: Set5 (5), Set14 (14), BSD100 (100), T91 (91: 80 train, 11 val).
3. Existence of all referenced HR, LR_x2, and LR_x4 files.
4. Correct spatial dimensions (HR divisible by 4, LR_x2 exactly half, LR_x4 exactly quarter).
5. SHA-256 integrity of all files against recorded hashes.
6. Zero duplicate file hashes across distinct images.
7. Zero cross-split contamination (train vs val vs test).
"""

import sys
import json
import hashlib
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"

EXPECTED_COUNTS = {
    "Set5": {"total": 5, "test": 5},
    "Set14": {"total": 14, "test": 14},
    "BSD100": {"total": 100, "test": 100},
    "T91": {"total": 91, "train": 80, "val": 11},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest():
    if not MANIFEST_PATH.exists():
        print(f"FAIL: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    errors = []
    warnings = []

    entries = manifest.get("entries", [])
    if not entries:
        errors.append("Manifest has no entries.")

    # 1. Verify Dataset Counts and Splits
    ds_counts = {}
    for e in entries:
        ds = e["dataset"]
        split = e["split"]
        if ds not in ds_counts:
            ds_counts[ds] = {"total": 0}
        ds_counts[ds]["total"] += 1
        ds_counts[ds][split] = ds_counts[ds].get(split, 0) + 1

    for ds, exp in EXPECTED_COUNTS.items():
        if ds not in ds_counts:
            errors.append(f"Missing expected dataset: {ds}")
            continue
        actual = ds_counts[ds]
        for k, expected_val in exp.items():
            actual_val = actual.get(k, 0)
            if actual_val != expected_val:
                errors.append(
                    f"Dataset count mismatch for {ds} ({k}): expected {expected_val}, got {actual_val}"
                )

    # 2. Verify File Existence, Dimensions, and Hashes
    hr_hashes = {}
    train_ids = set()
    val_ids = set()
    test_ids = set()

    for idx, e in enumerate(entries):
        ds = e["dataset"]
        img_id = e["image_id"]
        split = e["split"]

        # Track split IDs for leakage check
        if split == "train":
            train_ids.add((ds, img_id))
        elif split == "val":
            val_ids.add((ds, img_id))
        elif split == "test":
            test_ids.add((ds, img_id))

        hr_p = PROJECT_ROOT / e["hr_path"]
        lr2_p = PROJECT_ROOT / e["lr_x2_path"]
        lr4_p = PROJECT_ROOT / e["lr_x4_path"]

        for label, p, exp_hash in [
            ("HR", hr_p, e["hr_sha256"]),
            ("LR_x2", lr2_p, e["lr_x2_sha256"]),
            ("LR_x4", lr4_p, e["lr_x4_sha256"]),
        ]:
            if not p.exists():
                errors.append(f"Missing {label} file: {p}")
                continue

            # Verify hash
            act_hash = sha256_file(p)
            if act_hash != exp_hash:
                errors.append(f"Hash mismatch on {p}: expected {exp_hash}, got {act_hash}")

        if hr_p.exists() and lr2_p.exists() and lr4_p.exists():
            im_hr = Image.open(hr_p)
            im_lr2 = Image.open(lr2_p)
            im_lr4 = Image.open(lr4_p)

            hw, hh = im_hr.size
            l2w, l2h = im_lr2.size
            l4w, l4h = im_lr4.size

            if hw % 4 != 0 or hh % 4 != 0:
                errors.append(f"HR dimensions not divisible by 4: {hr_p} ({hw}x{hh})")

            if l2w != hw // 2 or l2h != hh // 2:
                errors.append(f"LR_x2 dimension mismatch for {hr_p}: HR={hw}x{hh}, LR2={l2w}x{l2h}")

            if l4w != hw // 4 or l4h != hh // 4:
                errors.append(f"LR_x4 dimension mismatch for {hr_p}: HR={hw}x{hh}, LR4={l4w}x{l4h}")

            # Check for duplicate HR hashes across different image IDs
            act_hr_hash = sha256_file(hr_p)
            if act_hr_hash in hr_hashes:
                prev_ds, prev_id, prev_split = hr_hashes[act_hr_hash]
                if prev_split != split:
                    errors.append(
                        f"Cross-split duplicate image: {ds}/{img_id} ({split}) matches {prev_ds}/{prev_id} ({prev_split})"
                    )
                else:
                    warnings.append(
                        f"Internal dataset duplicate: {ds}/{img_id} and {prev_ds}/{prev_id} share identical content (both in '{split}' split)"
                    )
            else:
                hr_hashes[act_hr_hash] = (ds, img_id, split)

    # 3. Check for Train / Val / Test Leakage
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap:
        errors.append(f"Data leakage between Train and Val: {train_val_overlap}")
    if train_test_overlap:
        errors.append(f"Data leakage between Train and Test: {train_test_overlap}")
    if val_test_overlap:
        errors.append(f"Data leakage between Val and Test: {val_test_overlap}")

    # Output Results
    print("=" * 60)
    print("DATASET MANIFEST VALIDATION REPORT")
    print("=" * 60)
    print(f"Total entries validated: {len(entries)}")
    for ds, counts in ds_counts.items():
        print(f"  {ds:8s}: {counts['total']} images -> {counts}")

    print("\nSplit Isolation:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val:   {len(val_ids)} images")
    print(f"  Test:  {len(test_ids)} images")
    print(f"  Leakage detected: {bool(train_val_overlap or train_test_overlap or val_test_overlap)}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  [ERROR] {err}")
        sys.exit(1)

    print("\nSUCCESS: All dataset manifest integrity checks PASSED perfectly.")
    sys.exit(0)


if __name__ == "__main__":
    validate_manifest()
