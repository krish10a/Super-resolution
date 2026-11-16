"""Verify SISR dataset integrity: counts, dimensions, alignment, readability."""
import os
from PIL import Image
import numpy as np

BASE = 'data/pairs'
SETS = ['Set5', 'Set14', 'BSD100', 'T91']
SCALES = [2, 4]


def count_images(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def read_image(path):
    try:
        img = Image.open(path).convert('RGB')
        arr = np.array(img)
        return img.size, arr.shape, None
    except Exception as e:
        return None, None, str(e)


def verify_set(name):
    hr_dir = os.path.join(BASE, f'{name}_HR')
    print(f"\n=== {name} ===")
    if not os.path.isdir(hr_dir):
        print(f"  MISSING HR dir: {hr_dir}")
        return False

    hr_files = sorted([f for f in os.listdir(hr_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  HR images: {len(hr_files)}")
    if len(hr_files) == 0:
        print("  ERROR: no HR images found")
        return False

    # Check readability + dimensions of HR
    dims = set()
    bad = 0
    for f in hr_files:
        size, shape, err = read_image(os.path.join(hr_dir, f))
        if err:
            print(f"  UNREADABLE HR: {f} -> {err}")
            bad += 1
        else:
            dims.add(size)
    print(f"  HR dims seen: {sorted(dims)}")
    print(f"  HR unreadable: {bad}")

    ok = True
    for scale in SCALES:
        lr_dir = os.path.join(BASE, f'{name}_LR_x{scale}')
        print(f"\n  --- x{scale} ---")
        if not os.path.isdir(lr_dir):
            print(f"    MISSING LR dir: {lr_dir}")
            ok = False
            continue

        lr_files = sorted([f for f in os.listdir(lr_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"    LR images: {len(lr_files)}")

        # Count match
        if len(lr_files) != len(hr_files):
            print(f"    COUNT MISMATCH: HR={len(hr_files)} LR={len(lr_files)}")
            ok = False

        # Alignment: same basenames
        hr_basenames = {os.path.splitext(f)[0] for f in hr_files}
        lr_basenames = {os.path.splitext(f)[0] for f in lr_files}
        missing_in_lr = hr_basenames - lr_basenames
        extra_in_lr = lr_basenames - hr_basenames
        if missing_in_lr:
            print(f"    LR missing for HR: {sorted(missing_in_lr)[:5]}")
            ok = False
        if extra_in_lr:
            print(f"    LR extra (no HR): {sorted(extra_in_lr)[:5]}")
            ok = False

        # Dimension check: LR should be HR/scale
        dim_mismatch = 0
        unreadable = 0
        for f in hr_files:
            base = os.path.splitext(f)[0]
            lr_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                cand = os.path.join(lr_dir, base + ext)
                if os.path.exists(cand):
                    lr_path = cand
                    break
            if lr_path is None:
                continue
            hr_size, _, hr_err = read_image(os.path.join(hr_dir, f))
            lr_size, _, lr_err = read_image(lr_path)
            if hr_err or lr_err:
                unreadable += 1
                continue
            if hr_size and lr_size:
                expected_w = hr_size[0] // scale
                expected_h = hr_size[1] // scale
                if lr_size[0] != expected_w or lr_size[1] != expected_h:
                    dim_mismatch += 1
                    if dim_mismatch <= 3:
                        print(f"    DIM MISMATCH: {f} HR={hr_size} LR={lr_size} expected~({expected_w},{expected_h})")
        print(f"    LR dim mismatches (vs HR//{scale}): {dim_mismatch}")
        print(f"    LR unreadable: {unreadable}")
        if dim_mismatch > 0 or unreadable > 0:
            ok = False

    return ok


all_ok = True
for s in SETS:
    if not verify_set(s):
        all_ok = False

print(f"\n{'='*50}")
print(f"OVERALL: {'ALL DATASETS VALID' if all_ok else 'SOME DATASETS INVALID'}")
print(f"{'='*50}")
