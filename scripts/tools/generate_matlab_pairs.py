"""
Generate corrected LR/HR benchmark pairs using MATLAB-compatible bicubic downsampling.

Creates a new dataset directory (data/pairs_matlab/) with:
- Set5 (5 images), Set14 (14 images), BSD100 (100 images) canonical HR images
- LR x2 and x4 generated via MATLAB-compatible bicubic imresize with antialiasing

Does NOT overwrite existing datasets or checkpoints.
"""

import os
import shutil
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.imresize_matlab import imresize_matlab, imresize_to_size

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
PAIRS_FIXED = os.path.join(DATA_DIR, 'pairs_fixed')
PAIRS_MATLAB = os.path.join(DATA_DIR, 'pairs_matlab')

# Canonical benchmark image names (matching create_clean_benchmarks.py)
CANONICAL = {
    'Set5': [f'Set5_{i:02d}' for i in range(5)],      # Set5_00 to Set5_04
    'Set14': [f'Set14_{i:02d}' for i in range(14)],    # Set14_00 to Set14_13
    'BSD100': [f'img_{i:03d}' for i in range(1, 101)], # img_001 to img_100
}

SCALES = [2, 4]


def find_hr_file(src_hr, name):
    """Find HR file with given basename, trying common extensions."""
    for ext in ['.png', '.jpg', '.jpeg']:
        src = os.path.join(src_hr, name + ext)
        if os.path.exists(src):
            return src
    return None


def generate_pairs():
    """Generate corrected LR/HR pairs using MATLAB-compatible bicubic."""
    os.makedirs(PAIRS_MATLAB, exist_ok=True)

    for dataset, names in CANONICAL.items():
        print(f'\nProcessing {dataset} ({len(names)} canonical images)...')

        src_hr = os.path.join(PAIRS_FIXED, f'{dataset}_HR')
        dst_hr = os.path.join(PAIRS_MATLAB, f'{dataset}_HR')
        os.makedirs(dst_hr, exist_ok=True)

        # Copy HR images as-is
        hr_count = 0
        for name in names:
            src = find_hr_file(src_hr, name)
            if src is None:
                print(f'  WARNING: {name} not found in {src_hr}')
                continue
            ext = os.path.splitext(src)[1]
            dst = os.path.join(dst_hr, name + ext)
            shutil.copy2(src, dst)
            hr_count += 1
        print(f'  {dataset}_HR: {hr_count} images copied')

        # Generate LR x2 and x4 via MATLAB-compatible bicubic downsampling
        for scale in SCALES:
            src_lr = os.path.join(PAIRS_FIXED, f'{dataset}_LR_x{scale}')
            dst_lr = os.path.join(PAIRS_MATLAB, f'{dataset}_LR_x{scale}')
            os.makedirs(dst_lr, exist_ok=True)

            lr_count = 0
            for name in names:
                # Find the HR source (regardless of extension)
                src_hr_path = find_hr_file(src_hr, name)
                if src_hr_path is None:
                    continue

                # Load HR image
                img = np.array(Image.open(src_hr_path).convert('RGB'))

                # Downscale using MATLAB-compatible bicubic with antialiasing.
                # For benchmark pairs, use exact division (HR // scale) to match
                # standard SISR benchmark convention.
                h, w = img.shape[:2]
                target_size = (w // scale, h // scale)
                lr_np = img.astype(np.float64)
                lr_img = imresize_to_size(lr_np, target_size, antialias=True)

                # Save with the same extension as the HR source
                ext = os.path.splitext(src_hr_path)[1]
                dst = os.path.join(dst_lr, name + ext)
                Image.fromarray(lr_img).save(dst)
                lr_count += 1

            print(f'  {dataset}_LR_x{scale}: {lr_count} images generated')

    print(f'\nCorrected dataset created at: {PAIRS_MATLAB}')


if __name__ == '__main__':
    generate_pairs()
