"""
Generate definitive corrected benchmark dataset using verified BasicSR-port imresize.

Creates data/pairs_reference/ with:
- Set5 (5), Set14 (14), BSD100 (100) canonical HR images
- LR x2 and x4 generated via BasicSR-port MATLAB-compatible bicubic (antialiasing)

Directory structure matches evaluate_all.py expectations:
  pairs_reference/{Set5,Set14,BSD100}_HR/
  pairs_reference/{Set5,Set14,BSD100}_LR_x2/
  pairs_reference/{Set5,Set14,BSD100}_LR_x4/

Does NOT overwrite existing datasets or checkpoints.

Protocol:
1. Load canonical HR image from data/pairs_fixed/
2. Downsample HR → LR using verified imresize (BasicSR-port, antialiasing enabled)
3. During evaluation, HR is mod-cropped on the fly to dimensions divisible by scale
"""

import os
import sys
import shutil
import hashlib
import json
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from data.imresize_matlab import imresize_matlab as imresize

PAIRS_FIXED = os.path.join(BASE, 'data', 'pairs_fixed')
PAIRS_REFERENCE = os.path.join(BASE, 'data', 'pairs_reference')

CANONICAL = {
    'Set5': [f'Set5_{i:02d}' for i in range(5)],
    'Set14': [f'Set14_{i:02d}' for i in range(14)],
    'BSD100': [f'img_{i:03d}' for i in range(1, 101)],
}

SCALES = [2, 4]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_hr_file(src_hr, name):
    for ext in ['.png', '.jpg', '.jpeg']:
        p = os.path.join(src_hr, name + ext)
        if os.path.exists(p):
            return p
    return None


def generate_pairs():
    os.makedirs(PAIRS_REFERENCE, exist_ok=True)

    manifest = {
        'dataset': 'pairs_reference',
        'protocol': 'BasicSR-port MATLAB-compatible bicubic',
        'downsampling': 'imresize_matlab (antialiasing=True, kernel_width=4, cubic a=-0.5)',
        'upsampling': 'imresize_matlab (antialiasing=False)',
        'modcrop': 'Applied on-the-fly during evaluation (HR cropped to dimensions divisible by scale)',
        'color_space': 'RGB',
        'dtype': 'uint8 [0,255]',
        'source': 'data/pairs_fixed/ (canonical images only)',
        'images': [],
    }

    for dataset, names in CANONICAL.items():
        print(f'\nProcessing {dataset} ({len(names)} canonical images)...')

        src_hr = os.path.join(PAIRS_FIXED, f'{dataset}_HR')

        dst_hr = os.path.join(PAIRS_REFERENCE, f'{dataset}_HR')
        os.makedirs(dst_hr, exist_ok=True)

        for scale in SCALES:
            dst_lr = os.path.join(PAIRS_REFERENCE, f'{dataset}_LR_x{scale}')
            os.makedirs(dst_lr, exist_ok=True)

        hr_count = 0
        for name in names:
            src = find_hr_file(src_hr, name)
            if src is None:
                print(f'  WARNING: {name} not found in {src_hr}')
                continue

            img = np.array(Image.open(src).convert('RGB'))
            ext = os.path.splitext(src)[1]
            src_hash = sha256(src)

            # Copy full HR image
            hr_dst_path = os.path.join(dst_hr, name + ext)
            shutil.copy2(src, hr_dst_path)
            hr_hash = sha256(hr_dst_path)

            for scale in SCALES:
                dst_lr = os.path.join(PAIRS_REFERENCE, f'{dataset}_LR_x{scale}')

                # Mod-crop HR so dimensions are divisible by scale (standard benchmark protocol)
                h, w = img.shape[:2]
                new_h = h - h % scale
                new_w = w - w % scale
                img_cropped = img[:new_h, :new_w, :]

                # Downsample mod-cropped HR to LR using verified imresize
                lr = imresize(img_cropped, 1.0 / scale, antialiasing=True)
                lr_dst_path = os.path.join(dst_lr, name + ext)
                Image.fromarray(lr).save(lr_dst_path)
                lr_hash = sha256(lr_dst_path)

                manifest['images'].append({
                    'name': name,
                    'dataset': dataset,
                    'scale': scale,
                    'ext': ext,
                    'src_hash': src_hash,
                    'hr_hash': hr_hash,
                    'lr_hash': lr_hash,
                    'hr_size': list(img.shape[:2]),
                    'lr_size': list(lr.shape[:2]),
                })

            hr_count += 1

        print(f'  {dataset}: {hr_count} HR images processed')

    manifest_path = os.path.join(PAIRS_REFERENCE, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f'\nReference dataset created at: {PAIRS_REFERENCE}')
    print(f'Manifest saved to: {manifest_path}')


if __name__ == '__main__':
    generate_pairs()
