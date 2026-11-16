"""
Generate verified T91 LR images using MATLAB-compatible bicubic imresize.

Creates data/pairs_verified/T91_LR_x2/ and T91_LR_x4/ from data/pairs/T91_HR/
using the verified imresize (antialiasing=True, kernel_width=4).

Does NOT overwrite existing datasets or checkpoints.
"""

import os
import sys
import hashlib
import json
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from data.imresize_matlab import imresize_matlab as imresize

T91_HR = os.path.join(BASE, 'data', 'pairs', 'T91_HR')
OUT_ROOT = os.path.join(BASE, 'data', 'pairs_verified')

SCALES = [2, 4]

os.makedirs(OUT_ROOT, exist_ok=True)

hr_files = sorted([f for f in os.listdir(T91_HR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {len(hr_files)} T91 HR images")

manifest = {
    'dataset': 'pairs_verified_T91',
    'source': 'data/pairs/T91_HR',
    'downsampling': 'imresize_matlab (antialiasing=True, kernel_width=4, cubic a=-0.5)',
    'modcrop': 'HR cropped to dimensions divisible by scale',
    'color_space': 'RGB',
    'dtype': 'uint8 [0,255]',
    'images': [],
}

for scale in SCALES:
    out_lr = os.path.join(OUT_ROOT, f'T91_LR_x{scale}')
    os.makedirs(out_lr, exist_ok=True)

for f in hr_files:
    name = os.path.splitext(f)[0]
    hr_path = os.path.join(T91_HR, f)
    ext = os.path.splitext(f)[1]

    img = np.array(Image.open(hr_path).convert('RGB'))
    h, w = img.shape[:2]

    for scale in SCALES:
        # Mod-crop HR so dimensions divisible by scale
        new_h = h - h % scale
        new_w = w - w % scale
        img_cropped = img[:new_h, :new_w, :]

        # Downsample using verified imresize with antialiasing
        lr = imresize(img_cropped, 1.0 / scale, antialiasing=True)

        out_lr = os.path.join(OUT_ROOT, f'T91_LR_x{scale}')
        lr_path = os.path.join(out_lr, name + ext)
        Image.fromarray(lr).save(lr_path)
        lr_hash = hashlib.sha256(open(lr_path, 'rb').read()).hexdigest()

        manifest['images'].append({
            'name': name,
            'scale': scale,
            'ext': ext,
            'hr_hash': hashlib.sha256(open(hr_path, 'rb').read()).hexdigest(),
            'lr_hash': lr_hash,
            'hr_size': [h, w],
            'lr_size': [lr.shape[0], lr.shape[1]],
        })

manifest_path = os.path.join(OUT_ROOT, 'manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2, default=str)

print(f'\nVerified T91 LR dataset created at: {OUT_ROOT}')
print(f'Manifest saved to: {manifest_path}')
