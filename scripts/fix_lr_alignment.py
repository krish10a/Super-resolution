"""Regenerate LR pairs from HR with consistent floor-division downsampling.

Fixes T91 x2 (76 mismatches) and Set14 x2 (3 mismatches) where LR dimensions
were off-by-one vs HR//scale. Regenerates ALL x2 LR images from HR so every
set/scale satisfies LR == HR // scale exactly.

Uses cv2.INTER_CUBIC to match data/degradation.py downscale_bicubic.
"""
import os
import cv2

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data', 'pairs')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'pairs_fixed')
SETS = ['Set5', 'Set14', 'BSD100', 'T91']
SCALES = [2, 4]


def regenerate_lr(hr_path, scale):
    img = cv2.imread(hr_path)  # BGR
    h, w = img.shape[:2]
    lr = cv2.resize(img, (max(1, w // scale), max(1, h // scale)),
                    interpolation=cv2.INTER_CUBIC)
    return lr


fixed = 0
checked = 0
for ds in SETS:
    for scale in SCALES:
        hr_dir = os.path.join(BASE, f'{ds}_HR')
        if not os.path.isdir(hr_dir):
            continue
        for f in sorted(os.listdir(hr_dir)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            checked += 1
            hr_path = os.path.join(hr_dir, f)
            img = cv2.imread(hr_path)
            h, w = img.shape[:2]
            expected = (max(1, w // scale), max(1, h // scale))
            out_dir = os.path.join(OUT, f'{ds}_LR_x{scale}')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f)
            # Only write if missing or dimensionally wrong
            existing = cv2.imread(out_path)
            if existing is None or tuple(existing.shape[:2]) != expected:
                lr = regenerate_lr(hr_path, scale)
                cv2.imwrite(out_path, lr)
                fixed += 1
                if fixed <= 15:
                    print(f'WRITTEN {ds}_LR_x{scale}/{f}: {expected}')

print(f'\nChecked {checked} LR images, wrote {fixed} fixed files to {OUT}/')

