"""
Independent bicubic baseline computation using BasicSR reference implementation.

Protocol:
1. Load canonical HR image from data/pairs_reference/{Dataset}_HR/.
2. During evaluation, mod-crop HR so dimensions are divisible by scale.
3. Upsample LR → HR size using imresize (antialiasing disabled).
4. Compare against mod-cropped HR using BasicSR metrics:
   - Y channel (BT.601), crop_border = scale
   - PSNR: 10*log10(255^2/mse) on [0,255] range
   - SSIM: 11x11 Gaussian sigma=1.5, c1=(0.01*255)^2, c2=(0.03*255)^2

Computes baselines using BOTH:
1. Our custom imresize (data/imresize_matlab.py) — verified equal to BasicSR
2. BasicSR reference imresize (outputs/evaluation/basicsr_reference.py)

Saves results to versioned output files.
Does NOT modify existing datasets or checkpoints.
"""

import os
import sys
import json
import hashlib
import csv
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'outputs', 'evaluation'))

from data.imresize_matlab import imresize_matlab as custom_imresize
from basicsr_reference import (
    basicSR_imresize as basicsr_imresize,
    basicSR_calculate_psnr as basicsr_psnr,
    basicSR_calculate_ssim as basicsr_ssim,
)

PAIRS_REFERENCE = os.path.join(BASE, 'data', 'pairs_reference')
OUT_DIR = os.path.join(BASE, 'outputs', 'evaluation', 'reference')
OUT_JSON = os.path.join(OUT_DIR, 'bicubic_baselines_reference.json')
OUT_CSV = os.path.join(OUT_DIR, 'bicubic_baselines_reference.csv')

os.makedirs(OUT_DIR, exist_ok=True)

TEST_SETS = ['Set5', 'Set14', 'BSD100']
SCALES = [2, 4]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_hr_file(hr_dir, name):
    for ext in ['.png', '.jpg', '.jpeg']:
        p = os.path.join(hr_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def modcrop(img, scale):
    """Crop HR image so dimensions are divisible by scale (standard SISR benchmark)."""
    h, w = img.shape[:2]
    new_h = h - h % scale
    new_w = w - w % scale
    if new_h != h or new_w != w:
        img = img[:new_h, :new_w, :]
    return img, (new_h, new_w)


def compute_baseline_for_pair(hr_path, lr_path, scale, upsample_impl):
    """Compute bicubic baseline PSNR/SSIM for one HR/LR pair."""
    hr = np.array(Image.open(hr_path).convert('RGB'))
    lr = np.array(Image.open(lr_path).convert('RGB'))

    # Mod-crop HR so dimensions are divisible by scale
    hr_cropped, crop_size = modcrop(hr, scale)

    # Upsample LR to HR-cropped size
    scale_up = float(hr_cropped.shape[0]) / lr.shape[0]
    if upsample_impl == 'custom':
        hr_up = custom_imresize(lr, scale=scale_up, antialiasing=False)
    else:
        hr_up = basicsr_imresize(lr, scale=scale_up, antialiasing=False)

    # Ensure exact size match
    h_min = min(hr_up.shape[0], hr_cropped.shape[0])
    w_min = min(hr_up.shape[1], hr_cropped.shape[1])
    hr_up = hr_up[:h_min, :w_min, :]
    hr_cropped = hr_cropped[:h_min, :w_min, :]

    # Compute metrics using BasicSR reference (Y channel, crop_border=scale)
    psnr = basicsr_psnr(hr_up, hr_cropped, crop_border=scale,
                        input_order='HWC', test_y_channel=True)
    ssim = basicsr_ssim(hr_up, hr_cropped, crop_border=scale,
                        input_order='HWC', test_y_channel=True)

    return psnr, ssim


def main():
    results = []

    for test_set in TEST_SETS:
        for scale in SCALES:
            lr_dir = os.path.join(PAIRS_REFERENCE, f'{test_set}_LR_x{scale}')
            hr_dir = os.path.join(PAIRS_REFERENCE, f'{test_set}_HR')
            if not os.path.isdir(lr_dir):
                print(f'Skipping {test_set} x{scale}: missing LR dir')
                continue

            lr_files = sorted([f for f in os.listdir(lr_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            set_result = {
                'dataset': test_set,
                'scale': scale,
                'downsample_impl': 'imresize_matlab (antialiasing=True, kernel_width=4)',
                'upsample_impls': ['custom', 'basicSR'],
                'metric_impl': 'basicSR_calculate_psnr/ssim (Y channel, BT.601, crop_border=scale)',
                'color_space': 'RGB',
                'quantisation': 'uint8 [0,255]',
                'modcrop': 'HR cropped to dimensions divisible by scale',
                'per_image': [],
            }

            for impl in ['custom', 'basicSR']:
                psnrs, ssims = [], []
                for f in lr_files:
                    name = os.path.splitext(f)[0]
                    lr_path = os.path.join(lr_dir, f)
                    hr_path = find_hr_file(hr_dir, name)
                    if hr_path is None:
                        continue

                    hr_hash = sha256(hr_path)
                    lr_hash = sha256(lr_path)

                    psnr, ssim = compute_baseline_for_pair(
                        hr_path, lr_path, scale, upsample_impl=impl,
                    )

                    psnrs.append(psnr)
                    ssims.append(ssim)

                    set_result['per_image'].append({
                        'image': name,
                        'psnr': psnr,
                        'ssim': ssim,
                        'hr_hash': hr_hash,
                        'lr_hash': lr_hash,
                        'upsample_impl': impl,
                    })

                avg_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
                avg_ssim = float(np.mean(ssims)) if ssims else float('nan')

                set_result[f'avg_psnr_{impl}'] = avg_psnr
                set_result[f'avg_ssim_{impl}'] = avg_ssim
                set_result[f'n_images_{impl}'] = len(psnrs)

                print(f'{test_set} x{scale} [{impl}]: PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} (n={len(psnrs)})')

            results.append(set_result)

    # Save JSON
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save CSV
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Scale', 'Upsample_Impl', 'N_Images', 'Avg_PSNR', 'Avg_SSIM'])
        for r in results:
            for impl in ['custom', 'basicSR']:
                writer.writerow([
                    r['dataset'], r['scale'], impl,
                    r[f'n_images_{impl}'],
                    f"{r[f'avg_psnr_{impl}']:.6f}",
                    f"{r[f'avg_ssim_{impl}']:.6f}",
                ])

    print(f'\nResults saved to {OUT_JSON} and {OUT_CSV}')
    return results


if __name__ == '__main__':
    main()
