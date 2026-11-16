"""
Metric parity test: project's metrics/psnr_ssim.py vs BasicSR reference.

Compares PSNR and SSIM on the same image pairs using both implementations,
with conventions aligned (Y channel, crop_border=scale, [0,255] range).
"""

import os
import sys
import json
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'outputs', 'evaluation'))

from metrics.psnr_ssim import compute_psnr_ssim as project_compute_psnr_ssim
from basicsr_reference import (
    basicSR_calculate_psnr as basicsr_psnr,
    basicSR_calculate_ssim as basicsr_ssim,
)

PAIRS_REFERENCE = os.path.join(BASE, 'data', 'pairs_reference')
OUT_DIR = os.path.join(BASE, 'outputs', 'evaluation', 'reference')
OUT_FILE = os.path.join(OUT_DIR, 'metric_parity_report.json')

os.makedirs(OUT_DIR, exist_ok=True)

TEST_SETS = ['Set5', 'Set14', 'BSD100']
SCALES = [2, 4]


def sha256(path):
    h = __import__('hashlib').sha256()
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


def run_metric_parity():
    """Compare project metrics vs BasicSR metrics on bicubic-upsampled pairs."""
    results = []

    for test_set in TEST_SETS:
        for scale in SCALES:
            lr_dir = os.path.join(PAIRS_REFERENCE, f'{test_set}_LR_x{scale}')
            hr_dir = os.path.join(PAIRS_REFERENCE, f'{test_set}_HR')
            if not os.path.isdir(lr_dir):
                continue

            lr_files = sorted([f for f in os.listdir(lr_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            set_result = {
                'dataset': test_set,
                'scale': scale,
                'project_impl': 'metrics/psnr_ssim.py (BT.601 Y, shave=scale, skimage SSIM)',
                'basicSR_impl': 'basicsr_reference.py (BT.601 Y, crop=scale, cv2 SSIM)',
                'per_image': [],
            }

            for f in lr_files[:5]:  # Test first 5 images per set for speed
                name = os.path.splitext(f)[0]
                lr_path = os.path.join(lr_dir, f)
                hr_path = find_hr_file(hr_dir, name)
                if hr_path is None:
                    continue

                lr = np.array(Image.open(lr_path).convert('RGB'))
                hr = np.array(Image.open(hr_path).convert('RGB'))

                # Upsample LR using verified imresize (BasicSR-port)
                from data.imresize_matlab import imresize_matlab as imresize
                hr_up = imresize(lr, scale=float(hr.shape[0]) / lr.shape[0], antialiasing=False)

                # Mod-crop HR to match
                h_min = min(hr_up.shape[0], hr.shape[0])
                w_min = min(hr_up.shape[1], hr.shape[1])
                hr_up = hr_up[:h_min, :w_min, :]
                hr_crop = hr[:h_min, :w_min, :]

                # Crop border by scale
                if scale > 0:
                    hr_up_c = hr_up[scale:-scale, scale:-scale, :] if h_min > 2*scale else hr_up
                    hr_crop_c = hr_crop[scale:-scale, scale:-scale, :] if h_min > 2*scale else hr_crop
                else:
                    hr_up_c = hr_up
                    hr_crop_c = hr_crop

                # Project metrics (on [0,1] tensors)
                hr_up_t = __import__('torch').from_numpy(
                    hr_up_c.astype(np.float32) / 255.0).permute(2, 0, 1)
                hr_crop_t = __import__('torch').from_numpy(
                    hr_crop_c.astype(np.float32) / 255.0).permute(2, 0, 1)
                p_proj, s_proj = project_compute_psnr_ssim(
                    hr_up_t.squeeze(0), hr_crop_t.squeeze(0), scale=scale)

                # BasicSR metrics (on [0,255] uint8)
                p_basic = basicsr_psnr(hr_up_c, hr_crop_c, crop_border=scale,
                                       input_order='HWC', test_y_channel=True)
                s_basic = basicsr_ssim(hr_up_c, hr_crop_c, crop_border=scale,
                                       input_order='HWC', test_y_channel=True)

                psnr_diff = abs(p_proj - p_basic)
                ssim_diff = abs(s_proj - s_basic)

                set_result['per_image'].append({
                    'image': name,
                    'project_psnr': p_proj,
                    'project_ssim': s_proj,
                    'basicSR_psnr': p_basic,
                    'basicSR_ssim': s_basic,
                    'psnr_diff': psnr_diff,
                    'ssim_diff': ssim_diff,
                    'hr_hash': sha256(hr_path),
                    'lr_hash': sha256(lr_path),
                })

            results.append(set_result)

    # Compute summary
    all_psnr_diffs = []
    all_ssim_diffs = []
    for r in results:
        for img in r['per_image']:
            all_psnr_diffs.append(img['psnr_diff'])
            all_ssim_diffs.append(img['ssim_diff'])

    summary = {
        'total_comparisons': len(all_psnr_diffs),
        'max_psnr_diff': float(max(all_psnr_diffs)) if all_psnr_diffs else 0.0,
        'mean_psnr_diff': float(np.mean(all_psnr_diffs)) if all_psnr_diffs else 0.0,
        'max_ssim_diff': float(max(all_ssim_diffs)) if all_ssim_diffs else 0.0,
        'mean_ssim_diff': float(np.mean(all_ssim_diffs)) if all_ssim_diffs else 0.0,
    }

    report = {
        'metric_parity_test': 'project vs BasicSR',
        'summary': summary,
        'results': results,
    }

    with open(OUT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Metric parity test complete: {summary['total_comparisons']} comparisons")
    print(f"Max PSNR diff: {summary['max_psnr_diff']:.6f}")
    print(f"Mean PSNR diff: {summary['mean_psnr_diff']:.6f}")
    print(f"Max SSIM diff: {summary['max_ssim_diff']:.6f}")
    print(f"Mean SSIM diff: {summary['mean_ssim_diff']:.6f}")
    print(f"Report saved to {OUT_FILE}")

    return report


if __name__ == '__main__':
    run_metric_parity()
