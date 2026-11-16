"""
Recalculate bicubic baselines using MATLAB-compatible bicubic upsampling.

Loads LR images from data/pairs_matlab/, upscales using imresize_to_size
(exact HR size), and computes PSNR/SSIM against HR using metrics.psnr_ssim.

Compares new baselines against existing OpenCV/PIL baselines from
outputs/evaluation/evaluation_results.json.

Does NOT modify existing datasets or checkpoints.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.imresize_matlab import imresize_to_size
from metrics.psnr_ssim import compute_psnr_ssim

BASE = os.path.dirname(os.path.abspath(__file__))
PAIRS_MATLAB = os.path.join(BASE, 'data', 'pairs_matlab')
EXISTING_RESULTS = os.path.join(BASE, 'outputs', 'evaluation', 'evaluation_results.json')
OUT_DIR = os.path.join(BASE, 'outputs', 'evaluation')
OUT_FILE = os.path.join(OUT_DIR, 'matlab_bicubic_baselines.json')

TEST_SETS = ['Set5', 'Set14', 'BSD100']
SCALES = [2, 4]


def find_hr_file(hr_dir, name):
    for ext in ['.png', '.jpg', '.jpeg']:
        src = os.path.join(hr_dir, name + ext)
        if os.path.exists(src):
            return src
    return None


def compute_baseline_on_set(lr_dir, hr_dir, scale):
    """Compute bicubic baseline PSNR/SSIM on a dataset using MATLAB-compatible upsampling."""
    files = sorted([f for f in os.listdir(lr_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    results = []
    psnrs, ssims = [], []

    for f in files:
        name = os.path.splitext(f)[0]
        lr_path = os.path.join(lr_dir, f)
        hr_path = find_hr_file(hr_dir, name)
        if hr_path is None:
            continue

        lr = Image.open(lr_path).convert('RGB')
        hr = Image.open(hr_path).convert('RGB')

        # MATLAB-compatible bicubic upsampling to exact HR size
        lr_np = np.array(lr)
        lr_up_np = imresize_to_size(lr_np, (hr.size[0], hr.size[1]), antialias=False)
        lr_up = Image.fromarray(lr_up_np)

        lr_t = torch.from_numpy(np.array(lr_up).astype(np.float32) / 255.0).permute(2, 0, 1)
        hr_t = torch.from_numpy(np.array(hr).astype(np.float32) / 255.0).permute(2, 0, 1)

        p, s = compute_psnr_ssim(lr_t, hr_t, scale=scale)
        psnrs.append(p)
        ssims.append(s)
        results.append({
            'image': name,
            'psnr': p,
            'ssim': s,
        })

    avg_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
    avg_ssim = float(np.mean(ssims)) if ssims else float('nan')
    return avg_psnr, avg_ssim, len(results), results


def load_existing_baselines():
    """Load existing bicubic baselines from evaluation_results.json."""
    if not os.path.exists(EXISTING_RESULTS):
        return {}
    with open(EXISTING_RESULTS, 'r') as f:
        data = json.load(f)
    baselines = {}
    for exp, exp_data in data.items():
        for test_set, res in exp_data.get('results', {}).items():
            key = f'{test_set}_x{exp_data["config"]["scale"]}'
            if key not in baselines:
                baselines[key] = {
                    'bicubic_psnr': res.get('bicubic_psnr'),
                    'bicubic_ssim': res.get('bicubic_ssim'),
                }
    return baselines


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    new_baselines = {}
    all_results = {}

    for test_set in TEST_SETS:
        for scale in SCALES:
            lr_dir = os.path.join(PAIRS_MATLAB, f'{test_set}_LR_x{scale}')
            hr_dir = os.path.join(PAIRS_MATLAB, f'{test_set}_HR')
            if not os.path.isdir(lr_dir):
                print(f'Skipping {test_set} x{scale}: missing LR dir')
                continue

            avg_psnr, avg_ssim, n, per_image = compute_baseline_on_set(lr_dir, hr_dir, scale)
            key = f'{test_set}_x{scale}'
            new_baselines[key] = {
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'n_images': n,
            }
            all_results[key] = {
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'n_images': n,
                'per_image': per_image,
            }
            print(f'MATLAB bicubic {test_set} x{scale}: PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} (n={n})')

    # Compare with existing baselines
    existing = load_existing_baselines()
    print('\n--- Comparison: MATLAB vs Existing (OpenCV/PIL) bicubic baselines ---')
    comparison = {}
    for key in sorted(new_baselines.keys()):
        new_psnr = new_baselines[key]['avg_psnr']
        new_ssim = new_baselines[key]['avg_ssim']
        existing_psnr = existing.get(key, {}).get('bicubic_psnr', float('nan'))
        existing_ssim = existing.get(key, {}).get('bicubic_ssim', float('nan'))

        psnr_diff = new_psnr - existing_psnr if existing_psnr == existing_psnr else float('nan')
        ssim_diff = new_ssim - existing_ssim if existing_ssim == existing_ssim else float('nan')

        comparison[key] = {
            'matlab_psnr': new_psnr,
            'matlab_ssim': new_ssim,
            'existing_psnr': existing_psnr,
            'existing_ssim': existing_ssim,
            'psnr_diff': psnr_diff,
            'ssim_diff': ssim_diff,
        }
        print(f'  {key}: MATLAB PSNR={new_psnr:.4f} vs Existing PSNR={existing_psnr:.4f} (diff={psnr_diff:+.4f}) | '
              f'MATLAB SSIM={new_ssim:.4f} vs Existing SSIM={existing_ssim:.4f} (diff={ssim_diff:+.4f})')

    # Established SISR benchmark bicubic values (from literature, Y channel, shaved by scale)
    established = {
        'Set5_x2': {'psnr': 33.66, 'ssim': 0.9299},
        'Set14_x2': {'psnr': 30.24, 'ssim': 0.8688},
        'BSD100_x2': {'psnr': 29.56, 'ssim': 0.8431},
        'Set5_x4': {'psnr': 28.42, 'ssim': 0.8105},
        'Set14_x4': {'psnr': 26.00, 'ssim': 0.7027},
        'BSD100_x4': {'psnr': 25.96, 'ssim': 0.6675},
    }

    print('\n--- Comparison: MATLAB bicubic vs Established benchmark values ---')
    for key in sorted(new_baselines.keys()):
        new_psnr = new_baselines[key]['avg_psnr']
        new_ssim = new_baselines[key]['avg_ssim']
        est = established.get(key, {})
        est_psnr = est.get('psnr', float('nan'))
        est_ssim = est.get('ssim', float('nan'))

        psnr_diff = new_psnr - est_psnr if est_psnr == est_psnr else float('nan')
        ssim_diff = new_ssim - est_ssim if est_ssim == est_ssim else float('nan')

        print(f'  {key}: MATLAB PSNR={new_psnr:.4f} vs Established PSNR={est_psnr:.2f} (diff={psnr_diff:+.4f}) | '
              f'MATLAB SSIM={new_ssim:.4f} vs Established SSIM={est_ssim:.4f} (diff={ssim_diff:+.4f})')

    # Save results
    output = {
        'new_baselines': new_baselines,
        'existing_baselines': existing,
        'comparison': comparison,
        'all_results': all_results,
    }
    with open(OUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nResults saved to {OUT_FILE}')


if __name__ == '__main__':
    main()
