"""
Numerical parity test: our custom imresize vs BasicSR reference implementation.

Compares actual pixel outputs for various images, scales, and dtypes.
Generates a machine-readable parity report.
"""

import os
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.imresize_matlab import imresize_matlab as custom_imresize
from outputs.evaluation.basicsr_reference import basicSR_imresize as basicsr_imresize

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS_MATLAB = os.path.join(BASE, 'data', 'pairs_matlab')
OUT_DIR = os.path.join(BASE, 'outputs', 'evaluation')
OUT_FILE = os.path.join(OUT_DIR, 'imresize_parity_report.json')

os.makedirs(OUT_DIR, exist_ok=True)


def load_test_images():
    """Load test images: canonical benchmarks + synthetic patterns."""
    images = {}

    # Canonical benchmark images
    for dataset, names in {
        'Set5': [f'Set5_{i:02d}' for i in range(5)],
        'Set14': [f'Set14_{i:02d}' for i in range(14)],
        'BSD100': [f'img_{i:03d}' for i in range(1, 101)],
    }.items():
        for name in names:
            for ext in ['.png', '.jpg', '.jpeg']:
                p = os.path.join(PAIRS_MATLAB, f'{dataset}_HR', name + ext)
                if os.path.exists(p):
                    images[f'{dataset}/{name}'] = np.array(Image.open(p).convert('RGB'))
                    break

    # Synthetic patterns
    images['synthetic/constant_red'] = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
    images['synthetic/constant_white'] = np.full((64, 64, 3), [255, 255, 255], dtype=np.uint8)
    images['synthetic/constant_black'] = np.zeros((64, 64, 3), dtype=np.uint8)

    # Impulse image (single bright pixel)
    impulse = np.zeros((32, 32, 3), dtype=np.uint8)
    impulse[16, 16, :] = 255
    images['synthetic/impulse'] = impulse

    # Checkerboard
    checker = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(32):
        for j in range(32):
            if (i // 4 + j // 4) % 2 == 0:
                checker[i, j, :] = 255
    images['synthetic/checkerboard'] = checker

    # Sharp edge pattern
    edge = np.zeros((32, 32, 3), dtype=np.uint8)
    edge[:, 16:, :] = 255
    images['synthetic/sharp_edge'] = edge

    # Odd-dimension image
    odd = np.array(Image.open(os.path.join(PAIRS_MATLAB, 'Set5_HR', 'Set5_00.png')).convert('RGB'))
    images['synthetic/odd_dim'] = odd[:100, :100, :]  # 100x100 (even) - use odd crop
    images['synthetic/odd_dim'] = odd[:99, :99, :]  # 99x99 odd

    return images


def compute_metrics(a, b):
    """Compute numerical comparison metrics between two arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    diff = np.abs(a - b)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    mse = float(np.mean((a - b) ** 2))
    rmse = float(np.sqrt(mse))

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(b), 1e-8)
    rel_err = float(np.mean(diff / denom))

    nan_count = int(np.sum(np.isnan(a)) + np.sum(np.isnan(b)))
    inf_count = int(np.sum(np.isinf(a)) + np.sum(np.isinf(b)))

    return {
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'mse': mse,
        'rmse': rmse,
        'mean_rel_err': rel_err,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'shape_a': list(a.shape),
        'shape_b': list(b.shape),
    }


def run_parity_tests():
    """Run all parity tests and generate report."""
    images = load_test_images()
    scales = [0.25, 0.5, 2.0, 4.0]  # down x4, down x2, up x2, up x4
    dtypes = [np.uint8]

    report = {
        'test_type': 'imresize_parity',
        'reference': 'BasicSR imresize (matlab_functions.py)',
        'custom': 'data/imresize_matlab.py',
        'results': [],
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'max_diff_overall': 0.0,
        },
    }

    tolerance = {
        np.uint8: {'max_abs': 3.0, 'mean_abs': 1.0},  # uint8 rounding tolerance
    }

    for img_name, img in images.items():
        for scale in scales:
            for dt in dtypes:
                img_dt = img.astype(dt)

                # Custom implementation
                custom_out = custom_imresize(img_dt, scale, antialiasing=(scale < 1))

                # BasicSR reference
                basicsr_out = basicsr_imresize(img_dt, scale, antialiasing=(scale < 1))

                metrics = compute_metrics(custom_out, basicsr_out)
                metrics['image'] = img_name
                metrics['scale'] = scale
                metrics['dtype'] = str(dt)
                metrics['custom_shape'] = list(custom_out.shape)
                metrics['basicsr_shape'] = list(basicsr_out.shape)

                # Pass/fail
                dt_tol = tolerance.get(dt, tolerance[np.uint8])
                passed = (
                    metrics['max_abs_diff'] <= dt_tol['max_abs'] and
                    metrics['mean_abs_diff'] <= dt_tol['mean_abs'] and
                    metrics['nan_count'] == 0 and
                    metrics['inf_count'] == 0 and
                    metrics['shape_a'] == metrics['shape_b']
                )
                metrics['passed'] = passed

                report['results'].append(metrics)
                report['summary']['total_tests'] += 1
                if passed:
                    report['summary']['passed'] += 1
                else:
                    report['summary']['failed'] += 1
                report['summary']['max_diff_overall'] = max(
                    report['summary']['max_diff_overall'], metrics['max_abs_diff']
                )

    with open(OUT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Parity tests complete: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
    print(f"Max abs diff overall: {report['summary']['max_diff_overall']:.6f}")
    print(f"Report saved to {OUT_FILE}")

    # Print failures
    for r in report['results']:
        if not r['passed']:
            print(f"  FAIL: {r['image']} scale={r['scale']} "
                  f"max_diff={r['max_abs_diff']:.4f} mean_diff={r['mean_abs_diff']:.4f} "
                  f"shapes={r['custom_shape']} vs {r['basicsr_shape']}")

    return report


if __name__ == '__main__':
    run_parity_tests()
