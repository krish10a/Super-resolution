"""
Verify alignment in clean benchmark directory.
"""
import os
from PIL import Image

BASE = r'C:\Users\ASUS\Desktop\cv\data\pairs_clean'

datasets = ['Set5', 'Set14', 'BSD100']
scales = [2, 4]

for dataset in datasets:
    hr_dir = os.path.join(BASE, f'{dataset}_HR')
    hr_files = sorted(os.listdir(hr_dir))
    print(f'\n{dataset}_HR: {len(hr_files)} files')

    for scale in scales:
        lr_dir = os.path.join(BASE, f'{dataset}_LR_x{scale}')
        lr_files = sorted(os.listdir(lr_dir))
        print(f'  {dataset}_LR_x{scale}: {len(lr_files)} files')

        hr_names = {os.path.splitext(f)[0] for f in hr_files}
        lr_names = {os.path.splitext(f)[0] for f in lr_files}
        if hr_names != lr_names:
            print(f'  NAME MISMATCH: HR-only={hr_names - lr_names}, LR-only={lr_names - hr_names}')

        mismatches = 0
        for f in hr_files:
            name = os.path.splitext(f)[0]
            lr_path = os.path.join(lr_dir, f)
            hr_path = os.path.join(hr_dir, f)
            if not os.path.exists(lr_path):
                continue
            with Image.open(hr_path) as hi, Image.open(lr_path) as li:
                hw, hh = hi.size
                lw, lh = li.size
                expected_w = hw // scale
                expected_h = hh // scale
                if lw != expected_w or lh != expected_h:
                    mismatches += 1
                    print(f'  ALIGN ISSUE {f}: HR={hi.size} LR={li.size} expected=({expected_w},{expected_h})')
        print(f'  Dimension mismatches: {mismatches}')
