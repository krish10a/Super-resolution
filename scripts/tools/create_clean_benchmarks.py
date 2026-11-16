"""
Create clean benchmark dataset directory with only canonical images.
Removes contamination from Set5_real and Set14_real images mixed into
Set5_HR and Set14_HR directories.
"""
import os
import shutil

BASE = r'C:\Users\ASUS\Desktop\cv'
PAIRS_FIXED = os.path.join(BASE, 'data', 'pairs_fixed')
PAIRS_CLEAN = os.path.join(BASE, 'data', 'pairs_clean')

# Canonical benchmark image sets
CANONICAL = {
    'Set5': [f'Set5_{i:02d}' for i in range(5)],      # Set5_00 to Set5_04
    'Set14': [f'Set14_{i:02d}' for i in range(14)],    # Set14_00 to Set14_13
    'BSD100': [f'img_{i:03d}' for i in range(1, 101)], # img_001 to img_100
}

SCALES = [2, 4]


def copy_canonical():
    os.makedirs(PAIRS_CLEAN, exist_ok=True)

    for dataset, names in CANONICAL.items():
        print(f'\nProcessing {dataset} ({len(names)} canonical images)...')

        # HR directory
        src_hr = os.path.join(PAIRS_FIXED, f'{dataset}_HR')
        dst_hr = os.path.join(PAIRS_CLEAN, f'{dataset}_HR')
        os.makedirs(dst_hr, exist_ok=True)

        for name in names:
            # Find the file with any extension
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                src = os.path.join(src_hr, name + ext)
                if os.path.exists(src):
                    dst = os.path.join(dst_hr, name + ext)
                    shutil.copy2(src, dst)
                    found = True
                    break
            if not found:
                print(f'  WARNING: {name} not found in {src_hr}')

        # LR directories
        for scale in SCALES:
            src_lr = os.path.join(PAIRS_FIXED, f'{dataset}_LR_x{scale}')
            dst_lr = os.path.join(PAIRS_CLEAN, f'{dataset}_LR_x{scale}')
            os.makedirs(dst_lr, exist_ok=True)

            for name in names:
                found = False
                for ext in ['.png', '.jpg', '.jpeg']:
                    src = os.path.join(src_lr, name + ext)
                    if os.path.exists(src):
                        dst = os.path.join(dst_lr, name + ext)
                        shutil.copy2(src, dst)
                        found = True
                        break
                if not found:
                    print(f'  WARNING: {name} not found in {src_lr} x{scale}')

        print(f'  {dataset}_HR: {len(os.listdir(dst_hr))} files')
        for scale in SCALES:
            dst_lr = os.path.join(PAIRS_CLEAN, f'{dataset}_LR_x{scale}')
            print(f'  {dataset}_LR_x{scale}: {len(os.listdir(dst_lr))} files')

    print(f'\nClean dataset created at: {PAIRS_CLEAN}')


if __name__ == '__main__':
    copy_canonical()
