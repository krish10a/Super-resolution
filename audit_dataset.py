"""
Dataset integrity audit for SISR project.
Checks: counts, duplicates, dimensions, alignment, leakage, provenance.
"""
import os
import hashlib
from PIL import Image
import numpy as np

BASE = r'C:\Users\ASUS\Desktop\cv'
RAW = os.path.join(BASE, 'data', 'raw')
PAIRS = os.path.join(BASE, 'data', 'pairs')
PAIRS_FIXED = os.path.join(BASE, 'data', 'pairs_fixed')


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted([f for f in os.listdir(d)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def dims(d, name):
    p = os.path.join(d, name)
    try:
        with Image.open(p) as im:
            return im.size, im.mode
    except Exception as e:
        return f'ERROR: {e}', None


def count_section(title, items):
    print(f'\n=== {title} ===')
    for k, v in items:
        print(f'  {k}: {v}')


def audit_raw():
    print('\n' + '=' * 70)
    print('STEP 1.1 — RAW DATASET AUDIT')
    print('=' * 70)

    expected = {
        'Set5': 5,
        'Set14': 14,
        'BSD100': 100,
        'T91': 91,
    }

    for name in ['Set5', 'Set14', 'Set5_real', 'Set14_real', 'Set14_png',
                 'BSD100_png', 'T91_png']:
        d = os.path.join(RAW, name)
        files = list_images(d)
        count_section(f'RAW/{name} ({len(files)} files)', [
            ('path', d),
            ('exists', os.path.isdir(d)),
        ])
        if not files:
            continue
        # Check duplicates by hash
        hashes = {}
        dupes = []
        dims_set = set()
        modes = set()
        for f in files:
            p = os.path.join(d, f)
            h = md5(p)
            if h in hashes:
                dupes.append((f, hashes[h]))
            hashes[h] = f
            sz, mode = dims(d, f)
            if isinstance(sz, tuple):
                dims_set.add(sz)
            if mode:
                modes.add(mode)
        count_section(f'  {name} details', [
            ('count', len(files)),
            ('expected', expected.get(name, 'N/A')),
            ('match_expected', len(files) == expected.get(name, len(files))),
            ('duplicate_hashes', len(dupes)),
            ('dimensions', sorted(dims_set)[:5] if dims_set else 'N/A'),
            ('modes', sorted(modes)),
        ])
        if dupes:
            print(f'  DUPLICATES: {dupes[:5]}')


def audit_pairs(label, base):
    print('\n' + '=' * 70)
    print(f'STEP 1.4 — PAIRS AUDIT: {label} ({base})')
    print('=' * 70)

    sets = ['Set5', 'Set14', 'BSD100', 'T91']
    scales = [2, 4]

    for s in sets:
        hr_dir = os.path.join(base, f'{s}_HR')
        hr_files = list_images(hr_dir)
        count_section(f'{s}_HR ({len(hr_files)} files)', [
            ('path', hr_dir),
            ('exists', os.path.isdir(hr_dir)),
        ])

        for scale in scales:
            lr_dir = os.path.join(base, f'{s}_LR_x{scale}')
            lr_files = list_images(lr_dir)
            count_section(f'  {s}_LR_x{scale} ({len(lr_files)} files)', [
                ('path', lr_dir),
                ('exists', os.path.isdir(lr_dir)),
            ])

            # Check HR/LR filename matching
            hr_names = {os.path.splitext(f)[0] for f in hr_files}
            lr_names = {os.path.splitext(f)[0] for f in lr_files}
            only_hr = hr_names - lr_names
            only_lr = lr_names - hr_names
            if only_hr or only_lr:
                print(f'  MISMATCH: HR-only={sorted(only_hr)[:5]}, LR-only={sorted(only_lr)[:5]}')

            # Check dimensions: HR should be scale * LR (or close)
            dim_issues = 0
            checked = 0
            for f in hr_files:
                name = os.path.splitext(f)[0]
                lr_path = os.path.join(lr_dir, f)
                hr_path = os.path.join(hr_dir, f)
                if not os.path.exists(lr_path):
                    continue
                try:
                    with Image.open(hr_path) as hi, Image.open(lr_path) as li:
                        hw, hh = hi.size
                        lw, lh = li.size
                        expected_w = hw // scale
                        expected_h = hh // scale
                        if lw != expected_w or lh != expected_h:
                            dim_issues += 1
                            if dim_issues <= 3:
                                print(f'  ALIGN ISSUE {f}: HR={hi.size} LR={li.size} expected_LR=({expected_w},{expected_h})')
                        checked += 1
                except Exception as e:
                    print(f'  READ ERROR {f}: {e}')
            count_section(f'  {s} x{scale} alignment', [
                ('pairs_checked', checked),
                ('dim_mismatches', dim_issues),
            ])


def audit_duplicates_across_dirs():
    print('\n' + '=' * 70)
    print('STEP 1.1 — CROSS-DIRECTORY DUPLICATE DETECTION')
    print('=' * 70)

    dirs = {}
    for name in ['Set5', 'Set14', 'Set5_real', 'Set14_real', 'Set14_png',
                 'BSD100_png', 'T91_png']:
        d = os.path.join(RAW, name)
        if os.path.isdir(d):
            dirs[f'RAW/{name}'] = d

    for name in ['Set5_HR', 'Set14_HR', 'BSD100_HR', 'T91_HR',
                 'Set5_LR_x2', 'Set14_LR_x2', 'BSD100_LR_x2', 'T91_LR_x2',
                 'Set5_LR_x4', 'Set14_LR_x4', 'BSD100_LR_x4', 'T91_LR_x4']:
        for base, label in [(PAIRS, 'pairs'), (PAIRS_FIXED, 'pairs_fixed')]:
            d = os.path.join(base, name)
            if os.path.isdir(d):
                dirs[f'{label}/{name}'] = d

    all_hashes = {}
    for label, d in dirs.items():
        for f in list_images(d):
            p = os.path.join(d, f)
            h = md5(p)
            if h in all_hashes:
                print(f'  CROSS-DIR DUP: {label}/{f} == {all_hashes[h]}')
            else:
                all_hashes[h] = f'{label}/{f}'


def audit_leakage():
    print('\n' + '=' * 70)
    print('STEP 1.5 — TRAIN/TEST LEAKAGE AUDIT')
    print('=' * 70)

    # T91 should only be in training dirs
    # Set5, Set14, BSD100 should only be in test dirs
    t91_dir = os.path.join(RAW, 'T91_png')
    if os.path.isdir(t91_dir):
        t91_files = list_images(t91_dir)
        t91_names = {os.path.splitext(f)[0] for f in t91_files}
        print(f'  T91_png: {len(t91_names)} unique images')

    # Check if any benchmark names appear in T91 training data
    benchmarks = {}
    for name in ['Set5', 'Set14', 'BSD100']:
        d = os.path.join(RAW, name)
        if os.path.isdir(d):
            benchmarks[name] = {os.path.splitext(f)[0] for f in list_images(d)}

    # Check pairs directories for cross-contamination
    for base, label in [(PAIRS, 'pairs'), (PAIRS_FIXED, 'pairs_fixed')]:
        for s in ['Set5', 'Set14', 'BSD100', 'T91']:
            hr_dir = os.path.join(base, f'{s}_HR')
            if os.path.isdir(hr_dir):
                names = {os.path.splitext(f)[0] for f in list_images(hr_dir)}
                for bname, bset in benchmarks.items():
                    overlap = names & bset
                    if overlap and bname != s:
                        print(f'  LEAKAGE: {label}/{s}_HR contains {bname} images: {sorted(overlap)[:5]}')


def main():
    audit_raw()
    audit_duplicates_across_dirs()
    audit_leakage()
    audit_pairs('pairs', PAIRS)
    audit_pairs('pairs_fixed', PAIRS_FIXED)


if __name__ == '__main__':
    main()
