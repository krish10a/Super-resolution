"""
Benchmark asset audit: verify canonical image identities, dimensions, hashes, duplicates.

Checks:
- data/raw/Set5, data/raw/Set14, data/raw/BSD100_png
- data/pairs_matlab/ (new MATLAB-compatible dataset)
- data/pairs_clean/ (existing clean dataset)
- data/pairs_fixed/ (existing fixed dataset)

Records SHA-256 hashes, dimensions, channel counts, duplicates, and format consistency.
"""

import os
import sys
import hashlib
import json
from PIL import Image
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, 'outputs', 'evaluation')
OUT_FILE = os.path.join(OUT_DIR, 'benchmark_asset_audit.json')

os.makedirs(OUT_DIR, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted([f for f in os.listdir(d)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def audit_directory(base, dir_rel, label):
    """Audit all images in a directory."""
    d = os.path.join(base, dir_rel)
    files = list_images(d)
    result = {
        'label': label,
        'path': d,
        'exists': os.path.isdir(d),
        'count': len(files),
        'images': [],
        'duplicates': [],
        'format_issues': [],
    }

    seen_hashes = {}
    for f in files:
        p = os.path.join(d, f)
        try:
            with Image.open(p) as im:
                width, height = im.size
                mode = im.mode
                fmt = im.format
        except Exception as e:
            result['format_issues'].append({
                'file': f,
                'error': str(e),
            })
            continue

        file_hash = sha256(p)
        entry = {
            'file': f,
            'sha256': file_hash,
            'width': width,
            'height': height,
            'mode': mode,
            'format': fmt,
        }
        result['images'].append(entry)

        # Check duplicates
        if file_hash in seen_hashes:
            result['duplicates'].append({
                'file': f,
                'duplicate_of': seen_hashes[file_hash],
                'sha256': file_hash,
            })
        else:
            seen_hashes[file_hash] = f

    return result


def main():
    audit = {
        'audit_type': 'benchmark_asset_audit',
        'base_dir': BASE,
        'directories': [],
    }

    # Source raw datasets
    raw_dirs = [
        ('data/raw/Set5', 'RAW Set5'),
        ('data/raw/Set14', 'RAW Set14'),
        ('data/raw/BSD100_png', 'RAW BSD100_png'),
    ]

    # Derived datasets
    derived_dirs = [
        ('data/pairs_matlab/Set5_HR', 'pairs_matlab Set5_HR'),
        ('data/pairs_matlab/Set5_LR_x2', 'pairs_matlab Set5_LR_x2'),
        ('data/pairs_matlab/Set5_LR_x4', 'pairs_matlab Set5_LR_x4'),
        ('data/pairs_matlab/Set14_HR', 'pairs_matlab Set14_HR'),
        ('data/pairs_matlab/Set14_LR_x2', 'pairs_matlab Set14_LR_x2'),
        ('data/pairs_matlab/Set14_LR_x4', 'pairs_matlab Set14_LR_x4'),
        ('data/pairs_matlab/BSD100_HR', 'pairs_matlab BSD100_HR'),
        ('data/pairs_matlab/BSD100_LR_x2', 'pairs_matlab BSD100_LR_x2'),
        ('data/pairs_matlab/BSD100_LR_x4', 'pairs_matlab BSD100_LR_x4'),
        ('data/pairs_clean/Set5_HR', 'pairs_clean Set5_HR'),
        ('data/pairs_clean/Set5_LR_x2', 'pairs_clean Set5_LR_x2'),
        ('data/pairs_clean/Set5_LR_x4', 'pairs_clean Set5_LR_x4'),
        ('data/pairs_clean/Set14_HR', 'pairs_clean Set14_HR'),
        ('data/pairs_clean/Set14_LR_x2', 'pairs_clean Set14_LR_x2'),
        ('data/pairs_clean/Set14_LR_x4', 'pairs_clean Set14_LR_x4'),
        ('data/pairs_clean/BSD100_HR', 'pairs_clean BSD100_HR'),
        ('data/pairs_clean/BSD100_LR_x2', 'pairs_clean BSD100_LR_x2'),
        ('data/pairs_clean/BSD100_LR_x4', 'pairs_clean BSD100_LR_x4'),
        ('data/pairs_fixed/Set5_HR', 'pairs_fixed Set5_HR'),
        ('data/pairs_fixed/Set5_LR_x2', 'pairs_fixed Set5_LR_x2'),
        ('data/pairs_fixed/Set5_LR_x4', 'pairs_fixed Set5_LR_x4'),
        ('data/pairs_fixed/Set14_HR', 'pairs_fixed Set14_HR'),
        ('data/pairs_fixed/Set14_LR_x2', 'pairs_fixed Set14_LR_x2'),
        ('data/pairs_fixed/Set14_LR_x4', 'pairs_fixed Set14_LR_x4'),
        ('data/pairs_fixed/BSD100_HR', 'pairs_fixed BSD100_HR'),
        ('data/pairs_fixed/BSD100_LR_x2', 'pairs_fixed BSD100_LR_x2'),
        ('data/pairs_fixed/BSD100_LR_x4', 'pairs_fixed BSD100_LR_x4'),
    ]

    for dir_rel, label in raw_dirs + derived_dirs:
        result = audit_directory(BASE, dir_rel, label)
        audit['directories'].append(result)

    # Canonical counts
    audit['canonical_counts'] = {
        'Set5': {'expected': 5, 'HR_dir': 'Set5_HR'},
        'Set14': {'expected': 14, 'HR_dir': 'Set14_HR'},
        'BSD100': {'expected': 100, 'HR_dir': 'BSD100_HR'},
    }

    with open(OUT_FILE, 'w') as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"Asset audit saved to {OUT_FILE}")
    print(f"\nSummary:")
    for d in audit['directories']:
        status = "OK" if d['exists'] else "MISSING"
        dup = f", {len(d['duplicates'])} dupes" if d['duplicates'] else ""
        issues = f", {len(d['format_issues'])} issues" if d['format_issues'] else ""
        print(f"  {d['label']}: {d['count']} images [{status}]{dup}{issues}")

    return audit


if __name__ == '__main__':
    main()
