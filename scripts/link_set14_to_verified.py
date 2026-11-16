"""
Link Set14 HR and LR data from pairs_reference into pairs_verified for validation.
Uses hard links (no data duplication).
"""

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_root = os.path.join(BASE, 'data', 'pairs_reference')
dst_root = os.path.join(BASE, 'data', 'pairs_verified')

dirs_to_link = [
    'Set14_HR',
    'Set14_LR_x2',
    'Set14_LR_x4',
]

for dir_name in dirs_to_link:
    src = os.path.join(src_root, dir_name)
    dst = os.path.join(dst_root, dir_name)
    if not os.path.isdir(src):
        print(f"SKIP: {src} does not exist")
        continue
    os.makedirs(dst, exist_ok=True)
    files = sorted([f for f in os.listdir(src)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    linked = 0
    for f in files:
        s = os.path.join(src, f)
        d = os.path.join(dst, f)
        if not os.path.exists(d):
            try:
                os.link(s, d)
                linked += 1
            except OSError:
                import shutil
                shutil.copy2(s, d)
                linked += 1
    print(f"Linked {linked} files: {dir_name}")

print('\nDone.')
