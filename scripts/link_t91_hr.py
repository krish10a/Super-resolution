"""
Create data/pairs_verified/T91_HR/ by linking to existing T91_HR files.
Uses hard links (no data duplication) to avoid data sprawl.
"""

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_hr = os.path.join(BASE, 'data', 'pairs', 'T91_HR')
dst_hr = os.path.join(BASE, 'data', 'pairs_verified', 'T91_HR')

os.makedirs(dst_hr, exist_ok=True)

files = sorted([f for f in os.listdir(src_hr)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {len(files)} T91 HR images to link")

linked = 0
for f in files:
    src = os.path.join(src_hr, f)
    dst = os.path.join(dst_hr, f)
    if not os.path.exists(dst):
        try:
            os.link(src, dst)  # hard link — no data duplication
            linked += 1
        except OSError:
            # Fall back to copy if hard link not supported
            import shutil
            shutil.copy2(src, dst)
            linked += 1

print(f"Linked/copied {linked} files to {dst_hr}")
