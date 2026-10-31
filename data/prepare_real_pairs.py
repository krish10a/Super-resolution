import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import raw_dir, pairs_real_dir

import cv2
import numpy as np

raw_set5 = raw_dir('Set5_real')
raw_set14 = raw_dir('Set14_real')
pairs = pairs_real_dir()
pairs.mkdir(parents=True, exist_ok=True)

def downsample(hr_img, scale):
    h, w = hr_img.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    lr = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    return lr

def prepare_dataset(name, src_dir):
    src = Path(src_dir)
    if not src.exists():
        print(f'Source {src} missing')
        return
    for scale in [2,4]:
        lr_dir = pairs / f'{name}_LR_x{scale}'
        hr_dir = pairs / f'{name}_HR'
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)
        for img_file in src.iterdir():
            if img_file.suffix.lower() not in ['.png','.jpg','.jpeg']:
                continue
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            lr = downsample(img, scale)
            # save LR with same basename as HR
            lr_name = img_file.name
            cv2.imwrite(str(lr_dir / lr_name), lr)
            # save HR once (or copy)
            if not (hr_dir / img_file.name).exists():
                cv2.imwrite(str(hr_dir / img_file.name), img)
    print(f'Prepared {name}')

prepare_dataset('Set5_real', raw_set5)
prepare_dataset('Set14_real', raw_set14)
print('Done')
