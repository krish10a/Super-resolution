import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import raw_dir, pairs_dir

from PIL import Image
import cv2
import numpy as np

def downsample(img, scale):
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    return lr, lr_up

raw = raw_dir()
out = pairs_dir()
for scale in [2,4]:
    for ds in ['Set5','Set14']:
        in_dir = os.path.join(raw, ds)
        out_lr = os.path.join(out, f'{ds}_LR_x{scale}')
        out_hr = os.path.join(out, f'{ds}_HR')
        os.makedirs(out_lr, exist_ok=True)
        os.makedirs(out_hr, exist_ok=True)
        for f in os.listdir(in_dir):
            if not f.endswith('.png'): continue
            img = cv2.imread(os.path.join(in_dir, f))
            lr, lr_up = downsample(img, scale)
            cv2.imwrite(os.path.join(out_lr, f), lr)
            # copy HR
            cv2.imwrite(os.path.join(out_hr, f), img)
print('Pairs prepared')
