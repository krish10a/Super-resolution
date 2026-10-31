import os
from PIL import Image, ImageDraw, ImageFont
import random

def make_synthetic_set(name, n, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for i in range(n):
        img = Image.new('RGB', (256, 256), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        draw = ImageDraw.Draw(img)
        # draw some text-like lines
        for _ in range(20):
            x0, y0 = random.randint(0,240), random.randint(0,240)
            x1, y1 = x0+random.randint(10,50), y0+random.randint(5,20)
            draw.line([x0,y0,x1,y1], fill=(0,0,0), width=2)
        img.save(os.path.join(base_dir, f'{name}_{i:02d}.png'))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import raw_dir

base = raw_dir()
os.makedirs(base, exist_ok=True)
make_synthetic_set('Set5', 5, os.path.join(base, 'Set5'))
make_synthetic_set('Set14', 14, os.path.join(base, 'Set14'))
print('Synthetic Set5/Set14 created')
