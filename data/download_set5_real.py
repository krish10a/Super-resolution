import requests, os
from pathlib import Path

base = Path('data/raw/Set5_real')
base.mkdir(parents=True, exist_ok=True)

images = ['baby','bird','butterfly','head','woman']
for name in images:
    url = f'https://raw.githubusercontent.com/linzino7/Set5-dataset/main/Set5/HR/{name}.png'
    print('Downloading', name)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out = base / f'{name}.png'
    out.write_bytes(r.content)
    print(' saved', out.stat().st_size)

print('Set5 done')
