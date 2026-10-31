import requests
from pathlib import Path

base = Path('data/raw/Set14_real')
base.mkdir(parents=True, exist_ok=True)

images = ['baboon','barbara','boat','cat','chair','city','comic','face','flowers','foreman','horse','lena','monarch','pepper']
for name in images:
    url = f'https://raw.githubusercontent.com/Benjamin-Wegener/Set14/main/{name}.jpeg'
    print('Downloading', name)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out = base / f'{name}.jpeg'
    out.write_bytes(r.content)
    print(' saved', out.stat().st_size)

print('Set14 done')
