import requests
from pathlib import Path

base = Path('data/raw/Set14_real')
base.mkdir(parents=True, exist_ok=True)

images = ['baboon','barbara','bridge','coastguard','collage','comic','face','flowers','foreman','lenna','man','monarch','pepper','ppt3','zebra']
# Actually that's 15, need 14. Let's use first 14 excluding one.
images = ['baboon','barbara','bridge','coastguard','collage','comic','face','flowers','foreman','lenna','man','monarch','pepper','ppt3']
for name in images:
    url = f'https://raw.githubusercontent.com/Benjamin-Wegener/Set14/main/{name}.jpeg'
    print('Downloading', name)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out = base / f'{name}.jpeg'
        out.write_bytes(r.content)
        print(' saved', out.stat().st_size)
    except Exception as e:
        print(' failed', e)

print('Set14 done')
