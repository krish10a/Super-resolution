import requests, os
from pathlib import Path

url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
out = Path('data/raw/DIV2K_train_HR.zip')
out.parent.mkdir(parents=True, exist_ok=True)

print('Downloading DIV2K train HR...')
with requests.get(url, stream=True, verify=False, timeout=30) as r:
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(out, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f'\r{downloaded/total*100:.2f}% ', end='')
print('\nDone')
