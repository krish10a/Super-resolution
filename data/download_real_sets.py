import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import raw_dir
from huggingface_hub import hf_hub_download

BASE = raw_dir()
BASE.mkdir(parents=True, exist_ok=True)

def download_zip(repo_id, filename, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {repo_id}/{filename}...')
    # hf_hub_download returns local cache path, we copy
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # For images, we can use datasets.load_dataset
    return local_path

# Alternative: use datasets library to load images
from datasets import load_dataset

def save_set5():
    ds = load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation')
    out_dir = BASE / 'Set5_real'
    out_dir.mkdir(exist_ok=True)
    for i in range(len(ds)):
        pass
    print('Set5 loaded')

if __name__ == '__main__':
    save_set5()
