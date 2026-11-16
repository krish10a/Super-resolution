"""Inspect all eight checkpoints: architecture, scale, shapes, metadata, duplicates."""
import os
import torch
from collections import OrderedDict

CKPT_DIR = 'outputs/checkpoints'

CKPTS = {
    'E1': 'srcnn_2x_mse.pt',
    'E2': 'srcnn_2x_mse_perceptual.pt',
    'E3': 'srcnn_4x_mse.pt',
    'E4': 'srcnn_4x_mse_perceptual.pt',
    'E5': 'espcn_2x_mse.pt',
    'E6': 'espcn_2x_mse_perceptual.pt',
    'E7': 'espcn_4x_mse.pt',
    'E8': 'espcn_4x_mse_perceptual.pt',
}


def count_params(sd):
    return sum(v.numel() for v in sd.values())


def inspect(name, path):
    print(f"\n{'='*60}")
    print(f"{name}: {path}")
    print(f"{'='*60}")
    if not os.path.isfile(path):
        print("  MISSING FILE")
        return

    sd = torch.load(path, map_location='cpu')
    print(f"  Type: {type(sd).__name__}")

    if isinstance(sd, dict) and any(k in sd for k in ('epoch', 'loss', 'optimizer', 'step')):
        print("  Contains training metadata:")
        for k in ('epoch', 'loss', 'step', 'best_psnr', 'val_psnr'):
            if k in sd:
                print(f"    {k}: {sd[k]}")
        sd = sd.get('state_dict', sd)
    elif isinstance(sd, dict):
        # Check if it's a raw state_dict (tensor values)
        first_val = next(iter(sd.values()))
        if torch.is_tensor(first_val):
            print("  Raw state_dict (no metadata wrapper)")
        else:
            print(f"  Dict with keys: {list(sd.keys())[:10]}")
            return

    n_params = count_params(sd)
    print(f"  Total parameters: {n_params:,}")

    # Check tensor shapes
    shapes = OrderedDict()
    for k, v in sd.items():
        if torch.is_tensor(v):
            shapes[k] = tuple(v.shape)
    print(f"  Num tensor keys: {len(shapes)}")

    # Architecture fingerprint
    if any('net.' in k for k in shapes):
        print("  Architecture: SRCNN (keys prefixed with 'net.')")
    elif any('features.' in k for k in shapes):
        print("  Architecture: ESPCN (keys include 'features.')")
    else:
        print("  Architecture: unknown / flat keys")

    # Scale inference from conv layers
    # SRCNN: final conv 32->3, 5x5
    # ESPCN: final subpixel conv, channels * scale^2
    print(f"  Sample shapes:")
    for k in list(shapes.keys())[:5]:
        print(f"    {k}: {shapes[k]}")
    print(f"    ... ({len(shapes)} total)")

    return sd


sds = {}
for exp, fname in CKPTS.items():
    path = os.path.join(CKPT_DIR, fname)
    sds[exp] = inspect(exp, path)

# Compare for duplicates
print(f"\n{'='*60}")
print("DUPLICATE / IDENTICAL WEIGHT CHECK")
print(f"{'='*60}")
exp_list = list(sds.keys())
for i in range(len(exp_list)):
    for j in range(i+1, len(exp_list)):
        e1, e2 = exp_list[i], exp_list[j]
        s1, s2 = sds.get(e1), sds.get(e2)
        if s1 is None or s2 is None:
            continue
        # Compare keys and values
        k1, k2 = set(s1.keys()), set(s2.keys())
        if k1 != k2:
            print(f"  {e1} vs {e2}: KEY MISMATCH (different architectures)")
            continue
        identical = all(torch.equal(s1[k], s2[k]) for k in k1)
        if identical:
            print(f"  {e1} vs {e2}: IDENTICAL WEIGHTS (duplicate!)")
        else:
            # Check if close (same init, not trained)
            max_diff = max((s1[k] - s2[k]).abs().max().item() for k in k1)
            print(f"  {e1} vs {e2}: different (max abs diff={max_diff:.6f})")
