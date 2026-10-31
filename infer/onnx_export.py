import argparse
import os
import sys

import torch

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srcnn import SRCNN
from models.espcn import ESPCN


def load_model(checkpoint_path, model_type, scale, device):
    if model_type == 'srcnn':
        model = SRCNN()
    elif model_type == 'espcn':
        model = ESPCN(scale=scale)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description='Export SRCNN/ESPCN to ONNX')
    p.add_argument('--ckpt', required=True, help='Path to model checkpoint (.pt)')
    p.add_argument('--out', required=True, help='Path to save exported .onnx')
    p.add_argument('--model-type', choices=['srcnn', 'espcn'], default='espcn')
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--height', type=int, default=64, help='Dummy input height (LR)')
    p.add_argument('--width', type=int, default=64, help='Dummy input width (LR)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')

    model = load_model(args.ckpt, args.model_type, args.scale, args.device)

    # SRCNN input is bicubic-upsampled to HR size; ESPCN input stays LR.
    if args.model_type == 'srcnn':
        in_h, in_w = args.height * args.scale, args.width * args.scale
    else:
        in_h, in_w = args.height, args.width

    dummy = torch.randn(args.batch, 3, in_h, in_w, device=args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )
    print(f'Exported {args.model_type} (scale {args.scale}x) to {args.out}')


if __name__ == '__main__':
    main()
