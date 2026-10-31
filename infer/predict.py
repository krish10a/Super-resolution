import argparse
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as T

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srcnn import SRCNN
from models.espcn import ESPCN


def load_model(checkpoint_path, model_type, scale, device):
    """Instantiate the model and load checkpoint weights."""
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


def preprocess(img, model_type, scale):
    """Prepare a PIL image for the given model.

    SRCNN: spec uses bicubic-upsampled LR as input (same size as HR target).
    ESPCN: PixelShuffle upscales internally, so feed raw LR.
    """
    if model_type == 'srcnn':
        w, h = img.size
        img = img.resize((w * scale, h * scale), Image.BICUBIC)
    return T.ToTensor()(img).unsqueeze(0)


@torch.no_grad()
def run_inference(model, x, device):
    x = x.to(device)
    out = model(x)
    return out.squeeze(0).clamp(0, 1)


def save_image(tensor, path):
    img = T.ToPILImage()(tensor.cpu())
    img.save(path)


def main():
    p = argparse.ArgumentParser(description='Single-image super-resolution inference')
    p.add_argument('--model', required=True, help='Path to model checkpoint (.pt)')
    p.add_argument('--img', required=True, help='Path to input LR image')
    p.add_argument('--out', default='output.png', help='Path to save SR output image')
    p.add_argument('--model-type', choices=['srcnn', 'espcn'], default='espcn',
                   help='Model architecture (default: espcn)')
    p.add_argument('--scale', type=int, default=4, help='Upscaling factor (default: 4)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f'Checkpoint not found: {args.model}')
    if not os.path.isfile(args.img):
        raise FileNotFoundError(f'Input image not found: {args.img}')

    model = load_model(args.model, args.model_type, args.scale, args.device)

    img = Image.open(args.img).convert('RGB')
    x = preprocess(img, args.model_type, args.scale)

    out = run_inference(model, x, args.device)
    save_image(out, args.out)
    print(f'Saved SR output to {args.out} (scale {args.scale}x, {args.model_type})')


if __name__ == '__main__':
    main()
