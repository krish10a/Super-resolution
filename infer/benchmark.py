import argparse
import os
import sys
import time

import torch

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


def dummy_input(model_type, scale, batch, height, width, device):
    """Build a dummy LR input matching each model's expected resolution.

    SRCNN: spec uses bicubic-upsampled LR as input (same size as HR target).
    ESPCN: PixelShuffle upscales internally, so feed raw LR.
    """
    if model_type == 'srcnn':
        in_h, in_w = height * scale, width * scale
    else:
        in_h, in_w = height, width
    return torch.randn(batch, 3, in_h, in_w, device=device)


@torch.no_grad()
def benchmark(model, x, iters=100, warmup=10, device='cpu'):
    """Measure inference latency and throughput.

    Returns (mean_ms, std_ms, fps).
    """
    model.eval()
    # Warmup (CUDA kernels need a few runs to settle)
    for _ in range(warmup):
        _ = model(x)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        if device.startswith('cuda'):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))  # ms
        else:
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    import numpy as np
    timings = np.array(timings, dtype=np.float64)
    mean_ms = float(timings.mean())
    std_ms = float(timings.std())
    fps = 1000.0 / mean_ms if mean_ms > 0 else float('inf')
    return mean_ms, std_ms, fps


def main():
    p = argparse.ArgumentParser(description='Benchmark SRCNN/ESPCN inference latency')
    p.add_argument('--ckpt', required=True, help='Path to model checkpoint (.pt)')
    p.add_argument('--model-type', choices=['srcnn', 'espcn'], default='espcn')
    p.add_argument('--scale', type=int, default=4, help='Upscaling factor')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--height', type=int, default=64, help='Dummy input height (LR)')
    p.add_argument('--width', type=int, default=64, help='Dummy input width (LR)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--iters', type=int, default=100, help='Timed iterations')
    p.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')

    model = load_model(args.ckpt, args.model_type, args.scale, args.device)
    x = dummy_input(args.model_type, args.scale, args.batch,
                    args.height, args.width, args.device)

    mean_ms, std_ms, fps = benchmark(model, x, args.iters, args.warmup, args.device)

    in_res = (args.height * args.scale, args.width * args.scale) if args.model_type == 'srcnn' \
        else (args.height, args.width)
    print(f'Model:        {args.model_type} (scale {args.scale}x)')
    print(f'Device:       {args.device}')
    print(f'Input res:    {in_res[0]}x{in_res[1]} (batch {args.batch})')
    print(f'Warmup:       {args.warmup} iters')
    print(f'Timed:        {args.iters} iters')
    print(f'Latency:      {mean_ms:.2f} +/- {std_ms:.2f} ms')
    print(f'Throughput:   {fps:.2f} fps (batch {args.batch})')


if __name__ == '__main__':
    main()
