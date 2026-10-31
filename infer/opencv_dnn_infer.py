import argparse
import os
import sys

import cv2
import numpy as np

# Ensure project root is on sys.path when run as a script (kept for consistency
# with the other infer scripts, though this script is OpenCV DNN only).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def preprocess(img, model_type, scale):
    """Prepare a BGR image (cv2.imread) for ONNX DNN inference.

    Returns a float32 NCHW array in [0, 1].

    SRCNN: bicubic-upsample LR to HR size, then normalize to [0, 1].
    ESPCN: feed raw LR, normalize to [0, 1].
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if model_type == 'srcnn':
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)    # CHW -> NCHW
    return arr


def postprocess(out):
    """Convert an NCHW float [0,1] tensor to a BGR uint8 image."""
    out = out.squeeze(0)                 # NCHW -> CHW
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).round().astype(np.uint8)
    out = np.transpose(out, (1, 2, 0))   # CHW -> HWC
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def main():
    p = argparse.ArgumentParser(description='Super-resolution via OpenCV DNN (ONNX)')
    p.add_argument('--model', required=True, help='Path to exported ONNX model (.onnx)')
    p.add_argument('--img', required=True, help='Path to input LR image')
    p.add_argument('--out', default='output.png', help='Path to save SR output image')
    p.add_argument('--model-type', choices=['srcnn', 'espcn'], default='espcn',
                   help='Model architecture (default: espcn)')
    p.add_argument('--scale', type=int, default=4, help='Upscaling factor (default: 4)')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                   help='OpenCV DNN backend target (default: cpu)')
    args = p.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f'ONNX model not found: {args.model}')
    if not os.path.isfile(args.img):
        raise FileNotFoundError(f'Input image not found: {args.img}')

    net = cv2.dnn.readNetFromONNX(args.model)
    if args.device == 'cuda':
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Failed to read image: {args.img}')

    blob = preprocess(img, args.model_type, args.scale)
    net.setInput(blob)
    out = net.forward()

    sr = postprocess(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    cv2.imwrite(args.out, sr)
    print(f'Saved SR output to {args.out} (scale {args.scale}x, {args.model_type}, {args.device})')


if __name__ == '__main__':
    main()
