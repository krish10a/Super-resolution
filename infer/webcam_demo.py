import argparse
import os
import sys

import cv2
import numpy as np
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


def preprocess_frame(frame, model_type, scale):
    """Downscale a BGR frame to LR and build the model input tensor.

    SRCNN: LR is bicubic-upscaled back to HR size before feeding (spec).
    ESPCN: raw LR is fed; PixelShuffle upscales internally.

    Returns (lr_frame, input_tensor).
    """
    h, w = frame.shape[:2]
    lr = cv2.resize(frame, (max(1, w // scale), max(1, h // scale)),
                    interpolation=cv2.INTER_AREA)
    if model_type == 'srcnn':
        model_in = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        model_in = lr
    rgb = cv2.cvtColor(model_in, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    return lr, tensor


def postprocess(out):
    """Convert a (1,3,H,W) torch tensor to a BGR uint8 frame."""
    out = out.squeeze(0).clamp(0, 1)
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser(description='Real-time webcam super-resolution demo')
    p.add_argument('--model', required=True, help='Path to model checkpoint (.pt)')
    p.add_argument('--model-type', choices=['srcnn', 'espcn'], default='espcn')
    p.add_argument('--scale', type=int, default=4, help='Upscaling factor (default: 4)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--camera-id', type=int, default=0, help='OpenCV camera index (default: 0)')
    args = p.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f'Checkpoint not found: {args.model}')

    model = load_model(args.model, args.model_type, args.scale, args.device)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {args.camera_id}')

    print('Press "q" to quit.')
    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                _, x = preprocess_frame(frame, args.model_type, args.scale)
                x = x.to(args.device)
                out = model(x)
                # SRCNN ends in Tanh ([-1, 1]); map to [0, 1] for display.
                if args.model_type == 'srcnn':
                    out = (out + 1.0) / 2.0
                sr = postprocess(out)

                # Side-by-side: original frame (left) and SR output (right).
                # Match heights by resizing SR to the frame height if needed.
                fh, fw = frame.shape[:2]
                sh, sw = sr.shape[:2]
                if sh != fh:
                    sr = cv2.resize(sr, (int(sw * fh / sh), fh),
                                    interpolation=cv2.INTER_CUBIC)
                combined = np.hstack([frame, sr])
                cv2.putText(combined, 'LR input', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(combined, f'SR {args.scale}x ({args.model_type})',
                            (fw + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 2)
                cv2.imshow('Super-Resolution Webcam', combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
