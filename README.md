# Deep Learning SISR for Surveillance

Single-Image Super-Resolution for low-quality surveillance feeds using SRCNN and ESPCN.
2×/4× upscaling on Set5/Set14. MSE vs Perceptual loss. PSNR/SSIM evaluation.

## Quick Start
1. Install: `pip install -r requirements.txt`
2. Prepare data: `python data/download_sets.py && python data/prepare_pairs.py`
3. Train: `python train/train_all.py --config configs/experiments.yaml --exp E4`
4. Infer: `python infer/predict.py --model outputs/checkpoints/espcn_4x.pt --img input.png`
5. Benchmark: `python infer/benchmark.py --onnx outputs/onnx/espcn_4x.onnx`

See `PROJECT_CONTEXT.md` for full design, `AGENTS.md` for agent rules.
