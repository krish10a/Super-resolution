# AGENTS.md — cv / Image Super-Resolution

## Project
Deep Learning Image Super-Resolution for Low-Quality Surveillance Feeds.
Single-Image Super-Resolution (SISR) 2× and 4× using SRCNN and ESPCN on Set5/Set14.
Compare MSE loss vs Perceptual loss, evaluate PSNR/SSIM on Y-channel.

## Hardware & Environment
- Local: Windows, Ryzen 7 5800HS, RTX 2050 4GB VRAM
- Optional: Google Colab T4 16GB for perceptual training
- OS quirks: Windows DataLoader `num_workers=0`, use `if __name__ == '__main__'` guards

## Tech Stack
- PyTorch CPU/GPU, torch.cuda.amp for RTX 2050
- OpenCV-Python, Pillow, NumPy, scikit-image, tqdm
- ONNX + OpenCV DNN for CPU inference export
- Config via `configs/experiments.yaml`

## Repo Structure (planned)
```
data/          # Set5/Set14 raw, precomputed LR/HR pairs, degradation simulator
models/        # srcnn.py, espcn.py, vgg_perceptual.py
losses/        # mse_loss.py, perceptual_loss.py (hybrid MSE + λ·Perceptual)
metrics/       # psnr_ssim.py (Y-channel, border shave)
train/         # trainer.py, train_all.py
infer/         # predict.py, benchmark.py, onnx_export.py, opencv_dnn_infer.py, webcam_demo.py
utils/         # visualization.py, cpu_affinity.py
configs/       # experiments.yaml
PROJECT_CONTEXT.md  # single source of truth for full design
```

## Critical Conventions
- Metrics are Y-channel only after RGB→YCbCr conversion, borders shaved by scale.
- Precompute LR/HR pairs to disk once. Do not downsample on-the-fly during training.
- Dataset RAM-caches patches; Tiny Set5/Set14 fits in RAM.
- No batch norm in SR models. Use He init, ReLU for SRCNN, tanh output.
- Perceptual loss: VGG11, frozen, features relu2_2/relu3_3, hybrid λ in 0.05-0.1.
- Training curriculum: Phase 1 MSE → Phase 2 MSE+Perceptual fine-tune.
- Self-ensemble at inference: flip/rotate average for +PSNR (EDSR trick).

## Developer Commands
- Train: `python train/train_all.py --config configs/experiments.yaml --exp E4`
- Train with degradation (E7/E8): `python train/train_all.py --config configs/experiments.yaml --exp E7 --degrade`
- Export to ONNX: `python infer/onnx_export.py --ckpt checkpoints/espcn_4x_mse.pt --out model.onnx`
- Benchmark: `python infer/benchmark.py --ckpt checkpoints/espcn_4x_mse.pt --device cpu`
- Single-image inference: `python infer/predict.py --model checkpoints/espcn_4x_mse.pt --img input.png --out output.png`
- OpenCV DNN inference: `python infer/opencv_dnn_infer.py --model model.onnx --img input.png --out output.png`
- Webcam demo: `python infer/webcam_demo.py --model checkpoints/espcn_4x_mse.pt --camera-id 0`
- Eval SRCNN best: `python eval_srcnn.py`
- Eval ESPCN best: `python eval_best.py`
- Prepare data: `python data/download_sets.py && python data/prepare_pairs.py`

## Windows / CPU Gotchas
- `torch.set_num_threads()` should be set to physical cores.
- No `pin_memory` on CPU.
- Mixed precision only on RTX 2050; perceptual loss heavy → prefer Colab T4.
- Checkpoint to disk every epoch; Google Drive sync for Colab runs.

## Deliverables
- Working PyTorch project + `PROJECT_CONTEXT.md` with full paper references, hyperparameters, results tables.
- Do not add adversarial/GAN training unless explicitly requested; scope is SRCNN/ESPCN MSE vs Perceptual.

## Non-Negotiable Rules
- Keep Set5 for training, Set14 for testing in cross-dataset evaluation.
- Never change loss weights without updating `PROJECT_CONTEXT.md`.
- All experiments must include bicubic baseline PSNR/SSIM.
