# PROJECT_CONTEXT.md — Deep Learning SISR for Low-Quality Surveillance

## 1. Problem Statement
Reconstruct high-frequency textural details and legible features from low-resolution surveillance footage using Single-Image Super-Resolution (SISR). Target scales: 2× and 4×. Baseline models: SRCNN and ESPCN. Losses: MSE vs Hybrid MSE + Perceptual. Datasets: Set5 and Set14. Metrics: PSNR and SSIM on Y-channel.

## 2. Objectives
- Implement SRCNN and ESPCN in PyTorch with reproducible training.
- Train with MSE loss and Hybrid MSE+Perceptual loss.
- Evaluate PSNR/SSIM against bicubic baseline on Set5→Set14 cross-dataset.
- Provide CPU/GPU inference, ONNX export, and webcam demo.
- Deliver a working project + this document as single source of truth.

## 3. Hardware & Environment
- Local: Windows, Ryzen 7 5800HS, RTX 2050 4GB VRAM
- Optional: Google Colab T4 16GB for perceptual experiments
- PyTorch CPU/GPU, Windows DataLoader num_workers=0

## 4. Research Foundations
- Dong et al., SRCNN, 2014/2016 — 9-1-5 conv layers, MSE, bicubic pre-upsample
- Shi et al., ESPCN, CVPR 2016 — Sub-pixel convolution / PixelShuffle
- Ledig et al., SRGAN, 2017 — Perceptual loss, MSE+Perceptual hybrid, Y-channel evaluation
- Johnson et al., Perceptual Losses, 2016 — VGG feature loss
- Lim et al., EDSR, 2017 — No batch norm, self-ensemble test-time augmentation
- Kim et al., VDSR, 2016 — Residual learning, gradient clipping, LR decay

## 5. Data Pipeline
- Set5 and Set14 HR PNGs downloaded to data/raw/
- Precompute LR pairs via bicubic downsample at 2× and 4×, save to data/pairs/x2/ and x4/
- Patch extraction: LR 48×48 → HR 192×192 for 4×; LR 32×32 → HR 64×64 for 2×
- Augmentation: random flip, rotation 0/90/180/270
- RAM-cached dataset; tiny Set5/Set14 fits fully in RAM
- Synthetic CCTV degradation: Gaussian noise σ5-15, JPEG q20-40, motion blur, low-light gamma

## 6. Model Architectures

### SRCNN
Input: bicubic upsampled LR → Output: HR
Conv1: 9×9, 64 ch, ReLU
Conv2: 1×1, 32 ch, ReLU
Conv3: 5×5, 3 ch, tanh
No batch norm. He init.

### ESPCN
Feature extractor: Conv 3×3 ×2 with 64 ch, ReLU
Upsample: PixelShuffle(scale) to produce HR
No batch norm. He init.

### VGG Perceptual
VGG11 features up to relu2_2 or relu3_3, frozen, ImageNet norm.

## 7. Loss Functions
MSE: nn.MSELoss()
Perceptual: MSE on VGG features
Hybrid: L = MSE + λ·Perceptual, λ =0.05-0.1
Curriculum: Phase1 MSE only → Phase2 MSE+Perceptual fine-tune

## 8. Training Protocol
Optimizer: Adam lr 1e-4
Scheduler: ReduceLROnPlateau patience5 factor0.5
Batch: local 4-8 MSE, 2-4 Perceptual with grad accum 4
Mixed precision AMP on RTX 2050
Gradient clip max norm 0.01
Early stopping patience 15 on val PSNR
Cross-dataset: train Set5, test Set14

## 9. Metrics
PSNR and SSIM on Y channel after RGB→YCbCr
Border shave = scale pixels
Compare vs bicubic baseline for every experiment

## 10. Experiment Matrix
E1 SRCNN 2× MSE Set5→Set14 Local
E2 SRCNN 4× MSE Set5→Set14 Local
E3 ESPCN 2× MSE Set5→Set14 Local
E4 ESPCN 4× MSE Set5→Set14 Local
E5 ESPCN 2× MSE+Perceptual Set5→Set14 Colab
E6 ESPCN 4× MSE+Perceptual Set5→Set14 Colab
E7 ESPCN 4× MSE degraded Set5→Set14 Local
E8 ESPCN 4× MSE+Perceptual degraded Set5→Set14 Colab

## 11. Inference & Deployment
ONNX export, OpenCV DNN CPU inference
Dynamic quantization for CPU
Self-ensemble flip/rotate average at test time
Webcam demo with FPS counter

## 12. File-by-File Guide
data/download_sets.py — download Set5/Set14
data/prepare_pairs.py — bicubic LR synthesis
data/degradation.py — synthetic CCTV degradation
data/dataset.py — RAM-cached PatchDataset
models/srcnn.py — SRCNN class
models/espcn.py — ESPCN class
models/vgg_perceptual.py — frozen VGG feature extractor
losses/mse_loss.py — MSE wrapper
losses/perceptual_loss.py — hybrid loss
metrics/psnr_ssim.py — PSNR/SSIM Y-channel
train/trainer.py — training loop with AMP, grad accum, checkpointing
train/train_all.py — run experiment matrix
infer/predict.py — single image inference
infer/benchmark.py — FPS benchmark
infer/onnx_export.py — export to ONNX
infer/opencv_dnn_infer.py — OpenCV DNN inference
infer/webcam_demo.py — live webcam SR
utils/cpu_affinity.py — torch.set_num_threads
utils/visualization.py — grids and zoom crops

## 13. Results Template
| Exp | Model | Scale | Loss | PSNR | SSIM | Params | FPS CPU | FPS GPU |
|-----|-------|-------|------|------|------|--------|---------|---------|

## 14. Troubleshooting
Windows DataLoader spawn error → num_workers=0
RTX 2050 OOM perceptual → reduce batch, VGG11, Colab T4
Color shift with perceptual → reduce λ, use hybrid
Overfitting → cross-dataset validation, augmentation

## 15. Future Roadmap
Video SR temporal consistency, Real-ESRGAN style degradation, TensorRT export, on-device edge deployment.

## References
SRCNN Dong 2016, ESPCN Shi 2016, SRGAN Ledig 2017, Perceptual Losses Johnson 2016, EDSR Lim 2017, VDSR Kim 2016.
