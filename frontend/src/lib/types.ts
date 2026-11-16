export type ExperimentId = 'E1' | 'E2' | 'E3' | 'E4' | 'E5' | 'E6' | 'E7' | 'E8';
export type ModelName = 'SRCNN' | 'ESPCN';
export type Scale = 2 | 4;
export type LossName = 'MSE' | 'MSE + Perceptual';

export interface Metrics {
  psnr: number;
  ssim: number;
  lpips?: number;
}

export interface PerImageResult {
  image: string;
  psnr: number;
  ssim: number;
  lpips?: number;
  hr_size: [number, number];
  lr_size: [number, number];
}

export interface DatasetSummary {
  avg_psnr: number;
  avg_ssim: number;
  avg_lpips?: number;
  n_images: number;
  bicubic_psnr: number;
  bicubic_ssim: number;
  psnr_delta: number;
  ssim_delta: number;
  beats_bicubic_psnr: string | boolean;
  beats_bicubic_ssim: string | boolean;
  per_image: PerImageResult[];
}

export interface ExperimentConfig {
  model: string;
  scale: number;
  ckpt: string;
  loss: string;
}

export interface ExperimentData {
  config: ExperimentConfig;
  results: {
    Set5: DatasetSummary;
    Set14: DatasetSummary;
    BSD100: DatasetSummary;
  };
}

export type EvaluationResults = Record<ExperimentId, ExperimentData>;

export interface TrainingEpoch {
  epoch: number;
  train_loss: number;
  val_psnr: number;
  val_ssim: number;
  best_psnr: number;
}

export interface SampleItem {
  id: string;
  group: 'Set5' | 'Set14' | 'BSD100' | 'Surveillance';
  name: string;
  thumbUrl: string;
  lrUrl?: string;
  lrUrls?: Record<string, string>;
  hrUrl?: string;
  isRealCctv?: boolean;
  hasGroundTruth?: boolean;
  datasetKey?: string;
  width?: number;
  height?: number;
}

export interface UpscaleResponse {
  lrUrl: string;
  bicubicUrl: string;
  srUrl: string;
  hrUrl?: string | null;
  metrics?: Metrics | null;
  bicubicMetrics?: Metrics | null;
  ms: number;
  device: string;
  inputSize: [number, number];
  outputSize: [number, number];
  modelId: string;
  scale: number;
  loss: string;
}

export interface DegradationSettings {
  enabled: boolean;
  blurSigma: number;      // 0.0 - 3.0
  noiseSigma: number;     // 0 - 30
  jpegQuality: number;    // 10 - 100
}

export type ViewerMode = 'slider' | 'side-by-side' | 'flicker' | 'error-map';
export type CompareAgainst = 'bicubic' | 'input' | 'ground-truth';

