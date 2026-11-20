import {
  EvaluationResults,
  ExperimentId,
  ModelName,
  Scale,
  LossName,
  DenoiseOption,
  SampleItem,
  TrainingEpoch,
  UpscaleResponse,
  DegradationSettings,
} from '../types';
import { resolveExperiment } from '../metricsMath';

let cachedResults: EvaluationResults | null = null;
let cachedSamples: SampleItem[] | null = null;
let cachedBaselines: any = null;

export async function getMockResults(): Promise<EvaluationResults> {
  if (cachedResults) return cachedResults;
  // Try /data/ then /mock/
  let res = await fetch('/data/evaluation_results.json');
  if (!res.ok) {
    res = await fetch('/mock/evaluation_results.json');
  }
  if (!res.ok) throw new Error('Failed to load evaluation_results.json');
  cachedResults = await res.json();
  return cachedResults!;
}

export async function getMockBaselines(): Promise<any> {
  if (cachedBaselines) return cachedBaselines;
  try {
    let res = await fetch('/data/matlab_bicubic_baselines.json');
    if (!res.ok) {
      res = await fetch('/mock/matlab_bicubic_baselines.json');
    }
    if (res.ok) {
      cachedBaselines = await res.json();
      return cachedBaselines;
    }
  } catch (e) {
    // fallback if missing
  }
  return null;
}

export async function getMockSamples(): Promise<SampleItem[]> {
  if (cachedSamples) return cachedSamples;
  const res = await fetch('/samples/samples.json');
  if (!res.ok) throw new Error('Failed to load samples.json');
  cachedSamples = await res.json();
  return cachedSamples!;
}

export async function getMockTrainingLog(exp: ExperimentId): Promise<TrainingEpoch[]> {
  const fileName = `${exp.toLowerCase()}_train.log`;
  let res = await fetch(`/data/${fileName}`);
  if (!res.ok) {
    res = await fetch(`/mock/${fileName}`);
  }
  if (!res.ok) throw new Error(`Failed to load ${fileName}`);
  const text = await res.text();
  const lines = text.trim().split('\n');
  const epochs: TrainingEpoch[] = [];

  // Format: epoch,train_loss,val_psnr,val_ssim,best_psnr
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    if (parts.length >= 4) {
      epochs.push({
        epoch: parseInt(parts[0], 10),
        train_loss: parseFloat(parts[1]),
        val_psnr: parseFloat(parts[2]),
        val_ssim: parseFloat(parts[3]),
        best_psnr: parseFloat(parts[4] === '-inf' ? '0' : parts[4] || '0'),
      });
    }
  }
  return epochs;
}

/**
 * Client-side mock upscale using Canvas Bicubic + Unsharp Mask.
 * Simulates deep learning SR inference with 600-1200ms delay.
 */
export async function mockSuperResolve(params: {
  imageSrc: string;
  sampleItem?: SampleItem | null;
  model: ModelName;
  scale: Scale;
  loss?: LossName;
  denoise?: DenoiseOption;
  degradation?: DegradationSettings;
  isLegacy?: boolean;
}): Promise<UpscaleResponse> {
  const startTime = performance.now();
  const delay = 400 + Math.random() * 300;
  await new Promise((r) => setTimeout(r, delay));

  const img = await loadImage(params.imageSrc);
  const srcW = img.width;
  const srcH = img.height;
  const targetW = srcW * params.scale;
  const targetH = srcH * params.scale;

  // 1. Generate Bicubic Upscale on Canvas
  const bicubicCanvas = document.createElement('canvas');
  bicubicCanvas.width = targetW;
  bicubicCanvas.height = targetH;
  const bicubicCtx = bicubicCanvas.getContext('2d')!;
  bicubicCtx.imageSmoothingEnabled = true;
  bicubicCtx.imageSmoothingQuality = 'high';
  bicubicCtx.drawImage(img, 0, 0, targetW, targetH);
  const bicubicUrl = bicubicCanvas.toDataURL('image/png');

  // 2. Generate Model Super-Resolution with edge enhancement / unsharp mask
  const srCanvas = document.createElement('canvas');
  srCanvas.width = targetW;
  srCanvas.height = targetH;
  const srCtx = srCanvas.getContext('2d', { willReadFrequently: true })!;
  srCtx.imageSmoothingEnabled = true;
  srCtx.imageSmoothingQuality = 'high';
  srCtx.drawImage(img, 0, 0, targetW, targetH);

  // Apply convolution unsharp mask
  const sharpness = params.model === 'EDSR' ? 1.6 : params.model === 'FSRCNN' ? 1.3 : 1.1;
  applyUnsharpMask(srCtx, targetW, targetH, sharpness);
  const srUrl = srCanvas.toDataURL('image/png');

  // 3. Resolve ground truth & metrics if sample item has GT
  const resolved = resolveExperiment(params.model, params.scale, params.loss, params.isLegacy);
  const sample = params.sampleItem;
  let metrics = null;
  let bicubicMetrics = null;
  let hrUrl: string | null = null;

  if (sample && sample.hasGroundTruth) {
    hrUrl = sample.hrUrl || null;
    const results = await getMockResults();
    const expId = resolved.id;
    const dsKey = sample.datasetKey || 'Set5_00';
    const group = sample.group as 'Set5' | 'Set14' | 'BSD100';

    const expData = results[expId]?.results?.[group];
    const itemResult = expData?.per_image?.find((p) => p.image === dsKey);

    if (itemResult) {
      metrics = {
        psnr: Math.round(itemResult.psnr * 100) / 100,
        ssim: Math.round(itemResult.ssim * 10000) / 10000,
        lpips: expData.avg_lpips ? Math.round(expData.avg_lpips * 1000) / 1000 : 0.128,
      };
      bicubicMetrics = {
        psnr: Math.round(expData.bicubic_psnr * 100) / 100,
        ssim: Math.round(expData.bicubic_ssim * 10000) / 10000,
        lpips: 0.155,
      };
    }
  }

  const duration = Math.round(performance.now() - startTime);

  return {
    lrUrl: params.imageSrc,
    bicubicUrl,
    srUrl,
    hrUrl,
    metrics,
    bicubicMetrics,
    ms: Math.min(duration, 85),
    device: 'CUDA · RTX 2050',
    inputSize: [srcW, srcH],
    outputSize: [targetW, targetH],
    modelId: resolved.modelId,
    scale: params.scale,
    loss: params.loss || 'L1',
  };
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image for upscaling'));
    img.src = src;
  });
}

function applyUnsharpMask(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  amount: number = 1.2
) {
  try {
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;
    const copy = new Uint8ClampedArray(data);

    // 3x3 Laplacian sharpening kernel
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) {
          const center = copy[idx + c];
          const top = copy[((y - 1) * w + x) * 4 + c];
          const bottom = copy[((y + 1) * w + x) * 4 + c];
          const left = copy[(y * w + (x - 1)) * 4 + c];
          const right = copy[(y * w + (x + 1)) * 4 + c];

          const laplacian = 4 * center - top - bottom - left - right;
          data[idx + c] = Math.max(0, Math.min(255, center + laplacian * (amount * 0.2)));
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  } catch (e) {
    // Canvas security/CORS fallback
  }
}
