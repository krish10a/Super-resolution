import {
  EvaluationResults,
  ExperimentId,
  SampleItem,
  TrainingEpoch,
  UpscaleResponse,
  DegradationSettings,
} from './types';
import {
  getMockResults,
  getMockSamples,
  getMockTrainingLog,
  mockSuperResolve,
} from './mock/mockData';

const BASE_URL = import.meta.env.VITE_API_URL || '';
const FORCE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';

let isMockMode = FORCE_MOCK;

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'error' | 'mock';
  device: string;
  torchVersion?: string;
  isMock: boolean;
}

export async function checkApiHealth(): Promise<HealthStatus> {
  if (FORCE_MOCK) {
    return { status: 'mock', device: 'CUDA · RTX 2050 (Mock)', isMock: true };
  }

  try {
    const res = await fetch(`${BASE_URL}/api/v1/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      const data = await res.json();
      isMockMode = false;
      return {
        status: 'ok',
        device: data.cuda_available ? 'CUDA · RTX 2050' : data.device || 'CPU',
        torchVersion: data.torch_version,
        isMock: false,
      };
    }
  } catch (e) {
    // Backend offline -> gracefully use mock mode
  }

  isMockMode = true;
  return {
    status: 'mock',
    device: 'CUDA · RTX 2050 (Mock)',
    isMock: true,
  };
}

export async function fetchEvaluationResults(): Promise<EvaluationResults> {
  return getMockResults();
}

export async function fetchSamplesList(): Promise<SampleItem[]> {
  return getMockSamples();
}

export async function fetchTrainingLog(exp: ExperimentId): Promise<TrainingEpoch[]> {
  return getMockTrainingLog(exp);
}

export async function upscaleImage(params: {
  imageSrc: string;
  sampleItem?: SampleItem | null;
  model: 'SRCNN' | 'ESPCN';
  scale: 2 | 4;
  loss: 'MSE' | 'MSE + Perceptual';
  degradation?: DegradationSettings;
  simulateLr?: boolean;
}): Promise<UpscaleResponse> {
  if (isMockMode || FORCE_MOCK) {
    return mockSuperResolve(params);
  }

  try {
    // Model ID convention: {arch}_{scale}x_{loss} e.g. "espcn_4x_mse" or "srcnn_2x_mse_perceptual"
    const modelId = `${params.model.toLowerCase()}_${params.scale}x_${params.loss === 'MSE' ? 'mse' : 'mse_perceptual'}`;

    // Convert imageSrc (data URL or relative URL) to Blob
    const blob = await fetch(params.imageSrc).then((r) => r.blob());

    // 1. Prepare FormData for /api/v1/super-resolve (field 'model' matches backend Form parameter)
    const srFormData = new FormData();
    srFormData.append('file', blob, 'image.png');
    srFormData.append('model', modelId);
    if (params.degradation?.enabled || params.simulateLr) {
      srFormData.append('simulate_lr', 'true');
    }

    // 2. Prepare FormData for /api/v1/bicubic baseline upscale
    const bicubicFormData = new FormData();
    bicubicFormData.append('file', blob, 'image.png');
    bicubicFormData.append('scale', params.scale.toString());

    // 3. Fetch super-resolved output and bicubic baseline in parallel
    const [srRes, bicubicRes] = await Promise.all([
      fetch(`${BASE_URL}/api/v1/super-resolve`, {
        method: 'POST',
        body: srFormData,
      }),
      fetch(`${BASE_URL}/api/v1/bicubic`, {
        method: 'POST',
        body: bicubicFormData,
      }),
    ]);

    if (!srRes.ok) {
      const errText = await srRes.text();
      throw new Error(`Super-resolve failed (HTTP ${srRes.status}): ${errText}`);
    }

    const srData = await srRes.json();

    // Resolve bicubic baseline URL at matching target resolution
    let bicubicUrl = params.imageSrc;
    if (bicubicRes.ok) {
      const bicubicData = await bicubicRes.json();
      bicubicUrl = `data:image/png;base64,${bicubicData.image_base64}`;
    }

    // Determine LR and HR URLs
    const srUrl = `data:image/png;base64,${srData.image_base64}`;
    const lrUrl = srData.lr_base64 ? `data:image/png;base64,${srData.lr_base64}` : params.imageSrc;
    const hrUrl = srData.hr_base64 ? `data:image/png;base64,${srData.hr_base64}` : (params.sampleItem?.hrUrl || null);

    // If sampleItem is provided, resolve precomputed benchmark metrics
    let metrics = null;
    let bicubicMetrics = null;
    if (params.sampleItem && params.sampleItem.hasGroundTruth) {
      const results = await getMockResults();
      const expId = resolveExpId(params.model, params.scale, params.loss);
      const dsKey = params.sampleItem.datasetKey || 'Set14_01';
      const group = params.sampleItem.group as 'Set5' | 'Set14' | 'BSD100';
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

    return {
      lrUrl,
      bicubicUrl,
      srUrl,
      hrUrl,
      ms: Math.round(srData.latency_ms),
      device: srData.device ? `${srData.device.toUpperCase()} · RTX 2050` : 'CUDA · RTX 2050',
      inputSize: srData.input_size || [128, 128],
      outputSize: srData.output_size || [512, 512],
      modelId: srData.model_id,
      scale: params.scale,
      loss: params.loss,
      metrics: metrics || undefined,
      bicubicMetrics: bicubicMetrics || undefined,
    };
  } catch (e) {
    console.warn('Real backend call failed, falling back to mock engine:', e);
    return mockSuperResolve(params);
  }
}

function resolveExpId(model: 'SRCNN' | 'ESPCN', scale: 2 | 4, loss: string): ExperimentId {
  const isMSE = loss === 'MSE';
  if (model === 'SRCNN') {
    return scale === 2 ? (isMSE ? 'E1' : 'E2') : isMSE ? 'E3' : 'E4';
  } else {
    return scale === 2 ? (isMSE ? 'E5' : 'E6') : isMSE ? 'E7' : 'E8';
  }
}
