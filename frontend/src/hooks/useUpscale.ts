import { useState, useCallback } from 'react';
import {
  ModelName,
  Scale,
  LossName,
  DenoiseOption,
  SampleItem,
  UpscaleResponse,
  DegradationSettings,
} from '../lib/types';
import { upscaleImage } from '../lib/api';

export function useUpscale() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeResult, setActiveResult] = useState<UpscaleResponse | null>(null);
  const [compare4Results, setCompare4Results] = useState<Record<string, UpscaleResponse>>({});

  const runUpscale = useCallback(
    async (
      imageSrc: string,
      model: ModelName,
      scale: Scale,
      loss?: LossName,
      sampleItem?: SampleItem | null,
      degradation?: DegradationSettings,
      simulateLr?: boolean,
      denoise?: DenoiseOption,
      isLegacy?: boolean
    ) => {
      setLoading(true);
      setError(null);
      try {
        const res = await upscaleImage({
          imageSrc,
          model,
          scale,
          loss,
          sampleItem,
          degradation,
          simulateLr,
          denoise,
          isLegacy,
        });
        setActiveResult(res);
        return res;
      } catch (err: any) {
        setError(err.message || 'Upscaling failed');
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const runCompare4 = useCallback(
    async (
      imageSrc: string,
      scale: Scale,
      sampleItem?: SampleItem | null,
      degradation?: DegradationSettings,
      simulateLr?: boolean,
      denoise?: DenoiseOption,
      isLegacy?: boolean
    ) => {
      setLoading(true);
      setError(null);
      try {
        const models: { model: ModelName; loss?: LossName; key: string; isLegacy?: boolean }[] = isLegacy
          ? [
              { model: 'SRCNN', loss: 'MSE', key: 'srcnn_mse', isLegacy: true },
              { model: 'SRCNN', loss: 'MSE + Perceptual', key: 'srcnn_perc', isLegacy: true },
              { model: 'ESPCN', loss: 'MSE', key: 'espcn_mse', isLegacy: true },
              { model: 'ESPCN', loss: 'MSE + Perceptual', key: 'espcn_perc', isLegacy: true },
            ]
          : [
              { model: 'EDSR', loss: 'L1', key: 'edsr_l1', isLegacy: false },
              { model: 'FSRCNN', loss: 'Charbonnier', key: 'fsrcnn_charb', isLegacy: false },
              { model: 'ESPCN', loss: 'Charbonnier', key: 'espcn_charb', isLegacy: false },
              { model: 'SRCNN', loss: 'MSE', key: 'srcnn_legacy', isLegacy: true },
            ];

        const results: Record<string, UpscaleResponse> = {};
        await Promise.all(
          models.map(async (m) => {
            const res = await upscaleImage({
              imageSrc,
              model: m.model,
              scale,
              loss: m.loss,
              sampleItem,
              degradation,
              simulateLr,
              denoise,
              isLegacy: m.isLegacy,
            });
            results[m.key] = res;
          })
        );
        setCompare4Results(results);
        return results;
      } catch (err: any) {
        setError(err.message || 'Compare 4 failed');
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    activeResult,
    compare4Results,
    runUpscale,
    runCompare4,
    setActiveResult,
  };
}

