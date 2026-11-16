import { useState, useCallback } from 'react';
import {
  ModelName,
  Scale,
  LossName,
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
      loss: LossName,
      sampleItem?: SampleItem | null,
      degradation?: DegradationSettings,
      simulateLr?: boolean
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
      simulateLr?: boolean
    ) => {
      setLoading(true);
      setError(null);
      try {
        const models: { model: ModelName; loss: LossName; key: string }[] = [
          { model: 'SRCNN', loss: 'MSE', key: 'srcnn_mse' },
          { model: 'SRCNN', loss: 'MSE + Perceptual', key: 'srcnn_perc' },
          { model: 'ESPCN', loss: 'MSE', key: 'espcn_mse' },
          { model: 'ESPCN', loss: 'MSE + Perceptual', key: 'espcn_perc' },
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

