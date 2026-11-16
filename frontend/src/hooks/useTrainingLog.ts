import { useState, useEffect } from 'react';
import { ExperimentId, TrainingEpoch } from '../lib/types';
import { fetchTrainingLog } from '../lib/api';

const logCache: Partial<Record<ExperimentId, TrainingEpoch[]>> = {};

export function useTrainingLog(expId: ExperimentId | null) {
  const [epochs, setEpochs] = useState<TrainingEpoch[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!expId) {
      setEpochs([]);
      return;
    }

    if (logCache[expId]) {
      setEpochs(logCache[expId]!);
      return;
    }

    let mounted = true;
    setLoading(true);
    fetchTrainingLog(expId)
      .then((data) => {
        if (mounted) {
          logCache[expId] = data;
          setEpochs(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message || 'Failed to load training log');
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, [expId]);

  // Perceptual runs (E2, E4, E6, E8) have 3-epoch warmup, boundary at epoch 3.5 or 4
  const hasWarmup = expId ? ['E2', 'E4', 'E6', 'E8'].includes(expId) : false;

  return { epochs, loading, error, hasWarmup };
}

