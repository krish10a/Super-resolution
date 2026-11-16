import { useState, useEffect } from 'react';
import { EvaluationResults, ExperimentId } from '../lib/types';
import { fetchEvaluationResults } from '../lib/api';
import {
  computeBenchmarkHeadline,
  computeGlobalStats,
  computeRank,
} from '../lib/metricsMath';

export function useResults() {
  const [results, setResults] = useState<EvaluationResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    fetchEvaluationResults()
      .then((data) => {
        if (mounted) {
          setResults(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message || 'Failed to load evaluation results');
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  const headline = computeBenchmarkHeadline(results);
  const globalStats = computeGlobalStats(results);

  const getRank = (
    expId: ExperimentId,
    dataset: 'Set5' | 'Set14' | 'BSD100',
    metric: 'psnr' | 'ssim' | 'lpips' = 'psnr'
  ) => computeRank(results, expId, dataset, metric);

  return {
    results,
    loading,
    error,
    headline,
    globalStats,
    getRank,
  };
}

