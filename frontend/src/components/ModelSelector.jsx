/**
 * Model + scale selector. Options are derived from /api/v1/models availability;
 * unavailable combinations are disabled with a tooltip.
 * Emits the selected model_id via onSelect(modelId).
 *
 * Resilient to backend/proxy failures: shows a "Try again" button on error
 * so the user can recover without a full page reload.
 */
import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';
import { Info, ShieldOff, AlertCircle, RotateCw } from 'lucide-react';
import { Card, Button, Spinner } from '../components/ui';

function cn(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function ModelSelector({ onSelect, disabled }) {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const mountedRef = useRef(true);

  const fetchModels = () => {
    setError(null);
    setLoading(true);
    console.log('[ModelSelector] fetching models');
    api.getModels()
      .then((m) => {
        if (!mountedRef.current) return;
        console.log('[ModelSelector] models received:', m.length);
        setModels(m);
        const firstAvail = m.find((x) => x.available) || m[0];
        if (firstAvail) {
          setSelected(firstAvail.model_id);
          onSelect(firstAvail.model_id);
        }
        setLoading(false);
      })
      .catch((err) => {
        if (!mountedRef.current) return;
        console.error('[ModelSelector] failed to load models:', err);
        setError(err.message || 'Failed to load models.');
        setLoading(false);
      });
  };

  useEffect(() => {
    mountedRef.current = true;
    fetchModels();
    return () => { mountedRef.current = false; };
  }, [onSelect]);

  useEffect(() => {
    if (selected && onSelect) onSelect(selected);
  }, [selected, onSelect]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-8">
        <Spinner className="mb-2" />
        <p className="text-sm text-slate-500">Loading models…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 rounded-xl border border-amber-200 bg-amber-50">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle size={20} />
          <span className="text-sm font-medium text-amber-800">{error}</span>
        </div>
        <Button variant="outline" size="sm" onClick={() => fetchModels()} className="w-full">
          <RotateCw size={14} className="mr-2" /> Try again
        </Button>
      </div>
    );
  }

  // Group models by scale
  const groups = {};
  models.forEach((m) => {
    (groups[m.scale] || (groups[m.scale] = [])).push(m);
  });

  const scaleGroups = Object.entries(groups);

  return (
    <div className="space-y-4">
      {scaleGroups.map(([scale, scaleModels]) => {
        // Group by architecture within this scale
        const archs = {};
        scaleModels.forEach((m) => {
          (archs[m.architecture] || (archs[m.architecture] = [])).push(m);
        });

        return (
          <div key={scale} className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                Scale {scale}×
              </span>
              <span className="ml-auto text-xs text-slate-400">
                {scaleModels.length} model(s)
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {Object.entries(archs).map(([arch, archModels]) => (
                <div key={arch}>
                  {archModels.map((m) => (
                    <button
                      key={m.model_id}
                      onClick={() => setSelected(m.model_id)}
                      disabled={!m.available || disabled}
                      className={cn(
                        'flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-200',
                        selected === m.model_id
                          ? 'bg-blue-50 border-blue-200 shadow-sm shadow-blue-100/50'
                          : 'bg-white border-slate-200 hover:border-slate-300 hover:shadow-md',
                        !m.available ? 'opacity-50 cursor-not-allowed' : ''
                      )}
                      title={!m.available ? 'Checkpoint not loadable' : undefined}
                    >
                      <div className="text-sm font-semibold text-slate-800">
                        {m.display_name}
                      </div>
                      <div className="text-xs text-slate-500">
                        {m.checkpoint}
                      </div>
                      {!m.available && (
                        <div className="flex items-center gap-1 text-slate-400 mt-1">
                          <ShieldOff size={12} />
                          <span className="text-xs">Unavailable</span>
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
