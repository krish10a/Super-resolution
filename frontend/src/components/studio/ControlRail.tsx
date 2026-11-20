import React, { useState } from 'react';
import { ModelName, Scale, LossName, DenoiseOption, SampleItem, DegradationSettings } from '../../lib/types';
import { resolveExperiment } from '../../lib/metricsMath';
import { Segmented } from '../ui/Segmented';
import { Dropzone } from '../ui/Dropzone';
import { Slider } from '../ui/Slider';
import { Button } from '../ui/Button';
import { SampleGallery } from './SampleGallery';

export interface ControlRailProps {
  model: ModelName;
  onModelChange: (model: ModelName) => void;
  scale: Scale;
  onScaleChange: (scale: Scale) => void;
  loss?: LossName;
  onLossChange?: (loss: LossName) => void;
  denoise?: DenoiseOption;
  onDenoiseChange?: (denoise: DenoiseOption) => void;
  isLegacy?: boolean;
  onIsLegacyChange?: (isLegacy: boolean) => void;
  samples: SampleItem[];
  selectedSample: SampleItem | null;
  onSampleSelect: (sample: SampleItem) => void;
  customImage: string | null;
  customDimensions: [number, number] | null;
  onCustomImageUpload: (file: File, dataUrl: string) => void;
  onClearCustomImage: () => void;
  degradation: DegradationSettings;
  onDegradationChange: (degradation: DegradationSettings) => void;
  onEnhance: () => void;
  loading: boolean;
  className?: string;
}

export const ControlRail: React.FC<ControlRailProps> = ({
  model,
  onModelChange,
  scale,
  onScaleChange,
  loss,
  onLossChange,
  denoise = 'none',
  onDenoiseChange,
  isLegacy = false,
  onIsLegacyChange,
  samples,
  selectedSample,
  onSampleSelect,
  customImage,
  customDimensions,
  onCustomImageUpload,
  onClearCustomImage,
  degradation,
  onDegradationChange,
  onEnhance,
  loading,
  className = '',
}) => {
  const [degradeOpen, setDegradeOpen] = useState(false);
  const resolved = resolveExperiment(model, scale, loss, isLegacy);

  // Degradation is enabled only when using sample presets (not raw uploads)
  const degradationAvailable = Boolean(selectedSample && !customImage);

  return (
    <aside
      className={`flex flex-col h-full bg-[var(--surface)] border-r border-[var(--border)] overflow-y-auto ${className}`}
    >
      <div className="flex-1 p-5 space-y-6">
        {/* Section 1: Source Input */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
            <span>1. Source Image</span>
          </div>

          <Dropzone
            onFileSelect={onCustomImageUpload}
            selectedPreview={customImage}
            dimensions={customDimensions}
            onClear={onClearCustomImage}
          />

          {!customImage && (
            <SampleGallery
              samples={samples}
              selectedId={selectedSample?.id || null}
              onSelect={onSampleSelect}
            />
          )}
        </div>

        {/* Section 2: Model Architecture */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
              2. Architecture
            </label>
            {onIsLegacyChange && (
              <label className="flex items-center gap-1.5 text-[11px] text-[var(--muted)] cursor-pointer">
                <input
                  type="checkbox"
                  checked={isLegacy}
                  onChange={(e) => {
                    const legacy = e.target.checked;
                    onIsLegacyChange(legacy);
                    if (legacy) {
                      onModelChange('SRCNN');
                      onLossChange?.('MSE');
                    } else {
                      onModelChange('EDSR');
                      onLossChange?.('L1');
                    }
                  }}
                  className="rounded border-[var(--border)] text-[var(--accent)] text-xs"
                />
                <span>Legacy</span>
              </label>
            )}
          </div>
          <div className="w-full">
            <Segmented
              id="model-selector"
              options={
                isLegacy
                  ? [
                      { value: 'SRCNN', label: 'SRCNN' },
                      { value: 'ESPCN', label: 'ESPCN' },
                    ]
                  : [
                      { value: 'EDSR', label: 'EDSR (Quality)' },
                      { value: 'ESPCN', label: 'ESPCN (Fast)' },
                      { value: 'FSRCNN', label: 'FSRCNN (Balanced)' },
                    ]
              }
              value={model}
              onChange={onModelChange}
              className="w-full justify-between"
            />
          </div>
        </div>

        {/* Section 3: Upscale Factor */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
            3. Scale Factor
          </label>
          <div className="w-full">
            <Segmented
              id="scale-selector"
              options={[
                { value: 2, label: '2× Upscale' },
                { value: 4, label: '4× Upscale' },
              ]}
              value={scale}
              onChange={onScaleChange}
              className="w-full justify-between"
            />
          </div>
        </div>

        {/* Section 4: CCTV Pre-Filter (Phase 4.3 Denoising) */}
        {onDenoiseChange && (
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
              4. CCTV Pre-Filter
            </label>
            <div className="w-full">
              <Segmented
                id="denoise-selector"
                options={[
                  { value: 'none', label: 'Off' },
                  { value: 'bilateral', label: 'Bilateral (+1.35 dB)' },
                ]}
                value={denoise}
                onChange={onDenoiseChange}
                className="w-full justify-between"
              />
            </div>
            <p className="text-[11px] text-[var(--muted)]">
              Edge-preserving filter before SISR inference. Rescues sensor & CCTV noise (+0.86ms).
            </p>
          </div>
        )}

        {/* Section 5: Training Loss (Legacy Only) */}
        {isLegacy && onLossChange && (
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
              5. Supervision Loss
            </label>
            <div className="w-full">
              <Segmented
                id="loss-selector"
                options={[
                  { value: 'MSE', label: 'MSE' },
                  { value: 'MSE + Perceptual', label: 'MSE + Perc' },
                ]}
                value={loss || 'MSE'}
                onChange={onLossChange}
                className="w-full justify-between"
              />
            </div>
          </div>
        )}

        {/* Section 6: Resolved Checkpoint */}
        <div className="p-3 rounded-[var(--radius-control)] bg-[var(--bg-base)] border border-[var(--border)] space-y-1">
          <div className="flex items-center justify-between text-[11px] text-[var(--muted)]">
            <span>Experiment</span>
            <span className="font-mono text-[var(--accent)] font-semibold">{resolved.id}</span>
          </div>
          <div className="font-mono text-xs text-[var(--text)] truncate" title={resolved.checkpoint}>
            {resolved.checkpoint}
          </div>
        </div>

        {/* Section 7: Surveillance Degradation Simulator */}
        <div className="border border-[var(--border)] rounded-[var(--radius-control)] bg-[var(--raised)]/40 overflow-hidden">
          <button
            type="button"
            onClick={() => setDegradeOpen(!degradeOpen)}
            disabled={!degradationAvailable}
            className={`w-full flex items-center justify-between p-3 text-xs font-medium text-left transition-colors ${
              !degradationAvailable
                ? 'opacity-40 cursor-not-allowed'
                : 'hover:bg-[var(--raised)] text-[var(--text)]'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[var(--amber)]" />
              <span>Simulate Surveillance Degradation</span>
            </div>
            <svg
              className={`w-4 h-4 text-[var(--muted)] transition-transform ${degradeOpen ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {degradeOpen && degradationAvailable && (
            <div className="p-3 pt-1 space-y-3 border-t border-[var(--border)]">
              <div className="flex items-center justify-between py-1 text-xs">
                <span className="text-[var(--muted)]">Enable Degradation</span>
                <input
                  type="checkbox"
                  checked={degradation.enabled}
                  onChange={(e) =>
                    onDegradationChange({ ...degradation, enabled: e.target.checked })
                  }
                  className="rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
                />
              </div>

              <Slider
                label="Gaussian Blur σ"
                min={0}
                max={3.0}
                step={0.1}
                value={degradation.blurSigma}
                disabled={!degradation.enabled}
                onChange={(v) => onDegradationChange({ ...degradation, blurSigma: v })}
              />

              <Slider
                label="Gaussian Noise σ"
                min={0}
                max={30}
                step={1}
                value={degradation.noiseSigma}
                disabled={!degradation.enabled}
                onChange={(v) => onDegradationChange({ ...degradation, noiseSigma: v })}
              />

              <Slider
                label="JPEG Quality"
                min={10}
                max={100}
                step={5}
                value={degradation.jpegQuality}
                disabled={!degradation.enabled}
                onChange={(v) => onDegradationChange({ ...degradation, jpegQuality: v })}
              />
            </div>
          )}
        </div>
      </div>

      {/* Sticky Bottom Action */}
      <div className="p-5 border-t border-[var(--border)] bg-[var(--surface)] sticky bottom-0">
        <Button
          variant="primary"
          size="lg"
          loading={loading}
          onClick={onEnhance}
          className="w-full font-semibold shadow-lg"
        >
          Enhance Resolution
        </Button>
      </div>
    </aside>
  );
};


