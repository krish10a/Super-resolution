import React from 'react';
import { SampleItem } from '../../lib/types';

export interface SampleGalleryProps {
  samples: SampleItem[];
  selectedId: string | null;
  onSelect: (sample: SampleItem) => void;
  className?: string;
}

export const SampleGallery: React.FC<SampleGalleryProps> = ({
  samples,
  selectedId,
  onSelect,
  className = '',
}) => {
  // Group by dataset
  const groups: Record<string, SampleItem[]> = {
    Surveillance: [],
    Set5: [],
    Set14: [],
    BSD100: [],
  };

  samples.forEach((s) => {
    if (groups[s.group]) {
      groups[s.group].push(s);
    }
  });

  return (
    <div className={`space-y-2.5 ${className}`}>
      <div className="flex items-center justify-between text-xs text-[var(--muted)] font-medium">
        <span>Curated Presets</span>
        <span className="text-[10px] font-mono">Set5 / Set14 / CCTV</span>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-2 pt-1 no-scrollbar scroll-smooth">
        {Object.entries(groups).map(([groupName, items]) => {
          if (items.length === 0) return null;
          return (
            <div key={groupName} className="flex items-center gap-1.5 shrink-0">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--muted)] -rotate-90 origin-center w-3 shrink-0">
                {groupName.slice(0, 3)}
              </span>
              <div className="flex gap-1.5">
                {items.map((item) => {
                  const isSelected = item.id === selectedId;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => onSelect(item)}
                      title={`${item.name} (${item.group})`}
                      className={`group relative w-16 h-16 rounded-[var(--radius-control)] overflow-hidden border transition-all shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${
                        isSelected
                          ? 'border-[var(--accent)] ring-1 ring-[var(--accent)] shadow-[0_0_10px_rgba(56,225,255,0.3)]'
                          : 'border-[var(--border)] hover:border-[var(--muted)] opacity-70 hover:opacity-100'
                      }`}
                    >
                      <img
                        src={item.thumbUrl}
                        alt={item.name}
                        className="w-full h-full object-cover bg-[var(--bg-base)]"
                        loading="lazy"
                      />
                      <div className="absolute inset-x-0 bottom-0 bg-black/80 px-1 py-0.5 text-[9px] font-mono text-[var(--text)] truncate text-center">
                        {item.name}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

