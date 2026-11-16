import React from 'react';
import { motion } from 'framer-motion';

export interface TabItem<T extends string> {
  id: T;
  label: React.ReactNode;
  count?: number;
}

export interface TabsProps<T extends string> {
  tabs: TabItem<T>[];
  activeTab: T;
  onChange: (tabId: T) => void;
  id?: string;
  className?: string;
}

export function Tabs<T extends string>({
  tabs,
  activeTab,
  onChange,
  id = 'tabs',
  className = '',
}: TabsProps<T>) {
  return (
    <div
      role="tablist"
      className={`flex items-center gap-1 border-b border-[var(--border)] ${className}`}
    >
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(tab.id)}
            className={`relative py-2.5 px-4 text-sm font-medium transition-colors select-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent)] ${
              isActive ? 'text-[var(--accent)] font-semibold' : 'text-[var(--muted)] hover:text-[var(--text)]'
            }`}
          >
            <span className="flex items-center gap-2">
              {tab.label}
              {typeof tab.count === 'number' && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[var(--raised)] rounded-full border border-[var(--border)]">
                  {tab.count}
                </span>
              )}
            </span>

            {isActive && (
              <motion.div
                layoutId={`active-tab-line-${id}`}
                className="absolute bottom-0 left-0 right-0 h-[2px] bg-[var(--accent)]"
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

