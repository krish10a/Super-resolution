/**
 * Shared UI primitives for the SISR Studio.
 * Clean, professional design system with explicit styling.
 */
import React from 'react';
import { cn } from '../lib/utils';

export function Card({ className, children, ...props }) {
  return (
    <div
      {...props}
      className={cn(
        'bg-white rounded-2xl border border-slate-200 shadow-sm p-6',
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className }) {
  return (
    <div className={cn('mb-4 flex items-center gap-2', className)}>{children}</div>
  );
}

export function CardTitle({ children, className }) {
  return (
    <h3 className={cn('text-lg font-semibold text-slate-900 tracking-tight', className)}>
      {children}
    </h3>
  );
}

export function Badge({ children, variant = 'default', className }) {
  const variants = {
    default: 'bg-slate-100 text-slate-600 border border-slate-200',
    available: 'bg-emerald-50 text-emerald-700 border border-emerald-100',
    unavailable: 'bg-slate-100 text-slate-400 border border-slate-200',
    brand: 'bg-blue-50 text-blue-700 border border-blue-100',
  };
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium border',
        variants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Button({ className, variant = 'default', size = 'default', children, ...props }) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed';
  const variants = {
    default:
      'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 shadow-sm hover:shadow-md active:scale-95',
    secondary:
      'bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 focus:ring-slate-300 hover:border-slate-300 active:scale-95',
    ghost:
      'bg-transparent text-slate-600 hover:bg-slate-100 focus:ring-slate-300 active:scale-95',
    danger:
      'bg-red-50 text-red-700 border border-red-100 hover:bg-red-100 focus:ring-red-200 active:scale-95',
    outline:
      'bg-transparent text-blue-600 border border-blue-200 hover:bg-blue-50 focus:ring-blue-500 active:scale-95',
  };
  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    default: 'px-4 py-2.5 text-sm',
    lg: 'px-6 py-3 text-base',
  };
  return (
    <button className={cn(base, variants[variant], sizes[size], className)} {...props}>
      {children}
    </button>
  );
}

export function Input({ className, ...props }) {
  return (
    <input
      {...props}
      className={cn(
        'block w-full px-3 py-2.5 text-sm text-slate-700 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors',
        className,
      )}
    />
  );
}

export function Label({ children, className }) {
  return (
    <label className={cn('block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5', className)}>
      {children}
    </label>
  );
}

export function StatCard({ icon, value, label, desc }) {
  return (
    <Card className="p-5 border border-slate-100">
      <div className="flex items-center justify-between mb-4">
        <div className="p-3 rounded-xl bg-blue-50 text-blue-600">{icon}</div>
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{value}</span>
      </div>
      <p className="text-sm font-medium text-slate-800">{label}</p>
      <p className="text-xs text-slate-500 mt-1">{desc}</p>
    </Card>
  );
}

export function SectionTitle({ children, className }) {
  return (
    <h2 className={cn('text-2xl font-bold text-slate-900 tracking-tight mb-1', className)}>
      {children}
    </h2>
  );
}

export function SectionSubtitle({ children, className }) {
  return (
    <p className={cn('text-sm text-slate-500 mt-0.5 mb-6', className)}>{children}</p>
  );
}

export function EmptyState({ icon, title, desc, action }) {
  return (
    <Card className="text-center p-12 border-dashed border-2 border-slate-200 bg-slate-50">
      <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4 text-slate-400">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-slate-900 mb-1">{title}</h3>
      <p className="text-sm text-slate-500 max-w-xs mx-auto mb-6">{desc}</p>
      {action}
    </Card>
  );
}

export function ImagePanel({ src, alt, onDownload, meta, className }) {
  return (
    <div
      className={cn(
        'relative group rounded-2xl border border-slate-200 bg-slate-50 overflow-hidden aspect-[4/3]',
        className,
      )}
    >
      <img src={src} alt={alt} className="w-full h-full object-contain bg-white" />
      <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
        <button
          className="w-10 h-10 rounded-full bg-white/90 backdrop-blur flex items-center justify-center text-slate-600 hover:bg-white hover:text-blue-600 transition-all shadow-lg"
          onClick={(e) => { e.stopPropagation(); onDownload?.(); }}
          title="Download"
        >
          ↓
        </button>
      </div>
      {meta && (
        <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent">
          <div className="flex justify-between text-xs text-white/90">
            <span>{meta.model_id || alt}</span>
            <span>{meta.scale ? `${meta.scale}×` : ''}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export function Skeleton({ className }) {
  return (
    <div
      className={cn(
        'animate-pulse rounded-md bg-slate-200',
        className,
      )}
    />
  );
}

export function Divider({ className }) {
  return (
    <div className={cn('h-px bg-slate-100 my-6', className)} />
  );
}

export function Spinner({ className }) {
  return (
    <div className={cn('animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600', className)} />
  );
}
