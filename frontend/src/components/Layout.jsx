/**
 * Application shell: top navigation bar + mobile drawer.
 * Professional dark header with glassmorphism, clear navigation.
 */
import React, { useState } from 'react';
import {
  LayoutDashboard,
  Image,
  Compare,
  Info,
  Menu,
  X,
  Shield,
  Zap,
} from 'lucide-react';
import { Button, Badge } from './ui';

const NAV = [
  { href: '#/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '#/studio', label: 'Studio', icon: Image },
  { href: '#/compare', label: 'Compare Models', icon: Compare },
  { href: '#/info', label: 'Model Info', icon: Info },
];

export default function Layout({ children }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-900">
      {/* Top bar */}
      <header className="fixed top-0 left-0 right-0 z-20 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-200/50">
              <Shield size={24} />
            </div>
            <div className="hidden md:flex flex-col">
              <span className="font-bold text-lg tracking-tight text-slate-900">
                SISR Studio
              </span>
              <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">
                Super-Resolution Studio
              </span>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-4">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-medium text-slate-600">Backend online</span>
          </div>
          <button
            className="md:hidden text-slate-600 hover:text-slate-900 rounded-lg p-2 hover:bg-slate-100 transition-colors"
            onClick={() => setOpen(!open)}
            aria-label="Toggle navigation"
          >
            {open ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Drawer (mobile) */}
        {open && (
          <div className="fixed inset-0 z-30 bg-slate-950/95 backdrop-blur-md flex flex-col md:hidden">
            <div className="p-6 border-b border-slate-800/50 flex justify-between items-center">
              <span className="font-bold text-white">Navigation</span>
              <button className="text-slate-400 hover:text-white" onClick={() => setOpen(false)}>
                <X size={20} />
              </button>
            </div>
            <nav className="flex-1 p-6 space-y-1">
              {NAV.map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  onClick={() => setOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
                >
                  <item.icon className="w-5 h-5" />
                  {item.label}
                </a>
              ))}
            </nav>
            <div className="p-6 border-t border-slate-800/50 text-xs text-slate-500">
              Real PyTorch inference through trained SRCNN / ESPCN checkpoints.
            </div>
          </div>
        )}
      </header>

      <main className="md:pl-0 pt-20 min-h-[calc(100vh-72px)] flex-1 overflow-y-auto">
        <div className="px-4 md:px-8 pb-12">{children}</div>
      </main>
    </div>
  );
}
