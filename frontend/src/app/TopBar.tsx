import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

export interface TopBarProps {
  device?: string;
  isMock?: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  device = 'CUDA · RTX 2050',
  isMock = false,
}) => {
  const location = useLocation();

  const navLinks = [
    { path: '/', label: 'Overview' },
    { path: '/studio', label: 'Studio' },
    { path: '/benchmarks', label: 'Benchmarks' },
    { path: '/models', label: 'Models' },
    { path: '/method', label: 'Method' },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14 bg-surface-container-low border-b border-outline-variant select-none">
      <div className="h-14 w-full px-6 flex items-center justify-between">
        {/* Left Brand & Wordmark */}
        <div className="flex items-center gap-6">
          <NavLink
            to="/"
            className="flex items-center gap-2.5 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary-container rounded"
          >
            <div className="w-6 h-6 rounded bg-surface-container-high border border-outline-variant flex items-center justify-center shadow-inner">
              <span className="w-2 h-2 rounded-sm bg-primary-container rotate-45 shadow-[0_0_8px_#38E1FF]" />
            </div>
            <span className="font-headline-md text-headline-md tracking-tight text-on-surface uppercase font-semibold">
              SISR<span className="text-primary-container">·</span>LAB
            </span>
          </NavLink>

          <div className="h-4 w-[1px] bg-outline-variant" />

          {/* Navigation Links with Shared Sliding Underline */}
          <nav className="flex items-center space-x-1 h-14">
            {navLinks.map((link) => {
              const isActive =
                link.path === '/'
                  ? location.pathname === '/'
                  : location.pathname.startsWith(link.path);

              return (
                <NavLink
                  key={link.path}
                  to={link.path}
                  className={`relative px-3.5 h-14 flex items-center font-body-sm-medium text-body-sm-medium transition-colors ${
                    isActive
                      ? 'text-on-surface font-medium'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  <span>{link.label}</span>

                  {isActive && (
                    <motion.div
                      layoutId="nav-underline"
                      className="absolute bottom-0 inset-x-2 h-[2px] bg-primary-container shadow-[0_0_10px_rgba(56,225,255,0.7)]"
                      transition={{ type: 'spring', stiffness: 450, damping: 32 }}
                    />
                  )}
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* Right Hardware Telemetry & Status Indicator */}
        <div className="flex items-center">
          <div className="flex items-center gap-2 bg-surface-container border border-outline-variant px-3 py-1 rounded-full shadow-inner">
            <span
              className={`w-2 h-2 rounded-full ${
                isMock ? 'bg-amber-400' : 'bg-[#34d399] shadow-[0_0_6px_#34d399]'
              }`}
            />
            <span className="font-mono-metric-sm text-mono-metric-sm text-on-surface-variant tracking-wider uppercase font-medium">
              {device}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
