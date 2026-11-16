import React, { useState, useEffect } from 'react';
import { useLocation, useOutlet } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { checkApiHealth, HealthStatus } from '../lib/api';
import { TopBar } from './TopBar';
import { StatusBar } from './StatusBar';

export const AppLayout: React.FC = () => {
  const location = useLocation();
  const outlet = useOutlet();

  const [health, setHealth] = useState<HealthStatus>({
    status: 'mock',
    device: 'CUDA · RTX 2050',
    isMock: true,
  });

  // Check health once on initial application mount (no repetitive terminal spam)
  useEffect(() => {
    checkApiHealth().then(setHealth).catch(console.error);
  }, []);

  return (
    <div className="bg-surface-container-lowest font-body-md text-on-surface antialiased min-h-screen flex flex-col justify-between selection:bg-primary-container selection:text-on-primary-container">
      {/* 56px Fixed Top Navigation Bar */}
      <TopBar device={health.device} isMock={health.isMock} />

      {/* Main Page Area */}
      <main className="w-full pt-14 pb-7 bg-surface-container-lowest min-h-[calc(100vh-28px)] relative flex-1 flex flex-col">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="flex-1 flex flex-col w-full"
          >
            {outlet}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* 28px Fixed Forensic Status Bar */}
      <StatusBar device={health.device.includes('CUDA') ? 'CUDA' : 'CPU'} />
    </div>
  );
};
