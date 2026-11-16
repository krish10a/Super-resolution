/**
 * Minimal dependency-free client-side router using hash-based routes.
 *
 * Routes:
 *   #/           Dashboard
 *   #/studio     Super-Resolution Studio
 *   #/compare    Model Comparison
 *   #/info       Model Information
 *
 * Hash routing works without server-side SPA fallback, so refreshing on any
 * route works with plain `vite` dev server and static hosting.
 */
import React, { useState, useEffect } from 'react';
import Dashboard from './pages/Dashboard';
import Studio from './pages/Studio';
import Comparison from './pages/Comparison';
import ModelInfo from './pages/ModelInfo';
import { ErrorBoundary } from './components/ErrorBoundary';

const ROUTES = {
  '': Dashboard,
  '/': Dashboard,
  '/dashboard': Dashboard,
  '/studio': Studio,
  '/compare': Comparison,
  '/info': ModelInfo,
};

export function AppRouter() {
  const [route, setRoute] = useState(() => location.hash.slice(1) || '/');

  useEffect(() => {
    const onHashChange = () => setRoute(location.hash.slice(1) || '/');
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const Page = ROUTES[route] || Dashboard;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <ErrorBoundary>
        <Page />
      </ErrorBoundary>
    </div>
  );
}

export default AppRouter;
