/**
 * Dashboard — project overview with live backend data (health + model registry).
 */
import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import {
  Database,
  Cpu,
  Activity,
  Layers,
  ShieldAlert,
  Zap,
  Server,
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  StatCard,
  SectionTitle,
  SectionSubtitle,
  EmptyState,
} from '../components/ui';

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const [h, m] = await Promise.all([api.health(), api.getModels()]);
        if (mounted) {
          setHealth(h);
          setModels(m);
        }
      } catch (err) {
        if (mounted) setError(err.message);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => { mounted = false; };
  }, []);

  const archs = [...new Set(models.map((m) => m.architecture))];
  const scales = [...new Set(models.map((m) => m.scale))];
  const available = models.filter((m) => m.available).length;

  if (loading) {
    return (
      <div className="space-y-6">
        <SectionTitle>Dashboard</SectionTitle>
        <SectionSubtitle>
          SISR Surveillance Super-Resolution Studio — real inference through
          trained SRCNN and ESPCN checkpoints.
        </SectionSubtitle>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <StatCard
              key={i}
              icon={<Zap className="w-6 h-6 animate-pulse" />}
              value="—"
              label="Loading"
              desc="Backend data"
            />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-red-100 bg-red-50/50">
        <CardHeader className="flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-red-600" />
          <CardTitle>Unable to reach backend</CardTitle>
        </CardHeader>
        <p className="text-sm text-red-700">{error}</p>
        <p className="text-sm text-slate-600 mt-3">
          Make sure the FastAPI server is running on port 8001:
        </p>
        <code className="block mt-2 bg-red-100 text-red-900 px-3 py-2 rounded-xl text-xs">
          python -m uvicorn backend.app.main:app --port 8001
        </code>
        <Button variant="ghost" size="sm" onClick={() => window.location.reload()}>
          Retry
        </Button>
      </Card>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <SectionTitle>Dashboard</SectionTitle>
        <SectionSubtitle>
          SISR Surveillance Super-Resolution Studio — real inference through
          trained SRCNN and ESPCN checkpoints.
        </SectionSubtitle>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Database className="w-6 h-6" />}
          value={models.length}
          label="Registered models"
          desc="E1–E8 checkpoints"
        />
        <StatCard
          icon={<Activity className="w-6 h-6" />}
          value={available}
          label="Available"
          desc="checkpoints loadable"
        />
        <StatCard
          icon={<Cpu className="w-6 h-6" />}
          value={health?.device || '—'}
          label="Device"
          desc={health?.cuda_available ? 'CUDA available' : 'CPU only'}
        />
        <StatCard
          icon={<Server className="w-6 h-6" />}
          value={health?.status || '—'}
          label="Backend"
          desc={health?.torch_version ? `PyTorch ${health.torch_version}` : '—'}
        />
      </div>

      {/* Capabilities */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <Layers className="w-5 h-5 text-blue-600" />
            <CardTitle>Architectures</CardTitle>
          </CardHeader>
          <ul className="space-y-2 text-sm text-slate-600">
            {archs.map((a) => (
              <li key={a} className="flex justify-between items-center py-2 px-3 rounded-lg hover:bg-slate-50 transition-colors cursor-default">
                <span className="capitalize">{a}</span>
                <Badge variant="default">{models.filter((m) => m.architecture === a).length} variants</Badge>
              </li>
            ))}
          </ul>
        </Card>
        <Card>
          <CardHeader>
            <Activity className="w-5 h-5 text-blue-600" />
            <CardTitle>Supported scales</CardTitle>
          </CardHeader>
          <ul className="space-y-2 text-sm text-slate-600">
            {scales.map((s) => (
              <li key={s} className="flex justify-between items-center py-2 px-3 rounded-lg hover:bg-slate-50 transition-colors cursor-default">
                <span>{s}×</span>
                <Badge variant="default">{models.filter((m) => m.scale === s).length} variants</Badge>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      {/* Model registry table */}
      <Card>
        <CardHeader>
          <Server className="w-5 h-5 text-slate-600" />
          <CardTitle>Model registry</CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs text-slate-500 uppercase">
                <th className="pb-3 pr-4 font-medium">Model</th>
                <th className="pb-3">Arch</th>
                <th className="pb-3">Scale</th>
                <th className="pb-3">Loss</th>
                <th className="pb-3">Checkpoint</th>
                <th className="pb-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {models.map((m) => (
                <tr key={m.model_id} className="hover:bg-slate-50/50 transition-colors group">
                  <td className="py-3 pr-4 font-medium text-slate-800">{m.model_id}</td>
                  <td className="py-3 capitalize text-slate-600">{m.architecture}</td>
                  <td className="py-3 font-mono text-slate-600">{m.scale}×</td>
                  <td className="py-3 capitalize text-slate-600">{m.loss.replace('_', ' ')}</td>
                  <td className="py-3 text-slate-500 font-mono text-xs">{m.checkpoint}</td>
                  <td className="py-3">
                    <Badge variant={m.available ? 'available' : 'unavailable'}>
                      {m.available ? 'Available' : 'Unavailable'}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Quick actions */}
      <div className="flex flex-col sm:flex-row gap-3">
        <Button asChild size="lg" className="flex-1 sm:flex-none">
          <a href="#/studio">Enhance an image →</a>
        </Button>
        <Button variant="secondary" asChild size="lg" className="flex-1 sm:flex-none">
          <a href="#/compare">Compare models →</a>
        </Button>
      </div>
    </div>
  );
}
