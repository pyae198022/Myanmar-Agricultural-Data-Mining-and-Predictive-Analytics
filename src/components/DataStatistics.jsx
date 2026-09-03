import React, { useState, useEffect } from 'react';
import {
  Database, BarChart3, Table2, ShieldCheck, Loader2, AlertCircle,
  Layers, MapPin, Sprout, CalendarRange, Droplets, Leaf, Rows3,
} from 'lucide-react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { getDataStatistics, humanizeApiError } from '../services/apiService';

const CHART_COLORS = ['#2d9f63', '#e5a03c', '#2563eb', '#8b5cf6', '#06b6d4', '#dc2626', '#ec4899', '#10b981', '#f59e0b'];

// Column aliases used purely for grouping/display of the raw categorical charts.
const CATEGORY_GROUPS = {
  Region: 'Region',
  Crop_Type: 'Crop Type',
  Soil_Type: 'Soil Type',
  Seeding_Season: 'Seeding Season',
  Water_Source: 'Water Source',
};

function OverviewCard({ icon: Icon, label, value }) {
  return (
    <div className="metric-card bg-white border border-forest-100">
      <div className="flex items-center gap-2 mb-1">
        <Icon size={15} className="text-forest-500" />
        <span className="text-xs font-medium text-gray-500">{label}</span>
      </div>
      <p className="text-xl font-bold text-gray-800">{value}</p>
    </div>
  );
}

function NumericTable({ numerical }) {
  return (
    <div className="glass-card p-5 overflow-hidden">
      <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
        <Table2 size={16} className="text-forest-500" /> Numerical Statistics
      </h4>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-200">
              <th className="py-2 pr-3 font-medium">Column</th>
              <th className="py-2 pr-3 font-medium">Count</th>
              <th className="py-2 pr-3 font-medium">Mean</th>
              <th className="py-2 pr-3 font-medium">Median</th>
              <th className="py-2 pr-3 font-medium">Std Dev</th>
              <th className="py-2 pr-3 font-medium">Min</th>
              <th className="py-2 pr-3 font-medium">Max</th>
            </tr>
          </thead>
          <tbody>
            {numerical.map(row => (
              <tr key={row.column} className="border-b border-gray-50 last:border-0">
                <td className="py-2 pr-3 font-medium text-gray-700">{row.column}</td>
                <td className="py-2 pr-3 text-gray-600">{row.count}</td>
                <td className="py-2 pr-3 text-gray-600">{row.mean != null ? row.mean : '—'}</td>
                <td className="py-2 pr-3 text-gray-600">{row.median != null ? row.median : '—'}</td>
                <td className="py-2 pr-3 text-gray-600">{row.std != null ? row.std : '—'}</td>
                <td className="py-2 pr-3 text-gray-600">{row.min != null ? row.min : '—'}</td>
                <td className="py-2 pr-3 text-gray-600">{row.max != null ? row.max : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CategoricalCard({ cat, showChart }) {
  const total = cat.distributions.reduce((s, d) => s + d.count, 0);
  const top = cat.distributions.slice(0, 12);
  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-gray-800">{cat.column}</h4>
        <span className="text-[11px] font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
          {cat.unique} unique
        </span>
      </div>

      {showChart && cat.column in CATEGORY_GROUPS && (
        <div className="h-56 mb-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top.map(d => ({ ...d, name: d.label }))} layout="vertical" margin={{ left: 8 }}>
              <XAxis type="number" tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} allowDecimals={false} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#374151' }} axisLine={false} tickLine={false} width={110} />
              <Tooltip
                formatter={(v) => [v, 'Count']}
                contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', fontSize: '12px' }}
              />
              <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                {top.map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Frequency distribution list */}
      <div className="space-y-1.5">
        {cat.distributions.slice(0, 10).map((d, i) => {
          const pct = total ? (d.count / total) * 100 : 0;
          return (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="text-gray-600 w-40 truncate shrink-0">{d.label}</span>
              <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-forest-500 to-forest-400"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-gray-500 w-16 text-right shrink-0">
                {d.count} ({pct.toFixed(1)}%)
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DataQualityTable({ dataQuality }) {
  const { missing, duplicate_rows, data_types } = dataQuality;
  return (
    <div className="glass-card p-5 overflow-hidden">
      <h4 className="text-sm font-semibold text-gray-800 mb-1 flex items-center gap-2">
        <ShieldCheck size={16} className="text-forest-500" /> Data Quality
      </h4>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs font-medium text-gray-500">
          Duplicate rows:
        </span>
        <span className="text-xs font-bold text-gray-700">{duplicate_rows}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-200">
              <th className="py-2 pr-3 font-medium">Column</th>
              <th className="py-2 pr-3 font-medium">Data Type</th>
              <th className="py-2 pr-3 font-medium">Type</th>
              <th className="py-2 pr-3 font-medium">Missing Values</th>
            </tr>
          </thead>
          <tbody>
            {data_types.map(col => {
              const miss = (missing.find(m => m.column === col.column) || {}).count;
              return (
                <tr key={col.column} className="border-b border-gray-50 last:border-0">
                  <td className="py-2 pr-3 font-medium text-gray-700">{col.column}</td>
                  <td className="py-2 pr-3 text-gray-600">{col.dtype}</td>
                  <td className="py-2 pr-3">
                    <span className={`text-[11px] font-medium px-2 py-0.5 rounded ${col.is_numeric ? 'bg-blue-50 text-blue-700' : 'bg-harvest-50 text-amber-700'}`}>
                      {col.is_numeric ? 'Numeric' : 'Categorical'}
                    </span>
                  </td>
                  <td className={`py-2 pr-3 ${miss ? 'text-red-600' : 'text-gray-600'}`}>
                    {miss || 0}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function DataStatistics() {
  const [stats, setStats] = useState(null);
  const [state, setState] = useState('loading'); // loading | loaded | error
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setState('loading');
    setError(null);
    getDataStatistics()
      .then(data => {
        if (!active) return;
        setStats(data);
        setState('loaded');
      })
      .catch(err => {
        if (!active) return;
        setError(humanizeApiError(err));
        setState('error');
      });
    return () => { active = false; };
  }, []);

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-forest-500 to-emerald-600 flex items-center justify-center">
          <Database size={22} className="text-white" />
        </div>
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-gray-900">Data Statistics</h1>
          <p className="text-sm text-gray-500">
            Descriptive statistics computed from the real dataset (cleaned_data.csv)
          </p>
        </div>
      </div>

      {/* Loading */}
      {state === 'loading' && (
        <div className="flex flex-col items-center justify-center py-20 text-gray-500">
          <Loader2 size={32} className="animate-spin text-forest-500 mb-4" />
          <p className="text-sm font-medium">Computing statistics from the dataset...</p>
        </div>
      )}

      {/* Error */}
      {state === 'error' && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-red-50 border border-red-200">
          <AlertCircle size={18} className="text-red-500 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-700">Failed to load dataset statistics</p>
            <p className="text-xs text-red-600 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Empty (defensive) */}
      {state === 'loaded' && (!stats || !stats.overview) && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-50 border border-amber-200">
          <AlertCircle size={18} className="text-amber-500 mt-0.5 shrink-0" />
          <p className="text-sm text-amber-700">
            No statistics are available for the dataset.
          </p>
        </div>
      )}

      {/* Loaded content */}
      {state === 'loaded' && stats && stats.overview && (
        <>
          {/* Dataset Overview */}
          <div>
            <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <Layers size={16} className="text-forest-500" /> Dataset Overview
            </h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <OverviewCard icon={Rows3} label="Total Records" value={stats.overview.total_records} />
              <OverviewCard icon={Table2} label="Total Attributes" value={stats.overview.total_attributes} />
              <OverviewCard icon={CalendarRange} label="Year Range" value={`${stats.overview.year_min}–${stats.overview.year_max}`} />
              <OverviewCard icon={MapPin} label="Regions" value={stats.overview.num_regions} />
              <OverviewCard icon={Sprout} label="Crop Types" value={stats.overview.num_crop_types} />
              <OverviewCard icon={Droplets} label="Soil Types" value={stats.overview.num_soil_types} />
              <OverviewCard icon={Leaf} label="Seeding Seasons" value={stats.overview.num_seeding_seasons} />
              <OverviewCard icon={Droplets} label="Water Sources" value={stats.overview.num_water_sources} />
            </div>
            {/* Year distribution chart */}
            <div className="glass-card p-5 mt-3">
              <h4 className="text-sm font-semibold text-gray-800 mb-3">Year Distribution</h4>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats.overview.year_counts.map(p => ({ year: p.year, count: p.count }))}>
                    <XAxis dataKey="year" tick={{ fontSize: 11, fill: '#374151' }} axisLine={false} tickLine={false} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
                    <Tooltip
                      formatter={(v) => [v, 'Records']}
                      contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', fontSize: '12px' }}
                    />
                    <Bar dataKey="count" fill="#2d9f63" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Numerical stats */}
          <NumericTable numerical={stats.numerical} />

          {/* Categorical stats + charts */}
          <div>
            <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <BarChart3 size={16} className="text-forest-500" /> Categorical Statistics
            </h3>
            <div className="grid lg:grid-cols-2 gap-4">
              {stats.categorical.map(cat => (
                <CategoricalCard key={cat.column} cat={cat} showChart />
              ))}
            </div>
          </div>

          {/* Data quality */}
          <DataQualityTable dataQuality={stats.data_quality} />

          <p className="text-xs text-gray-400">
            All values are computed at request time from the real dataset
            (<span className="font-mono">backend/data/cleaned_data.csv</span>). Nothing is hardcoded.
          </p>
        </>
      )}
    </div>
  );
}