import React, { useState, useEffect, useMemo } from 'react';
import {
  MapPin, Layers, Calendar, ArrowLeft, TrendingUp,
  TrendingDown, Minus, ChevronRight, AlertCircle, RotateCcw,
} from 'lucide-react';
import {
  LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip,
  CartesianGrid,
} from 'recharts';
import { getHistoricalOverview, humanizeApiError, REGIONS } from '../services/apiService';
import PageHeader from './PageHeader';

const TREND_META = {
  increasing: { label: 'Increasing', icon: TrendingUp, cls: 'text-forest-600 bg-forest-50' },
  decreasing: { label: 'Decreasing', icon: TrendingDown, cls: 'text-red-600 bg-red-50' },
  stable: { label: 'Stable', icon: Minus, cls: 'text-gray-600 bg-gray-100' },
};

function computeTrend(yields) {
  if (!yields || yields.length < 2) return 'stable';
  const vals = yields.filter(v => v != null);
  if (vals.length < 2) return 'stable';
  const first = vals[0];
  const last = vals[vals.length - 1];
  const pct = Math.abs(first) < 1e-9 ? 0 : ((last - first) / Math.abs(first)) * 100;
  if (pct > 3) return 'increasing';
  if (pct < -3) return 'decreasing';
  return 'stable';
}

function summarizeYield(crop, yearMin, yearMax) {
  const points = crop.years
    .map((y, i) => ({ year: y, yield: Number(crop.yields[i]) }))
    .filter(p => Number.isFinite(p.yield) && p.year >= yearMin && p.year <= yearMax)
    .sort((a, b) => a.year - b.year);

  if (points.length === 0) return null;

  const latest = points[points.length - 1];
  const prev = points.length >= 2 ? points[points.length - 2] : null;
  const pctChange = prev != null && Math.abs(prev.yield) > 1e-9
    ? ((latest.yield - prev.yield) / Math.abs(prev.yield)) * 100
    : null;

  const highest = points.reduce((a, b) => (b.yield > a.yield ? b : a), points[0]);
  const lowest = points.reduce((a, b) => (b.yield < a.yield ? b : a), points[0]);

  return {
    points,
    latestYield: latest.yield,
    latestYear: latest.year,
    prevYield: prev ? prev.yield : null,
    prevYear: prev ? prev.year : null,
    pctChange,
    highestYield: highest.yield,
    highestYear: highest.year,
    lowestYield: lowest.yield,
    lowestYear: lowest.year,
    trend: computeTrend(points.map(p => p.yield)),
  };
}

function formatYield(v, unit = 'tons/acre') {
  if (v == null) return '—';
  return `${Number(v).toFixed(2)} ${unit}`;
}

export default function HistoricalTrends({ onNavigate }) {
  const [region, setRegion] = useState('Mandalay');
  const [crop, setCrop] = useState('all');
  const [yearMin, setYearMin] = useState(null);
  const [yearMax, setYearMax] = useState(null);
  const [overview, setOverview] = useState(null);
  const [state, setState] = useState('loading'); // loading | loaded | error
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null); // crop name for detail

  const load = () => {
    setState('loading');
    setError(null);
    getHistoricalOverview(region)
      .then(res => {
        setOverview(res);
        setYearMin(prev => prev ?? res.year_min);
        setYearMax(prev => prev ?? res.year_max);
        setState('loaded');
      })
      .catch(e => {
        setState('error');
        setError(humanizeApiError(e));
      });
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region]);

  useEffect(() => {
    if (!overview || !overview.crops || !(crop in overview.crops)) {
      setSelected(null);
      return;
    }
    const s = summarizeYield(overview.crops[crop], yearMin ?? overview.year_min, yearMax ?? overview.year_max);
    setSelected(s ? { name: crop, ...s } : null);
  }, [crop, yearMin, yearMax, overview]);

  const cropNames = useMemo(() => {
    if (!overview || !overview.crops) return [];
    return Object.keys(overview.crops).sort();
  }, [overview]);

  const filteredCrops = useMemo(() => {
    if (!overview || !overview.crops) return [];
    return cropNames
      .map(name => ({ name, ...summarizeYield(overview.crops[name], yearMin ?? overview.year_min, yearMax ?? overview.year_max) }))
      .filter(c => c.latestYield != null)
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [overview, cropNames, yearMin, yearMax]);

  const validYearMin = yearMin ?? overview?.year_min;
  const validYearMax = yearMax ?? overview?.year_max;
  const yearRange = validYearMin != null && validYearMax != null ? `${validYearMin} – ${validYearMax}` : '—';

  return (
    <div className="page-shell">
      <PageHeader
        icon={TrendingUp}
        title="Historical Yield Trends"
        subtitle="Explore real crop yield trends across regions and years."
        gradient="from-cyan-500 to-blue-500"
        onBack={() => onNavigate('dashboard')}
      />

      {/* Filters */}
      <div className="glass-card p-5">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="space-y-1.5">
            <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700">
              <MapPin size={14} className="text-blue-500" />
              Region
            </label>
            <select
              value={region}
              onChange={e => { setRegion(e.target.value); setYearMin(null); setYearMax(null); }}
              className="input-field"
            >
              {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700">
              <Layers size={14} className="text-forest-500" />
              Crop
            </label>
            <select
              value={crop}
              onChange={e => setCrop(e.target.value)}
              className="input-field"
            >
              <option value="all">All Crops</option>
              {cropNames.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700">
              <Calendar size={14} className="text-harvest-500" />
              From Year
            </label>
            <input
              type="number"
              className="input-field"
              min={overview?.year_min ?? 2012}
              max={overview?.year_max ?? 2023}
              value={validYearMin ?? ''}
              onChange={e => setYearMin(Number(e.target.value))}
            />
          </div>

          <div className="space-y-1.5">
            <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700">
              <Calendar size={14} className="text-harvest-500" />
              To Year
            </label>
            <input
              type="number"
              className="input-field"
              min={overview?.year_min ?? 2012}
              max={overview?.year_max ?? 2023}
              value={validYearMax ?? ''}
              onChange={e => setYearMax(Number(e.target.value))}
            />
          </div>
        </div>
        <div className="mt-3 flex items-center justify-between text-xs text-gray-400">
          <span>{region} — {crop === 'all' ? 'All Crops' : crop} — {yearRange}</span>
          <button
            onClick={() => { setCrop('all'); setSelected(null); }}
            className="flex items-center gap-1 font-medium text-gray-500 hover:text-forest-600 transition-colors"
          >
            <RotateCcw size={12} /> Reset
          </button>
        </div>
      </div>

      {/* Loading */}
      {state === 'loading' && (
        <div className="space-y-5 animate-pulse">
          <div className="h-8 w-64 rounded-lg bg-gray-200" />
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="glass-card p-6 space-y-3">
                <div className="h-4 w-32 rounded bg-gray-200" />
                <div className="h-3 w-40 rounded bg-gray-100" />
                <div className="h-16 w-full rounded bg-gray-100" />
                <div className="h-3 w-24 rounded bg-gray-100" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {state === 'error' && (
        <div className="glass-card p-8 flex flex-col items-center justify-center text-center">
          <AlertCircle size={32} className="text-red-400 mb-3" />
          <p className="text-sm font-medium text-gray-600">Unable to load historical yield trends</p>
          <p className="text-xs text-gray-400 mt-1 max-w-md">{error}</p>
          <button
            onClick={load}
            className="mt-4 btn-primary flex items-center gap-1.5"
          >
            <RotateCcw size={14} /> Retry
          </button>
        </div>
      )}

      {/* Loaded */}
      {state === 'loaded' && (
        <>
          {/* Detail */}
          {selected && (
            <div className="glass-card p-6">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Historical Yield Trend</h2>
                  <p className="text-xs text-gray-500">{selected.name} — {region} Region</p>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="text-xs font-medium text-gray-500 hover:text-forest-600 transition-colors flex items-center gap-1"
                >
                  <ArrowLeft size={12} /> Back to all crops
                </button>
              </div>

              <div className="h-64 sm:h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={selected.points} margin={{ top: 10, right: 16, bottom: 0, left: -8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                    <XAxis dataKey="year" tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
                    <YAxis
                      tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false}
                      domain={['auto', 'auto']}
                      label={{ value: 'tons/acre', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af', textAnchor: 'middle' } }}
                    />
                    <Tooltip
                      formatter={(value) => [formatYield(value), 'Yield (tons/acre)']}
                      labelFormatter={(label) => `Year ${label}`}
                      contentStyle={{
                        borderRadius: '12px', border: '1px solid #e5e7eb',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.08)', fontSize: '12px',
                      }}
                    />
                    <Line type="monotone" dataKey="yield" stroke="#2d9f63" strokeWidth={2.5}
                      dot={{ r: 3, fill: '#2d9f63' }} activeDot={{ r: 5, fill: '#2d9f63', stroke: '#fff', strokeWidth: 2 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Summary stats */}
              <div className="mt-5 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <Stat label="Latest Yield" value={formatYield(selected.latestYield)} sub={selected.latestYear} />
                <Stat label="Previous Year" value={formatYield(selected.prevYield)} sub={selected.prevYear} />
                <Stat
                  label="% Change"
                  value={selected.pctChange == null ? '—' : `${selected.pctChange >= 0 ? '+' : ''}${selected.pctChange.toFixed(1)}%`}
                  tone={selected.pctChange == null ? 'neutral' : selected.pctChange >= 0 ? 'positive' : 'negative'}
                />
                <Stat label="Highest Year" value={formatYield(selected.highestYield)} sub={selected.highestYear} />
                <Stat label="Lowest Year" value={formatYield(selected.lowestYield)} sub={selected.lowestYear} />
                <Stat
                  label="Overall Trend"
                  value={TREND_META[selected.trend]?.label ?? 'Stable'}
                  badge={TREND_META[selected.trend]?.cls}
                />
              </div>
            </div>
          )}

          {/* Crop cards */}
          {!selected && (
            <div>
              <h2 className="text-lg font-semibold text-gray-800 mb-1">
                {crop === 'all' ? 'All Crops' : crop}
              </h2>
              <p className="text-xs text-gray-500 mb-4">{region} Region — {yearRange}</p>

              {filteredCrops.length === 0 ? (
                <div className="glass-card p-8 flex flex-col items-center justify-center text-center">
                  <TrendingUp size={32} className="text-gray-300 mb-3" />
                  <p className="text-sm text-gray-500">No historical yield data available for this crop and region.</p>
                </div>
              ) : (
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
                  {filteredCrops.map(c => {
                    const meta = TREND_META[c.trend] || TREND_META.stable;
                    const TrendIcon = meta.icon;
                    return (
                      <button
                        key={c.name}
                        onClick={() => setCrop(c.name)}
                        className="glass-card p-5 text-left group hover:shadow-xl transition-all duration-300"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-semibold text-gray-900">{c.name}</h3>
                          <span className={`inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full ${meta.cls}`}>
                            <TrendIcon size={12} />
                            {meta.label}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 mb-4">{region} Region</p>
                        <div className="h-16 mb-4">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={c.points} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
                              <Line type="monotone" dataKey="yield"
                                stroke={c.trend === 'increasing' ? '#16a34a' : c.trend === 'decreasing' ? '#dc2626' : '#6b7280'}
                                strokeWidth={2} dot={false} isAnimationActive={false} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-[11px] text-gray-400">Latest Yield</p>
                            <p className="text-sm font-semibold text-gray-800">{formatYield(c.latestYield)}</p>
                          </div>
                          <div className="flex items-center text-sm font-medium text-forest-600 group-hover:text-forest-700">
                            Details
                            <ChevronRight size={14} className="ml-0.5 group-hover:translate-x-1 transition-transform" />
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Stat({ label, value, sub, tone = 'neutral', badge }) {
  const toneCls = tone === 'positive' ? 'text-forest-600' : tone === 'negative' ? 'text-red-600' : 'text-gray-900';
  return (
    <div className="p-3 rounded-xl bg-gray-50 border border-gray-100">
      <p className="text-[11px] text-gray-400 font-medium uppercase tracking-wide mb-1">{label}</p>
      {badge ? (
        <span className={`inline-flex items-center text-sm font-semibold px-2 py-0.5 rounded-full ${badge}`}>{value}</span>
      ) : (
        <p className={`text-sm font-semibold ${toneCls}`}>{value}</p>
      )}
      {sub != null && <p className="text-[11px] text-gray-400 mt-0.5">{sub}</p>}
    </div>
  );
}
