import React, { useState, useEffect } from 'react';
import {
  Sprout, BarChart3, GitCompare, TrendingUp,
  Droplets, MapPin,
  ArrowRight, Leaf, Sun, Wheat, AlertCircle,
  Database, GitBranch, X, Search,
} from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts';
import {
  getHistoricalTrends, getModelComparison, humanizeApiError,
  REGIONS, SOIL_TYPES, WATER_SOURCES, CROP_TYPES,
} from '../services/apiService';

const CATALOGS = {
  crops: {
    title: 'Crop types',
    items: CROP_TYPES,
    icon: Sprout,
    color: 'text-forest-600',
    bg: 'bg-forest-50',
  },
  regions: {
    title: 'Regions',
    items: REGIONS,
    icon: MapPin,
    color: 'text-blue-600',
    bg: 'bg-blue-50',
  },
  soils: {
    title: 'Soil types',
    items: SOIL_TYPES,
    icon: Leaf,
    color: 'text-soil-600',
    bg: 'bg-soil-50',
  },
  water: {
    title: 'Water sources',
    items: WATER_SOURCES,
    icon: Droplets,
    color: 'text-cyan-600',
    bg: 'bg-cyan-50',
  },
};

const quickStats = [
  { id: 'crops', label: 'Crop Types', getValue: () => CROP_TYPES.length, icon: Sprout, color: 'text-forest-600', bg: 'bg-forest-50' },
  { id: 'regions', label: 'Regions', getValue: () => REGIONS.length, icon: MapPin, color: 'text-blue-600', bg: 'bg-blue-50' },
  { id: 'soils', label: 'Soil Types', getValue: () => SOIL_TYPES.length, icon: Leaf, color: 'text-soil-600', bg: 'bg-soil-50' },
  { id: 'water', label: 'Water Sources', getValue: () => WATER_SOURCES.length, icon: Droplets, color: 'text-cyan-600', bg: 'bg-cyan-50' },
];

const features = [
  {
    id: 'cropType',
    title: 'Crop Type Prediction',
    description: 'Predict the most suitable crop for your conditions using classification models',
    icon: Sprout,
    gradient: 'from-forest-500 to-earth-500',
    tag: 'Classification',
  },
  {
    id: 'cropYield',
    title: 'Crop Yield Prediction',
    description: 'Estimate production tonnage and yield levels using regression models',
    icon: BarChart3,
    gradient: 'from-harvest-500 to-harvest-600',
    tag: 'Regression',
  },
  {
    id: 'comparison',
    title: 'Model Comparison',
    description: 'Compare Baseline, Feature Engineering, and Advanced model performance side-by-side',
    icon: GitCompare,
    gradient: 'from-blue-500 to-indigo-500',
    tag: 'Analytics',
  },
];

function CatalogModal({ catalog, onClose }) {
  const [query, setQuery] = useState('');

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!catalog) return null;
  const Icon = catalog.icon;
  const q = query.trim().toLowerCase();
  const items = q
    ? catalog.items.filter(item => item.toLowerCase().includes(q))
    : catalog.items;
  const compact = catalog.items.length <= 4;

  return (
    <div className="fixed inset-0 z-[70] flex items-end sm:items-center justify-center p-0 sm:p-6">
      <button
        type="button"
        className="absolute inset-0 bg-forest-950/40 backdrop-blur-sm"
        aria-label="Close list"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="catalog-title"
        className="relative w-full sm:max-w-2xl max-h-[85vh] flex flex-col bg-white rounded-t-2xl sm:rounded-2xl shadow-xl"
      >
        <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-100">
          <div className={`w-10 h-10 rounded-xl ${catalog.bg} flex items-center justify-center`}>
            <Icon size={20} className={catalog.color} />
          </div>
          <div className="min-w-0 flex-1">
            <h2 id="catalog-title" className="text-base font-semibold text-gray-900">{catalog.title}</h2>
            <p className="text-xs text-gray-500">{catalog.items.length} in the dataset</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg text-gray-500 hover:bg-gray-100"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        {catalog.items.length > 6 && (
          <div className="px-5 pt-4">
            <label className="relative block">
              <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="search"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder={`Search ${catalog.title.toLowerCase()}…`}
                className="w-full pl-9 pr-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-forest-400"
              />
            </label>
          </div>
        )}

        <ul className={`overflow-y-auto p-4 ${compact ? 'grid grid-cols-1 gap-2' : 'grid sm:grid-cols-2 gap-2'}`}>
          {items.map((item) => (
            <li
              key={item}
              className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-gray-50 border border-gray-100 text-sm text-gray-800"
            >
              <span className="text-xs font-mono text-gray-400 w-5 shrink-0 pt-0.5">
                {catalog.items.indexOf(item) + 1}
              </span>
              <span>{item}</span>
            </li>
          ))}
          {items.length === 0 && (
            <li className="col-span-full text-center text-sm text-gray-500 py-8">
              No matches for “{query}”
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

export default function Dashboard({ onNavigate }) {
  const [openCatalog, setOpenCatalog] = useState(null);
  const [trendData, setTrendData] = useState([]);
  const [trendState, setTrendState] = useState('loading'); // loading | loaded | error
  const [trendError, setTrendError] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [metricsState, setMetricsState] = useState('loading');

  useEffect(() => {
    let active = true;

    // Historical trend (real backend data; shown as unavailable if not found).
    getHistoricalTrends('Mandalay', 'Paddy')
      .then(res => {
        if (!active) return;
        const years = res && Array.isArray(res.years) ? res.years : [];
        const mapped = years.map((year, i) => ({
          year,
          yield: res.yields ? res.yields[i] : null,
          rainfall: res.rainfall ? res.rainfall[i] : null,
          temperature: res.temperatures ? res.temperatures[i] : null,
          area: res.areas ? res.areas[i] : null,
        })).filter(d => d.yield != null);
        setTrendData(mapped);
        setTrendState(mapped.length > 0 ? 'loaded' : 'error');
        if (mapped.length === 0) setTrendError('No historical data available for this region/crop.');
      })
      .catch(e => {
        if (!active) return;
        setTrendState('error');
        setTrendError(humanizeApiError(e));
      });

    // Real model metrics (from the compare endpoint for crop_type).
    getModelComparison({
      region: 'Mandalay', year: 2023, cropType: 'Paddy', sownAcre: 200,
      soilType: SOIL_TYPES[0], avgTemperature: 28, totalRainfall: 1000,
      avgHumidity: 80, waterSource: WATER_SOURCES[1], seedingSeason: 'Rainy',
    }, 'crop_type')
      .then(res => {
        if (!active) return;
        setModelMetrics(res);
        setMetricsState('loaded');
      })
      .catch(() => {
        if (!active) return;
        setMetricsState('error');
      });

    return () => { active = false; };
  }, []);

  return (
    <div className="page-shell space-y-8">
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-forest-700 via-forest-800 to-earth-900 text-white p-6 sm:p-8">
        <div className="absolute -right-8 -top-10 w-40 h-40 rounded-full bg-harvest-400/20 blur-2xl" aria-hidden="true" />
        <div className="relative flex flex-col sm:flex-row sm:items-end sm:justify-between gap-5">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Sun size={18} className="text-harvest-300" />
              <span className="text-sm font-medium text-harvest-200">Welcome to Lal Yar Link</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
              Agricultural Decision Support
            </h1>
            <p className="text-forest-100 mt-2 max-w-lg text-sm leading-relaxed">
              Predict crops, estimate yield, and explore historical patterns from real Myanmar agricultural data.
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('cropType')}
            className="inline-flex items-center gap-2 self-start px-6 py-3 rounded-xl font-semibold bg-white text-forest-800 shadow-md hover:bg-harvest-50 transition-all active:scale-[0.98]"
          >
            <Wheat size={18} />
            Start prediction
            <ArrowRight size={16} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {quickStats.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <button
              key={stat.id}
              type="button"
              onClick={() => setOpenCatalog(stat.id)}
              className="glass-card p-4 sm:p-5 flex items-center gap-4 text-left hover:shadow-md hover:border-forest-200 transition-all"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className={`w-11 h-11 rounded-xl ${stat.bg} flex items-center justify-center`}>
                <Icon size={22} className={stat.color} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{stat.getValue()}</p>
                <p className="text-xs text-gray-500 font-medium">{stat.label}</p>
                <p className="text-[11px] text-forest-600 mt-0.5">View all</p>
              </div>
            </button>
          );
        })}
      </div>

      <CatalogModal
        key={openCatalog || 'closed'}
        catalog={openCatalog ? CATALOGS[openCatalog] : null}
        onClose={() => setOpenCatalog(null)}
      />

      <div>
        <div className="flex items-end justify-between mb-4">
          <div>
            <p className="section-label mb-1">Predict</p>
            <h2 className="text-lg font-semibold text-gray-800">Prediction modules</h2>
          </div>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <button
                key={feature.id}
                type="button"
                onClick={() => onNavigate(feature.id)}
                className="glass-card p-6 text-left group hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-105 transition-transform`}>
                  <Icon size={24} className="text-white" />
                </div>
                <h3 className="font-semibold text-gray-900">{feature.title}</h3>
                <span className="inline-block text-xs font-medium px-2.5 py-1 rounded-full bg-forest-50 text-forest-700 mt-2 mb-3">
                  {feature.tag}
                </span>
                <p className="text-sm text-gray-500 leading-relaxed">{feature.description}</p>
                <div className="mt-4 flex items-center text-sm font-medium text-forest-600 group-hover:text-forest-700">
                  Open module
                  <ArrowRight size={14} className="ml-1.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold text-gray-900">Historical yield trend</h3>
              <p className="text-xs text-gray-500">Paddy — Mandalay Region</p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('historicalTrends')}
              className="text-xs font-medium text-forest-600 hover:text-forest-800"
            >
              View all
            </button>
          </div>
          {trendState === 'loaded' && trendData.length > 0 ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData}>
                  <defs>
                    <linearGradient id="yieldGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2d9f63" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#2d9f63" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="year" tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{
                      borderRadius: '12px', border: '1px solid #e5e7eb',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.08)', fontSize: '12px',
                    }}
                  />
                  <Area type="monotone" dataKey="yield" stroke="#2d9f63" strokeWidth={2.5}
                    fill="url(#yieldGrad)" dot={{ r: 3, fill: '#2d9f63' }}
                    activeDot={{ r: 5, fill: '#2d9f63', stroke: '#fff', strokeWidth: 2 }} name="Yield (tons/acre)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-56 flex flex-col items-center justify-center text-center">
              {trendState === 'loading' ? (
                <>
                  <div className="w-8 h-8 border-3 border-forest-200 border-t-forest-600 rounded-full animate-spin mb-3" />
                  <p className="text-sm text-gray-500">Loading historical data...</p>
                </>
              ) : (
                <>
                  <AlertCircle size={26} className="text-amber-400 mb-2" />
                  <p className="text-sm font-medium text-gray-600">Historical data unavailable</p>
                  <p className="text-xs text-gray-400 mt-1 max-w-xs">{trendError}</p>
                </>
              )}
            </div>
          )}
        </div>

        <div className="glass-card p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">Model variants</h3>
            <button
              type="button"
              onClick={() => onNavigate('comparison')}
              className="text-xs font-medium text-forest-600 hover:text-forest-800"
            >
              Compare
            </button>
          </div>
          <div className="space-y-3 flex-1">
            {metricsState === 'loaded' && modelMetrics ? (
              modelMetrics.models.map(m => (
                <div key={m.variant} className="p-3 rounded-xl bg-gray-50 border border-gray-100">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-2.5 h-2.5 rounded-full bg-forest-500" />
                    <span className="text-sm font-semibold text-gray-800">{m.name}</span>
                    <span className="ml-auto text-xs font-mono font-bold text-forest-600">
                      {m.available && m.metrics ? `${(m.metrics.accuracy * 100).toFixed(1)}%` : 'Unavailable'}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 ml-[18px]">
                    {m.available ? 'Accuracy from backend' : m.error || 'Not available'}
                  </p>
                </div>
              ))
            ) : metricsState === 'error' ? (
              <p className="text-xs text-gray-500">Model metrics could not be loaded from backend.</p>
            ) : (
              <div className="space-y-2">
                <div className="skeleton h-16" />
                <div className="skeleton h-16" />
                <div className="skeleton h-16" />
              </div>
            )}
          </div>
        </div>
      </div>

      <div>
        <p className="section-label mb-1">Explore</p>
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Data & insights</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <button
            type="button"
            onClick={() => onNavigate('historicalTrends')}
            className="glass-card p-6 text-left group hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center group-hover:scale-105 transition-transform">
                <TrendingUp size={24} className="text-white" />
              </div>
              {trendData.length > 1 ? (
                <div className="w-24 h-10">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trendData}>
                      <defs>
                        <linearGradient id="othersGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <Area type="monotone" dataKey="yield" stroke="#0ea5e9" strokeWidth={2}
                        fill="url(#othersGrad)" dot={false} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : null}
            </div>
            <h3 className="font-semibold text-gray-900 mb-1">Historical yield trends</h3>
            <p className="text-sm text-gray-500 leading-relaxed mb-4">Compare yield over years by region and crop.</p>
            <div className="flex items-center text-sm font-medium text-cyan-600 group-hover:text-cyan-700">
              View trends
              <ArrowRight size={14} className="ml-1.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </button>

          <button
            type="button"
            onClick={() => onNavigate('dataStatistics')}
            className="glass-card p-6 text-left group hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
          >
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
              <Database size={24} className="text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-1">Data statistics</h3>
            <p className="text-sm text-gray-500 leading-relaxed mb-4">Dataset ranges, distributions, and overview counts.</p>
            <div className="flex items-center text-sm font-medium text-indigo-600 group-hover:text-indigo-700">
              View statistics
              <ArrowRight size={14} className="ml-1.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </button>

          <button
            type="button"
            onClick={() => onNavigate('descriptiveMining')}
            className="glass-card p-6 text-left group hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
          >
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
              <GitBranch size={24} className="text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-1">Descriptive mining</h3>
            <p className="text-sm text-gray-500 leading-relaxed mb-4">Correlations, association rules, and sequential patterns.</p>
            <div className="flex items-center text-sm font-medium text-amber-600 group-hover:text-amber-700">
              View mining
              <ArrowRight size={14} className="ml-1.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
