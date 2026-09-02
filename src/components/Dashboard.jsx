import React, { useState, useEffect } from 'react';
import {
  Sprout, BarChart3, GitCompare, TrendingUp,
  Thermometer, CloudRain, Droplets, MapPin,
  ArrowRight, Leaf, Sun, Wheat, AlertCircle,
} from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts';
import {
  getHistoricalTrends, getModelComparison, humanizeApiError,
  REGIONS, SOIL_TYPES, WATER_SOURCES, CROP_TYPES,
} from '../services/apiService';

const quickStats = [
  { label: 'Crop Types', getValue: () => CROP_TYPES.length, icon: Sprout, color: 'text-forest-600', bg: 'bg-forest-50' },
  { label: 'Regions', getValue: () => REGIONS.length, icon: MapPin, color: 'text-blue-600', bg: 'bg-blue-50' },
  { label: 'Soil Types', getValue: () => SOIL_TYPES.length, icon: Leaf, color: 'text-soil-600', bg: 'bg-soil-50' },
  { label: 'Water Sources', getValue: () => WATER_SOURCES.length, icon: Droplets, color: 'text-cyan-600', bg: 'bg-cyan-50' },
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

export default function Dashboard({ onNavigate }) {
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
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Sun size={20} className="text-harvest-500" />
            <span className="text-sm font-medium text-harvest-600">Welcome to Lal Yar Link</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">
            Agricultural Decision Support
          </h1>
          <p className="text-gray-500 mt-1 max-w-lg">
            Multi-model crop prediction and yield analysis system powered by machine learning.
          </p>
        </div>
        <button
          onClick={() => onNavigate('cropType')}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <Wheat size={18} />
          Start Prediction
          <ArrowRight size={16} />
        </button>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {quickStats.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div
              key={i}
              className="glass-card p-4 sm:p-5 flex items-center gap-4 animate-fade-in"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className={`w-11 h-11 rounded-xl ${stat.bg} flex items-center justify-center`}>
                <Icon size={22} className={stat.color} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{stat.getValue()}</p>
                <p className="text-xs text-gray-500 font-medium">{stat.label}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Feature Cards */}
      <div>
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Prediction Modules</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((feature, i) => {
            const Icon = feature.icon;
            return (
              <button
                key={feature.id}
                onClick={() => onNavigate(feature.id)}
                className="glass-card p-6 text-left group hover:shadow-xl transition-all duration-300 animate-fade-in"
                style={{ animationDelay: `${(i + 4) * 80}ms` }}
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon size={24} className="text-white" />
                </div>
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="font-semibold text-gray-900">{feature.title}</h3>
                </div>
                <span className="inline-block text-xs font-medium px-2.5 py-1 rounded-full bg-forest-50 text-forest-700 mb-3">
                  {feature.tag}
                </span>
                <p className="text-sm text-gray-500 leading-relaxed">{feature.description}</p>
                <div className="mt-4 flex items-center text-sm font-medium text-forest-600 group-hover:text-forest-700">
                  Get Started
                  <ArrowRight size={14} className="ml-1.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Trend Chart + Model Info */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="lg:col-span-2 glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold text-gray-900">Historical Yield Trend</h3>
              <p className="text-xs text-gray-500">Paddy — Mandalay Region</p>
            </div>
            <TrendingUp size={20} className="text-forest-500" />
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

        {/* Model Info */}
        <div className="glass-card p-6 flex flex-col gap-4">
          <h3 className="font-semibold text-gray-900">Model Variants</h3>
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
              <p className="text-xs text-gray-500">Loading model metrics...</p>
            )}
          </div>

          <div className="p-3 rounded-xl bg-harvest-50 border border-harvest-100">
            <div className="flex items-center gap-2">
              <Thermometer size={14} className="text-harvest-600" />
              <span className="text-xs font-medium text-harvest-700">Optimal Conditions</span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-harvest-600">
              <div className="flex items-center gap-1"><Thermometer size={12} /> 20-32°C</div>
              <div className="flex items-center gap-1"><CloudRain size={12} /> 50-250mm</div>
              <div className="flex items-center gap-1"><Droplets size={12} /> 40-80%</div>
              <div className="flex items-center gap-1"><MapPin size={12} /> {REGIONS.length} Regions</div>
            </div>
          </div>
        </div>
      </div>

      {/* Others */}
      <div>
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Others</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <button
            onClick={() => onNavigate('historicalTrends')}
            className="glass-card p-6 text-left group hover:shadow-xl transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center group-hover:scale-110 transition-transform">
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
              ) : (
                <div className="w-12 h-12 rounded-xl bg-cyan-50 flex items-center justify-center">
                  <TrendingUp size={22} className="text-cyan-500" />
                </div>
              )}
            </div>
            <h3 className="font-semibold text-gray-900 mb-1">Historical Yield Trend</h3>
            <p className="text-xs text-gray-500 mb-3">Paddy — Mandalay Region</p>
            <p className="text-sm text-gray-500 leading-relaxed mb-4">View historical crop yield trends</p>
            <div className="flex items-center text-sm font-medium text-cyan-600 group-hover:text-cyan-700">
              View Trends
              <ArrowRight size={14} className="ml-1.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
