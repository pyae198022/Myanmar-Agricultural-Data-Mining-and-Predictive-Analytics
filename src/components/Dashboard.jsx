import React from 'react';
import {
  Sprout, BarChart3, GitCompare, TrendingUp,
  Thermometer, CloudRain, Droplets, MapPin,
  ArrowRight, Leaf, Sun, Wheat,
} from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts';
import { getHistoricalTrends } from '../services/mockInference';

const quickStats = [
  { label: 'Crop Types', value: '12', icon: Sprout, color: 'text-forest-600', bg: 'bg-forest-50' },
  { label: 'Regions', value: '15', icon: MapPin, color: 'text-blue-600', bg: 'bg-blue-50' },
  { label: 'Soil Types', value: '8', icon: Leaf, color: 'text-soil-600', bg: 'bg-soil-50' },
  { label: 'Water Sources', value: '7', icon: Droplets, color: 'text-cyan-600', bg: 'bg-cyan-50' },
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
  const trendData = getHistoricalTrends('Mandalay', 'Rice');

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Sun size={20} className="text-harvest-500" />
            <span className="text-sm font-medium text-harvest-600">Welcome to AgriPredict</span>
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
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
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

      {/* Trend Chart + Info */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="lg:col-span-2 glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold text-gray-900">Historical Yield Trend</h3>
              <p className="text-xs text-gray-500">Rice production — Mandalay Region (2015–2025)</p>
            </div>
            <TrendingUp size={20} className="text-forest-500" />
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="yieldGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2d9f63" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#2d9f63" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="year"
                  tick={{ fontSize: 11, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                    fontSize: '12px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="yield"
                  stroke="#2d9f63"
                  strokeWidth={2.5}
                  fill="url(#yieldGrad)"
                  dot={{ r: 3, fill: '#2d9f63' }}
                  activeDot={{ r: 5, fill: '#2d9f63', stroke: '#fff', strokeWidth: 2 }}
                  name="Yield (tons)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Info */}
        <div className="glass-card p-6 flex flex-col gap-4">
          <h3 className="font-semibold text-gray-900">Model Variants</h3>
          <div className="space-y-3 flex-1">
            {[
              {
                name: 'Baseline',
                desc: 'Original unscaled features',
                accuracy: '~75%',
                color: 'bg-red-500',
              },
              {
                name: 'Feature Eng.',
                desc: 'Selected & scaled features',
                accuracy: '~85%',
                color: 'bg-blue-500',
              },
              {
                name: 'Advanced',
                desc: 'Apriori interaction features',
                accuracy: '~91%',
                color: 'bg-green-500',
              },
            ].map((model, i) => (
              <div key={i} className="p-3 rounded-xl bg-gray-50 border border-gray-100">
                <div className="flex items-center gap-2 mb-1">
                  <div className={`w-2.5 h-2.5 rounded-full ${model.color}`} />
                  <span className="text-sm font-semibold text-gray-800">{model.name}</span>
                  <span className="ml-auto text-xs font-mono font-bold text-forest-600">
                    {model.accuracy}
                  </span>
                </div>
                <p className="text-xs text-gray-500 ml-[18px]">{model.desc}</p>
              </div>
            ))}
          </div>

          <div className="p-3 rounded-xl bg-harvest-50 border border-harvest-100">
            <div className="flex items-center gap-2">
              <Thermometer size={14} className="text-harvest-600" />
              <span className="text-xs font-medium text-harvest-700">Optimal Conditions</span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-harvest-600">
              <div className="flex items-center gap-1">
                <Thermometer size={12} /> 20-32°C
              </div>
              <div className="flex items-center gap-1">
                <CloudRain size={12} /> 50-250mm
              </div>
              <div className="flex items-center gap-1">
                <Droplets size={12} /> 40-80%
              </div>
              <div className="flex items-center gap-1">
                <MapPin size={12} /> 15 Regions
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
