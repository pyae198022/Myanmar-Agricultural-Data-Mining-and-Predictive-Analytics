import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import {
  Award, Target, Percent, TrendingUp,
  CheckCircle2, Sparkles,
} from 'lucide-react';

const COLORS = ['#2d9f63', '#4fba7f', '#82d3a5', '#b4e6c7', '#d8f3e1'];

const COLOR_MAP = {
  forest: { bg: 'bg-forest-50', border: 'border-forest-100', icon: 'text-forest-500', text: 'text-forest-700' },
  blue:   { bg: 'bg-blue-50',   border: 'border-blue-100',   icon: 'text-blue-500',   text: 'text-blue-700' },
  harvest:{ bg: 'bg-amber-50',  border: 'border-amber-100',  icon: 'text-amber-500',  text: 'text-amber-700' },
  earth:  { bg: 'bg-lime-50',   border: 'border-lime-100',   icon: 'text-lime-600',   text: 'text-lime-700' },
  red:    { bg: 'bg-red-50',    border: 'border-red-100',    icon: 'text-red-500',    text: 'text-red-700' },
  indigo: { bg: 'bg-indigo-50', border: 'border-indigo-100', icon: 'text-indigo-500', text: 'text-indigo-700' },
  cyan:   { bg: 'bg-cyan-50',   border: 'border-cyan-100',   icon: 'text-cyan-500',   text: 'text-cyan-700' },
};

function MetricBadge({ label, value, icon: Icon, color = 'forest', suffix = '' }) {
  const c = COLOR_MAP[color] || COLOR_MAP.forest;
  return (
    <div className={`metric-card ${c.bg} ${c.border}`}>
      <div className="flex items-center gap-2 mb-1">
        <Icon size={15} className={c.icon} />
        <span className="text-xs font-medium text-gray-500">{label}</span>
      </div>
      <p className={`text-xl font-bold ${c.text}`}>
        {typeof value === 'number' ? (value * 100).toFixed(1) : value}{suffix || '%'}
      </p>
    </div>
  );
}

export default function CropTypeResults({ result }) {
  if (!result) return null;

  const { predictedCrop, confidence, topPredictions, featureImportance, modelMetrics } = result;

  // Radar chart data for metrics
  const radarData = [
    { metric: 'Accuracy', value: modelMetrics.accuracy },
    { metric: 'Precision', value: modelMetrics.precision },
    { metric: 'F1-Score', value: modelMetrics.f1Score },
    { metric: 'Recall', value: modelMetrics.recall },
    { metric: 'Confidence', value: confidence },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Result Card */}
      <div className="glass-card p-6 bg-gradient-to-br from-forest-50 to-earth-50 border-forest-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={18} className="text-harvest-500" />
              <span className="text-sm font-medium text-forest-600">Prediction Result</span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold text-forest-800 mb-1">
              🌾 {predictedCrop}
            </h2>
            <p className="text-sm text-gray-600">
              Recommended crop based on your environmental conditions
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-center">
              <div className="relative w-20 h-20">
                <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
                  <circle cx="40" cy="40" r="34" fill="none" stroke="#e5e7eb" strokeWidth="6" />
                  <circle
                    cx="40" cy="40" r="34" fill="none"
                    stroke="#2d9f63" strokeWidth="6" strokeLinecap="round"
                    strokeDasharray={`${confidence * 213.6} 213.6`}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-lg font-bold text-forest-700">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">Confidence</p>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricBadge label="Accuracy" value={modelMetrics.accuracy} icon={Target} color="forest" />
        <MetricBadge label="F1-Score" value={modelMetrics.f1Score} icon={Award} color="blue" />
        <MetricBadge label="Precision" value={modelMetrics.precision} icon={CheckCircle2} color="harvest" />
        <MetricBadge label="Recall" value={modelMetrics.recall} icon={TrendingUp} color="earth" />
      </div>

      {/* Charts Row */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Top Predictions Bar Chart */}
        <div className="glass-card p-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-4">Top Crop Probabilities</h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topPredictions} layout="vertical">
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  type="category"
                  dataKey="crop"
                  tick={{ fontSize: 11, fill: '#374151' }}
                  axisLine={false}
                  tickLine={false}
                  width={80}
                />
                <Tooltip
                  formatter={v => [`${(v * 100).toFixed(1)}%`, 'Probability']}
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
                  {topPredictions.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Importance Pie */}
        <div className="glass-card p-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-4">Feature Importance</h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={featureImportance.slice(0, 6)}
                  dataKey="importance"
                  nameKey="feature"
                  cx="50%"
                  cy="50%"
                  outerRadius={85}
                  innerRadius={40}
                  paddingAngle={2}
                >
                  {featureImportance.slice(0, 6).map((_, i) => (
                    <Cell
                      key={i}
                      fill={[
                        '#2d9f63', '#e5a03c', '#2563eb',
                        '#dc2626', '#8b5cf6', '#06b6d4',
                      ][i]}
                    />
                  ))}
                </Pie>
                <Tooltip
                  formatter={v => [`${(v * 100).toFixed(1)}%`, 'Importance']}
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    fontSize: '12px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          {/* Legend */}
          <div className="grid grid-cols-2 gap-1 mt-2">
            {featureImportance.slice(0, 6).map((f, i) => (
              <div key={i} className="flex items-center gap-1.5 text-xs text-gray-600">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: [
                      '#2d9f63', '#e5a03c', '#2563eb',
                      '#dc2626', '#8b5cf6', '#06b6d4',
                    ][i],
                  }}
                />
                {f.feature}
              </div>
            ))}
          </div>
        </div>

        {/* Radar Chart */}
        <div className="glass-card p-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-4">Model Performance Radar</h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{ fontSize: 10, fill: '#6b7280' }}
                />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 1]}
                  tick={{ fontSize: 9, fill: '#9ca3af' }}
                />
                <Radar
                  dataKey="value"
                  stroke="#2d9f63"
                  fill="#2d9f63"
                  fillOpacity={0.25}
                  strokeWidth={2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Feature Importance Table */}
      <div className="glass-card p-5">
        <h4 className="text-sm font-semibold text-gray-800 mb-4">Feature Importance Breakdown</h4>
        <div className="space-y-2.5">
          {featureImportance.map((f, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs font-medium text-gray-600 w-32 shrink-0 truncate">
                {f.feature}
              </span>
              <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-forest-500 to-forest-400 transition-all duration-700"
                  style={{ width: `${f.importance * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono font-semibold text-gray-700 w-14 text-right">
                {(f.importance * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
