import React from 'react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  ArrowUpDown, BarChart3,
  Target, Gauge, Sparkles,
} from 'lucide-react';

const LEVEL_CONFIG = {
  Low: { color: 'red', emoji: '📉', bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700' },
  High: { color: 'forest', emoji: '📈', bg: 'bg-forest-50', border: 'border-forest-200', text: 'text-forest-700' },
};

const REG_COLOR_MAP = {
  forest: { bg: 'bg-forest-50', border: 'border-forest-100', icon: 'text-forest-500', text: 'text-forest-700' },
  blue:   { bg: 'bg-blue-50',   border: 'border-blue-100',   icon: 'text-blue-500',   text: 'text-blue-700' },
  harvest:{ bg: 'bg-amber-50',  border: 'border-amber-100',  icon: 'text-amber-500',  text: 'text-amber-700' },
  red:    { bg: 'bg-red-50',    border: 'border-red-100',    icon: 'text-red-500',    text: 'text-red-700' },
  indigo: { bg: 'bg-indigo-50', border: 'border-indigo-100', icon: 'text-indigo-500', text: 'text-indigo-700' },
};

function RegressionMetricCard({ label, value, unit, icon: Icon, color, lowerIsBetter = false }) {
  if (value == null) return null;
  const c = REG_COLOR_MAP[color] || REG_COLOR_MAP.forest;
  return (
    <div className={`metric-card ${c.bg} ${c.border}`}>
      <div className="flex items-center gap-2 mb-1">
        <Icon size={15} className={c.icon} />
        <span className="text-xs font-medium text-gray-500">{label}</span>
        {lowerIsBetter && (
          <span className="ml-auto text-[10px] font-medium text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">
            Lower is better
          </span>
        )}
      </div>
      <p className={`text-xl font-bold ${c.text}`}>
        {value}{unit}
      </p>
    </div>
  );
}

export default function CropYieldResults({ result, inputs }) {
  if (!result) return null;

  const {
    predictedYield, featureImportance, modelMetrics, yieldLevelInfo,
  } = result;

  const m = modelMetrics || {};
  const r2 = m.r2 != null ? m.r2 : m.r2Score;
  const rmse = m.rmse;
  const mae = m.mae;

  // Binary yield level from the real yield-level endpoint (Low/High only).
  const level = yieldLevelInfo && yieldLevelInfo.predictedLevel ? yieldLevelInfo.predictedLevel : null;
  const levelConf = LEVEL_CONFIG[level];

  // Real metric keys for charts (ignore nulls).
  const barData = (featureImportance || []).filter(f => f.importance != null);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Result Card */}
      <div className={`glass-card p-6 ${levelConf ? `${levelConf.bg} ${levelConf.border} border` : 'border border-gray-100'}`}>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={18} className="text-harvest-500" />
              <span className="text-sm font-medium text-forest-600">Yield Prediction Result</span>
            </div>
            <div className="flex items-baseline gap-3 mb-2">
              <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">
                {Number(predictedYield).toFixed(2)}
              </h2>
              <span className="text-lg font-medium text-gray-500">tons</span>
            </div>
            <p className="text-sm text-gray-600">
              Predicted production for {inputs.cropType} in {inputs.region} ({inputs.sownAcre} acres)
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Yield Level Badge (binary Low/High from backend) */}
            {levelConf ? (
              <div className={`px-5 py-3 rounded-xl ${levelConf.bg} border ${levelConf.border} text-center`}>
                <span className="text-2xl">{levelConf.emoji}</span>
                <div className="flex items-center gap-1.5 mt-1">
                  <span className={`text-sm font-bold ${levelConf.text}`}>{level} Yield</span>
                </div>
                {yieldLevelInfo && (
                  <p className="text-[10px] text-gray-400 mt-0.5">
                    P(High) {Number(yieldLevelInfo.probabilityHigh).toFixed(1)} · P(Low) {Number(yieldLevelInfo.probabilityLow).toFixed(1)}
                  </p>
                )}
              </div>
            ) : (
              <div className="px-5 py-3 rounded-xl bg-gray-50 border border-gray-100 text-center">
                <span className="text-xs text-gray-500">Yield level model not available</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Regression Metrics */}
      <div>
        <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
          <BarChart3 size={16} className="text-forest-500" />
          Regression Metrics
        </h4>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          <RegressionMetricCard label="R² Score" value={r2 == null ? null : r2.toFixed(3)} unit="" icon={Target} color="forest" />
          <RegressionMetricCard label="RMSE" value={rmse == null ? null : rmse.toFixed(2)} unit="" icon={ArrowUpDown} color="red" lowerIsBetter />
          <RegressionMetricCard label="MAE" value={mae == null ? null : mae.toFixed(2)} unit="" icon={Gauge} color="harvest" lowerIsBetter />
        </div>
      </div>

      {/* Feature Importance */}
      <div className="glass-card p-5">
        <h4 className="text-sm font-semibold text-gray-800 mb-4">Feature Importance</h4>
        {barData.length > 0 ? (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} layout="vertical">
                <XAxis
                  type="number"
                  domain={[0, 'auto']}
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 10, fill: '#374151' }}
                  axisLine={false}
                  tickLine={false}
                  width={100}
                />
                <Tooltip
                  formatter={v => [`${(v * 100).toFixed(1)}%`, 'Importance']}
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="importance" radius={[0, 6, 6, 0]} fill="#2d9f63">
                  {barData.map((_, i) => (
                    <Cell key={i} fill={CELL_COLORS[i % CELL_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-sm text-gray-500">Feature importance is not available for this model.</p>
        )}
      </div>
    </div>
  );
}

const CELL_COLORS = ['#2d9f63', '#e5a03c', '#2563eb', '#dc2626', '#8b5cf6', '#06b6d4', '#ec4899', '#f59e0b', '#10b981'];
