import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area,
} from 'recharts';
import {
  TrendingUp, TrendingDown, ArrowUpDown, BarChart3,
  Target, Gauge, Award, CheckCircle2, Sparkles,
  ArrowUp, ArrowDown, Minus,
} from 'lucide-react';
import { getHistoricalTrends } from '../services/mockInference';

const LEVEL_CONFIG = {
  Low: { color: 'red', icon: TrendingDown, emoji: '📉', bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700' },
  Medium: { color: 'harvest', icon: Minus, emoji: '📊', bg: 'bg-harvest-50', border: 'border-harvest-200', text: 'text-harvest-700' },
  High: { color: 'forest', icon: TrendingUp, emoji: '📈', bg: 'bg-forest-50', border: 'border-forest-200', text: 'text-forest-700' },
};

const REG_COLOR_MAP = {
  forest: { bg: 'bg-forest-50', border: 'border-forest-100', icon: 'text-forest-500', text: 'text-forest-700' },
  blue:   { bg: 'bg-blue-50',   border: 'border-blue-100',   icon: 'text-blue-500',   text: 'text-blue-700' },
  harvest:{ bg: 'bg-amber-50',  border: 'border-amber-100',  icon: 'text-amber-500',  text: 'text-amber-700' },
  red:    { bg: 'bg-red-50',    border: 'border-red-100',    icon: 'text-red-500',    text: 'text-red-700' },
  indigo: { bg: 'bg-indigo-50', border: 'border-indigo-100', icon: 'text-indigo-500', text: 'text-indigo-700' },
};

function RegressionMetricCard({ label, value, unit, icon: Icon, color, lowerIsBetter = false }) {
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
    predictedYield, yieldLevel, confidenceInterval,
    featureImportance, modelMetrics,
  } = result;

  const levelConf = LEVEL_CONFIG[yieldLevel];
  const LevelIcon = levelConf.icon;

  // Historical trend for context
  const trendData = getHistoricalTrends(inputs.region, inputs.cropType);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Result Card */}
      <div className={`glass-card p-6 ${levelConf.bg} ${levelConf.border} border`}>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={18} className="text-harvest-500" />
              <span className="text-sm font-medium text-forest-600">Yield Prediction Result</span>
            </div>
            <div className="flex items-baseline gap-3 mb-2">
              <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">
                {predictedYield.toFixed(2)}
              </h2>
              <span className="text-lg font-medium text-gray-500">tons</span>
            </div>
            <p className="text-sm text-gray-600">
              Predicted production for {inputs.cropType} in {inputs.region} ({inputs.sownAcre} acres)
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Yield Level Badge */}
            <div className={`px-5 py-3 rounded-xl ${levelConf.bg} border ${levelConf.border} text-center`}>
              <span className="text-2xl">{levelConf.emoji}</span>
              <div className="flex items-center gap-1.5 mt-1">
                <LevelIcon size={14} className={levelConf.text} />
                <span className={`text-sm font-bold ${levelConf.text}`}>{yieldLevel} Yield</span>
              </div>
            </div>

            {/* Confidence Interval */}
            <div className="text-center p-3 rounded-xl bg-white/60 border border-gray-100">
              <p className="text-xs text-gray-500 mb-1">95% CI</p>
              <div className="flex items-center gap-1.5">
                <ArrowDown size={12} className="text-blue-400" />
                <span className="text-sm font-mono font-bold text-gray-700">
                  {confidenceInterval.lower}
                </span>
              </div>
              <div className="w-6 h-px bg-gray-300 mx-auto my-1" />
              <div className="flex items-center gap-1.5">
                <ArrowUp size={12} className="text-green-500" />
                <span className="text-sm font-mono font-bold text-gray-700">
                  {confidenceInterval.upper}
                </span>
              </div>
            </div>
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
          <RegressionMetricCard
            label="R² Score"
            value={modelMetrics.r2Score.toFixed(3)}
            unit=""
            icon={Target}
            color="forest"
          />
          <RegressionMetricCard
            label="RMSE"
            value={modelMetrics.rmse.toFixed(2)}
            unit=""
            icon={ArrowUpDown}
            color="red"
            lowerIsBetter
          />
          <RegressionMetricCard
            label="MAE"
            value={modelMetrics.mae.toFixed(2)}
            unit=""
            icon={Gauge}
            color="harvest"
            lowerIsBetter
          />
        </div>
      </div>

      {/* Classification Metrics */}
      <div>
        <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
          <CheckCircle2 size={16} className="text-blue-500" />
          Yield Level Classification
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <RegressionMetricCard
            label="Classification Accuracy"
            value={(modelMetrics.classificationAccuracy * 100).toFixed(1)}
            unit="%"
            icon={Award}
            color="blue"
          />
          <RegressionMetricCard
            label="Classification F1"
            value={(modelMetrics.classificationF1 * 100).toFixed(1)}
            unit="%"
            icon={Award}
            color="indigo"
          />
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Feature Importance */}
        <div className="glass-card p-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-4">Feature Importance</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportance} layout="vertical">
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
                  {featureImportance.map((_, i) => (
                    <Cell
                      key={i}
                      fill={[
                        '#2d9f63', '#e5a03c', '#2563eb', '#dc2626',
                        '#8b5cf6', '#06b6d4', '#ec4899', '#f59e0b', '#10b981',
                      ][i % 9]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Historical Yield Trend */}
        <div className="glass-card p-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-1">Historical Yield Trend</h4>
          <p className="text-xs text-gray-500 mb-4">
            {inputs.cropType} in {inputs.region} (2015–2025)
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="yieldAreaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2d9f63" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#2d9f63" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="rainAreaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="year"
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    fontSize: '12px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="yield"
                  stroke="#2d9f63"
                  strokeWidth={2}
                  fill="url(#yieldAreaGrad)"
                  name="Yield (tons)"
                />
                <Area
                  type="monotone"
                  dataKey="rainfall"
                  stroke="#2563eb"
                  strokeWidth={1.5}
                  fill="url(#rainAreaGrad)"
                  name="Rainfall (mm)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
