import React from 'react';
import { ArrowLeft } from 'lucide-react';

export default function PageHeader({
  icon: Icon,
  title,
  subtitle,
  gradient = 'from-forest-500 to-earth-500',
  onBack,
  actions,
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
      <div className="flex items-start gap-3 min-w-0">
        {onBack && (
          <button
            type="button"
            onClick={onBack}
            className="mt-0.5 w-10 h-10 rounded-xl bg-white border border-gray-200 flex items-center justify-center text-gray-600 hover:bg-gray-50 hover:border-forest-200 transition-colors shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest-400"
            aria-label="Back to dashboard"
          >
            <ArrowLeft size={18} />
          </button>
        )}
        {Icon && (
          <div
            className={`w-11 h-11 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-sm shrink-0`}
            aria-hidden="true"
          >
            <Icon size={22} className="text-white" />
          </div>
        )}
        <div className="min-w-0">
          <h1 className="text-xl sm:text-2xl font-bold text-gray-900 tracking-tight">
            {title}
          </h1>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-0.5 leading-relaxed">{subtitle}</p>
          )}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

export function InfoBanner({ icon: Icon, title, children, tone = 'forest' }) {
  const tones = {
    forest: 'bg-forest-50 border-forest-100',
    harvest: 'bg-harvest-50 border-harvest-100',
    blue: 'bg-blue-50 border-blue-100',
    red: 'bg-red-50 border-red-200',
    amber: 'bg-amber-50 border-amber-100',
  };
  const titleColor = {
    forest: 'text-forest-800',
    harvest: 'text-harvest-800',
    blue: 'text-blue-800',
    red: 'text-red-700',
    amber: 'text-amber-800',
  };
  const bodyColor = {
    forest: 'text-forest-600',
    harvest: 'text-harvest-600',
    blue: 'text-blue-600',
    red: 'text-red-600',
    amber: 'text-amber-700',
  };
  const iconColor = {
    forest: 'text-forest-600',
    harvest: 'text-harvest-600',
    blue: 'text-blue-600',
    red: 'text-red-500',
    amber: 'text-amber-600',
  };

  return (
    <div className={`flex items-start gap-3 p-4 rounded-xl border ${tones[tone] || tones.forest}`}>
      {Icon && <Icon size={18} className={`${iconColor[tone]} mt-0.5 shrink-0`} aria-hidden="true" />}
      <div className="min-w-0">
        {title && <p className={`text-sm font-medium ${titleColor[tone]}`}>{title}</p>}
        <div className={`text-xs leading-relaxed ${bodyColor[tone]} ${title ? 'mt-0.5' : ''}`}>
          {children}
        </div>
      </div>
    </div>
  );
}

export function ResultsPlaceholder({ title, hint }) {
  return (
    <div className="glass-card p-6 flex flex-col items-center justify-center text-center min-h-[140px] border-dashed">
      <p className="text-sm font-semibold text-gray-800">{title}</p>
      <p className="text-xs text-gray-500 mt-1.5 max-w-xl leading-relaxed">{hint}</p>
    </div>
  );
}
