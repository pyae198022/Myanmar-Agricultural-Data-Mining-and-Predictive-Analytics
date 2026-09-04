import React from 'react';
import {
  LayoutDashboard,
  Sprout,
  BarChart3,
  GitCompare,
  Leaf,
  TrendingUp,
  Database,
  GitBranch,
} from 'lucide-react';

const navGroups = [
  {
    label: 'Overview',
    items: [
      { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    ],
  },
  {
    label: 'Predict',
    items: [
      { id: 'cropType', label: 'Crop Prediction', icon: Sprout },
      { id: 'cropYield', label: 'Yield Prediction', icon: BarChart3 },
      { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
    ],
  },
  {
    label: 'Explore',
    items: [
      { id: 'historicalTrends', label: 'Historical Trends', icon: TrendingUp },
      { id: 'dataStatistics', label: 'Data Statistics', icon: Database },
      { id: 'descriptiveMining', label: 'Descriptive Mining', icon: GitBranch },
    ],
  },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="h-full flex flex-col bg-gradient-to-b from-forest-700 via-forest-800 to-forest-950 text-white">
      <div className="px-5 py-6 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-harvest-400 to-harvest-600 flex items-center justify-center shadow-lg">
            <Leaf size={22} className="text-white" aria-hidden="true" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">Lal Yar Link</h1>
            <p className="text-xs text-forest-200 font-medium">Decision Support System</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-5 overflow-y-auto space-y-5" aria-label="Main">
        {navGroups.map((group) => (
          <div key={group.label}>
            <p className="text-[11px] font-semibold text-forest-300 uppercase tracking-wider px-3 mb-2">
              {group.label}
            </p>
            <div className="space-y-1">
              {group.items.map((item) => {
                const Icon = item.icon;
                const isActive = activePage === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onNavigate(item.id)}
                    aria-current={isActive ? 'page' : undefined}
                    className={`sidebar-link w-full ${
                      isActive ? 'sidebar-link-active' : 'sidebar-link-inactive'
                    }`}
                  >
                    <Icon size={18} />
                    <span>{item.label}</span>
                    {isActive && (
                      <span className="ml-auto w-1.5 h-1.5 rounded-full bg-harvest-500" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="px-4 py-4 border-t border-white/10">
        <p className="text-[11px] text-forest-300 leading-relaxed">
          Multi-model crop prediction and yield analysis for Myanmar agricultural data.
        </p>
      </div>
    </aside>
  );
}
