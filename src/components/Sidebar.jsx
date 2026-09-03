import React from 'react';
import {
  LayoutDashboard,
  Sprout,
  BarChart3,
  GitCompare,
  Leaf,
  Sun,
  Droplets,
  Info,
} from 'lucide-react';

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'cropType', label: 'Crop Prediction', icon: Sprout },
  { id: 'cropYield', label: 'Yield Prediction', icon: BarChart3 },
  { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="h-full flex flex-col bg-gradient-to-b from-forest-700 via-forest-800 to-forest-950 text-white">
      {/* Logo / Brand */}
      <div className="px-6 py-6 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-harvest-400 to-harvest-600 flex items-center justify-center shadow-lg">
            <Leaf size={22} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">Lal Yar Link</h1>
            <p className="text-xs text-forest-200 font-medium">Decision Support System</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-1.5">
        <p className="text-xs font-semibold text-forest-300 uppercase tracking-wider px-4 mb-3">
          Navigation
        </p>
        {navItems.map(item => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`sidebar-link w-full ${
                isActive ? 'sidebar-link-active' : 'sidebar-link-inactive'
              }`}
            >
              <Icon size={20} />
              <span>{item.label}</span>
              {isActive && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-harvest-400" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Info Card */}
      <div className="px-4 pb-6">
        <div className="p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
          <div className="flex items-start gap-3">
            <Info size={18} className="text-harvest-300 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-semibold text-forest-100 mb-1">About</p>
              <p className="text-xs text-forest-300 leading-relaxed">
                Agricultural Decision Support with multi-model crop prediction & yield analysis.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-white/10">
        <div className="flex items-center gap-2 text-xs text-forest-300">
          <Sun size={14} className="text-harvest-400" />
          <span>Agricultural Data Analytics</span>
          <Droplets size={14} className="text-blue-300 ml-auto" />
        </div>
      </div>
    </aside>
  );
}
