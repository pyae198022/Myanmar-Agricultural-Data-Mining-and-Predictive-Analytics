import React, { useEffect, useRef, useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import CropTypePrediction from './components/CropTypePrediction';
import CropYieldPrediction from './components/CropYieldPrediction';
import ModelComparison from './components/ModelComparison';
import HistoricalTrends from './components/HistoricalTrends';
import DataStatistics from './components/DataStatistics';
import DescriptiveMining from './components/DescriptiveMining';
import { Menu, X } from 'lucide-react';

const PAGES = {
  dashboard: 'dashboard',
  cropType: 'cropType',
  cropYield: 'cropYield',
  comparison: 'comparison',
  historicalTrends: 'historicalTrends',
  dataStatistics: 'dataStatistics',
  descriptiveMining: 'descriptiveMining',
};

const PAGE_META = {
  dashboard: { title: 'Dashboard', crumb: 'Overview' },
  cropType: { title: 'Crop Prediction', crumb: 'Predict' },
  cropYield: { title: 'Yield Prediction', crumb: 'Predict' },
  comparison: { title: 'Model Comparison', crumb: 'Predict' },
  historicalTrends: { title: 'Historical Trends', crumb: 'Explore' },
  dataStatistics: { title: 'Data Statistics', crumb: 'Explore' },
  descriptiveMining: { title: 'Descriptive Mining', crumb: 'Explore' },
};

function App() {
  const [activePage, setActivePage] = useState(PAGES.dashboard);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const mainRef = useRef(null);

  const navigate = (page) => {
    setActivePage(page);
    setSidebarOpen(false);
  };

  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0 });
  }, [activePage]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') setSidebarOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const renderPage = () => {
    switch (activePage) {
      case PAGES.dashboard:
        return <Dashboard onNavigate={navigate} />;
      case PAGES.cropType:
        return <CropTypePrediction />;
      case PAGES.cropYield:
        return <CropYieldPrediction />;
      case PAGES.comparison:
        return <ModelComparison />;
      case PAGES.historicalTrends:
        return <HistoricalTrends onNavigate={navigate} />;
      case PAGES.dataStatistics:
        return <DataStatistics />;
      case PAGES.descriptiveMining:
        return <DescriptiveMining />;
      default:
        return <Dashboard onNavigate={navigate} />;
    }
  };

  const meta = PAGE_META[activePage] || PAGE_META.dashboard;

  return (
    <div className="flex h-screen overflow-hidden bg-[#f4f6f1]">
      <a
        href="#main-content"
        className="skip-link"
      >
        Skip to content
      </a>

      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-forest-950/40 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <div
        className={`fixed inset-y-0 left-0 z-50 w-72 transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <Sidebar activePage={activePage} onNavigate={navigate} />
      </div>

      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <header className="lg:hidden flex items-center justify-between px-4 py-3 bg-white/90 backdrop-blur border-b border-gray-200/80">
          <button
            type="button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            aria-label={sidebarOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={sidebarOpen}
          >
            {sidebarOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
          <div className="text-center min-w-0">
            <p className="text-sm font-bold text-forest-800 truncate">{meta.title}</p>
            <p className="text-[11px] text-gray-400">{meta.crumb}</p>
          </div>
          <div className="w-10" />
        </header>

        <header className="hidden lg:flex items-center justify-between px-8 py-3.5 bg-white/75 backdrop-blur-md border-b border-gray-200/70">
          <div>
            <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">
              {meta.crumb}
            </p>
            <h2 className="text-sm font-semibold text-gray-800">{meta.title}</h2>
          </div>
          <p className="text-xs text-gray-400">Lal Yar Link · Decision Support</p>
        </header>

        <main
          id="main-content"
          ref={mainRef}
          className="flex-1 overflow-y-auto"
          tabIndex={-1}
        >
          <div key={activePage} className="animate-fade-in">
            {renderPage()}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
