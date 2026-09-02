import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import CropTypePrediction from './components/CropTypePrediction';
import CropYieldPrediction from './components/CropYieldPrediction';
import ModelComparison from './components/ModelComparison';
import HistoricalTrends from './components/HistoricalTrends';
import { Menu, X } from 'lucide-react';

const PAGES = {
  dashboard: 'dashboard',
  cropType: 'cropType',
  cropYield: 'cropYield',
  comparison: 'comparison',
  historicalTrends: 'historicalTrends',
};

function App() {
  const [activePage, setActivePage] = useState(PAGES.dashboard);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const renderPage = () => {
    switch (activePage) {
      case PAGES.dashboard:
        return <Dashboard onNavigate={setActivePage} />;
      case PAGES.cropType:
        return <CropTypePrediction />;
      case PAGES.cropYield:
        return <CropYieldPrediction />;
      case PAGES.comparison:
        return <ModelComparison />;
      case PAGES.historicalTrends:
        return <HistoricalTrends onNavigate={setActivePage} />;
      default:
        return <Dashboard onNavigate={setActivePage} />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 z-50 w-72 transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <Sidebar
          activePage={activePage}
          onNavigate={(page) => {
            setActivePage(page);
            setSidebarOpen(false);
          }}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Mobile Header */}
        <header className="lg:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          <h1 className="text-lg font-bold text-forest-700">Lal Yar Link</h1>
          <div className="w-10" /> {/* Spacer */}
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="animate-fade-in">
            {renderPage()}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
