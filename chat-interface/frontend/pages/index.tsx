import { Activity, Brain, ChevronLeft, ChevronRight, Settings, Zap } from 'lucide-react';
import Head from 'next/head';
import { useEffect, useState } from 'react';
import { Toaster } from 'react-hot-toast';

import ChatInterface from '@/components/ChatInterface';
import SystemStatus from '@/components/SystemStatus';
import { SpectacularAPI } from '@/lib/api';
import { SystemStatus as SystemStatusType } from '@/types/api';

export default function Home() {
  const [showSystemStatus, setShowSystemStatus] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatusType | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(false);
  const [lastStatusUpdate, setLastStatusUpdate] = useState<Date | null>(null);

  // Load system status on mount and periodically
  useEffect(() => {
    loadSystemStatus();
    
    // Refresh status every 30 seconds
    const interval = setInterval(loadSystemStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const loadSystemStatus = async () => {
    setIsLoadingStatus(true);
    try {
      const status = await SpectacularAPI.getSystemStatus();
      setSystemStatus(status);
      setLastStatusUpdate(new Date());
    } catch (error) {
      console.error('Failed to load system status:', error);
    } finally {
      setIsLoadingStatus(false);
    }
  };

  const getSystemHealthColor = () => {
    if (!systemStatus) return 'text-gray-400';
    
    switch (systemStatus.status.toLowerCase()) {
      case 'healthy':
        return 'text-green-400';
      case 'warning':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <>
      <Head>
        <title>Spectacular - AI Reasoning with Triple Validation</title>
        <meta name="description" content="Advanced AI chat interface with Bayesian Evidence Network, environmental sensor integration, and triple validation plots" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full blur-3xl animate-pulse-slow"></div>
          <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-full blur-3xl animate-pulse-slow animation-delay-75"></div>
        </div>

        {/* Header */}
        <header className="relative z-10 border-b border-white/10 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              {/* Logo and Title */}
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <Brain className="w-8 h-8 text-blue-400" />
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-purple-400 rounded-full animate-ping"></div>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white text-shadow">Spectacular</h1>
                  <p className="text-sm text-gray-300">Bayesian Evidence Network</p>
                </div>
              </div>

              {/* System Status Toggle */}
              <div className="flex items-center space-x-4">
                {/* Status Indicator */}
                <div className="flex items-center space-x-2">
                  <Activity className={`w-4 h-4 ${getSystemHealthColor()}`} />
                  <span className={`text-sm font-medium ${getSystemHealthColor()}`}>
                    {systemStatus?.status || 'Unknown'}
                  </span>
                  {lastStatusUpdate && (
                    <span className="text-xs text-gray-400">
                      (Updated {lastStatusUpdate.toLocaleTimeString()})
                    </span>
                  )}
                </div>

                {/* Status Panel Toggle */}
                <button
                  onClick={() => setShowSystemStatus(!showSystemStatus)}
                  className="btn-secondary flex items-center space-x-2"
                >
                  <Settings className="w-4 h-4" />
                  <span className="hidden sm:inline">System</span>
                  {showSystemStatus ? (
                    <ChevronRight className="w-4 h-4" />
                  ) : (
                    <ChevronLeft className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>

            {/* Feature Highlights */}
            <div className="mt-4 flex items-center justify-center space-x-6 text-sm text-gray-300">
              <div className="flex items-center space-x-1">
                <Zap className="w-3 h-3 text-yellow-400" />
                <span>Triple Validation</span>
              </div>
              <div className="flex items-center space-x-1">
                <Brain className="w-3 h-3 text-purple-400" />
                <span>Bayesian Network</span>
              </div>
              <div className="flex items-center space-x-1">
                <Activity className="w-3 h-3 text-green-400" />
                <span>Environmental Sensors</span>
              </div>
              <div className="flex items-center space-x-1">
                <Settings className="w-3 h-3 text-blue-400" />
                <span>Dynamic Routing</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="relative z-10 flex h-[calc(100vh-120px)]">
          {/* Chat Interface */}
          <div className={`flex-1 transition-all duration-300 ${showSystemStatus ? 'mr-96' : ''}`}>
            <ChatInterface className="h-full" />
          </div>

          {/* System Status Sidebar */}
          <div className={`fixed top-[120px] right-0 h-[calc(100vh-120px)] w-96 transform transition-transform duration-300 z-20 ${
            showSystemStatus ? 'translate-x-0' : 'translate-x-full'
          }`}>
            <div className="h-full bg-black/20 backdrop-blur-md border-l border-white/10 overflow-y-auto">
              <div className="p-6">
                {systemStatus ? (
                  <SystemStatus status={systemStatus} isLoading={isLoadingStatus} />
                ) : (
                  <div className="glass-effect rounded-xl p-6">
                    <div className="animate-pulse space-y-4">
                      <div className="h-4 bg-gray-600 rounded w-3/4"></div>
                      <div className="h-4 bg-gray-600 rounded w-1/2"></div>
                      <div className="h-4 bg-gray-600 rounded w-2/3"></div>
                    </div>
                    <p className="text-gray-400 text-center mt-4">Loading system status...</p>
                  </div>
                )}

                {/* Refresh Button */}
                <button
                  onClick={loadSystemStatus}
                  disabled={isLoadingStatus}
                  className="w-full mt-4 btn-secondary disabled:opacity-50"
                >
                  {isLoadingStatus ? (
                    <>
                      <Activity className="w-4 h-4 mr-2 animate-spin" />
                      Refreshing...
                    </>
                  ) : (
                    <>
                      <Activity className="w-4 h-4 mr-2" />
                      Refresh Status
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Overlay for mobile when sidebar is open */}
          {showSystemStatus && (
            <div 
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-10 lg:hidden"
              onClick={() => setShowSystemStatus(false)}
            />
          )}
        </main>

        {/* Footer */}
        <footer className="relative z-10 border-t border-white/10 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
            <div className="flex items-center justify-between text-sm text-gray-400">
              <div className="flex items-center space-x-2">
                <span>Powered by Spectacular Framework</span>
                <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                <span>Multi-dimensional Reasoning</span>
              </div>
              <div className="flex items-center space-x-4">
                <span>Environmental Integration Active</span>
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              </div>
            </div>
          </div>
        </footer>

        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'rgba(0, 0, 0, 0.8)',
              backdropFilter: 'blur(12px)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.1)',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#000',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#000',
              },
            },
          }}
        />
      </div>
    </>
  );
}
