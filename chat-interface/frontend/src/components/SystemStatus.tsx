import { SystemStatusProps } from '@/types/api';
import {
    Activity,
    AlertTriangle,
    Brain,
    CheckCircle,
    Clock,
    Cpu,
    Eye,
    Gauge,
    GitBranch,
    Layers,
    Network
} from 'lucide-react';
import React from 'react';

const SystemStatus: React.FC<SystemStatusProps> = ({ status, isLoading = false }) => {
  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const getStatusColor = (statusValue: string) => {
    switch (statusValue.toLowerCase()) {
      case 'healthy':
      case 'running':
      case 'active':
        return 'text-green-400';
      case 'warning':
      case 'degraded':
        return 'text-yellow-400';
      case 'error':
      case 'failed':
      case 'inactive':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = (statusValue: string) => {
    switch (statusValue.toLowerCase()) {
      case 'healthy':
      case 'running':
      case 'active':
        return <CheckCircle className="w-4 h-4" />;
      case 'warning':
      case 'degraded':
        return <AlertTriangle className="w-4 h-4" />;
      case 'error':
      case 'failed':
      case 'inactive':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const renderEnvironmentalSensors = (sensors: Record<string, any>) => {
    if (!sensors || Object.keys(sensors).length === 0) {
      return <span className="text-gray-400">No sensor data</span>;
    }

    const sensorNames = [
      'biometric_data',
      'spatial_context', 
      'temporal_dynamics',
      'quantum_correlations',
      'computational_load',
      'network_coherence'
    ];

    return (
      <div className="grid grid-cols-2 gap-2 text-xs">
        {sensorNames.slice(0, 6).map((sensorName) => {
          const sensor = sensors[sensorName];
          if (!sensor) return null;
          
          return (
            <div key={sensorName} className="flex items-center justify-between">
              <span className="text-gray-300 capitalize">
                {sensorName.replace(/_/g, ' ').slice(0, 12)}:
              </span>
              <span className={`font-mono ${
                sensor.measurement > 0.7 ? 'text-green-400' :
                sensor.measurement > 0.4 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {(sensor.measurement * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  const renderBayesianHealth = (health: any) => {
    if (!health) return null;

    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-gray-300 flex items-center">
            <GitBranch className="w-3 h-3 mr-1" />
            Active Nodes:
          </span>
          <span className="text-blue-400 font-mono">{health.nodes_active || 0}</span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-gray-300 flex items-center">
            <Network className="w-3 h-3 mr-1" />
            Coherence:
          </span>
          <span className={`font-mono ${
            health.network_coherence > 0.7 ? 'text-green-400' :
            health.network_coherence > 0.4 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {(health.network_coherence * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-gray-300 flex items-center">
            <Eye className="w-3 h-3 mr-1" />
            Validators:
          </span>
          <span className="text-purple-400 font-mono">{health.external_validators_available || 0}</span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-gray-300 flex items-center">
            <Layers className="w-3 h-3 mr-1" />
            Embeddings:
          </span>
          <span className="text-cyan-400 font-mono">{health.embedding_paths_active || 0}</span>
        </div>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="glass-effect rounded-xl p-6 animate-pulse">
        <div className="flex items-center space-x-2 mb-4">
          <div className="w-5 h-5 bg-gray-600 rounded animate-pulse"></div>
          <div className="w-32 h-5 bg-gray-600 rounded animate-pulse"></div>
        </div>
        <div className="space-y-3">
          <div className="w-full h-4 bg-gray-600 rounded animate-pulse"></div>
          <div className="w-3/4 h-4 bg-gray-600 rounded animate-pulse"></div>
          <div className="w-1/2 h-4 bg-gray-600 rounded animate-pulse"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-effect rounded-xl p-6 hover:bg-white/15 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Activity className="w-6 h-6 text-blue-400" />
          <h2 className="text-xl font-bold text-white">System Status</h2>
        </div>
        <div className={`flex items-center space-x-2 ${getStatusColor(status.status)}`}>
          {getStatusIcon(status.status)}
          <span className="font-medium capitalize">{status.status}</span>
        </div>
      </div>

      {/* Main Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* System Metrics */}
        <div className="space-y-4">
          <h3 className="text-white font-semibold flex items-center">
            <Gauge className="w-4 h-4 mr-2 text-green-400" />
            Core Metrics
          </h3>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-300 flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                Uptime:
              </span>
              <span className="text-blue-400 font-mono">{formatUptime(status.uptime)}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-300 flex items-center">
                <Brain className="w-3 h-3 mr-1" />
                Orchestrator:
              </span>
              <span className={getStatusColor(status.orchestrator_status)}>
                {status.orchestrator_status || 'Unknown'}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Recent Executions:</span>
              <span className="text-green-400 font-mono">{status.recent_executions || 0}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Avg Process Time:</span>
              <span className="text-yellow-400 font-mono">
                {status.average_processing_time ? `${status.average_processing_time.toFixed(2)}s` : 'N/A'}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Success Rate:</span>
              <span className={`font-mono ${
                status.success_rate > 0.9 ? 'text-green-400' :
                status.success_rate > 0.7 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {status.success_rate ? `${(status.success_rate * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        {/* Environmental Sensors */}
        <div className="space-y-4">
          <h3 className="text-white font-semibold flex items-center">
            <Cpu className="w-4 h-4 mr-2 text-orange-400" />
            Environmental Sensors
          </h3>
          {renderEnvironmentalSensors(status.environmental_sensors)}
        </div>
      </div>

      {/* Bayesian Network Health */}
      {status.bayesian_network_health && (
        <div className="mt-6 pt-6 border-t border-white/10">
          <h3 className="text-white font-semibold flex items-center mb-4">
            <Network className="w-4 h-4 mr-2 text-purple-400" />
            Bayesian Network Health
          </h3>
          {renderBayesianHealth(status.bayesian_network_health)}
        </div>
      )}

      {/* Pipeline Components */}
      {status.pipeline_components && Object.keys(status.pipeline_components).length > 0 && (
        <div className="mt-6 pt-6 border-t border-white/10">
          <h3 className="text-white font-semibold flex items-center mb-4">
            <Layers className="w-4 h-4 mr-2 text-cyan-400" />
            Pipeline Components
          </h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(status.pipeline_components).slice(0, 8).map(([name, componentStatus]) => (
              <div key={name} className="flex items-center justify-between">
                <span className="text-gray-300 capitalize">
                  {name.replace(/_/g, ' ').slice(0, 15)}:
                </span>
                <span className={getStatusColor(String(componentStatus))}>
                  {String(componentStatus)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Status Indicator */}
      <div className="absolute top-2 right-2">
        <div className={`w-3 h-3 rounded-full ${
          status.status === 'healthy' ? 'bg-green-400 animate-pulse' :
          status.status === 'warning' ? 'bg-yellow-400 animate-pulse' :
          'bg-red-400 animate-pulse'
        }`}></div>
      </div>
    </div>
  );
};

export default SystemStatus;
