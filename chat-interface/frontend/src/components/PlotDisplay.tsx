import { PlotDisplayProps } from '@/types/api';
import { AlertTriangle, BarChart3, Brain, CheckCircle, Clock, Target } from 'lucide-react';
import React from 'react';

const PlotDisplay: React.FC<PlotDisplayProps> = ({ plotData, type, isLoading = false }) => {
  const getPlotIcon = (plotType: string) => {
    switch (plotType) {
      case 'ridiculous':
        return <AlertTriangle className="w-5 h-5 text-orange-400" />;
      case 'intent':
        return <Target className="w-5 h-5 text-blue-400" />;
      case 'reasoning':
        return <Brain className="w-5 h-5 text-purple-400" />;
      default:
        return <BarChart3 className="w-5 h-5 text-gray-400" />;
    }
  };

  const getPlotTypeLabel = (plotType: string) => {
    switch (plotType) {
      case 'ridiculous':
        return 'Pugachev-Cobra Boundary Test';
      case 'intent':
        return 'Intent Recognition Analysis';
      case 'reasoning':
        return 'Reasoning Validation';
      default:
        return 'Unknown Plot';
    }
  };

  const getPlotTypeDescription = (plotType: string) => {
    switch (plotType) {
      case 'ridiculous':
        return 'Tests solution space boundaries through ridiculous scenario generation';
      case 'intent':
        return '12-dimensional environmental analysis of user intent';
      case 'reasoning':
        return 'Visual validation of AI understanding and reasoning coherence';
      default:
        return 'Data visualization and analysis';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  const renderMetadata = (metadata: Record<string, any>) => {
    const importantKeys = [
      'ai_reasoning',
      'boundary_established',
      'environmental_integration',
      'pipeline_stage',
      'alternative_intents',
      'understanding_validated',
      'patterns_identified',
      'environmental_coherence'
    ];

    return (
      <div className="mt-4 space-y-2">
        {importantKeys.map((key) => {
          const value = metadata[key];
          if (value === undefined || value === null) return null;

          return (
            <div key={key} className="flex items-center justify-between text-sm">
              <span className="text-gray-300 capitalize">
                {key.replace(/_/g, ' ')}:
              </span>
              <span className="text-white ml-2">
                {typeof value === 'boolean' ? (
                  <span className={`flex items-center ${value ? 'text-green-400' : 'text-red-400'}`}>
                    {value ? <CheckCircle className="w-3 h-3 mr-1" /> : <AlertTriangle className="w-3 h-3 mr-1" />}
                    {value ? 'Yes' : 'No'}
                  </span>
                ) : Array.isArray(value) ? (
                  <span className="text-blue-300">{value.length} items</span>
                ) : (
                  <span className="text-white">{String(value)}</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="plot-container animate-pulse">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 bg-gray-600 rounded animate-pulse"></div>
            <div className="w-32 h-4 bg-gray-600 rounded animate-pulse"></div>
          </div>
          <div className="w-16 h-4 bg-gray-600 rounded animate-pulse"></div>
        </div>
        <div className="w-full h-64 bg-gray-700/50 rounded-lg flex items-center justify-center">
          <div className="flex items-center space-x-2 text-gray-400">
            <Clock className="w-5 h-5 animate-spin" />
            <span>Generating {getPlotTypeLabel(type)}...</span>
          </div>
        </div>
        <div className="mt-4 space-y-2">
          <div className="w-full h-3 bg-gray-600 rounded animate-pulse"></div>
          <div className="w-3/4 h-3 bg-gray-600 rounded animate-pulse"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="plot-container group">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          {getPlotIcon(type)}
          <div>
            <h3 className="text-white font-semibold text-lg">
              {plotData.title || getPlotTypeLabel(type)}
            </h3>
            <p className="text-gray-300 text-sm">
              {getPlotTypeDescription(type)}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">Confidence:</span>
          <span className={`font-bold ${getConfidenceColor(plotData.confidence)}`}>
            {(plotData.confidence * 100).toFixed(1)}%
          </span>
          <span className={`text-xs px-2 py-1 rounded-full ${
            plotData.confidence >= 0.8 ? 'bg-green-900/30 text-green-300' :
            plotData.confidence >= 0.6 ? 'bg-yellow-900/30 text-yellow-300' :
            'bg-red-900/30 text-red-300'
          }`}>
            {getConfidenceLabel(plotData.confidence)}
          </span>
        </div>
      </div>

      {/* Plot SVG */}
      <div className="relative mb-4">
        <div 
          className="plot-svg w-full min-h-[300px] max-h-[500px] overflow-auto rounded-lg border border-white/10 bg-gradient-to-br from-gray-900/50 to-gray-800/50 p-4"
          dangerouslySetInnerHTML={{ __html: plotData.svg_content }}
        />
        
        {/* Overlay for empty plots */}
        {plotData.svg_content.includes('No plot generated') && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 rounded-lg">
            <div className="text-center text-gray-400">
              <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>Plot generation in progress...</p>
              <p className="text-xs mt-1">Bayesian network is processing</p>
            </div>
          </div>
        )}
      </div>

      {/* Description */}
      <div className="mb-4">
        <p className="text-gray-200 leading-relaxed">
          {plotData.description}
        </p>
      </div>

      {/* Metadata */}
      {plotData.metadata && Object.keys(plotData.metadata).length > 0 && (
        <div className="border-t border-white/10 pt-4">
          <h4 className="text-white font-medium mb-2 flex items-center">
            <Brain className="w-4 h-4 mr-2 text-blue-400" />
            Analysis Details
          </h4>
          {renderMetadata(plotData.metadata)}
        </div>
      )}

      {/* Hover effect indicator */}
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
        <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
      </div>
    </div>
  );
};

export default PlotDisplay;
