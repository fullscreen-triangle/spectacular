// API Types for Spectacular Triple Validation Framework

export interface ChatRequest {
  message: string;
  context?: Record<string, any>;
  data?: any;
  conversation_id?: string;
}

export interface PlotData {
  svg_content: string;
  title: string;
  description: string;
  confidence: number;
  metadata: Record<string, any>;
}

export interface ValidationDetails {
  advanced_pipeline: boolean;
  pipeline_stages_completed: number;
  environmental_integration: boolean;
  sensor_data_collected: boolean;
  knowledge_items_synthesized: number;
  visual_embeddings_created: number;
  stage_timings: Record<string, number>;
  confidence_progression: number[];
  bayesian_intelligence?: boolean;
  dynamic_routing?: boolean;
  recursive_processing?: boolean;
  external_validation?: boolean;
  multi_dimensional_embeddings?: number;
  fuzzy_logic_nodes?: number;
  network_convergence?: boolean;
}

export interface ChatResponse {
  response_text: string;
  plots: {
    ridiculous: PlotData;
    intent: PlotData;
    reasoning: PlotData;
  };
  validation_passed: boolean;
  coherence_score: number;
  processing_time: number;
  conversation_id: string;
  timestamp: string;
  validation_details: ValidationDetails;
}

export interface SystemStatus {
  status: string;
  uptime: number;
  orchestrator_status: string;
  environmental_sensors: Record<string, any>;
  pipeline_components: Record<string, any>;
  recent_executions: number;
  average_processing_time: number;
  success_rate: number;
  bayesian_network_health?: {
    nodes_active: number;
    network_coherence: number;
    external_validators_available: number;
    embedding_paths_active: number;
  };
}

export interface EnvironmentalSnapshot {
  biometric_data: { measurement: number; confidence: number; timestamp: string };
  spatial_context: { measurement: number; confidence: number; timestamp: string };
  temporal_dynamics: { measurement: number; confidence: number; timestamp: string };
  quantum_correlations: { measurement: number; confidence: number; timestamp: string };
  atmospheric_conditions: { measurement: number; confidence: number; timestamp: string };
  electromagnetic_fields: { measurement: number; confidence: number; timestamp: string };
  thermal_patterns: { measurement: number; confidence: number; timestamp: string };
  acoustic_environment: { measurement: number; confidence: number; timestamp: string };
  luminosity_patterns: { measurement: number; confidence: number; timestamp: string };
  computational_load: { measurement: number; confidence: number; timestamp: string };
  network_coherence: { measurement: number; confidence: number; timestamp: string };
  cognitive_resonance: { measurement: number; confidence: number; timestamp: string };
}

export interface BayesianNetworkResults {
  nodes_converged: number;
  total_nodes: number;
  recursive_loops_executed: Record<string, number>;
  external_validations_passed: number;
  network_coherence: number;
  coherence_trajectory: number[];
  embedding_paths: Record<string, {
    dimensions: number[];
    coherence: number;
    environmental_stability: number;
    external_validations: number;
    similar_environments: string[];
  }>;
}

// UI State Types
export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  plots?: ChatResponse['plots'];
  validation_details?: ValidationDetails;
  bayesian_results?: BayesianNetworkResults;
}

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  currentConversationId: string | null;
  error: string | null;
}

// Component Props Types
export interface PlotDisplayProps {
  plotData: PlotData;
  type: 'ridiculous' | 'intent' | 'reasoning';
  isLoading?: boolean;
}

export interface SystemStatusProps {
  status: SystemStatus;
  isLoading?: boolean;
}

export interface ValidationDetailsProps {
  details: ValidationDetails;
  bayesianResults?: BayesianNetworkResults;
}

export interface EnvironmentalDataProps {
  snapshot: EnvironmentalSnapshot;
  showDetails?: boolean;
}
