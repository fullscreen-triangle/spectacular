//! Configuration structures for Spectacular components

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration for Spectacular system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectacularConfig {
    pub data_processing: DataProcessingConfig,
    pub pretoria: PretoriaConfig,
    pub huggingface: HuggingFaceConfig,
    pub orchestrator: OrchestratorConfig,
    pub javascript: JavaScriptConfig,
    pub system: SystemConfig,
}

impl Default for SpectacularConfig {
    fn default() -> Self {
        Self {
            data_processing: DataProcessingConfig::default(),
            pretoria: PretoriaConfig::default(),
            huggingface: HuggingFaceConfig::default(),
            orchestrator: OrchestratorConfig::default(),
            javascript: JavaScriptConfig::default(),
            system: SystemConfig::default(),
        }
    }
}

/// Configuration for data processing and reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessingConfig {
    pub max_points_threshold: usize,
    pub target_points_for_visualization: usize,
    pub enable_streaming: bool,
    pub cache_size_mb: usize,
    pub parallel_processing: bool,
    pub compression_enabled: bool,
}

impl Default for DataProcessingConfig {
    fn default() -> Self {
        Self {
            max_points_threshold: 100_000,
            target_points_for_visualization: 10_000,
            enable_streaming: true,
            cache_size_mb: 1024,
            parallel_processing: true,
            compression_enabled: true,
        }
    }
}

/// Configuration for Pretoria fuzzy logic engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretoriaConfig {
    pub max_fuzzy_rules: usize,
    pub confidence_threshold: f64,
    pub enable_logical_programming: bool,
    pub rule_learning_enabled: bool,
    pub optimization_level: u8,
}

impl Default for PretoriaConfig {
    fn default() -> Self {
        Self {
            max_fuzzy_rules: 1000,
            confidence_threshold: 0.7,
            enable_logical_programming: true,
            rule_learning_enabled: false,
            optimization_level: 2,
        }
    }
}

/// Configuration for HuggingFace integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    pub api_key: Option<String>,
    pub api_base_url: String,
    pub model_cache_dir: String,
    pub max_concurrent_requests: usize,
    pub timeout_seconds: u64,
    pub retry_attempts: usize,
    pub preferred_models: Vec<String>,
    pub enable_local_models: bool,
    pub gpu_enabled: bool,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_base_url: "https://api-inference.huggingface.co/models/".to_string(),
            model_cache_dir: "./models".to_string(),
            max_concurrent_requests: 5,
            timeout_seconds: 30,
            retry_attempts: 3,
            preferred_models: vec![
                "Salesforce/codet5-base".to_string(),
                "microsoft/DialoGPT-medium".to_string(),
            ],
            enable_local_models: false,
            gpu_enabled: false,
        }
    }
}

/// Configuration for external metacognitive orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub use_grpc: bool,
    pub grpc_port: u16,
    pub enable_streaming: bool,
}

/// Configuration for Autobahn metacognitive engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnConfig {
    pub endpoint: String,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub enable_streaming: bool,
    pub biological_authenticity_threshold: f64,
    pub quantum_coherence_reporting: bool,
    pub atp_budget_tracking: bool,
    pub consciousness_emergence_detection: bool,
    pub oscillation_sampling_rate: f64,
    pub membrane_state_precision: f64,
    pub atp_trajectory_resolution: f64,
    pub multi_scale_coupling: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8080".to_string(),
            api_key: None,
            timeout_seconds: 60,
            max_retries: 3,
            use_grpc: true,
            grpc_port: 50051,
            enable_streaming: false,
        }
    }
}

impl Default for AutobahnConfig {
    fn default() -> Self {
        Self {
            endpoint: "grpc://localhost:50051".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            enable_streaming: true,
            biological_authenticity_threshold: 0.85,
            quantum_coherence_reporting: true,
            atp_budget_tracking: true,
            consciousness_emergence_detection: true,
            oscillation_sampling_rate: 1000.0,
            membrane_state_precision: 0.001,
            atp_trajectory_resolution: 0.1,
            multi_scale_coupling: true,
        }
    }
}

/// Configuration for JavaScript debugging engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JavaScriptConfig {
    pub v8_heap_size_mb: usize,
    pub execution_timeout_ms: u64,
    pub enable_debugging: bool,
    pub enable_optimization: bool,
    pub strict_mode: bool,
    pub enable_typescript: bool,
    pub d3_version: String,
}

impl Default for JavaScriptConfig {
    fn default() -> Self {
        Self {
            v8_heap_size_mb: 512,
            execution_timeout_ms: 10000,
            enable_debugging: true,
            enable_optimization: true,
            strict_mode: true,
            enable_typescript: true,
            d3_version: "7.8.5".to_string(),
        }
    }
}

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub log_level: String,
    pub enable_metrics: bool,
    pub metrics_port: u16,
    pub health_check_port: u16,
    pub max_concurrent_queries: usize,
    pub query_timeout_seconds: u64,
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            enable_metrics: true,
            metrics_port: 9090,
            health_check_port: 8080,
            max_concurrent_queries: 100,
            query_timeout_seconds: 300,
            enable_caching: true,
            cache_ttl_seconds: 3600,
        }
    }
}

impl SpectacularConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        // HuggingFace API key
        if let Ok(api_key) = std::env::var("HUGGINGFACE_API_KEY") {
            config.huggingface.api_key = Some(api_key);
        }
        
        // Orchestrator endpoint
        if let Ok(endpoint) = std::env::var("ORCHESTRATOR_ENDPOINT") {
            config.orchestrator.endpoint = endpoint;
        }
        
        // Log level
        if let Ok(log_level) = std::env::var("LOG_LEVEL") {
            config.system.log_level = log_level;
        }
        
        // GPU enabling
        if let Ok(gpu_enabled) = std::env::var("ENABLE_GPU") {
            config.huggingface.gpu_enabled = gpu_enabled.to_lowercase() == "true";
        }
        
        config
    }
    
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: SpectacularConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.data_processing.max_points_threshold == 0 {
            return Err("max_points_threshold must be greater than 0".to_string());
        }
        
        if self.data_processing.target_points_for_visualization > self.data_processing.max_points_threshold {
            return Err("target_points_for_visualization cannot be greater than max_points_threshold".to_string());
        }
        
        if self.pretoria.confidence_threshold < 0.0 || self.pretoria.confidence_threshold > 1.0 {
            return Err("confidence_threshold must be between 0.0 and 1.0".to_string());
        }
        
        if self.system.max_concurrent_queries == 0 {
            return Err("max_concurrent_queries must be greater than 0".to_string());
        }
        
        Ok(())
    }
} 