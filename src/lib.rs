//! # Spectacular: High-Performance Scientific Visualization Engine
//!
//! A Rust-based system for generating optimized D3.js visualizations from large scientific datasets.
//! Features hybrid logical programming with fuzzy logic, intelligent data reduction, and automated
//! JavaScript debugging capabilities.

pub mod core;
pub mod data;
pub mod pretoria;
pub mod hf_integration;
pub mod orchestrator_client;
pub mod js_engine;
pub mod config;
pub mod error;

// Chigutiro: High-performance crossfiltering system
pub mod chigutiro;

// Re-export main types
pub use crate::core::{SpectacularEngine, QueryContext, VisualizationResult};
pub use crate::data::{DataProcessor, DataReduction, ScientificDataset};
pub use crate::pretoria::{PretoriaEngine, LogicalProgram, FuzzyRule};
pub use crate::hf_integration::{HuggingFaceClient, CodeGeneration, DebugResult};
pub use crate::orchestrator_client::MetacognitiveClient;
pub use crate::js_engine::{JavaScriptDebugger, D3CodeValidator};
pub use crate::config::SpectacularConfig;
pub use crate::error::{SpectacularError, Result};

// Re-export Chigutiro types
pub use crate::chigutiro::{
    Chigutiro, ChigutiroConfig, Record, JsonRecord, 
    PerformanceMetrics, ChigutiroError, ChigutiroResult
};

use tracing::{info, warn};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the Spectacular system with tracing
pub fn init_tracing() -> anyhow::Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "spectacular=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    info!("Spectacular {} initialized", VERSION);
    Ok(())
}

/// Main entry point for the Spectacular system
#[derive(Clone)]
pub struct Spectacular {
    engine: Arc<RwLock<SpectacularEngine>>,
    config: SpectacularConfig,
}

impl Spectacular {
    /// Create a new Spectacular instance
    pub async fn new(config: SpectacularConfig) -> Result<Self> {
        let engine = SpectacularEngine::new(config.clone()).await?;
        
        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
            config,
        })
    }
    
    /// Process a visualization query with large dataset optimization
    pub async fn generate_visualization(
        &self,
        query: &str,
        dataset: Option<ScientificDataset>,
    ) -> Result<VisualizationResult> {
        let engine = self.engine.read().await;
        let context = QueryContext::new(query, dataset);
        
        engine.process_query(context).await
    }
    
    /// Get system health and performance metrics
    pub async fn health_check(&self) -> Result<SystemHealth> {
        let engine = self.engine.read().await;
        Ok(engine.get_health().await?)
    }
}

/// System health information
#[derive(Debug, serde::Serialize)]
pub struct SystemHealth {
    pub status: String,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub active_queries: usize,
    pub cache_hit_rate: f64,
    pub hf_models_loaded: usize,
    pub pretoria_rules_active: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_initialization() {
        let config = SpectacularConfig::default();
        let spectacular = Spectacular::new(config).await;
        assert!(spectacular.is_ok());
    }
} 