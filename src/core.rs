//! Core engine for the Spectacular scientific visualization system

use crate::{
    config::SpectacularConfig,
    data::{DataProcessor, ScientificDataset, DataReduction},
    pretoria::PretoriaEngine,
    hf_integration::HuggingFaceClient,
    orchestrator_client::MetacognitiveClient,
    js_engine::JavaScriptDebugger,
    error::{Result, SpectacularError},
    SystemHealth,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn, error, debug, instrument};
use uuid::Uuid;
use dashmap::DashMap;
use tokio::sync::Mutex;
use std::sync::Arc;

/// Main query context containing all information needed for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub id: Uuid,
    pub query: String,
    pub dataset: Option<ScientificDataset>,
    pub complexity_estimate: f64,
    pub visual_complexity: f64,
    pub data_size: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user_preferences: HashMap<String, serde_json::Value>,
}

impl QueryContext {
    pub fn new(query: &str, dataset: Option<ScientificDataset>) -> Self {
        let data_size = dataset.as_ref().map(|d| d.estimated_size()).unwrap_or(0);
        
        Self {
            id: Uuid::new_v4(),
            query: query.to_string(),
            dataset,
            complexity_estimate: Self::estimate_complexity(query),
            visual_complexity: Self::estimate_visual_complexity(query),
            data_size,
            timestamp: chrono::Utc::now(),
            user_preferences: HashMap::new(),
        }
    }
    
    fn estimate_complexity(query: &str) -> f64 {
        let complexity_indicators = [
            "interactive", "dynamic", "real-time", "complex", "multi-dimensional",
            "hierarchical", "network", "animated", "drill-down", "filtering"
        ];
        
        let matches = complexity_indicators.iter()
            .filter(|&word| query.to_lowercase().contains(word))
            .count() as f64;
        
        (matches / complexity_indicators.len() as f64).min(1.0)
    }
    
    fn estimate_visual_complexity(query: &str) -> f64 {
        let visual_indicators = [
            "color", "gradient", "heatmap", "overlay", "multiple charts",
            "dashboard", "coordinated", "linked", "brush", "zoom"
        ];
        
        let matches = visual_indicators.iter()
            .filter(|&word| query.to_lowercase().contains(word))
            .count() as f64;
        
        (matches / visual_indicators.len() as f64).min(1.0)
    }
}

/// Result of a visualization generation process
#[derive(Debug, Serialize, Deserialize)]
pub struct VisualizationResult {
    pub query_id: Uuid,
    pub success: bool,
    pub d3_code: Option<String>,
    pub typescript_code: Option<String>,
    pub html_template: Option<String>,
    pub data_reduction_applied: bool,
    pub reduced_data_points: usize,
    pub original_data_points: usize,
    pub pretoria_rules_used: Vec<String>,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
    pub debug_info: Option<DebugInfo>,
    pub performance_metrics: PerformanceMetrics,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DebugInfo {
    pub syntax_errors: Vec<String>,
    pub runtime_warnings: Vec<String>,
    pub optimization_suggestions: Vec<String>,
    pub accessibility_issues: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub data_processing_ms: u64,
    pub fuzzy_logic_ms: u64,
    pub code_generation_ms: u64,
    pub validation_ms: u64,
    pub memory_peak_mb: f64,
    pub cpu_utilization: f64,
}

/// Core engine that orchestrates all Spectacular components
pub struct SpectacularEngine {
    config: SpectacularConfig,
    data_processor: DataProcessor,
    pretoria_engine: PretoriaEngine,
    hf_client: HuggingFaceClient,
    metacognitive_client: MetacognitiveClient,
    js_debugger: JavaScriptDebugger,
    
    // Performance tracking
    start_time: Instant,
    query_cache: Arc<DashMap<String, VisualizationResult>>,
    active_queries: Arc<Mutex<HashMap<Uuid, QueryContext>>>,
    
    // Statistics
    total_queries: std::sync::atomic::AtomicU64,
    successful_queries: std::sync::atomic::AtomicU64,
}

impl SpectacularEngine {
    /// Create a new SpectacularEngine instance
    #[instrument(skip(config))]
    pub async fn new(config: SpectacularConfig) -> Result<Self> {
        info!("Initializing Spectacular Engine");
        
        // Initialize all components
        let data_processor = DataProcessor::new(&config.data_processing).await?;
        let pretoria_engine = PretoriaEngine::new(&config.pretoria).await?;
        let hf_client = HuggingFaceClient::new(&config.huggingface).await?;
        let metacognitive_client = MetacognitiveClient::new(&config.orchestrator).await?;
        let js_debugger = JavaScriptDebugger::new(&config.javascript).await?;
        
        info!("All components initialized successfully");
        
        Ok(Self {
            config,
            data_processor,
            pretoria_engine,
            hf_client,
            metacognitive_client,
            js_debugger,
            start_time: Instant::now(),
            query_cache: Arc::new(DashMap::new()),
            active_queries: Arc::new(Mutex::new(HashMap::new())),
            total_queries: std::sync::atomic::AtomicU64::new(0),
            successful_queries: std::sync::atomic::AtomicU64::new(0),
        })
    }
    
    /// Process a visualization query end-to-end
    #[instrument(skip(self), fields(query_id = %context.id))]
    pub async fn process_query(&self, mut context: QueryContext) -> Result<VisualizationResult> {
        let start_time = Instant::now();
        
        // Track active query
        {
            let mut active = self.active_queries.lock().await;
            active.insert(context.id, context.clone());
        }
        
        // Check cache first
        let cache_key = format!("{}_{}", context.query, context.data_size);
        if let Some(cached_result) = self.query_cache.get(&cache_key) {
            info!("Cache hit for query {}", context.id);
            return Ok(cached_result.clone());
        }
        
        info!("Processing new visualization query: {}", &context.query[..50.min(context.query.len())]);
        
        let result = self.process_query_internal(context.clone()).await;
        
        // Update statistics
        self.total_queries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if result.is_ok() {
            self.successful_queries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        
        // Remove from active queries
        {
            let mut active = self.active_queries.lock().await;
            active.remove(&context.id);
        }
        
        // Cache successful results
        if let Ok(ref success_result) = result {
            if success_result.success {
                self.query_cache.insert(cache_key, success_result.clone());
            }
        }
        
        let processing_time = start_time.elapsed();
        info!("Query {} completed in {:?}", context.id, processing_time);
        
        result
    }
    
    async fn process_query_internal(&self, mut context: QueryContext) -> Result<VisualizationResult> {
        let mut performance_metrics = PerformanceMetrics {
            data_processing_ms: 0,
            fuzzy_logic_ms: 0,
            code_generation_ms: 0,
            validation_ms: 0,
            memory_peak_mb: 0.0,
            cpu_utilization: 0.0,
        };
        
        // Step 1: Data Processing and Reduction (if needed)
        let data_start = Instant::now();
        let (reduced_dataset, data_reduction_applied) = if let Some(ref dataset) = context.dataset {
            if dataset.estimated_size() > self.config.data_processing.max_points_threshold {
                info!("Large dataset detected ({} points), applying intelligent reduction", dataset.estimated_size());
                let reduced = self.data_processor.intelligent_reduce(dataset, &context).await?;
                (Some(reduced), true)
            } else {
                (context.dataset.clone(), false)
            }
        } else {
            (None, false)
        };
        
        context.dataset = reduced_dataset.clone();
        performance_metrics.data_processing_ms = data_start.elapsed().as_millis() as u64;
        
        // Step 2: Pretoria Fuzzy Logic Analysis
        let fuzzy_start = Instant::now();
        let pretoria_analysis = self.pretoria_engine.analyze_query(&context).await?;
        performance_metrics.fuzzy_logic_ms = fuzzy_start.elapsed().as_millis() as u64;
        
        // Step 3: Interface with External Metacognitive Orchestrator
        let orchestrator_response = self.metacognitive_client
            .request_visualization_strategy(&context, &pretoria_analysis)
            .await?;
        
        // Step 4: Code Generation via HuggingFace Models
        let codegen_start = Instant::now();
        let generated_code = self.hf_client.generate_d3_code(
            &context,
            &pretoria_analysis,
            &orchestrator_response,
        ).await?;
        performance_metrics.code_generation_ms = codegen_start.elapsed().as_millis() as u64;
        
        // Step 5: JavaScript Validation and Debugging
        let validation_start = Instant::now();
        let debug_result = self.js_debugger.validate_and_debug(&generated_code).await?;
        performance_metrics.validation_ms = validation_start.elapsed().as_millis() as u64;
        
        // Construct final result
        let original_points = context.dataset.as_ref()
            .map(|d| if data_reduction_applied { d.estimated_size() * 10 } else { d.estimated_size() })
            .unwrap_or(0);
        
        let reduced_points = context.dataset.as_ref()
            .map(|d| d.estimated_size())
            .unwrap_or(0);
        
        let result = VisualizationResult {
            query_id: context.id,
            success: debug_result.is_valid,
            d3_code: Some(debug_result.optimized_code),
            typescript_code: generated_code.typescript_wrapper,
            html_template: Some(self.generate_html_template(&debug_result.optimized_code)),
            data_reduction_applied,
            reduced_data_points: reduced_points,
            original_data_points: original_points,
            pretoria_rules_used: pretoria_analysis.applied_rules,
            confidence_score: self.calculate_confidence_score(&pretoria_analysis, &generated_code, &debug_result),
            processing_time_ms: data_start.elapsed().as_millis() as u64,
            debug_info: Some(DebugInfo {
                syntax_errors: debug_result.syntax_errors,
                runtime_warnings: debug_result.warnings,
                optimization_suggestions: debug_result.optimizations,
                accessibility_issues: debug_result.accessibility_issues,
            }),
            performance_metrics,
            error: if debug_result.is_valid { None } else { Some("Code validation failed".to_string()) },
        };
        
        Ok(result)
    }
    
    fn calculate_confidence_score(
        &self,
        pretoria_analysis: &crate::pretoria::PretoriaAnalysis,
        generated_code: &crate::hf_integration::GeneratedCode,
        debug_result: &crate::js_engine::DebugResult,
    ) -> f64 {
        let fuzzy_confidence = pretoria_analysis.confidence;
        let model_confidence = generated_code.confidence;
        let validation_score = if debug_result.is_valid { 1.0 } else { 0.0 };
        
        // Weighted average
        (fuzzy_confidence * 0.3 + model_confidence * 0.4 + validation_score * 0.3).min(1.0)
    }
    
    fn generate_html_template(&self, d3_code: &str) -> String {
        format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spectacular Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .visualization {{ border: 1px solid #ddd; padding: 20px; }}
        .loading {{ color: #666; font-style: italic; }}
    </style>
</head>
<body>
    <div id="visualization" class="visualization">
        <div class="loading">Loading visualization...</div>
    </div>
    
    <script>
        // Generated by Spectacular v{}
        // Timestamp: {}
        
        {}
    </script>
</body>
</html>"#, 
            crate::VERSION,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            d3_code
        )
    }
    
    /// Get system health information
    pub async fn get_health(&self) -> Result<SystemHealth> {
        let uptime = self.start_time.elapsed().as_secs();
        let active_queries = self.active_queries.lock().await.len();
        let total = self.total_queries.load(std::sync::atomic::Ordering::Relaxed);
        let successful = self.successful_queries.load(std::sync::atomic::Ordering::Relaxed);
        let cache_hit_rate = if total > 0 { 
            (self.query_cache.len() as f64 / total as f64) * 100.0 
        } else { 
            0.0 
        };
        
        // Get memory usage (simplified)
        let memory_usage = self.estimate_memory_usage().await?;
        
        Ok(SystemHealth {
            status: "healthy".to_string(),
            uptime_seconds: uptime,
            memory_usage_mb: memory_usage,
            active_queries,
            cache_hit_rate,
            hf_models_loaded: self.hf_client.loaded_models_count().await?,
            pretoria_rules_active: self.pretoria_engine.active_rules_count().await?,
        })
    }
    
    async fn estimate_memory_usage(&self) -> Result<f64> {
        // Simplified memory estimation
        // In a real implementation, you'd use system APIs
        let cache_size = self.query_cache.len() as f64 * 1.5; // Rough estimate in MB
        Ok(cache_size + 50.0) // Base system memory
    }
} 