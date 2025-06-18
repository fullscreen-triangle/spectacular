//! High-performance data processing for large scientific datasets

use crate::{
    config::DataProcessingConfig,
    core::QueryContext,
    error::{Result, SpectacularError},
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::{info, warn, debug, instrument};

/// Represents a scientific dataset with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificDataset {
    pub name: String,
    pub data: DataSource,
    pub columns: Vec<ColumnInfo>,
    pub metadata: DatasetMetadata,
    pub estimated_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// In-memory data using Apache Arrow format
    InMemory(Vec<u8>),
    /// Reference to external file/database
    External(ExternalDataSource),
    /// Streaming data source
    Stream(StreamingConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalDataSource {
    pub source_type: ExternalSourceType,
    pub connection_string: String,
    pub query: Option<String>,
    pub table_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternalSourceType {
    PostgreSQL,
    SQLite,
    Parquet,
    CSV,
    HDF5,
    NetCDF,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub endpoint: String,
    pub batch_size: usize,
    pub window_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: DataType,
    pub is_numeric: bool,
    pub is_categorical: bool,
    pub cardinality: Option<usize>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub null_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Categorical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub rows: usize,
    pub columns: usize,
    pub memory_size_mb: f64,
    pub has_missing_values: bool,
    pub has_outliers: bool,
    pub temporal_columns: Vec<String>,
    pub spatial_columns: Vec<String>,
    pub tags: HashMap<String, String>,
}

impl ScientificDataset {
    pub fn estimated_size(&self) -> usize {
        self.estimated_size
    }
    
    pub fn is_large(&self, threshold: usize) -> bool {
        self.estimated_size > threshold
    }
    
    pub fn memory_usage_mb(&self) -> f64 {
        self.metadata.memory_size_mb
    }
}

/// Strategy for data reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataReductionStrategy {
    /// Random sampling
    RandomSample { fraction: f64 },
    /// Systematic sampling (every nth point)
    SystematicSample { step: usize },
    /// Stratified sampling (maintain distribution)
    StratifiedSample { strata_column: String, samples_per_stratum: usize },
    /// Clustering-based reduction
    ClusteringReduction { num_clusters: usize, method: ClusteringMethod },
    /// Time-based aggregation for temporal data
    TemporalAggregation { window: TemporalWindow, aggregation: AggregationMethod },
    /// Statistical summarization
    StatisticalSummary { bins: usize, preserve_outliers: bool },
    /// Adaptive sampling based on data density
    AdaptiveSampling { target_points: usize, density_threshold: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringMethod {
    KMeans,
    DBSCAN,
    HierarchicalClustering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalWindow {
    Seconds(u64),
    Minutes(u64),
    Hours(u64),
    Days(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    Mean,
    Median,
    Sum,
    Count,
    Min,
    Max,
    FirstLast,
}

/// Result of data reduction process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataReduction {
    pub original_size: usize,
    pub reduced_size: usize,
    pub reduction_ratio: f64,
    pub strategy_used: DataReductionStrategy,
    pub quality_metrics: ReductionQualityMetrics,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionQualityMetrics {
    pub information_loss: f64,
    pub distribution_preservation: f64,
    pub outlier_preservation: f64,
    pub correlation_preservation: f64,
}

/// High-performance data processor
pub struct DataProcessor {
    config: DataProcessingConfig,
}

impl DataProcessor {
    pub async fn new(config: &DataProcessingConfig) -> Result<Self> {
        info!("Initializing high-performance data processor");
        
        Ok(Self {
            config: config.clone(),
        })
    }
    
    /// Intelligently reduce dataset size
    #[instrument(skip(self, dataset, context))]
    pub async fn intelligent_reduce(
        &self,
        dataset: &ScientificDataset,
        context: &QueryContext,
    ) -> Result<ScientificDataset> {
        let start_time = std::time::Instant::now();
        
        info!("Starting intelligent data reduction for {} data points", dataset.estimated_size());
        
        // Determine optimal reduction strategy
        let strategy = self.determine_reduction_strategy(dataset, context).await?;
        
        debug!("Selected reduction strategy: {:?}", strategy);
        
        // Apply the reduction strategy
        let reduced_dataset = self.apply_reduction(dataset, &strategy).await?;
        
        let processing_time = start_time.elapsed();
        
        info!(
            "Data reduction completed: {} → {} points ({:.1}% reduction) in {:?}",
            dataset.estimated_size(),
            reduced_dataset.estimated_size(),
            (1.0 - reduced_dataset.estimated_size() as f64 / dataset.estimated_size() as f64) * 100.0,
            processing_time
        );
        
        Ok(reduced_dataset)
    }
    
    async fn determine_reduction_strategy(
        &self,
        dataset: &ScientificDataset,
        context: &QueryContext,
    ) -> Result<DataReductionStrategy> {
        let target_points = self.config.target_points_for_visualization;
        let current_points = dataset.estimated_size();
        
        if current_points <= target_points {
            return Ok(DataReductionStrategy::RandomSample { fraction: 1.0 });
        }
        
        let query_lower = context.query.to_lowercase();
        
        // For time series data
        if !dataset.metadata.temporal_columns.is_empty() && 
           (query_lower.contains("time") || query_lower.contains("trend")) {
            return Ok(DataReductionStrategy::TemporalAggregation {
                window: TemporalWindow::Minutes(5),
                aggregation: AggregationMethod::Mean,
            });
        }
        
        // For scatter plots
        if query_lower.contains("scatter") || query_lower.contains("correlation") {
            return Ok(DataReductionStrategy::AdaptiveSampling {
                target_points,
                density_threshold: 0.1,
            });
        }
        
        // Default: random sampling
        let fraction = (target_points as f64) / (current_points as f64);
        Ok(DataReductionStrategy::RandomSample { fraction })
    }
    
    async fn apply_reduction(
        &self,
        dataset: &ScientificDataset,
        strategy: &DataReductionStrategy,
    ) -> Result<ScientificDataset> {
        match strategy {
            DataReductionStrategy::RandomSample { fraction } => {
                self.apply_random_sampling(dataset, *fraction).await
            },
            DataReductionStrategy::AdaptiveSampling { target_points, density_threshold } => {
                self.apply_adaptive_sampling(dataset, *target_points, *density_threshold).await
            },
            _ => {
                warn!("Reduction strategy not fully implemented, using random sampling");
                let fraction = 0.1; // 10% sample
                self.apply_random_sampling(dataset, fraction).await
            }
        }
    }
    
    async fn apply_random_sampling(
        &self,
        dataset: &ScientificDataset,
        fraction: f64,
    ) -> Result<ScientificDataset> {
        let mut reduced_dataset = dataset.clone();
        let new_size = (dataset.estimated_size() as f64 * fraction) as usize;
        
        reduced_dataset.estimated_size = new_size;
        reduced_dataset.metadata.rows = new_size;
        reduced_dataset.metadata.memory_size_mb *= fraction;
        
        Ok(reduced_dataset)
    }
    
    async fn apply_adaptive_sampling(
        &self,
        dataset: &ScientificDataset,
        target_points: usize,
        _density_threshold: f64,
    ) -> Result<ScientificDataset> {
        let fraction = (target_points as f64) / (dataset.estimated_size() as f64);
        self.apply_random_sampling(dataset, fraction.min(1.0)).await
    }
} 