use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

use crate::chigutiro::{ChigutiroResult, ChigutiroError, PerformanceMetrics};
use super::dimension::{Dimension, DimensionManager};
use super::filter::FilterManager;
use super::coordinator::ViewCoordinator;

/// Unique identifier for records in the crossfilter
pub type RecordId = u64;

/// Trait for data records that can be crossfiltered
pub trait Record: Send + Sync + Clone {
    /// Get a unique identifier for this record
    fn get_id(&self) -> RecordId;
    
    /// Get a field value by name (for dynamic field access)
    fn get_field(&self, field_name: &str) -> Option<serde_json::Value>;
    
    /// Get all field names available in this record
    fn field_names(&self) -> Vec<String>;
}

/// Default implementation for JSON-like records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRecord {
    pub id: RecordId,
    pub data: HashMap<String, serde_json::Value>,
}

impl Record for JsonRecord {
    fn get_id(&self) -> RecordId {
        self.id
    }
    
    fn get_field(&self, field_name: &str) -> Option<serde_json::Value> {
        self.data.get(field_name).cloned()
    }
    
    fn field_names(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }
}

/// Configuration for the Chigutiro system
#[derive(Debug, Clone)]
pub struct ChigutiroConfig {
    /// Maximum number of records to keep in memory
    pub max_records: usize,
    
    /// Enable probabilistic filtering for large datasets
    pub enable_probabilistic: bool,
    
    /// False positive rate for probabilistic structures (0.01 = 1%)
    pub false_positive_rate: f64,
    
    /// Number of threads to use for parallel operations
    pub num_threads: usize,
    
    /// Cache size for frequently accessed data
    pub cache_size_mb: usize,
    
    /// Enable adaptive algorithm selection
    pub adaptive_algorithms: bool,
    
    /// Minimum records before switching to probabilistic mode
    pub probabilistic_threshold: usize,
    
    /// Enable real-time metrics collection
    pub enable_metrics: bool,
}

impl Default for ChigutiroConfig {
    fn default() -> Self {
        Self {
            max_records: 10_000_000, // 10M records
            enable_probabilistic: true,
            false_positive_rate: 0.01, // 1% false positive rate
            num_threads: num_cpus::get(),
            cache_size_mb: 512, // 512MB cache
            adaptive_algorithms: true,
            probabilistic_threshold: 100_000, // Switch to probabilistic at 100K records
            enable_metrics: true,
        }
    }
}

/// Main Chigutiro crossfiltering system
pub struct Chigutiro<R: Record> {
    /// System configuration
    config: ChigutiroConfig,
    
    /// All records in the system
    records: Arc<RwLock<Vec<R>>>,
    
    /// Dimension manager for creating and managing dimensions
    dimension_manager: Arc<RwLock<DimensionManager<R>>>,
    
    /// Filter manager for coordinating filters across dimensions
    filter_manager: Arc<RwLock<FilterManager>>,
    
    /// View coordinator for managing chart updates
    view_coordinator: Arc<RwLock<ViewCoordinator>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// Next record ID to assign
    next_record_id: Arc<RwLock<RecordId>>,
}

impl<R: Record + 'static> Chigutiro<R> {
    /// Create a new Chigutiro instance
    pub fn new(config: ChigutiroConfig) -> Self {
        // Configure Rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global()
            .unwrap_or_else(|_| {}); // Ignore if already configured
        
        Self {
            config,
            records: Arc::new(RwLock::new(Vec::new())),
            dimension_manager: Arc::new(RwLock::new(DimensionManager::new())),
            filter_manager: Arc::new(RwLock::new(FilterManager::new())),
            view_coordinator: Arc::new(RwLock::new(ViewCoordinator::new())),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            next_record_id: Arc::new(RwLock::new(1)),
        }
    }
    
    /// Add a single record to the crossfilter
    pub fn add_record(&self, record: R) -> ChigutiroResult<RecordId> {
        let start_time = Instant::now();
        
        let record_id = record.get_id();
        
        // Add to records
        {
            let mut records = self.records.write().unwrap();
            records.push(record.clone());
        }
        
        // Update dimensions
        {
            let mut dim_manager = self.dimension_manager.write().unwrap();
            dim_manager.add_record(record)?;
        }
        
        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_records += 1;
            metrics.last_filter_time_us = start_time.elapsed().as_micros() as u64;
        }
        
        Ok(record_id)
    }
    
    /// Add multiple records in batch (more efficient)
    pub fn add_records(&self, records: Vec<R>) -> ChigutiroResult<Vec<RecordId>> {
        let start_time = Instant::now();
        
        let record_ids: Vec<RecordId> = records.iter().map(|r| r.get_id()).collect();
        
        // Add to records in batch
        {
            let mut record_store = self.records.write().unwrap();
            record_store.extend(records.clone());
        }
        
        // Update dimensions in parallel
        {
            let mut dim_manager = self.dimension_manager.write().unwrap();
            dim_manager.add_records_batch(records)?;
        }
        
        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_records += record_ids.len();
            metrics.last_filter_time_us = start_time.elapsed().as_micros() as u64;
        }
        
        Ok(record_ids)
    }
    
    /// Create a new dimension for filtering
    pub fn dimension<F>(&self, name: &str, accessor: F) -> ChigutiroResult<Arc<Dimension<R>>>
    where
        F: Fn(&R) -> serde_json::Value + Send + Sync + 'static,
    {
        let mut dim_manager = self.dimension_manager.write().unwrap();
        dim_manager.create_dimension(name, accessor, &self.config)
    }
    
    /// Get all records currently matching active filters
    pub fn all_filtered(&self) -> ChigutiroResult<Vec<R>> {
        let start_time = Instant::now();
        
        let records = self.records.read().unwrap();
        let filter_manager = self.filter_manager.read().unwrap();
        
        let filtered_records = if filter_manager.has_active_filters() {
            // Apply all active filters
            records
                .par_iter()
                .filter(|record| filter_manager.matches_all_filters(record))
                .cloned()
                .collect()
        } else {
            // No filters active, return all records
            records.clone()
        };
        
        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.filtered_records = filtered_records.len();
            metrics.last_filter_time_us = start_time.elapsed().as_micros() as u64;
        }
        
        Ok(filtered_records)
    }
    
    /// Get the total number of records matching current filters
    pub fn size(&self) -> usize {
        let filter_manager = self.filter_manager.read().unwrap();
        
        if filter_manager.has_active_filters() {
            let records = self.records.read().unwrap();
            records
                .par_iter()
                .filter(|record| filter_manager.matches_all_filters(record))
                .count()
        } else {
            let records = self.records.read().unwrap();
            records.len()
        }
    }
    
    /// Get current performance metrics
    pub fn metrics(&self) -> PerformanceMetrics {
        self.metrics.read().unwrap().clone()
    }
    
    /// Clear all records and reset the system
    pub fn clear(&self) -> ChigutiroResult<()> {
        {
            let mut records = self.records.write().unwrap();
            records.clear();
        }
        
        {
            let mut dim_manager = self.dimension_manager.write().unwrap();
            dim_manager.clear();
        }
        
        {
            let mut filter_manager = self.filter_manager.write().unwrap();
            filter_manager.clear_all_filters();
        }
        
        // Reset metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            *metrics = PerformanceMetrics::default();
        }
        
        Ok(())
    }
    
    /// Optimize the system for better performance
    pub fn optimize(&self) -> ChigutiroResult<()> {
        let start_time = Instant::now();
        
        // Optimize dimensions
        {
            let mut dim_manager = self.dimension_manager.write().unwrap();
            dim_manager.optimize_all()?;
        }
        
        // Consider switching to probabilistic mode if we have many records
        if self.config.adaptive_algorithms {
            let total_records = {
                let records = self.records.read().unwrap();
                records.len()
            };
            
            if total_records >= self.config.probabilistic_threshold {
                // Enable probabilistic filtering for large datasets
                log::info!("Switching to probabilistic mode for {} records", total_records);
                // This would trigger internal optimizations
            }
        }
        
        log::info!("Optimization completed in {:?}", start_time.elapsed());
        Ok(())
    }
    
    /// Get system statistics
    pub fn stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let records = self.records.read().unwrap();
        stats.insert("total_records".to_string(), serde_json::json!(records.len()));
        
        let dim_manager = self.dimension_manager.read().unwrap();
        stats.insert("dimensions".to_string(), serde_json::json!(dim_manager.dimension_count()));
        
        let filter_manager = self.filter_manager.read().unwrap();
        stats.insert("active_filters".to_string(), serde_json::json!(filter_manager.active_filter_count()));
        
        if self.config.enable_metrics {
            let metrics = self.metrics.read().unwrap();
            stats.insert("metrics".to_string(), serde_json::json!(*metrics));
        }
        
        stats.insert("config".to_string(), serde_json::json!({
            "max_records": self.config.max_records,
            "probabilistic_enabled": self.config.enable_probabilistic,
            "threads": self.config.num_threads,
            "cache_size_mb": self.config.cache_size_mb,
        }));
        
        stats
    }
} 