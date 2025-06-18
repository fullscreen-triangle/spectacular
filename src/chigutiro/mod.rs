//! # Chigutiro: High-Performance Multidimensional Crossfiltering
//!
//! A Rust implementation of multidimensional filtering inspired by Crossfilter.js
//! but designed for scientific datasets with millions of records. Features:
//!
//! - **Probabilistic Filtering**: Uses Bloom filters and probabilistic data structures
//! - **Adaptive Algorithms**: Switches between filtering strategies based on data characteristics
//! - **Real-time Performance**: Sub-millisecond filtering for coordinated views
//! - **Memory Efficient**: Uses compressed indexes and bit manipulation
//! - **Isolated Testing**: Designed as a standalone module for easy benchmarking

pub mod core;
pub mod dimension;
pub mod group;
pub mod filter;
pub mod index;
pub mod algorithms;
pub mod coordinator;
pub mod chart_redraw;
pub mod benchmark;

// Re-export main types for convenience
pub use core::{Chigutiro, ChigutiroConfig, Record, RecordId};
pub use dimension::{Dimension, DimensionType, ValueAccessor};
pub use group::{Group, GroupBy, Reducer, ReduceValue};
pub use filter::{Filter, FilterType, FilterPredicate, RangeFilter, SetFilter, CustomFilter};
pub use coordinator::{ViewCoordinator, ChartView, CoordinationStrategy};
pub use chart_redraw::{ChartRedrawEngine, RedrawStrategy, RenderTarget};

/// Main error type for Chigutiro operations
#[derive(Debug, thiserror::Error)]
pub enum ChigutiroError {
    #[error("Dimension not found: {name}")]
    DimensionNotFound { name: String },
    
    #[error("Invalid filter range: {start} to {end}")]
    InvalidFilterRange { start: String, end: String },
    
    #[error("Reducer error: {message}")]
    ReducerError { message: String },
    
    #[error("Index corruption detected")]
    IndexCorruption,
    
    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocation { size: usize },
    
    #[error("Probabilistic structure error: {message}")]
    ProbabilisticError { message: String },
    
    #[error("Chart coordination error: {message}")]
    CoordinationError { message: String },
}

/// Result type for Chigutiro operations
pub type ChigutiroResult<T> = Result<T, ChigutiroError>;

/// Performance metrics for monitoring system health
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_records: usize,
    pub active_filters: usize,
    pub filtered_records: usize,
    pub last_filter_time_us: u64,
    pub last_redraw_time_us: u64,
    pub memory_usage_bytes: usize,
    pub cache_hit_rate: f64,
    pub probabilistic_false_positive_rate: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_records: 0,
            active_filters: 0,
            filtered_records: 0,
            last_filter_time_us: 0,
            last_redraw_time_us: 0,
            memory_usage_bytes: 0,
            cache_hit_rate: 1.0,
            probabilistic_false_positive_rate: 0.01,
        }
    }
} 