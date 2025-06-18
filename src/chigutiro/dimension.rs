use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;
use rayon::prelude::*;

use crate::chigutiro::{ChigutiroResult, ChigutiroError};
use super::core::{Record, ChigutiroConfig};
use super::index::{SortedIndex, HashIndex, BloomIndex};
use super::filter::{Filter, FilterType};

/// Type of data in a dimension
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionType {
    Numeric,
    String,
    Boolean,
    DateTime,
    Array,
    Object,
}

/// Value accessor function type
pub type ValueAccessor<R> = Arc<dyn Fn(&R) -> Value + Send + Sync>;

/// A dimension represents a single field/attribute that can be filtered
pub struct Dimension<R: Record> {
    /// Name of the dimension
    pub name: String,
    
    /// Type of values in this dimension
    pub dimension_type: DimensionType,
    
    /// Function to extract values from records
    value_accessor: ValueAccessor<R>,
    
    /// Sorted index for range queries
    sorted_index: Option<SortedIndex>,
    
    /// Hash index for exact matches
    hash_index: Option<HashIndex>,
    
    /// Bloom filter for probabilistic filtering
    bloom_index: Option<BloomIndex>,
    
    /// Current filter applied to this dimension
    current_filter: Option<Filter>,
    
    /// Statistics about this dimension
    stats: DimensionStats,
}

#[derive(Debug, Clone)]
pub struct DimensionStats {
    pub unique_values: usize,
    pub min_value: Option<Value>,
    pub max_value: Option<Value>,
    pub null_count: usize,
    pub total_records: usize,
    pub cardinality_ratio: f64, // unique_values / total_records
}

impl Default for DimensionStats {
    fn default() -> Self {
        Self {
            unique_values: 0,
            min_value: None,
            max_value: None,
            null_count: 0,
            total_records: 0,
            cardinality_ratio: 0.0,
        }
    }
}

impl<R: Record> Dimension<R> {
    /// Create a new dimension
    pub fn new<F>(name: String, accessor: F) -> Self 
    where
        F: Fn(&R) -> Value + Send + Sync + 'static,
    {
        Self {
            name,
            dimension_type: DimensionType::Object, // Will be inferred
            value_accessor: Arc::new(accessor),
            sorted_index: None,
            hash_index: None,
            bloom_index: None,
            current_filter: None,
            stats: DimensionStats::default(),
        }
    }
    
    /// Add a record to this dimension's indexes
    pub fn add_record(&mut self, record: &R) -> ChigutiroResult<()> {
        let value = (self.value_accessor)(record);
        
        // Infer dimension type from first non-null value
        if self.stats.total_records == 0 && !value.is_null() {
            self.dimension_type = self.infer_type(&value);
        }
        
        // Update statistics
        self.update_stats(&value);
        
        // Update indexes
        if let Some(ref mut sorted_index) = self.sorted_index {
            sorted_index.add(record.get_id(), &value)?;
        }
        
        if let Some(ref mut hash_index) = self.hash_index {
            hash_index.add(record.get_id(), &value)?;
        }
        
        if let Some(ref mut bloom_index) = self.bloom_index {
            bloom_index.add(&value)?;
        }
        
        Ok(())
    }
    
    /// Add multiple records in batch
    pub fn add_records_batch(&mut self, records: &[R]) -> ChigutiroResult<()> {
        for record in records {
            self.add_record(record)?;
        }
        Ok(())
    }
    
    /// Apply a filter to this dimension
    pub fn filter(&mut self, filter: Filter) -> ChigutiroResult<()> {
        // Validate filter is compatible with dimension type
        self.validate_filter(&filter)?;
        
        self.current_filter = Some(filter);
        Ok(())
    }
    
    /// Clear the current filter
    pub fn clear_filter(&mut self) {
        self.current_filter = None;
    }
    
    /// Get all record IDs that match the current filter
    pub fn get_filtered_records(&self) -> ChigutiroResult<Vec<u64>> {
        if let Some(ref filter) = self.current_filter {
            match &filter.filter_type {
                FilterType::Exact(value) => {
                    if let Some(ref hash_index) = self.hash_index {
                        hash_index.get_records(value)
                    } else {
                        Err(ChigutiroError::IndexCorruption)
                    }
                },
                FilterType::Range(min, max) => {
                    if let Some(ref sorted_index) = self.sorted_index {
                        sorted_index.range_query(min, max)
                    } else {
                        Err(ChigutiroError::IndexCorruption)
                    }
                },
                FilterType::Set(values) => {
                    if let Some(ref hash_index) = self.hash_index {
                        let mut result = Vec::new();
                        for value in values {
                            result.extend(hash_index.get_records(value)?);
                        }
                        Ok(result)
                    } else {
                        Err(ChigutiroError::IndexCorruption)
                    }
                },
                FilterType::Custom(_) => {
                    // For custom filters, we need to scan all records
                    // This is a placeholder - would need access to all records
                    Ok(vec![])
                },
            }
        } else {
            // No filter, return empty (all records match)
            Ok(vec![])
        }
    }
    
    /// Check if a value passes the current filter
    pub fn matches_filter(&self, value: &Value) -> bool {
        if let Some(ref filter) = self.current_filter {
            filter.matches(value)
        } else {
            true // No filter means all values match
        }
    }
    
    /// Create appropriate indexes based on dimension characteristics
    pub fn create_indexes(&mut self, config: &ChigutiroConfig) -> ChigutiroResult<()> {
        // Always create hash index for exact matches
        self.hash_index = Some(HashIndex::new());
        
        // Create sorted index for range queries if numeric or string
        match self.dimension_type {
            DimensionType::Numeric | DimensionType::String | DimensionType::DateTime => {
                self.sorted_index = Some(SortedIndex::new());
            },
            _ => {},
        }
        
        // Create bloom filter for large datasets
        if config.enable_probabilistic && self.stats.total_records > config.probabilistic_threshold {
            self.bloom_index = Some(BloomIndex::new(
                self.stats.total_records, 
                config.false_positive_rate
            ));
        }
        
        Ok(())
    }
    
    /// Optimize indexes for better performance
    pub fn optimize(&mut self) -> ChigutiroResult<()> {
        if let Some(ref mut sorted_index) = self.sorted_index {
            sorted_index.optimize()?;
        }
        
        if let Some(ref mut hash_index) = self.hash_index {
            hash_index.optimize()?;
        }
        
        Ok(())
    }
    
    /// Get dimension statistics
    pub fn get_stats(&self) -> &DimensionStats {
        &self.stats
    }
    
    /// Infer the type of a value
    fn infer_type(&self, value: &Value) -> DimensionType {
        match value {
            Value::Number(_) => DimensionType::Numeric,
            Value::String(s) => {
                // Try to parse as datetime
                if chrono::DateTime::parse_from_rfc3339(s).is_ok() {
                    DimensionType::DateTime
                } else {
                    DimensionType::String
                }
            },
            Value::Bool(_) => DimensionType::Boolean,
            Value::Array(_) => DimensionType::Array,
            Value::Object(_) => DimensionType::Object,
            Value::Null => DimensionType::Object, // Default for null
        }
    }
    
    /// Update dimension statistics
    fn update_stats(&mut self, value: &Value) {
        if value.is_null() {
            self.stats.null_count += 1;
        } else {
            // Update min/max for comparable types
            match value {
                Value::Number(n) => {
                    if let Some(Value::Number(current_min)) = &self.stats.min_value {
                        if n < current_min {
                            self.stats.min_value = Some(value.clone());
                        }
                    } else {
                        self.stats.min_value = Some(value.clone());
                    }
                    
                    if let Some(Value::Number(current_max)) = &self.stats.max_value {
                        if n > current_max {
                            self.stats.max_value = Some(value.clone());
                        }
                    } else {
                        self.stats.max_value = Some(value.clone());
                    }
                },
                Value::String(s) => {
                    if let Some(Value::String(current_min)) = &self.stats.min_value {
                        if s < current_min {
                            self.stats.min_value = Some(value.clone());
                        }
                    } else {
                        self.stats.min_value = Some(value.clone());
                    }
                    
                    if let Some(Value::String(current_max)) = &self.stats.max_value {
                        if s > current_max {
                            self.stats.max_value = Some(value.clone());
                        }
                    } else {
                        self.stats.max_value = Some(value.clone());
                    }
                },
                _ => {},
            }
        }
        
        self.stats.total_records += 1;
        // Cardinality estimation would be updated by indexes
    }
    
    /// Validate that a filter is compatible with this dimension
    fn validate_filter(&self, filter: &Filter) -> ChigutiroResult<()> {
        match &filter.filter_type {
            FilterType::Range(min, max) => {
                match self.dimension_type {
                    DimensionType::Numeric | DimensionType::String | DimensionType::DateTime => Ok(()),
                    _ => Err(ChigutiroError::InvalidFilterRange {
                        start: format!("{:?}", min),
                        end: format!("{:?}", max),
                    }),
                }
            },
            _ => Ok(()), // Other filter types are generally compatible
        }
    }
}

/// Manager for all dimensions in the crossfilter
pub struct DimensionManager<R: Record> {
    /// All dimensions keyed by name
    dimensions: HashMap<String, Arc<Dimension<R>>>,
}

impl<R: Record> DimensionManager<R> {
    /// Create a new dimension manager
    pub fn new() -> Self {
        Self {
            dimensions: HashMap::new(),
        }
    }
    
    /// Create a new dimension
    pub fn create_dimension<F>(
        &mut self, 
        name: &str, 
        accessor: F,
        config: &ChigutiroConfig,
    ) -> ChigutiroResult<Arc<Dimension<R>>>
    where
        F: Fn(&R) -> Value + Send + Sync + 'static,
    {
        let mut dimension = Dimension::new(name.to_string(), accessor);
        dimension.create_indexes(config)?;
        
        let dimension_arc = Arc::new(dimension);
        self.dimensions.insert(name.to_string(), dimension_arc.clone());
        
        Ok(dimension_arc)
    }
    
    /// Get a dimension by name
    pub fn get_dimension(&self, name: &str) -> Option<Arc<Dimension<R>>> {
        self.dimensions.get(name).cloned()
    }
    
    /// Add a record to all dimensions
    pub fn add_record(&mut self, record: R) -> ChigutiroResult<()> {
        for dimension in self.dimensions.values_mut() {
            // We need a mutable reference, but Arc doesn't allow this
            // In practice, we'd use Arc<RwLock<Dimension<R>>> or similar
            // For now, this is a structural placeholder
        }
        Ok(())
    }
    
    /// Add multiple records to all dimensions
    pub fn add_records_batch(&mut self, records: Vec<R>) -> ChigutiroResult<()> {
        for record in records {
            self.add_record(record)?;
        }
        Ok(())
    }
    
    /// Optimize all dimensions
    pub fn optimize_all(&mut self) -> ChigutiroResult<()> {
        for dimension in self.dimensions.values_mut() {
            // Similar Arc mutability issue as above
        }
        Ok(())
    }
    
    /// Clear all dimensions
    pub fn clear(&mut self) {
        self.dimensions.clear();
    }
    
    /// Get the number of dimensions
    pub fn dimension_count(&self) -> usize {
        self.dimensions.len()
    }
    
    /// Get all dimension names
    pub fn dimension_names(&self) -> Vec<String> {
        self.dimensions.keys().cloned().collect()
    }
} 