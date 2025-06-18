use std::collections::{HashMap, HashSet};
use serde_json::Value;
use super::core::Record;

/// Types of filters that can be applied
#[derive(Debug, Clone)]
pub enum FilterType {
    /// Exact value match
    Exact(Value),
    
    /// Range filter (min, max) - inclusive
    Range(Value, Value),
    
    /// Set membership - matches any value in the set
    Set(HashSet<Value>),
    
    /// Custom predicate function
    Custom(FilterPredicate),
}

/// Function type for custom filter predicates
pub type FilterPredicate = fn(&Value) -> bool;

/// A filter that can be applied to a dimension
#[derive(Debug, Clone)]
pub struct Filter {
    /// The type of filter
    pub filter_type: FilterType,
    
    /// Optional name for the filter
    pub name: Option<String>,
    
    /// Whether the filter is currently active
    pub active: bool,
}

impl Filter {
    /// Create a new exact match filter
    pub fn exact(value: Value) -> Self {
        Self {
            filter_type: FilterType::Exact(value),
            name: None,
            active: true,
        }
    }
    
    /// Create a new range filter
    pub fn range(min: Value, max: Value) -> Self {
        Self {
            filter_type: FilterType::Range(min, max),
            name: None,
            active: true,
        }
    }
    
    /// Create a new set filter
    pub fn set(values: HashSet<Value>) -> Self {
        Self {
            filter_type: FilterType::Set(values),
            name: None,
            active: true,
        }
    }
    
    /// Create a new custom filter
    pub fn custom(predicate: FilterPredicate) -> Self {
        Self {
            filter_type: FilterType::Custom(predicate),
            name: None,
            active: true,
        }
    }
    
    /// Check if a value matches this filter
    pub fn matches(&self, value: &Value) -> bool {
        if !self.active {
            return true; // Inactive filters match everything
        }
        
        match &self.filter_type {
            FilterType::Exact(filter_value) => value == filter_value,
            FilterType::Range(min, max) => {
                self.value_in_range(value, min, max)
            },
            FilterType::Set(values) => values.contains(value),
            FilterType::Custom(predicate) => predicate(value),
        }
    }
    
    /// Check if a value is within a range
    fn value_in_range(&self, value: &Value, min: &Value, max: &Value) -> bool {
        match (value, min, max) {
            (Value::Number(v), Value::Number(min_n), Value::Number(max_n)) => {
                v >= min_n && v <= max_n
            },
            (Value::String(v), Value::String(min_s), Value::String(max_s)) => {
                v >= min_s && v <= max_s
            },
            _ => false, // Type mismatch
        }
    }
    
    /// Set the filter name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// Activate the filter
    pub fn activate(&mut self) {
        self.active = true;
    }
    
    /// Deactivate the filter
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// Convenient filter constructors
pub struct RangeFilter;

impl RangeFilter {
    /// Create a numeric range filter
    pub fn numeric(min: f64, max: f64) -> Filter {
        Filter::range(
            Value::Number(serde_json::Number::from_f64(min).unwrap()),
            Value::Number(serde_json::Number::from_f64(max).unwrap()),
        )
    }
    
    /// Create a string range filter
    pub fn string(min: String, max: String) -> Filter {
        Filter::range(Value::String(min), Value::String(max))
    }
    
    /// Create a date range filter (expecting RFC3339 format)
    pub fn date(min: String, max: String) -> Filter {
        Filter::range(Value::String(min), Value::String(max))
    }
}

/// Convenient set filter constructors
pub struct SetFilter;

impl SetFilter {
    /// Create a set filter from a vector of values
    pub fn from_values(values: Vec<Value>) -> Filter {
        let set: HashSet<Value> = values.into_iter().collect();
        Filter::set(set)
    }
    
    /// Create a set filter from string values
    pub fn from_strings(values: Vec<String>) -> Filter {
        let set: HashSet<Value> = values
            .into_iter()
            .map(Value::String)
            .collect();
        Filter::set(set)
    }
    
    /// Create a set filter from numeric values
    pub fn from_numbers(values: Vec<f64>) -> Filter {
        let set: HashSet<Value> = values
            .into_iter()
            .filter_map(|n| serde_json::Number::from_f64(n).map(Value::Number))
            .collect();
        Filter::set(set)
    }
}

/// Custom filter constructors
pub struct CustomFilter;

impl CustomFilter {
    /// Create a filter that checks if a string contains a substring
    pub fn contains(substring: String) -> Filter {
        Filter::custom(move |value| {
            if let Value::String(s) = value {
                s.contains(&substring)
            } else {
                false
            }
        })
    }
    
    /// Create a filter that checks if a string starts with a prefix
    pub fn starts_with(prefix: String) -> Filter {
        Filter::custom(move |value| {
            if let Value::String(s) = value {
                s.starts_with(&prefix)
            } else {
                false
            }
        })
    }
    
    /// Create a filter that checks if a string ends with a suffix
    pub fn ends_with(suffix: String) -> Filter {
        Filter::custom(move |value| {
            if let Value::String(s) = value {
                s.ends_with(&suffix)
            } else {
                false
            }
        })
    }
    
    /// Create a filter that checks if a number is even
    pub fn is_even() -> Filter {
        Filter::custom(|value| {
            if let Value::Number(n) = value {
                if let Some(i) = n.as_i64() {
                    i % 2 == 0
                } else {
                    false
                }
            } else {
                false
            }
        })
    }
    
    /// Create a filter that checks if a number is odd
    pub fn is_odd() -> Filter {
        Filter::custom(|value| {
            if let Value::Number(n) = value {
                if let Some(i) = n.as_i64() {
                    i % 2 != 0
                } else {
                    false
                }
            } else {
                false
            }
        })
    }
    
    /// Create a filter with a custom lambda
    pub fn lambda<F>(predicate: F) -> Filter 
    where 
        F: Fn(&Value) -> bool + 'static
    {
        // Note: This is a simplified version. In practice, we'd need to handle
        // the function pointer conversion more carefully
        Filter::custom(|_| false) // Placeholder
    }
}

/// Manager for coordinating filters across multiple dimensions
pub struct FilterManager {
    /// Filters for each dimension
    dimension_filters: HashMap<String, Vec<Filter>>,
    
    /// Global filters that apply to all records
    global_filters: Vec<Filter>,
    
    /// Filter combination mode
    combination_mode: FilterCombination,
}

/// How to combine multiple filters
#[derive(Debug, Clone, Copy)]
pub enum FilterCombination {
    /// All filters must match (AND)
    All,
    
    /// Any filter can match (OR)
    Any,
    
    /// Custom combination logic
    Custom,
}

impl FilterManager {
    /// Create a new filter manager
    pub fn new() -> Self {
        Self {
            dimension_filters: HashMap::new(),
            global_filters: Vec::new(),
            combination_mode: FilterCombination::All,
        }
    }
    
    /// Add a filter to a specific dimension
    pub fn add_dimension_filter(&mut self, dimension: String, filter: Filter) {
        self.dimension_filters
            .entry(dimension)
            .or_insert_with(Vec::new)
            .push(filter);
    }
    
    /// Add a global filter
    pub fn add_global_filter(&mut self, filter: Filter) {
        self.global_filters.push(filter);
    }
    
    /// Remove all filters for a dimension
    pub fn clear_dimension_filters(&mut self, dimension: &str) {
        self.dimension_filters.remove(dimension);
    }
    
    /// Remove all global filters
    pub fn clear_global_filters(&mut self) {
        self.global_filters.clear();
    }
    
    /// Remove all filters
    pub fn clear_all_filters(&mut self) {
        self.dimension_filters.clear();
        self.global_filters.clear();
    }
    
    /// Check if there are any active filters
    pub fn has_active_filters(&self) -> bool {
        !self.dimension_filters.is_empty() || !self.global_filters.is_empty()
    }
    
    /// Get the number of active filters
    pub fn active_filter_count(&self) -> usize {
        let dimension_count: usize = self.dimension_filters
            .values()
            .map(|filters| filters.iter().filter(|f| f.active).count())
            .sum();
        
        let global_count = self.global_filters
            .iter()
            .filter(|f| f.active)
            .count();
        
        dimension_count + global_count
    }
    
    /// Check if a record matches all active filters
    pub fn matches_all_filters<R: Record>(&self, record: &R) -> bool {
        // Check global filters
        for filter in &self.global_filters {
            if !filter.active {
                continue;
            }
            
            // For global filters, we'd need to specify which field to check
            // This is a structural limitation - in practice, global filters
            // would need to be implemented differently
        }
        
        // Check dimension-specific filters
        for (dimension_name, filters) in &self.dimension_filters {
            if let Some(field_value) = record.get_field(dimension_name) {
                let matches = match self.combination_mode {
                    FilterCombination::All => {
                        filters.iter().all(|filter| filter.matches(&field_value))
                    },
                    FilterCombination::Any => {
                        filters.iter().any(|filter| filter.matches(&field_value))
                    },
                    FilterCombination::Custom => {
                        // Custom logic would go here
                        filters.iter().all(|filter| filter.matches(&field_value))
                    },
                };
                
                if !matches {
                    return false;
                }
            } else {
                // Field not found in record - depending on policy, this could
                // be treated as non-matching or matching
                return false;
            }
        }
        
        true
    }
    
    /// Set the filter combination mode
    pub fn set_combination_mode(&mut self, mode: FilterCombination) {
        self.combination_mode = mode;
    }
    
    /// Get filter statistics
    pub fn get_filter_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("total_dimensions".to_string(), 
                    serde_json::json!(self.dimension_filters.len()));
        
        stats.insert("global_filters".to_string(), 
                    serde_json::json!(self.global_filters.len()));
        
        stats.insert("active_filters".to_string(), 
                    serde_json::json!(self.active_filter_count()));
        
        stats.insert("combination_mode".to_string(), 
                    serde_json::json!(format!("{:?}", self.combination_mode)));
        
        // Per-dimension filter counts
        let mut dimension_stats = HashMap::new();
        for (dim, filters) in &self.dimension_filters {
            dimension_stats.insert(dim.clone(), filters.len());
        }
        stats.insert("dimension_filter_counts".to_string(), 
                    serde_json::json!(dimension_stats));
        
        stats
    }
} 