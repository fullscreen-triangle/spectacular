use std::collections::HashMap;
use serde_json::Value;
use crate::chigutiro::{ChigutiroResult, ChigutiroError};

/// Group by function type
pub type GroupBy<R> = fn(&R) -> Value;

/// Reduce function for aggregating grouped data
pub type Reducer<T> = fn(T, T) -> T;

/// Value that can be reduced/aggregated
pub trait ReduceValue: Clone + Send + Sync {
    fn reduce_add(&mut self, other: &Self);
    fn reduce_remove(&mut self, other: &Self);
    fn empty() -> Self;
}

/// A group represents aggregated data for a dimension value
pub struct Group<R, T> {
    /// Key for this group
    pub key: Value,
    
    /// Aggregated value
    pub value: T,
    
    /// Records in this group
    pub records: Vec<R>,
}

impl<R, T> Group<R, T> {
    pub fn new(key: Value, value: T) -> Self {
        Self {
            key,
            value,
            records: Vec::new(),
        }
    }
} 