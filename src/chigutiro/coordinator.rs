use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde_json::Value;

use crate::chigutiro::{ChigutiroResult, ChigutiroError};

/// Strategy for coordinating updates between views
#[derive(Debug, Clone, Copy)]
pub enum CoordinationStrategy {
    /// Update all views immediately when any filter changes
    Immediate,
    
    /// Batch updates and apply them together
    Batched,
    
    /// Use priorities to update most important views first
    Prioritized,
    
    /// Lazy evaluation - only update views when they're accessed
    Lazy,
    
    /// Adaptive strategy that switches based on performance
    Adaptive,
}

/// Represents a chart/visualization view
pub struct ChartView {
    /// Unique identifier for the view
    pub id: String,
    
    /// Human-readable name
    pub name: String,
    
    /// Priority for updates (higher = more important)
    pub priority: u32,
    
    /// Whether the view needs updating
    pub needs_update: bool,
    
    /// Last update timestamp
    pub last_updated: std::time::Instant,
    
    /// Configuration for this view
    pub config: ViewConfig,
}

/// Configuration for a chart view
#[derive(Debug, Clone)]
pub struct ViewConfig {
    /// Chart type (bar, line, scatter, etc.)
    pub chart_type: String,
    
    /// Data dimensions this view depends on
    pub dimensions: Vec<String>,
    
    /// Update frequency limit (minimum time between updates)
    pub min_update_interval_ms: u64,
    
    /// Whether this view can be updated incrementally
    pub supports_incremental_updates: bool,
    
    /// Custom properties
    pub properties: HashMap<String, Value>,
}

impl ChartView {
    /// Create a new chart view
    pub fn new(id: String, name: String, config: ViewConfig) -> Self {
        Self {
            id,
            name,
            priority: 100, // Default priority
            needs_update: true,
            last_updated: std::time::Instant::now(),
            config,
        }
    }
    
    /// Check if this view can be updated now
    pub fn can_update(&self) -> bool {
        let elapsed = self.last_updated.elapsed();
        elapsed.as_millis() >= self.config.min_update_interval_ms as u128
    }
    
    /// Mark this view as needing an update
    pub fn mark_dirty(&mut self) {
        self.needs_update = true;
    }
    
    /// Mark this view as updated
    pub fn mark_updated(&mut self) {
        self.needs_update = false;
        self.last_updated = std::time::Instant::now();
    }
}

/// Coordinates updates between multiple chart views
pub struct ViewCoordinator {
    /// All registered views
    views: HashMap<String, Arc<RwLock<ChartView>>>,
    
    /// Coordination strategy
    strategy: CoordinationStrategy,
    
    /// Update queue for batched updates
    update_queue: Vec<String>,
    
    /// Statistics
    stats: CoordinatorStats,
}

#[derive(Debug, Clone, Default)]
pub struct CoordinatorStats {
    pub total_updates: u64,
    pub batched_updates: u64,
    pub failed_updates: u64,
    pub avg_update_time_ms: f64,
    pub last_update_time: Option<std::time::Instant>,
}

impl ViewCoordinator {
    /// Create a new view coordinator
    pub fn new() -> Self {
        Self {
            views: HashMap::new(),
            strategy: CoordinationStrategy::Adaptive,
            update_queue: Vec::new(),
            stats: CoordinatorStats::default(),
        }
    }
    
    /// Register a new view
    pub fn register_view(&mut self, view: ChartView) -> Arc<RwLock<ChartView>> {
        let view_id = view.id.clone();
        let view_arc = Arc::new(RwLock::new(view));
        self.views.insert(view_id, view_arc.clone());
        view_arc
    }
    
    /// Unregister a view
    pub fn unregister_view(&mut self, view_id: &str) {
        self.views.remove(view_id);
    }
    
    /// Notify that a dimension has been filtered
    pub fn notify_filter_change(&mut self, dimension: &str) -> ChigutiroResult<()> {
        // Find views that depend on this dimension
        let affected_views: Vec<String> = self.views
            .iter()
            .filter_map(|(id, view_arc)| {
                let view = view_arc.read().unwrap();
                if view.config.dimensions.contains(&dimension.to_string()) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        
        // Apply coordination strategy
        match self.strategy {
            CoordinationStrategy::Immediate => {
                self.update_views_immediately(&affected_views)?;
            },
            CoordinationStrategy::Batched => {
                self.queue_views_for_update(&affected_views);
            },
            CoordinationStrategy::Prioritized => {
                self.update_views_by_priority(&affected_views)?;
            },
            CoordinationStrategy::Lazy => {
                self.mark_views_dirty(&affected_views);
            },
            CoordinationStrategy::Adaptive => {
                self.adaptive_update(&affected_views)?;
            },
        }
        
        Ok(())
    }
    
    /// Update views immediately
    fn update_views_immediately(&mut self, view_ids: &[String]) -> ChigutiroResult<()> {
        let start_time = std::time::Instant::now();
        
        for view_id in view_ids {
            if let Some(view_arc) = self.views.get(view_id) {
                let mut view = view_arc.write().unwrap();
                if view.can_update() {
                    // Simulate update (in real implementation, this would trigger chart redraw)
                    view.mark_updated();
                    self.stats.total_updates += 1;
                }
            }
        }
        
        self.update_stats(start_time);
        Ok(())
    }
    
    /// Queue views for batched update
    fn queue_views_for_update(&mut self, view_ids: &[String]) {
        for view_id in view_ids {
            if !self.update_queue.contains(view_id) {
                self.update_queue.push(view_id.clone());
            }
        }
    }
    
    /// Update views by priority order
    fn update_views_by_priority(&mut self, view_ids: &[String]) -> ChigutiroResult<()> {
        let start_time = std::time::Instant::now();
        
        // Sort by priority (highest first)
        let mut sorted_views: Vec<_> = view_ids
            .iter()
            .filter_map(|id| {
                self.views.get(id).map(|view_arc| {
                    let view = view_arc.read().unwrap();
                    (id.clone(), view.priority)
                })
            })
            .collect();
        
        sorted_views.sort_by_key(|(_, priority)| std::cmp::Reverse(*priority));
        
        for (view_id, _) in sorted_views {
            if let Some(view_arc) = self.views.get(&view_id) {
                let mut view = view_arc.write().unwrap();
                if view.can_update() {
                    view.mark_updated();
                    self.stats.total_updates += 1;
                }
            }
        }
        
        self.update_stats(start_time);
        Ok(())
    }
    
    /// Mark views as needing updates (lazy)
    fn mark_views_dirty(&mut self, view_ids: &[String]) {
        for view_id in view_ids {
            if let Some(view_arc) = self.views.get(view_id) {
                let mut view = view_arc.write().unwrap();
                view.mark_dirty();
            }
        }
    }
    
    /// Adaptive update strategy
    fn adaptive_update(&mut self, view_ids: &[String]) -> ChigutiroResult<()> {
        // Choose strategy based on current load and performance
        let strategy = if view_ids.len() > 5 {
            CoordinationStrategy::Batched // Many views, use batching
        } else if self.stats.avg_update_time_ms > 100.0 {
            CoordinationStrategy::Lazy // Slow updates, use lazy
        } else {
            CoordinationStrategy::Immediate // Fast updates, use immediate
        };
        
        match strategy {
            CoordinationStrategy::Immediate => self.update_views_immediately(view_ids),
            CoordinationStrategy::Batched => {
                self.queue_views_for_update(view_ids);
                Ok(())
            },
            CoordinationStrategy::Lazy => {
                self.mark_views_dirty(view_ids);
                Ok(())
            },
            _ => self.update_views_immediately(view_ids), // Fallback
        }
    }
    
    /// Process batched updates
    pub fn process_batched_updates(&mut self) -> ChigutiroResult<()> {
        if self.update_queue.is_empty() {
            return Ok(());
        }
        
        let start_time = std::time::Instant::now();
        let queue = std::mem::take(&mut self.update_queue);
        
        for view_id in &queue {
            if let Some(view_arc) = self.views.get(view_id) {
                let mut view = view_arc.write().unwrap();
                if view.can_update() {
                    view.mark_updated();
                    self.stats.total_updates += 1;
                }
            }
        }
        
        self.stats.batched_updates += 1;
        self.update_stats(start_time);
        
        Ok(())
    }
    
    /// Force update a specific view
    pub fn force_update_view(&mut self, view_id: &str) -> ChigutiroResult<()> {
        if let Some(view_arc) = self.views.get(view_id) {
            let mut view = view_arc.write().unwrap();
            view.mark_updated();
            self.stats.total_updates += 1;
            Ok(())
        } else {
            Err(ChigutiroError::CoordinationError {
                message: format!("View not found: {}", view_id),
            })
        }
    }
    
    /// Get all views that need updating
    pub fn get_dirty_views(&self) -> Vec<String> {
        self.views
            .iter()
            .filter_map(|(id, view_arc)| {
                let view = view_arc.read().unwrap();
                if view.needs_update {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Set coordination strategy
    pub fn set_strategy(&mut self, strategy: CoordinationStrategy) {
        self.strategy = strategy;
    }
    
    /// Get coordinator statistics
    pub fn get_stats(&self) -> &CoordinatorStats {
        &self.stats
    }
    
    /// Update performance statistics
    fn update_stats(&mut self, start_time: std::time::Instant) {
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        
        // Update running average
        let total_updates = self.stats.total_updates as f64;
        self.stats.avg_update_time_ms = 
            (self.stats.avg_update_time_ms * (total_updates - 1.0) + elapsed_ms) / total_updates;
        
        self.stats.last_update_time = Some(std::time::Instant::now());
    }
    
    /// Get view count
    pub fn view_count(&self) -> usize {
        self.views.len()
    }
    
    /// Clear all views
    pub fn clear_views(&mut self) {
        self.views.clear();
        self.update_queue.clear();
    }
} 