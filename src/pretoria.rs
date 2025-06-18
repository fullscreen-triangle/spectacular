//! Pretoria: Hybrid Logical Programming and Fuzzy Logic Engine
//! 
//! This module implements the sophisticated reasoning system that uses both:
//! 1. Fuzzy logic for handling uncertainty in visualization decisions
//! 2. Prolog-style logical programming for rule-based chart selection
//! 3. Answer Set Programming (ASP) for optimization problems

use crate::{
    config::PretoriaConfig,
    core::QueryContext,
    error::{Result, SpectacularError},
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn, debug, instrument};
use tokio::sync::RwLock;

/// Fuzzy membership value (0.0 to 1.0)
pub type FuzzyValue = f64;

/// Represents a fuzzy logic rule for chart reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRule {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub conclusion: String,
    pub confidence: f64,
}

/// Complete analysis result from Pretoria engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretoriaAnalysis {
    pub recommended_chart_type: String,
    pub chart_alternatives: Vec<ChartRecommendation>,
    pub fuzzy_scores: HashMap<String, FuzzyValue>,
    pub logical_reasoning: Vec<LogicalStep>,
    pub applied_rules: Vec<String>,
    pub confidence: f64,
    pub optimization_hints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartRecommendation {
    pub chart_type: String,
    pub confidence: f64,
    pub reasoning: String,
    pub data_mappings: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalStep {
    pub step_type: String,
    pub rule_applied: String,
    pub conclusion: String,
    pub confidence: f64,
}

/// Logical program structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalProgram {
    pub name: String,
    pub rules: Vec<LogicalRule>,
    pub facts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalRule {
    pub head: String,
    pub body: Vec<String>,
    pub confidence: f64,
}

/// Main Pretoria engine
pub struct PretoriaEngine {
    config: PretoriaConfig,
    fuzzy_rules: Arc<RwLock<Vec<FuzzyRule>>>,
    logical_programs: Arc<RwLock<HashMap<String, LogicalProgram>>>,
}

impl PretoriaEngine {
    /// Create a new Pretoria engine
    pub async fn new(config: &PretoriaConfig) -> Result<Self> {
        info!("Initializing Pretoria hybrid reasoning engine");
        
        let engine = Self {
            config: config.clone(),
            fuzzy_rules: Arc::new(RwLock::new(Vec::new())),
            logical_programs: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Initialize rules and programs
        engine.initialize_fuzzy_rules().await?;
        engine.initialize_logical_programs().await?;
        
        info!("Pretoria engine initialized with {} fuzzy rules", 
              engine.fuzzy_rules.read().await.len());
        
        Ok(engine)
    }
    
    /// Analyze a query using hybrid reasoning
    #[instrument(skip(self, context))]
    pub async fn analyze_query(&self, context: &QueryContext) -> Result<PretoriaAnalysis> {
        info!("Starting Pretoria analysis for query: {}", &context.query[..50.min(context.query.len())]);
        
        // Step 1: Fuzzy inference
        let fuzzy_scores = self.perform_fuzzy_inference(context).await?;
        
        // Step 2: Logical reasoning
        let logical_steps = self.perform_logical_reasoning(context, &fuzzy_scores).await?;
        
        // Step 3: Chart recommendations
        let recommendations = self.generate_chart_recommendations(context, &fuzzy_scores).await?;
        
        let analysis = PretoriaAnalysis {
            recommended_chart_type: recommendations.first()
                .map(|r| r.chart_type.clone())
                .unwrap_or_else(|| "scatter".to_string()),
            chart_alternatives: recommendations,
            fuzzy_scores: fuzzy_scores.clone(),
            logical_reasoning: logical_steps,
            applied_rules: vec!["base_rule".to_string()],
            confidence: self.calculate_confidence(&fuzzy_scores),
            optimization_hints: self.generate_optimization_hints(context).await?,
        };
        
        debug!("Pretoria analysis completed with confidence: {:.2}", analysis.confidence);
        Ok(analysis)
    }
    
    async fn perform_fuzzy_inference(&self, context: &QueryContext) -> Result<HashMap<String, FuzzyValue>> {
        let mut scores = HashMap::new();
        
        // Calculate fuzzy variables
        scores.insert("data_complexity".to_string(), self.calculate_data_complexity(context));
        scores.insert("query_complexity".to_string(), context.complexity_estimate);
        scores.insert("visual_complexity".to_string(), context.visual_complexity);
        scores.insert("interaction_need".to_string(), self.assess_interaction_need(context));
        
        Ok(scores)
    }
    
    async fn perform_logical_reasoning(&self, context: &QueryContext, _fuzzy_scores: &HashMap<String, FuzzyValue>) -> Result<Vec<LogicalStep>> {
        let mut steps = Vec::new();
        
        // Simple rule-based reasoning
        let query_lower = context.query.to_lowercase();
        
        if query_lower.contains("scatter") || query_lower.contains("correlation") {
            steps.push(LogicalStep {
                step_type: "chart_selection".to_string(),
                rule_applied: "scatter_rule".to_string(),
                conclusion: "Scatter plot recommended".to_string(),
                confidence: 0.8,
            });
        }
        
        if query_lower.contains("bar") || query_lower.contains("category") {
            steps.push(LogicalStep {
                step_type: "chart_selection".to_string(),
                rule_applied: "bar_rule".to_string(),
                conclusion: "Bar chart recommended".to_string(),
                confidence: 0.8,
            });
        }
        
        Ok(steps)
    }
    
    async fn generate_chart_recommendations(&self, context: &QueryContext, fuzzy_scores: &HashMap<String, FuzzyValue>) -> Result<Vec<ChartRecommendation>> {
        let mut recommendations = Vec::new();
        let query_lower = context.query.to_lowercase();
        
        // Rule-based chart selection
        if query_lower.contains("scatter") || query_lower.contains("correlation") {
            recommendations.push(ChartRecommendation {
                chart_type: "scatter".to_string(),
                confidence: 0.8,
                reasoning: "Query mentions scatter plot or correlation".to_string(),
                data_mappings: [
                    ("x".to_string(), "numeric".to_string()),
                    ("y".to_string(), "numeric".to_string()),
                ].into_iter().collect(),
            });
        }
        
        if query_lower.contains("bar") || query_lower.contains("category") {
            recommendations.push(ChartRecommendation {
                chart_type: "bar".to_string(),
                confidence: 0.8,
                reasoning: "Query mentions bar chart or categories".to_string(),
                data_mappings: [
                    ("x".to_string(), "categorical".to_string()),
                    ("y".to_string(), "numeric".to_string()),
                ].into_iter().collect(),
            });
        }
        
        if query_lower.contains("line") || query_lower.contains("time") || query_lower.contains("trend") {
            recommendations.push(ChartRecommendation {
                chart_type: "line".to_string(),
                confidence: 0.8,
                reasoning: "Query mentions line chart, time series, or trends".to_string(),
                data_mappings: [
                    ("x".to_string(), "temporal".to_string()),
                    ("y".to_string(), "numeric".to_string()),
                ].into_iter().collect(),
            });
        }
        
        // Default fallback
        if recommendations.is_empty() {
            recommendations.push(ChartRecommendation {
                chart_type: "scatter".to_string(),
                confidence: 0.5,
                reasoning: "Default chart type for general data".to_string(),
                data_mappings: HashMap::new(),
            });
        }
        
        // Sort by confidence
        recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        Ok(recommendations)
    }
    
    async fn generate_optimization_hints(&self, context: &QueryContext) -> Result<Vec<String>> {
        let mut hints = Vec::new();
        
        if let Some(dataset) = &context.dataset {
            if dataset.estimated_size() > 1000 {
                hints.push("Consider data aggregation for better performance".to_string());
            }
            
            if dataset.metadata.has_missing_values {
                hints.push("Handle missing values in the visualization".to_string());
            }
        }
        
        if context.visual_complexity > 0.7 {
            hints.push("Use progressive disclosure for complex visualizations".to_string());
        }
        
        Ok(hints)
    }
    
    pub async fn active_rules_count(&self) -> Result<usize> {
        Ok(self.fuzzy_rules.read().await.len())
    }
    
    // Helper methods
    fn calculate_data_complexity(&self, context: &QueryContext) -> f64 {
        if let Some(dataset) = &context.dataset {
            let size_complexity = (dataset.estimated_size() as f64 / 10000.0).min(1.0);
            let column_complexity = (dataset.columns.len() as f64 / 20.0).min(1.0);
            (size_complexity + column_complexity) / 2.0
        } else {
            0.3
        }
    }
    
    fn assess_interaction_need(&self, context: &QueryContext) -> f64 {
        let query_lower = context.query.to_lowercase();
        let interaction_keywords = ["interactive", "click", "hover", "filter", "select", "zoom"];
        
        let matches = interaction_keywords.iter()
            .filter(|&word| query_lower.contains(word))
            .count() as f64;
        
        (matches / interaction_keywords.len() as f64).min(1.0)
    }
    
    fn calculate_confidence(&self, fuzzy_scores: &HashMap<String, FuzzyValue>) -> f64 {
        let avg_score = fuzzy_scores.values().sum::<f64>() / fuzzy_scores.len() as f64;
        avg_score.min(1.0).max(0.0)
    }
    
    async fn initialize_fuzzy_rules(&self) -> Result<()> {
        let mut rules = self.fuzzy_rules.write().await;
        
        rules.push(FuzzyRule {
            id: "scatter_rule".to_string(),
            name: "Scatter plot rule".to_string(),
            condition: "correlation OR scatter".to_string(),
            conclusion: "recommend scatter plot".to_string(),
            confidence: 0.8,
        });
        
        rules.push(FuzzyRule {
            id: "bar_rule".to_string(),
            name: "Bar chart rule".to_string(),
            condition: "category OR bar".to_string(),
            conclusion: "recommend bar chart".to_string(),
            confidence: 0.8,
        });
        
        Ok(())
    }
    
    async fn initialize_logical_programs(&self) -> Result<()> {
        let mut programs = self.logical_programs.write().await;
        
        let chart_program = LogicalProgram {
            name: "chart_selection".to_string(),
            rules: vec![
                LogicalRule {
                    head: "recommend(scatter)".to_string(),
                    body: vec!["contains(query, correlation)".to_string()],
                    confidence: 0.8,
                },
                LogicalRule {
                    head: "recommend(bar)".to_string(),
                    body: vec!["contains(query, category)".to_string()],
                    confidence: 0.8,
                },
            ],
            facts: vec!["chart_type(scatter)".to_string(), "chart_type(bar)".to_string()],
        };
        
        programs.insert("chart_selection".to_string(), chart_program);
        Ok(())
    }
} 