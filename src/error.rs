//! Error handling for Spectacular

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, SpectacularError>;

/// Main error type for Spectacular system
#[derive(Error, Debug)]
pub enum SpectacularError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    
    #[error("Fuzzy logic error: {0}")]
    FuzzyLogic(String),
    
    #[error("HuggingFace integration error: {0}")]
    HuggingFace(String),
    
    #[error("JavaScript engine error: {0}")]
    JavaScript(String),
    
    #[error("Orchestrator communication error: {0}")]
    Orchestrator(String),
    
    #[error("Unsupported data source: {0}")]
    UnsupportedDataSource(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Timeout error: operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    #[error("System resource error: {0}")]
    SystemResource(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl SpectacularError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            SpectacularError::Network(_) => true,
            SpectacularError::Timeout { .. } => true,
            SpectacularError::HuggingFace(_) => true,
            SpectacularError::Orchestrator(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            SpectacularError::Configuration(_) => ErrorSeverity::Critical,
            SpectacularError::Internal(_) => ErrorSeverity::Critical,
            SpectacularError::SystemResource(_) => ErrorSeverity::High,
            SpectacularError::DataProcessing(_) => ErrorSeverity::Medium,
            SpectacularError::Validation(_) => ErrorSeverity::Medium,
            SpectacularError::JavaScript(_) => ErrorSeverity::Medium,
            SpectacularError::Network(_) => ErrorSeverity::Low,
            SpectacularError::Timeout { .. } => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub query_id: Option<uuid::Uuid>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: &str, component: &str) -> Self {
        Self {
            operation: operation.to_string(),
            component: component.to_string(),
            query_id: None,
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_query_id(mut self, query_id: uuid::Uuid) -> Self {
        self.query_id = Some(query_id);
        self
    }
    
    pub fn with_info(mut self, key: &str, value: &str) -> Self {
        self.additional_info.insert(key.to_string(), value.to_string());
        self
    }
} 