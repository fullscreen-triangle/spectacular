//! Main entry point for Spectacular

use spectacular::{
    init_tracing, Spectacular, SpectacularConfig,
    data::{ScientificDataset, DataSource, ColumnInfo, DatasetMetadata, DataType},
};

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use tracing::info;

#[derive(Parser)]
#[command(name = "spectacular")]
#[command(about = "High-performance scientific visualization system")]
#[command(long_about = "Spectacular generates optimized D3.js visualizations from large scientific datasets using hybrid logical programming and fuzzy logic.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    #[arg(long, default_value = "info")]
    log_level: String,
    
    #[arg(long)]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a visualization from a query
    Generate {
        /// Natural language query describing the desired visualization
        query: String,
        
        /// Optional data file (CSV, Parquet, etc.)
        #[arg(long)]
        data: Option<String>,
        
        /// Output file for the generated HTML
        #[arg(long, default_value = "output.html")]
        output: String,
    },
    
    /// Start interactive mode
    Interactive,
    
    /// Run system health check
    Health,
    
    /// Show system configuration
    Config,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize tracing
    init_tracing()?;
    
    // Load configuration
    let config = if let Some(config_path) = cli.config {
        SpectacularConfig::from_file(&config_path)?
    } else {
        SpectacularConfig::from_env()
    };
    
    // Validate configuration
    config.validate().map_err(|e| format!("Configuration error: {}", e))?;
    
    // Initialize Spectacular system
    let spectacular = Spectacular::new(config).await?;
    
    match cli.command {
        Commands::Generate { query, data, output } => {
            handle_generate(spectacular, query, data, output).await?;
        },
        Commands::Interactive => {
            handle_interactive(spectacular).await?;
        },
        Commands::Health => {
            handle_health(spectacular).await?;
        },
        Commands::Config => {
            handle_config().await?;
        },
    }
    
    Ok(())
}

async fn handle_generate(
    spectacular: Spectacular,
    query: String,
    data_file: Option<String>,
    output: String,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating visualization for query: {}", query);
    
    // Load dataset if provided
    let dataset = if let Some(path) = data_file {
        Some(load_dataset(&path).await?)
    } else {
        None
    };
    
    // Generate visualization
    let result = spectacular.generate_visualization(&query, dataset).await?;
    
    if result.success {
        // Write HTML output
        if let Some(html) = result.html_template {
            std::fs::write(&output, html)?;
            println!("✅ Visualization generated successfully!");
            println!("📁 Output saved to: {}", output);
            println!("🎯 Confidence: {:.1}%", result.confidence_score * 100.0);
            
            if result.data_reduction_applied {
                println!("📊 Data reduced: {} → {} points ({:.1}% reduction)",
                    result.original_data_points,
                    result.reduced_data_points,
                    (1.0 - result.reduced_data_points as f64 / result.original_data_points as f64) * 100.0
                );
            }
        } else {
            println!("⚠️  Visualization generated but no HTML template available");
        }
    } else {
        println!("❌ Visualization generation failed");
        if let Some(error) = result.error {
            println!("Error: {}", error);
        }
    }
    
    Ok(())
}

async fn handle_interactive(spectacular: Spectacular) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Spectacular Interactive Mode");
    println!("Type 'quit' to exit, 'help' for commands");
    println!();
    
    loop {
        print!("spectacular> ");
        use std::io::{self, Write};
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        match input {
            "quit" | "exit" => break,
            "help" => {
                println!("Commands:");
                println!("  <query>  - Generate visualization from natural language");
                println!("  health   - Show system health");
                println!("  quit     - Exit interactive mode");
            },
            "health" => {
                let health = spectacular.health_check().await?;
                println!("System Status: {}", health.status);
                println!("Uptime: {}s", health.uptime_seconds);
                println!("Memory: {:.1}MB", health.memory_usage_mb);
                println!("Active queries: {}", health.active_queries);
            },
            query if !query.is_empty() => {
                println!("🔄 Processing: {}", query);
                
                match spectacular.generate_visualization(query, None).await {
                    Ok(result) => {
                        println!("✅ Generated {} chart (confidence: {:.1}%)",
                            result.query_id, result.confidence_score * 100.0);
                        
                        if let Some(debug_info) = result.debug_info {
                            if !debug_info.syntax_errors.is_empty() {
                                println!("⚠️  Syntax errors: {}", debug_info.syntax_errors.len());
                            }
                        }
                    },
                    Err(e) => {
                        println!("❌ Error: {}", e);
                    }
                }
            },
            _ => {
                println!("Unknown command. Type 'help' for available commands.");
            }
        }
        
        println!();
    }
    
    println!("👋 Goodbye!");
    Ok(())
}

async fn handle_health(spectacular: Spectacular) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏥 Spectacular System Health Check");
    println!();
    
    let health = spectacular.health_check().await?;
    
    println!("Status: {}", health.status);
    println!("Uptime: {} seconds", health.uptime_seconds);
    println!("Memory Usage: {:.1} MB", health.memory_usage_mb);
    println!("Active Queries: {}", health.active_queries);
    println!("Cache Hit Rate: {:.1}%", health.cache_hit_rate);
    println!("HF Models Loaded: {}", health.hf_models_loaded);
    println!("Pretoria Rules Active: {}", health.pretoria_rules_active);
    
    Ok(())
}

async fn handle_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚙️  Spectacular Configuration");
    println!();
    
    let config = SpectacularConfig::from_env();
    
    println!("Data Processing:");
    println!("  Max Points Threshold: {}", config.data_processing.max_points_threshold);
    println!("  Target Points: {}", config.data_processing.target_points_for_visualization);
    println!("  Parallel Processing: {}", config.data_processing.parallel_processing);
    
    println!();
    println!("Pretoria Engine:");
    println!("  Max Fuzzy Rules: {}", config.pretoria.max_fuzzy_rules);
    println!("  Confidence Threshold: {}", config.pretoria.confidence_threshold);
    println!("  Logical Programming: {}", config.pretoria.enable_logical_programming);
    
    println!();
    println!("HuggingFace:");
    println!("  API Key Set: {}", config.huggingface.api_key.is_some());
    println!("  Local Models: {}", config.huggingface.enable_local_models);
    println!("  GPU Enabled: {}", config.huggingface.gpu_enabled);
    
    Ok(())
}

async fn load_dataset(path: &str) -> Result<ScientificDataset, Box<dyn std::error::Error>> {
    info!("Loading dataset from: {}", path);
    
    // Placeholder implementation
    Ok(ScientificDataset {
        name: path.to_string(),
        data: DataSource::External(crate::data::ExternalDataSource {
            source_type: crate::data::ExternalSourceType::CSV,
            connection_string: path.to_string(),
            query: None,
            table_name: None,
        }),
        columns: vec![
            ColumnInfo {
                name: "x".to_string(),
                data_type: DataType::Float,
                is_numeric: true,
                is_categorical: false,
                cardinality: None,
                min_value: Some(0.0),
                max_value: Some(100.0),
                null_count: 0,
            },
            ColumnInfo {
                name: "y".to_string(),
                data_type: DataType::Float,
                is_numeric: true,
                is_categorical: false,
                cardinality: None,
                min_value: Some(0.0),
                max_value: Some(100.0),
                null_count: 0,
            },
        ],
        metadata: DatasetMetadata {
            rows: 1000,
            columns: 2,
            memory_size_mb: 1.5,
            has_missing_values: false,
            has_outliers: false,
            temporal_columns: vec![],
            spatial_columns: vec![],
            tags: HashMap::new(),
        },
        estimated_size: 1000,
    })
} 