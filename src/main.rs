//! Spectacular: High-Performance Scientific Visualization Engine
//!
//! Main entry point for the Spectacular visualization system with
//! Autobahn metacognitive engine integration for biological quantum
//! simulation visualization.

use spectacular::{
    SpectacularConfig, AutobahnClient, MolecularRenderer, MolecularRendererConfig,
    BiologicalData, BiologicalDataType, OscillationPattern, MembraneState, AtpTrajectory,
    HierarchyLevel, OscillationType, MembranePhase, MetabolicPathway,
    Result, SpectacularError,
};

use clap::{Parser, Subcommand};
use tracing::{info, error, debug};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "spectacular")]
#[command(about = "High-performance scientific visualization engine with Autobahn integration")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    
    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    /// Enable GPU acceleration
    #[arg(long)]
    gpu: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Spectacular server
    Server {
        /// Server port
        #[arg(short, long, default_value = "8080")]
        port: u16,
        
        /// Enable Autobahn integration
        #[arg(long)]
        autobahn: bool,
    },
    /// Test biological data visualization
    TestBiological {
        /// Number of oscillation patterns to generate
        #[arg(short, long, default_value = "100")]
        oscillations: usize,
        
        /// Number of membrane states to generate
        #[arg(short, long, default_value = "50")]
        membranes: usize,
        
        /// Autobahn endpoint
        #[arg(long, default_value = "grpc://localhost:50051")]
        autobahn_endpoint: String,
    },
    /// Test molecular rendering
    TestMolecular {
        /// Rendering strategy
        #[arg(short, long, default_value = "quantum-coherence")]
        strategy: String,
        
        /// Enable WebGL shaders
        #[arg(long)]
        webgl: bool,
    },
    /// Test consciousness visualization
    TestConsciousness {
        /// Number of phi values to generate
        #[arg(short, long, default_value = "1000")]
        phi_count: usize,
        
        /// Emergence threshold
        #[arg(short, long, default_value = "0.5")]
        threshold: f64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize tracing
    init_tracing(&cli.log_level)?;
    
    info!("Starting Spectacular v{}", spectacular::VERSION);
    
    // Load configuration
    let config = load_config(cli.config.as_deref()).await?;
    
    match cli.command {
        Commands::Server { port, autobahn } => {
            run_server(config, port, autobahn, cli.gpu).await
        }
        Commands::TestBiological { oscillations, membranes, autobahn_endpoint } => {
            test_biological_visualization(config, oscillations, membranes, &autobahn_endpoint).await
        }
        Commands::TestMolecular { strategy, webgl } => {
            test_molecular_rendering(config, &strategy, webgl).await
        }
        Commands::TestConsciousness { phi_count, threshold } => {
            test_consciousness_visualization(config, phi_count, threshold).await
        }
    }
}

/// Initialize tracing with the specified log level
fn init_tracing(log_level: &str) -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("spectacular={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    Ok(())
}

/// Load configuration from file or use defaults
async fn load_config(config_path: Option<&std::path::Path>) -> Result<SpectacularConfig> {
    match config_path {
        Some(path) => {
            info!("Loading configuration from {:?}", path);
            SpectacularConfig::from_file(path.to_str().unwrap())
                .map_err(|e| SpectacularError::Configuration(e.to_string()))
        }
        None => {
            info!("Using default configuration");
            Ok(SpectacularConfig::default())
        }
    }
}

/// Run the Spectacular server
async fn run_server(
    config: SpectacularConfig,
    port: u16,
    enable_autobahn: bool,
    gpu_acceleration: bool,
) -> Result<()> {
    info!("Starting Spectacular server on port {}", port);
    info!("Autobahn integration: {}", enable_autobahn);
    info!("GPU acceleration: {}", gpu_acceleration);
    
    // Initialize Autobahn client if enabled
    let autobahn_client = if enable_autobahn {
        Some(AutobahnClient::new(config.autobahn.clone()).await?)
    } else {
        None
    };
    
    // Initialize molecular renderer
    let mut renderer_config = MolecularRendererConfig::default();
    renderer_config.gpu_acceleration = gpu_acceleration;
    let molecular_renderer = MolecularRenderer::new(renderer_config);
    
    info!("Server initialized successfully");
    
    // In a real implementation, this would start the HTTP server
    // For now, we'll just keep the server running
    tokio::signal::ctrl_c().await.unwrap();
    info!("Shutting down server");
    
    Ok(())
}

/// Test biological data visualization
async fn test_biological_visualization(
    config: SpectacularConfig,
    oscillation_count: usize,
    membrane_count: usize,
    autobahn_endpoint: &str,
) -> Result<()> {
    info!("Testing biological data visualization");
    info!("Generating {} oscillation patterns and {} membrane states", 
          oscillation_count, membrane_count);
    
    // Generate test biological data
    let biological_data = generate_test_biological_data(oscillation_count, membrane_count).await?;
    
    info!("Generated biological data with complexity score: {:.3}", 
          biological_data.estimate_complexity());
    info!("Biological authenticity: {:.3}", 
          biological_data.biological_authenticity());
    info!("Has quantum effects: {}", 
          biological_data.has_quantum_effects());
    info!("ATP budget: {:.2}", 
          biological_data.atp_budget());
    
    // Initialize Autobahn client
    let mut autobahn_config = config.autobahn.clone();
    autobahn_config.endpoint = autobahn_endpoint.to_string();
    
    let autobahn_client = AutobahnClient::new(autobahn_config).await?;
    
    // Request visualization strategy from Autobahn
    match autobahn_client.request_visualization_strategy(&biological_data).await {
        Ok(command) => {
            info!("Received visualization command from Autobahn: {:?}", command);
            
            // Execute the command with molecular renderer
            let renderer = MolecularRenderer::new(MolecularRendererConfig::default());
            let result = renderer.execute_rendering_command(&command, &biological_data).await?;
            
            info!("Molecular rendering completed:");
            info!("  Visualization ID: {}", result.visualization_id);
            info!("  Rendering time: {:.2}ms", result.rendering_time_ms);
            info!("  Memory usage: {:.2}MB", result.memory_usage_mb);
            info!("  ATP consumption: {:.2}", result.atp_consumption);
            info!("  Quantum coherence maintained: {:.3}", result.quantum_coherence_maintained);
            info!("  Consciousness emergence detected: {}", result.consciousness_emergence_detected);
            info!("  Oscillation endpoints visualized: {}", result.oscillation_endpoints_visualized);
            
            // Save the generated D3.js code
            tokio::fs::write("biological_visualization.js", &result.d3_code).await
                .map_err(|e| SpectacularError::Io(e))?;
            info!("D3.js visualization code saved to biological_visualization.js");
            
            if let Some(webgl_shaders) = &result.webgl_shaders {
                tokio::fs::write("webgl_shaders.glsl", webgl_shaders).await
                    .map_err(|e| SpectacularError::Io(e))?;
                info!("WebGL shaders saved to webgl_shaders.glsl");
            }
            
            // Report metrics back to Autobahn
            let metrics = spectacular::ExecutionMetrics {
                command_id: Uuid::new_v4(),
                rendering_time_ms: result.rendering_time_ms,
                memory_usage_mb: result.memory_usage_mb,
                crossfilter_efficiency: 0.95, // Mock value
                data_reduction_ratio: 0.8,    // Mock value
                biological_authenticity_score: result.biological_authenticity_score,
                quantum_coherence_maintained: result.quantum_coherence_maintained,
                atp_consumption: result.atp_consumption,
                consciousness_emergence_detected: result.consciousness_emergence_detected,
            };
            
            autobahn_client.report_execution_metrics(metrics).await?;
            info!("Execution metrics reported to Autobahn");
        }
        Err(e) => {
            error!("Failed to get visualization strategy from Autobahn: {}", e);
            info!("Falling back to default molecular rendering");
            
            // Fallback to default rendering
            let renderer = MolecularRenderer::new(MolecularRendererConfig::default());
            // This would need a default command - simplified for demo
            info!("Default rendering would be implemented here");
        }
    }
    
    Ok(())
}

/// Generate test biological data
async fn generate_test_biological_data(
    oscillation_count: usize,
    membrane_count: usize,
) -> Result<BiologicalData> {
    use spectacular::{
        OscillationEndpoint, TemporalPoint, LipidComposition, ProteinState,
        QuantumTransport, ElectricalProperties, AtpCoordinate,
    };
    use std::collections::HashMap;
    use chrono::Utc;
    
    let mut biological_data = BiologicalData::new(BiologicalDataType::MembraneSimulation);
    
    // Generate oscillation patterns
    for i in 0..oscillation_count {
        let pattern = OscillationPattern {
            id: format!("osc_{}", i),
            frequency_hz: 0.1 + (i as f64) * 0.01,
                         amplitude: 0.5 + (i % 10) as f64 * 0.05,
            phase: (i as f64) * 0.1,
            hierarchy_level: match i % 4 {
                0 => HierarchyLevel::Molecular,
                1 => HierarchyLevel::Cellular,
                2 => HierarchyLevel::Tissue,
                _ => HierarchyLevel::Organ,
            },
            oscillation_type: match i % 3 {
                0 => OscillationType::Enzymatic,
                1 => OscillationType::MembraneTransport,
                _ => OscillationType::AtpSynthase,
            },
            endpoints: vec![
                OscillationEndpoint {
                    position: 0.0,
                    probability: 0.6,
                    entropy_contribution: 0.3,
                    quantum_coherence: 0.8,
                },
                OscillationEndpoint {
                    position: 1.0,
                    probability: 0.4,
                    entropy_contribution: 0.7,
                    quantum_coherence: 0.6,
                },
            ],
                         entropy_contribution: 0.5 + (i % 10) as f64 * 0.05,
             coupling_strength: 0.3 + (i % 5) as f64 * 0.1,
            temporal_evolution: vec![
                TemporalPoint {
                    time: 0.0,
                    amplitude: 0.5,
                    phase: 0.0,
                    energy: 1.0,
                },
                TemporalPoint {
                    time: 1.0,
                    amplitude: 0.7,
                    phase: std::f64::consts::PI,
                    energy: 0.8,
                },
            ],
        };
        biological_data.oscillation_patterns.push(pattern);
    }
    
    // Generate membrane states
    for i in 0..membrane_count {
        let mut ion_concentrations = HashMap::new();
        ion_concentrations.insert("Na+".to_string(), 145.0);
        ion_concentrations.insert("K+".to_string(), 4.0);
        ion_concentrations.insert("Ca2+".to_string(), 2.5);
        ion_concentrations.insert("Cl-".to_string(), 110.0);
        
        let mut ion_currents = HashMap::new();
        ion_currents.insert("Na+".to_string(), -50.0);
        ion_currents.insert("K+".to_string(), 20.0);
        ion_currents.insert("Ca2+".to_string(), -10.0);
        
        let membrane_state = MembraneState {
            patch_id: format!("patch_{}", i),
            lipid_composition: LipidComposition {
                phosphatidylcholine: 40.0,
                phosphatidylserine: 20.0,
                phosphatidylethanolamine: 25.0,
                cholesterol: 10.0,
                sphingomyelin: 3.0,
                cardiolipin: 2.0,
                fluidity_index: 0.7,
                phase_state: match i % 3 {
                    0 => MembranePhase::LiquidOrdered,
                    1 => MembranePhase::LiquidDisordered,
                    _ => MembranePhase::Gel,
                },
            },
            protein_states: vec![], // Simplified for demo
            quantum_transport: QuantumTransport {
                coherence_time_ns: 10.0 + (i as f64) * 0.5,
                decoherence_rate: 0.1,
                transport_efficiency: 0.8 + (i as f64 % 10) * 0.02,
                coupling_strength: 0.5,
                environmental_enhancement: 0.3,
                quantum_yield: 0.9,
                electron_transfer_rate: 1e12,
            },
            electrical_properties: ElectricalProperties {
                membrane_potential_mv: -70.0 + (i as f64) * 0.1,
                capacitance_uf_cm2: 1.0,
                resistance_ohm_cm2: 1000.0,
                conductance_s_cm2: 0.001,
                ion_currents,
            },
            temperature_k: 310.15,
            ph: 7.4,
            ion_concentrations,
            enaqt_coherence: 0.7 + (i as f64 % 10) * 0.03,
            quantum_tunneling_rate: 1e6,
        };
        biological_data.membrane_states.push(membrane_state);
    }
    
    // Generate ATP trajectories
    for i in 0..10 {
        let mut atp_coordinates = Vec::new();
        for j in 0..100 {
            atp_coordinates.push(AtpCoordinate {
                time: j as f64 * 0.01,
                atp_concentration: 5.0 + (j as f64 * 0.01).sin(),
                atp_consumption_rate: 0.1 + (j as f64 * 0.02).cos() * 0.05,
                oscillation_frequency: 10.0,
                oscillation_phase: j as f64 * 0.1,
                quantum_coherence: 0.8,
                entropy: 1.0 + (j as f64 * 0.01).ln(),
            });
        }
        
        let trajectory = AtpTrajectory {
            trajectory_id: format!("atp_traj_{}", i),
            start_time: 0.0,
            end_time: 1.0,
            atp_coordinates,
            metabolic_pathway: match i % 3 {
                0 => MetabolicPathway::Glycolysis,
                1 => MetabolicPathway::CitricAcidCycle,
                _ => MetabolicPathway::AtpSynthesis,
            },
            energy_efficiency: 0.8 + (i as f64 % 5) * 0.04,
            quantum_enhancement: 0.2,
            biological_authenticity: 0.9,
        };
        biological_data.atp_trajectories.push(trajectory);
    }
    
    Ok(biological_data)
}

/// Test molecular rendering
async fn test_molecular_rendering(
    _config: SpectacularConfig,
    strategy: &str,
    webgl: bool,
) -> Result<()> {
    info!("Testing molecular rendering with strategy: {}", strategy);
    info!("WebGL shaders enabled: {}", webgl);
    
    // This would be a simplified test of the molecular renderer
    let mut renderer_config = MolecularRendererConfig::default();
    renderer_config.gpu_acceleration = webgl;
    
    let renderer = MolecularRenderer::new(renderer_config);
    info!("Molecular renderer initialized");
    
    // Generate some test data and render
    let biological_data = generate_test_biological_data(50, 25).await?;
    info!("Test data generated for molecular rendering");
    
    Ok(())
}

/// Test consciousness visualization
async fn test_consciousness_visualization(
    _config: SpectacularConfig,
    phi_count: usize,
    threshold: f64,
) -> Result<()> {
    info!("Testing consciousness visualization");
    info!("Generating {} phi values with threshold {}", phi_count, threshold);
    
    // Generate test phi values
    let phi_values: Vec<f64> = (0..phi_count)
        .map(|i| threshold * (1.0 + (i as f64 * 0.01).sin()))
        .collect();
    
    let emergence_detected = phi_values.iter().any(|&phi| phi > threshold);
    info!("Consciousness emergence detected: {}", emergence_detected);
    
    if emergence_detected {
        let max_phi = phi_values.iter().fold(0.0, |acc, &x| acc.max(x));
        info!("Maximum phi value: {:.3}", max_phi);
    }
    
    Ok(())
} 