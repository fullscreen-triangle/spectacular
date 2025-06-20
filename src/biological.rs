//! Biological Data Structures
//!
//! This module defines the specialized data structures for handling
//! quantum-biological simulation data from Bene Gesserit membrane
//! simulations and Nebuchadnezzar circuit systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Main biological data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalData {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub data_type: BiologicalDataType,
    pub oscillation_patterns: Vec<OscillationPattern>,
    pub membrane_states: Vec<MembraneState>,
    pub atp_trajectories: Vec<AtpTrajectory>,
    pub quantum_coherence_data: Option<QuantumCoherenceData>,
    pub consciousness_metrics: Option<ConsciousnessMetrics>,
    pub circuit_states: Option<Vec<CircuitState>>,
    pub metadata: BiologicalMetadata,
}

/// Types of biological data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiologicalDataType {
    MembraneSimulation,
    CircuitIntegration,
    ConsciousnessEmergence,
    AtpMetabolism,
    QuantumTransport,
    OscillatoryDynamics,
    FireCircleCommunication,
}

/// Oscillation patterns from universal oscillatory dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationPattern {
    pub id: String,
    pub frequency_hz: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub hierarchy_level: HierarchyLevel,
    pub oscillation_type: OscillationType,
    pub endpoints: Vec<OscillationEndpoint>,
    pub entropy_contribution: f64,
    pub coupling_strength: f64,
    pub temporal_evolution: Vec<TemporalPoint>,
}

/// Hierarchy levels for oscillatory dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchyLevel {
    Planck,      // 10^-44 s
    Quantum,     // 10^-21 s
    Atomic,      // 10^-18 s
    Molecular,   // 10^-12 s
    Cellular,    // 10^-6 s
    Tissue,      // 10^-3 s
    Organ,       // 1 s
    Organism,    // 10^3 s
    Ecosystem,   // 10^6 s
    Cosmic,      // 10^13 s
}

/// Types of oscillations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OscillationType {
    Enzymatic,
    MembraneTransport,
    IonChannel,
    AtpSynthase,
    ProteinConformational,
    Metabolic,
    Neural,
    Circadian,
    Consciousness,
}

/// Oscillation endpoint for entropy calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationEndpoint {
    pub position: f64,
    pub probability: f64,
    pub entropy_contribution: f64,
    pub quantum_coherence: f64,
}

/// Temporal evolution point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPoint {
    pub time: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub energy: f64,
}

/// Membrane quantum states from Bene Gesserit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembraneState {
    pub patch_id: String,
    pub lipid_composition: LipidComposition,
    pub protein_states: Vec<ProteinState>,
    pub quantum_transport: QuantumTransport,
    pub electrical_properties: ElectricalProperties,
    pub temperature_k: f64,
    pub ph: f64,
    pub ion_concentrations: HashMap<String, f64>,
    pub enaqt_coherence: f64,
    pub quantum_tunneling_rate: f64,
}

/// Lipid bilayer composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipidComposition {
    pub phosphatidylcholine: f64,
    pub phosphatidylserine: f64,
    pub phosphatidylethanolamine: f64,
    pub cholesterol: f64,
    pub sphingomyelin: f64,
    pub cardiolipin: f64,
    pub fluidity_index: f64,
    pub phase_state: MembranePhase,
}

/// Membrane phase states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MembranePhase {
    Gel,
    LiquidOrdered,
    LiquidDisordered,
    RipplePhase,
    Transition,
}

/// Protein conformational states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinState {
    pub protein_id: String,
    pub conformation: ProteinConformation,
    pub activity_level: f64,
    pub binding_sites: Vec<BindingSite>,
    pub quantum_effects: bool,
    pub allosteric_state: AllostericState,
}

/// Protein conformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProteinConformation {
    Native,
    Intermediate,
    Denatured,
    Molten,
    Aggregated,
}

/// Binding sites on proteins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingSite {
    pub site_id: String,
    pub ligand: Option<String>,
    pub binding_affinity: f64,
    pub occupancy: f64,
}

/// Allosteric states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllostericState {
    Relaxed,
    Tense,
    Intermediate,
}

/// Environment-Assisted Quantum Transport data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTransport {
    pub coherence_time_ns: f64,
    pub decoherence_rate: f64,
    pub transport_efficiency: f64,
    pub coupling_strength: f64,
    pub environmental_enhancement: f64,
    pub quantum_yield: f64,
    pub electron_transfer_rate: f64,
}

/// Electrical properties of membranes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricalProperties {
    pub membrane_potential_mv: f64,
    pub capacitance_uf_cm2: f64,
    pub resistance_ohm_cm2: f64,
    pub conductance_s_cm2: f64,
    pub ion_currents: HashMap<String, f64>,
}

/// ATP-constrained trajectories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpTrajectory {
    pub trajectory_id: String,
    pub start_time: f64,
    pub end_time: f64,
    pub atp_coordinates: Vec<AtpCoordinate>,
    pub metabolic_pathway: MetabolicPathway,
    pub energy_efficiency: f64,
    pub quantum_enhancement: f64,
    pub biological_authenticity: f64,
}

/// ATP coordinate in dx/dATP space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpCoordinate {
    pub time: f64,
    pub atp_concentration: f64,
    pub atp_consumption_rate: f64,
    pub oscillation_frequency: f64,
    pub oscillation_phase: f64,
    pub quantum_coherence: f64,
    pub entropy: f64,
}

/// Metabolic pathways
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetabolicPathway {
    Glycolysis,
    CitricAcidCycle,
    ElectronTransport,
    AtpSynthesis,
    Phosphorylation,
    Custom(String),
}

/// Quantum coherence data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCoherenceData {
    pub coherence_matrix: Vec<Vec<f64>>,
    pub entanglement_measures: Vec<f64>,
    pub decoherence_times: Vec<f64>,
    pub quantum_correlations: Vec<QuantumCorrelation>,
    pub ion_channel_coherence: HashMap<String, f64>,
    pub fire_light_coupling: Option<FireLightCoupling>,
}

/// Quantum correlations between system components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCorrelation {
    pub component_a: String,
    pub component_b: String,
    pub correlation_strength: f64,
    pub correlation_type: CorrelationType,
}

/// Types of quantum correlations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    Entanglement,
    Discord,
    Coherence,
    Superposition,
}

/// Fire-light quantum coupling (650nm optimization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireLightCoupling {
    pub wavelength_nm: f64,
    pub coupling_efficiency: f64,
    pub consciousness_enhancement: f64,
    pub ion_channel_resonance: HashMap<String, f64>,
}

/// Consciousness emergence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    pub phi_value: f64,
    pub integrated_information: f64,
    pub global_workspace_activity: f64,
    pub self_awareness_level: f64,
    pub metacognitive_activity: f64,
    pub agency_illusion_strength: f64,
    pub persistence_illusion_strength: f64,
    pub frame_selection_patterns: Vec<FrameSelection>,
    pub bmds_active: usize,
}

/// Frame selection by Biological Maxwell Demons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameSelection {
    pub frame_id: String,
    pub selection_probability: f64,
    pub cognitive_load: f64,
    pub associative_strength: f64,
    pub emotional_weighting: f64,
}

/// Circuit states from Nebuchadnezzar integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitState {
    pub circuit_id: String,
    pub topology: CircuitTopology,
    pub quantum_classical_interface: QuantumClassicalInterface,
    pub temporal_coordinates: TemporalCoordinates,
    pub predetermined_states: Vec<PredeterminedState>,
    pub entropy_optimization: f64,
}

/// Circuit topology based on membrane dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitTopology {
    pub nodes: Vec<CircuitNode>,
    pub connections: Vec<CircuitConnection>,
    pub hierarchical_level: u32,
    pub oscillation_coupling: f64,
}

/// Circuit nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitNode {
    pub node_id: String,
    pub node_type: CircuitNodeType,
    pub activation_level: f64,
    pub quantum_state: Option<Vec<f64>>,
    pub biological_origin: String,
}

/// Types of circuit nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitNodeType {
    MembraneProtein,
    IonChannel,
    AtpSynthase,
    Enzyme,
    Receptor,
    Transporter,
    QuantumDot,
}

/// Circuit connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConnection {
    pub from_node: String,
    pub to_node: String,
    pub connection_strength: f64,
    pub connection_type: ConnectionType,
    pub quantum_entanglement: bool,
}

/// Types of circuit connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Electrical,
    Chemical,
    Quantum,
    Oscillatory,
    Metabolic,
}

/// Quantum-classical interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumClassicalInterface {
    pub interface_efficiency: f64,
    pub decoherence_management: f64,
    pub measurement_protocol: MeasurementProtocol,
    pub classical_parameters: HashMap<String, f64>,
}

/// Quantum measurement protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementProtocol {
    VonNeumann,
    Weak,
    Continuous,
    Quantum,
}

/// Temporal coordinates for predetermined navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCoordinates {
    pub current_state: Vec<f64>,
    pub target_state: Vec<f64>,
    pub convergence_probability: f64,
    pub navigation_efficiency: f64,
    pub predetermined_path: Vec<TemporalWaypoint>,
}

/// Waypoints in predetermined temporal navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalWaypoint {
    pub time_coordinate: f64,
    pub state_vector: Vec<f64>,
    pub probability: f64,
    pub entropy: f64,
}

/// Predetermined optimal states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredeterminedState {
    pub state_id: String,
    pub optimality_score: f64,
    pub thermodynamic_efficiency: f64,
    pub consciousness_compatibility: f64,
    pub biological_authenticity: f64,
}

/// Metadata for biological data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalMetadata {
    pub simulation_parameters: SimulationParameters,
    pub data_quality: DataQuality,
    pub biological_authenticity: f64,
    pub quantum_fidelity: f64,
    pub computational_cost: ComputationalCost,
    pub validation_status: ValidationStatus,
}

/// Simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub ph: f64,
    pub ionic_strength: f64,
    pub simulation_time_ns: f64,
    pub time_step_fs: f64,
    pub ensemble: EnsembleType,
}

/// Ensemble types for simulations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleType {
    Nve,  // Constant number, volume, energy
    Nvt,  // Constant number, volume, temperature
    Npt,  // Constant number, pressure, temperature
    Nph,  // Constant number, pressure, enthalpy
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    pub completeness: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub consistency: f64,
    pub temporal_resolution: f64,
    pub spatial_resolution: f64,
}

/// Computational cost tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalCost {
    pub cpu_hours: f64,
    pub memory_gb_hours: f64,
    pub gpu_hours: Option<f64>,
    pub atp_equivalent: f64,  // Metabolic cost equivalent
    pub energy_efficiency: f64,
}

/// Validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatus {
    pub experimental_validation: bool,
    pub theoretical_consistency: bool,
    pub biological_plausibility: f64,
    pub quantum_mechanical_validity: bool,
    pub thermodynamic_consistency: bool,
}

impl BiologicalData {
    /// Create new biological data container
    pub fn new(data_type: BiologicalDataType) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            data_type,
            oscillation_patterns: Vec::new(),
            membrane_states: Vec::new(),
            atp_trajectories: Vec::new(),
            quantum_coherence_data: None,
            consciousness_metrics: None,
            circuit_states: None,
            metadata: BiologicalMetadata::default(),
        }
    }
    
    /// Estimate data complexity for visualization planning
    pub fn estimate_complexity(&self) -> f64 {
        let oscillation_complexity = self.oscillation_patterns.len() as f64 * 0.1;
        let membrane_complexity = self.membrane_states.len() as f64 * 0.2;
        let atp_complexity = self.atp_trajectories.len() as f64 * 0.15;
        let quantum_complexity = if self.quantum_coherence_data.is_some() { 0.3 } else { 0.0 };
        let consciousness_complexity = if self.consciousness_metrics.is_some() { 0.4 } else { 0.0 };
        let circuit_complexity = self.circuit_states.as_ref()
            .map(|states| states.len() as f64 * 0.25)
            .unwrap_or(0.0);
        
        (oscillation_complexity + membrane_complexity + atp_complexity + 
         quantum_complexity + consciousness_complexity + circuit_complexity).min(1.0)
    }
    
    /// Get biological authenticity score
    pub fn biological_authenticity(&self) -> f64 {
        self.metadata.biological_authenticity
    }
    
    /// Check if quantum effects are present
    pub fn has_quantum_effects(&self) -> bool {
        self.quantum_coherence_data.is_some() || 
        self.membrane_states.iter().any(|state| state.quantum_transport.coherence_time_ns > 0.0)
    }
    
    /// Get ATP budget for visualization
    pub fn atp_budget(&self) -> f64 {
        self.atp_trajectories.iter()
            .map(|traj| traj.atp_coordinates.iter()
                .map(|coord| coord.atp_concentration)
                .fold(0.0, f64::max))
            .fold(0.0, f64::max)
    }
}

impl Default for BiologicalMetadata {
    fn default() -> Self {
        Self {
            simulation_parameters: SimulationParameters {
                temperature_k: 310.15,  // 37°C
                pressure_pa: 101325.0,  // 1 atm
                ph: 7.4,
                ionic_strength: 0.15,   // Physiological
                simulation_time_ns: 1000.0,
                time_step_fs: 2.0,
                ensemble: EnsembleType::Npt,
            },
            data_quality: DataQuality {
                completeness: 1.0,
                accuracy: 0.95,
                precision: 0.98,
                consistency: 0.97,
                temporal_resolution: 1e-12,  // ps
                spatial_resolution: 1e-10,   // Angstrom
            },
            biological_authenticity: 0.9,
            quantum_fidelity: 0.85,
            computational_cost: ComputationalCost {
                cpu_hours: 0.0,
                memory_gb_hours: 0.0,
                gpu_hours: None,
                atp_equivalent: 0.0,
                energy_efficiency: 0.8,
            },
            validation_status: ValidationStatus {
                experimental_validation: false,
                theoretical_consistency: true,
                biological_plausibility: 0.9,
                quantum_mechanical_validity: true,
                thermodynamic_consistency: true,
            },
        }
    }
} 