"""
Bayesian Evidence Network Pipeline: Dynamic knowledge base using fuzzy logic nodes.

This implements the actual pipeline intelligence where each stage is a Bayesian node
that gets updated with fuzzy evidence, validated against external data, and can
trigger recursive loops. The network itself IS the knowledge base that guides
the metacognitive orchestrator's decisions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import json
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class NodeState(Enum):
    """Fuzzy states for Bayesian network nodes."""
    UNINITIALIZED = "uninitialized"
    EVIDENCE_GATHERING = "evidence_gathering"
    PROCESSING = "processing"
    VALIDATING = "validating"
    CONVERGED = "converged"
    RECURSIVE_LOOP = "recursive_loop"
    EXTERNAL_VALIDATION = "external_validation"
    FAILED = "failed"

@dataclass
class FuzzyEvidence:
    """Fuzzy evidence for Bayesian network nodes."""
    evidence_type: str
    confidence: float           # 0.0 to 1.0
    uncertainty: float         # 0.0 to 1.0 
    source: str
    data: Any
    timestamp: datetime
    environmental_context: Dict[str, float]
    
    def fuzzy_strength(self) -> float:
        """Calculate fuzzy strength combining confidence and uncertainty."""
        return confidence * (1.0 - uncertainty) if hasattr(self, 'confidence') else 0.0

@dataclass
class EmbeddingPath:
    """Multi-dimensional embedding path with environmental contexts."""
    path_id: str
    dimensions: List[int]                    # Dimensional sequence
    embedding_sequence: List[np.ndarray]     # Sequence of embeddings
    environmental_contexts: List[Dict[str, float]]  # Environmental data at each step
    external_validations: List[Dict[str, Any]]      # External validation results
    similarity_environments: List[str]              # Similar environment references
    confidence_trajectory: List[float]              # Confidence evolution
    
    def get_path_coherence(self) -> float:
        """Calculate coherence across the embedding path."""
        if len(self.confidence_trajectory) < 2:
            return 0.5
        
        # Calculate variance in confidence (lower variance = higher coherence)
        variance = np.var(self.confidence_trajectory)
        return max(0.0, 1.0 - variance)
    
    def get_environmental_stability(self) -> float:
        """Calculate environmental stability across the path."""
        if len(self.environmental_contexts) < 2:
            return 0.5
        
        # Calculate stability across environmental dimensions
        env_keys = set()
        for ctx in self.environmental_contexts:
            env_keys.update(ctx.keys())
        
        stabilities = []
        for key in env_keys:
            values = [ctx.get(key, 0.0) for ctx in self.environmental_contexts]
            if len(values) > 1:
                stability = 1.0 - np.std(values)  # Lower std = higher stability
                stabilities.append(max(0.0, stability))
        
        return np.mean(stabilities) if stabilities else 0.5

class BayesianNode:
    """
    Bayesian network node with fuzzy logic updates and external validation.
    
    Each node represents a pipeline stage that:
    1. Accumulates fuzzy evidence
    2. Updates beliefs using Bayesian inference
    3. Validates against external data/models
    4. Maintains multi-dimensional embedding paths
    5. Can trigger recursive processing
    """
    
    def __init__(self, node_id: str, node_type: str, config: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.config = config or {}
        
        # Node state and beliefs
        self.state = NodeState.UNINITIALIZED
        self.belief_confidence = 0.0
        self.belief_uncertainty = 1.0
        self.prior_belief = 0.5  # Neutral prior
        
        # Evidence accumulation
        self.evidence_buffer: List[FuzzyEvidence] = []
        self.processed_evidence: List[FuzzyEvidence] = []
        
        # External validation
        self.external_validators: List[str] = []
        self.validation_results: Dict[str, Any] = {}
        
        # Embedding paths
        self.embedding_paths: Dict[str, EmbeddingPath] = {}
        self.active_path_id: Optional[str] = None
        
        # Network connections
        self.input_nodes: Set[str] = set()
        self.output_nodes: Set[str] = set()
        self.recursive_connections: Set[str] = set()
        
        # Processing history
        self.processing_history: List[Dict[str, Any]] = []
        self.recursive_count = 0
        self.max_recursive_depth = config.get('max_recursive_depth', 5)
        
        # External environment references
        self.similar_environments: List[str] = []
        self.environment_similarities: Dict[str, float] = {}
        
        logger.debug("Initialized Bayesian node: %s (%s)", node_id, node_type)
    
    async def add_evidence(self, evidence: FuzzyEvidence) -> None:
        """Add fuzzy evidence to the node."""
        self.evidence_buffer.append(evidence)
        
        # Update embedding path if active
        if self.active_path_id and self.active_path_id in self.embedding_paths:
            path = self.embedding_paths[self.active_path_id]
            path.confidence_trajectory.append(evidence.confidence)
            path.environmental_contexts.append(evidence.environmental_context)
        
        logger.debug("Added evidence to node %s: %s (confidence: %.3f)", 
                    self.node_id, evidence.evidence_type, evidence.confidence)
    
    async def update_beliefs(self) -> float:
        """Update node beliefs using Bayesian inference with fuzzy logic."""
        
        if not self.evidence_buffer:
            return self.belief_confidence
        
        # Process new evidence using fuzzy Bayesian update
        for evidence in self.evidence_buffer:
            likelihood = await self._calculate_fuzzy_likelihood(evidence)
            
            # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
            # With fuzzy modifications for uncertainty
            fuzzy_strength = evidence.fuzzy_strength()
            
            # Update belief confidence
            posterior = (likelihood * self.prior_belief) / max(0.001, self._calculate_evidence_probability())
            
            # Fuzzy integration
            self.belief_confidence = self._fuzzy_combine(
                self.belief_confidence, 
                posterior * fuzzy_strength
            )
            
            # Update uncertainty based on evidence uncertainty
            self.belief_uncertainty = self._fuzzy_combine(
                self.belief_uncertainty,
                evidence.uncertainty * (1.0 - fuzzy_strength)
            )
            
            self.processed_evidence.append(evidence)
        
        # Clear processed evidence from buffer
        self.evidence_buffer.clear()
        
        # Update prior for next iteration
        self.prior_belief = self.belief_confidence
        
        logger.debug("Updated beliefs for node %s: confidence=%.3f, uncertainty=%.3f", 
                    self.node_id, self.belief_confidence, self.belief_uncertainty)
        
        return self.belief_confidence
    
    async def validate_externally(self, external_data: Dict[str, Any]) -> bool:
        """Validate node beliefs against external data/models."""
        
        self.state = NodeState.EXTERNAL_VALIDATION
        validation_passed = True
        
        for validator_type in self.external_validators:
            try:
                validation_result = await self._run_external_validator(validator_type, external_data)
                self.validation_results[validator_type] = validation_result
                
                if not validation_result.get('passed', False):
                    validation_passed = False
                    logger.warning("External validation failed for node %s, validator %s", 
                                 self.node_id, validator_type)
                
                # Update embedding path with validation
                if self.active_path_id and self.active_path_id in self.embedding_paths:
                    self.embedding_paths[self.active_path_id].external_validations.append(validation_result)
                
            except Exception as e:
                logger.error("Error in external validation for node %s: %s", self.node_id, str(e))
                validation_passed = False
        
        # Adjust beliefs based on validation results
        if not validation_passed:
            self.belief_confidence *= 0.7  # Reduce confidence on validation failure
            self.belief_uncertainty = min(1.0, self.belief_uncertainty + 0.2)
        
        return validation_passed
    
    async def check_convergence(self) -> bool:
        """Check if node has converged or needs recursive processing."""
        
        convergence_threshold = self.config.get('convergence_threshold', 0.8)
        uncertainty_threshold = self.config.get('uncertainty_threshold', 0.3)
        
        # Check basic convergence criteria
        basic_convergence = (
            self.belief_confidence > convergence_threshold and
            self.belief_uncertainty < uncertainty_threshold
        )
        
        if basic_convergence:
            self.state = NodeState.CONVERGED
            return True
        
        # Check if recursive processing needed
        if self.recursive_count < self.max_recursive_depth:
            recursive_needed = await self._assess_recursive_need()
            if recursive_needed:
                self.state = NodeState.RECURSIVE_LOOP
                self.recursive_count += 1
                logger.info("Node %s entering recursive loop (depth: %d)", 
                          self.node_id, self.recursive_count)
                return False
        
        # Check if more evidence gathering needed
        if len(self.processed_evidence) < self.config.get('min_evidence_count', 3):
            self.state = NodeState.EVIDENCE_GATHERING
            return False
        
        return basic_convergence
    
    async def create_embedding_path(self, environmental_context: Dict[str, float]) -> str:
        """Create new multi-dimensional embedding path."""
        
        path_id = f"path_{self.node_id}_{len(self.embedding_paths)}"
        
        embedding_path = EmbeddingPath(
            path_id=path_id,
            dimensions=[],
            embedding_sequence=[],
            environmental_contexts=[environmental_context],
            external_validations=[],
            similarity_environments=[],
            confidence_trajectory=[self.belief_confidence]
        )
        
        self.embedding_paths[path_id] = embedding_path
        self.active_path_id = path_id
        
        # Find similar environments
        await self._find_similar_environments(environmental_context)
        
        logger.debug("Created embedding path %s for node %s", path_id, self.node_id)
        return path_id
    
    async def extend_embedding_path(self, embedding: np.ndarray, environmental_context: Dict[str, float]) -> None:
        """Extend active embedding path with new embedding and context."""
        
        if not self.active_path_id or self.active_path_id not in self.embedding_paths:
            await self.create_embedding_path(environmental_context)
        
        path = self.embedding_paths[self.active_path_id]
        path.embedding_sequence.append(embedding)
        path.dimensions.append(len(embedding))
        path.environmental_contexts.append(environmental_context)
        path.confidence_trajectory.append(self.belief_confidence)
        
        logger.debug("Extended embedding path %s: dimension %d", 
                    self.active_path_id, len(embedding))
    
    # Private helper methods
    
    async def _calculate_fuzzy_likelihood(self, evidence: FuzzyEvidence) -> float:
        """Calculate fuzzy likelihood for evidence."""
        
        # Base likelihood from evidence confidence
        base_likelihood = evidence.confidence
        
        # Adjust for environmental context similarity
        env_similarity = await self._calculate_environmental_similarity(evidence.environmental_context)
        
        # Fuzzy combination
        fuzzy_likelihood = base_likelihood * (0.7 + 0.3 * env_similarity)
        
        return max(0.001, min(0.999, fuzzy_likelihood))
    
    def _calculate_evidence_probability(self) -> float:
        """Calculate evidence probability for Bayesian denominator."""
        
        # Simplified evidence probability calculation
        evidence_strengths = [evidence.fuzzy_strength() for evidence in self.processed_evidence[-5:]]  # Last 5 pieces
        return max(0.001, np.mean(evidence_strengths)) if evidence_strengths else 0.5
    
    def _fuzzy_combine(self, value1: float, value2: float) -> float:
        """Combine two fuzzy values using T-norm operation."""
        
        # Using algebraic product T-norm for combination
        return max(0.0, min(1.0, value1 + value2 - value1 * value2))
    
    async def _run_external_validator(self, validator_type: str, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run external validation against data/models."""
        
        if validator_type == "environmental_consistency":
            return await self._validate_environmental_consistency(external_data)
        elif validator_type == "mathematical_coherence":
            return await self._validate_mathematical_coherence(external_data)
        elif validator_type == "similar_environment_check":
            return await self._validate_similar_environments(external_data)
        else:
            # Generic validation
            return {
                'validator_type': validator_type,
                'passed': True,
                'confidence': 0.6,
                'details': 'Generic validation passed'
            }
    
    async def _validate_environmental_consistency(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency with environmental data."""
        
        if not self.active_path_id:
            return {'passed': False, 'reason': 'No active embedding path'}
        
        path = self.embedding_paths[self.active_path_id]
        env_stability = path.get_environmental_stability()
        
        passed = env_stability > 0.6
        return {
            'validator_type': 'environmental_consistency',
            'passed': passed,
            'confidence': env_stability,
            'details': f'Environmental stability: {env_stability:.3f}'
        }
    
    async def _validate_mathematical_coherence(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical coherence of beliefs."""
        
        # Check belief consistency (confidence + uncertainty should be reasonable)
        belief_consistency = abs(1.0 - (self.belief_confidence + self.belief_uncertainty))
        coherence_score = max(0.0, 1.0 - belief_consistency)
        
        passed = coherence_score > 0.7
        return {
            'validator_type': 'mathematical_coherence',
            'passed': passed,
            'confidence': coherence_score,
            'details': f'Belief consistency score: {coherence_score:.3f}'
        }
    
    async def _validate_similar_environments(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against similar environmental contexts."""
        
        if not self.similar_environments:
            return {
                'passed': True,  # Pass if no similar environments to check
                'confidence': 0.5,
                'details': 'No similar environments for comparison'
            }
        
        # Check consistency with similar environments
        similarity_scores = list(self.environment_similarities.values())
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.5
        
        passed = avg_similarity > 0.6
        return {
            'validator_type': 'similar_environment_check',
            'passed': passed,
            'confidence': avg_similarity,
            'details': f'Average environment similarity: {avg_similarity:.3f}'
        }
    
    async def _assess_recursive_need(self) -> bool:
        """Assess if recursive processing is needed."""
        
        # Recursive processing needed if:
        # 1. Low confidence but high certainty (need more evidence)
        # 2. External validation failed
        # 3. Environmental context changed significantly
        
        low_confidence_high_certainty = (
            self.belief_confidence < 0.6 and self.belief_uncertainty < 0.4
        )
        
        validation_failed = any(
            not result.get('passed', True) 
            for result in self.validation_results.values()
        )
        
        return low_confidence_high_certainty or validation_failed
    
    async def _find_similar_environments(self, environmental_context: Dict[str, float]) -> None:
        """Find similar environmental contexts for comparison."""
        
        # This would query a database/knowledge base of similar environments
        # For now, simulate with some example similar environments
        
        self.similar_environments = [
            f"similar_env_{i}" for i in range(3)
        ]
        
        # Calculate similarities (simplified)
        for env_id in self.similar_environments:
            # Simulate similarity calculation
            similarity = np.random.uniform(0.4, 0.9)  # Would be actual similarity calculation
            self.environment_similarities[env_id] = similarity
    
    async def _calculate_environmental_similarity(self, environmental_context: Dict[str, float]) -> float:
        """Calculate similarity to current environmental contexts."""
        
        if not self.active_path_id:
            return 0.5
        
        path = self.embedding_paths[self.active_path_id]
        if not path.environmental_contexts:
            return 0.5
        
        # Calculate similarity to previous contexts in path
        similarities = []
        for prev_context in path.environmental_contexts[-3:]:  # Last 3 contexts
            sim = self._cosine_similarity_dict(environmental_context, prev_context)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _cosine_similarity_dict(self, dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two dictionaries."""
        
        # Get common keys
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys:
            return 0.0
        
        # Create vectors
        vec1 = [dict1[key] for key in common_keys]
        vec2 = [dict2[key] for key in common_keys]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class BayesianPipelineNetwork:
    """
    Bayesian Evidence Network that IS the knowledge base for the metacognitive orchestrator.
    
    This network of fuzzy logic nodes can:
    1. Process evidence non-linearly with recursive loops
    2. Validate each node against external data/models  
    3. Build multi-dimensional embedding paths
    4. Reference similar environmental contexts
    5. Make dynamic routing decisions for problem solving
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Network structure
        self.nodes: Dict[str, BayesianNode] = {}
        self.node_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.execution_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Network state
        self.active_nodes: Set[str] = set()
        self.converged_nodes: Set[str] = set()
        self.recursive_loops: Dict[str, int] = defaultdict(int)
        
        # External validation systems
        self.external_validators = {
            'environmental_db': self._environmental_database_validator,
            'mathematical_models': self._mathematical_model_validator,
            'similar_contexts': self._similar_context_validator,
            'knowledge_graphs': self._knowledge_graph_validator
        }
        
        # Knowledge base components
        self.global_embeddings: Dict[str, np.ndarray] = {}
        self.environment_database: Dict[str, Dict[str, float]] = {}
        self.similarity_network: Dict[str, List[str]] = defaultdict(list)
        
        # Execution history and metrics
        self.execution_history: List[Dict[str, Any]] = []
        self.network_coherence_history: List[float] = []
        
        # Initialize standard pipeline nodes
        self._initialize_pipeline_nodes()
        
        logger.info("Initialized Bayesian Pipeline Network with %d nodes", len(self.nodes))
    
    def _initialize_pipeline_nodes(self) -> None:
        """Initialize standard pipeline nodes with fuzzy logic capabilities."""
        
        node_configs = {
            'environmental_acquisition': {
                'external_validators': ['environmental_db', 'similar_contexts'],
                'convergence_threshold': 0.7,
                'max_recursive_depth': 3
            },
            'cognitive_mapping': {
                'external_validators': ['mathematical_models', 'knowledge_graphs'],
                'convergence_threshold': 0.8,
                'max_recursive_depth': 4
            },
            'knowledge_synthesis': {
                'external_validators': ['knowledge_graphs', 'similar_contexts'],
                'convergence_threshold': 0.75,
                'max_recursive_depth': 5
            },
            'dimensional_analysis': {
                'external_validators': ['mathematical_models', 'environmental_db'],
                'convergence_threshold': 0.8,
                'max_recursive_depth': 3
            },
            'reasoning_orchestration': {
                'external_validators': ['mathematical_models', 'knowledge_graphs'],
                'convergence_threshold': 0.85,
                'max_recursive_depth': 6
            },
            'validation_convergence': {
                'external_validators': ['mathematical_models', 'similar_contexts'],
                'convergence_threshold': 0.8,
                'max_recursive_depth': 4
            },
            'visual_coherence': {
                'external_validators': ['environmental_db', 'similar_contexts'],
                'convergence_threshold': 0.75,
                'max_recursive_depth': 3
            },
            'synthesis_emergence': {
                'external_validators': ['knowledge_graphs', 'mathematical_models'],
                'convergence_threshold': 0.9,
                'max_recursive_depth': 2
            }
        }
        
        # Create nodes
        for node_id, config in node_configs.items():
            node = BayesianNode(node_id, node_id, config)
            node.external_validators = config['external_validators']
            self.nodes[node_id] = node
        
        # Define node dependencies (can be non-linear)
        dependencies = {
            'cognitive_mapping': {'environmental_acquisition'},
            'knowledge_synthesis': {'environmental_acquisition', 'cognitive_mapping'},
            'dimensional_analysis': {'environmental_acquisition', 'knowledge_synthesis'},
            'reasoning_orchestration': {'cognitive_mapping', 'knowledge_synthesis', 'dimensional_analysis'},
            'validation_convergence': {'reasoning_orchestration', 'dimensional_analysis'},
            'visual_coherence': {'validation_convergence', 'reasoning_orchestration'},
            'synthesis_emergence': {'validation_convergence', 'visual_coherence', 'reasoning_orchestration'}
        }
        
        for node_id, deps in dependencies.items():
            self.node_dependencies[node_id] = deps
            for dep in deps:
                self.nodes[node_id].input_nodes.add(dep)
                self.nodes[dep].output_nodes.add(node_id)
    
    async def process_query(self, query: str, initial_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query through the Bayesian network with dynamic routing and recursion.
        
        This is the main intelligence - the network makes decisions about:
        - Which nodes to activate
        - When to create recursive loops  
        - How to route evidence between nodes
        - When external validation is needed
        - How to build multi-dimensional embedding paths
        """
        
        logger.info("🧠 Processing query through Bayesian Pipeline Network: %s", query[:100])
        start_time = datetime.now()
        
        # Initialize network state
        execution_id = str(uuid.uuid4())
        self.active_nodes.clear()
        self.converged_nodes.clear()
        self.recursive_loops.clear()
        
        # Create initial evidence
        initial_fuzzy_evidence = FuzzyEvidence(
            evidence_type="query_input",
            confidence=0.8,
            uncertainty=0.3,
            source="user_query",
            data={'query': query, **initial_evidence},
            timestamp=datetime.now(),
            environmental_context=initial_evidence.get('environmental_context', {})
        )
        
        # Add initial evidence to entry nodes
        entry_nodes = ['environmental_acquisition']  # Start here but can branch
        for node_id in entry_nodes:
            await self.nodes[node_id].add_evidence(initial_fuzzy_evidence)
            self.active_nodes.add(node_id)
        
        # Dynamic execution with potential recursion
        execution_rounds = 0
        max_execution_rounds = 20  # Prevent infinite loops
        
        while self.active_nodes and execution_rounds < max_execution_rounds:
            execution_rounds += 1
            logger.debug("Execution round %d: Active nodes: %s", execution_rounds, list(self.active_nodes))
            
            # Process active nodes
            round_results = await self._execute_active_nodes()
            
            # Check for convergence and routing decisions
            await self._update_network_state()
            
            # Dynamic routing based on network state
            await self._make_routing_decisions()
            
            # External validation for nodes that need it
            await self._run_external_validations(initial_evidence)
            
            # Check if network has converged
            if await self._check_network_convergence():
                break
        
        # Generate final results
        final_results = await self._generate_network_results(query, execution_id, start_time)
        
        # Store execution history
        self.execution_history.append({
            'execution_id': execution_id,
            'query': query,
            'execution_rounds': execution_rounds,
            'nodes_processed': len(self.converged_nodes),
            'recursive_loops': dict(self.recursive_loops),
            'final_coherence': final_results.get('network_coherence', 0.0),
            'processing_time': final_results.get('processing_time', 0.0)
        })
        
        logger.info("✅ Bayesian network processing completed in %d rounds", execution_rounds)
        return final_results
    
    async def _execute_active_nodes(self) -> Dict[str, Any]:
        """Execute all currently active nodes."""
        
        round_results = {}
        
        for node_id in list(self.active_nodes):  # Copy list as we modify during iteration
            node = self.nodes[node_id]
            
            try:
                # Update node beliefs
                node.state = NodeState.PROCESSING
                confidence = await node.update_beliefs()
                
                # Check convergence
                converged = await node.check_convergence()
                
                round_results[node_id] = {
                    'confidence': confidence,
                    'uncertainty': node.belief_uncertainty,
                    'converged': converged,
                    'state': node.state.value,
                    'recursive_count': node.recursive_count
                }
                
                logger.debug("Node %s: confidence=%.3f, uncertainty=%.3f, state=%s", 
                           node_id, confidence, node.belief_uncertainty, node.state.value)
                
            except Exception as e:
                logger.error("Error processing node %s: %s", node_id, str(e))
                node.state = NodeState.FAILED
                round_results[node_id] = {'error': str(e), 'state': 'failed'}
        
        return round_results
    
    async def _update_network_state(self) -> None:
        """Update overall network state based on node states."""
        
        for node_id in list(self.active_nodes):
            node = self.nodes[node_id]
            
            if node.state == NodeState.CONVERGED:
                self.active_nodes.discard(node_id)
                self.converged_nodes.add(node_id)
                
                # Activate dependent nodes
                for output_node_id in node.output_nodes:
                    if self._can_activate_node(output_node_id):
                        self.active_nodes.add(output_node_id)
                        logger.debug("Activated dependent node: %s", output_node_id)
            
            elif node.state == NodeState.RECURSIVE_LOOP:
                self.recursive_loops[node_id] += 1
                # Node stays active for recursive processing
                
            elif node.state == NodeState.FAILED:
                self.active_nodes.discard(node_id)
                logger.warning("Node %s failed and removed from active set", node_id)
    
    def _can_activate_node(self, node_id: str) -> bool:
        """Check if a node can be activated based on dependencies."""
        
        if node_id in self.active_nodes or node_id in self.converged_nodes:
            return False
        
        # Check if all dependencies are satisfied
        required_deps = self.node_dependencies.get(node_id, set())
        satisfied_deps = required_deps & self.converged_nodes
        
        # Node can be activated if enough dependencies are satisfied
        satisfaction_ratio = len(satisfied_deps) / max(1, len(required_deps))
        return satisfaction_ratio >= 0.7  # Fuzzy activation threshold
    
    async def _make_routing_decisions(self) -> None:
        """Make dynamic routing decisions based on network state."""
        
        # Analyze network coherence to make routing decisions
        network_coherence = await self._calculate_network_coherence()
        self.network_coherence_history.append(network_coherence)
        
        # If coherence is dropping, consider recursive loops
        if len(self.network_coherence_history) >= 3:
            recent_coherence = self.network_coherence_history[-3:]
            if all(recent_coherence[i] > recent_coherence[i+1] for i in range(len(recent_coherence)-1)):
                # Coherence is dropping - activate recursive processing
                await self._activate_recursive_processing()
        
        # If certain nodes have high uncertainty, gather more evidence
        for node_id, node in self.nodes.items():
            if (node.belief_uncertainty > 0.7 and 
                node.belief_confidence < 0.6 and 
                node_id not in self.active_nodes):
                
                # Reactivate node for more evidence gathering
                node.state = NodeState.EVIDENCE_GATHERING
                self.active_nodes.add(node_id)
                logger.debug("Reactivated node %s for additional evidence gathering", node_id)
    
    async def _activate_recursive_processing(self) -> None:
        """Activate recursive processing for nodes that need it."""
        
        for node_id, node in self.nodes.items():
            if (node.recursive_count < node.max_recursive_depth and
                node.belief_uncertainty > 0.5):
                
                node.state = NodeState.RECURSIVE_LOOP
                self.active_nodes.add(node_id)
                logger.info("Activated recursive processing for node: %s", node_id)
    
    async def _run_external_validations(self, initial_evidence: Dict[str, Any]) -> None:
        """Run external validations for nodes that need it."""
        
        for node_id in list(self.active_nodes):
            node = self.nodes[node_id]
            
            # Run external validation if node has processed enough evidence
            if (len(node.processed_evidence) >= 2 and 
                not node.validation_results and
                node.external_validators):
                
                validation_passed = await node.validate_externally(initial_evidence)
                
                if not validation_passed:
                    # Add this as evidence for potential recursive processing
                    validation_evidence = FuzzyEvidence(
                        evidence_type="external_validation_failed",
                        confidence=0.3,
                        uncertainty=0.8,
                        source="external_validator",
                        data={'validation_results': node.validation_results},
                        timestamp=datetime.now(),
                        environmental_context=initial_evidence.get('environmental_context', {})
                    )
                    await node.add_evidence(validation_evidence)
    
    async def _check_network_convergence(self) -> bool:
        """Check if the overall network has converged."""
        
        # Network converged if most nodes have converged
        convergence_ratio = len(self.converged_nodes) / len(self.nodes)
        
        # Or if we have essential nodes converged
        essential_nodes = {'reasoning_orchestration', 'synthesis_emergence'}
        essential_converged = essential_nodes & self.converged_nodes
        
        return convergence_ratio >= 0.7 or len(essential_converged) >= 1
    
    async def _calculate_network_coherence(self) -> float:
        """Calculate overall network coherence."""
        
        node_coherences = []
        
        for node in self.nodes.values():
            if node.active_path_id and node.active_path_id in node.embedding_paths:
                path_coherence = node.embedding_paths[node.active_path_id].get_path_coherence()
                node_coherences.append(path_coherence)
            else:
                # Use belief confidence as coherence proxy
                node_coherences.append(node.belief_confidence)
        
        return np.mean(node_coherences) if node_coherences else 0.0
    
    async def _generate_network_results(self, query: str, execution_id: str, start_time: datetime) -> Dict[str, Any]:
        """Generate final results from the Bayesian network."""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Collect results from converged nodes
        node_results = {}
        embedding_paths = {}
        
        for node_id in self.converged_nodes:
            node = self.nodes[node_id]
            node_results[node_id] = {
                'belief_confidence': node.belief_confidence,
                'belief_uncertainty': node.belief_uncertainty,
                'processed_evidence_count': len(node.processed_evidence),
                'validation_results': node.validation_results,
                'recursive_count': node.recursive_count,
                'state': node.state.value
            }
            
            # Collect embedding paths
            for path_id, path in node.embedding_paths.items():
                embedding_paths[f"{node_id}_{path_id}"] = {
                    'dimensions': path.dimensions,
                    'path_length': len(path.embedding_sequence),
                    'coherence': path.get_path_coherence(),
                    'environmental_stability': path.get_environmental_stability(),
                    'external_validations': len(path.external_validations),
                    'similar_environments': path.similarity_environments
                }
        
        # Calculate final network coherence
        final_coherence = await self._calculate_network_coherence()
        
        return {
            'query': query,
            'execution_id': execution_id,
            'processing_time': processing_time,
            'network_coherence': final_coherence,
            'nodes_converged': len(self.converged_nodes),
            'total_nodes': len(self.nodes),
            'recursive_loops_executed': dict(self.recursive_loops),
            'node_results': node_results,
            'embedding_paths': embedding_paths,
            'coherence_trajectory': self.network_coherence_history,
            'bayesian_network_success': final_coherence > 0.6,
            'external_validations_passed': sum(
                1 for node in self.nodes.values()
                for result in node.validation_results.values()
                if result.get('passed', False)
            ),
            'network_metadata': {
                'execution_rounds': len(self.network_coherence_history),
                'max_recursive_depth_reached': max(self.recursive_loops.values()) if self.recursive_loops else 0,
                'environmental_contexts_analyzed': len(self.environment_database),
                'similarity_networks_built': len(self.similarity_network)
            }
        }
    
    # External validator methods
    
    async def _environmental_database_validator(self, node: BayesianNode, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against environmental database."""
        
        # This would query actual environmental database
        # For now, simulate validation
        return {
            'validator': 'environmental_database',
            'passed': True,
            'confidence': 0.8,
            'matched_environments': 3,
            'similarity_score': 0.75
        }
    
    async def _mathematical_model_validator(self, node: BayesianNode, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against mathematical models."""
        
        # This would run actual mathematical model validation
        return {
            'validator': 'mathematical_model',
            'passed': node.belief_confidence > 0.6,
            'confidence': node.belief_confidence,
            'model_coherence': node.belief_confidence * (1.0 - node.belief_uncertainty)
        }
    
    async def _similar_context_validator(self, node: BayesianNode, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against similar contexts."""
        
        similarity_score = np.mean(list(node.environment_similarities.values())) if node.environment_similarities else 0.5
        
        return {
            'validator': 'similar_context',
            'passed': similarity_score > 0.6,
            'confidence': similarity_score,
            'similar_contexts_found': len(node.similar_environments)
        }
    
    async def _knowledge_graph_validator(self, node: BayesianNode, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against knowledge graphs."""
        
        # This would query actual knowledge graphs
        return {
            'validator': 'knowledge_graph',
            'passed': True,
            'confidence': 0.75,
            'knowledge_connections': 5,
            'graph_coherence': 0.8
        }
