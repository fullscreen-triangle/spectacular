"""
Mzekezeke: Bayesian Evidence Network for Spectacular

This module implements a Bayesian evidence network with an objective function
that can be optimized for probabilistic reasoning in visualization generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import json

# Probabilistic modeling dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Install with: pip install networkx")

try:
    import scipy.stats as stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Install with: pip install scipy")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """Types of evidence in the network."""
    QUERY_FEATURE = "query_feature"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    VISUAL_PATTERN = "visual_pattern"
    CODE_PATTERN = "code_pattern"
    USER_PREFERENCE = "user_preference"
    HISTORICAL_SUCCESS = "historical_success"


class NodeType(Enum):
    """Types of nodes in the Bayesian network."""
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    UTILITY = "utility"
    DECISION = "decision"


@dataclass
class EvidenceNode:
    """Represents an evidence node in the network."""
    id: str
    name: str
    evidence_type: EvidenceType
    value: float
    confidence: float
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisNode:
    """Represents a hypothesis node in the network."""
    id: str
    name: str
    prior_probability: float
    posterior_probability: float
    likelihood_cache: Dict[str, float] = field(default_factory=dict)
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)


@dataclass
class BayesianUpdate:
    """Represents a Bayesian update operation."""
    hypothesis_id: str
    evidence_id: str
    prior: float
    likelihood: float
    posterior: float
    update_time: float
    confidence_gain: float


class BayesianEvidenceNetwork:
    """
    Advanced Bayesian Evidence Network for probabilistic reasoning.
    
    This class implements:
    1. Dynamic Bayesian network construction
    2. Evidence accumulation and belief updating
    3. Objective function optimization
    4. Probabilistic inference for visualization generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Bayesian evidence network."""
        self.config = config or {}
        self.ready = False
        
        # Network components
        self.network = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.evidence_nodes: Dict[str, EvidenceNode] = {}
        self.hypothesis_nodes: Dict[str, HypothesisNode] = {}
        self.update_history: List[BayesianUpdate] = []
        
        # Configuration parameters
        self.max_evidence_nodes = config.get('max_evidence_nodes', 1000)
        self.max_hypothesis_nodes = config.get('max_hypothesis_nodes', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.learning_rate = config.get('learning_rate', 0.01)
        
        # Objective function parameters
        self.objective_weights = config.get('objective_weights', {
            'accuracy': 0.4,
            'consistency': 0.3,
            'novelty': 0.2,
            'efficiency': 0.1
        })
        
        # Initialize the network
        asyncio.create_task(self._initialize_network())
        
        logger.info("Mzekezeke Bayesian Evidence Network initialized")
    
    async def _initialize_network(self):
        """Initialize the Bayesian network structure."""
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX not available for network construction")
            self.ready = False
            return
        
        try:
            # Create initial hypothesis nodes
            await self._create_initial_hypotheses()
            
            # Set up objective function
            await self._setup_objective_function()
            
            self.ready = True
            logger.info("Bayesian network initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Bayesian network: {e}")
            self.ready = False
    
    async def _create_initial_hypotheses(self):
        """Create initial hypothesis nodes for visualization generation."""
        
        initial_hypotheses = [
            # Visualization type hypotheses
            HypothesisNode(
                id="H001",
                name="bar_chart_appropriate",
                prior_probability=0.3,
                posterior_probability=0.3
            ),
            HypothesisNode(
                id="H002", 
                name="line_chart_appropriate",
                prior_probability=0.25,
                posterior_probability=0.25
            ),
            HypothesisNode(
                id="H003",
                name="scatter_plot_appropriate", 
                prior_probability=0.2,
                posterior_probability=0.2
            ),
            HypothesisNode(
                id="H004",
                name="heatmap_appropriate",
                prior_probability=0.15,
                posterior_probability=0.15
            ),
            HypothesisNode(
                id="H005",
                name="network_diagram_appropriate",
                prior_probability=0.1,
                posterior_probability=0.1
            ),
            
            # Complexity hypotheses
            HypothesisNode(
                id="H006",
                name="simple_visualization_needed",
                prior_probability=0.4,
                posterior_probability=0.4
            ),
            HypothesisNode(
                id="H007",
                name="interactive_features_needed",
                prior_probability=0.3,
                posterior_probability=0.3
            ),
            HypothesisNode(
                id="H008",
                name="animation_needed",
                prior_probability=0.2,
                posterior_probability=0.2
            ),
            
            # Data structure hypotheses
            HypothesisNode(
                id="H009",
                name="hierarchical_data_structure",
                prior_probability=0.25,
                posterior_probability=0.25
            ),
            HypothesisNode(
                id="H010",
                name="time_series_data",
                prior_probability=0.3,
                posterior_probability=0.3
            )
        ]
        
        for hypothesis in initial_hypotheses:
            self.hypothesis_nodes[hypothesis.id] = hypothesis
            if NETWORKX_AVAILABLE:
                self.network.add_node(hypothesis.id, 
                                    type=NodeType.HYPOTHESIS,
                                    data=hypothesis)
        
        logger.info(f"Created {len(initial_hypotheses)} initial hypothesis nodes")
    
    async def _setup_objective_function(self):
        """Set up the objective function for optimization."""
        
        # Objective function: maximize expected utility
        # J = Σ w_i * U_i(h, e) - λ * R(θ)
        # where w_i are weights, U_i are utility functions, R(θ) is regularization
        
        self.objective_function = {
            'components': [
                {'name': 'accuracy', 'weight': self.objective_weights['accuracy']},
                {'name': 'consistency', 'weight': self.objective_weights['consistency']}, 
                {'name': 'novelty', 'weight': self.objective_weights['novelty']},
                {'name': 'efficiency', 'weight': self.objective_weights['efficiency']}
            ],
            'regularization_lambda': 0.01,
            'optimization_history': []
        }
        
        logger.info("Objective function configured")
    
    async def update_beliefs(self, query_context, reasoning_results: Dict) -> Dict[str, Any]:
        """Update beliefs in the network based on new evidence."""
        logger.info("Updating beliefs with new evidence...")
        
        # Extract evidence from query context and reasoning results
        evidence_list = await self._extract_evidence(query_context, reasoning_results)
        
        # Add evidence to network
        update_results = []
        for evidence in evidence_list:
            result = await self._add_evidence_and_update(evidence)
            update_results.append(result)
        
        # Optimize objective function
        optimization_result = await self._optimize_objective_function()
        
        # Calculate network statistics
        network_stats = await self._calculate_network_statistics()
        
        return {
            "evidence_added": len(evidence_list),
            "beliefs_updated": len(update_results),
            "network_stats": network_stats,
            "optimization_result": optimization_result,
            "top_hypotheses": await self._get_top_hypotheses(5),
            "confidence": np.mean([r.get('confidence', 0.5) for r in update_results])
        }
    
    async def _extract_evidence(self, query_context, reasoning_results: Dict) -> List[EvidenceNode]:
        """Extract evidence from context and reasoning results."""
        evidence_list = []
        
        # Evidence from query context
        if hasattr(query_context, 'query'):
            # Query complexity evidence
            evidence_list.append(EvidenceNode(
                id=f"E{len(self.evidence_nodes) + len(evidence_list) + 1:03d}",
                name="query_complexity",
                evidence_type=EvidenceType.QUERY_FEATURE,
                value=getattr(query_context, 'complexity_score', 0.5),
                confidence=0.8,
                timestamp=asyncio.get_event_loop().time(),
                source="query_context"
            ))
            
            # Domain specificity evidence
            evidence_list.append(EvidenceNode(
                id=f"E{len(self.evidence_nodes) + len(evidence_list) + 1:03d}",
                name="domain_specificity",
                evidence_type=EvidenceType.DOMAIN_KNOWLEDGE,
                value=getattr(query_context, 'domain_specificity', 0.5),
                confidence=0.8,
                timestamp=asyncio.get_event_loop().time(),
                source="query_context"
            ))
            
            # Visual complexity evidence  
            evidence_list.append(EvidenceNode(
                id=f"E{len(self.evidence_nodes) + len(evidence_list) + 1:03d}",
                name="visual_complexity",
                evidence_type=EvidenceType.VISUAL_PATTERN,
                value=getattr(query_context, 'visual_complexity', 0.5),
                confidence=0.8,
                timestamp=asyncio.get_event_loop().time(),
                source="query_context"
            ))
        
        # Evidence from reasoning results
        for step_name, step_data in reasoning_results.get('steps', {}).items():
            if isinstance(step_data, dict) and 'confidence' in step_data:
                evidence_list.append(EvidenceNode(
                    id=f"E{len(self.evidence_nodes) + len(evidence_list) + 1:03d}",
                    name=f"reasoning_{step_name}",
                    evidence_type=EvidenceType.CODE_PATTERN,
                    value=step_data['confidence'],
                    confidence=step_data.get('reliability', 0.7),
                    timestamp=asyncio.get_event_loop().time(),
                    source=step_name,
                    metadata={'step_data': step_data}
                ))
        
        return evidence_list
    
    async def _add_evidence_and_update(self, evidence: EvidenceNode) -> Dict[str, Any]:
        """Add evidence to network and perform Bayesian updates."""
        
        # Add evidence node to network
        self.evidence_nodes[evidence.id] = evidence
        if NETWORKX_AVAILABLE:
            self.network.add_node(evidence.id, type=NodeType.EVIDENCE, data=evidence)
        
        # Determine relevant hypotheses
        relevant_hypotheses = await self._find_relevant_hypotheses(evidence)
        
        # Perform Bayesian updates
        updates = []
        for hypothesis_id in relevant_hypotheses:
            update = await self._bayesian_update(hypothesis_id, evidence)
            if update:
                updates.append(update)
                self.update_history.append(update)
        
        return {
            "evidence_id": evidence.id,
            "relevant_hypotheses": len(relevant_hypotheses),
            "updates_performed": len(updates),
            "confidence": evidence.confidence
        }
    
    async def _find_relevant_hypotheses(self, evidence: EvidenceNode) -> List[str]:
        """Find hypotheses relevant to the given evidence."""
        relevant = []
        
        # Simple relevance rules based on evidence type and name
        evidence_name = evidence.name.lower()
        
        for hypothesis_id, hypothesis in self.hypothesis_nodes.items():
            hypothesis_name = hypothesis.name.lower()
            
            # Relevance rules
            if evidence.evidence_type == EvidenceType.QUERY_FEATURE:
                if 'complexity' in evidence_name and 'simple' in hypothesis_name:
                    relevant.append(hypothesis_id)
                elif 'complexity' in evidence_name and 'interactive' in hypothesis_name:
                    relevant.append(hypothesis_id)
            
            elif evidence.evidence_type == EvidenceType.VISUAL_PATTERN:
                if 'visual' in evidence_name and any(viz in hypothesis_name for viz in ['chart', 'plot', 'diagram']):
                    relevant.append(hypothesis_id)
            
            elif evidence.evidence_type == EvidenceType.DOMAIN_KNOWLEDGE:
                if 'domain' in evidence_name:
                    relevant.append(hypothesis_id)
            
            # General relevance for all hypotheses if high confidence evidence
            if evidence.confidence > 0.9:
                relevant.append(hypothesis_id)
        
        return list(set(relevant))  # Remove duplicates
    
    async def _bayesian_update(self, hypothesis_id: str, evidence: EvidenceNode) -> Optional[BayesianUpdate]:
        """Perform Bayesian update for a hypothesis given evidence."""
        
        hypothesis = self.hypothesis_nodes.get(hypothesis_id)
        if not hypothesis:
            return None
        
        # Calculate likelihood P(E|H)
        likelihood = await self._calculate_likelihood(hypothesis, evidence)
        
        if likelihood is None:
            return None
        
        # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        prior = hypothesis.posterior_probability
        
        # Simplified Bayesian update (assuming uniform evidence probability)
        # In practice, this would be more sophisticated
        evidence_prob = 0.5  # P(E) - could be calculated more precisely
        
        posterior = (likelihood * prior) / evidence_prob
        posterior = np.clip(posterior, 0.0, 1.0)  # Ensure valid probability
        
        # Update hypothesis
        old_posterior = hypothesis.posterior_probability
        hypothesis.posterior_probability = posterior
        
        # Calculate confidence gain
        confidence_gain = abs(posterior - prior) * evidence.confidence
        
        return BayesianUpdate(
            hypothesis_id=hypothesis_id,
            evidence_id=evidence.id,
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            update_time=asyncio.get_event_loop().time(),
            confidence_gain=confidence_gain
        )
    
    async def _calculate_likelihood(self, hypothesis: HypothesisNode, evidence: EvidenceNode) -> Optional[float]:
        """Calculate likelihood P(E|H) for given hypothesis and evidence."""
        
        # Cache key for likelihood
        cache_key = f"{evidence.id}_{evidence.value:.3f}"
        if cache_key in hypothesis.likelihood_cache:
            return hypothesis.likelihood_cache[cache_key]
        
        # Calculate likelihood based on evidence type and hypothesis
        likelihood = None
        
        hypothesis_name = hypothesis.name.lower()
        evidence_name = evidence.name.lower()
        
        if evidence.evidence_type == EvidenceType.QUERY_FEATURE:
            if 'complexity' in evidence_name:
                if 'simple' in hypothesis_name:
                    # High complexity evidence makes simple visualization less likely
                    likelihood = 1.0 - evidence.value
                elif 'interactive' in hypothesis_name:
                    # High complexity evidence makes interactive features more likely
                    likelihood = evidence.value
                else:
                    likelihood = 0.5  # Neutral
        
        elif evidence.evidence_type == EvidenceType.VISUAL_PATTERN:
            if 'visual_complexity' in evidence_name:
                if any(viz in hypothesis_name for viz in ['bar', 'line', 'scatter']):
                    # Simple chart types are less likely with high visual complexity
                    likelihood = 1.0 - evidence.value * 0.7
                elif 'heatmap' in hypothesis_name or 'network' in hypothesis_name:
                    # Complex visualizations more likely with high visual complexity
                    likelihood = evidence.value * 0.8 + 0.2
        
        # Default likelihood if not calculated above
        if likelihood is None:
            # Use Gaussian likelihood centered at 0.5
            likelihood = stats.norm.pdf(evidence.value, loc=0.5, scale=0.3)
            likelihood = likelihood / stats.norm.pdf(0.5, loc=0.5, scale=0.3)  # Normalize
        
        # Ensure valid probability
        likelihood = np.clip(likelihood, 0.01, 0.99)
        
        # Cache the result
        hypothesis.likelihood_cache[cache_key] = likelihood
        
        return likelihood
    
    async def _optimize_objective_function(self) -> Dict[str, Any]:
        """Optimize the objective function."""
        logger.info("Optimizing objective function...")
        
        if not SCIPY_AVAILABLE:
            return {"status": "scipy_not_available", "objective_value": 0.0}
        
        # Current objective value
        current_objective = await self._calculate_objective_value()
        
        # Simple optimization: adjust hypothesis priors based on evidence
        optimization_steps = []
        
        for hypothesis_id, hypothesis in self.hypothesis_nodes.items():
            # Calculate evidence support for this hypothesis
            support = await self._calculate_evidence_support(hypothesis_id)
            
            # Adjust prior based on support (simple learning rule)
            old_prior = hypothesis.prior_probability
            adjustment = self.learning_rate * (support - hypothesis.prior_probability)
            new_prior = np.clip(hypothesis.prior_probability + adjustment, 0.01, 0.99)
            
            hypothesis.prior_probability = new_prior
            
            optimization_steps.append({
                "hypothesis_id": hypothesis_id,
                "old_prior": old_prior,
                "new_prior": new_prior,
                "support": support,
                "adjustment": adjustment
            })
        
        # Calculate new objective value
        new_objective = await self._calculate_objective_value()
        improvement = new_objective - current_objective
        
        self.objective_function['optimization_history'].append({
            "timestamp": asyncio.get_event_loop().time(),
            "old_objective": current_objective,
            "new_objective": new_objective,
            "improvement": improvement,
            "steps": len(optimization_steps)
        })
        
        return {
            "status": "completed",
            "objective_improvement": improvement,
            "optimization_steps": len(optimization_steps),
            "new_objective_value": new_objective
        }
    
    async def _calculate_objective_value(self) -> float:
        """Calculate current objective function value."""
        
        # Simplified objective function calculation
        objective_value = 0.0
        
        # Accuracy component: how well posterior probabilities match evidence
        accuracy = 0.0
        if self.hypothesis_nodes:
            accuracy = np.mean([h.posterior_probability for h in self.hypothesis_nodes.values()])
        
        # Consistency component: how consistent beliefs are across similar hypotheses
        consistency = await self._calculate_consistency()
        
        # Novelty component: reward for discovering new patterns
        novelty = len(self.evidence_nodes) / self.max_evidence_nodes
        
        # Efficiency component: penalize excessive complexity
        efficiency = 1.0 - (len(self.hypothesis_nodes) / self.max_hypothesis_nodes)
        
        # Weighted sum
        objective_value = (
            self.objective_weights['accuracy'] * accuracy +
            self.objective_weights['consistency'] * consistency +
            self.objective_weights['novelty'] * novelty +
            self.objective_weights['efficiency'] * efficiency
        )
        
        return objective_value
    
    async def _calculate_consistency(self) -> float:
        """Calculate consistency metric for beliefs."""
        if len(self.hypothesis_nodes) < 2:
            return 1.0
        
        # Calculate variance in posterior probabilities as inverse consistency measure
        posteriors = [h.posterior_probability for h in self.hypothesis_nodes.values()]
        variance = np.var(posteriors)
        consistency = 1.0 / (1.0 + variance)  # Convert variance to consistency score
        
        return consistency
    
    async def _calculate_evidence_support(self, hypothesis_id: str) -> float:
        """Calculate total evidence support for a hypothesis."""
        support = 0.0
        count = 0
        
        for evidence in self.evidence_nodes.values():
            # Find updates related to this hypothesis and evidence
            for update in self.update_history:
                if update.hypothesis_id == hypothesis_id and update.evidence_id == evidence.id:
                    support += update.confidence_gain
                    count += 1
        
        return support / max(1, count)  # Average support
    
    async def _calculate_network_statistics(self) -> Dict[str, Any]:
        """Calculate network statistics."""
        stats = {
            "total_evidence_nodes": len(self.evidence_nodes),
            "total_hypothesis_nodes": len(self.hypothesis_nodes),
            "total_updates": len(self.update_history),
            "average_posterior": 0.0,
            "highest_confidence_hypothesis": None,
            "evidence_types": {}
        }
        
        if self.hypothesis_nodes:
            posteriors = [h.posterior_probability for h in self.hypothesis_nodes.values()]
            stats["average_posterior"] = np.mean(posteriors)
            
            # Find highest confidence hypothesis
            best_hypothesis = max(self.hypothesis_nodes.values(), 
                                key=lambda h: h.posterior_probability)
            stats["highest_confidence_hypothesis"] = {
                "id": best_hypothesis.id,
                "name": best_hypothesis.name,
                "probability": best_hypothesis.posterior_probability
            }
        
        # Count evidence types
        for evidence in self.evidence_nodes.values():
            evidence_type = evidence.evidence_type.value
            stats["evidence_types"][evidence_type] = stats["evidence_types"].get(evidence_type, 0) + 1
        
        return stats
    
    async def _get_top_hypotheses(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N hypotheses by posterior probability."""
        sorted_hypotheses = sorted(
            self.hypothesis_nodes.values(),
            key=lambda h: h.posterior_probability,
            reverse=True
        )
        
        return [
            {
                "id": h.id,
                "name": h.name,
                "posterior_probability": h.posterior_probability,
                "prior_probability": h.prior_probability
            }
            for h in sorted_hypotheses[:n]
        ]
    
    def is_ready(self) -> bool:
        """Check if the Bayesian network is ready."""
        return self.ready
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get a summary of the network state."""
        return {
            "ready": self.ready,
            "evidence_nodes": len(self.evidence_nodes),
            "hypothesis_nodes": len(self.hypothesis_nodes),
            "total_updates": len(self.update_history),
            "objective_function": self.objective_function,
            "networkx_available": NETWORKX_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE
        }
    
    async def export_network(self, filepath: str) -> bool:
        """Export network to JSON file."""
        try:
            network_data = {
                "evidence_nodes": {
                    eid: {
                        "id": e.id,
                        "name": e.name,
                        "evidence_type": e.evidence_type.value,
                        "value": e.value,
                        "confidence": e.confidence,
                        "timestamp": e.timestamp,
                        "source": e.source,
                        "metadata": e.metadata
                    }
                    for eid, e in self.evidence_nodes.items()
                },
                "hypothesis_nodes": {
                    hid: {
                        "id": h.id,
                        "name": h.name,
                        "prior_probability": h.prior_probability,
                        "posterior_probability": h.posterior_probability
                    }
                    for hid, h in self.hypothesis_nodes.items()
                },
                "objective_function": self.objective_function,
                "config": self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(network_data, f, indent=2)
            
            logger.info(f"Network exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting network: {e}")
            return False