"""
Pretoria: Fuzzy Logic Programming Engine for Spectacular

This module implements the fuzzy logic reasoning system that generates
internal logical programming scripts for code generation strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Fuzzy logic dependencies
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("scikit-fuzzy not available. Install with: pip install scikit-fuzzy")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuzzyVariable(Enum):
    """Enumeration of fuzzy variables used in the system."""
    QUERY_COMPLEXITY = "query_complexity"
    DOMAIN_SPECIFICITY = "domain_specificity"
    VISUAL_COMPLEXITY = "visual_complexity"
    DATA_COMPLEXITY = "data_complexity"
    PROMPT_STRATEGY = "prompt_strategy"
    REASONING_DEPTH = "reasoning_depth"


@dataclass
class FuzzyRule:
    """Represents a fuzzy rule in the system."""
    id: str
    antecedent: str
    consequent: str
    weight: float
    confidence: float
    description: str


@dataclass
class FuzzyAnalysisResult:
    """Result of fuzzy analysis."""
    complexity: float
    domain_specificity: float
    visual_complexity: float
    strategy: str
    reasoning_depth: str
    confidence: float
    applied_rules: List[str]


class FuzzyLogicEngine:
    """
    Advanced fuzzy logic programming engine for code generation strategies.
    
    This class implements fuzzy inference systems for:
    1. Query complexity analysis
    2. Code generation strategy selection
    3. Reasoning depth determination
    4. Prompt optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the fuzzy logic engine."""
        self.config = config or {}
        self.ready = False
        
        # Fuzzy variables and rules
        self.fuzzy_vars = {}
        self.fuzzy_rules = []
        self.control_system = None
        self.simulation = None
        
        # Configuration
        self.max_rules = config.get('max_rules', 500)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Initialize the fuzzy system
        asyncio.create_task(self._initialize_fuzzy_system())
        
        logger.info("Pretoria Fuzzy Logic Engine initialized")
    
    async def _initialize_fuzzy_system(self):
        """Initialize the fuzzy inference system."""
        if not FUZZY_AVAILABLE:
            logger.error("Fuzzy logic library not available")
            return
        
        try:
            # Define fuzzy variables
            await self._define_fuzzy_variables()
            
            # Load fuzzy rules
            await self._load_fuzzy_rules()
            
            # Create control system
            await self._create_control_system()
            
            self.ready = True
            logger.info("Fuzzy system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing fuzzy system: {e}")
            self.ready = False
    
    async def _define_fuzzy_variables(self):
        """Define fuzzy variables and their membership functions."""
        
        # Input variables
        self.fuzzy_vars['query_complexity'] = ctrl.Antecedent(
            np.arange(0, 1.1, 0.1), 'query_complexity'
        )
        self.fuzzy_vars['domain_specificity'] = ctrl.Antecedent(
            np.arange(0, 1.1, 0.1), 'domain_specificity'
        )
        self.fuzzy_vars['visual_complexity'] = ctrl.Antecedent(
            np.arange(0, 1.1, 0.1), 'visual_complexity'
        )
        self.fuzzy_vars['data_complexity'] = ctrl.Antecedent(
            np.arange(0, 1.1, 0.1), 'data_complexity'
        )
        
        # Output variables
        self.fuzzy_vars['prompt_strategy'] = ctrl.Consequent(
            np.arange(0, 1.1, 0.1), 'prompt_strategy'
        )
        self.fuzzy_vars['reasoning_depth'] = ctrl.Consequent(
            np.arange(0, 1.1, 0.1), 'reasoning_depth'
        )
        
        # Define membership functions for query complexity
        self.fuzzy_vars['query_complexity']['low'] = fuzz.trimf(
            self.fuzzy_vars['query_complexity'].universe, [0, 0, 0.4]
        )
        self.fuzzy_vars['query_complexity']['medium'] = fuzz.trimf(
            self.fuzzy_vars['query_complexity'].universe, [0.2, 0.5, 0.8]
        )
        self.fuzzy_vars['query_complexity']['high'] = fuzz.trimf(
            self.fuzzy_vars['query_complexity'].universe, [0.6, 1.0, 1.0]
        )
        
        # Define membership functions for domain specificity
        self.fuzzy_vars['domain_specificity']['low'] = fuzz.trimf(
            self.fuzzy_vars['domain_specificity'].universe, [0, 0, 0.4]
        )
        self.fuzzy_vars['domain_specificity']['medium'] = fuzz.trimf(
            self.fuzzy_vars['domain_specificity'].universe, [0.2, 0.5, 0.8]
        )
        self.fuzzy_vars['domain_specificity']['high'] = fuzz.trimf(
            self.fuzzy_vars['domain_specificity'].universe, [0.6, 1.0, 1.0]
        )
        
        # Define membership functions for visual complexity
        self.fuzzy_vars['visual_complexity']['simple'] = fuzz.trimf(
            self.fuzzy_vars['visual_complexity'].universe, [0, 0, 0.4]
        )
        self.fuzzy_vars['visual_complexity']['moderate'] = fuzz.trimf(
            self.fuzzy_vars['visual_complexity'].universe, [0.2, 0.5, 0.8]
        )
        self.fuzzy_vars['visual_complexity']['complex'] = fuzz.trimf(
            self.fuzzy_vars['visual_complexity'].universe, [0.6, 1.0, 1.0]
        )
        
        # Define membership functions for data complexity
        self.fuzzy_vars['data_complexity']['simple'] = fuzz.trimf(
            self.fuzzy_vars['data_complexity'].universe, [0, 0, 0.4]
        )
        self.fuzzy_vars['data_complexity']['moderate'] = fuzz.trimf(
            self.fuzzy_vars['data_complexity'].universe, [0.2, 0.5, 0.8]
        )
        self.fuzzy_vars['data_complexity']['complex'] = fuzz.trimf(
            self.fuzzy_vars['data_complexity'].universe, [0.6, 1.0, 1.0]
        )
        
        # Define membership functions for prompt strategy
        self.fuzzy_vars['prompt_strategy']['simple'] = fuzz.trimf(
            self.fuzzy_vars['prompt_strategy'].universe, [0, 0, 0.4]
        )
        self.fuzzy_vars['prompt_strategy']['multi_step'] = fuzz.trimf(
            self.fuzzy_vars['prompt_strategy'].universe, [0.2, 0.5, 0.8]
        )
        self.fuzzy_vars['prompt_strategy']['advanced'] = fuzz.trimf(
            self.fuzzy_vars['prompt_strategy'].universe, [0.6, 1.0, 1.0]
        )
        
        # Define membership functions for reasoning depth
        self.fuzzy_vars['reasoning_depth']['shallow'] = fuzz.trimf(
            self.fuzzy_vars['reasoning_depth'].universe, [0, 0, 0.4]
        )
        self.fuzzy_vars['reasoning_depth']['moderate'] = fuzz.trimf(
            self.fuzzy_vars['reasoning_depth'].universe, [0.2, 0.5, 0.8]
        )
        self.fuzzy_vars['reasoning_depth']['deep'] = fuzz.trimf(
            self.fuzzy_vars['reasoning_depth'].universe, [0.6, 1.0, 1.0]
        )
    
    async def _load_fuzzy_rules(self):
        """Load fuzzy rules for the inference system."""
        
        # Core rules for code generation strategy
        rules = [
            # Simple cases
            FuzzyRule(
                id="R001",
                antecedent="query_complexity[low] & domain_specificity[low]",
                consequent="prompt_strategy[simple] & reasoning_depth[shallow]",
                weight=1.0,
                confidence=0.9,
                description="Simple queries require basic prompt strategies"
            ),
            
            # Medium complexity cases
            FuzzyRule(
                id="R002",
                antecedent="query_complexity[medium] & visual_complexity[moderate]",
                consequent="prompt_strategy[multi_step] & reasoning_depth[moderate]",
                weight=0.9,
                confidence=0.8,
                description="Medium complexity requires multi-step reasoning"
            ),
            
            # High complexity cases
            FuzzyRule(
                id="R003",
                antecedent="query_complexity[high] & domain_specificity[high]",
                consequent="prompt_strategy[advanced] & reasoning_depth[deep]",
                weight=1.0,
                confidence=0.9,
                description="Complex domain-specific queries need advanced strategies"
            ),
            
            # Visual complexity rules
            FuzzyRule(
                id="R004",
                antecedent="visual_complexity[complex] & data_complexity[complex]",
                consequent="prompt_strategy[advanced] & reasoning_depth[deep]",
                weight=0.9,
                confidence=0.85,
                description="Complex visualizations need advanced reasoning"
            ),
            
            # Data-driven rules
            FuzzyRule(
                id="R005",
                antecedent="data_complexity[simple] & visual_complexity[simple]",
                consequent="prompt_strategy[simple] & reasoning_depth[shallow]",
                weight=0.8,
                confidence=0.8,
                description="Simple data and visuals use basic strategies"
            ),
            
            # Mixed complexity rules
            FuzzyRule(
                id="R006",
                antecedent="query_complexity[high] & visual_complexity[simple]",
                consequent="prompt_strategy[multi_step] & reasoning_depth[moderate]",
                weight=0.7,
                confidence=0.75,
                description="High query complexity with simple visuals"
            ),
            
            # Domain-specific rules
            FuzzyRule(
                id="R007",
                antecedent="domain_specificity[high] & data_complexity[moderate]",
                consequent="prompt_strategy[multi_step] & reasoning_depth[deep]",
                weight=0.8,
                confidence=0.8,
                description="Domain-specific knowledge requires deeper reasoning"
            )
        ]
        
        self.fuzzy_rules = rules
        logger.info(f"Loaded {len(self.fuzzy_rules)} fuzzy rules")
    
    async def _create_control_system(self):
        """Create the fuzzy control system."""
        if not FUZZY_AVAILABLE:
            return
        
        # Convert fuzzy rules to scikit-fuzzy format
        rules = []
        
        for rule in self.fuzzy_rules:
            try:
                # Parse antecedent
                antecedent_parts = rule.antecedent.split(' & ')
                antecedent_conditions = []
                
                for part in antecedent_parts:
                    var_name, condition = part.split('[')
                    condition = condition.rstrip(']')
                    antecedent_conditions.append(self.fuzzy_vars[var_name][condition])
                
                # Parse consequent
                consequent_parts = rule.consequent.split(' & ')
                consequent_conditions = []
                
                for part in consequent_parts:
                    var_name, condition = part.split('[')
                    condition = condition.rstrip(']')
                    consequent_conditions.append(self.fuzzy_vars[var_name][condition])
                
                # Create the rule
                if len(antecedent_conditions) == 1:
                    antecedent = antecedent_conditions[0]
                else:
                    antecedent = antecedent_conditions[0]
                    for cond in antecedent_conditions[1:]:
                        antecedent = antecedent & cond
                
                if len(consequent_conditions) == 1:
                    consequent = consequent_conditions[0]
                else:
                    consequent = consequent_conditions[0]
                    for cond in consequent_conditions[1:]:
                        consequent = consequent & cond
                
                fuzzy_rule = ctrl.Rule(antecedent, consequent, label=rule.id)
                rules.append(fuzzy_rule)
                
            except Exception as e:
                logger.warning(f"Failed to create rule {rule.id}: {e}")
        
        # Create control system
        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
        logger.info(f"Created control system with {len(rules)} active rules")
    
    async def analyze_query_complexity(self, query: str) -> Dict[str, float]:
        """Analyze query complexity using fuzzy logic."""
        logger.info(f"Analyzing query complexity: {query[:50]}...")
        
        # Extract features from query (simplified NLP analysis)
        features = await self._extract_query_features(query)
        
        if not self.ready or not FUZZY_AVAILABLE:
            # Fallback to simple heuristics
            return await self._simple_complexity_analysis(query, features)
        
        try:
            # Set inputs
            self.simulation.input['query_complexity'] = features['query_complexity']
            self.simulation.input['domain_specificity'] = features['domain_specificity']
            self.simulation.input['visual_complexity'] = features['visual_complexity']
            self.simulation.input['data_complexity'] = features['data_complexity']
            
            # Compute the result
            self.simulation.compute()
            
            # Extract outputs
            prompt_strategy = self.simulation.output['prompt_strategy']
            reasoning_depth = self.simulation.output['reasoning_depth']
            
            result = {
                'complexity': features['query_complexity'],
                'domain_specificity': features['domain_specificity'],
                'visual_complexity': features['visual_complexity'],
                'data_complexity': features['data_complexity'],
                'prompt_strategy': self._interpret_strategy(prompt_strategy),
                'reasoning_depth': self._interpret_depth(reasoning_depth),
                'confidence': 0.8
            }
            
            logger.info(f"Fuzzy analysis complete: strategy={result['prompt_strategy']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in fuzzy analysis: {e}")
            return await self._simple_complexity_analysis(query, features)
    
    async def generate_code_strategy(self, query_context) -> Dict[str, Any]:
        """Generate code generation strategy based on fuzzy analysis."""
        logger.info("Generating code strategy with fuzzy logic...")
        
        # Perform fuzzy analysis
        analysis = await self.analyze_query_complexity(query_context.query)
        
        # Generate strategy based on analysis
        strategy = {
            "approach": analysis['prompt_strategy'],
            "reasoning_depth": analysis['reasoning_depth'],
            "confidence": analysis['confidence'],
            "fuzzy_analysis": analysis,
            "recommended_actions": []
        }
        
        # Add specific recommendations based on strategy
        if analysis['prompt_strategy'] == 'advanced':
            strategy["recommended_actions"].extend([
                "use_multi_modal_reasoning",
                "apply_domain_knowledge",
                "validate_intermediate_steps"
            ])
        elif analysis['prompt_strategy'] == 'multi_step':
            strategy["recommended_actions"].extend([
                "break_down_problem",
                "iterative_refinement",
                "context_validation"
            ])
        else:
            strategy["recommended_actions"].extend([
                "direct_generation",
                "simple_validation"
            ])
        
        # Add reasoning depth recommendations
        if analysis['reasoning_depth'] == 'deep':
            strategy["recommended_actions"].extend([
                "comprehensive_analysis",
                "multiple_perspectives",
                "detailed_validation"
            ])
        
        return strategy
    
    async def _extract_query_features(self, query: str) -> Dict[str, float]:
        """Extract features from query for fuzzy analysis."""
        
        # Simplified feature extraction (in practice, this would be much more sophisticated)
        query_lower = query.lower()
        
        # Query complexity indicators
        complexity_indicators = [
            'complex', 'advanced', 'detailed', 'comprehensive', 'sophisticated',
            'multi', 'interactive', 'dynamic', 'hierarchical', 'nested'
        ]
        
        # Domain specificity indicators
        domain_indicators = [
            'd3', 'visualization', 'chart', 'graph', 'plot', 'axis', 'scale',
            'svg', 'canvas', 'data', 'dataset', 'json', 'csv'
        ]
        
        # Visual complexity indicators
        visual_indicators = [
            'animation', 'transition', 'interaction', 'brush', 'zoom', 'pan',
            'tooltip', 'legend', 'multiple', 'facet', 'grid', 'layout'
        ]
        
        # Calculate scores
        query_words = query_lower.split()
        total_words = len(query_words)
        
        complexity_score = sum(1 for word in query_words if any(ind in word for ind in complexity_indicators))
        domain_score = sum(1 for word in query_words if any(ind in word for ind in domain_indicators))
        visual_score = sum(1 for word in query_words if any(ind in word for ind in visual_indicators))
        
        # Normalize scores
        features = {
            'query_complexity': min(1.0, complexity_score / max(1, total_words * 0.1)),
            'domain_specificity': min(1.0, domain_score / max(1, total_words * 0.2)),
            'visual_complexity': min(1.0, visual_score / max(1, total_words * 0.15)),
            'data_complexity': min(1.0, len(query) / 1000)  # Simple length-based heuristic
        }
        
        # Ensure minimum values
        for key in features:
            features[key] = max(0.1, features[key])
        
        return features
    
    async def _simple_complexity_analysis(self, query: str, features: Dict) -> Dict[str, float]:
        """Fallback complexity analysis when fuzzy logic is unavailable."""
        
        # Simple rule-based analysis
        avg_complexity = np.mean(list(features.values()))
        
        if avg_complexity > 0.7:
            strategy = 'advanced'
            depth = 'deep'
        elif avg_complexity > 0.4:
            strategy = 'multi_step'
            depth = 'moderate'
        else:
            strategy = 'simple'
            depth = 'shallow'
        
        return {
            'complexity': features['query_complexity'],
            'domain_specificity': features['domain_specificity'],
            'visual_complexity': features['visual_complexity'],
            'data_complexity': features['data_complexity'],
            'prompt_strategy': strategy,
            'reasoning_depth': depth,
            'confidence': 0.6  # Lower confidence for fallback
        }
    
    def _interpret_strategy(self, value: float) -> str:
        """Interpret fuzzy output for prompt strategy."""
        if value < 0.4:
            return 'simple'
        elif value < 0.7:
            return 'multi_step'
        else:
            return 'advanced'
    
    def _interpret_depth(self, value: float) -> str:
        """Interpret fuzzy output for reasoning depth."""
        if value < 0.4:
            return 'shallow'
        elif value < 0.7:
            return 'moderate'
        else:
            return 'deep'
    
    def is_ready(self) -> bool:
        """Check if the fuzzy logic engine is ready."""
        return self.ready
    
    def get_fuzzy_rules(self) -> List[FuzzyRule]:
        """Get all fuzzy rules."""
        return self.fuzzy_rules.copy()
    
    def add_fuzzy_rule(self, rule: FuzzyRule) -> bool:
        """Add a new fuzzy rule to the system."""
        if len(self.fuzzy_rules) >= self.max_rules:
            logger.warning(f"Maximum number of rules ({self.max_rules}) reached")
            return False
        
        self.fuzzy_rules.append(rule)
        logger.info(f"Added fuzzy rule: {rule.id}")
        
        # Recreate control system
        asyncio.create_task(self._create_control_system())
        
        return True
    
    async def optimize_rules(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Optimize fuzzy rules based on feedback data."""
        logger.info("Optimizing fuzzy rules based on feedback...")
        
        # This is a placeholder for rule optimization
        # In practice, this would use machine learning to improve rules
        
        optimization_result = {
            "rules_modified": 0,
            "rules_added": 0,
            "rules_removed": 0,
            "performance_improvement": 0.0
        }
        
        return optimization_result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "ready": self.ready,
            "fuzzy_available": FUZZY_AVAILABLE,
            "total_rules": len(self.fuzzy_rules),
            "max_rules": self.max_rules,
            "confidence_threshold": self.confidence_threshold
        }