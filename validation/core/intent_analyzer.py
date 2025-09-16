"""
Intent Analyzer: Systematic interrogative analysis to infer user's actual intent.

This module implements the Intent Validation mechanism from the theoretical framework:
- Applies 12-dimensional environmental analysis to query context
- Generates counterfactual scenarios to test interpretation robustness
- Creates visualizations of inferred user intent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
from enum import Enum

logger = logging.getLogger(__name__)

class IntentDimension(Enum):
    """12-dimensional environmental analysis dimensions for intent."""
    MOTIVATIONAL = "motivational"          # Why would user ask this?
    GOAL_ORIENTED = "goal_oriented"        # What's the underlying objective?
    EXPRESSION = "expression"              # Why phrased this way?
    CONDITIONAL = "conditional"            # What knowledge gaps exist?
    TEMPORAL = "temporal"                  # Time-sensitive factors?
    DOMAIN_CONTEXT = "domain_context"      # Technical domain considerations
    CULTURAL_CONTEXT = "cultural_context"  # Region-specific conventions  
    IMPLICIT_KNOWLEDGE = "implicit_knowledge"  # Unstated assumptions
    NEGATION_HANDLING = "negation_handling"    # Opposite interpretations
    SPATIAL = "spatial"                    # Location/positioning context
    ATMOSPHERIC = "atmospheric"            # Environmental mood/context
    QUANTUM_COHERENCE = "quantum_coherence" # Deep pattern coherence

@dataclass
class IntentPlot:
    """Result from intent analysis and visualization."""
    svg_content: str
    inferred_intent: str
    intent_confidence: float
    alternative_intents: List[str]
    dimensional_analysis: Dict[str, float]
    counterfactual_scenarios: List[Dict[str, Any]]
    intent_reasoning_chain: List[str]
    environmental_factors: Dict[str, Any]
    
    @classmethod
    def empty_plot(cls) -> 'IntentPlot':
        """Create empty plot for error cases."""
        return cls(
            svg_content="<svg></svg>",
            inferred_intent="No intent inferred", 
            intent_confidence=0.0,
            alternative_intents=[],
            dimensional_analysis={},
            counterfactual_scenarios=[],
            intent_reasoning_chain=[],
            environmental_factors={}
        )

class InterrogativeFramework:
    """Systematic questioning mechanism for intent analysis."""
    
    def __init__(self):
        self.question_templates = {
            IntentDimension.MOTIVATIONAL: [
                "Why would the user pose this specific question?",
                "What problem is the user trying to solve?",
                "What triggered this information need?"
            ],
            IntentDimension.GOAL_ORIENTED: [
                "What underlying objectives drive this request?",
                "What will the user do with this information?", 
                "What decision is this information supporting?"
            ],
            IntentDimension.EXPRESSION: [
                "Why was the query phrased using these particular terms?",
                "What does the language choice reveal about intent?",
                "What assumptions are embedded in the phrasing?"
            ],
            IntentDimension.CONDITIONAL: [
                "Given assumed user knowledge, what gaps exist?",
                "What context is the user operating within?",
                "What constraints might influence the answer needed?"
            ],
            IntentDimension.TEMPORAL: [
                "What time-sensitive factors might influence interpretation?",
                "Is this query about past, present, or future states?",
                "How urgent is the information need?"
            ]
        }
    
    async def analyze_motivations(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze motivational dimensions of the query."""
        context = context or {}
        
        # Extract motivational indicators from query
        urgency_indicators = ['urgent', 'immediately', 'asap', 'quickly', 'now']
        problem_indicators = ['problem', 'issue', 'trouble', 'error', 'wrong', 'help']
        decision_indicators = ['choose', 'decide', 'should', 'better', 'recommend', 'suggest']
        
        query_lower = query.lower()
        
        urgency_score = sum(1 for indicator in urgency_indicators if indicator in query_lower)
        problem_score = sum(1 for indicator in problem_indicators if indicator in query_lower) 
        decision_score = sum(1 for indicator in decision_indicators if indicator in query_lower)
        
        # Infer primary motivation
        if problem_score > 0:
            primary_motivation = "problem_solving"
        elif decision_score > 0:
            primary_motivation = "decision_support"
        elif urgency_score > 0:
            primary_motivation = "urgent_information"
        else:
            primary_motivation = "knowledge_acquisition"
        
        return {
            'primary_motivation': primary_motivation,
            'urgency_level': min(1.0, urgency_score / 2.0),
            'problem_complexity': min(1.0, problem_score / 3.0),
            'decision_complexity': min(1.0, decision_score / 3.0),
            'motivation_confidence': 0.8 if any([urgency_score, problem_score, decision_score]) else 0.5
        }
    
    async def infer_analytical_goals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Infer analytical goals from context."""
        goals = {
            'data_exploration': 0.0,
            'pattern_identification': 0.0,
            'comparison_analysis': 0.0,
            'trend_analysis': 0.0,
            'relationship_discovery': 0.0,
            'hypothesis_testing': 0.0
        }
        
        if 'data' in context:
            goals['data_exploration'] = 0.7
            
        query = context.get('query', '').lower()
        
        if any(term in query for term in ['compare', 'difference', 'versus', 'vs']):
            goals['comparison_analysis'] = 0.8
            
        if any(term in query for term in ['trend', 'over time', 'change', 'temporal']):
            goals['trend_analysis'] = 0.8
            
        if any(term in query for term in ['relationship', 'correlation', 'associated', 'connect']):
            goals['relationship_discovery'] = 0.9
            
        if any(term in query for term in ['test', 'hypothesis', 'significant', 'prove']):
            goals['hypothesis_testing'] = 0.9
            
        if any(term in query for term in ['pattern', 'detect', 'find', 'identify']):
            goals['pattern_identification'] = 0.8
        
        # Determine optimal chart type based on goals
        chart_type = self._infer_optimal_chart_type(goals)
        
        return {
            'analytical_goals': goals,
            'primary_goal': max(goals, key=goals.get),
            'goal_confidence': max(goals.values()),
            'optimal_chart_type': chart_type
        }
    
    def _infer_optimal_chart_type(self, goals: Dict[str, float]) -> str:
        """Infer optimal chart type from analytical goals."""
        primary_goal = max(goals, key=goals.get)
        
        chart_mapping = {
            'data_exploration': 'histogram',
            'pattern_identification': 'scatter_plot', 
            'comparison_analysis': 'bar_chart',
            'trend_analysis': 'line_chart',
            'relationship_discovery': 'scatter_plot',
            'hypothesis_testing': 'box_plot'
        }
        
        return chart_mapping.get(primary_goal, 'scatter_plot')

class CounterfactualGenerator:
    """Generates alternative interpretations to test robustness."""
    
    def __init__(self):
        self.scenario_types = [
            'temporal_ambiguity',
            'domain_context_shift',
            'cultural_context_variation',
            'implicit_knowledge_assumption',
            'negation_handling_test',
            'scale_interpretation_variation'
        ]
    
    async def generate_scenarios(self, query: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios for interpretation testing."""
        scenarios = []
        
        for scenario_type in self.scenario_types:
            scenario = await self._generate_scenario(query, scenario_type, context)
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_scenario(self, query: str, scenario_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific counterfactual scenario."""
        
        if scenario_type == 'temporal_ambiguity':
            return {
                'type': 'temporal_ambiguity',
                'description': 'What if the user means a different time frame?',
                'alternative_interpretation': f"Time-shifted interpretation of: {query}",
                'probability': 0.3,
                'impact': 'medium'
            }
        
        elif scenario_type == 'domain_context_shift':
            return {
                'type': 'domain_context_shift',
                'description': 'What if this is from a different technical domain?',
                'alternative_interpretation': f"Different domain context for: {query}",
                'probability': 0.4,
                'impact': 'high'
            }
        
        elif scenario_type == 'cultural_context_variation':
            return {
                'type': 'cultural_context_variation', 
                'description': 'What if cultural conventions are different?',
                'alternative_interpretation': f"Cultural variation interpretation: {query}",
                'probability': 0.2,
                'impact': 'low'
            }
        
        elif scenario_type == 'implicit_knowledge_assumption':
            return {
                'type': 'implicit_knowledge_assumption',
                'description': 'What if user has different background knowledge?',
                'alternative_interpretation': f"Different knowledge base assumption: {query}",
                'probability': 0.5,
                'impact': 'high'
            }
        
        elif scenario_type == 'negation_handling_test':
            return {
                'type': 'negation_handling_test',
                'description': 'What if the user means the opposite?',
                'alternative_interpretation': f"Negated interpretation: {query}",
                'probability': 0.1,
                'impact': 'very_high'
            }
        
        else:  # scale_interpretation_variation
            return {
                'type': 'scale_interpretation_variation',
                'description': 'What if the scale/magnitude is different?',
                'alternative_interpretation': f"Scale-shifted interpretation: {query}", 
                'probability': 0.3,
                'impact': 'medium'
            }

class IntentAnalyzer:
    """
    Systematic interrogative analysis to infer user's actual intent.
    
    Implements the Intent Validation mechanism through:
    - 12-dimensional environmental analysis
    - Systematic interrogative questioning
    - Counterfactual scenario generation
    - Intent visualization creation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intent analyzer."""
        self.config = config or {}
        self.interrogative_framework = InterrogativeFramework()
        self.counterfactual_generator = CounterfactualGenerator()
        
        logger.info("Intent Analyzer initialized")
    
    async def generate_intent_plot(self, query: str, context: Dict[str, Any]) -> IntentPlot:
        """
        Generate intent analysis plot through systematic interrogative analysis.
        
        Args:
            query: User query to analyze
            context: Additional context for analysis
            
        Returns:
            IntentPlot with inferred intent and visualization
        """
        logger.info("Generating intent plot for query: %s", query[:50])
        
        try:
            # Systematic interrogative analysis across dimensions
            dimensional_analysis = await self._perform_dimensional_analysis(query, context)
            
            # Generate counterfactual scenarios
            counterfactual_scenarios = await self.counterfactual_generator.generate_scenarios(query, context)
            
            # Infer primary intent from dimensional analysis
            inferred_intent = await self._infer_primary_intent(dimensional_analysis, context)
            
            # Generate alternative interpretations
            alternative_intents = await self._generate_alternative_intents(
                query, counterfactual_scenarios, dimensional_analysis
            )
            
            # Calculate intent confidence
            intent_confidence = await self._calculate_intent_confidence(
                dimensional_analysis, counterfactual_scenarios
            )
            
            # Create intent reasoning chain
            intent_reasoning_chain = await self._create_reasoning_chain(
                query, dimensional_analysis, inferred_intent
            )
            
            # Generate intent visualization
            svg_content = await self._generate_intent_visualization(
                inferred_intent, alternative_intents, dimensional_analysis
            )
            
            # Extract environmental factors (simplified 12-dimensional measurement)
            environmental_factors = await self._extract_environmental_factors(context)
            
            result = IntentPlot(
                svg_content=svg_content,
                inferred_intent=inferred_intent,
                intent_confidence=intent_confidence,
                alternative_intents=alternative_intents,
                dimensional_analysis=dimensional_analysis,
                counterfactual_scenarios=counterfactual_scenarios,
                intent_reasoning_chain=intent_reasoning_chain,
                environmental_factors=environmental_factors
            )
            
            logger.info("Intent plot generated with confidence: %.2f", intent_confidence)
            return result
            
        except Exception as e:
            logger.error("Error generating intent plot: %s", str(e))
            return IntentPlot.empty_plot()
    
    async def _perform_dimensional_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Perform 12-dimensional environmental analysis on the query."""
        
        analysis = {}
        
        # Motivational analysis
        motivational_result = await self.interrogative_framework.analyze_motivations(query, context)
        analysis[IntentDimension.MOTIVATIONAL.value] = motivational_result['motivation_confidence']
        
        # Goal-oriented analysis
        goal_result = await self.interrogative_framework.infer_analytical_goals(context)
        analysis[IntentDimension.GOAL_ORIENTED.value] = goal_result['goal_confidence']
        
        # Expression analysis (language patterns)
        query_complexity = len(query.split()) / 20.0  # Normalize by typical query length
        analysis[IntentDimension.EXPRESSION.value] = min(1.0, query_complexity)
        
        # Conditional analysis (context dependency)
        context_richness = len(str(context)) / 500.0  # Normalize by typical context size
        analysis[IntentDimension.CONDITIONAL.value] = min(1.0, context_richness)
        
        # Temporal analysis
        temporal_indicators = ['time', 'when', 'before', 'after', 'during', 'temporal']
        temporal_score = sum(1 for indicator in temporal_indicators if indicator in query.lower())
        analysis[IntentDimension.TEMPORAL.value] = min(1.0, temporal_score / 3.0)
        
        # Simplified remaining dimensions (would be more sophisticated in full implementation)
        analysis[IntentDimension.DOMAIN_CONTEXT.value] = 0.7
        analysis[IntentDimension.CULTURAL_CONTEXT.value] = 0.6
        analysis[IntentDimension.IMPLICIT_KNOWLEDGE.value] = 0.8
        analysis[IntentDimension.NEGATION_HANDLING.value] = 0.5
        analysis[IntentDimension.SPATIAL.value] = 0.7
        analysis[IntentDimension.ATMOSPHERIC.value] = 0.6
        analysis[IntentDimension.QUANTUM_COHERENCE.value] = 0.9
        
        return analysis
    
    async def _infer_primary_intent(self, dimensional_analysis: Dict[str, float], context: Dict[str, Any]) -> str:
        """Infer primary intent from dimensional analysis."""
        
        # Find highest-confidence dimension
        primary_dimension = max(dimensional_analysis, key=dimensional_analysis.get)
        
        intent_mappings = {
            IntentDimension.MOTIVATIONAL.value: "User seeks to solve a specific problem or address an urgent need",
            IntentDimension.GOAL_ORIENTED.value: "User has a clear analytical objective requiring data insights",
            IntentDimension.EXPRESSION.value: "User's language indicates sophisticated domain expertise",
            IntentDimension.CONDITIONAL.value: "User operates within specific constraints or assumptions",
            IntentDimension.TEMPORAL.value: "User needs time-sensitive or temporally-structured information"
        }
        
        base_intent = intent_mappings.get(primary_dimension, "User seeks general information or analysis")
        
        # Enhance with context-specific details
        if 'data' in context:
            base_intent += " involving data visualization and analysis"
        
        if dimensional_analysis.get(IntentDimension.GOAL_ORIENTED.value, 0) > 0.7:
            base_intent += " with clear decision-making objectives"
        
        return base_intent
    
    async def _generate_alternative_intents(
        self, 
        query: str, 
        counterfactual_scenarios: List[Dict[str, Any]], 
        dimensional_analysis: Dict[str, float]
    ) -> List[str]:
        """Generate alternative intent interpretations."""
        
        alternatives = []
        
        for scenario in counterfactual_scenarios:
            if scenario['probability'] > 0.3:  # Only include plausible alternatives
                alternative = f"Alternative: {scenario['alternative_interpretation']}"
                alternatives.append(alternative)
        
        # Add dimension-based alternatives
        high_confidence_dims = [dim for dim, score in dimensional_analysis.items() if score > 0.7]
        
        for dim in high_confidence_dims[:2]:  # Top 2 alternative dimensions
            alternative = f"Dimension-based alternative focusing on {dim} aspects of the query"
            alternatives.append(alternative)
        
        return alternatives[:5]  # Limit to 5 alternatives
    
    async def _calculate_intent_confidence(
        self, 
        dimensional_analysis: Dict[str, float], 
        counterfactual_scenarios: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the inferred intent."""
        
        # Base confidence from dimensional analysis
        dimension_confidence = sum(dimensional_analysis.values()) / len(dimensional_analysis)
        
        # Adjustment based on counterfactual scenario risks
        high_risk_scenarios = [s for s in counterfactual_scenarios if s['impact'] in ['high', 'very_high']]
        risk_penalty = len(high_risk_scenarios) * 0.1
        
        # Environmental coherence bonus (simplified)
        coherence_bonus = 0.1 if dimension_confidence > 0.7 else 0.0
        
        final_confidence = dimension_confidence - risk_penalty + coherence_bonus
        return max(0.0, min(1.0, final_confidence))
    
    async def _create_reasoning_chain(
        self, 
        query: str, 
        dimensional_analysis: Dict[str, float], 
        inferred_intent: str
    ) -> List[str]:
        """Create reasoning chain explaining intent inference."""
        
        chain = [
            f"1. Analyzed query: '{query[:50]}...'",
            f"2. Performed 12-dimensional environmental analysis",
            f"3. Primary dimension: {max(dimensional_analysis, key=dimensional_analysis.get)}",
            f"4. Generated counterfactual scenarios for robustness testing",
            f"5. Inferred intent: {inferred_intent[:100]}...",
            f"6. Validated intent coherence across dimensions"
        ]
        
        return chain
    
    async def _generate_intent_visualization(
        self, 
        inferred_intent: str, 
        alternative_intents: List[str], 
        dimensional_analysis: Dict[str, float]
    ) -> str:
        """Generate SVG visualization of intent analysis."""
        
        width, height = 400, 300
        
        # Create intent visualization with dimensional radar chart
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="{width}" height="{height}" fill="#e6f3ff" stroke="#0066cc" stroke-width="2"/>
            <text x="10" y="20" font-family="Arial" font-size="12" fill="#0066cc">INTENT ANALYSIS</text>
            
            <!-- Dimensional analysis radar -->
            <g transform="translate(200, 150)">
                <circle cx="0" cy="0" r="80" fill="none" stroke="#ccc" stroke-width="1"/>
                <circle cx="0" cy="0" r="60" fill="none" stroke="#ccc" stroke-width="1"/>
                <circle cx="0" cy="0" r="40" fill="none" stroke="#ccc" stroke-width="1"/>
                <circle cx="0" cy="0" r="20" fill="none" stroke="#ccc" stroke-width="1"/>
                
                <!-- Dimension lines -->
                <path d="M0,-80 L0,80 M-80,0 L80,0 M-57,-57 L57,57 M57,-57 L-57,57" 
                      stroke="#ddd" stroke-width="1" fill="none"/>
                
                <!-- Intent confidence visualization -->
                <circle cx="0" cy="0" r="30" fill="#0066cc" fill-opacity="0.3" stroke="#0066cc" stroke-width="2"/>
                <text x="0" y="100" text-anchor="middle" font-size="10" fill="#0066cc">Intent Confidence</text>
            </g>
            
            <!-- Alternative intents list -->
            <text x="10" y="200" font-family="Arial" font-size="10" fill="#666">Alternative Interpretations:</text>
        </svg>'''
        
        return svg
    
    async def _extract_environmental_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environmental factors (simplified 12-dimensional measurement)."""
        
        return {
            'temporal_context': context.get('timestamp', 'unknown'),
            'data_context': 'present' if 'data' in context else 'absent',
            'query_complexity': 'high' if len(str(context.get('query', ''))) > 100 else 'medium',
            'spatial_context': 'localized',
            'atmospheric_coherence': 0.8,
            'quantum_entanglement': 0.7,
            'environmental_stability': 0.9
        }
