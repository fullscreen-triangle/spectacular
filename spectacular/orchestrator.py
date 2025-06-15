"""
Metacognitive Orchestrator - Central coordination hub for the Spectacular system.

This module implements the core metacognitive reasoning engine that coordinates
all other modules and manages the overall reasoning process.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

from .pretoria import FuzzyLogicEngine
from .mzekezeke import BayesianEvidenceNetwork
from .zengeza import MDPModule
from .nicotine import ContextualSketchingModule
from .hf_integration import HuggingFaceHub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningState(Enum):
    """Enumeration of possible reasoning states."""
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    GENERATING = "generating"
    VALIDATING = "validating"
    REFINING = "refining"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class QueryContext:
    """Context information for processing queries."""
    query: str
    complexity_score: float
    domain_specificity: float
    visual_complexity: float
    data_schema: Optional[Dict] = None
    user_preferences: Optional[Dict] = None
    session_history: Optional[List] = None


@dataclass
class ReasoningStep:
    """Individual step in the reasoning process."""
    step_id: str
    module: str
    action: str
    input_data: Any
    output_data: Any
    confidence: float
    timestamp: datetime
    processing_time: float


class MetacognitiveOrchestrator:
    """
    Central metacognitive orchestrator that coordinates all reasoning modules.
    
    This class implements the core metacognitive loop:
    1. Metacognitive analysis of the input
    2. Dynamic module selection and coordination
    3. Process monitoring and self-evaluation
    4. Adaptive refinement based on feedback
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the orchestrator with all reasoning modules."""
        self.config = config or {}
        self.state = ReasoningState.INITIALIZING
        self.reasoning_history: List[ReasoningStep] = []
        self.session_id = self._generate_session_id()
        
        # Initialize all reasoning modules
        self.pretoria = FuzzyLogicEngine(config.get('pretoria', {}))
        self.mzekezeke = BayesianEvidenceNetwork(config.get('mzekezeke', {}))
        self.zengeza = MDPModule(config.get('zengeza', {}))
        self.nicotine = ContextualSketchingModule(config.get('nicotine', {}))
        self.hf_hub = HuggingFaceHub(config.get('hf_integration', {}))
        
        # Metacognitive parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.max_refinement_cycles = config.get('max_refinement_cycles', 3)
        self.reasoning_weights = config.get('reasoning_weights', {
            'fuzzy': 0.2,
            'bayesian': 0.25,
            'mdp': 0.2,
            'contextual': 0.2,
            'hf_models': 0.15
        })
        
        logger.info(f"Metacognitive Orchestrator initialized with session {self.session_id}")
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point for processing queries through metacognitive reasoning.
        
        Args:
            query: Natural language query for visualization
            context: Additional context information
            
        Returns:
            Dictionary containing generated code, metadata, and reasoning trace
        """
        try:
            self.state = ReasoningState.ANALYZING
            
            # Step 1: Metacognitive analysis of query
            query_context = await self._analyze_query(query, context)
            
            # Step 2: Dynamic module coordination
            reasoning_plan = await self._create_reasoning_plan(query_context)
            
            # Step 3: Execute reasoning chain
            result = await self._execute_reasoning_chain(reasoning_plan, query_context)
            
            # Step 4: Metacognitive validation
            validated_result = await self._validate_and_refine(result, query_context)
            
            self.state = ReasoningState.COMPLETED
            logger.info(f"Query processed successfully: {query[:50]}...")
            
            return validated_result
            
        except Exception as e:
            self.state = ReasoningState.ERROR
            logger.error(f"Error processing query: {e}")
            return {"error": str(e), "state": self.state.value}
    
    async def _analyze_query(self, query: str, context: Optional[Dict]) -> QueryContext:
        """Perform metacognitive analysis of the input query."""
        logger.info("Performing metacognitive query analysis...")
        
        # Use fuzzy logic to assess query characteristics
        fuzzy_analysis = await self.pretoria.analyze_query_complexity(query)
        
        # Extract complexity metrics
        complexity_score = fuzzy_analysis.get('complexity', 0.5)
        domain_specificity = fuzzy_analysis.get('domain_specificity', 0.5)
        visual_complexity = fuzzy_analysis.get('visual_complexity', 0.5)
        
        query_context = QueryContext(
            query=query,
            complexity_score=complexity_score,
            domain_specificity=domain_specificity,
            visual_complexity=visual_complexity,
            data_schema=context.get('data_schema') if context else None,
            user_preferences=context.get('preferences') if context else None,
            session_history=self.reasoning_history[-10:]  # Last 10 steps
        )
        
        # Record reasoning step
        step = ReasoningStep(
            step_id=self._generate_step_id(),
            module="orchestrator",
            action="query_analysis",
            input_data={"query": query, "context": context},
            output_data=query_context,
            confidence=0.9,
            timestamp=datetime.now(),
            processing_time=0.1
        )
        self.reasoning_history.append(step)
        
        return query_context
    
    async def _create_reasoning_plan(self, query_context: QueryContext) -> List[Dict[str, Any]]:
        """Create a dynamic reasoning plan based on query characteristics."""
        logger.info("Creating dynamic reasoning plan...")
        
        plan = []
        
        # Always start with initial sketch
        plan.append({
            "module": "nicotine",
            "action": "create_initial_sketch",
            "priority": 1.0,
            "required": True
        })
        
        # Add fuzzy logic analysis
        plan.append({
            "module": "pretoria",
            "action": "generate_fuzzy_rules",
            "priority": 0.8,
            "required": True
        })
        
        # Bayesian evidence accumulation
        plan.append({
            "module": "mzekezeke",
            "action": "accumulate_evidence",
            "priority": 0.9,
            "required": True
        })
        
        # MDP planning based on complexity
        if query_context.complexity_score > 0.6:
            plan.append({
                "module": "zengeza",
                "action": "optimize_generation_strategy",
                "priority": 0.8,
                "required": True
            })
        
        # HuggingFace model selection
        hf_models = await self._select_hf_models(query_context)
        for model in hf_models:
            plan.append({
                "module": "hf_hub",
                "action": f"invoke_{model}",
                "priority": 0.7,
                "required": False
            })
        
        # Sort plan by priority
        plan.sort(key=lambda x: x["priority"], reverse=True)
        
        return plan
    
    async def _execute_reasoning_chain(self, plan: List[Dict], query_context: QueryContext) -> Dict[str, Any]:
        """Execute the reasoning chain according to the plan."""
        logger.info(f"Executing reasoning chain with {len(plan)} steps...")
        
        self.state = ReasoningState.GENERATING
        results = {"steps": {}, "final_output": None}
        
        for step_config in plan:
            module_name = step_config["module"]
            action = step_config["action"]
            
            try:
                start_time = datetime.now()
                
                # Execute step based on module
                if module_name == "nicotine":
                    step_result = await self._execute_nicotine_step(action, query_context, results)
                elif module_name == "pretoria":
                    step_result = await self._execute_pretoria_step(action, query_context, results)
                elif module_name == "mzekezeke":
                    step_result = await self._execute_mzekezeke_step(action, query_context, results)
                elif module_name == "zengeza":
                    step_result = await self._execute_zengeza_step(action, query_context, results)
                elif module_name == "hf_hub":
                    step_result = await self._execute_hf_step(action, query_context, results)
                else:
                    logger.warning(f"Unknown module: {module_name}")
                    continue
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Record step
                reasoning_step = ReasoningStep(
                    step_id=self._generate_step_id(),
                    module=module_name,
                    action=action,
                    input_data=query_context,
                    output_data=step_result,
                    confidence=step_result.get("confidence", 0.5),
                    timestamp=datetime.now(),
                    processing_time=processing_time
                )
                self.reasoning_history.append(reasoning_step)
                results["steps"][f"{module_name}_{action}"] = step_result
                
            except Exception as e:
                logger.error(f"Error in {module_name}.{action}: {e}")
                if step_config["required"]:
                    raise
        
        # Synthesize final output
        results["final_output"] = await self._synthesize_results(results, query_context)
        
        return results
    
    async def _validate_and_refine(self, result: Dict, query_context: QueryContext) -> Dict[str, Any]:
        """Perform metacognitive validation and refinement."""
        logger.info("Performing metacognitive validation...")
        
        self.state = ReasoningState.VALIDATING
        
        # Calculate overall confidence
        step_confidences = [
            step_data.get("confidence", 0.5) 
            for step_data in result["steps"].values()
        ]
        overall_confidence = np.mean(step_confidences) if step_confidences else 0.5
        
        # Check if refinement is needed
        if overall_confidence < self.confidence_threshold:
            logger.info(f"Confidence {overall_confidence:.2f} below threshold {self.confidence_threshold}")
            self.state = ReasoningState.REFINING
            
            # Perform refinement (simplified for now)
            refinement_result = await self._refine_output(result, query_context)
            result.update(refinement_result)
        
        # Add metacognitive metadata
        result["metacognitive_data"] = {
            "session_id": self.session_id,
            "reasoning_steps": len(self.reasoning_history),
            "overall_confidence": overall_confidence,
            "processing_state": self.state.value,
            "refinement_cycles": getattr(self, '_refinement_cycles', 0)
        }
        
        return result
    
    async def _execute_nicotine_step(self, action: str, context: QueryContext, results: Dict) -> Dict:
        """Execute a step in the Nicotine module."""
        if action == "create_initial_sketch":
            return await self.nicotine.create_initial_sketch(context.query)
        elif action == "validate_context":
            return await self.nicotine.validate_context(results)
        else:
            return {"error": f"Unknown nicotine action: {action}"}
    
    async def _execute_pretoria_step(self, action: str, context: QueryContext, results: Dict) -> Dict:
        """Execute a step in the Pretoria module."""
        if action == "generate_fuzzy_rules":
            return await self.pretoria.generate_code_strategy(context)
        else:
            return {"error": f"Unknown pretoria action: {action}"}
    
    async def _execute_mzekezeke_step(self, action: str, context: QueryContext, results: Dict) -> Dict:
        """Execute a step in the Mzekezeke module."""
        if action == "accumulate_evidence":
            return await self.mzekezeke.update_beliefs(context, results)
        else:
            return {"error": f"Unknown mzekezeke action: {action}"}
    
    async def _execute_zengeza_step(self, action: str, context: QueryContext, results: Dict) -> Dict:
        """Execute a step in the Zengeza module."""
        if action == "optimize_generation_strategy":
            return await self.zengeza.optimize_strategy(context, results)
        else:
            return {"error": f"Unknown zengeza action: {action}"}
    
    async def _execute_hf_step(self, action: str, context: QueryContext, results: Dict) -> Dict:
        """Execute a step in the HuggingFace hub."""
        model_name = action.replace("invoke_", "")
        return await self.hf_hub.invoke_model(model_name, context, results)
    
    async def _select_hf_models(self, context: QueryContext) -> List[str]:
        """Select appropriate HuggingFace models based on query context."""
        models = []
        
        if context.complexity_score > 0.7:
            models.extend(["CodeT5-large", "BLIP-2"])
        else:
            models.append("CodeT5-base")
        
        if context.visual_complexity > 0.6:
            models.append("CLIP")
        
        # Always include Plot-BERT for domain understanding
        models.append("Plot-BERT")
        
        return models
    
    async def _synthesize_results(self, results: Dict, context: QueryContext) -> Dict[str, Any]:
        """Synthesize final output from all reasoning steps."""
        logger.info("Synthesizing final results...")
        
        # This is a simplified synthesis - in practice, this would be much more sophisticated
        d3_code = "// Generated D3.js code placeholder\n"
        d3_code += f"// Query: {context.query}\n"
        d3_code += "// Generated by Spectacular metacognitive reasoning\n"
        
        # Extract key information from steps
        sketch_data = results["steps"].get("nicotine_create_initial_sketch", {})
        fuzzy_strategy = results["steps"].get("pretoria_generate_fuzzy_rules", {})
        
        return {
            "d3_code": d3_code,
            "visualization_type": sketch_data.get("visualization_type", "unknown"),
            "generation_strategy": fuzzy_strategy.get("strategy", "default"),
            "confidence": 0.85,
            "metadata": {
                "reasoning_modules_used": list(results["steps"].keys()),
                "query_complexity": context.complexity_score
            }
        }
    
    async def _refine_output(self, result: Dict, context: QueryContext) -> Dict[str, Any]:
        """Refine output through additional reasoning cycles."""
        logger.info("Performing output refinement...")
        
        # Simplified refinement logic
        refinement_cycles = getattr(self, '_refinement_cycles', 0) + 1
        self._refinement_cycles = refinement_cycles
        
        if refinement_cycles < self.max_refinement_cycles:
            # Could re-run parts of the reasoning chain here
            pass
        
        return {"refinement_applied": True, "cycles": refinement_cycles}
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        from uuid import uuid4
        return str(uuid4())[:8]
    
    def _generate_step_id(self) -> str:
        """Generate a unique step ID."""
        from uuid import uuid4
        return str(uuid4())[:8]
    
    def get_reasoning_trace(self) -> List[ReasoningStep]:
        """Get the complete reasoning trace for debugging/analysis."""
        return self.reasoning_history.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics."""
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "reasoning_steps": len(self.reasoning_history),
            "modules_active": {
                "pretoria": self.pretoria.is_ready(),
                "mzekezeke": self.mzekezeke.is_ready(),
                "zengeza": self.zengeza.is_ready(),
                "nicotine": self.nicotine.is_ready(),
                "hf_hub": self.hf_hub.is_ready()
            }
        } 