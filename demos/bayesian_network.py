"""
Simplified Bayesian Evidence Network for Demo

This is a simplified version that demonstrates the core concepts:
- Fuzzy logic nodes with confidence/uncertainty tracking
- Evidence accumulation and belief updates
- Recursive processing when uncertainty is high
- External validation through LLM queries
- Step-by-step reasoning trace
"""

import time
import random
import math
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class NodeState(Enum):
    UNINITIALIZED = "uninitialized"
    PROCESSING = "processing"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    VALIDATING = "validating"
    CONVERGED = "converged"
    RECURSIVE_LOOP = "recursive_loop"
    FAILED = "failed"

@dataclass
class Evidence:
    """Simplified evidence structure"""
    content: str
    confidence: float  # 0.0 to 1.0
    uncertainty: float  # 0.0 to 1.0
    source: str
    timestamp: datetime
    
    def strength(self) -> float:
        return self.confidence * (1.0 - self.uncertainty)

@dataclass
class ProcessingStep:
    """Records each step of processing for transparency"""
    step_number: int
    node_id: str
    action: str
    input_data: Any
    output_data: Any
    confidence_before: float
    confidence_after: float
    uncertainty_before: float
    uncertainty_after: float
    reasoning: str
    duration: float
    timestamp: datetime

class SimpleBayesianNode:
    """Simplified Bayesian node with fuzzy logic"""
    
    def __init__(self, node_id: str, node_type: str, description: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.node_type = node_type
        self.description = description
        self.config = config
        
        # Bayesian state
        self.confidence = 0.5  # Start neutral
        self.uncertainty = 1.0  # High uncertainty initially
        self.prior_belief = 0.5
        
        # Processing state
        self.state = NodeState.UNINITIALIZED
        self.evidence_list: List[Evidence] = []
        self.processing_history: List[ProcessingStep] = []
        self.recursive_count = 0
        self.max_recursive_depth = config.get('max_recursive_depth', 3)
        
        # Thresholds
        self.convergence_threshold = config.get('convergence_threshold', 0.8)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.3)
    
    async def add_evidence(self, evidence: Evidence) -> ProcessingStep:
        """Add evidence and update beliefs"""
        step_start = time.time()
        confidence_before = self.confidence
        uncertainty_before = self.uncertainty
        
        self.evidence_list.append(evidence)
        
        # Bayesian update
        old_confidence = self.confidence
        
        # Simple Bayesian update: P(H|E) ∝ P(E|H) * P(H)
        likelihood = evidence.confidence
        
        # Update using weighted average with evidence strength
        evidence_weight = evidence.strength()
        self.confidence = (
            self.confidence * (1 - evidence_weight) + 
            likelihood * evidence_weight
        )
        
        # Update uncertainty (reduces with more evidence)
        self.uncertainty = max(0.1, self.uncertainty * (1 - evidence_weight * 0.3))
        
        reasoning = f"Updated beliefs based on evidence '{evidence.content[:50]}...' "
        reasoning += f"Evidence strength: {evidence_weight:.3f}, "
        reasoning += f"Confidence: {old_confidence:.3f} → {self.confidence:.3f}"
        
        step = ProcessingStep(
            step_number=len(self.processing_history) + 1,
            node_id=self.node_id,
            action="add_evidence",
            input_data=evidence.content,
            output_data={"confidence": self.confidence, "uncertainty": self.uncertainty},
            confidence_before=confidence_before,
            confidence_after=self.confidence,
            uncertainty_before=uncertainty_before,
            uncertainty_after=self.uncertainty,
            reasoning=reasoning,
            duration=time.time() - step_start,
            timestamp=datetime.now()
        )
        
        self.processing_history.append(step)
        return step
    
    async def check_convergence(self) -> Tuple[bool, ProcessingStep]:
        """Check if node has converged or needs recursive processing"""
        step_start = time.time()
        confidence_before = self.confidence
        uncertainty_before = self.uncertainty
        
        converged = (
            self.confidence >= self.convergence_threshold and
            self.uncertainty <= self.uncertainty_threshold
        )
        
        if converged:
            self.state = NodeState.CONVERGED
            reasoning = f"✅ Node converged: confidence={self.confidence:.3f}, uncertainty={self.uncertainty:.3f}"
        elif self.uncertainty > 0.7 and self.recursive_count < self.max_recursive_depth:
            self.state = NodeState.RECURSIVE_LOOP
            self.recursive_count += 1
            reasoning = f"🔄 High uncertainty ({self.uncertainty:.3f}) → Entering recursive loop #{self.recursive_count}"
        elif self.confidence < 0.3:
            self.state = NodeState.NEEDS_MORE_EVIDENCE
            reasoning = f"❓ Low confidence ({self.confidence:.3f}) → Need more evidence"
        else:
            self.state = NodeState.CONVERGED  # Accept current state
            reasoning = f"⚡ Partial convergence accepted: confidence={self.confidence:.3f}"
        
        step = ProcessingStep(
            step_number=len(self.processing_history) + 1,
            node_id=self.node_id,
            action="check_convergence",
            input_data={"confidence": self.confidence, "uncertainty": self.uncertainty},
            output_data={"converged": converged, "state": self.state.value},
            confidence_before=confidence_before,
            confidence_after=self.confidence,
            uncertainty_before=uncertainty_before,
            uncertainty_after=self.uncertainty,
            reasoning=reasoning,
            duration=time.time() - step_start,
            timestamp=datetime.now()
        )
        
        self.processing_history.append(step)
        return converged, step
    
    def get_ascii_visualization(self) -> str:
        """Generate ASCII art representation of node state"""
        
        confidence_bar = "█" * int(self.confidence * 20) + "░" * (20 - int(self.confidence * 20))
        uncertainty_bar = "█" * int(self.uncertainty * 20) + "░" * (20 - int(self.uncertainty * 20))
        
        state_icon = {
            NodeState.UNINITIALIZED: "⭕",
            NodeState.PROCESSING: "⚙️",
            NodeState.NEEDS_MORE_EVIDENCE: "❓",
            NodeState.VALIDATING: "🔍",
            NodeState.CONVERGED: "✅",
            NodeState.RECURSIVE_LOOP: "🔄",
            NodeState.FAILED: "❌"
        }.get(self.state, "⚪")
        
        return f"""
┌─ {self.node_id} {state_icon} ──────────────────┐
│ Type: {self.node_type:<25} │
│ Confidence: [{confidence_bar}] {self.confidence:.3f} │
│ Uncertainty:[{uncertainty_bar}] {self.uncertainty:.3f} │
│ Evidence: {len(self.evidence_list)} items, Recursive: {self.recursive_count}  │
│ {self.description:<35} │
└────────────────────────────────────────┘"""

class SimpleBayesianNetwork:
    """Simplified Bayesian Evidence Network"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, SimpleBayesianNode] = {}
        self.processing_steps: List[ProcessingStep] = []
        self.network_coherence = 0.0
        
        # Initialize nodes from config
        for node_config in config.get('nodes', []):
            node = SimpleBayesianNode(
                node_id=node_config['id'],
                node_type=node_config['type'],
                description=node_config['description'],
                config=config
            )
            self.nodes[node_config['id']] = node
    
    async def process_query(self, query: str, context: List[str], llm_client) -> Dict[str, Any]:
        """Process query through the Bayesian network"""
        
        print("🧠 Starting Bayesian Evidence Network Processing...")
        start_time = time.time()
        
        # Step 1: Query Analysis
        analysis_step = await self._analyze_query(query, context, llm_client)
        
        # Step 2: Knowledge Retrieval
        knowledge_step = await self._retrieve_knowledge(query, context, llm_client)
        
        # Step 3: Reasoning Validation
        validation_step = await self._validate_reasoning(query, llm_client)
        
        # Step 4: Visual Generation
        visual_step = await self._generate_visuals(query, llm_client)
        
        # Step 5: Final Synthesis
        synthesis_step = await self._synthesize_results(query, llm_client)
        
        # Calculate final network coherence
        self.network_coherence = self._calculate_network_coherence()
        
        total_time = time.time() - start_time
        
        return {
            'query': query,
            'context': context,
            'processing_steps': self.processing_steps,
            'network_coherence': self.network_coherence,
            'total_processing_time': total_time,
            'nodes_final_state': {
                node_id: {
                    'confidence': node.confidence,
                    'uncertainty': node.uncertainty,
                    'state': node.state.value,
                    'evidence_count': len(node.evidence_list),
                    'recursive_count': node.recursive_count
                }
                for node_id, node in self.nodes.items()
            },
            'final_response': synthesis_step.output_data if synthesis_step else "Processing incomplete"
        }
    
    async def _analyze_query(self, query: str, context: List[str], llm_client) -> ProcessingStep:
        """Step 1: Analyze the query"""
        node = self.nodes['query_analysis']
        node.state = NodeState.PROCESSING
        
        print(f"📝 Step 1: Query Analysis")
        
        # Use LLM to analyze query
        analysis_prompt = f"""
        Analyze this query for intent, complexity, and required approach:
        Query: "{query}"
        Context: {context}
        
        Provide:
        1. User intent (what they really want to know)
        2. Complexity level (1-10)
        3. Required knowledge domains
        4. Suggested approach for explanation
        
        Be concise but thorough.
        """
        
        try:
            response = await llm_client.get_completion(analysis_prompt)
            
            # Create evidence from LLM analysis
            evidence = Evidence(
                content=response,
                confidence=0.8,  # High confidence in LLM analysis
                uncertainty=0.2,
                source="llm_analysis",
                timestamp=datetime.now()
            )
            
            step = await node.add_evidence(evidence)
            await node.check_convergence()
            
        except Exception as e:
            step = ProcessingStep(
                step_number=len(self.processing_steps) + 1,
                node_id=node.node_id,
                action="analyze_query_failed",
                input_data=query,
                output_data=f"Error: {str(e)}",
                confidence_before=node.confidence,
                confidence_after=node.confidence,
                uncertainty_before=node.uncertainty,
                uncertainty_after=node.uncertainty,
                reasoning=f"Query analysis failed: {str(e)}",
                duration=0.0,
                timestamp=datetime.now()
            )
        
        self.processing_steps.append(step)
        return step
    
    async def _retrieve_knowledge(self, query: str, context: List[str], llm_client) -> ProcessingStep:
        """Step 2: Retrieve relevant knowledge"""
        node = self.nodes['knowledge_retrieval']
        node.state = NodeState.PROCESSING
        
        print(f"📚 Step 2: Knowledge Retrieval")
        
        # Use LLM to gather knowledge
        knowledge_prompt = f"""
        Provide comprehensive knowledge about: "{query}"
        
        Include:
        1. Core concepts and definitions
        2. Key relationships and principles
        3. Common examples and applications
        4. Potential misconceptions to address
        
        Focus on accuracy and clarity.
        """
        
        try:
            response = await llm_client.get_completion(knowledge_prompt)
            
            # Assess knowledge quality (simulate)
            quality_score = min(1.0, len(response.split()) / 100)  # Simple quality heuristic
            
            evidence = Evidence(
                content=response,
                confidence=quality_score,
                uncertainty=0.3,
                source="llm_knowledge",
                timestamp=datetime.now()
            )
            
            step = await node.add_evidence(evidence)
            
            # Check if we need recursive processing
            converged, conv_step = await node.check_convergence()
            
            if node.state == NodeState.RECURSIVE_LOOP:
                print(f"🔄 Knowledge retrieval needs more depth...")
                # In a real system, this would trigger additional knowledge gathering
                
                # Simulate additional knowledge gathering
                deeper_prompt = f"""
                Provide deeper, more detailed knowledge about: "{query}"
                Focus on advanced concepts, mathematical relationships, and expert-level insights.
                """
                
                deeper_response = await llm_client.get_completion(deeper_prompt)
                
                deeper_evidence = Evidence(
                    content=deeper_response,
                    confidence=0.9,
                    uncertainty=0.1,
                    source="llm_deep_knowledge",
                    timestamp=datetime.now()
                )
                
                await node.add_evidence(deeper_evidence)
                await node.check_convergence()
            
        except Exception as e:
            step = ProcessingStep(
                step_number=len(self.processing_steps) + 1,
                node_id=node.node_id,
                action="knowledge_retrieval_failed",
                input_data=query,
                output_data=f"Error: {str(e)}",
                confidence_before=node.confidence,
                confidence_after=node.confidence,
                uncertainty_before=node.uncertainty,
                uncertainty_after=node.uncertainty,
                reasoning=f"Knowledge retrieval failed: {str(e)}",
                duration=0.0,
                timestamp=datetime.now()
            )
        
        self.processing_steps.append(step)
        return step
    
    async def _validate_reasoning(self, query: str, llm_client) -> ProcessingStep:
        """Step 3: Validate the reasoning"""
        node = self.nodes['reasoning_validation']
        node.state = NodeState.PROCESSING
        
        print(f"🔍 Step 3: Reasoning Validation")
        
        # Get the knowledge from previous step
        knowledge_node = self.nodes['knowledge_retrieval']
        if knowledge_node.evidence_list:
            knowledge_content = knowledge_node.evidence_list[-1].content
        else:
            knowledge_content = "No knowledge available"
        
        validation_prompt = f"""
        Validate this explanation for logical consistency and accuracy:
        
        Query: "{query}"
        Knowledge: {knowledge_content[:500]}...
        
        Check:
        1. Logical consistency
        2. Scientific accuracy
        3. Completeness of explanation
        4. Potential gaps or errors
        
        Rate the explanation quality (1-10) and explain your reasoning.
        """
        
        try:
            response = await llm_client.get_completion(validation_prompt)
            
            # Extract quality score (simulate)
            quality_score = 0.8  # Would parse from response in real system
            
            evidence = Evidence(
                content=response,
                confidence=quality_score,
                uncertainty=0.2,
                source="llm_validation",
                timestamp=datetime.now()
            )
            
            step = await node.add_evidence(evidence)
            await node.check_convergence()
            
        except Exception as e:
            step = ProcessingStep(
                step_number=len(self.processing_steps) + 1,
                node_id=node.node_id,
                action="reasoning_validation_failed",
                input_data=query,
                output_data=f"Error: {str(e)}",
                confidence_before=node.confidence,
                confidence_after=node.confidence,
                uncertainty_before=node.uncertainty,
                uncertainty_after=node.uncertainty,
                reasoning=f"Reasoning validation failed: {str(e)}",
                duration=0.0,
                timestamp=datetime.now()
            )
        
        self.processing_steps.append(step)
        return step
    
    async def _generate_visuals(self, query: str, llm_client) -> ProcessingStep:
        """Step 4: Generate visual representations"""
        node = self.nodes['visual_generation']
        node.state = NodeState.PROCESSING
        
        print(f"🎨 Step 4: Visual Generation")
        
        visual_prompt = f"""
        Create visual descriptions for: "{query}"
        
        Provide:
        1. ASCII art diagram if applicable
        2. Description of key visual relationships
        3. Analogies that would help visualize the concept
        4. Suggestions for plots or charts that would illustrate the concept
        
        Focus on making abstract concepts concrete and visual.
        """
        
        try:
            response = await llm_client.get_completion(visual_prompt)
            
            # Create simple ASCII visualization
            ascii_visual = self._create_concept_visualization(query)
            
            combined_visual = f"{response}\n\n--- Generated ASCII Visualization ---\n{ascii_visual}"
            
            evidence = Evidence(
                content=combined_visual,
                confidence=0.7,
                uncertainty=0.3,
                source="visual_generation",
                timestamp=datetime.now()
            )
            
            step = await node.add_evidence(evidence)
            await node.check_convergence()
            
        except Exception as e:
            step = ProcessingStep(
                step_number=len(self.processing_steps) + 1,
                node_id=node.node_id,
                action="visual_generation_failed",
                input_data=query,
                output_data=f"Error: {str(e)}",
                confidence_before=node.confidence,
                confidence_after=node.confidence,
                uncertainty_before=node.uncertainty,
                uncertainty_after=node.uncertainty,
                reasoning=f"Visual generation failed: {str(e)}",
                duration=0.0,
                timestamp=datetime.now()
            )
        
        self.processing_steps.append(step)
        return step
    
    async def _synthesize_results(self, query: str, llm_client) -> ProcessingStep:
        """Step 5: Synthesize final results"""
        node = self.nodes['final_synthesis']
        node.state = NodeState.PROCESSING
        
        print(f"🎭 Step 5: Final Synthesis")
        
        # Gather all evidence from previous nodes
        all_evidence = []
        for node_id, other_node in self.nodes.items():
            if node_id != 'final_synthesis' and other_node.evidence_list:
                all_evidence.extend([f"{node_id}: {evidence.content[:200]}..." 
                                   for evidence in other_node.evidence_list])
        
        synthesis_prompt = f"""
        Synthesize a comprehensive answer for: "{query}"
        
        Based on the following analysis:
        {chr(10).join(all_evidence)}
        
        Provide:
        1. Clear, comprehensive answer
        2. Key insights and takeaways  
        3. Visual elements that support understanding
        4. Confidence assessment of the response
        
        Make it educational and accessible.
        """
        
        try:
            response = await llm_client.get_completion(synthesis_prompt)
            
            # Calculate synthesis confidence based on all nodes
            avg_confidence = sum(node.confidence for node in self.nodes.values()) / len(self.nodes)
            
            evidence = Evidence(
                content=response,
                confidence=avg_confidence,
                uncertainty=0.1,  # Low uncertainty for final synthesis
                source="final_synthesis",
                timestamp=datetime.now()
            )
            
            step = await node.add_evidence(evidence)
            await node.check_convergence()
            
        except Exception as e:
            step = ProcessingStep(
                step_number=len(self.processing_steps) + 1,
                node_id=node.node_id,
                action="synthesis_failed",
                input_data=query,
                output_data=f"Error: {str(e)}",
                confidence_before=node.confidence,
                confidence_after=node.confidence,
                uncertainty_before=node.uncertainty,
                uncertainty_after=node.uncertainty,
                reasoning=f"Final synthesis failed: {str(e)}",
                duration=0.0,
                timestamp=datetime.now()
            )
        
        self.processing_steps.append(step)
        return step
    
    def _create_concept_visualization(self, query: str) -> str:
        """Create simple ASCII visualization based on query"""
        
        if "newton" in query.lower() or "f=ma" in query.lower():
            return """
F = m × a

Force ──► Mass × Acceleration

    📦 (mass)
     │
     ▼
    🏃‍♂️ (acceleration)
     │
     ▼
    💨 (force)

Example:
Car (1000kg) accelerating at 2 m/s²
F = 1000 × 2 = 2000 Newtons
"""
        elif "quantum" in query.lower():
            return """
Quantum Entanglement:

Particle A ◄─────────► Particle B
    ↓ measure           ↓ instantly
   Spin ↑              Spin ↓
   
🌌 Spooky action at a distance! 🌌
"""
        elif "energy" in query.lower() and "mass" in query.lower():
            return """
E = mc²

Energy = Mass × (Speed of Light)²

💡 Energy ◄─── 🧪 Mass
              │
              ▼
         🚀 c² (very big number!)

Small mass → HUGE energy!
"""
        else:
            return """
Generic Concept Visualization:

Input ──► Processing ──► Output
  │           │            │
  ▼           ▼            ▼
[Data]    [Analysis]   [Result]
"""
    
    def _calculate_network_coherence(self) -> float:
        """Calculate overall network coherence"""
        
        if not self.nodes:
            return 0.0
        
        # Average confidence weighted by evidence count
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for node in self.nodes.values():
            weight = max(1, len(node.evidence_list))  # At least weight of 1
            total_weighted_confidence += node.confidence * weight
            total_weight += weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def get_network_visualization(self) -> str:
        """Generate ASCII visualization of entire network"""
        
        viz = "🧠 BAYESIAN EVIDENCE NETWORK STATE\n"
        viz += "=" * 50 + "\n\n"
        
        for node in self.nodes.values():
            viz += node.get_ascii_visualization() + "\n\n"
        
        viz += f"🌐 Overall Network Coherence: {self.network_coherence:.3f}\n"
        viz += f"📊 Total Processing Steps: {len(self.processing_steps)}\n"
        
        return viz
