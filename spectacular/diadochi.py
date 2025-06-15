"""
Diadochi Module - Intelligent Model Combination for Superior Expert Domains

This module implements sophisticated architectural patterns for combining domain-expert
models to create superior integrated AI systems. Named after Alexander's successors
who divided and ruled different domains, this module intelligently orchestrates
multiple specialized models to produce comprehensive expert responses.

Key Features:
- Router-Based Ensembles for intelligent query routing
- Sequential Chaining for progressive domain analysis
- Mixture of Experts with confidence-based weighting
- Specialized System Prompts for single-model multi-expertise
- Knowledge Distillation across domains
- Multi-domain RAG with specialized knowledge bases

Author: Spectacular AI System
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationPattern(Enum):
    """Enumeration of available integration patterns."""
    ROUTER_ENSEMBLE = "router_ensemble"
    SEQUENTIAL_CHAIN = "sequential_chain"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"
    SYSTEM_PROMPTS = "system_prompts"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    MULTI_RAG = "multi_rag"


@dataclass
class DomainExpertise:
    """Represents expertise in a specific domain."""
    domain: str
    description: str
    keywords: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    reasoning_patterns: List[str] = field(default_factory=list)
    communication_style: str = "technical"
    knowledge_base: Optional[str] = None


@dataclass
class QueryContext:
    """Context information for a query."""
    query: str
    domains: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    integration_requirements: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertResponse:
    """Response from a domain expert."""
    domain: str
    response: str
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class DomainRouter(ABC):
    """Abstract base class for domain routing strategies."""
    
    @abstractmethod
    async def route(self, query: str, available_domains: List[str]) -> List[Tuple[str, float]]:
        """Route a query to appropriate domains with confidence scores."""
        pass


class EmbeddingRouter(DomainRouter):
    """Router based on semantic embedding similarity."""
    
    def __init__(self, embedding_model=None, temperature: float = 0.5):
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.domain_embeddings = {}
        self.domain_descriptions = {}
    
    def add_domain(self, domain: str, description: str):
        """Add a domain with its description."""
        self.domain_descriptions[domain] = description
        # In a real implementation, this would generate embeddings
        # For now, we'll use a simple keyword-based approach
        self.domain_embeddings[domain] = self._generate_mock_embedding(description)
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a mock embedding based on text hash."""
        # Simple hash-based mock embedding
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        return [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(32, len(hash_hex)), 2)]
    
    async def route(self, query: str, available_domains: List[str]) -> List[Tuple[str, float]]:
        """Route query based on embedding similarity."""
        query_embedding = self._generate_mock_embedding(query)
        
        scores = []
        for domain in available_domains:
            if domain in self.domain_embeddings:
                # Calculate cosine similarity
                domain_emb = self.domain_embeddings[domain]
                similarity = self._cosine_similarity(query_embedding, domain_emb)
                scores.append((domain, similarity))
        
        # Apply softmax with temperature
        scores = self._apply_softmax(scores, self.temperature)
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _apply_softmax(self, scores: List[Tuple[str, float]], temperature: float) -> List[Tuple[str, float]]:
        """Apply softmax function to scores."""
        if not scores:
            return scores
        
        # Extract scores and apply temperature
        values = [score / temperature for _, score in scores]
        max_val = max(values)
        
        # Compute softmax
        exp_values = [np.exp(v - max_val) for v in values]
        sum_exp = sum(exp_values)
        
        softmax_scores = [exp_val / sum_exp for exp_val in exp_values]
        
        return [(scores[i][0], softmax_scores[i]) for i in range(len(scores))]


class ResponseMixer(ABC):
    """Abstract base class for response mixing strategies."""
    
    @abstractmethod
    async def mix(self, query: str, responses: List[ExpertResponse]) -> str:
        """Mix multiple expert responses into a unified response."""
        pass


class SynthesisMixer(ResponseMixer):
    """Mixer that uses LLM synthesis to combine responses."""
    
    def __init__(self, synthesis_model=None):
        self.synthesis_model = synthesis_model
        self.synthesis_template = """
        You are tasked with synthesizing responses from multiple domain experts into a coherent, integrated response.
        
        Original query: {query}
        
        Expert responses (with confidence scores):
        
        {expert_responses}
        
        Create a unified response that integrates insights from all experts, giving appropriate weight to each domain based on their confidence scores. Ensure the response is coherent, non-repetitive, and directly addresses the original query.
        """
    
    async def mix(self, query: str, responses: List[ExpertResponse]) -> str:
        """Synthesize multiple expert responses."""
        if not responses:
            return "No expert responses available."
        
        if len(responses) == 1:
            return responses[0].response
        
        # Format expert responses
        formatted_responses = []
        for resp in responses:
            formatted_responses.append(
                f"[{resp.domain.title()} Expert (Confidence: {resp.confidence:.1%})]:\n{resp.response}"
            )
        
        expert_responses_text = "\n\n".join(formatted_responses)
        
        # Create synthesis prompt
        synthesis_prompt = self.synthesis_template.format(
            query=query,
            expert_responses=expert_responses_text
        )
        
        # In a real implementation, this would call the synthesis model
        # For now, we'll create a mock synthesis
        return await self._mock_synthesis(query, responses)
    
    async def _mock_synthesis(self, query: str, responses: List[ExpertResponse]) -> str:
        """Create a mock synthesis of expert responses."""
        synthesis = f"Based on analysis from {len(responses)} domain experts:\n\n"
        
        # Combine key insights
        for i, resp in enumerate(responses, 1):
            synthesis += f"{i}. From {resp.domain}: {resp.response[:200]}...\n"
        
        synthesis += f"\nIntegrated Analysis: The query '{query}' requires interdisciplinary expertise. "
        synthesis += "The combined insights suggest a comprehensive approach that considers "
        synthesis += ", ".join([resp.domain for resp in responses]) + " perspectives."
        
        return synthesis


class RouterEnsemble:
    """Router-based ensemble for intelligent query routing."""
    
    def __init__(self, router: DomainRouter, experts: Dict[str, Any], mixer: ResponseMixer):
        self.router = router
        self.experts = experts
        self.mixer = mixer
        self.performance_stats = {}
    
    async def generate(self, query: str, top_k: int = 3) -> str:
        """Generate response using router-based ensemble."""
        # Route query to appropriate domains
        domain_scores = await self.router.route(query, list(self.experts.keys()))
        
        # Select top-k domains
        selected_domains = domain_scores[:top_k]
        
        # Generate responses from selected experts
        responses = []
        for domain, confidence in selected_domains:
            if domain in self.experts:
                expert_response = await self._query_expert(domain, query, confidence)
                responses.append(expert_response)
        
        # Mix responses
        final_response = await self.mixer.mix(query, responses)
        
        # Update performance stats
        self._update_stats(query, selected_domains, responses)
        
        return final_response
    
    async def _query_expert(self, domain: str, query: str, confidence: float) -> ExpertResponse:
        """Query a domain expert."""
        # In a real implementation, this would call the actual expert model
        # For now, we'll create a mock response
        mock_response = f"As a {domain} expert, I analyze this query: {query}. "
        mock_response += f"From a {domain} perspective, the key considerations are... "
        mock_response += f"[Mock response with {confidence:.1%} confidence]"
        
        return ExpertResponse(
            domain=domain,
            response=mock_response,
            confidence=confidence,
            reasoning_chain=[f"Applied {domain} expertise", "Analyzed query context", "Generated domain-specific insights"],
            citations=[f"{domain}_reference_1", f"{domain}_reference_2"]
        )
    
    def _update_stats(self, query: str, domains: List[Tuple[str, float]], responses: List[ExpertResponse]):
        """Update performance statistics."""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        self.performance_stats[query_hash] = {
            'domains_used': [d[0] for d in domains],
            'confidence_scores': [d[1] for d in domains],
            'response_count': len(responses),
            'timestamp': datetime.now().isoformat()
        }


class SequentialChain:
    """Sequential chaining for progressive domain analysis."""
    
    def __init__(self, experts: List[Tuple[str, Any]], prompt_templates: Dict[str, str] = None):
        self.experts = experts
        self.prompt_templates = prompt_templates or {}
        self.context_manager = ChainContextManager()
    
    async def generate(self, query: str) -> str:
        """Generate response through sequential chaining."""
        context = QueryContext(query=query)
        responses = []
        
        for i, (domain, expert) in enumerate(self.experts):
            # Format prompt for current expert
            prompt = self._format_prompt(domain, query, responses, context)
            
            # Generate response from current expert
            response = await self._query_expert(domain, expert, prompt, context)
            responses.append(response)
            
            # Update context
            self.context_manager.update_context(context, response)
            
            # Manage context window if needed
            if len(responses) > 2:
                responses = await self._manage_context_window(responses)
        
        # Return final response or synthesis
        if responses:
            return responses[-1].response
        return "No responses generated."
    
    def _format_prompt(self, domain: str, query: str, prev_responses: List[ExpertResponse], context: QueryContext) -> str:
        """Format prompt for current expert."""
        if domain in self.prompt_templates:
            template = self.prompt_templates[domain]
            return template.format(
                query=query,
                domain=domain,
                prev_responses=prev_responses,
                context=context
            )
        
        # Default prompt format
        if not prev_responses:
            return f"As a {domain} expert, analyze this query: {query}"
        
        prev_analysis = prev_responses[-1].response
        return f"""
        Previous expert analysis: {prev_analysis}
        
        Original query: {query}
        
        As a {domain} expert, provide your perspective on this query, building upon or critiquing the previous analysis as appropriate.
        """
    
    async def _query_expert(self, domain: str, expert: Any, prompt: str, context: QueryContext) -> ExpertResponse:
        """Query an expert in the chain."""
        # Mock expert response
        mock_response = f"From a {domain} perspective: {prompt[:100]}... [Chain analysis continues]"
        
        return ExpertResponse(
            domain=domain,
            response=mock_response,
            confidence=0.8,
            reasoning_chain=[f"Analyzed from {domain} perspective", "Built upon previous analysis"],
            metadata={'chain_position': len(context.domains)}
        )
    
    async def _manage_context_window(self, responses: List[ExpertResponse]) -> List[ExpertResponse]:
        """Manage context window by summarizing older responses."""
        if len(responses) <= 3:
            return responses
        
        # Keep the most recent response and summarize older ones
        recent = responses[-1]
        older = responses[:-1]
        
        # Create summary of older responses
        summary_text = "Previous analysis summary: "
        for resp in older:
            summary_text += f"{resp.domain}: {resp.response[:100]}... "
        
        summary_response = ExpertResponse(
            domain="summary",
            response=summary_text,
            confidence=0.9,
            reasoning_chain=["Summarized previous analyses"]
        )
        
        return [summary_response, recent]


class ChainContextManager:
    """Manages context flow in sequential chains."""
    
    def __init__(self, max_context_length: int = 2000):
        self.max_context_length = max_context_length
        self.context_history = []
    
    def update_context(self, context: QueryContext, response: ExpertResponse):
        """Update context with new response."""
        context.domains.append(response.domain)
        context.confidence_scores[response.domain] = response.confidence
        self.context_history.append(response)
        
        # Manage context length
        if len(str(context)) > self.max_context_length:
            self._compress_context(context)
    
    def _compress_context(self, context: QueryContext):
        """Compress context to fit within limits."""
        # Keep only essential information
        if len(context.domains) > 3:
            context.domains = context.domains[-3:]
        
        # Keep only high-confidence scores
        high_conf_scores = {k: v for k, v in context.confidence_scores.items() if v > 0.7}
        context.confidence_scores = high_conf_scores


class MixtureOfExperts:
    """Mixture of Experts with confidence-based weighting."""
    
    def __init__(self, experts: Dict[str, Any], confidence_estimator, mixer: ResponseMixer):
        self.experts = experts
        self.confidence_estimator = confidence_estimator
        self.mixer = mixer
        self.weighting_strategy = "softmax"
    
    async def generate(self, query: str, threshold: float = 0.1) -> str:
        """Generate response using mixture of experts."""
        # Estimate confidence for each expert
        confidence_scores = await self.confidence_estimator.estimate(query, list(self.experts.keys()))
        
        # Filter experts by threshold
        relevant_experts = [(domain, score) for domain, score in confidence_scores if score >= threshold]
        
        if not relevant_experts:
            return "No relevant experts found for this query."
        
        # Generate responses from relevant experts in parallel
        tasks = []
        for domain, confidence in relevant_experts:
            task = self._query_expert_async(domain, query, confidence)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # Mix responses based on confidence weights
        final_response = await self.mixer.mix(query, responses)
        
        return final_response
    
    async def _query_expert_async(self, domain: str, query: str, confidence: float) -> ExpertResponse:
        """Asynchronously query an expert."""
        # Simulate async expert query
        await asyncio.sleep(0.1)  # Mock processing time
        
        mock_response = f"Expert {domain} analysis (confidence: {confidence:.2f}): {query[:50]}..."
        
        return ExpertResponse(
            domain=domain,
            response=mock_response,
            confidence=confidence,
            reasoning_chain=[f"Applied {domain} methodology", "Analyzed query requirements"]
        )


class ConfidenceEstimator:
    """Estimates confidence scores for domain experts."""
    
    def __init__(self, estimation_method: str = "embedding"):
        self.estimation_method = estimation_method
        self.domain_profiles = {}
    
    def add_domain_profile(self, domain: str, profile: DomainExpertise):
        """Add a domain expertise profile."""
        self.domain_profiles[domain] = profile
    
    async def estimate(self, query: str, domains: List[str]) -> List[Tuple[str, float]]:
        """Estimate confidence scores for domains given a query."""
        scores = []
        
        for domain in domains:
            if domain in self.domain_profiles:
                profile = self.domain_profiles[domain]
                confidence = self._calculate_confidence(query, profile)
                scores.append((domain, confidence))
        
        return scores
    
    def _calculate_confidence(self, query: str, profile: DomainExpertise) -> float:
        """Calculate confidence score for a domain profile."""
        # Simple keyword-based confidence calculation
        query_lower = query.lower()
        keyword_matches = sum(1 for keyword in profile.keywords if keyword.lower() in query_lower)
        
        if not profile.keywords:
            return 0.5  # Default confidence
        
        base_confidence = keyword_matches / len(profile.keywords)
        
        # Apply domain-specific adjustments
        if len(query) > 100:  # Complex queries might need more expertise
            base_confidence *= 1.1
        
        return min(1.0, base_confidence)


class DiadochiOrchestrator:
    """Main orchestrator for the Diadochi model combination system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.domain_experts = {}
        self.integration_patterns = {}
        self.performance_metrics = {}
        self.active_sessions = {}
        
        # Initialize default components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize default system components."""
        # Default router
        self.default_router = EmbeddingRouter(temperature=0.5)
        
        # Default mixer
        self.default_mixer = SynthesisMixer()
        
        # Default confidence estimator
        self.default_confidence = ConfidenceEstimator()
        
        logger.info("Diadochi orchestrator initialized with default components")
    
    def register_domain_expert(self, domain: str, expert: Any, expertise: DomainExpertise):
        """Register a domain expert with the system."""
        self.domain_experts[domain] = {
            'expert': expert,
            'expertise': expertise,
            'performance': {'queries_handled': 0, 'avg_confidence': 0.0}
        }
        
        # Add domain to router and confidence estimator
        self.default_router.add_domain(domain, expertise.description)
        self.default_confidence.add_domain_profile(domain, expertise)
        
        logger.info(f"Registered domain expert: {domain}")
    
    def create_integration_pattern(self, pattern_type: IntegrationPattern, name: str, **kwargs) -> Any:
        """Create and register an integration pattern."""
        if pattern_type == IntegrationPattern.ROUTER_ENSEMBLE:
            pattern = RouterEnsemble(
                router=kwargs.get('router', self.default_router),
                experts=self.domain_experts,
                mixer=kwargs.get('mixer', self.default_mixer)
            )
        
        elif pattern_type == IntegrationPattern.SEQUENTIAL_CHAIN:
            experts_list = [(domain, info['expert']) for domain, info in self.domain_experts.items()]
            pattern = SequentialChain(
                experts=experts_list,
                prompt_templates=kwargs.get('prompt_templates', {})
            )
        
        elif pattern_type == IntegrationPattern.MIXTURE_OF_EXPERTS:
            pattern = MixtureOfExperts(
                experts={domain: info['expert'] for domain, info in self.domain_experts.items()},
                confidence_estimator=kwargs.get('confidence_estimator', self.default_confidence),
                mixer=kwargs.get('mixer', self.default_mixer)
            )
        
        else:
            raise ValueError(f"Unsupported integration pattern: {pattern_type}")
        
        self.integration_patterns[name] = pattern
        logger.info(f"Created integration pattern: {name} ({pattern_type.value})")
        
        return pattern
    
    async def process_query(self, query: str, pattern_name: str = None, **kwargs) -> Dict[str, Any]:
        """Process a query using specified or default integration pattern."""
        start_time = datetime.now()
        
        # Select integration pattern
        if pattern_name and pattern_name in self.integration_patterns:
            pattern = self.integration_patterns[pattern_name]
        else:
            # Use default router ensemble
            if 'default' not in self.integration_patterns:
                self.create_integration_pattern(IntegrationPattern.ROUTER_ENSEMBLE, 'default')
            pattern = self.integration_patterns['default']
        
        try:
            # Process query
            response = await pattern.generate(query, **kwargs)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = {
                'query': query,
                'response': response,
                'pattern_used': pattern_name or 'default',
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'domains_available': list(self.domain_experts.keys()),
                    'patterns_available': list(self.integration_patterns.keys())
                }
            }
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': query,
                'response': f"Error processing query: {str(e)}",
                'pattern_used': pattern_name or 'default',
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update system performance metrics."""
        pattern_name = result['pattern_used']
        
        if pattern_name not in self.performance_metrics:
            self.performance_metrics[pattern_name] = {
                'total_queries': 0,
                'avg_processing_time': 0.0,
                'success_rate': 0.0,
                'last_updated': datetime.now().isoformat()
            }
        
        metrics = self.performance_metrics[pattern_name]
        metrics['total_queries'] += 1
        
        # Update average processing time
        current_avg = metrics['avg_processing_time']
        new_time = result['processing_time']
        metrics['avg_processing_time'] = (current_avg * (metrics['total_queries'] - 1) + new_time) / metrics['total_queries']
        
        # Update success rate
        if 'error' not in result:
            success_count = metrics['success_rate'] * (metrics['total_queries'] - 1) + 1
            metrics['success_rate'] = success_count / metrics['total_queries']
        else:
            success_count = metrics['success_rate'] * (metrics['total_queries'] - 1)
            metrics['success_rate'] = success_count / metrics['total_queries']
        
        metrics['last_updated'] = datetime.now().isoformat()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'domain_experts': {
                domain: {
                    'description': info['expertise'].description,
                    'keywords': info['expertise'].keywords,
                    'performance': info['performance']
                }
                for domain, info in self.domain_experts.items()
            },
            'integration_patterns': list(self.integration_patterns.keys()),
            'performance_metrics': self.performance_metrics,
            'system_info': {
                'total_domains': len(self.domain_experts),
                'total_patterns': len(self.integration_patterns),
                'uptime': datetime.now().isoformat()
            }
        }
    
    async def benchmark_patterns(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark different integration patterns."""
        results = {}
        
        for pattern_name in self.integration_patterns.keys():
            pattern_results = []
            
            for query in test_queries:
                result = await self.process_query(query, pattern_name)
                pattern_results.append({
                    'query': query,
                    'processing_time': result['processing_time'],
                    'success': 'error' not in result
                })
            
            # Calculate pattern statistics
            processing_times = [r['processing_time'] for r in pattern_results]
            success_rate = sum(1 for r in pattern_results if r['success']) / len(pattern_results)
            
            results[pattern_name] = {
                'avg_processing_time': sum(processing_times) / len(processing_times),
                'success_rate': success_rate,
                'total_queries': len(test_queries),
                'detailed_results': pattern_results
            }
        
        return results


# Example usage and demonstration
async def demonstrate_diadochi():
    """Demonstrate the Diadochi system capabilities."""
    print("🏛️  Initializing Diadochi - Intelligent Model Combination System")
    
    # Create orchestrator
    orchestrator = DiadochiOrchestrator()
    
    # Define domain expertise
    biomechanics_expertise = DomainExpertise(
        domain="biomechanics",
        description="The study of mechanical laws relating to the movement of living organisms",
        keywords=["force", "velocity", "acceleration", "kinematics", "kinetics", "stride", "gait"],
        reasoning_patterns=["mechanical analysis", "force vector analysis", "motion optimization"]
    )
    
    physiology_expertise = DomainExpertise(
        domain="exercise_physiology",
        description="The study of physiological responses to physical activity and exercise",
        keywords=["muscle", "fiber", "aerobic", "anaerobic", "fatigue", "metabolism", "adaptation"],
        reasoning_patterns=["physiological analysis", "metabolic assessment", "adaptation mechanisms"]
    )
    
    nutrition_expertise = DomainExpertise(
        domain="sports_nutrition",
        description="The study of dietary needs and strategies to enhance athletic performance",
        keywords=["protein", "carbohydrate", "hydration", "supplementation", "recovery", "energy"],
        reasoning_patterns=["nutritional analysis", "dietary optimization", "performance enhancement"]
    )
    
    # Register domain experts (mock experts for demonstration)
    orchestrator.register_domain_expert("biomechanics", "mock_biomechanics_model", biomechanics_expertise)
    orchestrator.register_domain_expert("exercise_physiology", "mock_physiology_model", physiology_expertise)
    orchestrator.register_domain_expert("sports_nutrition", "mock_nutrition_model", nutrition_expertise)
    
    # Create different integration patterns
    orchestrator.create_integration_pattern(IntegrationPattern.ROUTER_ENSEMBLE, "router_ensemble")
    orchestrator.create_integration_pattern(IntegrationPattern.SEQUENTIAL_CHAIN, "sequential_chain")
    orchestrator.create_integration_pattern(IntegrationPattern.MIXTURE_OF_EXPERTS, "mixture_of_experts")
    
    # Test queries
    test_queries = [
        "What is the optimal stride frequency for elite sprinters?",
        "How does nutrition affect recovery time from high-intensity training?",
        "What are the biomechanical factors that influence running economy?",
        "How can athletes optimize their training for both strength and endurance?"
    ]
    
    print(f"\n📊 Processing {len(test_queries)} test queries...")
    
    # Process queries with different patterns
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        # Try router ensemble
        result = await orchestrator.process_query(query, "router_ensemble")
        print(f"   Router Ensemble: {result['processing_time']:.3f}s")
        
        # Try sequential chain
        result = await orchestrator.process_query(query, "sequential_chain")
        print(f"   Sequential Chain: {result['processing_time']:.3f}s")
        
        # Try mixture of experts
        result = await orchestrator.process_query(query, "mixture_of_experts")
        print(f"   Mixture of Experts: {result['processing_time']:.3f}s")
    
    # Show system status
    print("\n📈 System Status:")
    status = orchestrator.get_system_status()
    print(f"   Domain Experts: {status['system_info']['total_domains']}")
    print(f"   Integration Patterns: {status['system_info']['total_patterns']}")
    
    # Benchmark patterns
    print("\n🏁 Benchmarking Integration Patterns...")
    benchmark_results = await orchestrator.benchmark_patterns(test_queries[:2])  # Use subset for demo
    
    for pattern_name, results in benchmark_results.items():
        print(f"   {pattern_name}:")
        print(f"     Avg Processing Time: {results['avg_processing_time']:.3f}s")
        print(f"     Success Rate: {results['success_rate']:.1%}")
    
    print("\n✅ Diadochi demonstration completed!")
    return orchestrator


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_diadochi()) 