"""
Main entry point for the Spectacular metacognitive visualization system.

This module demonstrates the complete end-to-end process of generating
D3.js visualizations using the metacognitive orchestrator and all
specialized reasoning modules.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .config import get_config, setup_logging
from .orchestrator import MetacognitiveOrchestrator
from .diadochi import DiadochiOrchestrator, DomainExpertise, IntegrationPattern

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context object for visualization queries."""
    query: str
    complexity_score: float = 0.5
    visual_complexity: float = 0.5
    domain_specificity: float = 0.5
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}


class SpectacularSystem:
    """
    Main system class that coordinates the entire Spectacular pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Spectacular system."""
        # Load configuration
        self.config = get_config()
        setup_logging(self.config)
        
        # Initialize the metacognitive orchestrator
        self.orchestrator = None
        self.ready = False
        
        logger.info("Spectacular System initialized")
    
    async def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing Spectacular system components...")
        
        try:
            # Initialize the metacognitive orchestrator
            self.orchestrator = MetacognitiveOrchestrator(self.config.to_dict())
            
            # Wait for all modules to be ready
            await self._wait_for_readiness()
            
            self.ready = True
            logger.info("Spectacular system initialization complete")
            
        except Exception as e:
            logger.error(f"Error during system initialization: {e}")
            raise
    
    async def _wait_for_readiness(self, timeout: int = 60):
        """Wait for all modules to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.orchestrator and self.orchestrator.is_ready():
                logger.info("All modules are ready")
                return
            
            logger.info("Waiting for modules to initialize...")
            await asyncio.sleep(2)
        
        raise TimeoutError("System initialization timed out")
    
    async def generate_visualization(self, query: str, 
                                   complexity_score: float = None,
                                   visual_complexity: float = None) -> Dict[str, Any]:
        """
        Generate a D3.js visualization from a natural language query.
        
        Args:
            query: Natural language description of the desired visualization
            complexity_score: Optional complexity score (0-1)
            visual_complexity: Optional visual complexity score (0-1)
            
        Returns:
            Dict containing the generated visualization and metadata
        """
        if not self.ready:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logger.info(f"Generating visualization for query: {query[:100]}...")
        
        # Create query context
        context = QueryContext(
            query=query,
            complexity_score=complexity_score or self._estimate_complexity(query),
            visual_complexity=visual_complexity or self._estimate_visual_complexity(query)
        )
        
        # Use the metacognitive orchestrator to process the query
        result = await self.orchestrator.process_query(context)
        
        return result
    
    def _estimate_complexity(self, query: str) -> float:
        """Simple heuristic to estimate query complexity."""
        # Count complexity indicators
        complexity_words = [
            'interactive', 'complex', 'multiple', 'nested', 'hierarchical',
            'animated', 'real-time', 'dynamic', 'advanced', 'sophisticated'
        ]
        
        query_lower = query.lower()
        complexity_count = sum(1 for word in complexity_words if word in query_lower)
        
        # Normalize by query length and complexity words found
        base_complexity = min(0.9, len(query) / 200.0)  # Longer queries are more complex
        word_complexity = min(0.8, complexity_count / 3.0)  # More complexity words = higher complexity
        
        return (base_complexity + word_complexity) / 2.0
    
    def _estimate_visual_complexity(self, query: str) -> float:
        """Simple heuristic to estimate visual complexity."""
        visual_words = [
            'color', 'colors', 'heatmap', 'gradient', 'multi-dimensional',
            'layered', 'overlay', 'detailed', 'intricate', 'elaborate'
        ]
        
        query_lower = query.lower()
        visual_count = sum(1 for word in visual_words if word in query_lower)
        
        return min(0.9, visual_count / 3.0)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of all system modules."""
        if not self.orchestrator:
            return {"status": "not_initialized"}
        
        return await self.orchestrator.get_system_status()
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down Spectacular system...")
        
        if self.orchestrator:
            # Any cleanup needed
            pass
        
        logger.info("Spectacular system shutdown complete")


async def demo_visualization_generation():
    """Demonstration of the Spectacular system with various queries."""
    
    # Initialize the system
    system = SpectacularSystem()
    await system.initialize()
    
    # Test queries with increasing complexity
    test_queries = [
        {
            "query": "Create a simple bar chart showing sales by month",
            "description": "Simple bar chart"
        },
        {
            "query": "Generate an interactive scatter plot showing the correlation between temperature and sales, with color coding by region and tooltips",
            "description": "Interactive scatter plot with multiple dimensions"
        },
        {
            "query": "Build a complex hierarchical network diagram showing company organizational structure with animated transitions and drill-down capabilities",
            "description": "Complex hierarchical network with animations"
        },
        {
            "query": "Create a multi-layered heatmap dashboard with real-time data updates, brush selection, and coordinated filtering across multiple charts",
            "description": "Advanced dashboard with multiple coordinated views"
        }
    ]
    
    print("\n" + "="*80)
    print("SPECTACULAR SYSTEM DEMONSTRATION")
    print("="*80)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"Query: {test_case['query']}")
        
        try:
            start_time = time.time()
            result = await system.generate_visualization(test_case['query'])
            end_time = time.time()
            
            print(f"\n✅ Generation completed in {end_time - start_time:.2f} seconds")
            print(f"Overall confidence: {result.get('overall_confidence', 0.0):.2f}")
            print(f"Final state: {result.get('final_state', 'unknown')}")
            
            # Print module results summary
            if 'module_results' in result:
                print("\nModule Results:")
                for module, module_result in result['module_results'].items():
                    confidence = module_result.get('confidence', 0.0)
                    print(f"  - {module}: {confidence:.2f} confidence")
            
            # Print reasoning chain summary
            if 'reasoning_chain' in result:
                print(f"\nReasoning steps: {len(result['reasoning_chain'])}")
                for step in result['reasoning_chain'][-3:]:  # Show last 3 steps
                    print(f"  - {step.get('step_type', 'unknown')}: {step.get('description', 'no description')}")
            
            # Print generated code preview
            if 'generated_code' in result:
                code = result['generated_code']
                if isinstance(code, str):
                    preview = code[:200] + "..." if len(code) > 200 else code
                    print(f"\nGenerated D3.js code preview:\n{preview}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("-" * 80)
    
    # Show system status
    print("\n--- System Status ---")
    status = await system.get_system_status()
    print(f"System ready: {status.get('ready', False)}")
    print(f"Modules initialized: {status.get('modules_ready', 0)}/5")
    print(f"Total queries processed: {status.get('total_queries_processed', 0)}")
    
    if 'module_stats' in status:
        print("\nModule Statistics:")
        for module, stats in status['module_stats'].items():
            if isinstance(stats, dict) and 'ready' in stats:
                print(f"  - {module}: {'✓' if stats['ready'] else '✗'}")
    
    # Cleanup
    await system.shutdown()
    print("\nDemonstration complete!")


async def demo_diadochi_system():
    """Demonstration of the Diadochi intelligent model combination system."""
    
    print("\n" + "="*80)
    print("DIADOCHI INTELLIGENT MODEL COMBINATION DEMONSTRATION")
    print("="*80)
    print("🏛️  Initializing Diadochi - Combining Expert Domains for Superior AI")
    
    # Initialize the Diadochi orchestrator
    diadochi = DiadochiOrchestrator()
    
    # Define domain expertise for data visualization
    visualization_expertise = DomainExpertise(
        domain="data_visualization",
        description="Expert in creating effective data visualizations using D3.js, understanding visual encoding principles, and chart selection",
        keywords=["chart", "graph", "plot", "visualization", "d3", "svg", "interactive", "dashboard"],
        reasoning_patterns=["visual encoding analysis", "chart type selection", "aesthetic optimization"]
    )
    
    statistics_expertise = DomainExpertise(
        domain="statistics",
        description="Expert in statistical analysis, data interpretation, and mathematical modeling for data insights",
        keywords=["correlation", "regression", "distribution", "significance", "analysis", "model", "statistical"],
        reasoning_patterns=["statistical analysis", "hypothesis testing", "data interpretation"]
    )
    
    design_expertise = DomainExpertise(
        domain="ui_ux_design",
        description="Expert in user interface and user experience design principles for creating intuitive and accessible visualizations",
        keywords=["usability", "accessibility", "interaction", "user", "interface", "experience", "design"],
        reasoning_patterns=["user-centered design", "accessibility assessment", "interaction design"]
    )
    
    # Register domain experts
    diadochi.register_domain_expert("data_visualization", "mock_viz_model", visualization_expertise)
    diadochi.register_domain_expert("statistics", "mock_stats_model", statistics_expertise)
    diadochi.register_domain_expert("ui_ux_design", "mock_design_model", design_expertise)
    
    # Create different integration patterns
    diadochi.create_integration_pattern(IntegrationPattern.ROUTER_ENSEMBLE, "router_ensemble")
    diadochi.create_integration_pattern(IntegrationPattern.SEQUENTIAL_CHAIN, "sequential_chain")
    diadochi.create_integration_pattern(IntegrationPattern.MIXTURE_OF_EXPERTS, "mixture_of_experts")
    
    # Test queries that span multiple domains
    test_queries = [
        {
            "query": "Create an accessible scatter plot showing correlation between variables with clear statistical significance indicators",
            "description": "Multi-domain query requiring visualization, statistics, and design expertise"
        },
        {
            "query": "Design an interactive dashboard for exploring sales data with intuitive filtering and statistical summaries",
            "description": "Complex dashboard requiring all three domains"
        },
        {
            "query": "Build a user-friendly chart that shows distribution patterns with appropriate statistical annotations",
            "description": "Distribution visualization with statistical and UX considerations"
        },
        {
            "query": "Generate a color-blind accessible heatmap with statistical clustering and hover interactions",
            "description": "Accessibility-focused visualization with statistical analysis"
        }
    ]
    
    print(f"\n📊 Testing {len(test_queries)} multi-domain queries...")
    
    # Test each integration pattern
    patterns_to_test = ["router_ensemble", "sequential_chain", "mixture_of_experts"]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {test_case['description']}")
        print(f"   Query: {test_case['query']}")
        
        # Test with different integration patterns
        for pattern in patterns_to_test:
            try:
                start_time = time.time()
                result = await diadochi.process_query(test_case['query'], pattern)
                end_time = time.time()
                
                print(f"   {pattern.replace('_', ' ').title()}: {end_time - start_time:.3f}s")
                print(f"     Domains: {', '.join(result.get('domains_used', []))}")
                print(f"     Confidence: {max(result.get('confidence_scores', {}).values(), default=0.0):.2f}")
                
            except Exception as e:
                print(f"   {pattern}: Error - {str(e)}")
        
        print("-" * 60)
    
    # Show system status
    print("\n📈 Diadochi System Status:")
    status = diadochi.get_system_status()
    print(f"   Domain Experts: {status['system_info']['total_domains']}")
    print(f"   Available Domains: {', '.join(status['domain_experts'].keys())}")
    
    # Show domain expertise details
    print("\n🎯 Domain Expertise Profiles:")
    for domain, info in status['domain_experts'].items():
        print(f"   {domain.replace('_', ' ').title()}:")
        print(f"     Description: {info['description'][:80]}...")
        print(f"     Keywords: {', '.join(info['keywords'][:5])}...")
    
    # Benchmark patterns
    print("\n🏁 Benchmarking Integration Patterns...")
    benchmark_queries = [test_case['query'] for test_case in test_queries[:2]]  # Use subset for demo
    benchmark_results = await diadochi.benchmark_patterns(benchmark_queries)
    
    for pattern_name, results in benchmark_results.items():
        print(f"   {pattern_name.replace('_', ' ').title()}:")
        print(f"     Avg Processing Time: {results['avg_processing_time']:.3f}s")
        print(f"     Success Rate: {results['success_rate']:.1%}")
        print(f"     Total Queries: {results['total_queries']}")
    
    print("\n✅ Diadochi demonstration completed!")
    print("🏛️  The successors of Alexander would be proud - domains united under intelligent orchestration!")
    
    return diadochi


async def interactive_mode():
    """Interactive mode for testing queries."""
    
    system = SpectacularSystem()
    await system.initialize()
    
    print("\n" + "="*60)
    print("SPECTACULAR INTERACTIVE MODE")
    print("="*60)
    print("Enter visualization queries (type 'quit' to exit)")
    print("Examples:")
    print("  - 'Create a bar chart of sales data'")
    print("  - 'Generate an interactive scatter plot'")
    print("  - 'Build a network diagram'")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\nProcessing: {query}")
            start_time = time.time()
            
            result = await system.generate_visualization(query)
            
            end_time = time.time()
            print(f"✅ Completed in {end_time - start_time:.2f} seconds")
            print(f"Confidence: {result.get('overall_confidence', 0.0):.2f}")
            
            # Show brief summary
            if 'module_results' in result:
                print("Module confidences:", end=" ")
                confidences = [str(round(r.get('confidence', 0.0), 2)) 
                             for r in result['module_results'].values()]
                print(", ".join(confidences))
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    await system.shutdown()
    print("Goodbye!")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            asyncio.run(interactive_mode())
        elif sys.argv[1] == "diadochi":
            asyncio.run(demo_diadochi_system())
        elif sys.argv[1] == "full":
            # Run both demonstrations
            async def full_demo():
                await demo_visualization_generation()
                await demo_diadochi_system()
            asyncio.run(full_demo())
        else:
            print("Usage: python -m spectacular.main [interactive|diadochi|full]")
            print("  interactive: Run in interactive mode")
            print("  diadochi: Run Diadochi model combination demo")
            print("  full: Run both main demo and diadochi demo")
            print("  (no args): Run main visualization demo")
    else:
        asyncio.run(demo_visualization_generation())


if __name__ == "__main__":
    main() 