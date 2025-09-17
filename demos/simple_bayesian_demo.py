#!/usr/bin/env python3
"""
Spectacular Simple Bayesian Demo

A simplified CLI demonstration of the Bayesian Evidence Network concepts
with complete transparency of all reasoning steps.

Usage:
    python simple_bayesian_demo.py --config demo_config.yaml
    python simple_bayesian_demo.py --help
"""

import argparse
import asyncio
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Import our demo modules
from bayesian_network import SimpleBayesianNetwork
from llm_client import SimpleLLMClient
from markdown_report import MarkdownReportGenerator

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please create a config file or use the provided demo_config.yaml")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate that required configuration is present"""
    
    required_sections = ['openai', 'query', 'bayesian_network']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required config section: {section}")
            return False
    
    # Check OpenAI config
    if 'api_key' not in config['openai']:
        print("❌ Missing OpenAI API key in config")
        print("Please set your API key in the config file or environment variable OPENAI_API_KEY")
        return False
    
    # Check for placeholder API key
    api_key = config['openai']['api_key']
    if api_key in ['your-openai-api-key-here', '', None]:
        print("❌ Please set a valid OpenAI API key in the config file")
        print("You can get one from: https://platform.openai.com/api-keys")
        return False
    
    # Check query
    if 'prompt' not in config['query']:
        print("❌ Missing query prompt in config")
        return False
    
    print("✅ Configuration validation passed")
    return True

def print_banner():
    """Print demo banner"""
    
    banner = """
🧠 ═══════════════════════════════════════════════════════════════════════
    SPECTACULAR BAYESIAN EVIDENCE NETWORK DEMO
    
    Simplified demonstration with complete reasoning transparency
    Every step is logged and explained in plain English
═══════════════════════════════════════════════════════════════════════ 🧠
"""
    print(banner)

async def run_demo(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete demo"""
    
    print_banner()
    
    # Extract configuration
    openai_config = config['openai']
    query_config = config['query']
    bayesian_config = config['bayesian_network']
    
    # Initialize components
    print("🚀 Initializing system components...")
    
    # Initialize LLM client
    llm_client = SimpleLLMClient(
        api_key=openai_config['api_key'],
        model=openai_config.get('model', 'gpt-3.5-turbo'),
        temperature=openai_config.get('temperature', 0.7),
        max_tokens=openai_config.get('max_tokens', 1000)
    )
    
    # Test LLM connection
    if not llm_client.test_connection():
        print("❌ Failed to connect to OpenAI API")
        sys.exit(1)
    
    # Initialize Bayesian network
    bayesian_network = SimpleBayesianNetwork(bayesian_config)
    
    # Extract query and context
    query = query_config['prompt']
    context = query_config.get('context', [])
    
    print(f"📝 Processing Query: {query}")
    print(f"📋 Context: {context}")
    print("")
    
    # Process the query through the Bayesian network
    print("🧠 Starting Bayesian Evidence Network processing...")
    print("=" * 60)
    
    results = await bayesian_network.process_query(query, context, llm_client)
    
    print("=" * 60)
    print("✅ Processing complete!")
    print("")
    
    # Print network visualization
    if config.get('output', {}).get('include_ascii_art', True):
        print("🌐 Final Network State:")
        print(bayesian_network.get_network_visualization())
    
    return results

def generate_report(config: Dict[str, Any], results: Dict[str, Any]) -> str:
    """Generate markdown report"""
    
    print("📄 Generating comprehensive markdown report...")
    
    report_generator = MarkdownReportGenerator(config)
    report_content = report_generator.generate_report(results)
    
    # Save report
    report_path = report_generator.save_report(report_content)
    
    print(f"✅ Report saved to: {report_path}")
    print(f"📊 Report contains {len(report_content.split())} words")
    
    return report_path

def print_summary(results: Dict[str, Any], report_path: str):
    """Print execution summary"""
    
    print("\n🎯 EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Query: {results['query']}")
    print(f"Network Coherence: {results['network_coherence']:.3f}/1.0")
    print(f"Processing Time: {results['total_processing_time']:.2f} seconds")
    print(f"Steps Executed: {len(results['processing_steps'])}")
    
    nodes_state = results['nodes_final_state']
    converged = sum(1 for state in nodes_state.values() if state['state'] == 'converged')
    recursive = sum(state['recursive_count'] for state in nodes_state.values())
    
    print(f"Nodes Converged: {converged}/{len(nodes_state)}")
    print(f"Recursive Loops: {recursive}")
    
    print(f"\n📄 Complete reasoning trace: {report_path}")
    
    # Quality assessment
    if results['network_coherence'] >= 0.8:
        print("🟢 High quality reasoning with strong confidence")
    elif results['network_coherence'] >= 0.6:
        print("🟡 Moderate quality reasoning with acceptable confidence")
    else:
        print("🟠 Lower quality reasoning - results should be used cautiously")

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Spectacular Simple Bayesian Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python simple_bayesian_demo.py --config demo_config.yaml
    python simple_bayesian_demo.py --config my_physics_question.yaml

The demo will:
1. Load your configuration file
2. Connect to OpenAI API
3. Process your query through the Bayesian Evidence Network
4. Show every reasoning step in real-time
5. Generate a comprehensive markdown report
6. Save everything to the output/ directory

Make sure to set your OpenAI API key in the config file!
        """
    )
    
    parser.add_argument(
        '--config',
        default='demo_config.yaml',
        help='Path to configuration YAML file (default: demo_config.yaml)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output during processing'
    )
    
    args = parser.parse_args()
    
    try:
        # Load and validate configuration
        config = load_config(args.config)
        
        if not validate_config(config):
            sys.exit(1)
        
        # Run the demo
        results = await run_demo(config)
        
        # Generate report
        report_path = generate_report(config, results)
        
        # Print summary
        if not args.quiet:
            print_summary(results, report_path)
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📖 Open {report_path} to see the complete reasoning trace")
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
