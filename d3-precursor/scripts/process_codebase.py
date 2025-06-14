#!/usr/bin/env python3
"""
Process D3 codebase and generate knowledge base.

This script processes a D3 codebase (JavaScript and Python implementations)
and extracts structured knowledge and embeddings for use by the LLM.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the src directory to the path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from parsers.js_parser import D3JSParser
from parsers.py_parser import D3PyParser
from parsers.doc_parser import D3DocParser
from parsers.example_parser import D3ExampleParser


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process D3 codebase and generate knowledge base")
    
    parser.add_argument("--d3-js-dir", type=str, help="Path to D3.js source code")
    parser.add_argument("--d3-py-dir", type=str, help="Path to Python D3 bindings")
    parser.add_argument("--docs-dir", type=str, help="Path to D3 documentation")
    parser.add_argument("--examples-dir", type=str, help="Path to D3 examples")
    parser.add_argument("--output-dir", type=str, default="../data/processed", help="Output directory")
    
    return parser.parse_args()


def process_js_codebase(source_dir, output_dir):
    """Process D3.js codebase."""
    print(f"Processing D3.js codebase from {source_dir}...")
    
    # Parse JavaScript codebase
    js_parser = D3JSParser(source_dir)
    js_ast = js_parser.parse_directory()
    
    # Extract API definitions
    api_defs = js_parser.extract_api_definitions()
    
    # Analyze usage patterns
    patterns = js_parser.analyze_usage_patterns()
    
    # Save results
    output_path = Path(output_dir) / "js"
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "api_definitions.json", "w") as f:
        json.dump(api_defs, f, indent=2)
    
    with open(output_path / "usage_patterns.json", "w") as f:
        json.dump(patterns, f, indent=2)
    
    print(f"JavaScript parsing results saved to {output_path}")
    return api_defs, patterns


def process_py_codebase(source_dir, output_dir):
    """Process Python D3 bindings."""
    print(f"Processing Python D3 bindings from {source_dir}...")
    
    # Parse Python codebase
    py_parser = D3PyParser(source_dir)
    py_ast = py_parser.parse_directory()
    
    # Extract API definitions
    api_defs = py_parser.extract_api_definitions()
    
    # Find D3 imports
    imports = py_parser.find_d3_imports()
    
    # Save results
    output_path = Path(output_dir) / "python"
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "api_definitions.json", "w") as f:
        json.dump(api_defs, f, indent=2)
    
    with open(output_path / "imports.json", "w") as f:
        json.dump(imports, f, indent=2)
    
    print(f"Python parsing results saved to {output_path}")
    return api_defs, imports


def process_documentation(docs_dir, output_dir):
    """Process D3 documentation."""
    print(f"Processing D3 documentation from {docs_dir}...")
    
    # Parse documentation
    doc_parser = D3DocParser(docs_dir)
    docs = doc_parser.process_directory()
    
    # Extract API documentation
    api_docs = doc_parser.extract_api_docs()
    
    # Save results
    output_path = Path(output_dir) / "docs"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save directly using the doc parser's method
    doc_parser.save_docs(str(output_path))
    
    print(f"Documentation parsing results saved to {output_path}")
    return docs, api_docs


def process_examples(examples_dir, output_dir):
    """Process D3 examples."""
    print(f"Processing D3 examples from {examples_dir}...")
    
    # Parse examples
    example_parser = D3ExampleParser(examples_dir)
    examples = example_parser.parse_directory()
    
    # Extract patterns
    patterns = example_parser.extract_patterns()
    
    # Categorize examples
    categories = example_parser.categorize_examples()
    
    # Save results
    output_path = Path(output_dir) / "examples"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save directly using the example parser's method
    example_parser.save_results(str(output_path))
    
    print(f"Example parsing results saved to {output_path}")
    return examples, patterns, categories


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "js": None,
        "python": None,
        "docs": None,
        "examples": None
    }
    
    # Process each component if the directory is provided
    if args.d3_js_dir:
        js_api_defs, js_patterns = process_js_codebase(args.d3_js_dir, args.output_dir)
        results["js"] = {
            "api_definitions": js_api_defs,
            "patterns": js_patterns
        }
    
    if args.d3_py_dir:
        py_api_defs, py_imports = process_py_codebase(args.d3_py_dir, args.output_dir)
        results["python"] = {
            "api_definitions": py_api_defs,
            "imports": py_imports
        }
    
    if args.docs_dir:
        docs, api_docs = process_documentation(args.docs_dir, args.output_dir)
        results["docs"] = {
            "api_docs": api_docs
        }
    
    if args.examples_dir:
        examples, ex_patterns, categories = process_examples(args.examples_dir, args.output_dir)
        results["examples"] = {
            "patterns": ex_patterns,
            "categories": categories
        }
    
    # Save summary
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, "w") as f:
        # Just save the structure, not the full data
        summary = {
            "js": bool(results["js"]),
            "python": bool(results["python"]),
            "docs": bool(results["docs"]),
            "examples": bool(results["examples"]),
            "timestamp": import_datetime.datetime.now().isoformat()
        }
        json.dump(summary, f, indent=2)
    
    print(f"Processing complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    try:
        import datetime
    except ImportError:
        import datetime as import_datetime
    
    main() 