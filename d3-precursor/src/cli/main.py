import os
import sys
import argparse
import logging
from typing import List, Optional

from .commands import (
    extract_command,
    generate_command,
    query_command,
    analyze_command
)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration.
    
    Args:
        level: The logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout)
        ]
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.
    
    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="D3 Precursor - Knowledge extraction and generation for D3.js",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to execute"
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract knowledge from D3.js source code or documentation"
    )
    extract_parser.add_argument(
        "source",
        type=str,
        help="Source directory or file to extract from"
    )
    extract_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for extracted knowledge"
    )
    extract_parser.add_argument(
        "--format",
        choices=["json", "markdown", "db"],
        default="json",
        help="Output format for extracted knowledge"
    )
    extract_parser.add_argument(
        "--extractors",
        nargs="+",
        choices=["api", "pattern", "concept", "relation"],
        default=["api"],
        help="Types of extractors to use"
    )
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate D3.js code for visualizations"
    )
    generate_parser.add_argument(
        "type",
        choices=["visualization", "component"],
        help="Type of code to generate"
    )
    generate_parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Description of what to generate"
    )
    generate_parser.add_argument(
        "--output",
        type=str,
        help="Output file for generated code"
    )
    generate_parser.add_argument(
        "--data-format",
        type=str,
        help="Format of data for the visualization"
    )
    generate_parser.add_argument(
        "--requirements",
        nargs="+",
        help="Specific requirements for the generation"
    )
    generate_parser.add_argument(
        "--use-template",
        action="store_true",
        help="Use a template if available"
    )
    generate_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model to use for generation"
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the D3.js knowledge base"
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="Query string"
    )
    query_parser.add_argument(
        "--type",
        choices=["api", "pattern", "concept", "relation"],
        help="Type of elements to query for"
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return"
    )
    query_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format for query results"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze D3.js code"
    )
    analyze_parser.add_argument(
        "source",
        type=str,
        help="Source file to analyze"
    )
    analyze_parser.add_argument(
        "--focus",
        choices=["performance", "patterns", "bugs", "general"],
        default="general",
        help="Focus of the analysis"
    )
    analyze_parser.add_argument(
        "--output",
        type=str,
        help="Output file for analysis results"
    )
    analyze_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format for analysis results"
    )
    analyze_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model to use for analysis"
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command-line arguments. If None, uses sys.argv.
        
    Returns:
        Exit code.
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    log_level = logging.DEBUG if parsed_args.debug else logging.INFO
    setup_logging(log_level)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    
    # Execute the command
    try:
        if parsed_args.command == "extract":
            return extract_command(parsed_args)
        elif parsed_args.command == "generate":
            return generate_command(parsed_args)
        elif parsed_args.command == "query":
            return query_command(parsed_args)
        elif parsed_args.command == "analyze":
            return analyze_command(parsed_args)
        else:
            parser.print_help()
            return 0
    except Exception as e:
        logger.exception(f"Error executing command: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 