import os
import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ..extractors import (
    ApiExtractor,
    PatternExtractor,
    ConceptExtractor,
    RelationExtractor
)
from ..knowledge_base import (
    KnowledgeStorage,
    KnowledgeQuery,
    ApiElement,
    ApiElementType
)
from ..llm import (
    OpenAIProvider,
    AnthropicProvider,
    AnalysisTemplate
)
from ..code_gen import (
    VisualizationGenerator,
    ComponentGenerator,
    TemplateLibrary
)
from .progress import spinner, create_progress_bar
from .colors import error, warning, info, highlight, api_name, code, heading


# Set up logger
logger = logging.getLogger(__name__)


def get_llm_provider(args: argparse.Namespace):
    """Get the LLM provider based on command-line arguments.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        The configured LLM provider.
    """
    model = args.model if hasattr(args, "model") else "gpt-4"
    
    # Determine provider type based on model name
    if model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIProvider(api_key=api_key, model=model)


def extract_command(args: argparse.Namespace) -> int:
    """Execute the extract command.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    source_path = Path(args.source)
    
    # Determine output path
    output_path = Path(args.output) if args.output else Path.cwd() / "d3_knowledge"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if source exists
    if not source_path.exists():
        logger.error(f"Source path does not exist: {source_path}")
        return 1
    
    # Initialize extractors based on command-line arguments
    extractors = []
    if "api" in args.extractors:
        extractors.append(ApiExtractor())
    if "pattern" in args.extractors:
        extractors.append(PatternExtractor())
    if "concept" in args.extractors:
        extractors.append(ConceptExtractor())
    if "relation" in args.extractors:
        extractors.append(RelationExtractor())
    
    # Initialize knowledge storage
    storage = KnowledgeStorage(output_path / "db")
    
    # Create a spinner for the initialization phase
    spin = spinner("Initializing extractors")
    spin.start()
    
    try:
        # Extract knowledge
        total_extracted = 0
        for extractor in extractors:
            # Stop the spinner with success
            spin.stop(True)
            
            logger.info(f"Running {extractor.__class__.__name__}...")
            
            # Run the extractor with progress tracking
            spin = spinner(f"Running {extractor.__class__.__name__}")
            spin.start()
            
            try:
                extracted_elements = extractor.extract(source_path)
                
                # Stop spinner once extraction is complete
                spin.stop(True)
                
                # Show progress bar for saving elements
                element_count = len(extracted_elements)
                progress = create_progress_bar(
                    total=element_count,
                    desc=f"Saving {element_count} elements from {extractor.__class__.__name__}"
                )
                
                # Save the extracted elements with progress
                for element in extracted_elements:
                    element_id = storage.add_element(element)
                    total_extracted += 1
                    
                    # Also save as JSON if requested
                    if args.format == "json":
                        element_json = element_to_json(element)
                        element_type = element.type.name.lower()
                        json_path = output_path / "json" / element_type
                        json_path.mkdir(parents=True, exist_ok=True)
                        
                        with open(json_path / f"{element.name}.json", "w") as f:
                            json.dump(element_json, f, indent=2)
                    
                    # Also save as Markdown if requested
                    if args.format == "markdown":
                        markdown = element_to_markdown(element)
                        md_path = output_path / "markdown" / element.type.name.lower()
                        md_path.mkdir(parents=True, exist_ok=True)
                        
                        with open(md_path / f"{element.name}.md", "w") as f:
                            f.write(markdown)
                    
                    # Update progress bar
                    progress.update()
                
            except Exception as e:
                # Stop spinner with failure if an error occurs
                spin.stop(False)
                logger.error(f"Error running {extractor.__class__.__name__}: {e}")
                if args.debug:
                    logger.exception(e)
    except Exception as e:
        # Stop spinner with failure if an error occurs
        spin.stop(False)
        logger.error(f"Error during extraction: {e}")
        if args.debug:
            logger.exception(e)
        return 1
    
    logger.info(f"Extracted {total_extracted} elements to {output_path}")
    return 0


def generate_command(args: argparse.Namespace) -> int:
    """Execute the generate command.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    # Get LLM provider
    try:
        llm_provider = get_llm_provider(args)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Initialize template library
    template_library = TemplateLibrary()
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        # Create a default output filename based on the description
        sanitized_name = "".join(c if c.isalnum() else "_" for c in args.description.lower())
        sanitized_name = sanitized_name[:30]  # Limit length
        
        if args.type == "visualization":
            output_file = Path.cwd() / f"{sanitized_name}.html"
        else:
            output_file = Path.cwd() / f"{sanitized_name}.js"
    
    # Create the code generator
    if args.type == "visualization":
        generator = VisualizationGenerator(llm_provider, template_library)
        
        # Default data format if not provided
        data_format = args.data_format or "Array of objects with name and value properties"
        
        # Run the generator
        try:
            # Use asyncio to run the async function
            code = asyncio.run(generator.generate_visualization(
                visualization_type=sanitized_name,
                data_format=data_format,
                description=args.description,
                requirements=args.requirements,
                use_template=args.use_template
            ))
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return 1
    else:
        generator = ComponentGenerator(llm_provider, template_library)
        
        # Run the generator
        try:
            # Use asyncio to run the async function
            code = asyncio.run(generator.generate_component(
                component_type=sanitized_name,
                description=args.description,
                dependencies=args.requirements,
                use_template=args.use_template
            ))
        except Exception as e:
            logger.error(f"Error generating component: {e}")
            return 1
    
    # Save the code
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(code)
        logger.info(f"Generated code saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving generated code: {e}")
        return 1
    
    return 0


def query_command(args: argparse.Namespace) -> int:
    """Execute the query command.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    query_string = args.query
    
    # Get output format
    output_format = args.format
    
    # Initialize the knowledge query
    try:
        knowledge_query = KnowledgeQuery()
    except Exception as e:
        logger.error(f"Error initializing knowledge query: {e}")
        return 1
        
    # Apply query filters
    if args.type:
        # Convert string to ApiElementType enum
        try:
            element_type = ApiElementType[args.type.upper()]
            knowledge_query.filter_by_type(element_type)
        except KeyError:
            # Handle invalid type
            logger.error(f"Invalid element type: {args.type}")
            return 1
    
    # Create spinner for query execution
    spin = spinner(f"Searching for '{highlight(query_string)}'")
    spin.start()
    
    try:
        # Execute the query
        results = knowledge_query.search(query_string, limit=args.limit)
        
        # Stop spinner
        spin.stop(True)
        
        # Format and display results
        if output_format == "json":
            # Convert to JSON
            results_json = []
            for result in results:
                results_json.append(element_to_json(result.element))
                
            print(json.dumps(results_json, indent=2))
            
        elif output_format == "markdown":
            # Convert to Markdown
            for i, result in enumerate(results):
                print(heading(f"Result {i+1} (Score: {result.score:.2f})"))
                print(element_to_markdown(result.element))
                print()
                
        else:  # text format
            if not results:
                print(warning("No results found."))
            else:
                print(heading(f"Found {len(results)} results:"))
                print()
                
                for i, result in enumerate(results):
                    element = result.element
                    print(heading(f"Result {i+1} (Score: {result.score:.2f})"))
                    print(f"Name: {api_name(element.name)}")
                    print(f"Type: {info(element.type.name)}")
                    
                    if element.description:
                        print(f"Description: {element.description}")
                    
                    if hasattr(element, "code") and element.code:
                        print(f"Code: {code(element.code[:200])}" + ("..." if len(element.code) > 200 else ""))
                    
                    print()
        
        return 0
        
    except Exception as e:
        # Stop spinner with error
        spin.stop(False)
        logger.error(f"Error executing query: {e}")
        if args.debug:
            logger.exception(e)
        return 1


def analyze_command(args: argparse.Namespace) -> int:
    """Execute the analyze command.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    source_path = Path(args.source)
    
    # Check if source exists
    if not source_path.exists():
        logger.error(f"Source file does not exist: {source_path}")
        return 1
    
    # Read the source file
    try:
        with open(source_path, "r") as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Error reading source file: {e}")
        return 1
    
    # Get LLM provider
    try:
        llm_provider = get_llm_provider(args)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Create analysis template
    template = AnalysisTemplate()
    
    # Format the prompt
    prompt = template.format_for_analysis(
        code=code,
        focus=args.focus
    )
    
    # Run the analysis
    try:
        # Use asyncio to run the async function
        response = asyncio.run(llm_provider.generate(prompt))
        analysis = response.content
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        return 1
    
    # Format the analysis
    if args.format == "json":
        # Try to parse the analysis as JSON, or wrap it in a JSON object
        try:
            json_analysis = json.loads(analysis)
            output = json.dumps(json_analysis, indent=2)
        except json.JSONDecodeError:
            output = json.dumps({"analysis": analysis}, indent=2)
    elif args.format == "markdown":
        output = f"# Analysis of {source_path.name}\n\n{analysis}"
    else:
        output = analysis
    
    # Output the analysis
    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(output)
            logger.info(f"Analysis saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return 1
    else:
        print(output)
    
    return 0


def element_to_json(element: ApiElement) -> Dict[str, Any]:
    """Convert an API element to a JSON-serializable dictionary.
    
    Args:
        element: The API element to convert.
        
    Returns:
        A JSON-serializable dictionary representation of the element.
    """
    # Create a dictionary with basic properties
    result = {
        "name": element.name,
        "type": element.type.name,
        "description": element.description,
        "tags": element.tags or [],
        "related_elements": element.related_elements or []
    }
    
    # Add parameters if present
    if element.parameters:
        result["parameters"] = [
            {
                "name": param.name,
                "type": param.type.name,
                "description": param.description,
                "required": param.required,
                "default_value": param.default_value
            }
            for param in element.parameters
        ]
    
    # Add return value if present
    if element.return_value:
        result["return_value"] = {
            "type": element.return_value.type.name if hasattr(element.return_value.type, "name") else [t.name for t in element.return_value.type],
            "description": element.return_value.description,
            "example": element.return_value.example
        }
    
    # Add examples if present
    if element.examples:
        result["examples"] = [
            {
                "code": example.code,
                "description": example.description
            }
            for example in element.examples
        ]
    
    # Add usage patterns if present
    if element.usage_patterns:
        result["usage_patterns"] = [
            {
                "name": pattern.name,
                "description": pattern.description,
                "code_template": pattern.code_template,
                "visualization_type": pattern.visualization_type.name if pattern.visualization_type else None
            }
            for pattern in element.usage_patterns
        ]
    
    # Add other metadata
    if element.source_url:
        result["source_url"] = element.source_url
    if element.version_introduced:
        result["version_introduced"] = element.version_introduced
    if element.version_deprecated:
        result["version_deprecated"] = element.version_deprecated
    if element.metadata:
        result["metadata"] = element.metadata
    
    return result


def element_to_markdown(element: ApiElement) -> str:
    """Convert an API element to a Markdown representation.
    
    Args:
        element: The API element to convert.
        
    Returns:
        A Markdown representation of the element.
    """
    lines = []
    
    # Add title and type
    lines.append(f"# {element.name}")
    lines.append(f"**Type**: {element.type.name}")
    lines.append("")
    
    # Add description
    lines.append(element.description)
    lines.append("")
    
    # Add parameters if present
    if element.parameters:
        lines.append("## Parameters")
        lines.append("")
        
        for param in element.parameters:
            required_str = "Required" if param.required else "Optional"
            default_str = f", Default: `{param.default_value}`" if param.default_value is not None else ""
            lines.append(f"### {param.name}")
            lines.append(f"**Type**: {param.type.name}  ")
            lines.append(f"**Requirement**: {required_str}{default_str}")
            lines.append("")
            lines.append(param.description)
            lines.append("")
    
    # Add return value if present
    if element.return_value:
        lines.append("## Return Value")
        lines.append("")
        
        if hasattr(element.return_value.type, "name"):
            type_str = element.return_value.type.name
        else:
            type_str = ", ".join(t.name for t in element.return_value.type)
        
        lines.append(f"**Type**: {type_str}")
        lines.append("")
        lines.append(element.return_value.description)
        
        if element.return_value.example:
            lines.append("")
            lines.append("**Example**:")
            lines.append("```js")
            lines.append(str(element.return_value.example))
            lines.append("```")
        
        lines.append("")
    
    # Add examples if present
    if element.examples:
        lines.append("## Examples")
        lines.append("")
        
        for i, example in enumerate(element.examples, 1):
            lines.append(f"### Example {i}")
            lines.append("")
            lines.append(example.description)
            lines.append("")
            lines.append("```js")
            lines.append(example.code)
            lines.append("```")
            lines.append("")
    
    # Add usage patterns if present
    if element.usage_patterns:
        lines.append("## Usage Patterns")
        lines.append("")
        
        for pattern in element.usage_patterns:
            lines.append(f"### {pattern.name}")
            lines.append("")
            lines.append(pattern.description)
            
            if pattern.visualization_type:
                lines.append(f"**Visualization Type**: {pattern.visualization_type.name}")
            
            lines.append("")
            lines.append("```js")
            lines.append(pattern.code_template)
            lines.append("```")
            lines.append("")
    
    # Add metadata
    lines.append("## Metadata")
    lines.append("")
    
    if element.tags:
        lines.append(f"**Tags**: {', '.join(element.tags)}")
    
    if element.related_elements:
        lines.append(f"**Related Elements**: {', '.join(element.related_elements)}")
    
    if element.source_url:
        lines.append(f"**Source**: [{element.source_url}]({element.source_url})")
    
    if element.version_introduced:
        lines.append(f"**Introduced in Version**: {element.version_introduced}")
    
    if element.version_deprecated:
        lines.append(f"**Deprecated in Version**: {element.version_deprecated}")
    
    return "\n".join(lines)


def element_to_text(element: ApiElement) -> str:
    """Convert an API element to a plain text representation.
    
    Args:
        element: The API element to convert.
        
    Returns:
        A plain text representation of the element.
    """
    lines = []
    
    # Add title and type
    lines.append(f"{element.name} ({element.type.name})")
    lines.append("=" * len(f"{element.name} ({element.type.name})"))
    lines.append("")
    
    # Add description
    lines.append(element.description)
    lines.append("")
    
    # Add parameters if present
    if element.parameters:
        lines.append("Parameters:")
        lines.append("-" * 10)
        
        for param in element.parameters:
            required_str = "Required" if param.required else "Optional"
            default_str = f", Default: {param.default_value}" if param.default_value is not None else ""
            lines.append(f"  {param.name} ({param.type.name}, {required_str}{default_str})")
            lines.append(f"    {param.description}")
            lines.append("")
    
    # Add return value if present
    if element.return_value:
        lines.append("Return Value:")
        lines.append("-" * 12)
        
        if hasattr(element.return_value.type, "name"):
            type_str = element.return_value.type.name
        else:
            type_str = ", ".join(t.name for t in element.return_value.type)
        
        lines.append(f"  Type: {type_str}")
        lines.append(f"  {element.return_value.description}")
        
        if element.return_value.example:
            lines.append(f"  Example: {element.return_value.example}")
        
        lines.append("")
    
    # Add metadata
    if element.tags or element.related_elements or element.source_url or element.version_introduced or element.version_deprecated:
        lines.append("Metadata:")
        lines.append("-" * 9)
        
        if element.tags:
            lines.append(f"  Tags: {', '.join(element.tags)}")
        
        if element.related_elements:
            lines.append(f"  Related Elements: {', '.join(element.related_elements)}")
        
        if element.source_url:
            lines.append(f"  Source: {element.source_url}")
        
        if element.version_introduced:
            lines.append(f"  Introduced in Version: {element.version_introduced}")
        
        if element.version_deprecated:
            lines.append(f"  Deprecated in Version: {element.version_deprecated}")
    
    return "\n".join(lines) 