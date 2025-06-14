import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio

from ..llm import LLMProvider, LLMResponse
from ..llm.prompt_templates import CodeGenTemplate
from .templates import TemplateLibrary, VisualizationTemplate, ComponentTemplate


class CodeGenerator:
    """Base class for D3.js code generators."""
    
    def __init__(self, llm_provider: LLMProvider, template_library: Optional[TemplateLibrary] = None):
        """Initialize the code generator.
        
        Args:
            llm_provider: The LLM provider to use for code generation.
            template_library: Library of templates to use. If None, a new one is created.
        """
        self.llm = llm_provider
        self.template_library = template_library or TemplateLibrary()
        self.prompt_template = CodeGenTemplate()
    
    async def _generate_code(self, prompt: str, **kwargs) -> str:
        """Generate code using the LLM provider.
        
        Args:
            prompt: The prompt to send to the LLM.
            **kwargs: Additional arguments for the LLM provider.
            
        Returns:
            The generated code.
        """
        # Set up a more code-focused system prompt
        system_prompt = """You are an expert D3.js developer. Generate clean, efficient, and well-commented code.
Follow modern JavaScript practices and D3.js conventions. Include all necessary HTML, CSS, and JS.
Focus on code quality, readability, and best practices."""
        
        response = await self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=kwargs.get("temperature", 0.2),  # Lower temperature for more deterministic code
            max_tokens=kwargs.get("max_tokens", 3000)
        )
        
        # Extract code blocks from the response
        code = self._extract_code_blocks(response.content)
        
        return code
    
    def _extract_code_blocks(self, content: str) -> str:
        """Extract code blocks from the LLM response.
        
        This handles cases where the LLM might wrap the code in markdown code blocks
        or add explanatory text.
        
        Args:
            content: The raw content from the LLM.
            
        Returns:
            The extracted code.
        """
        # Check if the content is wrapped in markdown code blocks
        if "```html" in content or "```js" in content or "```javascript" in content:
            # Extract all code blocks
            code_blocks = []
            lines = content.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.startswith("```") and not in_code_block:
                    in_code_block = True
                    # Skip this line (the opening ```)
                    continue
                elif line.startswith("```") and in_code_block:
                    in_code_block = False
                    # Join the block and add to the list
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                    # Skip this line (the closing ```)
                    continue
                
                if in_code_block:
                    current_block.append(line)
            
            # Join all code blocks with appropriate spacing
            return '\n\n'.join(code_blocks)
        
        # If no code blocks are found, return the content as is
        return content
    
    def save_code(self, code: str, file_path: Union[str, Path], overwrite: bool = False) -> bool:
        """Save the generated code to a file.
        
        Args:
            code: The code to save.
            file_path: The path to save the code to.
            overwrite: Whether to overwrite an existing file.
            
        Returns:
            True if the file was saved successfully, False otherwise.
        """
        file_path = Path(file_path)
        
        # Check if the file already exists
        if file_path.exists() and not overwrite:
            return False
        
        # Create the directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the code to the file
        with open(file_path, 'w') as f:
            f.write(code)
        
        return True


class VisualizationGenerator(CodeGenerator):
    """Generator for D3.js visualizations."""
    
    async def generate_visualization(
        self,
        visualization_type: str,
        data_format: str,
        description: str,
        requirements: List[str] = None,
        use_template: bool = True,
        **kwargs
    ) -> str:
        """Generate code for a D3.js visualization.
        
        Args:
            visualization_type: The type of visualization to generate (bar_chart, line_chart, etc.).
            data_format: Description or example of the data format.
            description: Description of the visualization to generate.
            requirements: List of specific requirements for the visualization.
            use_template: Whether to use a template from the library if available.
            **kwargs: Additional arguments for the LLM provider.
            
        Returns:
            The generated visualization code.
        """
        requirements = requirements or []
        
        # Check if we have a template for this visualization type
        if use_template and visualization_type in self.template_library.visualization_templates:
            template = self.template_library.visualization_templates[visualization_type]
            return self._apply_template(template, data_format, requirements)
        
        # Format the prompt for code generation
        prompt = self.prompt_template.format_for_visualization(
            description=f"{visualization_type}: {description}",
            data_format=data_format,
            requirements=requirements,
            additional_context=kwargs.get("additional_context", "")
        )
        
        # Generate the code
        return await self._generate_code(prompt, **kwargs)
    
    def _apply_template(
        self,
        template: VisualizationTemplate,
        data_format: str,
        requirements: List[str]
    ) -> str:
        """Apply a visualization template to generate code.
        
        Args:
            template: The template to apply.
            data_format: Description or example of the data format.
            requirements: List of specific requirements for the visualization.
            
        Returns:
            The generated code from the template.
        """
        # Convert data format from description to actual format if needed
        data_variable = "const data = " + data_format
        if "const data" not in data_format and "[" not in data_format:
            # This is a description, so use a sample data set
            if "time series" in data_format.lower():
                data_variable = template.sample_time_data
            elif "categorical" in data_format.lower():
                data_variable = template.sample_categorical_data
            elif "network" in data_format.lower() or "graph" in data_format.lower():
                data_variable = template.sample_network_data
            elif "geographic" in data_format.lower() or "map" in data_format.lower():
                data_variable = template.sample_geo_data
            else:
                data_variable = template.sample_data
        
        # Apply any customizations from requirements
        customizations = {}
        for req in requirements:
            if "color" in req.lower() or "palette" in req.lower():
                customizations["colorScale"] = self._extract_color_requirement(req)
            if "width" in req.lower() or "height" in req.lower():
                dimensions = self._extract_dimensions(req)
                if "width" in dimensions:
                    customizations["width"] = dimensions["width"]
                if "height" in dimensions:
                    customizations["height"] = dimensions["height"]
            if "title" in req.lower():
                customizations["title"] = self._extract_title(req)
            if "margin" in req.lower():
                customizations["margin"] = self._extract_margins(req)
            if "animation" in req.lower() or "transition" in req.lower():
                customizations["animate"] = True
                
        # Apply the template
        return template.apply(data_variable, customizations)
    
    def _extract_color_requirement(self, requirement: str) -> str:
        """Extract color scheme from a requirement.
        
        Args:
            requirement: The requirement string.
            
        Returns:
            A D3 color scale definition.
        """
        if "categorical" in requirement.lower():
            return "d3.scaleOrdinal(d3.schemeCategory10)"
        if "sequential" in requirement.lower():
            return "d3.scaleSequential(d3.interpolateViridis)"
        if "diverging" in requirement.lower():
            return "d3.scaleDiverging(d3.interpolateRdBu)"
        
        # Default to category10
        return "d3.scaleOrdinal(d3.schemeCategory10)"
    
    def _extract_dimensions(self, requirement: str) -> Dict[str, int]:
        """Extract width and height from a requirement.
        
        Args:
            requirement: The requirement string.
            
        Returns:
            A dictionary with width and/or height.
        """
        dimensions = {}
        
        # Extract numbers from the requirement
        import re
        numbers = re.findall(r'\d+', requirement)
        
        if len(numbers) >= 2:
            dimensions["width"] = int(numbers[0])
            dimensions["height"] = int(numbers[1])
        elif len(numbers) == 1:
            if "width" in requirement.lower():
                dimensions["width"] = int(numbers[0])
            elif "height" in requirement.lower():
                dimensions["height"] = int(numbers[0])
        
        return dimensions
    
    def _extract_title(self, requirement: str) -> str:
        """Extract title from a requirement.
        
        Args:
            requirement: The requirement string.
            
        Returns:
            The extracted title.
        """
        if ":" in requirement:
            return requirement.split(":", 1)[1].strip()
        return ""
    
    def _extract_margins(self, requirement: str) -> Dict[str, int]:
        """Extract margins from a requirement.
        
        Args:
            requirement: The requirement string.
            
        Returns:
            A dictionary with margin values.
        """
        margins = {"top": 20, "right": 20, "bottom": 30, "left": 40}
        
        # Extract numbers from the requirement
        import re
        numbers = re.findall(r'\d+', requirement)
        
        if len(numbers) == 1:
            # Same margin for all sides
            value = int(numbers[0])
            margins = {"top": value, "right": value, "bottom": value, "left": value}
        elif len(numbers) == 4:
            # Different margins for each side (top, right, bottom, left)
            margins = {
                "top": int(numbers[0]),
                "right": int(numbers[1]),
                "bottom": int(numbers[2]),
                "left": int(numbers[3])
            }
        
        return margins


class ComponentGenerator(CodeGenerator):
    """Generator for D3.js components and utilities."""
    
    async def generate_component(
        self,
        component_type: str,
        description: str,
        dependencies: List[str] = None,
        use_template: bool = True,
        **kwargs
    ) -> str:
        """Generate code for a D3.js component.
        
        Args:
            component_type: The type of component to generate (tooltip, legend, etc.).
            description: Description of the component to generate.
            dependencies: List of dependencies for the component.
            use_template: Whether to use a template from the library if available.
            **kwargs: Additional arguments for the LLM provider.
            
        Returns:
            The generated component code.
        """
        dependencies = dependencies or []
        requirements = kwargs.get("requirements", [])
        
        # Check if we have a template for this component type
        if use_template and component_type in self.template_library.component_templates:
            template = self.template_library.component_templates[component_type]
            return self._apply_template(template, description, requirements)
        
        # Format the prompt for code generation
        additional_context = f"""
Dependencies: {', '.join(dependencies)}

{kwargs.get('additional_context', '')}
"""
        
        prompt = self.prompt_template.format_for_visualization(
            description=f"{component_type} component: {description}",
            data_format=kwargs.get("data_format", "Not applicable for this component."),
            requirements=requirements,
            additional_context=additional_context
        )
        
        # Generate the code
        return await self._generate_code(prompt, **kwargs)
    
    def _apply_template(
        self,
        template: ComponentTemplate,
        description: str,
        requirements: List[str]
    ) -> str:
        """Apply a component template to generate code.
        
        Args:
            template: The template to apply.
            description: Description of the component.
            requirements: List of specific requirements for the component.
            
        Returns:
            The generated code from the template.
        """
        # Apply any customizations from requirements
        customizations = {"description": description}
        
        for req in requirements:
            if "function name" in req.lower():
                customizations["functionName"] = req.split(":", 1)[1].strip() if ":" in req else ""
            if "style" in req.lower():
                customizations["style"] = req.split(":", 1)[1].strip() if ":" in req else ""
            if "behavior" in req.lower():
                customizations["behavior"] = req.split(":", 1)[1].strip() if ":" in req else ""
                
        # Apply the template
        return template.apply(customizations) 