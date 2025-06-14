from string import Template
from typing import Dict, Any, List, Optional


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, template: str):
        """Initialize the prompt template.
        
        Args:
            template: The template string with placeholders.
        """
        self.template = Template(template)
    
    def format(self, **kwargs) -> str:
        """Format the template with the given values.
        
        Args:
            **kwargs: Values for the template placeholders.
            
        Returns:
            The formatted prompt.
        """
        return self.template.safe_substitute(**kwargs)


class CodeGenTemplate(PromptTemplate):
    """Template for D3.js code generation."""
    
    # Default template for generating D3.js code
    DEFAULT_TEMPLATE = """
Generate D3.js code for the following visualization:

Description: ${description}

Data format: 
${data_format}

Requirements:
${requirements}

${additional_context}

The code should follow D3.js best practices and be well-commented.
Include all necessary HTML, CSS, and JavaScript for a complete solution.
"""
    
    def __init__(self, template: Optional[str] = None):
        """Initialize the code generation template.
        
        Args:
            template: Custom template string. If None, uses the default template.
        """
        super().__init__(template or self.DEFAULT_TEMPLATE)
    
    def format_for_visualization(
        self,
        description: str,
        data_format: str,
        requirements: List[str],
        additional_context: str = ""
    ) -> str:
        """Format the template for a D3.js visualization.
        
        Args:
            description: Description of the visualization to generate.
            data_format: Format of the data to be visualized.
            requirements: List of specific requirements for the visualization.
            additional_context: Any additional context or instructions.
            
        Returns:
            The formatted prompt.
        """
        # Format the requirements as a bulleted list
        requirements_formatted = "\n".join(f"- {req}" for req in requirements)
        
        return self.format(
            description=description,
            data_format=data_format,
            requirements=requirements_formatted,
            additional_context=additional_context
        )


class DocGenTemplate(PromptTemplate):
    """Template for generating D3.js documentation."""
    
    # Default template for generating D3.js documentation
    DEFAULT_TEMPLATE = """
Generate comprehensive documentation for the following D3.js ${element_type}:

${code}

Requirements:
${requirements}

${additional_context}

Your documentation should include:
- A clear description of what the ${element_type} does
- Parameter explanations with types
- Return value explanation
- Usage examples
- Common patterns or pitfalls
"""
    
    def __init__(self, template: Optional[str] = None):
        """Initialize the documentation generation template.
        
        Args:
            template: Custom template string. If None, uses the default template.
        """
        super().__init__(template or self.DEFAULT_TEMPLATE)
    
    def format_for_element(
        self,
        code: str,
        element_type: str = "function",
        requirements: List[str] = None,
        additional_context: str = ""
    ) -> str:
        """Format the template for D3.js element documentation.
        
        Args:
            code: The code to document.
            element_type: Type of element being documented (function, method, class, etc.).
            requirements: Specific requirements for the documentation.
            additional_context: Any additional context or instructions.
            
        Returns:
            The formatted prompt.
        """
        requirements = requirements or []
        requirements_formatted = "\n".join(f"- {req}" for req in requirements)
        
        return self.format(
            code=code,
            element_type=element_type,
            requirements=requirements_formatted,
            additional_context=additional_context
        )


class AnalysisTemplate(PromptTemplate):
    """Template for analyzing D3.js code."""
    
    # Default template for analyzing D3.js code
    DEFAULT_TEMPLATE = """
Analyze the following D3.js code:

${code}

Analysis focus: ${focus}

${additional_context}

Provide insights on:
- The visualization type and purpose
- Key D3.js features and patterns used
- Potential optimizations or improvements
- Any potential issues or bugs
"""
    
    def __init__(self, template: Optional[str] = None):
        """Initialize the analysis template.
        
        Args:
            template: Custom template string. If None, uses the default template.
        """
        super().__init__(template or self.DEFAULT_TEMPLATE)
    
    def format_for_analysis(
        self,
        code: str,
        focus: str = "general",
        additional_context: str = ""
    ) -> str:
        """Format the template for D3.js code analysis.
        
        Args:
            code: The code to analyze.
            focus: The focus of the analysis (performance, readability, etc.).
            additional_context: Any additional context or instructions.
            
        Returns:
            The formatted prompt.
        """
        return self.format(
            code=code,
            focus=focus,
            additional_context=additional_context
        ) 