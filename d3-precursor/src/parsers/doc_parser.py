"""
Documentation parser for D3.

This module provides functionality to parse D3 documentation
in Markdown and HTML formats and extract structured information.
"""

import os
import re
import json
import markdown
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional, Set, Tuple


class D3DocParser:
    """Parse D3 documentation in Markdown and HTML formats."""
    
    def __init__(self, docs_path: str):
        """
        Initialize the parser with the path to documentation files.
        
        Args:
            docs_path: Path to the documentation directory
        """
        self.docs_path = docs_path
        self.api_docs = {}
        self.examples = []
        self.api_index = {}  # Maps API names to their documentation
        
    def parse_markdown(self, file_path: str) -> Dict[str, Any]:
        """
        Parse markdown documentation into structured format.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dictionary with structured documentation
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Convert markdown to HTML
        html = markdown.markdown(content, extensions=['fenced_code', 'tables'])
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structured information from HTML
        result = {
            'title': self._extract_title(soup),
            'sections': self._extract_sections(soup),
            'api_elements': self._extract_api_elements(soup),
            'examples': self._extract_examples(soup),
            'source': file_path
        }
        
        return result
    
    def parse_html(self, file_path: str) -> Dict[str, Any]:
        """
        Parse HTML documentation into structured format.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Dictionary with structured documentation
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract structured information from HTML
        result = {
            'title': self._extract_title(soup),
            'sections': self._extract_sections(soup),
            'api_elements': self._extract_api_elements(soup),
            'examples': self._extract_examples(soup),
            'source': file_path
        }
        
        return result
    
    def _extract_title(self, soup) -> str:
        """
        Extract title from parsed document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Document title
        """
        h1 = soup.find('h1')
        if h1:
            return h1.text.strip()
            
        # Fallback to looking for the first heading
        for tag in ['h2', 'h3', 'h4']:
            heading = soup.find(tag)
            if heading:
                return heading.text.strip()
                
        return "Untitled Document"
    
    def _extract_sections(self, soup) -> List[Dict[str, Any]]:
        """
        Extract sections from document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Get all headings
        headings = soup.find_all(['h2', 'h3', 'h4'])
        
        for i, heading in enumerate(headings):
            # Get content until the next heading
            content = []
            current = heading.next_sibling
            
            while current and (i == len(headings) - 1 or current != headings[i + 1]):
                if current.name:  # Only process HTML elements, not NavigableString
                    content.append(str(current))
                current = current.next_sibling
                
                # Break if we hit the next heading
                if current in headings:
                    break
            
            sections.append({
                'heading': heading.text.strip(),
                'level': int(heading.name[1]),
                'content': ''.join(content)
            })
            
        return sections
    
    def _extract_api_elements(self, soup) -> List[Dict[str, Any]]:
        """
        Extract API documentation elements.
        
        Looks for function/method signatures, parameters, and descriptions.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of API element dictionaries
        """
        api_elements = []
        
        # Look for code blocks that might contain API signatures
        code_blocks = soup.find_all('code')
        
        for code in code_blocks:
            # Check if this looks like an API signature
            text = code.text.strip()
            
            # Look for function/method signatures like "d3.select()", "selection.attr()", etc.
            if self._is_api_signature(text):
                element = self._parse_api_signature(text)
                
                # Get description from surrounding text
                description = self._get_surrounding_text(code)
                element['description'] = description
                
                # Check for parameter descriptions
                element['parameters'] = self._extract_parameters(code)
                
                # Check for return value descriptions
                element['returns'] = self._extract_return_value(code)
                
                api_elements.append(element)
        
        return api_elements
    
    def _is_api_signature(self, text: str) -> bool:
        """Check if text looks like a D3 API signature."""
        # Patterns to match D3 API signatures
        patterns = [
            r'^d3\.\w+\s*\(',  # d3.funcName(
            r'^d3\.\w+\.\w+\s*\(',  # d3.namespace.funcName(
            r'^\w+\.\w+\s*\(',  # selection.attr(
            r'^\w+\s*\(\s*\[?\s*\w+',  # scale(domain
        ]
        
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _parse_api_signature(self, text: str) -> Dict[str, Any]:
        """Parse an API signature into its components."""
        # Clean up the signature
        text = text.replace('\n', ' ').strip()
        
        # Extract name and namespace
        pattern = r'((?:\w+\.)+)?(\w+)\s*\((.*)\)'
        match = re.search(pattern, text)
        
        if match:
            namespace = match.group(1).rstrip('.') if match.group(1) else None
            name = match.group(2)
            param_str = match.group(3)
            
            # Parse parameters
            params = []
            if param_str:
                # Handle complex parameter strings
                param_list = self._split_parameters(param_str)
                for p in param_list:
                    p = p.strip()
                    if p:
                        # Check for parameter with default value
                        param_match = re.match(r'(\w+)(?:\s*=\s*(.+))?', p)
                        if param_match:
                            param_name = param_match.group(1)
                            default_value = param_match.group(2) if param_match.group(2) else None
                            params.append({
                                'name': param_name,
                                'default': default_value
                            })
                        else:
                            params.append({'name': p})
            
            return {
                'type': 'function',
                'namespace': namespace,
                'name': name,
                'full_name': f"{namespace}.{name}" if namespace else name,
                'signature': text,
                'param_list': params
            }
        
        return {
            'type': 'unknown',
            'signature': text
        }
    
    def _split_parameters(self, param_str: str) -> List[str]:
        """Split parameter string handling nested brackets correctly."""
        params = []
        current = ""
        bracket_depth = 0
        
        for char in param_str:
            if char == ',' and bracket_depth == 0:
                params.append(current.strip())
                current = ""
            else:
                current += char
                if char in '[{(':
                    bracket_depth += 1
                elif char in ']})':
                    bracket_depth -= 1
        
        if current.strip():
            params.append(current.strip())
            
        return params
    
    def _get_surrounding_text(self, element) -> str:
        """
        Get descriptive text surrounding a code element.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Surrounding text as a string
        """
        # Get parent paragraph or list item if any
        parent = element.parent
        
        # Check if the element is inside a paragraph
        if parent.name == 'p':
            return parent.text.strip()
            
        # Check if the element is inside a list item
        if parent.name == 'li':
            return parent.text.strip()
            
        # Check if the element is inside a header
        if parent.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return parent.text.strip()
            
        # Try to get the next paragraph
        next_p = element.find_next('p')
        if next_p:
            return next_p.text.strip()
            
        # Return empty string if no description found
        return ""
    
    def _extract_parameters(self, element) -> List[Dict[str, Any]]:
        """
        Extract parameter descriptions for an API element.
        
        Args:
            element: BeautifulSoup element containing API signature
            
        Returns:
            List of parameter dictionaries with name and description
        """
        parameters = []
        
        # Look for parameter descriptions in subsequent list
        next_list = element.find_next(['ul', 'ol', 'dl'])
        
        if next_list and next_list.name == 'dl':
            # Handle definition lists
            terms = next_list.find_all('dt')
            for term in terms:
                # Check if this looks like a parameter name
                param_name = term.text.strip()
                
                # Skip if it doesn't look like a parameter
                if not param_name or '(' in param_name or '=' in param_name:
                    continue
                
                # Get description from following dd element
                desc_elem = term.find_next('dd')
                description = desc_elem.text.strip() if desc_elem else ""
                
                parameters.append({
                    'name': param_name,
                    'description': description
                })
        elif next_list:
            # Handle regular lists
            items = next_list.find_all('li')
            for item in items:
                text = item.text.strip()
                
                # Try to extract parameter name and description
                param_match = re.match(r'^(\w+)(?:\s*:\s*[^-]*)?\s*[-–]\s*(.+)$', text)
                if param_match:
                    param_name = param_match.group(1)
                    description = param_match.group(2)
                    
                    parameters.append({
                        'name': param_name,
                        'description': description
                    })
        
        return parameters
    
    def _extract_return_value(self, element) -> Optional[Dict[str, Any]]:
        """
        Extract return value description for an API element.
        
        Args:
            element: BeautifulSoup element containing API signature
            
        Returns:
            Dictionary with return type and description, or None
        """
        # Look for return value description in text after signature
        parent = element.parent
        
        # Check for text that mentions "returns"
        return_pattern = r'(?:returns|returning)\s+(?:a|an|the)?\s*(\w+)(?:[.,:]|\s+(\w+))?\s+(.+?)(?:\.|\n|$)'
        
        # Search in surrounding paragraphs
        for para in parent.find_next_siblings('p', limit=3):
            text = para.text.lower()
            if 'return' in text:
                match = re.search(return_pattern, text, re.IGNORECASE)
                if match:
                    return_type = match.group(1)
                    if match.group(2):  # Handle cases like "returns a new selection"
                        return_type += " " + match.group(2)
                    description = match.group(3)
                    
                    return {
                        'type': return_type,
                        'description': description
                    }
        
        # Check for explicit "Returns:" section
        for sibling in parent.find_next_siblings():
            if sibling.name in ['h3', 'h4', 'h5', 'strong', 'b'] and 'return' in sibling.text.lower():
                description = ""
                next_elem = sibling.find_next(['p', 'div'])
                if next_elem:
                    description = next_elem.text.strip()
                
                return {
                    'type': 'unknown',
                    'description': description
                }
        
        return None
    
    def _extract_examples(self, soup) -> List[Dict[str, Any]]:
        """
        Extract code examples from document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of example dictionaries
        """
        examples = []
        
        # Find all pre/code blocks
        code_blocks = soup.find_all(['pre', 'code'])
        
        for block in code_blocks:
            # Extract the code content
            code = block.text.strip()
            
            # Skip if too short
            if len(code) < 10:
                continue
                
            # Try to determine the language
            language = "javascript"  # Default to JavaScript for D3 docs
            
            # Check for class hints about language
            if block.has_attr('class'):
                classes = block['class']
                for cls in classes:
                    if 'language-' in cls:
                        language = cls.replace('language-', '')
                    elif cls in ['js', 'javascript', 'html', 'css', 'python']:
                        language = cls
            
            # Check if this is a D3 example
            if self._is_d3_example(code, language):
                # Get description from surrounding text
                description = ""
                
                # Check for preceding paragraph
                prev_p = block.find_previous('p')
                if prev_p:
                    description = prev_p.text.strip()
                
                # Check for heading above
                prev_h = block.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                title = prev_h.text.strip() if prev_h else ""
                
                examples.append({
                    'code': code,
                    'language': language,
                    'title': title,
                    'description': description,
                    'source': self._get_source_path(block)
                })
        
        return examples
    
    def _get_source_path(self, element) -> str:
        """Get source file path or URL for an element."""
        # Try to find a link to the source
        for parent in element.parents:
            if parent.name == 'a' and parent.has_attr('href'):
                return parent['href']
        
        # Default to the source document
        return "unknown"
    
    def _is_d3_example(self, code: str, language: str) -> bool:
        """Check if a code block is a D3 example."""
        if language.lower() in ['js', 'javascript']:
            return 'd3.' in code or 'D3' in code
        elif language.lower() == 'html':
            return '<script' in code and ('d3.' in code or 'D3' in code)
        elif language.lower() == 'python':
            return 'import d3' in code or 'from d3' in code
        
        return False
    
    def process_directory(self, dir_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all documentation files in a directory.
        
        Args:
            dir_path: Path to the directory. If None, uses docs_path.
            
        Returns:
            Dictionary with processed documentation
        """
        if dir_path is None:
            dir_path = self.docs_path
            
        results = {
            'api_elements': [],
            'examples': [],
            'processed_files': []
        }
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.md'):
                    doc = self.parse_markdown(file_path)
                elif file.endswith('.html'):
                    doc = self.parse_html(file_path)
                else:
                    continue
                
                results['api_elements'].extend(doc['api_elements'])
                results['examples'].extend(doc['examples'])
                results['processed_files'].append(file_path)
                
                # Update API index
                for api in doc['api_elements']:
                    if 'full_name' in api:
                        self.api_index[api['full_name']] = api
        
        self.api_docs = results
        return results
    
    def extract_api_docs(self) -> Dict[str, Any]:
        """
        Extract and structure API documentation.
        
        Returns:
            Dictionary mapping API names to their documentation
        """
        # Make sure we've processed the documentation
        if not self.api_docs:
            self.process_directory()
            
        return self.api_index
    
    def get_api_doc(self, api_name: str) -> Optional[Dict[str, Any]]:
        """
        Get documentation for a specific API element.
        
        Args:
            api_name: Name of the API element
            
        Returns:
            API documentation or None if not found
        """
        return self.api_index.get(api_name)
    
    def save_docs(self, output_dir: str) -> None:
        """
        Save processed documentation to files.
        
        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save API documentation
        api_path = os.path.join(output_dir, 'api_docs.json')
        with open(api_path, 'w', encoding='utf-8') as f:
            json.dump(self.api_index, f, indent=2)
            
        # Save examples
        examples_path = os.path.join(output_dir, 'examples.json')
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(self.api_docs['examples'], f, indent=2)
            
        print(f"Documentation saved to {output_dir}")
        
    def get_examples(self, api_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get code examples, optionally filtered by API name.
        
        Args:
            api_name: Name of the API to get examples for
            
        Returns:
            List of matching examples
        """
        if not self.api_docs:
            self.process_directory()
            
        if api_name is None:
            return self.api_docs['examples']
            
        # Filter examples that use the specified API
        matching_examples = []
        for example in self.api_docs['examples']:
            code = example['code']
            if api_name in code:
                matching_examples.append(example)
                
        return matching_examples 