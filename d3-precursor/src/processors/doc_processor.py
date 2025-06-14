import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import logging


class DocProcessor:
    """Process D3.js documentation."""
    
    def __init__(self):
        """Initialize the documentation processor."""
        # Logger for this module
        self.logger = logging.getLogger(__name__)
        
        # Patterns for API documentation
        self.api_header_pattern = r'^#+\s+([a-zA-Z0-9_$.]+)\s*$'
        self.parameter_pattern = r'^\s*(?:\*\s*)?(?:@param|Parameter:?)\s+(?:\{([^}]+)\}\s+)?([a-zA-Z0-9_$.]+)(?:\s+-\s+|\s*:\s*|\s+)(.+)$'
        self.return_pattern = r'^\s*(?:\*\s*)?(?:@returns?|Returns:?)\s+(?:\{([^}]+)\}\s+)?(.+)$'
        self.example_pattern = r'(?:Example|Usage):?\s*(?:\n```(?:js|javascript)?\n([\s\S]*?)\n```)?'
        
        # Pattern for code blocks
        self.code_block_pattern = r'```(?:js|javascript)?\n([\s\S]*?)\n```'
        
        # Pattern to identify API function signatures
        self.signature_pattern = r'([a-zA-Z0-9_$.]+)\(([^)]*)\)'
    
    def process_markdown(self, content: str) -> List[Dict[str, Any]]:
        """Process Markdown documentation content.
        
        Args:
            content: The Markdown content to process.
            
        Returns:
            A list of dictionaries with processed documentation.
        """
        api_docs = []
        lines = content.split('\n')
        current_doc = None
        
        for i, line in enumerate(lines):
            # Check for API headers
            header_match = re.match(self.api_header_pattern, line)
            if header_match:
                # If we were processing a doc, add it to the list
                if current_doc is not None:
                    api_docs.append(current_doc)
                
                # Start a new doc
                current_doc = {
                    'name': header_match.group(1),
                    'description': '',
                    'parameters': [],
                    'returns': None,
                    'examples': []
                }
                continue
            
            if current_doc is None:
                continue
            
            # Check for parameter documentation
            param_match = re.match(self.parameter_pattern, line)
            if param_match:
                param_type = param_match.group(1) if param_match.group(1) else None
                param_name = param_match.group(2)
                param_desc = param_match.group(3)
                
                current_doc['parameters'].append({
                    'name': param_name,
                    'type': param_type,
                    'description': param_desc
                })
                continue
            
            # Check for return documentation
            return_match = re.match(self.return_pattern, line)
            if return_match:
                return_type = return_match.group(1) if return_match.group(1) else None
                return_desc = return_match.group(2)
                
                current_doc['returns'] = {
                    'type': return_type,
                    'description': return_desc
                }
                continue
            
            # Check for example blocks
            if line.strip().startswith('Example') or line.strip().startswith('Usage'):
                # Find the code block that follows
                example_text = '\n'.join(lines[i:i+20])  # Look ahead a few lines
                example_match = re.search(self.code_block_pattern, example_text)
                if example_match:
                    current_doc['examples'].append({
                        'code': example_match.group(1),
                        'description': line.strip()
                    })
                continue
            
            # Add non-matched lines to description
            if line.strip() and not line.startswith('#') and not line.strip().startswith('---'):
                if current_doc['description']:
                    current_doc['description'] += '\n' + line
                else:
                    current_doc['description'] = line
        
        # Add the last doc if there is one
        if current_doc is not None:
            api_docs.append(current_doc)
        
        # Post-process the docs
        for doc in api_docs:
            # Clean up description
            doc['description'] = doc['description'].strip()
            
            # Extract signature from the name if it's a function
            signature_match = re.match(self.signature_pattern, doc['name'])
            if signature_match:
                func_name = signature_match.group(1)
                params_text = signature_match.group(2)
                
                # Update the name to just the function name
                doc['name'] = func_name
                
                # Add signature params if they're not already documented
                if params_text and not doc['parameters']:
                    params = [p.strip() for p in params_text.split(',') if p.strip()]
                    doc['parameters'] = [{'name': p, 'type': None, 'description': ''} for p in params]
        
        return api_docs
    
    def process_jsdoc(self, content: str) -> List[Dict[str, Any]]:
        """Process JSDoc documentation content.
        
        Args:
            content: The JSDoc content to process.
            
        Returns:
            A list of dictionaries with processed documentation.
        """
        api_docs = []
        
        # Extract JSDoc comment blocks
        jsdoc_pattern = r'/\*\*\s*([\s\S]*?)\s*\*/'
        jsdoc_blocks = re.findall(jsdoc_pattern, content)
        
        for block in jsdoc_blocks:
            # Split into lines
            lines = block.split('\n')
            
            # Clean up lines
            clean_lines = []
            for line in lines:
                # Remove leading asterisks and spaces
                clean_line = re.sub(r'^\s*\*\s?', '', line)
                clean_lines.append(clean_line)
            
            # Init doc
            doc = {
                'name': '',
                'description': '',
                'parameters': [],
                'returns': None,
                'examples': []
            }
            
            # Process lines
            description_lines = []
            for line in clean_lines:
                # Check for tags
                if line.startswith('@'):
                    # Parameter
                    param_match = re.match(r'@param\s+(?:\{([^}]+)\}\s+)?([a-zA-Z0-9_$.]+)(?:\s+-\s+|\s*\s*)(.+)?', line)
                    if param_match:
                        param_type = param_match.group(1) if param_match.group(1) else None
                        param_name = param_match.group(2)
                        param_desc = param_match.group(3) if param_match.group(3) else ''
                        
                        doc['parameters'].append({
                            'name': param_name,
                            'type': param_type,
                            'description': param_desc
                        })
                        continue
                    
                    # Return
                    return_match = re.match(r'@returns?\s+(?:\{([^}]+)\}\s+)?(.+)?', line)
                    if return_match:
                        return_type = return_match.group(1) if return_match.group(1) else None
                        return_desc = return_match.group(2) if return_match.group(2) else ''
                        
                        doc['returns'] = {
                            'type': return_type,
                            'description': return_desc
                        }
                        continue
                    
                    # Example
                    example_match = re.match(r'@example\s+(.+)?', line)
                    if example_match:
                        # Look for a code block in the following lines
                        example_desc = example_match.group(1) if example_match.group(1) else 'Example'
                        doc['examples'].append({
                            'code': '',  # Fill this in later
                            'description': example_desc
                        })
                        continue
                    
                    # Function name
                    name_match = re.match(r'@(func|function|method)\s+([a-zA-Z0-9_$.]+)', line)
                    if name_match:
                        doc['name'] = name_match.group(2)
                        continue
                else:
                    # If not a tag, add to description
                    if line.strip():
                        description_lines.append(line)
            
            # Set description
            doc['description'] = '\n'.join(description_lines).strip()
            
            # Try to extract function name from code that follows JSDoc
            if not doc['name']:
                # Find the code after this JSDoc block
                block_end = content.find('*/', content.find(block)) + 2
                next_line = content[block_end:].strip().split('\n')[0] if block_end < len(content) else ''
                
                # Check for function declaration
                func_match = re.match(r'(?:function\s+([a-zA-Z0-9_$.]+)|(?:const|let|var)?\s+([a-zA-Z0-9_$.]+)\s*=\s*function)', next_line)
                if func_match:
                    doc['name'] = func_match.group(1) if func_match.group(1) else func_match.group(2)
            
            # Add to list if we have a name
            if doc['name']:
                api_docs.append(doc)
        
        return api_docs
    
    def process_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Process a documentation file.
        
        Args:
            file_path: Path to the file to process.
            
        Returns:
            A list of dictionaries with processed documentation.
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            return []
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process based on file extension
        if file_path.suffix.lower() in ['.md', '.markdown']:
            return self.process_markdown(content)
        elif file_path.suffix.lower() in ['.js', '.ts', '.jsx', '.tsx']:
            return self.process_jsdoc(content)
        else:
            self.logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
    
    def extract_api_info(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract API information from processed documentation.
        
        Args:
            docs: The processed documentation.
            
        Returns:
            A list of dictionaries with API information.
        """
        api_info = []
        
        for doc in docs:
            # Basic API info
            api = {
                'name': doc['name'],
                'description': doc['description'],
                'type': self._infer_api_type(doc),
                'parameters': doc['parameters'],
                'returns': doc['returns'],
                'examples': doc['examples']
            }
            
            # Infer additional information
            api['tags'] = self._extract_tags(doc)
            api['related'] = self._extract_related(doc)
            
            api_info.append(api)
        
        return api_info
    
    def _infer_api_type(self, doc: Dict[str, Any]) -> str:
        """Infer the API element type from the documentation.
        
        Args:
            doc: The processed documentation.
            
        Returns:
            The inferred API element type.
        """
        # Check name for common patterns
        name = doc['name']
        
        if name.startswith('d3.'):
            # Check if it's a namespace
            if '.' not in name[3:]:
                return 'NAMESPACE'
        
        # Check if it has parameters
        if doc['parameters']:
            return 'FUNCTION'
        
        # Check for "new" in description
        if 'new ' in doc['description'] or 'constructor' in doc['description'].lower():
            return 'CLASS'
        
        # Default to function
        return 'FUNCTION'
    
    def _extract_tags(self, doc: Dict[str, Any]) -> List[str]:
        """Extract tags from the documentation.
        
        Args:
            doc: The processed documentation.
            
        Returns:
            A list of tags.
        """
        tags = []
        
        # Check description for common keywords
        description = doc['description'].lower()
        
        if 'scale' in description:
            tags.append('scale')
        if 'axis' in description:
            tags.append('axis')
        if 'select' in description:
            tags.append('selection')
        if 'transition' in description:
            tags.append('transition')
        if 'interpolate' in description:
            tags.append('interpolation')
        if 'color' in description:
            tags.append('color')
        if 'layout' in description:
            tags.append('layout')
        if 'array' in description:
            tags.append('array')
        if 'format' in description:
            tags.append('formatting')
        if 'geo' in description or 'map' in description:
            tags.append('geographic')
        if 'zoom' in description:
            tags.append('zoom')
        if 'drag' in description:
            tags.append('drag')
        if 'brush' in description:
            tags.append('brush')
        
        # Check if it's a core function
        if doc['name'].startswith('d3.'):
            tags.append('core')
        
        return tags
    
    def _extract_related(self, doc: Dict[str, Any]) -> List[str]:
        """Extract related API elements from the documentation.
        
        Args:
            doc: The processed documentation.
            
        Returns:
            A list of related API element names.
        """
        related = []
        
        # Check for references to other functions in description
        description = doc['description']
        
        # Find function-like references
        refs = re.findall(r'([a-zA-Z0-9_$.]+)\(\)', description)
        for ref in refs:
            if ref != doc['name'] and ref not in related:
                related.append(ref)
        
        # Find d3.* references
        d3_refs = re.findall(r'd3\.([a-zA-Z0-9_$.]+)', description)
        for ref in d3_refs:
            full_ref = f"d3.{ref}"
            if full_ref != doc['name'] and full_ref not in related:
                related.append(full_ref)
        
        return related 