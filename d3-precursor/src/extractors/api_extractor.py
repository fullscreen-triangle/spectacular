"""
API extractor for D3 knowledge.

This module extracts API elements from parsed D3 code and documentation.
"""

import os
from typing import Dict, List, Any, Optional, Set, Tuple
import json


class D3APIExtractor:
    """Extract API elements from parsed D3 code and documentation."""
    
    def __init__(self):
        """Initialize the API extractor."""
        self.api_elements = {}
        self.js_api_elements = {}
        self.py_api_elements = {}
        self.doc_api_elements = {}
    
    def extract_from_js_parser(self, js_api_defs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract API elements from JavaScript parser output.
        
        Args:
            js_api_defs: API definitions from D3JSParser
            
        Returns:
            Extracted and normalized API elements
        """
        api_elements = {}
        
        for file_path, definitions in js_api_defs.items():
            for definition in definitions:
                # Extract necessary information
                api_type = definition.get('type', 'unknown')
                name = definition.get('name', 'unknown')
                
                # Generate a unique ID
                api_id = f"js_{api_type}_{name}".replace('.', '_')
                
                # Normalize and structure the API element
                element = {
                    'id': api_id,
                    'name': name,
                    'type': api_type,
                    'language': 'javascript',
                    'source': file_path,
                    'params': definition.get('params', []),
                    'is_export': api_type.startswith('export'),
                    'description': '',  # Will be filled from docs later
                    'examples': []  # Will be filled from examples later
                }
                
                api_elements[api_id] = element
        
        self.js_api_elements = api_elements
        return api_elements
    
    def extract_from_py_parser(self, py_api_defs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract API elements from Python parser output.
        
        Args:
            py_api_defs: API definitions from D3PyParser
            
        Returns:
            Extracted and normalized API elements
        """
        api_elements = {}
        
        for file_path, definitions in py_api_defs.items():
            for definition in definitions:
                # Extract necessary information
                api_type = definition.get('type', 'unknown')
                name = definition.get('name', 'unknown')
                docstring = definition.get('docstring', '')
                
                # Generate a unique ID
                api_id = f"py_{api_type}_{name}".replace('.', '_')
                
                # Normalize and structure the API element
                element = {
                    'id': api_id,
                    'name': name,
                    'type': api_type,
                    'language': 'python',
                    'source': file_path,
                    'params': definition.get('params', []),
                    'class': definition.get('class', None),
                    'description': docstring,
                    'examples': []  # Will be filled from examples later
                }
                
                api_elements[api_id] = element
        
        self.py_api_elements = api_elements
        return api_elements
    
    def extract_from_doc_parser(self, doc_api_elements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract API elements from documentation parser output.
        
        Args:
            doc_api_elements: API elements from D3DocParser
            
        Returns:
            Extracted and normalized API elements
        """
        api_elements = {}
        
        for api_name, api_doc in doc_api_elements.items():
            # Generate a unique ID
            api_id = f"doc_{api_doc.get('type', 'unknown')}_{api_name}".replace('.', '_')
            
            # Normalize and structure the API element
            element = {
                'id': api_id,
                'name': api_name,
                'full_name': api_doc.get('full_name', api_name),
                'type': api_doc.get('type', 'unknown'),
                'language': 'javascript',  # Documentation is primarily for JS
                'signature': api_doc.get('signature', ''),
                'description': self._extract_description(api_doc),
                'params': self._normalize_params(api_doc.get('parameters', [])),
                'returns': api_doc.get('returns', {}).get('description', ''),
                'examples': []  # Will be filled from examples later
            }
            
            api_elements[api_id] = element
        
        self.doc_api_elements = api_elements
        return api_elements
    
    def _extract_description(self, api_doc: Dict[str, Any]) -> str:
        """Extract description from API documentation."""
        description = api_doc.get('description', '')
        
        # If no description, try to get it from surrounding text
        if not description and 'surrounding_text' in api_doc:
            description = api_doc['surrounding_text']
            
        return description
    
    def _normalize_params(self, params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize parameter information."""
        normalized = []
        
        for param in params:
            normalized.append({
                'name': param.get('name', 'unknown'),
                'description': param.get('description', ''),
                'default': param.get('default', None),
                'type': param.get('type', None)
            })
            
        return normalized
    
    def merge_api_elements(self) -> Dict[str, Any]:
        """
        Merge API elements from different sources.
        
        Returns:
            Merged API elements with enriched information
        """
        merged = {}
        
        # First, add all JavaScript API elements
        for api_id, element in self.js_api_elements.items():
            merged[api_id] = element
        
        # Add Python API elements
        for api_id, element in self.py_api_elements.items():
            merged[api_id] = element
        
        # Enrich with documentation information
        for api_id, element in self.doc_api_elements.items():
            name = element['name']
            
            # Try to find matching element in JS or Python API
            matching_js = self._find_matching_element(name, self.js_api_elements)
            matching_py = self._find_matching_element(name, self.py_api_elements)
            
            if matching_js:
                # Enrich JavaScript API with documentation
                js_id = matching_js['id']
                merged[js_id]['description'] = element['description']
                merged[js_id]['params'] = element['params']
                merged[js_id]['returns'] = element['returns']
            elif matching_py:
                # Enrich Python API with documentation
                py_id = matching_py['id']
                if not merged[py_id]['description']:
                    merged[py_id]['description'] = element['description']
                merged[py_id]['params'] = element['params']
            else:
                # No matching API found, add the documentation as a new element
                merged[api_id] = element
        
        self.api_elements = merged
        return merged
    
    def _find_matching_element(self, name: str, elements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a matching API element by name."""
        # Try exact match
        for element in elements.values():
            if element['name'] == name:
                return element
            
        # Try matching by removing namespaces
        base_name = name.split('.')[-1]
        for element in elements.values():
            element_name = element['name'].split('.')[-1]
            if element_name == base_name:
                return element
                
        return None
    
    def add_examples_to_api(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add examples to API elements.
        
        Args:
            examples: List of code examples
            
        Returns:
            Updated API elements with examples
        """
        for example in examples:
            code = example.get('code', '')
            language = example.get('language', 'javascript')
            description = example.get('description', '')
            
            # Find API elements used in this example
            for api_id, element in self.api_elements.items():
                name = element['name']
                
                # Check if the API is used in the example
                if name in code:
                    # Add example to the API element
                    if 'examples' not in element:
                        element['examples'] = []
                        
                    element['examples'].append({
                        'code': code,
                        'language': language,
                        'description': description
                    })
        
        return self.api_elements
    
    def save_api_elements(self, output_path: str) -> None:
        """
        Save API elements to file.
        
        Args:
            output_path: Path to save the API elements
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.api_elements, f, indent=2)
            
        print(f"API elements saved to {output_path}")
    
    def get_api_elements(self) -> Dict[str, Any]:
        """
        Get the extracted API elements.
        
        Returns:
            Dictionary of API elements
        """
        return self.api_elements 