"""
Concept extractor for D3 knowledge.

This module extracts D3 concepts from parsed D3 code, documentation, and examples.
"""

import os
from typing import Dict, List, Any, Optional, Set
import json
from collections import Counter, defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class D3ConceptExtractor:
    """Extract D3 concepts from parsed code, documentation, and examples."""
    
    def __init__(self):
        """Initialize the concept extractor."""
        # Initialize NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.d3_concepts = {}
        self.concepts_by_module = {}
        self.core_concepts = set()
        self.domain_concepts = set()
        
    def extract_from_docs(self, doc_content: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from documentation parser output.
        
        Args:
            doc_content: Documentation parser output
            
        Returns:
            Dictionary of extracted concepts
        """
        # Initialize concepts dictionary
        concepts = {
            'core_concepts': [],
            'domain_concepts': []
        }
        
        # Extract from API documentation
        if 'apis' in doc_content:
            for api in doc_content['apis']:
                # Extract concepts from API name
                name = api.get('name', '')
                if name:
                    self._process_api_name(name, concepts)
                
                # Extract concepts from description
                description = api.get('description', '')
                if description:
                    self._process_description(description, 'core', concepts)
                
                # Extract concepts from parameters
                parameters = api.get('parameters', [])
                for param in parameters:
                    param_desc = param.get('description', '')
                    if param_desc:
                        self._process_description(param_desc, 'domain', concepts)
        
        # Extract from tutorials and guides
        if 'guides' in doc_content:
            for guide in doc_content['guides']:
                title = guide.get('title', '')
                content = guide.get('content', '')
                
                if title:
                    self._process_title(title, concepts)
                
                if content:
                    self._process_description(content, 'both', concepts)
        
        # Process extracted concepts
        self._prune_and_categorize_concepts(concepts)
        
        # Update class instance variables
        self.d3_concepts = concepts
        
        # Return the concepts dictionary
        return concepts
    
    def extract_from_js_parser(self, js_content: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from JavaScript parser output.
        
        Args:
            js_content: JavaScript parser output
            
        Returns:
            Dictionary of extracted concepts
        """
        # Initialize concepts dictionary
        concepts = {
            'core_concepts': [],
            'domain_concepts': []
        }
        
        # Extract from modules
        if 'modules' in js_content:
            for module_name, module_data in js_content['modules'].items():
                module_concepts = self._extract_module_concepts(module_name, module_data)
                
                # Add to concepts by module
                self.concepts_by_module[module_name] = module_concepts
                
                # Add to overall concepts
                for concept in module_concepts.get('core_concepts', []):
                    if concept not in [c['concept'] for c in concepts['core_concepts']]:
                        concepts['core_concepts'].append({
                            'concept': concept,
                            'source': module_name,
                            'weight': module_concepts.get('weights', {}).get(concept, 1)
                        })
                
                for concept in module_concepts.get('domain_concepts', []):
                    if concept not in [c['concept'] for c in concepts['domain_concepts']]:
                        concepts['domain_concepts'].append({
                            'concept': concept,
                            'source': module_name,
                            'weight': module_concepts.get('weights', {}).get(concept, 1)
                        })
        
        # Extract from API
        if 'api' in js_content:
            for api in js_content['api']:
                name = api.get('name', '')
                description = api.get('description', '')
                
                if name:
                    self._process_api_name(name, concepts)
                
                if description:
                    self._process_description(description, 'core', concepts)
        
        # Process extracted concepts
        self._prune_and_categorize_concepts(concepts)
        
        # Update class instance variables
        self.d3_concepts = concepts
        
        # Return the concepts dictionary
        return concepts
    
    def extract_from_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract concepts from examples parser output.
        
        Args:
            examples: Examples parser output
            
        Returns:
            Dictionary of extracted concepts
        """
        # Initialize concepts dictionary
        concepts = {
            'core_concepts': [],
            'domain_concepts': []
        }
        
        for example in examples:
            title = example.get('title', '')
            description = example.get('description', '')
            code = example.get('code', '')
            
            if title:
                self._process_title(title, concepts)
            
            if description:
                self._process_description(description, 'both', concepts)
            
            if code:
                self._process_code(code, concepts)
        
        # Process extracted concepts
        self._prune_and_categorize_concepts(concepts)
        
        # Update class instance variables
        self.d3_concepts = concepts
        
        # Return the concepts dictionary
        return concepts
    
    def _extract_module_concepts(self, module_name: str, module_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract concepts from a D3 module.
        
        Args:
            module_name: Name of the module
            module_data: Module data from parser
            
        Returns:
            Dictionary of concepts for the module
        """
        module_concepts = {
            'core_concepts': [],
            'domain_concepts': [],
            'weights': {}
        }
        
        # Process module name
        name_parts = module_name.split('-')
        if len(name_parts) > 1:
            # For modules like d3-scale, d3-array, etc.
            for part in name_parts[1:]:
                if part not in module_concepts['core_concepts'] and part not in self.stop_words:
                    module_concepts['core_concepts'].append(part)
                    module_concepts['weights'][part] = 5  # Higher weight for module name
        
        # Process functions
        if 'functions' in module_data:
            for func in module_data['functions']:
                name = func.get('name', '')
                description = func.get('description', '')
                
                if name:
                    # Process function name for concepts
                    name_concepts = self._extract_concepts_from_name(name)
                    for concept in name_concepts:
                        if concept not in module_concepts['core_concepts']:
                            module_concepts['core_concepts'].append(concept)
                            module_concepts['weights'][concept] = 3  # Medium-high weight for function names
                
                if description:
                    # Process description
                    desc_concepts = self._extract_concepts_from_text(description)
                    for concept, is_core in desc_concepts:
                        if is_core:
                            if concept not in module_concepts['core_concepts']:
                                module_concepts['core_concepts'].append(concept)
                                module_concepts['weights'][concept] = 2  # Medium weight for descriptions
                        else:
                            if concept not in module_concepts['domain_concepts']:
                                module_concepts['domain_concepts'].append(concept)
                                module_concepts['weights'][concept] = 1  # Lower weight for domain concepts
        
        # Process classes
        if 'classes' in module_data:
            for cls in module_data['classes']:
                name = cls.get('name', '')
                description = cls.get('description', '')
                
                if name:
                    # Process class name for concepts
                    name_concepts = self._extract_concepts_from_name(name)
                    for concept in name_concepts:
                        if concept not in module_concepts['core_concepts']:
                            module_concepts['core_concepts'].append(concept)
                            module_concepts['weights'][concept] = 4  # Higher weight for class names
                
                if description:
                    # Process description
                    desc_concepts = self._extract_concepts_from_text(description)
                    for concept, is_core in desc_concepts:
                        if is_core:
                            if concept not in module_concepts['core_concepts']:
                                module_concepts['core_concepts'].append(concept)
                                module_concepts['weights'][concept] = 2  # Medium weight for descriptions
                        else:
                            if concept not in module_concepts['domain_concepts']:
                                module_concepts['domain_concepts'].append(concept)
                                module_concepts['weights'][concept] = 1  # Lower weight for domain concepts
        
        return module_concepts
    
    def _process_api_name(self, name: str, concepts: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Process an API name for concepts.
        
        Args:
            name: API name
            concepts: Dictionary to update with extracted concepts
        """
        name_concepts = self._extract_concepts_from_name(name)
        for concept in name_concepts:
            if concept not in [c['concept'] for c in concepts['core_concepts']]:
                concepts['core_concepts'].append({
                    'concept': concept,
                    'source': 'api_name',
                    'weight': 4  # Higher weight for API names
                })
    
    def _process_description(self, description: str, type_hint: str, concepts: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Process a description for concepts.
        
        Args:
            description: Text description
            type_hint: Hint about the type of concepts to extract (core, domain, or both)
            concepts: Dictionary to update with extracted concepts
        """
        desc_concepts = self._extract_concepts_from_text(description)
        
        for concept, is_core in desc_concepts:
            if (type_hint == 'core' or type_hint == 'both') and is_core:
                if concept not in [c['concept'] for c in concepts['core_concepts']]:
                    concepts['core_concepts'].append({
                        'concept': concept,
                        'source': 'description',
                        'weight': 2  # Medium weight for descriptions
                    })
            
            if (type_hint == 'domain' or type_hint == 'both') and not is_core:
                if concept not in [c['concept'] for c in concepts['domain_concepts']]:
                    concepts['domain_concepts'].append({
                        'concept': concept,
                        'source': 'description',
                        'weight': 1  # Lower weight for domain concepts
                    })
    
    def _process_title(self, title: str, concepts: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Process a title for concepts.
        
        Args:
            title: Title text
            concepts: Dictionary to update with extracted concepts
        """
        title_concepts = self._extract_concepts_from_text(title)
        
        for concept, is_core in title_concepts:
            if is_core:
                if concept not in [c['concept'] for c in concepts['core_concepts']]:
                    concepts['core_concepts'].append({
                        'concept': concept,
                        'source': 'title',
                        'weight': 3  # Medium-high weight for titles
                    })
            else:
                if concept not in [c['concept'] for c in concepts['domain_concepts']]:
                    concepts['domain_concepts'].append({
                        'concept': concept,
                        'source': 'title',
                        'weight': 2  # Medium weight for domain concepts in titles
                    })
    
    def _process_code(self, code: str, concepts: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Process code for concepts.
        
        Args:
            code: D3 code
            concepts: Dictionary to update with extracted concepts
        """
        # Extract D3 method calls
        d3_calls = re.findall(r'd3\.(\w+)', code)
        
        for call in d3_calls:
            # Split camelCase into separate words
            words = re.findall(r'[A-Z]?[a-z]+', call)
            
            for word in words:
                word = word.lower()
                if word not in self.stop_words and len(word) > 2:
                    if word not in [c['concept'] for c in concepts['core_concepts']]:
                        concepts['core_concepts'].append({
                            'concept': word,
                            'source': 'code',
                            'weight': 3  # Medium-high weight for code
                        })
    
    def _extract_concepts_from_name(self, name: str) -> List[str]:
        """
        Extract concepts from a name (function, class, API).
        
        Args:
            name: The name to process
            
        Returns:
            List of extracted concepts
        """
        # Remove d3. prefix if present
        name = name.replace('d3.', '')
        
        # Split camelCase into separate words
        words = re.findall(r'[A-Z]?[a-z]+', name)
        
        concepts = []
        for word in words:
            word = word.lower()
            if word not in self.stop_words and len(word) > 2:
                concepts.append(word)
        
        return concepts
    
    def _extract_concepts_from_text(self, text: str) -> List[Tuple[str, bool]]:
        """
        Extract concepts from text with indication of core vs domain concepts.
        
        Args:
            text: The text to process
            
        Returns:
            List of tuples (concept, is_core_concept)
        """
        # D3-specific terms to always treat as core concepts
        d3_core_terms = {
            'selection', 'scale', 'axis', 'transition', 'svg', 'data', 'join', 
            'enter', 'exit', 'update', 'append', 'attr', 'style', 'domain', 
            'range', 'transform', 'zoom', 'drag', 'brush', 'format', 'color',
            'path', 'line', 'arc', 'area', 'symbol', 'histogram', 'contour',
            'tree', 'cluster', 'pack', 'partition', 'treemap', 'force',
            'simulation', 'node', 'link', 'chord', 'pie', 'stack', 'shape',
            'interpolate', 'ease', 'timer', 'interval', 'timeout', 'time',
            'projection', 'geo', 'map', 'voronoi', 'quadtree', 'polygon',
            'hull', 'centroid', 'hierarchy', 'nest', 'array', 'collection'
        }
        
        # Tokenize and preprocess text
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        
        # Extract potential concepts
        concepts = []
        for token in tokens:
            if len(token) > 2:  # Filter very short words
                # Check if it's a core D3 concept
                is_core = token in d3_core_terms
                concepts.append((token, is_core))
        
        return concepts
    
    def _prune_and_categorize_concepts(self, concepts: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Prune low-quality concepts and categorize them better.
        
        Args:
            concepts: Dictionary of concepts to prune and categorize
        """
        # Filter out very common words or irrelevant terms
        common_irrelevant = {'function', 'method', 'object', 'value', 'return', 'parameter', 'example'}
        
        # Prune core concepts
        concepts['core_concepts'] = [
            concept for concept in concepts['core_concepts']
            if concept['concept'] not in common_irrelevant
        ]
        
        # Prune domain concepts
        concepts['domain_concepts'] = [
            concept for concept in concepts['domain_concepts']
            if concept['concept'] not in common_irrelevant
        ]
        
        # Update class instance variables
        self.core_concepts = set(c['concept'] for c in concepts['core_concepts'])
        self.domain_concepts = set(c['concept'] for c in concepts['domain_concepts'])
    
    def extract_concept_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships between concepts.
        
        Returns:
            Dictionary of concept relationships
        """
        relationships = {
            'core_to_core': [],
            'core_to_domain': []
        }
        
        # Build relationships between core concepts from co-occurrence in documentation
        core_concepts = list(self.core_concepts)
        
        for i, concept1 in enumerate(core_concepts):
            for concept2 in core_concepts[i+1:]:
                # Check co-occurrence in modules
                co_occurred = False
                strength = 0
                
                for module_name, module_concepts in self.concepts_by_module.items():
                    if (concept1 in module_concepts.get('core_concepts', []) and 
                        concept2 in module_concepts.get('core_concepts', [])):
                        co_occurred = True
                        strength += 1
                
                if co_occurred:
                    relationships['core_to_core'].append({
                        'source': concept1,
                        'target': concept2,
                        'strength': strength,
                        'type': 'co-occurrence'
                    })
        
        # Build relationships between core and domain concepts
        for core_concept in self.core_concepts:
            for domain_concept in self.domain_concepts:
                # For now, just create a relationship if they appear in the same module
                for module_name, module_concepts in self.concepts_by_module.items():
                    if (core_concept in module_concepts.get('core_concepts', []) and 
                        domain_concept in module_concepts.get('domain_concepts', [])):
                        relationships['core_to_domain'].append({
                            'source': core_concept,
                            'target': domain_concept,
                            'module': module_name,
                            'type': 'association'
                        })
                        break
        
        return relationships
    
    def get_top_concepts(self, category: str = 'core', limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the top concepts by weight.
        
        Args:
            category: 'core' or 'domain'
            limit: Maximum number of concepts to return
            
        Returns:
            List of top concepts with weights
        """
        if category == 'core':
            concepts = self.d3_concepts.get('core_concepts', [])
        else:
            concepts = self.d3_concepts.get('domain_concepts', [])
        
        # Sort by weight descending
        sorted_concepts = sorted(concepts, key=lambda x: x.get('weight', 0), reverse=True)
        
        return sorted_concepts[:limit]
    
    def save_concepts(self, output_path: str) -> None:
        """
        Save concepts to file.
        
        Args:
            output_path: Path to save the concepts
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create the full concepts dictionary
        full_concepts = {
            'core_concepts': self.d3_concepts.get('core_concepts', []),
            'domain_concepts': self.d3_concepts.get('domain_concepts', []),
            'concepts_by_module': self.concepts_by_module,
            'relationships': self.extract_concept_relationships()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_concepts, f, indent=2)
            
        print(f"Concepts saved to {output_path}")
    
    def get_concepts(self) -> Dict[str, Any]:
        """
        Get the extracted concepts.
        
        Returns:
            Dictionary of concepts and relationships
        """
        return {
            'core_concepts': self.d3_concepts.get('core_concepts', []),
            'domain_concepts': self.d3_concepts.get('domain_concepts', []),
            'relationships': self.extract_concept_relationships()
        } 