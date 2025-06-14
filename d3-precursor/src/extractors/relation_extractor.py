"""
Relation extractor for D3 knowledge.

This module extracts relationships between D3 entities from parsed D3 code, documentation, and examples.
"""

import os
from typing import Dict, List, Any, Optional, Set, Tuple
import json
import re
from collections import defaultdict, Counter


class D3RelationExtractor:
    """Extract relationships between D3 entities from parsed code, documentation, and examples."""
    
    def __init__(self):
        """Initialize the relation extractor."""
        self.relations = {}
        self.api_relations = {}
        self.pattern_relations = {}
        self.concept_relations = {}
        self.usage_relations = {}
        
    def extract_from_js_parser(self, js_content: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships from JavaScript parser output.
        
        Args:
            js_content: JavaScript parser output
            
        Returns:
            Dictionary of extracted relationships
        """
        relations = {
            'function_to_function': [],
            'class_to_method': [],
            'module_to_function': [],
            'function_to_parameter': [],
            'inheritance': [],
            'usage': []
        }
        
        # Extract relationships from modules
        if 'modules' in js_content:
            for module_name, module_data in js_content['modules'].items():
                # Module to function relationships
                self._extract_module_function_relations(module_name, module_data, relations)
                
                # Function to function dependencies
                self._extract_function_relations(module_name, module_data, relations)
                
                # Class to method relationships
                self._extract_class_relations(module_name, module_data, relations)
                
                # Function to parameter relationships
                self._extract_parameter_relations(module_name, module_data, relations)
                
                # Extract inheritance relationships
                self._extract_inheritance_relations(module_name, module_data, relations)
                
                # Extract usage patterns
                self._extract_usage_relations(module_name, module_data, relations)
        
        # Store and return the relations
        self.api_relations = relations
        return relations
    
    def extract_from_patterns(self, patterns: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships from patterns.
        
        Args:
            patterns: Pattern data
            
        Returns:
            Dictionary of extracted relationships
        """
        relations = {
            'pattern_to_pattern': [],
            'pattern_to_function': [],
            'pattern_to_visualization': []
        }
        
        # Extract pattern to pattern relationships
        if 'method_chaining' in patterns:
            self._extract_pattern_cooccurrence(patterns['method_chaining'], 'method_chaining', relations)
        
        if 'selections' in patterns:
            self._extract_pattern_cooccurrence(patterns['selections'], 'selections', relations)
            
        if 'data_binding' in patterns:
            self._extract_pattern_cooccurrence(patterns['data_binding'], 'data_binding', relations)
            
        if 'scaling' in patterns:
            self._extract_pattern_cooccurrence(patterns['scaling'], 'scaling', relations)
            
        if 'layouts' in patterns:
            self._extract_pattern_cooccurrence(patterns['layouts'], 'layouts', relations)
        
        # Extract pattern to function relationships
        if 'method_chaining' in patterns and 'functions' in patterns:
            self._extract_pattern_function_relations(
                patterns['method_chaining'], 
                patterns['functions'],
                'method_chaining',
                relations
            )
        
        # Extract pattern to visualization relationships
        if 'patterns_by_visualization' in patterns:
            self._extract_pattern_visualization_relations(patterns['patterns_by_visualization'], relations)
        
        # Store and return the relations
        self.pattern_relations = relations
        return relations
    
    def extract_from_concepts(self, concepts: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships from concepts.
        
        Args:
            concepts: Concept data
            
        Returns:
            Dictionary of extracted relationships
        """
        # We'll just pass through any relationships that were already extracted by the concept extractor
        relations = {}
        
        if 'relationships' in concepts:
            relations = concepts['relationships']
        
        # Store and return the relations
        self.concept_relations = relations
        return relations
    
    def extract_from_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships from examples.
        
        Args:
            examples: Examples data
            
        Returns:
            Dictionary of extracted relationships
        """
        relations = {
            'function_usage': [],
            'pattern_usage': [],
            'visualization_techniques': []
        }
        
        for example in examples:
            example_id = example.get('id', '')
            title = example.get('title', '')
            code = example.get('code', '')
            
            if not code:
                continue
            
            # Extract function usage relationships
            function_calls = self._extract_function_calls(code)
            for func_name, count in function_calls.items():
                relations['function_usage'].append({
                    'example_id': example_id,
                    'title': title,
                    'function': func_name,
                    'count': count
                })
            
            # Extract pattern usage relationships
            patterns = self._extract_patterns_from_code(code)
            for pattern_type, pattern_instances in patterns.items():
                for pattern, count in pattern_instances.items():
                    relations['pattern_usage'].append({
                        'example_id': example_id,
                        'title': title,
                        'pattern_type': pattern_type,
                        'pattern': pattern,
                        'count': count
                    })
            
            # Extract visualization techniques
            vis_techniques = self._extract_visualization_techniques(code, title)
            for technique, confidence in vis_techniques:
                relations['visualization_techniques'].append({
                    'example_id': example_id,
                    'title': title,
                    'technique': technique,
                    'confidence': confidence
                })
        
        # Store and return the relations
        self.usage_relations = relations
        return relations
    
    def _extract_module_function_relations(self, module_name: str, module_data: Dict[str, Any], 
                                          relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract module to function relationships.
        
        Args:
            module_name: Name of the module
            module_data: Module data
            relations: Relations dictionary to update
        """
        if 'functions' in module_data:
            for func in module_data['functions']:
                func_name = func.get('name', '')
                if func_name:
                    relations['module_to_function'].append({
                        'module': module_name,
                        'function': func_name,
                        'exported': func.get('exported', True)
                    })
    
    def _extract_function_relations(self, module_name: str, module_data: Dict[str, Any], 
                                   relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract function to function relationships.
        
        Args:
            module_name: Name of the module
            module_data: Module data
            relations: Relations dictionary to update
        """
        if 'functions' in module_data:
            # Build a map of all functions in the module
            function_map = {func.get('name', ''): func for func in module_data['functions'] if func.get('name')}
            
            # For each function, check if it calls other functions
            for func_name, func_data in function_map.items():
                dependencies = func_data.get('dependencies', [])
                source = func_data.get('source', '')
                
                # Add explicit dependencies
                for dep in dependencies:
                    if dep in function_map:
                        relations['function_to_function'].append({
                            'source_function': func_name,
                            'target_function': dep,
                            'module': module_name,
                            'type': 'explicit_dependency'
                        })
                
                # Try to find implicit dependencies in the function source
                if source:
                    for other_func_name in function_map.keys():
                        if other_func_name != func_name and re.search(r'\b' + re.escape(other_func_name) + r'\s*\(', source):
                            relations['function_to_function'].append({
                                'source_function': func_name,
                                'target_function': other_func_name,
                                'module': module_name,
                                'type': 'implicit_call'
                            })
    
    def _extract_class_relations(self, module_name: str, module_data: Dict[str, Any], 
                               relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract class to method relationships.
        
        Args:
            module_name: Name of the module
            module_data: Module data
            relations: Relations dictionary to update
        """
        if 'classes' in module_data:
            for cls in module_data['classes']:
                class_name = cls.get('name', '')
                
                if not class_name:
                    continue
                
                # Class to method relationships
                methods = cls.get('methods', [])
                for method in methods:
                    method_name = method.get('name', '')
                    if method_name:
                        relations['class_to_method'].append({
                            'class': class_name,
                            'method': method_name,
                            'module': module_name,
                            'static': method.get('static', False),
                            'visibility': method.get('visibility', 'public')
                        })
    
    def _extract_parameter_relations(self, module_name: str, module_data: Dict[str, Any], 
                                    relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract function to parameter relationships.
        
        Args:
            module_name: Name of the module
            module_data: Module data
            relations: Relations dictionary to update
        """
        if 'functions' in module_data:
            for func in module_data['functions']:
                func_name = func.get('name', '')
                parameters = func.get('parameters', [])
                
                if not func_name:
                    continue
                
                for param in parameters:
                    param_name = param.get('name', '')
                    if param_name:
                        relations['function_to_parameter'].append({
                            'function': func_name,
                            'parameter': param_name,
                            'module': module_name,
                            'type': param.get('type', 'any'),
                            'optional': param.get('optional', False),
                            'default_value': param.get('default', None)
                        })
    
    def _extract_inheritance_relations(self, module_name: str, module_data: Dict[str, Any], 
                                      relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract inheritance relationships.
        
        Args:
            module_name: Name of the module
            module_data: Module data
            relations: Relations dictionary to update
        """
        if 'classes' in module_data:
            for cls in module_data['classes']:
                class_name = cls.get('name', '')
                extends = cls.get('extends', [])
                
                if not class_name:
                    continue
                
                for parent_class in extends:
                    relations['inheritance'].append({
                        'child_class': class_name,
                        'parent_class': parent_class,
                        'module': module_name
                    })
    
    def _extract_usage_relations(self, module_name: str, module_data: Dict[str, Any], 
                                relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract usage relationships.
        
        Args:
            module_name: Name of the module
            module_data: Module data
            relations: Relations dictionary to update
        """
        # Build a map of all functions and classes in the module
        entities = {}
        
        if 'functions' in module_data:
            for func in module_data['functions']:
                func_name = func.get('name', '')
                if func_name:
                    entities[func_name] = {
                        'type': 'function',
                        'source': func.get('source', '')
                    }
        
        if 'classes' in module_data:
            for cls in module_data['classes']:
                class_name = cls.get('name', '')
                if class_name:
                    entities[class_name] = {
                        'type': 'class',
                        'source': cls.get('source', '')
                    }
                    
                    # Add class methods
                    methods = cls.get('methods', [])
                    for method in methods:
                        method_name = method.get('name', '')
                        if method_name:
                            full_method_name = f"{class_name}.{method_name}"
                            entities[full_method_name] = {
                                'type': 'method',
                                'source': method.get('source', '')
                            }
        
        # For each entity, check usage of other entities
        for entity_name, entity_data in entities.items():
            source = entity_data.get('source', '')
            
            if not source:
                continue
            
            for other_entity, other_data in entities.items():
                if entity_name != other_entity and re.search(r'\b' + re.escape(other_entity) + r'\b', source):
                    relations['usage'].append({
                        'source_entity': entity_name,
                        'source_type': entity_data['type'],
                        'target_entity': other_entity,
                        'target_type': other_data['type'],
                        'module': module_name
                    })
    
    def _extract_pattern_cooccurrence(self, patterns: List[Dict[str, Any]], pattern_type: str, 
                                     relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract pattern co-occurrence relationships.
        
        Args:
            patterns: List of patterns
            pattern_type: Type of pattern
            relations: Relations dictionary to update
        """
        # Create a map from examples to patterns
        example_patterns = defaultdict(list)
        
        for pattern in patterns:
            pattern_key = pattern.get('pattern', '')
            examples = pattern.get('examples', [])
            
            if not pattern_key or not examples:
                continue
            
            for example in examples:
                example_patterns[example].append(pattern_key)
        
        # Find patterns that co-occur in the same examples
        for example, patterns_list in example_patterns.items():
            if len(patterns_list) <= 1:
                continue
            
            for i, pattern1 in enumerate(patterns_list):
                for pattern2 in patterns_list[i+1:]:
                    # Check if this relationship already exists
                    existing = False
                    for rel in relations['pattern_to_pattern']:
                        if (rel['pattern1'] == pattern1 and rel['pattern2'] == pattern2 and 
                            rel['pattern_type'] == pattern_type):
                            rel['count'] += 1
                            existing = True
                            break
                        elif (rel['pattern1'] == pattern2 and rel['pattern2'] == pattern1 and 
                              rel['pattern_type'] == pattern_type):
                            rel['count'] += 1
                            existing = True
                            break
                    
                    if not existing:
                        relations['pattern_to_pattern'].append({
                            'pattern1': pattern1,
                            'pattern2': pattern2,
                            'pattern_type': pattern_type,
                            'count': 1
                        })
    
    def _extract_pattern_function_relations(self, patterns: List[Dict[str, Any]], functions: List[Dict[str, Any]], 
                                           pattern_type: str, relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract pattern to function relationships.
        
        Args:
            patterns: List of patterns
            functions: List of functions
            pattern_type: Type of pattern
            relations: Relations dictionary to update
        """
        # Create a map of function names
        function_names = {func.get('name', ''): func for func in functions if func.get('name', '')}
        
        for pattern in patterns:
            pattern_key = pattern.get('pattern', '')
            pattern_code = pattern.get('code', '')
            
            if not pattern_key or not pattern_code:
                continue
            
            # Find functions used in the pattern
            for func_name in function_names.keys():
                if re.search(r'\b' + re.escape(func_name) + r'\s*\(', pattern_code):
                    relations['pattern_to_function'].append({
                        'pattern': pattern_key,
                        'pattern_type': pattern_type,
                        'function': func_name
                    })
    
    def _extract_pattern_visualization_relations(self, patterns_by_vis: Dict[str, List[Dict[str, Any]]],
                                               relations: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Extract pattern to visualization relationships.
        
        Args:
            patterns_by_vis: Patterns organized by visualization type
            relations: Relations dictionary to update
        """
        for vis_type, patterns in patterns_by_vis.items():
            for pattern in patterns:
                pattern_key = pattern.get('pattern', '')
                pattern_type = pattern.get('type', '')
                
                if not pattern_key or not pattern_type:
                    continue
                
                relations['pattern_to_visualization'].append({
                    'pattern': pattern_key,
                    'pattern_type': pattern_type,
                    'visualization': vis_type,
                    'importance': pattern.get('importance', 'medium')
                })
    
    def _extract_function_calls(self, code: str) -> Dict[str, int]:
        """
        Extract D3 function calls from code.
        
        Args:
            code: D3 code
            
        Returns:
            Dictionary mapping function names to call counts
        """
        # Find all function calls in the format d3.xxx() or d3.xxx.yyy()
        function_calls = re.findall(r'd3\.([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\s*\(', code)
        
        # Count occurrences
        function_counts = Counter(function_calls)
        
        return function_counts
    
    def _extract_patterns_from_code(self, code: str) -> Dict[str, Dict[str, int]]:
        """
        Extract patterns from code.
        
        Args:
            code: D3 code
            
        Returns:
            Dictionary mapping pattern types to pattern counts
        """
        patterns = {
            'method_chaining': Counter(),
            'selections': Counter(),
            'data_binding': Counter(),
            'scaling': Counter(),
            'layouts': Counter()
        }
        
        # Method chaining patterns - chains of at least 3 methods
        method_chains = re.findall(r'(\w+(?:\.\w+){2,})', code)
        for chain in method_chains:
            if chain.startswith('d3'):
                patterns['method_chaining'][chain] += 1
        
        # Selection patterns - d3.select or d3.selectAll
        selection_patterns = re.findall(r'd3\.select(?:All)?\([^)]+\)(?:\.\w+\([^)]*\))+', code)
        for pattern in selection_patterns:
            patterns['selections'][pattern] += 1
        
        # Data binding patterns - .data()
        data_binding_patterns = re.findall(r'\.data\([^)]+\)(?:\.\w+\([^)]*\))*', code)
        for pattern in data_binding_patterns:
            patterns['data_binding'][pattern] += 1
        
        # Scaling patterns - d3.scaleXxx
        scaling_patterns = re.findall(r'd3\.scale\w+\([^)]*\)(?:\.\w+\([^)]*\))*', code)
        for pattern in scaling_patterns:
            patterns['scaling'][pattern] += 1
        
        # Layout patterns - d3.xxxLayout
        layout_patterns = re.findall(r'd3\.\w+Layout\([^)]*\)(?:\.\w+\([^)]*\))*', code)
        for pattern in layout_patterns:
            patterns['layouts'][pattern] += 1
        
        return patterns
    
    def _extract_visualization_techniques(self, code: str, title: str) -> List[Tuple[str, float]]:
        """
        Extract visualization techniques from code and title.
        
        Args:
            code: D3 code
            title: Example title
            
        Returns:
            List of tuples (technique, confidence)
        """
        techniques = []
        
        # Dictionary mapping keywords to visualization techniques
        vis_keywords = {
            'bar chart': ['bar', 'histogram', 'column'],
            'line chart': ['line', 'path', 'curve', 'trend'],
            'pie chart': ['pie', 'donut', 'circle', 'arc'],
            'scatter plot': ['scatter', 'bubble', 'point'],
            'map': ['map', 'geo', 'projection', 'topology'],
            'network': ['force', 'node', 'link', 'graph', 'network'],
            'tree': ['tree', 'hierarchy', 'cluster', 'dendrogram'],
            'heatmap': ['heatmap', 'grid', 'color scale'],
            'sankey': ['sankey', 'flow'],
            'chord': ['chord', 'matrix'],
            'treemap': ['treemap', 'partition']
        }
        
        # Check title for keywords
        title_lower = title.lower()
        for technique, keywords in vis_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    techniques.append((technique, 0.9))  # High confidence if in title
                    break
        
        # Check code for visualization-specific functions
        vis_functions = {
            'bar chart': [r'd3\.scaleBand', r'\.attr\([\'"]width', r'\.attr\([\'"]x'],
            'line chart': [r'd3\.line', r'path', r'\.curve'],
            'pie chart': [r'd3\.pie', r'\.arc', r'\.innerRadius'],
            'scatter plot': [r'\.attr\([\'"]cx', r'\.attr\([\'"]cy', r'\.attr\([\'"]r'],
            'map': [r'd3\.geo', r'projection', r'path'],
            'network': [r'd3\.force', r'\.links', r'\.nodes'],
            'tree': [r'd3\.tree', r'd3\.hierarchy', r'\.descendants'],
            'heatmap': [r'\.attr\([\'"]fill', r'colorScale'],
            'sankey': [r'd3\.sankey', r'\.nodeWidth'],
            'chord': [r'd3\.chord', r'\.ribbon'],
            'treemap': [r'd3\.treemap', r'\.tile']
        }
        
        # Check for each visualization type
        for technique, patterns in vis_functions.items():
            confidence = 0.0
            for pattern in patterns:
                if re.search(pattern, code):
                    confidence += 0.3  # Increase confidence for each matching pattern
                    
            if confidence > 0:
                # Limit confidence to 0.8 for code-based detection
                techniques.append((technique, min(0.8, confidence)))
        
        # Remove duplicates (keeping highest confidence)
        unique_techniques = {}
        for technique, confidence in techniques:
            if technique not in unique_techniques or confidence > unique_techniques[technique]:
                unique_techniques[technique] = confidence
        
        return [(t, c) for t, c in unique_techniques.items()]
    
    def merge_all_relations(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Merge all relations into a comprehensive view.
        
        Returns:
            Dictionary of all relations
        """
        all_relations = {
            'api': self.api_relations,
            'patterns': self.pattern_relations,
            'concepts': self.concept_relations,
            'usage': self.usage_relations
        }
        
        return all_relations
    
    def save_relations(self, output_path: str) -> None:
        """
        Save relations to file.
        
        Args:
            output_path: Path to save the relations
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Merge all relations
        all_relations = self.merge_all_relations()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_relations, f, indent=2)
            
        print(f"Relations saved to {output_path}")
    
    def get_relations(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get the extracted relations.
        
        Returns:
            Dictionary of all relations
        """
        return self.merge_all_relations() 