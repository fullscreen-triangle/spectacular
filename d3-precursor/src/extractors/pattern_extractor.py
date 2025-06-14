"""
Pattern extractor for D3 knowledge.

This module extracts common D3 usage patterns from parsed D3 code and examples.
"""

import os
from typing import Dict, List, Any, Optional, Set, Tuple
import json
import re


class D3PatternExtractor:
    """Extract D3 usage patterns from parsed code and examples."""
    
    def __init__(self):
        """Initialize the pattern extractor."""
        self.patterns = {}
        self.method_chaining_patterns = []
        self.selection_patterns = []
        self.data_binding_patterns = []
        self.scaling_patterns = []
        self.layout_patterns = []
        
    def extract_from_js_parser(self, usage_patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract patterns from JavaScript parser output.
        
        Args:
            usage_patterns: Usage patterns from D3JSParser
            
        Returns:
            Categorized and normalized patterns
        """
        patterns = {
            'method_chaining': [],
            'selections': [],
            'data_binding': [],
            'scales': [],
            'layouts': []
        }
        
        # Process method chaining patterns
        for pattern in usage_patterns.get('method_chaining', []):
            if 'code' in pattern:
                normalized = self._normalize_method_chain(pattern['code'])
                patterns['method_chaining'].append({
                    'pattern': normalized,
                    'code': pattern['code'],
                    'source': pattern.get('source', '')
                })
        
        # Process selection patterns
        for pattern in usage_patterns.get('selections', []):
            if 'code' in pattern:
                normalized = self._normalize_selection(pattern['code'])
                patterns['selections'].append({
                    'pattern': normalized,
                    'code': pattern['code'],
                    'source': pattern.get('source', '')
                })
        
        # Process data binding patterns
        for pattern in usage_patterns.get('data_binding', []):
            if 'code' in pattern:
                normalized = self._normalize_data_binding(pattern['code'])
                patterns['data_binding'].append({
                    'pattern': normalized,
                    'code': pattern['code'],
                    'source': pattern.get('source', '')
                })
        
        # Process scale patterns
        for pattern in usage_patterns.get('scales', []):
            if 'code' in pattern:
                normalized = self._normalize_scale(pattern['code'])
                patterns['scales'].append({
                    'pattern': normalized,
                    'code': pattern['code'],
                    'source': pattern.get('source', '')
                })
                
        self.patterns = patterns
        return patterns
    
    def extract_from_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract patterns from examples.
        
        Args:
            examples: Examples from D3ExampleParser
            
        Returns:
            Categorized and normalized patterns
        """
        for example in examples:
            code = example.get('code', '')
            source = example.get('source', '')
            
            # Extract method chaining patterns
            self._extract_method_chaining(code, source)
            
            # Extract selection patterns
            self._extract_selections(code, source)
            
            # Extract data binding patterns
            self._extract_data_binding(code, source)
            
            # Extract scaling patterns
            self._extract_scaling(code, source)
            
            # Extract layout patterns
            self._extract_layouts(code, source)
        
        # Consolidate patterns into the main dictionary
        if self.method_chaining_patterns:
            self.patterns['method_chaining'] = self.method_chaining_patterns
        
        if self.selection_patterns:
            self.patterns['selections'] = self.selection_patterns
        
        if self.data_binding_patterns:
            self.patterns['data_binding'] = self.data_binding_patterns
        
        if self.scaling_patterns:
            self.patterns['scales'] = self.scaling_patterns
        
        if self.layout_patterns:
            self.patterns['layouts'] = self.layout_patterns
            
        return self.patterns
    
    def _extract_method_chaining(self, code: str, source: str) -> None:
        """Extract method chaining patterns from code."""
        # Look for method chains with at least 3 methods
        pattern = r'(\w+(?:\.\w+)*\([^)]*\))(?:\.\w+\([^)]*\)){2,}'
        
        for match in re.finditer(pattern, code, re.DOTALL):
            chain = match.group(0)
            normalized = self._normalize_method_chain(chain)
            
            self.method_chaining_patterns.append({
                'pattern': normalized,
                'code': chain,
                'source': source
            })
    
    def _extract_selections(self, code: str, source: str) -> None:
        """Extract selection patterns from code."""
        # Look for d3.select or d3.selectAll
        pattern = r'd3\.select(?:All)?\([^)]*\)(?:\.\w+\([^)]*\))*'
        
        for match in re.finditer(pattern, code, re.DOTALL):
            selection = match.group(0)
            normalized = self._normalize_selection(selection)
            
            self.selection_patterns.append({
                'pattern': normalized,
                'code': selection,
                'source': source
            })
    
    def _extract_data_binding(self, code: str, source: str) -> None:
        """Extract data binding patterns from code."""
        # Look for .data() followed by .enter(), .exit(), etc.
        pattern = r'\.data\([^)]*\)(?:\.(?:enter|exit|append|join)\([^)]*\))*'
        
        for match in re.finditer(pattern, code, re.DOTALL):
            binding = match.group(0)
            normalized = self._normalize_data_binding(binding)
            
            self.data_binding_patterns.append({
                'pattern': normalized,
                'code': binding,
                'source': source
            })
    
    def _extract_scaling(self, code: str, source: str) -> None:
        """Extract scaling patterns from code."""
        # Look for scale creation and configuration
        pattern = r'd3\.(?:scale\w+|scaleLinear|scaleBand|scaleOrdinal|scaleTime)\(\)(?:\.(?:domain|range|padding|round|nice)\([^)]*\))*'
        
        for match in re.finditer(pattern, code, re.DOTALL):
            scale = match.group(0)
            normalized = self._normalize_scale(scale)
            
            self.scaling_patterns.append({
                'pattern': normalized,
                'code': scale,
                'source': source
            })
    
    def _extract_layouts(self, code: str, source: str) -> None:
        """Extract layout patterns from code."""
        # Look for layout creation and configuration
        layout_types = [
            'tree', 'cluster', 'pack', 'partition', 'treemap', 
            'forceSimulation', 'force', 'pie', 'stack', 'histogram'
        ]
        pattern_parts = []
        
        for layout in layout_types:
            pattern_parts.append(f'd3\\.{layout}\\(\\)(?:\\.\\w+\\([^)]*\\))*')
            
        pattern = '|'.join(pattern_parts)
        
        for match in re.finditer(pattern, code, re.DOTALL):
            layout = match.group(0)
            normalized = self._normalize_layout(layout)
            
            self.layout_patterns.append({
                'pattern': normalized,
                'code': layout,
                'source': source
            })
    
    def _normalize_method_chain(self, chain: str) -> str:
        """Normalize a method chain to a pattern descriptor."""
        # Extract the sequence of method calls
        methods = re.findall(r'\.(\w+)\(', chain)
        
        if methods:
            return ' -> '.join(methods)
        else:
            return "unknown_chain"
    
    def _normalize_selection(self, selection: str) -> str:
        """Normalize a selection pattern to a descriptor."""
        # Check if it's selectAll
        is_all = 'selectAll' in selection
        
        # Extract the selector
        selector_match = re.search(r'select(?:All)?\(\s*[\'"]([^\'"]*)[\'"]', selection)
        selector = selector_match.group(1) if selector_match else 'unknown'
        
        # Check for common patterns
        patterns = []
        if '.attr(' in selection:
            patterns.append('attr')
        if '.style(' in selection:
            patterns.append('style')
        if '.on(' in selection:
            patterns.append('event')
        if '.append(' in selection:
            patterns.append('append')
        if '.data(' in selection:
            patterns.append('data')
            
        pattern_str = '+'.join(patterns) if patterns else 'basic'
        
        return f"{'selectAll' if is_all else 'select'}({selector}) [{pattern_str}]"
    
    def _normalize_data_binding(self, binding: str) -> str:
        """Normalize a data binding pattern to a descriptor."""
        # Check what data binding operations are used
        operations = []
        
        if '.data(' in binding:
            operations.append('data')
            
        # Check for enter/exit/update pattern
        if '.enter(' in binding:
            operations.append('enter')
        if '.exit(' in binding:
            operations.append('exit')
        if '.update(' in binding or '.merge(' in binding:
            operations.append('update')
            
        # Check for append/insert operations
        if '.append(' in binding:
            operations.append('append')
        if '.insert(' in binding:
            operations.append('insert')
            
        # Check for join (D3v5+)
        if '.join(' in binding:
            operations.append('join')
            
        # Extract what is being appended if possible
        append_match = re.search(r'\.append\(\s*[\'"](\w+)[\'"]', binding)
        element_type = append_match.group(1) if append_match else ""
        
        # Create the pattern descriptor
        pattern_parts = []
        if operations:
            pattern_parts.append('-'.join(operations))
        if element_type:
            pattern_parts.append(f"to-{element_type}")
            
        return " ".join(pattern_parts) if pattern_parts else "basic-binding"
    
    def _normalize_scale(self, scale: str) -> str:
        """Normalize a scale pattern to a descriptor."""
        # Extract scale type
        scale_type_match = re.search(r'd3\.(?:scale)?(\w+)', scale)
        scale_type = scale_type_match.group(1) if scale_type_match else "unknown"
        
        # Check if it explicitly uses linear/band/log/etc.
        if scale_type.lower() == "scale":
            for scale_name in ["Linear", "Band", "Ordinal", "Log", "Time", "Sqrt", "Pow", "Quantize", "Quantile", "Threshold", "Sequential", "Diverging"]:
                if f"scale{scale_name}" in scale:
                    scale_type = scale_name.lower()
                    break
        
        # Check for domain/range configuration
        operations = []
        if '.domain(' in scale:
            operations.append('domain')
        if '.range(' in scale:
            operations.append('range')
        if '.rangeRound(' in scale:
            operations.append('rangeRound')
        if '.padding(' in scale or '.paddingInner(' in scale or '.paddingOuter(' in scale:
            operations.append('padding')
        if '.clamp(' in scale:
            operations.append('clamp')
        if '.nice(' in scale:
            operations.append('nice')
        
        # Create pattern descriptor
        if operations:
            return f"{scale_type}-scale with {', '.join(operations)}"
        else:
            return f"{scale_type}-scale basic"
    
    def _normalize_layout(self, layout: str) -> str:
        """Normalize a layout pattern to a descriptor."""
        # Extract layout type
        layout_types = [
            'tree', 'cluster', 'pack', 'partition', 'treemap', 
            'forceSimulation', 'force', 'pie', 'stack', 'histogram'
        ]
        
        layout_type = "unknown"
        for lt in layout_types:
            if f"d3.{lt}" in layout:
                layout_type = lt
                break
                
        # Check for configuration operations
        operations = []
        
        # Force layout specific operations
        if layout_type in ['force', 'forceSimulation']:
            if '.force(' in layout or any(f".force{force}" in layout for force in ['Center', 'Collide', 'Link', 'ManyBody', 'X', 'Y', 'Radial']):
                operations.append('force-config')
            if '.on(' in layout:
                operations.append('event-handler')
            if '.alpha(' in layout or '.alphaTarget(' in layout or '.alphaMin(' in layout or '.alphaDecay(' in layout:
                operations.append('alpha-config')
                
        # Hierarchy layout operations
        elif layout_type in ['tree', 'cluster', 'pack', 'partition', 'treemap']:
            if '.size(' in layout:
                operations.append('size')
            if '.nodeSize(' in layout:
                operations.append('nodeSize')
            if '.separation(' in layout:
                operations.append('separation')
            if '.padding(' in layout:
                operations.append('padding')
                
        # Pie layout operations
        elif layout_type == 'pie':
            if '.value(' in layout:
                operations.append('value')
            if '.sort(' in layout:
                operations.append('sort')
            if '.startAngle(' in layout or '.endAngle(' in layout:
                operations.append('angle-config')
                
        # Stack layout operations
        elif layout_type == 'stack':
            if '.keys(' in layout:
                operations.append('keys')
            if '.value(' in layout:
                operations.append('value')
            if '.order(' in layout:
                operations.append('order')
            if '.offset(' in layout:
                operations.append('offset')
                
        # Create the pattern descriptor
        if operations:
            return f"{layout_type}-layout with {', '.join(operations)}"
        else:
            return f"{layout_type}-layout basic"
    
    def categorize_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize patterns by visualization type.
        
        Returns:
            Dictionary mapping visualization types to patterns
        """
        categorized = {}
        
        # Combine all patterns for categorization
        all_patterns = []
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                pattern['category'] = category
                all_patterns.append(pattern)
        
        # Define visualization types and their associated keywords
        viz_types = {
            'bar_chart': ['bar', 'bars', 'rect', 'column'],
            'line_chart': ['line', 'path', 'curve'],
            'scatter_plot': ['circle', 'scatter', 'dots'],
            'pie_chart': ['pie', 'arc', 'wedge'],
            'force_graph': ['force', 'simulation', 'nodes', 'links'],
            'tree': ['tree', 'hierarchy', 'node'],
            'map': ['geo', 'map', 'projection', 'path'],
            'heatmap': ['heatmap', 'rect', 'cell']
        }
        
        # Categorize each pattern
        for pattern in all_patterns:
            code = pattern.get('code', '')
            
            for viz_type, keywords in viz_types.items():
                if any(keyword in code.lower() for keyword in keywords):
                    if viz_type not in categorized:
                        categorized[viz_type] = []
                    
                    categorized[viz_type].append(pattern)
                    break
            else:
                # If no specific viz type is found, add to 'generic'
                if 'generic' not in categorized:
                    categorized['generic'] = []
                
                categorized['generic'].append(pattern)
        
        return categorized
    
    def find_common_patterns(self, min_occurrences: int = 3) -> Dict[str, List[str]]:
        """
        Find common patterns that occur multiple times.
        
        Args:
            min_occurrences: Minimum number of occurrences to be considered common
            
        Returns:
            Dictionary mapping pattern categories to common patterns
        """
        common = {}
        
        for category, patterns in self.patterns.items():
            # Count occurrences of each normalized pattern
            pattern_counts = {}
            
            for pattern in patterns:
                normalized = pattern.get('pattern', '')
                if normalized:
                    if normalized not in pattern_counts:
                        pattern_counts[normalized] = 0
                    pattern_counts[normalized] += 1
            
            # Filter to patterns with at least min_occurrences
            common_patterns = [
                pattern for pattern, count in pattern_counts.items()
                if count >= min_occurrences
            ]
            
            if common_patterns:
                common[category] = common_patterns
                
        return common
    
    def save_patterns(self, output_path: str) -> None:
        """
        Save patterns to file.
        
        Args:
            output_path: Path to save the patterns
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, indent=2)
            
        print(f"Patterns saved to {output_path}")
    
    def get_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the extracted patterns.
        
        Returns:
            Dictionary of patterns by category
        """
        return self.patterns 