"""
Example parser for D3 visualization examples.

This module provides functionality to parse D3 examples
and extract patterns, techniques, and best practices.
"""

import os
import re
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional, Set, Tuple
from .js_parser import D3JSParser
from .py_parser import D3PyParser


class D3ExampleParser:
    """Parse D3 examples to extract patterns and best practices."""
    
    def __init__(self, examples_path: str):
        """
        Initialize the parser with the path to example files.
        
        Args:
            examples_path: Path to the examples directory
        """
        self.examples_path = examples_path
        self.examples = []
        self.patterns = {}
        self.visualization_types = set()
        
    def parse_example_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a single example file.
        
        Args:
            file_path: Path to the example file
            
        Returns:
            Dictionary with parsed example information
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine the file type and appropriate parser
        if file_path.endswith('.js'):
            return self._parse_js_example(content, file_path)
        elif file_path.endswith('.py'):
            return self._parse_py_example(content, file_path)
        elif file_path.endswith('.html'):
            return self._parse_html_example(content, file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return {}
    
    def _parse_js_example(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse a JavaScript D3 example."""
        # Use the JavaScript parser to get detailed info
        js_parser = D3JSParser("")  # We don't need a source directory
        ast = js_parser._fallback_parse(content, file_path)
        
        # Extract D3 specific elements
        d3_calls = self._extract_d3_calls(content)
        visualization_type = self._detect_visualization_type(content, d3_calls)
        techniques = self._detect_techniques(content, d3_calls)
        
        return {
            'type': 'javascript',
            'source': file_path,
            'filename': os.path.basename(file_path),
            'code': content,
            'd3_calls': d3_calls,
            'visualization_type': visualization_type,
            'techniques': techniques
        }
    
    def _parse_py_example(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse a Python D3 example."""
        # Use the Python parser to get detailed info
        py_parser = D3PyParser("")  # We don't need a source directory
        
        # Extract D3 specific elements
        d3_imports = self._extract_d3_imports(content)
        visualization_type = self._detect_visualization_type_py(content)
        techniques = self._detect_techniques_py(content)
        
        return {
            'type': 'python',
            'source': file_path,
            'filename': os.path.basename(file_path),
            'code': content,
            'd3_imports': d3_imports,
            'visualization_type': visualization_type,
            'techniques': techniques
        }
    
    def _parse_html_example(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse an HTML file that contains D3 code."""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract JavaScript code from script tags
        js_code = ""
        for script in soup.find_all('script'):
            if script.string and ('d3' in script.string or 'D3' in script.string):
                js_code += script.string + "\n"
        
        # Use the same logic as the JavaScript parser
        if js_code:
            d3_calls = self._extract_d3_calls(js_code)
            visualization_type = self._detect_visualization_type(js_code, d3_calls)
            techniques = self._detect_techniques(js_code, d3_calls)
        else:
            d3_calls = []
            visualization_type = "unknown"
            techniques = []
        
        # Extract title and description
        title = ""
        description = ""
        
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.text.strip()
            
        # Look for a description in meta tags or first paragraph
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and 'content' in meta_desc.attrs:
            description = meta_desc['content']
        else:
            first_p = soup.find('p')
            if first_p:
                description = first_p.text.strip()
        
        return {
            'type': 'html',
            'source': file_path,
            'filename': os.path.basename(file_path),
            'code': js_code,
            'html': content,
            'title': title,
            'description': description,
            'd3_calls': d3_calls,
            'visualization_type': visualization_type,
            'techniques': techniques
        }
    
    def _extract_d3_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract D3 function calls from code."""
        d3_calls = []
        
        # Pattern to match d3.xxx() calls
        pattern = r'd3\.([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?\s*\(([^)]*)\)'
        for match in re.finditer(pattern, content):
            namespace = match.group(1)
            method = match.group(2)
            params = match.group(3).strip()
            
            call = {
                'namespace': namespace,
                'method': method,
                'params': params,
                'full': match.group(0)
            }
            
            if method:
                call['full_name'] = f"d3.{namespace}.{method}"
            else:
                call['full_name'] = f"d3.{namespace}"
                
            d3_calls.append(call)
            
        # Pattern to match method chaining: .xxx()
        chain_pattern = r'\.([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
        chains = []
        
        for match in re.finditer(chain_pattern, content):
            method = match.group(1)
            params = match.group(2).strip()
            
            if method not in ('then', 'catch', 'finally', 'map', 'filter', 'forEach'):  # Skip JavaScript standard methods
                chains.append({
                    'method': method,
                    'params': params,
                    'full': match.group(0)
                })
                
        # Add a chaining attribute if we detect chains
        if chains:
            d3_calls.append({
                'type': 'chaining',
                'chains': chains
            })
            
        return d3_calls
    
    def _extract_d3_imports(self, content: str) -> List[Dict[str, Any]]:
        """Extract D3 imports from Python code."""
        imports = []
        
        # Pattern for Python imports
        import_pattern = r'from\s+(d3[a-zA-Z0-9_.]*)\s+import\s+([^#\n]+)'
        for match in re.finditer(import_pattern, content):
            module = match.group(1)
            imported = [name.strip() for name in match.group(2).split(',')]
            
            imports.append({
                'module': module,
                'imported': imported,
                'full': match.group(0)
            })
            
        # Direct import pattern
        direct_pattern = r'import\s+(d3[a-zA-Z0-9_.]*)'
        for match in re.finditer(direct_pattern, content):
            module = match.group(1)
            
            imports.append({
                'module': module,
                'full': match.group(0)
            })
            
        return imports
    
    def _detect_visualization_type(self, content: str, d3_calls: List[Dict[str, Any]]) -> str:
        """Detect the type of visualization in the example."""
        # Look for specific D3 functions that indicate visualization types
        visualization_patterns = {
            'bar chart': [r'\.bar', r'\.bars', r'bar\s+chart', r'barchart'],
            'line chart': [r'\.line', r'line\s+chart', r'linechart', r'path.*?d3\.line'],
            'scatter plot': [r'\.scatter', r'scatter\s+plot', r'scatterplot'],
            'pie chart': [r'\.pie', r'pie\s+chart', r'piechart', r'arc', r'd3\.arc'],
            'force layout': [r'\.force', r'force\s+layout', r'd3\.forceSimulation'],
            'tree': [r'\.tree', r'd3\.tree', r'treeLayout'],
            'map': [r'\.geo', r'd3\.geo', r'topojson', r'projection'],
            'network': [r'\.links', r'\.nodes', r'network', r'graph'],
            'histogram': [r'\.histogram', r'histogram', r'bin'],
            'heatmap': [r'heatmap', r'heat\s+map'],
            'chord diagram': [r'\.chord', r'chord', r'd3\.chord'],
            'sankey diagram': [r'\.sankey', r'sankey', r'd3\.sankey'],
            'pack layout': [r'\.pack', r'd3\.pack', r'circle\s+packing'],
            'streamgraph': [r'\.stack', r'streamgraph', r'stream\s+graph'],
            'treemap': [r'\.treemap', r'd3\.treemap'],
            'sunburst': [r'\.sunburst', r'sunburst', r'partition']
        }
        
        # Check for each visualization type
        for viz_type, patterns in visualization_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return viz_type
                    
        # Check function calls
        for call in d3_calls:
            namespace = call.get('namespace', '')
            method = call.get('method', '')
            
            # Check for specific D3 modules
            if namespace in ('pie', 'arc', 'line', 'area', 'bar', 'stack'):
                return f"{namespace} chart"
            elif method in ('pie', 'arc', 'line', 'area', 'bar', 'stack'):
                return f"{method} chart"
                
        # Default if nothing specific is found
        return "generic visualization"
    
    def _detect_visualization_type_py(self, content: str) -> str:
        """Detect the type of visualization in Python example."""
        # Similar logic to JavaScript but with Python patterns
        visualization_patterns = {
            'bar chart': [r'bar_chart', r'barchart', r'\.bar\(', r'bar\s+chart'],
            'line chart': [r'line_chart', r'linechart', r'\.line\(', r'line\s+chart'],
            'scatter plot': [r'scatter_plot', r'scatterplot', r'\.scatter\(', r'scatter\s+plot'],
            'pie chart': [r'pie_chart', r'piechart', r'\.pie\(', r'pie\s+chart'],
            'map': [r'\.geo', r'geopandas', r'folium', r'choropleth', r'map\s+plot'],
            'network': [r'networkx', r'graph', r'network'],
            'heatmap': [r'heatmap', r'heat_map', r'heat\s+map', r'\.imshow']
        }
        
        # Check for each visualization type
        for viz_type, patterns in visualization_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return viz_type
                    
        # Default if nothing specific is found
        return "generic visualization"
    
    def _detect_techniques(self, content: str, d3_calls: List[Dict[str, Any]]) -> List[str]:
        """Detect visualization techniques used in the example."""
        techniques = []
        
        # Look for common techniques
        technique_patterns = {
            'transitions': [r'\.transition', r'\.duration'],
            'data binding': [r'\.data\(', r'\.datum\(', r'\.enter\(', r'\.exit\('],
            'scales': [r'\.scale', r'd3\.scale', r'scaleLinear', r'scaleBand'],
            'axes': [r'\.axis', r'd3\.axis', r'axisLeft', r'axisBottom'],
            'color schemes': [r'\.scheme', r'd3\.scheme', r'interpolate', r'schemeCategory'],
            'zoom': [r'\.zoom', r'd3\.zoom'],
            'brush': [r'\.brush', r'd3\.brush'],
            'tooltips': [r'tooltip', r'\.hover', r'mouseover', r'mouseout'],
            'responsive': [r'viewBox', r'media query', r'resize', r'responsive'],
            'update pattern': [r'\.merge\(', r'\.enter\(.*\.append', r'\.exit\(.*\.remove'],
            'force simulation': [r'\.force', r'forceSimulation', r'forceManyBody'],
            'geo projection': [r'\.projection', r'geoPath', r'geoMercator'],
            'voronoi': [r'\.voronoi', r'd3\.voronoi', r'delaunay'],
            'contour': [r'\.contour', r'd3\.contour']
        }
        
        # Check for each technique
        for technique, patterns in technique_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    techniques.append(technique)
                    break  # Found this technique, move to next
                    
        # Add method chaining if detected
        for call in d3_calls:
            if call.get('type') == 'chaining' and len(call.get('chains', [])) > 2:
                techniques.append('method chaining')
                break
                
        return list(set(techniques))  # Remove duplicates
    
    def _detect_techniques_py(self, content: str) -> List[str]:
        """Detect visualization techniques used in the Python example."""
        techniques = []
        
        # Look for common Python D3 techniques
        technique_patterns = {
            'interactive': [r'interact', r'widget', r'ipywidgets', r'event'],
            'animation': [r'animation', r'animate', r'transition'],
            'scales': [r'scale', r'normalize', r'standardize'],
            'colormap': [r'colormap', r'cmap', r'color_scale'],
            'tooltips': [r'tooltip', r'hover', r'popup'],
            'responsive': [r'responsive', r'resize', r'figure\s*\(\s*figsize'],
            'geographic': [r'projection', r'geopandas', r'folium', r'map'],
            'statistics': [r'pandas', r'groupby', r'mean\(', r'sum\('],
        }
        
        # Check for each technique
        for technique, patterns in technique_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    techniques.append(technique)
                    break  # Found this technique, move to next
                    
        return list(set(techniques))  # Remove duplicates
    
    def parse_directory(self, dir_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse all example files in a directory.
        
        Args:
            dir_path: Path to the directory. If None, uses examples_path.
            
        Returns:
            List of parsed examples
        """
        if dir_path is None:
            dir_path = self.examples_path
            
        examples = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip non-supported files
                if not any(file.endswith(ext) for ext in ['.js', '.py', '.html']):
                    continue
                    
                try:
                    example = self.parse_example_file(file_path)
                    if example:
                        examples.append(example)
                        
                        # Update visualization types set
                        if 'visualization_type' in example:
                            self.visualization_types.add(example['visualization_type'])
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
        
        self.examples = examples
        return examples
    
    def extract_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract common patterns from parsed examples.
        
        Returns:
            Dictionary of patterns grouped by category
        """
        if not self.examples:
            self.parse_directory()
            
        patterns = {
            'data_binding': [],
            'scales': [],
            'axes': [],
            'transitions': [],
            'layouts': [],
            'interactions': [],
            'method_chaining': []
        }
        
        # Process examples to find patterns
        for example in self.examples:
            # Extract method chaining patterns
            for call in example.get('d3_calls', []):
                if call.get('type') == 'chaining':
                    # Only consider significant chains (more than 3 methods)
                    chains = call.get('chains', [])
                    if len(chains) > 3:
                        pattern = {
                            'methods': [chain['method'] for chain in chains],
                            'source': example['source'],
                            'visualization_type': example.get('visualization_type', 'unknown')
                        }
                        patterns['method_chaining'].append(pattern)
            
            # Group by techniques
            for technique in example.get('techniques', []):
                category = None
                
                # Map technique to pattern category
                if technique in ('data binding', 'update pattern'):
                    category = 'data_binding'
                elif technique in ('scales', 'color schemes'):
                    category = 'scales'
                elif technique == 'axes':
                    category = 'axes'
                elif technique in ('transitions', 'animation'):
                    category = 'transitions'
                elif technique in ('force simulation', 'tree', 'pack layout', 'treemap'):
                    category = 'layouts'
                elif technique in ('zoom', 'brush', 'tooltips', 'interactive'):
                    category = 'interactions'
                    
                if category:
                    # Extract a code snippet that illustrates this technique
                    snippet = self._extract_snippet_for_technique(example, technique)
                    if snippet:
                        pattern = {
                            'technique': technique,
                            'visualization_type': example.get('visualization_type', 'unknown'),
                            'code_snippet': snippet,
                            'source': example['source']
                        }
                        patterns[category].append(pattern)
        
        self.patterns = patterns
        return patterns
    
    def _extract_snippet_for_technique(self, example: Dict[str, Any], technique: str) -> str:
        """Extract a code snippet that illustrates a specific technique."""
        code = example.get('code', '')
        
        # Define patterns to match snippets for each technique
        technique_patterns = {
            'transitions': r'\.transition\s*\([^)]*\).*?;',
            'data binding': r'\.data\s*\([^)]*\)(.*?)(?:\.enter\(\)|;)',
            'update pattern': r'\.data\s*\([^)]*\).*?\.exit\(\).*?\.remove\(\)',
            'scales': r'(?:d3\.scale\w+|scaleLinear|scaleBand).*?(?:\.domain|\.range)',
            'axes': r'(?:d3\.axis\w+|axisLeft|axisBottom).*?\.call\s*\([^)]*\)',
            'color schemes': r'(?:d3\.scheme\w+|interpolate\w+|schemeCategory)',
            'zoom': r'\.call\s*\(\s*d3\.zoom\s*\([^)]*\)\s*\)',
            'brush': r'\.call\s*\(\s*d3\.brush\s*\([^)]*\)\s*\)',
            'tooltips': r'(?:\.on\s*\(\s*[\'"]mouseover[\'"]\s*,.*?\.on\s*\(\s*[\'"]mouseout[\'"]\s*,|\.append\s*\(\s*[\'"]title[\'"]\s*\))',
            'force simulation': r'd3\.forceSimulation\s*\([^)]*\).*?\.force\s*\(',
            'geo projection': r'\.projection\s*\(\s*d3\.geo\w+\s*\(\s*\)\s*\)'
        }
        
        pattern = technique_patterns.get(technique)
        if pattern:
            match = re.search(pattern, code, re.DOTALL)
            if match:
                return match.group(0)
                
        # For other techniques, return a small section around any keyword
        keywords = {
            'method chaining': [r'\.attr', r'\.style'],
            'responsive': [r'viewBox', r'resize'],
            'voronoi': [r'voronoi', r'delaunay'],
            'contour': [r'contour'],
            'interactive': [r'interact', r'widget'],
            'animation': [r'animate'],
            'colormap': [r'colormap', r'cmap'],
            'geographic': [r'projection'],
            'statistics': [r'groupby', r'mean\(']
        }
        
        for key, words in keywords.items():
            if technique == key:
                for word in words:
                    match = re.search(f'(?:.*?{word}.*?){{1,5}}', code, re.DOTALL)
                    if match:
                        # Extract a reasonable snippet (at most 5 lines)
                        snippet = match.group(0)
                        lines = snippet.splitlines()
                        if len(lines) > 5:
                            snippet = '\n'.join(lines[:5]) + '\n...'
                        return snippet
                        
        return ""
    
    def categorize_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize examples by visualization type.
        
        Returns:
            Dictionary mapping visualization types to examples
        """
        if not self.examples:
            self.parse_directory()
            
        categorized = {}
        
        for example in self.examples:
            viz_type = example.get('visualization_type', 'unknown')
            
            if viz_type not in categorized:
                categorized[viz_type] = []
                
            categorized[viz_type].append(example)
            
        return categorized
    
    def save_results(self, output_dir: str) -> None:
        """
        Save parsing results to output directory.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save examples
        examples_path = os.path.join(output_dir, 'examples.json')
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, indent=2)
            
        # Save patterns
        if not self.patterns:
            self.extract_patterns()
            
        patterns_path = os.path.join(output_dir, 'patterns.json')
        with open(patterns_path, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, indent=2)
            
        # Save categorized examples
        categorized = self.categorize_examples()
        categories_path = os.path.join(output_dir, 'categorized_examples.json')
        with open(categories_path, 'w', encoding='utf-8') as f:
            json.dump(categorized, f, indent=2)
            
        print(f"Results saved to {output_dir}")
        
    def get_examples_by_technique(self, technique: str) -> List[Dict[str, Any]]:
        """
        Get examples that use a specific technique.
        
        Args:
            technique: Name of the technique to find
            
        Returns:
            List of examples using the technique
        """
        if not self.examples:
            self.parse_directory()
            
        matching = []
        for example in self.examples:
            if technique in example.get('techniques', []):
                matching.append(example)
                
        return matching
    
    def get_examples_by_visualization(self, viz_type: str) -> List[Dict[str, Any]]:
        """
        Get examples of a specific visualization type.
        
        Args:
            viz_type: Type of visualization to find
            
        Returns:
            List of examples of the visualization type
        """
        if not self.examples:
            self.parse_directory()
            
        matching = []
        for example in self.examples:
            if example.get('visualization_type', '').lower() == viz_type.lower():
                matching.append(example)
                
        return matching 