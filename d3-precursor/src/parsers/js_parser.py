"""
JavaScript parser for D3.js source code.

This module provides functionality to parse D3.js source code and extract
information about API structure, function definitions, and usage patterns.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple

# Try to import esprima, but provide a fallback if it's not available
try:
    import esprima
    HAS_ESPRIMA = True
except ImportError:
    HAS_ESPRIMA = False
    print("Warning: esprima not installed. Will use regex-based parsing.")


class D3JSParser:
    """Parse D3.js source code to extract API structure and usage patterns."""
    
    def __init__(self, source_path: str):
        """
        Initialize the parser with the path to D3.js source files.
        
        Args:
            source_path: Path to the D3.js source code directory
        """
        self.source_path = source_path
        self.ast_cache = {}
        self.api_definitions = {}
        self.module_structure = {}
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a JavaScript file into AST representation.
        
        Args:
            file_path: Path to the JavaScript file
            
        Returns:
            Dictionary representing the AST of the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if HAS_ESPRIMA:
            try:
                ast = esprima.parseScript(content, {'loc': True, 'comment': True})
                self.ast_cache[file_path] = ast
                return ast.toDict()
            except Exception as e:
                print(f"Error parsing {file_path} with esprima: {e}")
                return self._fallback_parse(content, file_path)
        else:
            return self._fallback_parse(content, file_path)
    
    def _fallback_parse(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Fallback parsing method using regex when esprima is not available.
        
        Args:
            content: JavaScript source code content
            file_path: Path to the source file for reference
            
        Returns:
            Dictionary with basic parsing information
        """
        # Simple regex-based extraction of function declarations and exports
        functions = []
        
        # Find function declarations: function name(...) {...}
        func_pattern = r'function\s+(\w+)\s*\(([\w\s,]*)\)\s*\{'
        for match in re.finditer(func_pattern, content):
            functions.append({
                'type': 'FunctionDeclaration',
                'name': match.group(1),
                'params': [p.strip() for p in match.group(2).split(',') if p.strip()],
                'range': [match.start(), match.end()]
            })
        
        # Find ES6 export declarations
        export_pattern = r'export\s+(?:(default)\s+)?(?:function|const|let|var)\s+(\w+)'
        exports = []
        for match in re.finditer(export_pattern, content):
            is_default = match.group(1) is not None
            name = match.group(2)
            exports.append({
                'type': 'ExportDeclaration',
                'name': name,
                'default': is_default,
                'range': [match.start(), match.end()]
            })
            
        # Find class declarations
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{'
        classes = []
        for match in re.finditer(class_pattern, content):
            classes.append({
                'type': 'ClassDeclaration',
                'name': match.group(1),
                'extends': match.group(2),
                'range': [match.start(), match.end()]
            })
            
        return {
            'functions': functions,
            'exports': exports,
            'classes': classes,
            'source': file_path
        }
            
    def parse_directory(self, dir_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse all JavaScript files in a directory.
        
        Args:
            dir_path: Path to the directory. If None, uses the source_path.
            
        Returns:
            Dictionary mapping file paths to their AST representations
        """
        if dir_path is None:
            dir_path = self.source_path
            
        results = {}
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.js'):
                    file_path = os.path.join(root, file)
                    results[file_path] = self.parse_file(file_path)
                    
        return results
    
    def extract_api_definitions(self) -> Dict[str, Any]:
        """
        Extract API definitions from parsed ASTs.
        
        Returns:
            Dictionary mapping file paths to API definitions
        """
        definitions = {}
        
        for file_path, ast in self.ast_cache.items():
            # Process declarations, exports, and function definitions
            file_definitions = self._process_declarations(ast)
            definitions[file_path] = file_definitions
            
        self.api_definitions = definitions
        return definitions
    
    def _process_declarations(self, ast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process declarations in AST to extract API components.
        
        Args:
            ast: AST representation of a JavaScript file
            
        Returns:
            List of API component definitions
        """
        definitions = []
        
        # Handle AST from esprima vs fallback parser
        if HAS_ESPRIMA and 'body' in ast:
            # Esprima AST
            for node in ast['body']:
                node_type = node.get('type')
                
                if node_type == 'FunctionDeclaration':
                    definitions.append(self._process_function_declaration(node))
                elif node_type == 'VariableDeclaration':
                    for decl in node.get('declarations', []):
                        if decl.get('init', {}).get('type') == 'FunctionExpression':
                            definitions.append(self._process_function_expression(decl))
                        elif decl.get('init', {}).get('type') == 'ArrowFunctionExpression':
                            definitions.append(self._process_arrow_function(decl))
                        elif decl.get('init', {}).get('type') == 'ObjectExpression':
                            # Process object properties that might contain methods
                            obj_defs = self._process_object_expression(decl)
                            if obj_defs:
                                definitions.extend(obj_defs)
                elif node_type == 'ExportNamedDeclaration' or node_type == 'ExportDefaultDeclaration':
                    export_def = self._process_export(node)
                    if export_def:
                        definitions.append(export_def)
                elif node_type == 'ClassDeclaration':
                    class_defs = self._process_class_declaration(node)
                    if class_defs:
                        definitions.extend(class_defs)
                elif node_type == 'ExpressionStatement':
                    # Check for method/prototype assignments like d3.select = function() {...}
                    expr = node.get('expression', {})
                    if expr.get('type') == 'AssignmentExpression':
                        assign_def = self._process_assignment(expr)
                        if assign_def:
                            definitions.append(assign_def)
        else:
            # Fallback parser AST
            for func in ast.get('functions', []):
                definitions.append({
                    'type': 'function',
                    'name': func['name'],
                    'params': func['params'],
                    'source': ast.get('source', '')
                })
            
            for exp in ast.get('exports', []):
                definitions.append({
                    'type': 'export',
                    'name': exp['name'],
                    'default': exp.get('default', False),
                    'source': ast.get('source', '')
                })
                
            for cls in ast.get('classes', []):
                definitions.append({
                    'type': 'class',
                    'name': cls['name'],
                    'extends': cls.get('extends'),
                    'source': ast.get('source', '')
                })
        
        return definitions
    
    def _process_function_declaration(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Process a function declaration node."""
        return {
            'type': 'function',
            'name': node.get('id', {}).get('name', 'anonymous'),
            'params': [param.get('name', '') for param in node.get('params', [])],
            'location': node.get('loc', {})
        }
    
    def _process_function_expression(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Process a variable declaration with function expression."""
        return {
            'type': 'function',
            'name': node.get('id', {}).get('name', 'anonymous'),
            'params': [param.get('name', '') for param in node.get('init', {}).get('params', [])],
            'location': node.get('loc', {})
        }
    
    def _process_arrow_function(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Process a variable declaration with arrow function expression."""
        return {
            'type': 'function',
            'name': node.get('id', {}).get('name', '') or node.get('left', {}).get('name', ''),
            'params': [param.get('name', '') for param in node.get('init', {}).get('params', [])],
            'location': node.get('loc', {}),
            'is_arrow': True
        }
    
    def _process_object_expression(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process an object expression to extract method definitions."""
        definitions = []
        obj_name = node.get('id', {}).get('name', 'anonymous')
        
        properties = node.get('init', {}).get('properties', [])
        for prop in properties:
            if prop.get('value', {}).get('type') in ['FunctionExpression', 'ArrowFunctionExpression']:
                key = prop.get('key', {})
                method_name = key.get('name', '') if key.get('type') == 'Identifier' else ''
                
                if method_name:
                    definitions.append({
                        'type': 'method',
                        'name': f"{obj_name}.{method_name}",
                        'parent': obj_name,
                        'params': [param.get('name', '') for param in prop.get('value', {}).get('params', [])],
                        'location': prop.get('loc', {})
                    })
        
        return definitions
    
    def _process_class_declaration(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a class declaration node."""
        definitions = []
        class_name = node.get('id', {}).get('name', 'AnonymousClass')
        
        # Get superclass if any
        extends = None
        if node.get('superClass'):
            if node.get('superClass', {}).get('type') == 'Identifier':
                extends = node.get('superClass', {}).get('name')
        
        # Add class definition
        definitions.append({
            'type': 'class',
            'name': class_name,
            'extends': extends,
            'location': node.get('loc', {})
        })
        
        # Process class body for methods
        for method_node in node.get('body', {}).get('body', []):
            if method_node.get('type') == 'MethodDefinition':
                method_name = method_node.get('key', {}).get('name', '')
                is_static = method_node.get('static', False)
                
                if method_name:
                    definitions.append({
                        'type': 'method',
                        'name': f"{class_name}.{method_name}",
                        'parent': class_name,
                        'is_static': is_static,
                        'params': [param.get('name', '') for param in method_node.get('value', {}).get('params', [])],
                        'location': method_node.get('loc', {})
                    })
        
        return definitions
    
    def _process_assignment(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an assignment expression that might define an API component."""
        left = node.get('left', {})
        right = node.get('right', {})
        
        # Only process assignments to functions/methods
        if right.get('type') not in ['FunctionExpression', 'ArrowFunctionExpression']:
            return None
        
        # Check if it's assigning to a property (like d3.select = function...)
        if left.get('type') == 'MemberExpression':
            obj = left.get('object', {})
            prop = left.get('property', {})
            
            obj_name = self._get_node_name(obj)
            prop_name = self._get_node_name(prop)
            
            if obj_name and prop_name:
                return {
                    'type': 'method',
                    'name': f"{obj_name}.{prop_name}",
                    'parent': obj_name,
                    'params': [param.get('name', '') for param in right.get('params', [])],
                    'location': node.get('loc', {})
                }
        
        return None
    
    def _get_node_name(self, node: Dict[str, Any]) -> str:
        """Extract name from a node that might be an identifier or another member expression."""
        if node.get('type') == 'Identifier':
            return node.get('name', '')
        elif node.get('type') == 'MemberExpression':
            obj_name = self._get_node_name(node.get('object', {}))
            prop_name = self._get_node_name(node.get('property', {}))
            return f"{obj_name}.{prop_name}" if obj_name and prop_name else ''
        return ''
    
    def _process_export(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Process an export declaration."""
        is_default = node.get('type') == 'ExportDefaultDeclaration'
        
        declaration = node.get('declaration', {})
        decl_type = declaration.get('type', '')
        
        if decl_type == 'FunctionDeclaration':
            return {
                'type': 'export_function',
                'name': declaration.get('id', {}).get('name', 'default_export'),
                'default': is_default,
                'params': [param.get('name', '') for param in declaration.get('params', [])],
                'location': node.get('loc', {})
            }
        
        return {
            'type': 'export',
            'name': 'unknown_export',
            'default': is_default,
            'location': node.get('loc', {})
        }
    
    def analyze_usage_patterns(self) -> Dict[str, Any]:
        """
        Identify common D3 usage patterns in the codebase.
        
        Returns:
            Dictionary of identified patterns by category
        """
        patterns = {
            'method_chaining': [],
            'selections': [],
            'data_binding': [],
            'scales': [],
            'transitions': []
        }
        
        for file_path, ast in self.ast_cache.items():
            self._extract_patterns_from_ast(ast, patterns, file_path)
            
        return patterns
    
    def _extract_patterns_from_ast(self, ast: Dict[str, Any], patterns: Dict[str, List], file_path: str) -> None:
        """Extract patterns from a single AST."""
        # For regex fallback, we need to examine the original source
        if not HAS_ESPRIMA:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Method chaining pattern
            method_chain_pattern = r'(\w+(?:\.\w+)*)\s*(\(.*?\))\s*\.\s*\w+\s*\('
            for match in re.finditer(method_chain_pattern, content):
                patterns['method_chaining'].append({
                    'code': match.group(0),
                    'source': file_path
                })
            
            # Selection pattern
            selection_pattern = r'd3\.select(?:All)?\s*\([^)]*\)'
            for match in re.finditer(selection_pattern, content):
                patterns['selections'].append({
                    'code': match.group(0),
                    'source': file_path
                })
            
            # Data binding pattern
            data_binding_pattern = r'\.data\s*\([^)]*\)'
            for match in re.finditer(data_binding_pattern, content):
                patterns['data_binding'].append({
                    'code': match.group(0),
                    'source': file_path
                })
        
        # If esprima is available, use AST to extract patterns
        # This would be more complex but also more accurate
        
        return 