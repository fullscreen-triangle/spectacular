"""
Python parser for D3 bindings.

This module provides functionality to parse Python D3 bindings
and extract information about API structure and usage patterns.
"""

import os
import ast
import astunparse
from typing import Dict, List, Any, Optional, Set, Tuple


class D3PyParser:
    """Parse Python D3 bindings and examples."""
    
    def __init__(self, source_path: str):
        """
        Initialize the parser with the path to Python D3 source files.
        
        Args:
            source_path: Path to the Python D3 source code directory
        """
        self.source_path = source_path
        self.ast_cache = {}
        self.api_definitions = {}
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Python file into AST representation.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary representing the AST of the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            self.ast_cache[file_path] = tree
            return self._ast_to_dict(tree, file_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {'error': str(e), 'source': file_path}
    
    def _ast_to_dict(self, node, file_path: str) -> Dict[str, Any]:
        """
        Convert AST node to dictionary representation.
        
        Args:
            node: AST node
            file_path: Source file path
            
        Returns:
            Dictionary representation of the AST
        """
        if isinstance(node, ast.Module):
            return {
                'type': 'Module',
                'body': [self._process_node(n) for n in node.body],
                'source': file_path
            }
        elif isinstance(node, ast.FunctionDef):
            return self._process_function_def(node)
        elif isinstance(node, ast.ClassDef):
            return self._process_class_def(node)
        elif isinstance(node, ast.Import):
            return self._process_import(node)
        elif isinstance(node, ast.ImportFrom):
            return self._process_import_from(node)
        elif isinstance(node, ast.Assign):
            return self._process_assignment(node)
        elif isinstance(node, ast.Expr):
            return self._process_expression(node)
        elif isinstance(node, ast.Call):
            return self._process_call(node)
        elif isinstance(node, ast.If):
            return {
                'type': 'if_statement',
                'test': self._ast_to_dict(node.test, file_path) if hasattr(node, 'test') else None,
                'body': [self._ast_to_dict(n, file_path) for n in node.body],
                'orelse': [self._ast_to_dict(n, file_path) for n in node.orelse],
                'source': file_path
            }
        elif isinstance(node, ast.For):
            return {
                'type': 'for_loop',
                'target': self._ast_to_dict(node.target, file_path) if hasattr(node, 'target') else None,
                'iter': self._ast_to_dict(node.iter, file_path) if hasattr(node, 'iter') else None,
                'body': [self._ast_to_dict(n, file_path) for n in node.body],
                'source': file_path
            }
        elif isinstance(node, ast.While):
            return {
                'type': 'while_loop',
                'test': self._ast_to_dict(node.test, file_path) if hasattr(node, 'test') else None,
                'body': [self._ast_to_dict(n, file_path) for n in node.body],
                'source': file_path
            }
        elif isinstance(node, ast.With):
            return {
                'type': 'with_statement',
                'items': [self._ast_to_dict(n, file_path) for n in node.items],
                'body': [self._ast_to_dict(n, file_path) for n in node.body],
                'source': file_path
            }
        elif isinstance(node, ast.Try):
            return {
                'type': 'try_statement',
                'body': [self._ast_to_dict(n, file_path) for n in node.body],
                'handlers': [self._ast_to_dict(n, file_path) for n in node.handlers],
                'finalbody': [self._ast_to_dict(n, file_path) for n in node.finalbody],
                'source': file_path
            }
        elif isinstance(node, ast.ExceptHandler):
            return {
                'type': 'except_handler',
                'type_node': self._ast_to_dict(node.type, file_path) if node.type else None,
                'name': node.name,
                'body': [self._ast_to_dict(n, file_path) for n in node.body],
                'source': file_path
            }
        elif isinstance(node, ast.Name):
            return {
                'type': 'name',
                'id': node.id,
                'source': file_path
            }
        elif isinstance(node, ast.Attribute):
            return {
                'type': 'attribute',
                'value': self._ast_to_dict(node.value, file_path) if hasattr(node, 'value') else None,
                'attr': node.attr,
                'source': file_path
            }
        elif isinstance(node, ast.Constant):
            return {
                'type': 'constant',
                'value': node.value if hasattr(node, 'value') else None,
                'source': file_path
            }
        elif isinstance(node, ast.Str):  # For Python < 3.8 compatibility
            return {
                'type': 'constant',
                'value': node.s,
                'source': file_path
            }
        elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return {
                'type': 'constant',
                'value': node.n,
                'source': file_path
            }
        elif isinstance(node, ast.List):
            return {
                'type': 'list',
                'elts': [self._ast_to_dict(n, file_path) for n in node.elts],
                'source': file_path
            }
        elif isinstance(node, ast.Dict):
            return {
                'type': 'dict',
                'keys': [self._ast_to_dict(n, file_path) if n else None for n in node.keys],
                'values': [self._ast_to_dict(n, file_path) if n else None for n in node.values],
                'source': file_path
            }
        
        # Default case for nodes we don't explicitly handle
        return {
            'type': type(node).__name__,
            'source': file_path
        }
    
    def _process_expression(self, node: ast.Expr) -> Dict[str, Any]:
        """Process an expression node."""
        if isinstance(node.value, ast.Str):
            # Detect docstrings
            return {
                'type': 'docstring',
                'value': node.value.s,
            }
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # For Python 3.8+ where Str is deprecated
            return {
                'type': 'docstring',
                'value': node.value.value,
            }
        elif isinstance(node.value, ast.Call):
            # Function/method calls
            return self._process_call(node.value)
        
        return {'type': 'expression', 'expr_type': type(node.value).__name__}
    
    def _process_call(self, node: ast.Call) -> Dict[str, Any]:
        """Process a function/method call node."""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = self._get_name(node.func)
        
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args.append({'type': 'arg', 'name': arg.id})
            elif isinstance(arg, ast.Constant):
                args.append({'type': 'arg', 'value': arg.value})
            elif hasattr(arg, 's'):  # str in Python < 3.8
                args.append({'type': 'arg', 'value': arg.s})
            elif hasattr(arg, 'n'):  # num in Python < 3.8
                args.append({'type': 'arg', 'value': arg.n})
            else:
                args.append({'type': 'arg', 'arg_type': type(arg).__name__})
        
        # Process keyword arguments
        kwargs = []
        for kwarg in node.keywords:
            value = None
            if isinstance(kwarg.value, ast.Name):
                value = {'type': 'name', 'id': kwarg.value.id}
            elif isinstance(kwarg.value, ast.Constant):
                value = {'type': 'constant', 'value': kwarg.value.value}
            elif hasattr(kwarg.value, 's'):  # str in Python < 3.8
                value = {'type': 'constant', 'value': kwarg.value.s}
            elif hasattr(kwarg.value, 'n'):  # num in Python < 3.8
                value = {'type': 'constant', 'value': kwarg.value.n}
            else:
                value = {'type': type(kwarg.value).__name__}
            
            kwargs.append({
                'arg': kwarg.arg,
                'value': value
            })
        
        return {
            'type': 'call',
            'func': func_name,
            'args': args,
            'keywords': kwargs
        }
    
    def _process_node(self, node) -> Dict[str, Any]:
        """Process a single AST node."""
        if isinstance(node, ast.FunctionDef):
            return self._process_function_def(node)
        elif isinstance(node, ast.ClassDef):
            return self._process_class_def(node)
        elif isinstance(node, ast.Import):
            return self._process_import(node)
        elif isinstance(node, ast.ImportFrom):
            return self._process_import_from(node)
        elif isinstance(node, ast.Assign):
            return self._process_assignment(node)
        
        return {'type': 'Other', 'node_type': type(node).__name__}
    
    def _process_function_def(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Process a function definition node."""
        params = []
        for arg in node.args.args:
            param = {'name': arg.arg}
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                param['annotation'] = self._get_annotation_name(arg.annotation)
            params.append(param)
            
        return {
            'type': 'function',
            'name': node.name,
            'params': params,
            'decorator_list': [self._get_name(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node)
        }
    
    def _process_class_def(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Process a class definition node."""
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._process_function_def(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({'name': target.id})
        
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_name(base))
        
        return {
            'type': 'class',
            'name': node.name,
            'bases': bases,
            'methods': methods,
            'attributes': attributes,
            'docstring': ast.get_docstring(node)
        }
    
    def _process_import(self, node: ast.Import) -> Dict[str, Any]:
        """Process an import statement."""
        names = [{'name': n.name, 'asname': n.asname} for n in node.names]
        return {'type': 'import', 'names': names}
    
    def _process_import_from(self, node: ast.ImportFrom) -> Dict[str, Any]:
        """Process an import from statement."""
        names = [{'name': n.name, 'asname': n.asname} for n in node.names]
        return {
            'type': 'import_from',
            'module': node.module,
            'names': names
        }
    
    def _process_assignment(self, node: ast.Assign) -> Dict[str, Any]:
        """Process an assignment statement."""
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, ast.Attribute):
                targets.append(self._get_name(target))
        
        return {
            'type': 'assignment',
            'targets': targets
        }
    
    def _get_name(self, node) -> str:
        """Get name from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"
    
    def _get_annotation_name(self, node) -> str:
        """Get annotation name from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_name(node)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        return "unknown"
        
    def parse_directory(self, dir_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse all Python files in a directory.
        
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
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    results[file_path] = self.parse_file(file_path)
                    
        return results
    
    def extract_api_definitions(self) -> Dict[str, Any]:
        """
        Extract API definitions from parsed Python files.
        
        Returns:
            Dictionary mapping file paths to API definitions
        """
        definitions = {}
        
        for file_path, ast_dict in self.ast_cache.items():
            file_definitions = self._extract_definitions_from_ast(ast_dict, file_path)
            definitions[file_path] = file_definitions
            
        self.api_definitions = definitions
        return definitions
    
    def _extract_definitions_from_ast(self, ast_dict: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
        """Extract API definitions from a parsed AST."""
        definitions = []
        
        # Extract from parsed AST dict
        if 'body' in ast_dict:
            for item in ast_dict['body']:
                if item['type'] == 'function':
                    is_d3_related = self._is_d3_related(item)
                    if is_d3_related:
                        definitions.append({
                            'type': 'function',
                            'name': item['name'],
                            'params': item.get('params', []),
                            'docstring': item.get('docstring', ''),
                            'source': file_path
                        })
                elif item['type'] == 'class':
                    is_d3_related = self._is_d3_related(item)
                    if is_d3_related:
                        for method in item.get('methods', []):
                            definitions.append({
                                'type': 'method',
                                'name': f"{item['name']}.{method['name']}",
                                'params': method.get('params', []),
                                'docstring': method.get('docstring', ''),
                                'class': item['name'],
                                'source': file_path
                            })
                            
        return definitions
    
    def _is_d3_related(self, item: Dict[str, Any]) -> bool:
        """
        Determine if an item is related to D3.
        
        This is a heuristic function that checks if the item is likely
        to be a D3 API item based on naming, docstrings, etc.
        """
        # Check name for d3 or common visualization terms
        name = item.get('name', '').lower()
        d3_terms = ['d3', 'visualization', 'chart', 'graph', 'plot', 'svg', 'axis', 'scale']
        
        if any(term in name for term in d3_terms):
            return True
        
        # Check docstring for d3 mentions
        docstring = item.get('docstring', '').lower()
        if 'd3' in docstring:
            return True
            
        return False
    
    def find_d3_imports(self) -> Dict[str, List[str]]:
        """
        Find D3-related imports in the Python files.
        
        Returns:
            Dictionary mapping file paths to lists of D3 imports
        """
        d3_imports = {}
        
        for file_path, ast_dict in self.ast_cache.items():
            imports = []
            
            if 'body' in ast_dict:
                for item in ast_dict['body']:
                    if item['type'] == 'import':
                        for name_info in item.get('names', []):
                            if 'd3' in name_info.get('name', '').lower():
                                imports.append({
                                    'type': 'import',
                                    'name': name_info.get('name', ''),
                                    'asname': name_info.get('asname')
                                })
                    elif item['type'] == 'import_from':
                        module = item.get('module', '')
                        if 'd3' in module.lower():
                            for name_info in item.get('names', []):
                                imports.append({
                                    'type': 'import_from',
                                    'module': module,
                                    'name': name_info.get('name', ''),
                                    'asname': name_info.get('asname')
                                })
            
            if imports:
                d3_imports[file_path] = imports
                
        return d3_imports 