import re
from typing import Dict, List, Any, Optional, Union
import ast
import json


class CodeProcessor:
    """Process and analyze code snippets."""
    
    def __init__(self):
        """Initialize the code processor."""
        # Patterns for detecting comments
        self.js_single_line_comment = r'\/\/.*?$'
        self.js_multi_line_comment = r'\/\*[\s\S]*?\*\/'
        
        # Pattern for detecting function declarations
        self.js_function_pattern = r'(?:function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)|(?:const|let|var)?\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:function)?\s*\(([^)]*)\)|\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*(?:function)?\s*\(([^)]*)\))'
        
        # Pattern for detecting class declarations
        self.js_class_pattern = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        
        # Pattern for detecting d3 chaining patterns
        self.d3_chain_pattern = r'd3\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)(?:\s*\.\s*[a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\))+' 
    
    def extract_comments(self, code: str) -> List[str]:
        """Extract comments from code.
        
        Args:
            code: The code to extract comments from.
            
        Returns:
            A list of extracted comments.
        """
        comments = []
        
        # Extract single line comments
        single_line = re.findall(self.js_single_line_comment, code, re.MULTILINE)
        for comment in single_line:
            clean = comment.replace('//', '').strip()
            if clean:
                comments.append(clean)
        
        # Extract multi-line comments
        multi_line = re.findall(self.js_multi_line_comment, code, re.DOTALL)
        for comment in multi_line:
            # Clean and split by lines
            clean = comment.replace('/*', '').replace('*/', '').strip()
            # Remove leading asterisks common in JSDoc style comments
            clean = re.sub(r'^\s*\*\s*', '', clean, flags=re.MULTILINE)
            if clean:
                # Split multi-line comments into separate lines
                comment_lines = [line.strip() for line in clean.split('\n')]
                # Filter out empty lines
                comment_lines = [line for line in comment_lines if line]
                comments.extend(comment_lines)
        
        return comments
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function declarations from code.
        
        Args:
            code: The code to extract functions from.
            
        Returns:
            A list of dictionaries with function details.
        """
        functions = []
        
        # First, remove comments to avoid false positives
        code_no_comments = re.sub(self.js_single_line_comment, '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(self.js_multi_line_comment, '', code_no_comments, flags=re.DOTALL)
        
        # Find function declarations
        matches = re.finditer(self.js_function_pattern, code_no_comments, re.MULTILINE)
        
        for match in matches:
            # The pattern has multiple capture groups for different function declaration styles
            # Only one of these groups will match for each function
            name = None
            params = None
            
            if match.group(1):
                # function name(params) style
                name = match.group(1)
                params = match.group(2)
            elif match.group(3):
                # const name = function(params) or const name = (params) => style
                name = match.group(3)
                params = match.group(4)
            elif match.group(5):
                # object method style: name: function(params)
                name = match.group(5)
                params = match.group(6)
            
            if name and params is not None:
                # Extract parameter list
                param_list = [p.strip() for p in params.split(',') if p.strip()]
                
                # Look for JSDoc comment before the function
                func_pos = match.start()
                comment_end = func_pos
                while comment_end > 0 and code[comment_end-1].isspace():
                    comment_end -= 1
                
                docstring = None
                if comment_end > 0 and code[comment_end-2:comment_end] == '*/':
                    # Find start of comment
                    comment_start = code[:comment_end].rfind('/*')
                    if comment_start != -1:
                        docstring = code[comment_start+2:comment_end-2].strip()
                
                functions.append({
                    'name': name,
                    'parameters': param_list,
                    'docstring': docstring,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return functions
    
    def extract_d3_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Extract D3.js usage patterns from code.
        
        Args:
            code: The code to extract patterns from.
            
        Returns:
            A list of dictionaries with pattern details.
        """
        patterns = []
        
        # First, remove comments to avoid false positives
        code_no_comments = re.sub(self.js_single_line_comment, '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(self.js_multi_line_comment, '', code_no_comments, flags=re.DOTALL)
        
        # Find D3 method chains
        chain_matches = re.finditer(self.d3_chain_pattern, code_no_comments, re.MULTILINE | re.DOTALL)
        
        for match in chain_matches:
            chain_text = match.group(0)
            
            # Extract the starting method
            start_method = match.group(1)
            
            # Extract all method calls in the chain
            method_calls = re.findall(r'\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)', chain_text)
            
            patterns.append({
                'type': 'method_chain',
                'start_method': start_method,
                'method_calls': [{'method': method, 'args': args.strip()} for method, args in method_calls],
                'start': match.start(),
                'end': match.end(),
                'code': chain_text
            })
        
        # Find common D3 patterns like selections
        selection_pattern = r'd3\.select(?:All)?\s*\([^)]*\)'
        selection_matches = re.finditer(selection_pattern, code_no_comments, re.MULTILINE)
        
        for match in selection_matches:
            selection_text = match.group(0)
            
            # Only include standalone selections (not part of a chain)
            chain_found = False
            for chain in patterns:
                if chain['start'] <= match.start() <= chain['end']:
                    chain_found = True
                    break
            
            if not chain_found:
                patterns.append({
                    'type': 'selection',
                    'start': match.start(),
                    'end': match.end(),
                    'code': selection_text
                })
        
        return patterns
    
    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze the structure of the code.
        
        Args:
            code: The code to analyze.
            
        Returns:
            A dictionary with the analysis results.
        """
        result = {
            'functions': self.extract_functions(code),
            'patterns': self.extract_d3_patterns(code),
            'comments': self.extract_comments(code),
        }
        
        # Count lines of code
        code_lines = code.split('\n')
        result['line_count'] = len(code_lines)
        
        # Count non-empty lines of code
        result['non_empty_line_count'] = sum(1 for line in code_lines if line.strip())
        
        # Count D3.js related lines
        d3_lines = sum(1 for line in code_lines if 'd3.' in line)
        result['d3_line_count'] = d3_lines
        
        # Calculate D3 usage density
        if result['non_empty_line_count'] > 0:
            result['d3_density'] = d3_lines / result['non_empty_line_count']
        else:
            result['d3_density'] = 0
        
        return result
    
    def extract_api_usage(self, code: str) -> List[str]:
        """Extract D3.js API method usage from code.
        
        Args:
            code: The code to extract API usage from.
            
        Returns:
            A list of D3.js API methods used in the code.
        """
        # Remove comments
        code_no_comments = re.sub(self.js_single_line_comment, '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(self.js_multi_line_comment, '', code_no_comments, flags=re.DOTALL)
        
        # Find all d3.method calls
        api_calls = re.findall(r'd3\.([a-zA-Z_$][a-zA-Z0-9_$]*)', code_no_comments)
        
        # Remove duplicates
        unique_calls = list(set(api_calls))
        
        return unique_calls
    
    def clean_code(self, code: str) -> str:
        """Clean code by removing unnecessary whitespace, etc.
        
        Args:
            code: The code to clean.
            
        Returns:
            The cleaned code.
        """
        # Remove trailing whitespace
        code = re.sub(r'\s+$', '', code, flags=re.MULTILINE)
        
        # Remove multiple blank lines
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        # Normalize indentation to spaces
        code = self._normalize_indentation(code)
        
        return code
    
    def _normalize_indentation(self, code: str, spaces: int = 2) -> str:
        """Normalize indentation to a fixed number of spaces.
        
        Args:
            code: The code to normalize.
            spaces: Number of spaces to use for indentation.
            
        Returns:
            Code with normalized indentation.
        """
        lines = code.split('\n')
        result = []
        
        # Determine if the code uses tabs or spaces
        tab_pattern = re.compile(r'^\t+')
        space_pattern = re.compile(r'^( +)')
        
        # Infer indentation style from the code
        uses_tabs = False
        for line in lines:
            if tab_pattern.match(line):
                uses_tabs = True
                break
        
        # Normalize indentation
        for line in lines:
            if line.strip() == '':
                # Preserve empty lines
                result.append('')
                continue
                
            if uses_tabs:
                # Replace tabs with spaces
                indent_match = tab_pattern.match(line)
                if indent_match:
                    tab_count = len(indent_match.group(0))
                    new_indent = ' ' * (tab_count * spaces)
                    result.append(new_indent + line[tab_count:])
                else:
                    result.append(line)
            else:
                # Standardize space indentation
                indent_match = space_pattern.match(line)
                if indent_match:
                    current_spaces = len(indent_match.group(1))
                    indent_level = current_spaces // spaces if spaces > 0 else 0
                    new_indent = ' ' * (indent_level * spaces)
                    result.append(new_indent + line[current_spaces:])
                else:
                    result.append(line)
        
        return '\n'.join(result)
    
    def normalize_code_style(self, code: str) -> str:
        """Normalize code style (brackets, semicolons, etc).
        
        Args:
            code: The code to normalize.
            
        Returns:
            Code with normalized style.
        """
        # First, remove comments
        code_no_comments = re.sub(self.js_single_line_comment, '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(self.js_multi_line_comment, '', code_no_comments, flags=re.DOTALL)
        
        # Standardize curly brace style
        # Convert "function() {" to "function() {"
        code_no_comments = re.sub(r'(function\s*\([^)]*\))\s*\n\s*{', r'\1 {', code_no_comments)
        
        # Convert "if (condition) {" to "if (condition) {"
        code_no_comments = re.sub(r'(if\s*\([^)]*\))\s*\n\s*{', r'\1 {', code_no_comments)
        
        # Convert "} else {" to "} else {"
        code_no_comments = re.sub(r'}\s*\n\s*else\s*\n\s*{', r'} else {', code_no_comments)
        
        # Standardize spacing around operators
        # a=b to a = b
        code_no_comments = re.sub(r'(\w+)\s*([=+\-*/%])\s*(\w+)', r'\1 \2 \3', code_no_comments)
        
        # Ensure semicolons at end of statements
        # This is a simplistic approach - a full solution would need syntax parsing
        code_no_comments = re.sub(r'([\w)\]"\'`])\s*\n', r'\1;\n', code_no_comments)
        
        # Handle special case for existing semicolons
        code_no_comments = re.sub(r';\s*;\n', r';\n', code_no_comments)
        
        # Reintroduce original comments
        # This is a complex task that would require tracking comment positions
        # For simplicity, we'll just return the style-normalized code without comments
        
        return code_no_comments
    
    def normalize_d3_method_chains(self, code: str) -> str:
        """Normalize D3.js method chains for consistent formatting.
        
        Args:
            code: The code to normalize.
            
        Returns:
            Code with normalized D3 method chains.
        """
        # Match D3 method chains
        chain_pattern = r'd3\.([a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\))(\s*\.\s*[a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\))+' 
        
        def format_chain(match):
            chain = match.group(0)
            
            # Split the chain into method calls
            methods = re.findall(r'(?:d3\.([a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\)))|(?:\.([a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\)))', chain)
            
            # Format into a consistent style with proper indentation
            formatted = []
            for i, method in enumerate(methods):
                method_str = method[0] if method[0] else method[1]
                if i == 0:
                    # First method in chain (d3.something())
                    formatted.append(f"d3.{method_str}")
                else:
                    # Subsequent methods (.something())
                    formatted.append(f"  .{method_str}")
            
            return '\n'.join(formatted)
        
        # Replace each chain with its formatted version
        formatted_code = re.sub(chain_pattern, format_chain, code, flags=re.DOTALL)
        return formatted_code
    
    def normalize_code(self, code: str) -> str:
        """Apply all normalization steps to the code.
        
        Args:
            code: The code to normalize.
            
        Returns:
            Fully normalized code.
        """
        # Apply normalization steps in sequence
        cleaned = self.clean_code(code)
        indented = self._normalize_indentation(cleaned)
        formatted_chains = self.normalize_d3_method_chains(indented)
        
        return formatted_chains 