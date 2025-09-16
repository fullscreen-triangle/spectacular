"""
Mathematical Visualization Engine: Creates sophisticated mathematical visualizations.

This module provides advanced mathematical visualization capabilities for the
reasoning validation system, generating accurate mathematical plots, functions,
and data relationships.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import math
import re

logger = logging.getLogger(__name__)

@dataclass
class MathematicalFunction:
    """Represents a mathematical function with its properties."""
    expression: str                  # Mathematical expression (e.g., "x^2 + 2*x + 1")
    domain: Tuple[float, float]      # Function domain
    range: Tuple[float, float]       # Function range  
    function_type: str               # linear, quadratic, exponential, etc.
    parameters: Dict[str, float]     # Function parameters (slope, intercept, etc.)
    derivative: Optional[str]        # Derivative expression if calculable
    properties: Dict[str, Any]       # Function properties (monotonic, continuous, etc.)

@dataclass  
class MathVisualization:
    """Container for mathematical visualization results."""
    svg_content: str
    functions: List[MathematicalFunction]
    coordinate_system: Dict[str, Any]
    visualization_type: str
    mathematical_accuracy: float
    computational_metadata: Dict[str, Any]

class FunctionParser:
    """Parses and analyzes mathematical functions from text and data."""
    
    def __init__(self):
        self.function_patterns = {
            'linear': r'([+-]?[\d]*\.?\d*)\s*\*?\s*x\s*([+-]\s*[\d]+\.?\d*)?',
            'quadratic': r'([+-]?[\d]*\.?\d*)\s*\*?\s*x\^2\s*([+-]?[\d]*\.?\d*)\s*\*?\s*x?\s*([+-]?\s*[\d]+\.?\d*)?',
            'exponential': r'([+-]?[\d]*\.?\d*)\s*\*?\s*(e\^|exp\()\s*([+-]?[\d]*\.?\d*)\s*\*?\s*x',
            'logarithmic': r'([+-]?[\d]*\.?\d*)\s*\*?\s*(log|ln)\s*\(\s*([+-]?[\d]*\.?\d*)\s*\*?\s*x\s*\)',
            'power': r'([+-]?[\d]*\.?\d*)\s*\*?\s*x\^([+-]?[\d]+\.?\d*)',
            'trigonometric': r'([+-]?[\d]*\.?\d*)\s*\*?\s*(sin|cos|tan)\s*\(\s*([+-]?[\d]*\.?\d*)\s*\*?\s*x\s*\)'
        }
    
    async def parse_function_from_expression(self, expression: str) -> Optional[MathematicalFunction]:
        """Parse mathematical function from string expression."""
        
        expression = expression.replace(' ', '')
        
        for func_type, pattern in self.function_patterns.items():
            match = re.search(pattern, expression, re.IGNORECASE)
            if match:
                return await self._create_function_from_match(func_type, match.groups(), expression)
        
        return None
    
    async def infer_function_from_data(self, x_data: List[float], y_data: List[float]) -> Optional[MathematicalFunction]:
        """Infer mathematical function from data points."""
        
        if len(x_data) != len(y_data) or len(x_data) < 3:
            return None
        
        try:
            # Try linear regression first
            linear_func = await self._fit_linear_function(x_data, y_data)
            if linear_func and linear_func.properties.get('r_squared', 0) > 0.8:
                return linear_func
            
            # Try polynomial regression
            poly_func = await self._fit_polynomial_function(x_data, y_data, degree=2)
            if poly_func and poly_func.properties.get('r_squared', 0) > 0.8:
                return poly_func
            
            # Try exponential fitting
            exp_func = await self._fit_exponential_function(x_data, y_data)
            if exp_func and exp_func.properties.get('r_squared', 0) > 0.8:
                return exp_func
            
            # Return the best linear fit as fallback
            return linear_func
            
        except Exception as e:
            logger.error("Error fitting function to data: %s", str(e))
            return None
    
    async def _create_function_from_match(self, func_type: str, groups: Tuple, expression: str) -> MathematicalFunction:
        """Create function object from regex match groups."""
        
        parameters = {}
        
        if func_type == 'linear':
            slope = float(groups[0]) if groups[0] and groups[0] != '' else 1.0
            intercept = float(groups[1].replace(' ', '')) if groups[1] else 0.0
            parameters = {'slope': slope, 'intercept': intercept}
            expression_clean = f"{slope}*x + {intercept}"
            derivative = str(slope)
            
        elif func_type == 'quadratic':
            a = float(groups[0]) if groups[0] and groups[0] != '' else 1.0
            b = float(groups[1].replace(' ', '')) if groups[1] else 0.0
            c = float(groups[2].replace(' ', '')) if groups[2] else 0.0
            parameters = {'a': a, 'b': b, 'c': c}
            expression_clean = f"{a}*x^2 + {b}*x + {c}"
            derivative = f"{2*a}*x + {b}"
            
        elif func_type == 'exponential':
            coeff = float(groups[0]) if groups[0] and groups[0] != '' else 1.0
            exp_coeff = float(groups[2]) if groups[2] else 1.0
            parameters = {'coefficient': coeff, 'exponent': exp_coeff}
            expression_clean = f"{coeff}*exp({exp_coeff}*x)"
            derivative = f"{coeff*exp_coeff}*exp({exp_coeff}*x)"
            
        else:
            parameters = {'coefficient': 1.0}
            expression_clean = expression
            derivative = None
        
        # Estimate domain and range (simplified)
        domain = (-10.0, 10.0)  # Default domain
        range_val = await self._estimate_function_range(func_type, parameters, domain)
        
        # Calculate function properties
        properties = await self._analyze_function_properties(func_type, parameters)
        
        return MathematicalFunction(
            expression=expression_clean,
            domain=domain,
            range=range_val,
            function_type=func_type,
            parameters=parameters,
            derivative=derivative,
            properties=properties
        )
    
    async def _fit_linear_function(self, x_data: List[float], y_data: List[float]) -> Optional[MathematicalFunction]:
        """Fit linear function to data points."""
        
        try:
            x_arr = np.array(x_data)
            y_arr = np.array(y_data)
            
            # Linear regression
            A = np.vstack([x_arr, np.ones(len(x_arr))]).T
            slope, intercept = np.linalg.lstsq(A, y_arr, rcond=None)[0]
            
            # Calculate R-squared
            y_pred = slope * x_arr + intercept
            ss_res = np.sum((y_arr - y_pred) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            domain = (min(x_data), max(x_data))
            range_val = (min(y_data), max(y_data))
            
            parameters = {'slope': float(slope), 'intercept': float(intercept)}
            properties = {
                'r_squared': r_squared,
                'monotonic': slope != 0,
                'continuous': True,
                'differentiable': True
            }
            
            return MathematicalFunction(
                expression=f"{slope:.3f}*x + {intercept:.3f}",
                domain=domain,
                range=range_val,
                function_type='linear',
                parameters=parameters,
                derivative=str(slope),
                properties=properties
            )
            
        except Exception as e:
            logger.error("Error fitting linear function: %s", str(e))
            return None
    
    async def _fit_polynomial_function(self, x_data: List[float], y_data: List[float], degree: int) -> Optional[MathematicalFunction]:
        """Fit polynomial function to data points."""
        
        try:
            coefficients = np.polyfit(x_data, y_data, degree)
            
            # Calculate R-squared
            y_pred = np.polyval(coefficients, x_data)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Create expression
            if degree == 2:
                a, b, c = coefficients
                expression = f"{a:.3f}*x^2 + {b:.3f}*x + {c:.3f}"
                derivative = f"{2*a:.3f}*x + {b:.3f}"
                parameters = {'a': float(a), 'b': float(b), 'c': float(c)}
                func_type = 'quadratic'
            else:
                expression = " + ".join([f"{coeff:.3f}*x^{degree-i}" for i, coeff in enumerate(coefficients)])
                derivative = None  # Complex for higher degrees
                parameters = {f'coeff_{i}': float(coeff) for i, coeff in enumerate(coefficients)}
                func_type = f'polynomial_degree_{degree}'
            
            domain = (min(x_data), max(x_data))
            range_val = (min(y_data), max(y_data))
            
            properties = {
                'r_squared': r_squared,
                'continuous': True,
                'differentiable': True,
                'degree': degree
            }
            
            return MathematicalFunction(
                expression=expression,
                domain=domain,
                range=range_val,
                function_type=func_type,
                parameters=parameters,
                derivative=derivative,
                properties=properties
            )
            
        except Exception as e:
            logger.error("Error fitting polynomial function: %s", str(e))
            return None
    
    async def _fit_exponential_function(self, x_data: List[float], y_data: List[float]) -> Optional[MathematicalFunction]:
        """Fit exponential function to data points."""
        
        try:
            # Check if all y values are positive (required for log transformation)
            if any(y <= 0 for y in y_data):
                return None
            
            # Transform to linear: ln(y) = ln(a) + b*x
            log_y = np.log(y_data)
            
            # Linear regression on transformed data
            A = np.vstack([x_data, np.ones(len(x_data))]).T
            b, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]
            a = np.exp(log_a)
            
            # Calculate R-squared on original scale
            y_pred = a * np.exp(b * np.array(x_data))
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            domain = (min(x_data), max(x_data))
            range_val = (min(y_data), max(y_data))
            
            parameters = {'coefficient': float(a), 'exponent': float(b)}
            properties = {
                'r_squared': r_squared,
                'monotonic': b != 0,
                'continuous': True,
                'differentiable': True
            }
            
            return MathematicalFunction(
                expression=f"{a:.3f}*exp({b:.3f}*x)",
                domain=domain,
                range=range_val,
                function_type='exponential',
                parameters=parameters,
                derivative=f"{a*b:.3f}*exp({b:.3f}*x)",
                properties=properties
            )
            
        except Exception as e:
            logger.error("Error fitting exponential function: %s", str(e))
            return None
    
    async def _estimate_function_range(self, func_type: str, parameters: Dict[str, float], domain: Tuple[float, float]) -> Tuple[float, float]:
        """Estimate function range given type, parameters, and domain."""
        
        x_min, x_max = domain
        
        try:
            if func_type == 'linear':
                slope = parameters.get('slope', 1)
                intercept = parameters.get('intercept', 0)
                y_min = slope * x_min + intercept
                y_max = slope * x_max + intercept
                if y_min > y_max:
                    y_min, y_max = y_max, y_min
                
            elif func_type == 'quadratic':
                a = parameters.get('a', 1)
                b = parameters.get('b', 0)
                c = parameters.get('c', 0)
                
                # Find vertex
                vertex_x = -b / (2 * a) if a != 0 else 0
                vertex_y = a * vertex_x**2 + b * vertex_x + c
                
                # Evaluate at domain endpoints
                y_min_boundary = a * x_min**2 + b * x_min + c
                y_max_boundary = a * x_max**2 + b * x_max + c
                
                if x_min <= vertex_x <= x_max:
                    if a > 0:  # Parabola opens upward
                        y_min = vertex_y
                        y_max = max(y_min_boundary, y_max_boundary)
                    else:  # Parabola opens downward
                        y_max = vertex_y
                        y_min = min(y_min_boundary, y_max_boundary)
                else:
                    y_min = min(y_min_boundary, y_max_boundary)
                    y_max = max(y_min_boundary, y_max_boundary)
                
            elif func_type == 'exponential':
                coeff = parameters.get('coefficient', 1)
                exp_coeff = parameters.get('exponent', 1)
                
                y_min_val = coeff * math.exp(exp_coeff * x_min)
                y_max_val = coeff * math.exp(exp_coeff * x_max)
                y_min = min(y_min_val, y_max_val)
                y_max = max(y_min_val, y_max_val)
                
            else:
                # Default range estimation
                y_min = -10.0
                y_max = 10.0
            
            return (y_min, y_max)
            
        except Exception as e:
            logger.error("Error estimating function range: %s", str(e))
            return (-10.0, 10.0)  # Default range
    
    async def _analyze_function_properties(self, func_type: str, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Analyze mathematical properties of the function."""
        
        properties = {
            'continuous': True,      # Most functions we handle are continuous
            'differentiable': True,  # Most functions we handle are differentiable
        }
        
        if func_type == 'linear':
            slope = parameters.get('slope', 1)
            properties.update({
                'monotonic': slope != 0,
                'increasing': slope > 0,
                'decreasing': slope < 0,
                'bounded': False,
                'periodic': False
            })
            
        elif func_type == 'quadratic':
            a = parameters.get('a', 1)
            properties.update({
                'monotonic': False,  # Quadratics are not monotonic over entire domain
                'bounded_above': a < 0,
                'bounded_below': a > 0,
                'periodic': False,
                'has_extremum': True
            })
            
        elif func_type == 'exponential':
            exp_coeff = parameters.get('exponent', 1)
            properties.update({
                'monotonic': exp_coeff != 0,
                'increasing': exp_coeff > 0,
                'decreasing': exp_coeff < 0,
                'bounded_below': True,  # Exponentials are bounded below by 0
                'periodic': False
            })
            
        elif func_type == 'trigonometric':
            properties.update({
                'periodic': True,
                'bounded': True,
                'monotonic': False
            })
        
        return properties

class MathVisualizationEngine:
    """
    Creates sophisticated mathematical visualizations with high accuracy.
    
    Generates SVG visualizations for mathematical functions, data relationships,
    and mathematical concepts with precise mathematical rendering.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mathematical visualization engine."""
        
        self.config = config or {}
        self.function_parser = FunctionParser()
        
        # Visualization parameters
        self.default_width = config.get('width', 400)
        self.default_height = config.get('height', 300)
        self.margin = config.get('margin', 40)
        self.grid_enabled = config.get('grid_enabled', True)
        self.axis_labels = config.get('axis_labels', True)
        
        # Mathematical accuracy settings
        self.function_resolution = config.get('function_resolution', 200)  # Points per function
        self.numerical_precision = config.get('numerical_precision', 1e-10)
        
        logger.info("Mathematical Visualization Engine initialized")
    
    async def create_function_visualization(
        self,
        functions: List[MathematicalFunction],
        title: str = "Mathematical Function",
        domain: Optional[Tuple[float, float]] = None,
        range_override: Optional[Tuple[float, float]] = None
    ) -> MathVisualization:
        """
        Create visualization of mathematical functions.
        
        Args:
            functions: List of mathematical functions to visualize
            title: Title for the visualization
            domain: X-axis domain override
            range_override: Y-axis range override
            
        Returns:
            MathVisualization with SVG content and metadata
        """
        
        logger.info("Creating function visualization for %d functions", len(functions))
        
        try:
            if not functions:
                return await self._create_empty_visualization("No functions provided")
            
            # Determine overall domain and range
            overall_domain = domain or self._calculate_overall_domain(functions)
            overall_range = range_override or self._calculate_overall_range(functions, overall_domain)
            
            # Create coordinate system
            coord_system = self._create_coordinate_system(overall_domain, overall_range)
            
            # Generate SVG content
            svg_content = await self._generate_function_svg(
                functions, title, coord_system, overall_domain, overall_range
            )
            
            # Calculate mathematical accuracy
            accuracy = await self._calculate_mathematical_accuracy(functions, coord_system)
            
            # Create computational metadata
            metadata = {
                'function_count': len(functions),
                'domain': overall_domain,
                'range': overall_range,
                'resolution': self.function_resolution,
                'coordinate_system': coord_system,
                'generation_timestamp': datetime.now().isoformat()
            }
            
            return MathVisualization(
                svg_content=svg_content,
                functions=functions,
                coordinate_system=coord_system,
                visualization_type='function_plot',
                mathematical_accuracy=accuracy,
                computational_metadata=metadata
            )
            
        except Exception as e:
            logger.error("Error creating function visualization: %s", str(e))
            return await self._create_empty_visualization(f"Error: {str(e)}")
    
    async def create_data_visualization(
        self,
        x_data: List[float],
        y_data: List[float], 
        title: str = "Data Visualization",
        fit_function: bool = True
    ) -> MathVisualization:
        """
        Create visualization from data points with optional function fitting.
        
        Args:
            x_data: X-coordinate data points
            y_data: Y-coordinate data points  
            title: Title for the visualization
            fit_function: Whether to fit and display a function
            
        Returns:
            MathVisualization with data points and fitted function if requested
        """
        
        logger.info("Creating data visualization with %d points", len(x_data))
        
        try:
            functions = []
            
            # Fit function to data if requested
            if fit_function and len(x_data) >= 3:
                fitted_function = await self.function_parser.infer_function_from_data(x_data, y_data)
                if fitted_function:
                    functions.append(fitted_function)
            
            # Calculate domain and range from data
            domain = (min(x_data), max(x_data))
            y_range = (min(y_data), max(y_data))
            
            # Add padding to range
            y_padding = (y_range[1] - y_range[0]) * 0.1
            overall_range = (y_range[0] - y_padding, y_range[1] + y_padding)
            
            # Create coordinate system
            coord_system = self._create_coordinate_system(domain, overall_range)
            
            # Generate SVG content with data points
            svg_content = await self._generate_data_svg(
                x_data, y_data, functions, title, coord_system, domain, overall_range
            )
            
            # Calculate accuracy
            accuracy = await self._calculate_data_accuracy(x_data, y_data, functions)
            
            # Create metadata
            metadata = {
                'data_point_count': len(x_data),
                'fitted_functions': len(functions),
                'domain': domain,
                'range': overall_range,
                'coordinate_system': coord_system,
                'generation_timestamp': datetime.now().isoformat()
            }
            
            return MathVisualization(
                svg_content=svg_content,
                functions=functions,
                coordinate_system=coord_system,
                visualization_type='data_plot',
                mathematical_accuracy=accuracy,
                computational_metadata=metadata
            )
            
        except Exception as e:
            logger.error("Error creating data visualization: %s", str(e))
            return await self._create_empty_visualization(f"Error: {str(e)}")
    
    async def _generate_function_svg(
        self,
        functions: List[MathematicalFunction],
        title: str,
        coord_system: Dict[str, Any],
        domain: Tuple[float, float],
        range_val: Tuple[float, float]
    ) -> str:
        """Generate SVG content for function visualization."""
        
        width = self.default_width
        height = self.default_height
        margin = self.margin
        
        # SVG header
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="#fafafa" stroke="#333" stroke-width="1"/>',
            f'<text x="{width/2}" y="20" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold">{title}</text>'
        ]
        
        # Coordinate transformation functions
        x_scale = (width - 2 * margin) / (domain[1] - domain[0])
        y_scale = (height - 2 * margin) / (range_val[1] - range_val[0])
        
        def transform_x(x): return margin + (x - domain[0]) * x_scale
        def transform_y(y): return height - margin - (y - range_val[0]) * y_scale
        
        # Draw coordinate axes
        svg_parts.extend(await self._draw_axes(transform_x, transform_y, domain, range_val, coord_system))
        
        # Draw grid if enabled
        if self.grid_enabled:
            svg_parts.extend(await self._draw_grid(transform_x, transform_y, domain, range_val))
        
        # Draw functions
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, function in enumerate(functions):
            color = colors[i % len(colors)]
            function_path = await self._generate_function_path(function, domain, transform_x, transform_y)
            
            if function_path:
                svg_parts.append(f'<path d="{function_path}" stroke="{color}" stroke-width="2" fill="none"/>')
                
                # Add function label
                label_x = width - 120
                label_y = 50 + i * 20
                svg_parts.append(f'<text x="{label_x}" y="{label_y}" font-family="Arial" font-size="10" fill="{color}">{function.expression[:30]}</text>')
        
        svg_parts.append('</svg>')
        return ''.join(svg_parts)
    
    async def _generate_data_svg(
        self,
        x_data: List[float],
        y_data: List[float], 
        functions: List[MathematicalFunction],
        title: str,
        coord_system: Dict[str, Any],
        domain: Tuple[float, float],
        range_val: Tuple[float, float]
    ) -> str:
        """Generate SVG content for data visualization."""
        
        width = self.default_width
        height = self.default_height
        margin = self.margin
        
        # SVG header
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="#fafafa" stroke="#333" stroke-width="1"/>',
            f'<text x="{width/2}" y="20" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold">{title}</text>'
        ]
        
        # Coordinate transformation
        x_scale = (width - 2 * margin) / (domain[1] - domain[0])
        y_scale = (height - 2 * margin) / (range_val[1] - range_val[0])
        
        def transform_x(x): return margin + (x - domain[0]) * x_scale
        def transform_y(y): return height - margin - (y - range_val[0]) * y_scale
        
        # Draw coordinate axes
        svg_parts.extend(await self._draw_axes(transform_x, transform_y, domain, range_val, coord_system))
        
        # Draw data points
        for x, y in zip(x_data, y_data):
            svg_x = transform_x(x)
            svg_y = transform_y(y)
            svg_parts.append(f'<circle cx="{svg_x}" cy="{svg_y}" r="3" fill="#e74c3c" stroke="#c0392b" stroke-width="1"/>')
        
        # Draw fitted functions if any
        for i, function in enumerate(functions):
            color = '#3498db'
            function_path = await self._generate_function_path(function, domain, transform_x, transform_y)
            
            if function_path:
                svg_parts.append(f'<path d="{function_path}" stroke="{color}" stroke-width="2" fill="none"/>')
                
                # Add R-squared if available
                r_squared = function.properties.get('r_squared', 0)
                svg_parts.append(f'<text x="{width-120}" y="50" font-family="Arial" font-size="10" fill="{color}">R² = {r_squared:.3f}</text>')
                svg_parts.append(f'<text x="{width-120}" y="65" font-family="Arial" font-size="10" fill="{color}">{function.expression[:30]}</text>')
        
        svg_parts.append('</svg>')
        return ''.join(svg_parts)
    
    async def _generate_function_path(
        self,
        function: MathematicalFunction,
        domain: Tuple[float, float],
        transform_x: Callable,
        transform_y: Callable
    ) -> str:
        """Generate SVG path for a mathematical function."""
        
        try:
            # Generate function points
            x_points = np.linspace(domain[0], domain[1], self.function_resolution)
            y_points = []
            
            for x in x_points:
                y = await self._evaluate_function(function, x)
                if y is not None and not math.isnan(y) and not math.isinf(y):
                    y_points.append(y)
                else:
                    y_points.append(None)  # Mark discontinuities
            
            # Build path string
            path_parts = []
            first_point = True
            
            for x, y in zip(x_points, y_points):
                if y is not None:
                    svg_x = transform_x(x)
                    svg_y = transform_y(y)
                    
                    if first_point:
                        path_parts.append(f'M {svg_x} {svg_y}')
                        first_point = False
                    else:
                        path_parts.append(f'L {svg_x} {svg_y}')
                else:
                    first_point = True  # Start new path segment after discontinuity
            
            return ' '.join(path_parts)
            
        except Exception as e:
            logger.error("Error generating function path: %s", str(e))
            return ""
    
    async def _evaluate_function(self, function: MathematicalFunction, x: float) -> Optional[float]:
        """Evaluate mathematical function at given x value."""
        
        try:
            if function.function_type == 'linear':
                slope = function.parameters['slope']
                intercept = function.parameters['intercept']
                return slope * x + intercept
            
            elif function.function_type == 'quadratic':
                a = function.parameters['a']
                b = function.parameters['b']
                c = function.parameters['c']
                return a * x**2 + b * x + c
            
            elif function.function_type == 'exponential':
                coeff = function.parameters['coefficient']
                exp_coeff = function.parameters['exponent']
                return coeff * math.exp(exp_coeff * x)
            
            else:
                # For other function types, would need more sophisticated evaluation
                return None
                
        except (OverflowError, ValueError, KeyError):
            return None
    
    # Helper methods for coordinate system, axes, grid, etc.
    
    def _create_coordinate_system(self, domain: Tuple[float, float], range_val: Tuple[float, float]) -> Dict[str, Any]:
        """Create coordinate system specification."""
        return {
            'type': 'cartesian',
            'domain': domain,
            'range': range_val,
            'x_axis': {'min': domain[0], 'max': domain[1]},
            'y_axis': {'min': range_val[0], 'max': range_val[1]},
            'grid_enabled': self.grid_enabled,
            'axis_labels': self.axis_labels
        }
    
    def _calculate_overall_domain(self, functions: List[MathematicalFunction]) -> Tuple[float, float]:
        """Calculate overall domain from all functions."""
        if not functions:
            return (-10.0, 10.0)
        
        min_domain = min(func.domain[0] for func in functions)
        max_domain = max(func.domain[1] for func in functions)
        
        return (min_domain, max_domain)
    
    def _calculate_overall_range(self, functions: List[MathematicalFunction], domain: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate overall range from all functions."""
        if not functions:
            return (-10.0, 10.0)
        
        min_range = min(func.range[0] for func in functions)
        max_range = max(func.range[1] for func in functions)
        
        # Add padding
        range_padding = (max_range - min_range) * 0.1
        return (min_range - range_padding, max_range + range_padding)
    
    async def _draw_axes(self, transform_x, transform_y, domain, range_val, coord_system):
        """Draw coordinate axes."""
        
        axes = []
        
        # X-axis
        y_zero = transform_y(0) if range_val[0] <= 0 <= range_val[1] else transform_y(range_val[0])
        x_start = transform_x(domain[0])
        x_end = transform_x(domain[1])
        axes.append(f'<line x1="{x_start}" y1="{y_zero}" x2="{x_end}" y2="{y_zero}" stroke="#333" stroke-width="1"/>')
        
        # Y-axis
        x_zero = transform_x(0) if domain[0] <= 0 <= domain[1] else transform_x(domain[0])
        y_start = transform_y(range_val[0])
        y_end = transform_y(range_val[1])
        axes.append(f'<line x1="{x_zero}" y1="{y_start}" x2="{x_zero}" y2="{y_end}" stroke="#333" stroke-width="1"/>')
        
        return axes
    
    async def _draw_grid(self, transform_x, transform_y, domain, range_val):
        """Draw coordinate grid."""
        
        grid = []
        
        # Vertical grid lines
        x_step = (domain[1] - domain[0]) / 10
        for i in range(11):
            x = domain[0] + i * x_step
            svg_x = transform_x(x)
            y_start = transform_y(range_val[0])
            y_end = transform_y(range_val[1])
            grid.append(f'<line x1="{svg_x}" y1="{y_start}" x2="{svg_x}" y2="{y_end}" stroke="#ddd" stroke-width="0.5"/>')
        
        # Horizontal grid lines
        y_step = (range_val[1] - range_val[0]) / 10
        for i in range(11):
            y = range_val[0] + i * y_step
            svg_y = transform_y(y)
            x_start = transform_x(domain[0])
            x_end = transform_x(domain[1])
            grid.append(f'<line x1="{x_start}" y1="{svg_y}" x2="{x_end}" y2="{svg_y}" stroke="#ddd" stroke-width="0.5"/>')
        
        return grid
    
    async def _calculate_mathematical_accuracy(self, functions, coord_system):
        """Calculate mathematical accuracy score."""
        if not functions:
            return 0.0
        
        # Simple accuracy based on function properties
        total_accuracy = 0
        for func in functions:
            r_squared = func.properties.get('r_squared', 0.5)
            total_accuracy += r_squared
        
        return total_accuracy / len(functions)
    
    async def _calculate_data_accuracy(self, x_data, y_data, functions):
        """Calculate accuracy for data visualization."""
        if not functions:
            return 0.8  # Base accuracy for data visualization
        
        # Use R-squared from fitted function
        return functions[0].properties.get('r_squared', 0.8)
    
    async def _create_empty_visualization(self, message):
        """Create empty visualization for error cases."""
        
        svg_content = f'''<svg width="{self.default_width}" height="{self.default_height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="{self.default_width}" height="{self.default_height}" fill="#f8f8f8" stroke="#ccc"/>
            <text x="{self.default_width/2}" y="{self.default_height/2}" text-anchor="middle" font-family="Arial" font-size="14" fill="#666">
                {message}
            </text>
        </svg>'''
        
        return MathVisualization(
            svg_content=svg_content,
            functions=[],
            coordinate_system={},
            visualization_type='empty',
            mathematical_accuracy=0.0,
            computational_metadata={'error': message}
        )

# Alias for convenience
MathVisualizer = MathVisualizationEngine
