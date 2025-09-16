"""
Visual Embeddings: Higher-dimensional representation of visual information.

This module creates embeddings from visual content that encode significantly more
information than traditional text embeddings, supporting geometric relationships,
mathematical patterns, and spatial reasoning.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import re
import json
from enum import Enum

logger = logging.getLogger(__name__)

class EmbeddingDimension(Enum):
    """Dimensions for visual embedding representation."""
    GEOMETRIC = "geometric"              # Shapes, angles, proportions
    SPATIAL = "spatial"                  # Position, orientation, scaling
    TOPOLOGICAL = "topological"         # Connectivity, continuity, boundaries
    CHROMATIC = "chromatic"              # Color, saturation, contrast
    TEXTURAL = "textural"                # Pattern density, roughness, regularity
    TEMPORAL = "temporal"                # Animation, sequence, transitions
    MATHEMATICAL = "mathematical"        # Functions, relationships, equations
    SEMANTIC = "semantic"                # Meaning, labels, annotations
    COMPOSITIONAL = "compositional"      # Layout, arrangement, hierarchy
    INTERACTIVE = "interactive"          # User elements, controls, feedback

@dataclass
class VisualEmbedding:
    """Multi-dimensional visual embedding with rich spatial-temporal information."""
    
    # Core embedding vectors
    geometric_features: np.ndarray       # Shape, angle, proportion features
    spatial_features: np.ndarray         # Position, orientation, scale features
    topological_features: np.ndarray     # Connectivity and boundary features
    chromatic_features: np.ndarray       # Color and visual style features
    mathematical_features: np.ndarray    # Mathematical relationship features
    
    # Metadata and context
    embedding_id: str
    source_content: str                  # Original SVG or visual content
    content_type: str                    # svg, canvas, plot, etc.
    extraction_method: str               # How features were extracted
    confidence_scores: Dict[str, float]  # Confidence in each dimension
    temporal_sequence: Optional[int]     # Position in sequence if applicable
    
    # Relationships and context
    spatial_relationships: Dict[str, Any]    # Relationships to other visual elements
    mathematical_context: Dict[str, Any]     # Mathematical interpretation context
    semantic_annotations: List[str]          # Human-readable descriptions
    
    def __post_init__(self):
        """Validate and normalize embedding vectors."""
        # Ensure all feature vectors are numpy arrays
        for field_name in ['geometric_features', 'spatial_features', 'topological_features', 
                          'chromatic_features', 'mathematical_features']:
            field_value = getattr(self, field_name)
            if not isinstance(field_value, np.ndarray):
                setattr(self, field_name, np.array(field_value, dtype=float))
    
    def get_combined_embedding(self) -> np.ndarray:
        """Get combined embedding vector from all dimensions."""
        return np.concatenate([
            self.geometric_features,
            self.spatial_features, 
            self.topological_features,
            self.chromatic_features,
            self.mathematical_features
        ])
    
    def calculate_similarity(self, other: 'VisualEmbedding') -> float:
        """Calculate similarity with another visual embedding."""
        if not isinstance(other, VisualEmbedding):
            return 0.0
        
        # Get combined embeddings
        embedding1 = self.get_combined_embedding()
        embedding2 = other.get_combined_embedding()
        
        # Ensure same dimensionality
        min_dim = min(len(embedding1), len(embedding2))
        embedding1 = embedding1[:min_dim]
        embedding2 = embedding2[:min_dim]
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))
    
    def get_dimension_summary(self) -> Dict[str, Any]:
        """Get summary of embedding dimensions and their characteristics."""
        return {
            'geometric_dimensionality': len(self.geometric_features),
            'spatial_dimensionality': len(self.spatial_features),
            'topological_dimensionality': len(self.topological_features),
            'chromatic_dimensionality': len(self.chromatic_features),
            'mathematical_dimensionality': len(self.mathematical_features),
            'total_dimensionality': len(self.get_combined_embedding()),
            'confidence_scores': self.confidence_scores,
            'extraction_method': self.extraction_method,
            'content_type': self.content_type
        }

class GeometricFeatureExtractor:
    """Extracts geometric features from visual content."""
    
    def __init__(self):
        self.shape_patterns = {
            'circle': r'<circle|<ellipse|border-radius:\s*50%',
            'rectangle': r'<rect|<div[^>]*style[^>]*width|rectangle',
            'line': r'<line|<path[^>]*d[^>]*[ML]|stroke-width',
            'polygon': r'<polygon|<path[^>]*d[^>]*[zZ]',
            'curve': r'<path[^>]*d[^>]*[QqCc]|curve|bezier',
            'text': r'<text|font-|<span|<p'
        }
    
    async def extract_geometric_features(self, content: str) -> np.ndarray:
        """Extract geometric features from visual content."""
        
        features = []
        
        # Shape complexity analysis
        for shape_name, pattern in self.shape_patterns.items():
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            features.append(matches)
        
        # Size and proportion analysis
        dimensions = self._extract_dimensions(content)
        features.extend([
            dimensions.get('width', 0),
            dimensions.get('height', 0), 
            dimensions.get('aspect_ratio', 1.0)
        ])
        
        # Geometric complexity measures
        features.extend([
            content.count('<path'),           # Path complexity
            content.count('transform'),       # Transformation complexity
            content.count('stroke'),          # Stroke elements
            content.count('fill'),            # Fill elements
            self._calculate_coordinate_variance(content)  # Coordinate spread
        ])
        
        return np.array(features, dtype=float)
    
    def _extract_dimensions(self, content: str) -> Dict[str, float]:
        """Extract dimensional information from content."""
        
        width_match = re.search(r'width["\s]*[:=]["\s]*(\d+)', content)
        height_match = re.search(r'height["\s]*[:=]["\s]*(\d+)', content)
        
        width = float(width_match.group(1)) if width_match else 400.0
        height = float(height_match.group(1)) if height_match else 300.0
        
        aspect_ratio = width / height if height > 0 else 1.0
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio
        }
    
    def _calculate_coordinate_variance(self, content: str) -> float:
        """Calculate variance in coordinate usage (measure of spatial spread)."""
        
        # Extract numeric coordinates from the content
        coordinates = re.findall(r'[xy][12]?["\s]*[:=]["\s]*([0-9.-]+)', content, re.IGNORECASE)
        coordinates.extend(re.findall(r'[ML]\s*([0-9.-]+)[,\s]+([0-9.-]+)', content))
        
        if not coordinates:
            return 0.0
        
        # Flatten coordinate list and convert to numbers
        flat_coords = []
        for coord in coordinates:
            if isinstance(coord, tuple):
                flat_coords.extend(coord)
            else:
                flat_coords.append(coord)
        
        try:
            numeric_coords = [float(c) for c in flat_coords]
            return np.var(numeric_coords) if numeric_coords else 0.0
        except (ValueError, TypeError):
            return 0.0

class SpatialFeatureExtractor:
    """Extracts spatial relationship features from visual content."""
    
    async def extract_spatial_features(self, content: str) -> np.ndarray:
        """Extract spatial positioning and relationship features."""
        
        features = []
        
        # Positioning analysis
        positioning_data = self._analyze_positioning(content)
        features.extend([
            positioning_data['absolute_positions'],
            positioning_data['relative_positions'], 
            positioning_data['centered_elements'],
            positioning_data['margin_usage']
        ])
        
        # Grouping and hierarchy analysis
        grouping_data = self._analyze_grouping(content)
        features.extend([
            grouping_data['nested_groups'],
            grouping_data['sibling_elements'],
            grouping_data['hierarchical_depth']
        ])
        
        # Spatial distribution measures
        distribution_data = self._analyze_spatial_distribution(content)
        features.extend([
            distribution_data['clustering_coefficient'],
            distribution_data['spatial_entropy'],
            distribution_data['boundary_utilization']
        ])
        
        # Alignment and grid analysis
        alignment_data = self._analyze_alignment(content)
        features.extend([
            alignment_data['horizontal_alignment'],
            alignment_data['vertical_alignment'],
            alignment_data['grid_regularity']
        ])
        
        return np.array(features, dtype=float)
    
    def _analyze_positioning(self, content: str) -> Dict[str, float]:
        """Analyze positioning patterns in the content."""
        
        # Count different positioning methods
        absolute_positions = len(re.findall(r'[xy]["\s]*[:=]["\s]*\d+', content, re.IGNORECASE))
        relative_positions = len(re.findall(r'translate|relative|%', content, re.IGNORECASE))
        centered_elements = len(re.findall(r'text-anchor["\s]*[:=]["\s]*["\']?middle|center', content, re.IGNORECASE))
        margin_usage = content.count('margin') + content.count('padding')
        
        return {
            'absolute_positions': absolute_positions,
            'relative_positions': relative_positions,
            'centered_elements': centered_elements,
            'margin_usage': margin_usage
        }
    
    def _analyze_grouping(self, content: str) -> Dict[str, float]:
        """Analyze grouping and hierarchical structure."""
        
        nested_groups = content.count('<g') + content.count('<div')
        sibling_elements = len(re.findall(r'<(\w+)[^>]*>', content)) - nested_groups
        
        # Calculate hierarchical depth (simplified)
        max_nesting = 0
        current_nesting = 0
        for char in content:
            if char == '<':
                # Check if it's an opening tag
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == '>':
                current_nesting = max(0, current_nesting - 1)
        
        return {
            'nested_groups': nested_groups,
            'sibling_elements': sibling_elements,
            'hierarchical_depth': max_nesting
        }
    
    def _analyze_spatial_distribution(self, content: str) -> Dict[str, float]:
        """Analyze spatial distribution of elements."""
        
        # Extract coordinates for distribution analysis
        coordinates = []
        coord_matches = re.findall(r'[xy][12]?["\s]*[:=]["\s]*([0-9.-]+)', content, re.IGNORECASE)
        
        for match in coord_matches:
            try:
                coordinates.append(float(match))
            except ValueError:
                pass
        
        if not coordinates:
            return {
                'clustering_coefficient': 0.0,
                'spatial_entropy': 0.0,
                'boundary_utilization': 0.0
            }
        
        # Simple clustering coefficient (variance-based)
        coord_variance = np.var(coordinates)
        clustering_coefficient = 1.0 / (1.0 + coord_variance) if coord_variance > 0 else 1.0
        
        # Spatial entropy (distribution uniformity)
        coord_range = max(coordinates) - min(coordinates) if len(coordinates) > 1 else 0
        spatial_entropy = coord_range / 1000.0 if coord_range > 0 else 0.0  # Normalize
        
        # Boundary utilization (how much of space is used)
        boundary_utilization = min(1.0, len(coordinates) / 50.0)  # Normalize by expected element count
        
        return {
            'clustering_coefficient': clustering_coefficient,
            'spatial_entropy': min(1.0, spatial_entropy),
            'boundary_utilization': boundary_utilization
        }
    
    def _analyze_alignment(self, content: str) -> Dict[str, float]:
        """Analyze alignment and grid patterns."""
        
        # Look for alignment indicators
        horizontal_alignment = len(re.findall(r'align.*center|text-align|justify', content, re.IGNORECASE))
        vertical_alignment = len(re.findall(r'vertical-align|middle|baseline', content, re.IGNORECASE))
        
        # Look for grid-like patterns
        grid_indicators = content.count('grid') + content.count('flex') + content.count('column')
        grid_regularity = min(1.0, grid_indicators / 5.0)  # Normalize
        
        return {
            'horizontal_alignment': horizontal_alignment,
            'vertical_alignment': vertical_alignment,
            'grid_regularity': grid_regularity
        }

class MathematicalFeatureExtractor:
    """Extracts mathematical relationship features from visual content."""
    
    def __init__(self):
        self.math_patterns = {
            'linear': r'[Ll]inear|y\s*=\s*[^*]*x|straight.*line',
            'exponential': r'[Ee]xp|exponential|e\^|growth',
            'logarithmic': r'[Ll]og|logarithm|ln',
            'polynomial': r'polynomial|x\^|degree',
            'trigonometric': r'sin|cos|tan|periodic|wave',
            'statistical': r'mean|average|distribution|correlation|regression'
        }
    
    async def extract_mathematical_features(self, content: str) -> np.ndarray:
        """Extract mathematical relationship features."""
        
        features = []
        
        # Mathematical pattern recognition
        for pattern_name, pattern in self.math_patterns.items():
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            features.append(matches)
        
        # Equation and formula analysis
        equations = self._extract_equations(content)
        features.extend([
            len(equations),                    # Number of equations
            self._calculate_equation_complexity(equations),  # Average complexity
            self._count_variables(content),    # Variable usage
            self._count_operators(content)     # Mathematical operators
        ])
        
        # Numerical analysis
        numbers = self._extract_numbers(content)
        if numbers:
            features.extend([
                len(numbers),                  # Count of numbers
                np.mean(numbers),             # Mean value
                np.std(numbers),              # Standard deviation
                max(numbers) - min(numbers)   # Range
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Geometric relationship analysis
        geometric_relationships = self._analyze_geometric_relationships(content)
        features.extend([
            geometric_relationships['proportional_relationships'],
            geometric_relationships['angular_relationships'],
            geometric_relationships['scaling_relationships']
        ])
        
        return np.array(features, dtype=float)
    
    def _extract_equations(self, content: str) -> List[str]:
        """Extract mathematical equations from content."""
        
        # Look for equation-like patterns
        equation_patterns = [
            r'y\s*=\s*[^<>]+',
            r'f\([^)]*\)\s*=\s*[^<>]+',
            r'[a-zA-Z]\s*=\s*[^<>]+[+\-*/][^<>]+',
            r'\w+\s*[+\-*/]\s*\w+\s*=\s*\w+'
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            equations.extend(matches)
        
        return equations
    
    def _calculate_equation_complexity(self, equations: List[str]) -> float:
        """Calculate average complexity of equations."""
        
        if not equations:
            return 0.0
        
        total_complexity = 0
        for equation in equations:
            # Simple complexity measure: count of operators and functions
            complexity = (
                equation.count('+') + equation.count('-') + 
                equation.count('*') + equation.count('/') +
                equation.count('^') + equation.count('exp') +
                equation.count('log') + equation.count('sin') +
                equation.count('cos') + equation.count('tan')
            )
            total_complexity += complexity
        
        return total_complexity / len(equations)
    
    def _count_variables(self, content: str) -> int:
        """Count mathematical variables in content."""
        
        # Look for single-letter variables (common in math)
        variables = set(re.findall(r'\b[a-zA-Z]\b', content))
        # Filter out common non-variable words
        non_variables = {'a', 'i', 'x', 'y', 'z', 'e', 'g', 'p', 'r'}  # Keep some common variables
        math_variables = variables.intersection(non_variables)
        return len(math_variables)
    
    def _count_operators(self, content: str) -> int:
        """Count mathematical operators in content."""
        
        operators = ['+', '-', '*', '/', '^', '=', '<', '>', '≤', '≥', '≠']
        total_operators = sum(content.count(op) for op in operators)
        return total_operators
    
    def _extract_numbers(self, content: str) -> List[float]:
        """Extract numerical values from content."""
        
        number_pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
        number_strings = re.findall(number_pattern, content)
        
        numbers = []
        for num_str in number_strings:
            try:
                numbers.append(float(num_str))
            except ValueError:
                pass
        
        return numbers
    
    def _analyze_geometric_relationships(self, content: str) -> Dict[str, float]:
        """Analyze geometric relationships in visual content."""
        
        # Look for proportional relationships
        proportion_indicators = content.count('proportion') + content.count('ratio') + content.count('scale')
        
        # Look for angular relationships
        angle_indicators = content.count('angle') + content.count('degree') + content.count('rotate')
        
        # Look for scaling relationships
        scale_indicators = content.count('scale') + content.count('zoom') + content.count('transform')
        
        return {
            'proportional_relationships': proportion_indicators,
            'angular_relationships': angle_indicators,
            'scaling_relationships': scale_indicators
        }

class VisualEmbeddingProcessor:
    """
    Main processor for creating multi-dimensional visual embeddings.
    
    Creates embeddings that encode significantly more information than text
    embeddings, supporting geometric, spatial, mathematical, and temporal
    reasoning capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the visual embedding processor."""
        
        self.config = config or {}
        self.geometric_extractor = GeometricFeatureExtractor()
        self.spatial_extractor = SpatialFeatureExtractor()
        self.mathematical_extractor = MathematicalFeatureExtractor()
        
        # Feature dimensions configuration
        self.feature_dimensions = {
            'geometric': 14,      # Shape, size, proportion features
            'spatial': 12,        # Position, relationship features  
            'topological': 8,     # Connectivity features (simplified)
            'chromatic': 6,       # Color features (simplified)
            'mathematical': 18    # Mathematical relationship features
        }
        
        logger.info("Visual Embedding Processor initialized with %d total dimensions", 
                   sum(self.feature_dimensions.values()))
    
    async def create_visual_embedding(
        self, 
        visual_content: str, 
        content_type: str = "svg",
        context: Optional[Dict[str, Any]] = None
    ) -> VisualEmbedding:
        """
        Create multi-dimensional visual embedding from visual content.
        
        Args:
            visual_content: SVG, Canvas, or other visual content
            content_type: Type of visual content (svg, canvas, plot, etc.)
            context: Additional context for embedding creation
            
        Returns:
            VisualEmbedding with rich multi-dimensional representation
        """
        
        logger.info("Creating visual embedding for %s content (%d chars)", 
                   content_type, len(visual_content))
        
        try:
            # Extract features from different dimensions simultaneously
            geometric_task = self.geometric_extractor.extract_geometric_features(visual_content)
            spatial_task = self.spatial_extractor.extract_spatial_features(visual_content)
            mathematical_task = self.mathematical_extractor.extract_mathematical_features(visual_content)
            
            geometric_features, spatial_features, mathematical_features = await asyncio.gather(
                geometric_task, spatial_task, mathematical_task
            )
            
            # Create simplified topological and chromatic features
            topological_features = await self._create_topological_features(visual_content)
            chromatic_features = await self._create_chromatic_features(visual_content)
            
            # Calculate confidence scores for each dimension
            confidence_scores = await self._calculate_confidence_scores(
                visual_content, geometric_features, spatial_features, 
                mathematical_features, topological_features, chromatic_features
            )
            
            # Generate spatial relationships
            spatial_relationships = await self._extract_spatial_relationships(visual_content, context)
            
            # Generate mathematical context
            mathematical_context = await self._extract_mathematical_context(visual_content, context)
            
            # Generate semantic annotations
            semantic_annotations = await self._generate_semantic_annotations(
                visual_content, context
            )
            
            # Create embedding
            embedding = VisualEmbedding(
                geometric_features=geometric_features,
                spatial_features=spatial_features,
                topological_features=topological_features,
                chromatic_features=chromatic_features,
                mathematical_features=mathematical_features,
                embedding_id=f"visual_embed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_content=visual_content[:1000],  # Truncate for storage
                content_type=content_type,
                extraction_method="multi_dimensional_feature_extraction",
                confidence_scores=confidence_scores,
                temporal_sequence=context.get('sequence_position') if context else None,
                spatial_relationships=spatial_relationships,
                mathematical_context=mathematical_context,
                semantic_annotations=semantic_annotations
            )
            
            logger.info("Visual embedding created with %d total dimensions", 
                       len(embedding.get_combined_embedding()))
            
            return embedding
            
        except Exception as e:
            logger.error("Error creating visual embedding: %s", str(e))
            # Return minimal embedding for error cases
            return VisualEmbedding(
                geometric_features=np.zeros(self.feature_dimensions['geometric']),
                spatial_features=np.zeros(self.feature_dimensions['spatial']),
                topological_features=np.zeros(self.feature_dimensions['topological']),
                chromatic_features=np.zeros(self.feature_dimensions['chromatic']),
                mathematical_features=np.zeros(self.feature_dimensions['mathematical']),
                embedding_id="error_embedding",
                source_content="",
                content_type=content_type,
                extraction_method="error_fallback",
                confidence_scores={dim: 0.0 for dim in self.feature_dimensions.keys()},
                temporal_sequence=None,
                spatial_relationships={},
                mathematical_context={},
                semantic_annotations=["Error in embedding creation"]
            )
    
    async def _create_topological_features(self, content: str) -> np.ndarray:
        """Create simplified topological features."""
        
        features = [
            content.count('<g'),              # Group elements (connectivity)
            content.count('</g>'),            # Group closures
            content.count('<path'),           # Path elements (continuity)
            content.count('stroke'),          # Boundary elements
            content.count('fill="none"'),     # Open shapes
            content.count('fill='),           # Closed shapes
            len(re.findall(r'[zZ]', content)), # Path closures
            content.count('transform')        # Topological transformations
        ]
        
        return np.array(features, dtype=float)
    
    async def _create_chromatic_features(self, content: str) -> np.ndarray:
        """Create simplified chromatic features."""
        
        features = [
            content.count('fill='),           # Fill color usage
            content.count('stroke='),         # Stroke color usage
            len(re.findall(r'#[0-9a-fA-F]{6}', content)),  # Hex colors
            content.count('rgb'),             # RGB color usage
            content.count('opacity'),         # Transparency usage
            len(re.findall(r'[Cc]olor', content))  # Color references
        ]
        
        return np.array(features, dtype=float)
    
    async def _calculate_confidence_scores(
        self, 
        content: str,
        geometric_features: np.ndarray,
        spatial_features: np.ndarray,
        mathematical_features: np.ndarray,
        topological_features: np.ndarray,
        chromatic_features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate confidence scores for each embedding dimension."""
        
        # Content length factor (more content = higher confidence)
        content_factor = min(1.0, len(content) / 1000.0)
        
        # Feature non-zero ratio (more active features = higher confidence)
        geometric_confidence = content_factor * (np.count_nonzero(geometric_features) / len(geometric_features))
        spatial_confidence = content_factor * (np.count_nonzero(spatial_features) / len(spatial_features))
        mathematical_confidence = content_factor * (np.count_nonzero(mathematical_features) / len(mathematical_features))
        topological_confidence = content_factor * (np.count_nonzero(topological_features) / len(topological_features))
        chromatic_confidence = content_factor * (np.count_nonzero(chromatic_features) / len(chromatic_features))
        
        return {
            'geometric': geometric_confidence,
            'spatial': spatial_confidence,
            'mathematical': mathematical_confidence,
            'topological': topological_confidence,
            'chromatic': chromatic_confidence
        }
    
    async def _extract_spatial_relationships(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract spatial relationships between elements."""
        
        relationships = {
            'element_count': len(re.findall(r'<[a-zA-Z][^>]*>', content)),
            'nested_structure': content.count('<g'),
            'positioning_method': 'absolute' if 'x=' in content else 'relative',
            'layout_complexity': min(10, content.count('transform') + content.count('translate')),
            'spatial_coherence': 0.8  # Simplified coherence measure
        }
        
        return relationships
    
    async def _extract_mathematical_context(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract mathematical context from visual content."""
        
        mathematical_context = {
            'contains_equations': len(re.findall(r'[yf]\s*=', content)) > 0,
            'numerical_precision': len(re.findall(r'\d+\.\d{2,}', content)),
            'mathematical_notation': bool(re.search(r'[∑∏∫∆∇]|sum|integral|derivative', content, re.IGNORECASE)),
            'coordinate_system': 'cartesian' if ('x=' in content and 'y=' in content) else 'other',
            'function_complexity': content.count('path') + content.count('curve')
        }
        
        return mathematical_context
    
    async def _generate_semantic_annotations(self, content: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate human-readable semantic annotations."""
        
        annotations = []
        
        # Analyze content type
        if '<circle' in content:
            annotations.append("Contains circular elements")
        if '<rect' in content:
            annotations.append("Contains rectangular elements") 
        if '<path' in content:
            annotations.append("Contains path-based drawings")
        if '<text' in content:
            annotations.append("Contains text elements")
        
        # Analyze complexity
        element_count = len(re.findall(r'<[a-zA-Z]', content))
        if element_count > 20:
            annotations.append("High visual complexity")
        elif element_count > 10:
            annotations.append("Medium visual complexity")
        else:
            annotations.append("Simple visual structure")
        
        # Analyze mathematical content
        if re.search(r'y\s*=|f\(', content):
            annotations.append("Contains mathematical functions")
        if re.search(r'\d+\.\d+', content):
            annotations.append("Contains numerical data")
        
        return annotations
    
    async def compare_embeddings(self, embedding1: VisualEmbedding, embedding2: VisualEmbedding) -> Dict[str, float]:
        """Compare two visual embeddings across all dimensions."""
        
        similarity_scores = {}
        
        # Compare each dimension
        dimensions = ['geometric', 'spatial', 'topological', 'chromatic', 'mathematical']
        
        for dim in dimensions:
            features1 = getattr(embedding1, f'{dim}_features')
            features2 = getattr(embedding2, f'{dim}_features')
            
            # Ensure same length
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
            # Calculate cosine similarity
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(features1, features2) / (norm1 * norm2)
                similarity_scores[dim] = max(0.0, similarity)
            else:
                similarity_scores[dim] = 0.0
        
        # Overall similarity
        similarity_scores['overall'] = embedding1.calculate_similarity(embedding2)
        
        return similarity_scores
