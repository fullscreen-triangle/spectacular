"""
Spatial Reasoning Engine: Advanced spatial relationship analysis for visual content.

This module provides sophisticated spatial reasoning capabilities that understand
geometric relationships, positioning, and spatial context in visualizations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import math

logger = logging.getLogger(__name__)

@dataclass
class SpatialContext:
    """Context for spatial reasoning operations."""
    coordinate_system: str           # cartesian, polar, spherical, etc.
    dimension_count: int             # 2D, 3D, etc.
    reference_frame: str             # absolute, relative, normalized
    scale_factor: float              # Scaling information
    bounds: Dict[str, Tuple[float, float]]  # min/max bounds per dimension
    transformation_matrix: Optional[np.ndarray]  # Transformation matrix if applicable
    spatial_units: str               # pixels, meters, normalized, etc.
    
    def get_bounds_info(self) -> Dict[str, Any]:
        """Get comprehensive bounds information."""
        return {
            'bounds': self.bounds,
            'aspect_ratios': {
                dim: (bounds[1] - bounds[0]) for dim, bounds in self.bounds.items()
            },
            'total_area': self._calculate_total_area(),
            'center_point': self._calculate_center_point()
        }
    
    def _calculate_total_area(self) -> float:
        """Calculate total area of the spatial context."""
        if 'x' in self.bounds and 'y' in self.bounds:
            width = self.bounds['x'][1] - self.bounds['x'][0]
            height = self.bounds['y'][1] - self.bounds['y'][0]
            return width * height
        return 0.0
    
    def _calculate_center_point(self) -> Dict[str, float]:
        """Calculate center point of spatial context."""
        center = {}
        for dim, (min_val, max_val) in self.bounds.items():
            center[dim] = (min_val + max_val) / 2.0
        return center

@dataclass
class SpatialRelationship:
    """Represents a spatial relationship between elements."""
    element1_id: str
    element2_id: str
    relationship_type: str           # adjacent, overlapping, contained, etc.
    distance: float                  # Spatial distance
    angle: float                     # Relative angle in radians
    relative_position: str           # above, below, left, right, etc.
    strength: float                  # Relationship strength (0-1)
    context: SpatialContext

class SpatialReasoningEngine:
    """
    Advanced spatial reasoning engine for visual content analysis.
    
    Provides sophisticated spatial relationship analysis, positioning logic,
    and geometric reasoning capabilities for the visual reasoning framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the spatial reasoning engine."""
        self.config = config or {}
        
        # Spatial relationship types and their detection methods
        self.relationship_types = {
            'adjacent': self._detect_adjacency,
            'overlapping': self._detect_overlap,
            'contained': self._detect_containment,
            'aligned': self._detect_alignment,
            'parallel': self._detect_parallelism,
            'perpendicular': self._detect_perpendicularity,
            'clustered': self._detect_clustering,
            'distributed': self._detect_distribution
        }
        
        # Spatial analysis thresholds
        self.adjacency_threshold = config.get('adjacency_threshold', 10.0)
        self.overlap_threshold = config.get('overlap_threshold', 0.1)
        self.alignment_threshold = config.get('alignment_threshold', 5.0)
        self.angle_tolerance = config.get('angle_tolerance', math.pi / 12)  # 15 degrees
        
        logger.info("Spatial Reasoning Engine initialized with %d relationship types", 
                   len(self.relationship_types))
    
    async def analyze_spatial_context(self, visual_content: str) -> SpatialContext:
        """
        Analyze spatial context from visual content.
        
        Args:
            visual_content: SVG or other visual content to analyze
            
        Returns:
            SpatialContext with coordinate system and bounds information
        """
        
        logger.info("Analyzing spatial context from visual content")
        
        try:
            # Extract coordinate system information
            coordinate_system = await self._detect_coordinate_system(visual_content)
            
            # Determine dimensionality
            dimension_count = await self._detect_dimensionality(visual_content)
            
            # Extract bounds information
            bounds = await self._extract_spatial_bounds(visual_content)
            
            # Detect transformation matrix if present
            transformation_matrix = await self._extract_transformation_matrix(visual_content)
            
            # Determine reference frame and scaling
            reference_frame = await self._detect_reference_frame(visual_content)
            scale_factor = await self._calculate_scale_factor(visual_content, bounds)
            
            # Determine spatial units
            spatial_units = await self._detect_spatial_units(visual_content)
            
            context = SpatialContext(
                coordinate_system=coordinate_system,
                dimension_count=dimension_count,
                reference_frame=reference_frame,
                scale_factor=scale_factor,
                bounds=bounds,
                transformation_matrix=transformation_matrix,
                spatial_units=spatial_units
            )
            
            logger.info("Spatial context analyzed: %s coordinate system, %dD", 
                       coordinate_system, dimension_count)
            
            return context
            
        except Exception as e:
            logger.error("Error analyzing spatial context: %s", str(e))
            # Return default context
            return SpatialContext(
                coordinate_system="cartesian",
                dimension_count=2,
                reference_frame="absolute",
                scale_factor=1.0,
                bounds={'x': (0, 400), 'y': (0, 300)},
                transformation_matrix=None,
                spatial_units="pixels"
            )
    
    async def extract_spatial_relationships(
        self, 
        visual_content: str,
        context: SpatialContext
    ) -> List[SpatialRelationship]:
        """
        Extract spatial relationships between elements in visual content.
        
        Args:
            visual_content: Visual content to analyze
            context: Spatial context for the content
            
        Returns:
            List of detected spatial relationships
        """
        
        logger.info("Extracting spatial relationships from visual content")
        
        try:
            # Extract visual elements with their positions
            elements = await self._extract_visual_elements(visual_content)
            
            if len(elements) < 2:
                return []  # Need at least 2 elements for relationships
            
            relationships = []
            
            # Analyze pairwise relationships
            for i, element1 in enumerate(elements):
                for j, element2 in enumerate(elements[i+1:], i+1):
                    
                    # Calculate spatial relationship for each type
                    for relationship_type, detector in self.relationship_types.items():
                        relationship = await detector(element1, element2, context)
                        
                        if relationship and relationship.strength > 0.3:  # Threshold for meaningful relationships
                            relationships.append(relationship)
            
            # Sort by relationship strength
            relationships.sort(key=lambda r: r.strength, reverse=True)
            
            logger.info("Extracted %d spatial relationships", len(relationships))
            return relationships[:20]  # Limit to top 20 relationships
            
        except Exception as e:
            logger.error("Error extracting spatial relationships: %s", str(e))
            return []
    
    async def analyze_spatial_patterns(
        self, 
        relationships: List[SpatialRelationship]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in spatial relationships.
        
        Args:
            relationships: List of spatial relationships to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        
        if not relationships:
            return {'patterns': [], 'complexity': 0.0, 'coherence': 0.0}
        
        try:
            # Group relationships by type
            relationship_groups = {}
            for rel in relationships:
                if rel.relationship_type not in relationship_groups:
                    relationship_groups[rel.relationship_type] = []
                relationship_groups[rel.relationship_type].append(rel)
            
            # Analyze patterns
            patterns = []
            
            # Regular arrangement patterns
            if 'aligned' in relationship_groups and len(relationship_groups['aligned']) >= 3:
                patterns.append({
                    'type': 'regular_alignment',
                    'strength': np.mean([r.strength for r in relationship_groups['aligned']]),
                    'count': len(relationship_groups['aligned'])
                })
            
            # Clustering patterns
            if 'clustered' in relationship_groups:
                patterns.append({
                    'type': 'clustering',
                    'strength': np.mean([r.strength for r in relationship_groups['clustered']]),
                    'count': len(relationship_groups['clustered'])
                })
            
            # Hierarchical patterns (containment)
            if 'contained' in relationship_groups:
                patterns.append({
                    'type': 'hierarchical_structure',
                    'strength': np.mean([r.strength for r in relationship_groups['contained']]),
                    'count': len(relationship_groups['contained'])
                })
            
            # Calculate overall complexity and coherence
            complexity = self._calculate_spatial_complexity(relationships)
            coherence = self._calculate_spatial_coherence(relationships, patterns)
            
            return {
                'patterns': patterns,
                'relationship_distribution': {k: len(v) for k, v in relationship_groups.items()},
                'complexity': complexity,
                'coherence': coherence,
                'total_relationships': len(relationships)
            }
            
        except Exception as e:
            logger.error("Error analyzing spatial patterns: %s", str(e))
            return {'patterns': [], 'complexity': 0.0, 'coherence': 0.0}
    
    # Detection Methods for Different Relationship Types
    
    async def _detect_adjacency(
        self, 
        element1: Dict[str, Any], 
        element2: Dict[str, Any], 
        context: SpatialContext
    ) -> Optional[SpatialRelationship]:
        """Detect adjacency relationship between elements."""
        
        pos1 = element1.get('position', {})
        pos2 = element2.get('position', {})
        
        if not pos1 or not pos2:
            return None
        
        # Calculate distance between elements
        distance = math.sqrt(
            (pos2.get('x', 0) - pos1.get('x', 0)) ** 2 +
            (pos2.get('y', 0) - pos1.get('y', 0)) ** 2
        )
        
        # Check if within adjacency threshold
        if distance <= self.adjacency_threshold:
            strength = max(0.0, 1.0 - (distance / self.adjacency_threshold))
            
            # Determine relative position
            relative_pos = self._determine_relative_position(pos1, pos2)
            
            # Calculate angle
            angle = math.atan2(
                pos2.get('y', 0) - pos1.get('y', 0),
                pos2.get('x', 0) - pos1.get('x', 0)
            )
            
            return SpatialRelationship(
                element1_id=element1.get('id', 'unknown'),
                element2_id=element2.get('id', 'unknown'),
                relationship_type='adjacent',
                distance=distance,
                angle=angle,
                relative_position=relative_pos,
                strength=strength,
                context=context
            )
        
        return None
    
    async def _detect_overlap(
        self, 
        element1: Dict[str, Any], 
        element2: Dict[str, Any], 
        context: SpatialContext
    ) -> Optional[SpatialRelationship]:
        """Detect overlap relationship between elements."""
        
        bounds1 = element1.get('bounds')
        bounds2 = element2.get('bounds')
        
        if not bounds1 or not bounds2:
            return None
        
        # Calculate overlap area
        overlap_area = self._calculate_overlap_area(bounds1, bounds2)
        total_area = self._calculate_total_area(bounds1) + self._calculate_total_area(bounds2)
        
        if total_area > 0:
            overlap_ratio = overlap_area / total_area
            
            if overlap_ratio > self.overlap_threshold:
                # Calculate center-to-center distance
                center1 = self._calculate_bounds_center(bounds1)
                center2 = self._calculate_bounds_center(bounds2)
                distance = math.sqrt(
                    (center2['x'] - center1['x']) ** 2 +
                    (center2['y'] - center1['y']) ** 2
                )
                
                angle = math.atan2(
                    center2['y'] - center1['y'],
                    center2['x'] - center1['x']
                )
                
                return SpatialRelationship(
                    element1_id=element1.get('id', 'unknown'),
                    element2_id=element2.get('id', 'unknown'),
                    relationship_type='overlapping',
                    distance=distance,
                    angle=angle,
                    relative_position='overlapping',
                    strength=min(1.0, overlap_ratio * 2),  # Scale strength
                    context=context
                )
        
        return None
    
    async def _detect_alignment(
        self, 
        element1: Dict[str, Any], 
        element2: Dict[str, Any], 
        context: SpatialContext
    ) -> Optional[SpatialRelationship]:
        """Detect alignment relationship between elements."""
        
        pos1 = element1.get('position', {})
        pos2 = element2.get('position', {})
        
        if not pos1 or not pos2:
            return None
        
        x1, y1 = pos1.get('x', 0), pos1.get('y', 0)
        x2, y2 = pos2.get('x', 0), pos2.get('y', 0)
        
        # Check for horizontal alignment
        horizontal_diff = abs(y1 - y2)
        vertical_diff = abs(x1 - x2)
        
        alignment_type = None
        alignment_diff = 0
        
        if horizontal_diff <= self.alignment_threshold:
            alignment_type = 'horizontal'
            alignment_diff = horizontal_diff
        elif vertical_diff <= self.alignment_threshold:
            alignment_type = 'vertical'
            alignment_diff = vertical_diff
        
        if alignment_type:
            strength = max(0.0, 1.0 - (alignment_diff / self.alignment_threshold))
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = math.atan2(y2 - y1, x2 - x1)
            
            return SpatialRelationship(
                element1_id=element1.get('id', 'unknown'),
                element2_id=element2.get('id', 'unknown'),
                relationship_type='aligned',
                distance=distance,
                angle=angle,
                relative_position=f'{alignment_type}_aligned',
                strength=strength,
                context=context
            )
        
        return None
    
    # Helper Methods
    
    async def _detect_coordinate_system(self, content: str) -> str:
        """Detect coordinate system used in visual content."""
        
        if 'viewBox' in content and ('x=' in content or 'y=' in content):
            return 'cartesian'
        elif 'transform' in content and 'polar' in content.lower():
            return 'polar'
        else:
            return 'cartesian'  # Default
    
    async def _detect_dimensionality(self, content: str) -> int:
        """Detect spatial dimensionality of visual content."""
        
        if 'z=' in content or '3d' in content.lower():
            return 3
        else:
            return 2  # Default for SVG
    
    async def _extract_spatial_bounds(self, content: str) -> Dict[str, Tuple[float, float]]:
        """Extract spatial bounds from visual content."""
        
        import re
        
        # Extract viewBox if present
        viewbox_match = re.search(r'viewBox["\s]*=["\s]*([0-9.-]+)[,\s]+([0-9.-]+)[,\s]+([0-9.-]+)[,\s]+([0-9.-]+)', content)
        if viewbox_match:
            min_x, min_y, width, height = [float(x) for x in viewbox_match.groups()]
            return {
                'x': (min_x, min_x + width),
                'y': (min_y, min_y + height)
            }
        
        # Extract width/height attributes
        width_match = re.search(r'width["\s]*=["\s]*([0-9.-]+)', content)
        height_match = re.search(r'height["\s]*=["\s]*([0-9.-]+)', content)
        
        width = float(width_match.group(1)) if width_match else 400
        height = float(height_match.group(1)) if height_match else 300
        
        return {
            'x': (0, width),
            'y': (0, height)
        }
    
    async def _extract_visual_elements(self, content: str) -> List[Dict[str, Any]]:
        """Extract visual elements with position and bounds information."""
        
        import re
        
        elements = []
        element_id = 0
        
        # Extract different element types
        element_patterns = [
            (r'<circle[^>]*cx["\s]*=["\s]*([0-9.-]+)[^>]*cy["\s]*=["\s]*([0-9.-]+)[^>]*r["\s]*=["\s]*([0-9.-]+)', 'circle'),
            (r'<rect[^>]*x["\s]*=["\s]*([0-9.-]+)[^>]*y["\s]*=["\s]*([0-9.-]+)[^>]*width["\s]*=["\s]*([0-9.-]+)[^>]*height["\s]*=["\s]*([0-9.-]+)', 'rectangle'),
            (r'<line[^>]*x1["\s]*=["\s]*([0-9.-]+)[^>]*y1["\s]*=["\s]*([0-9.-]+)[^>]*x2["\s]*=["\s]*([0-9.-]+)[^>]*y2["\s]*=["\s]*([0-9.-]+)', 'line'),
        ]
        
        for pattern, element_type in element_patterns:
            matches = re.findall(pattern, content)
            
            for match in matches:
                element = {
                    'id': f'element_{element_id}',
                    'type': element_type
                }
                
                if element_type == 'circle':
                    cx, cy, r = [float(x) for x in match]
                    element['position'] = {'x': cx, 'y': cy}
                    element['bounds'] = {
                        'x_min': cx - r, 'x_max': cx + r,
                        'y_min': cy - r, 'y_max': cy + r
                    }
                elif element_type == 'rectangle':
                    x, y, w, h = [float(x) for x in match]
                    element['position'] = {'x': x + w/2, 'y': y + h/2}  # Center position
                    element['bounds'] = {
                        'x_min': x, 'x_max': x + w,
                        'y_min': y, 'y_max': y + h
                    }
                elif element_type == 'line':
                    x1, y1, x2, y2 = [float(x) for x in match]
                    element['position'] = {'x': (x1 + x2)/2, 'y': (y1 + y2)/2}  # Midpoint
                    element['bounds'] = {
                        'x_min': min(x1, x2), 'x_max': max(x1, x2),
                        'y_min': min(y1, y2), 'y_max': max(y1, y2)
                    }
                
                elements.append(element)
                element_id += 1
        
        return elements
    
    def _determine_relative_position(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> str:
        """Determine relative position between two points."""
        
        dx = pos2.get('x', 0) - pos1.get('x', 0)
        dy = pos2.get('y', 0) - pos1.get('y', 0)
        
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'below' if dy > 0 else 'above'
    
    def _calculate_overlap_area(self, bounds1: Dict[str, float], bounds2: Dict[str, float]) -> float:
        """Calculate overlap area between two bounding boxes."""
        
        x_overlap = max(0, min(bounds1['x_max'], bounds2['x_max']) - max(bounds1['x_min'], bounds2['x_min']))
        y_overlap = max(0, min(bounds1['y_max'], bounds2['y_max']) - max(bounds1['y_min'], bounds2['y_min']))
        
        return x_overlap * y_overlap
    
    def _calculate_total_area(self, bounds: Dict[str, float]) -> float:
        """Calculate total area of bounding box."""
        
        width = bounds['x_max'] - bounds['x_min']
        height = bounds['y_max'] - bounds['y_min']
        return width * height
    
    def _calculate_bounds_center(self, bounds: Dict[str, float]) -> Dict[str, float]:
        """Calculate center of bounding box."""
        
        return {
            'x': (bounds['x_min'] + bounds['x_max']) / 2,
            'y': (bounds['y_min'] + bounds['y_max']) / 2
        }
    
    def _calculate_spatial_complexity(self, relationships: List[SpatialRelationship]) -> float:
        """Calculate spatial complexity score."""
        
        if not relationships:
            return 0.0
        
        # Base complexity from relationship count
        relationship_complexity = min(1.0, len(relationships) / 20.0)
        
        # Type diversity complexity
        relationship_types = set(r.relationship_type for r in relationships)
        type_complexity = len(relationship_types) / len(self.relationship_types)
        
        # Average relationship strength
        avg_strength = np.mean([r.strength for r in relationships])
        
        return (relationship_complexity + type_complexity + avg_strength) / 3.0
    
    def _calculate_spatial_coherence(self, relationships: List[SpatialRelationship], patterns: List[Dict[str, Any]]) -> float:
        """Calculate spatial coherence score."""
        
        if not relationships:
            return 0.0
        
        # Pattern coherence (regularity in relationships)
        pattern_coherence = min(1.0, len(patterns) / 3.0)
        
        # Relationship strength coherence (consistency in strengths)
        strengths = [r.strength for r in relationships]
        strength_variance = np.var(strengths)
        strength_coherence = max(0.0, 1.0 - strength_variance)
        
        return (pattern_coherence + strength_coherence) / 2.0
    
    # Placeholder methods for remaining relationship types
    async def _detect_containment(self, element1, element2, context): return None
    async def _detect_parallelism(self, element1, element2, context): return None  
    async def _detect_perpendicularity(self, element1, element2, context): return None
    async def _detect_clustering(self, element1, element2, context): return None
    async def _detect_distribution(self, element1, element2, context): return None
    async def _extract_transformation_matrix(self, content): return None
    async def _detect_reference_frame(self, content): return "absolute"
    async def _calculate_scale_factor(self, content, bounds): return 1.0
    async def _detect_spatial_units(self, content): return "pixels"
