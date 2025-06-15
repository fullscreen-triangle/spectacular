"""
Nicotine: Contextual Sketching Module for Spectacular

This module implements contextual sketching functionality that creates initial
visualization sketches and maintains context through progressive refinement
and predictive validation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import json
import base64

# Image processing dependencies
try:
    from PIL import Image, ImageDraw, ImageFont
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Install with: pip install Pillow")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Install with: pip install matplotlib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SketchType(Enum):
    """Types of visualization sketches."""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    NETWORK_DIAGRAM = "network_diagram"
    PIE_CHART = "pie_chart"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    TREEMAP = "treemap"
    SANKEY_DIAGRAM = "sankey_diagram"


class SketchElement(Enum):
    """Elements that can be present in a sketch."""
    TITLE = "title"
    X_AXIS = "x_axis"
    Y_AXIS = "y_axis"
    LEGEND = "legend"
    DATA_POINTS = "data_points"
    GRID_LINES = "grid_lines"
    ANNOTATIONS = "annotations"
    INTERACTIONS = "interactions"
    COLOR_SCALE = "color_scale"
    TOOLTIPS = "tooltips"


@dataclass
class SketchComponent:
    """Represents a component in the sketch."""
    id: str
    element_type: SketchElement
    position: Tuple[int, int]  # (x, y)
    size: Tuple[int, int]      # (width, height)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    predicted: bool = False


@dataclass
class ContextualSketch:
    """Represents a complete contextual sketch."""
    id: str
    sketch_type: SketchType
    components: List[SketchComponent]
    canvas_size: Tuple[int, int]
    creation_time: datetime
    refinement_history: List[str] = field(default_factory=list)
    context_score: float = 1.0
    image_data: Optional[str] = None  # Base64 encoded image


@dataclass
class ContextValidation:
    """Result of context validation."""
    step_number: int
    missing_components: List[SketchElement]
    predicted_components: List[SketchComponent]
    confidence_score: float
    context_maintained: bool
    validation_time: datetime


class ContextualSketchingModule:
    """
    Advanced contextual sketching module for visualization generation.
    
    This class implements:
    1. Initial sketch creation from queries
    2. Progressive sketch refinement
    3. Context validation through prediction
    4. Visual component tracking
    5. Sketch-to-code mapping
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the contextual sketching module."""
        self.config = config or {}
        self.ready = False
        
        # Sketching components
        self.current_sketch: Optional[ContextualSketch] = None
        self.sketch_history: List[ContextualSketch] = []
        self.context_validations: List[ContextValidation] = []
        
        # Configuration
        self.canvas_width = config.get('canvas_width', 800)
        self.canvas_height = config.get('canvas_height', 600)
        self.validation_interval = config.get('validation_interval', 3)  # Every 3 steps
        self.context_threshold = config.get('context_threshold', 0.7)
        
        # Component libraries
        self.sketch_templates = {}
        self.component_predictors = {}
        
        # Statistics
        self.total_sketches = 0
        self.successful_predictions = 0
        self.context_maintained_count = 0
        
        # Initialize the module
        asyncio.create_task(self._initialize_sketching())
        
        logger.info("Nicotine Contextual Sketching Module initialized")
    
    async def _initialize_sketching(self):
        """Initialize the sketching system."""
        try:
            # Load sketch templates
            await self._load_sketch_templates()
            
            # Initialize component predictors
            await self._initialize_predictors()
            
            self.ready = True
            logger.info("Contextual sketching system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sketching system: {e}")
            self.ready = False
    
    async def _load_sketch_templates(self):
        """Load predefined sketch templates."""
        
        templates = {
            SketchType.BAR_CHART: {
                "required_components": [
                    SketchElement.X_AXIS,
                    SketchElement.Y_AXIS,
                    SketchElement.DATA_POINTS
                ],
                "optional_components": [
                    SketchElement.TITLE,
                    SketchElement.LEGEND,
                    SketchElement.GRID_LINES
                ],
                "layout": {
                    "margins": {"top": 50, "right": 50, "bottom": 50, "left": 50},
                    "chart_area": (100, 100, 650, 450)
                }
            },
            SketchType.LINE_CHART: {
                "required_components": [
                    SketchElement.X_AXIS,
                    SketchElement.Y_AXIS,
                    SketchElement.DATA_POINTS
                ],
                "optional_components": [
                    SketchElement.TITLE,
                    SketchElement.LEGEND,
                    SketchElement.GRID_LINES,
                    SketchElement.ANNOTATIONS
                ],
                "layout": {
                    "margins": {"top": 50, "right": 50, "bottom": 50, "left": 50},
                    "chart_area": (100, 100, 650, 450)
                }
            },
            SketchType.SCATTER_PLOT: {
                "required_components": [
                    SketchElement.X_AXIS,
                    SketchElement.Y_AXIS,
                    SketchElement.DATA_POINTS
                ],
                "optional_components": [
                    SketchElement.TITLE,
                    SketchElement.LEGEND,
                    SketchElement.COLOR_SCALE,
                    SketchElement.TOOLTIPS
                ],
                "layout": {
                    "margins": {"top": 50, "right": 100, "bottom": 50, "left": 50},
                    "chart_area": (100, 100, 600, 450)
                }
            },
            SketchType.HEATMAP: {
                "required_components": [
                    SketchElement.X_AXIS,
                    SketchElement.Y_AXIS,
                    SketchElement.DATA_POINTS,
                    SketchElement.COLOR_SCALE
                ],
                "optional_components": [
                    SketchElement.TITLE,
                    SketchElement.ANNOTATIONS,
                    SketchElement.TOOLTIPS
                ],
                "layout": {
                    "margins": {"top": 50, "right": 120, "bottom": 50, "left": 80},
                    "chart_area": (100, 100, 600, 450)
                }
            }
        }
        
        self.sketch_templates = templates
        logger.info(f"Loaded {len(templates)} sketch templates")
    
    async def _initialize_predictors(self):
        """Initialize component prediction models."""
        
        # Simple rule-based predictors for demonstration
        # In practice, these would be machine learning models
        
        predictors = {
            "title_predictor": {
                "description": "Predicts if title should be present",
                "rules": ["always_predict_title", "check_query_complexity"]
            },
            "legend_predictor": {
                "description": "Predicts if legend is needed",
                "rules": ["check_multiple_series", "check_categorical_data"]
            },
            "interaction_predictor": {
                "description": "Predicts interactive elements",
                "rules": ["check_data_size", "check_complexity"]
            }
        }
        
        self.component_predictors = predictors
        logger.info("Initialized component predictors")
    
    async def create_initial_sketch(self, query: str) -> Dict[str, Any]:
        """Create an initial sketch based on the query."""
        logger.info(f"Creating initial sketch for query: {query[:50]}...")
        
        # Analyze query to determine sketch type
        sketch_type = await self._analyze_query_for_sketch_type(query)
        
        # Create sketch components
        components = await self._create_sketch_components(sketch_type, query)
        
        # Generate the sketch
        sketch = ContextualSketch(
            id=f"sketch_{self.total_sketches + 1:03d}",
            sketch_type=sketch_type,
            components=components,
            canvas_size=(self.canvas_width, self.canvas_height),
            creation_time=datetime.now(),
            context_score=1.0
        )
        
        # Generate visual representation
        if PIL_AVAILABLE or MATPLOTLIB_AVAILABLE:
            image_data = await self._generate_sketch_image(sketch)
            sketch.image_data = image_data
        
        # Store sketch
        self.current_sketch = sketch
        self.sketch_history.append(sketch)
        self.total_sketches += 1
        
        return {
            "sketch_id": sketch.id,
            "sketch_type": sketch_type.value,
            "components_count": len(components),
            "canvas_size": sketch.canvas_size,
            "confidence": 0.9,
            "image_data": sketch.image_data,
            "context_score": sketch.context_score
        }
    
    async def _analyze_query_for_sketch_type(self, query: str) -> SketchType:
        """Analyze query to determine appropriate sketch type."""
        
        query_lower = query.lower()
        
        # Simple keyword-based analysis
        if any(word in query_lower for word in ['bar', 'column', 'categorical']):
            return SketchType.BAR_CHART
        elif any(word in query_lower for word in ['line', 'trend', 'time', 'temporal']):
            return SketchType.LINE_CHART
        elif any(word in query_lower for word in ['scatter', 'correlation', 'relationship']):
            return SketchType.SCATTER_PLOT
        elif any(word in query_lower for word in ['heatmap', 'matrix', 'correlation']):
            return SketchType.HEATMAP
        elif any(word in query_lower for word in ['network', 'graph', 'node', 'edge']):
            return SketchType.NETWORK_DIAGRAM
        elif any(word in query_lower for word in ['pie', 'proportion', 'percentage']):
            return SketchType.PIE_CHART
        elif any(word in query_lower for word in ['histogram', 'distribution', 'frequency']):
            return SketchType.HISTOGRAM
        else:
            # Default to bar chart
            return SketchType.BAR_CHART
    
    async def _create_sketch_components(self, sketch_type: SketchType, query: str) -> List[SketchComponent]:
        """Create sketch components based on type and query."""
        
        components = []
        template = self.sketch_templates.get(sketch_type, {})
        layout = template.get("layout", {})
        chart_area = layout.get("chart_area", (100, 100, 650, 450))
        
        # Required components
        required = template.get("required_components", [])
        
        component_id = 1
        
        for element_type in required:
            component = await self._create_component(
                component_id, element_type, chart_area, query
            )
            components.append(component)
            component_id += 1
        
        # Optional components based on query analysis
        optional = template.get("optional_components", [])
        
        for element_type in optional:
            if await self._should_include_component(element_type, query):
                component = await self._create_component(
                    component_id, element_type, chart_area, query
                )
                components.append(component)
                component_id += 1
        
        return components
    
    async def _create_component(self, comp_id: int, element_type: SketchElement, 
                              chart_area: Tuple[int, int, int, int], query: str) -> SketchComponent:
        """Create a specific sketch component."""
        
        x1, y1, x2, y2 = chart_area
        width = x2 - x1
        height = y2 - y1
        
        # Position and size based on component type
        if element_type == SketchElement.TITLE:
            position = (x1 + width // 2 - 100, 20)
            size = (200, 30)
            properties = {"text": "Visualization Title", "align": "center"}
        
        elif element_type == SketchElement.X_AXIS:
            position = (x1, y2)
            size = (width, 30)
            properties = {"label": "X Axis", "orientation": "horizontal"}
        
        elif element_type == SketchElement.Y_AXIS:
            position = (x1 - 30, y1)
            size = (30, height)
            properties = {"label": "Y Axis", "orientation": "vertical"}
        
        elif element_type == SketchElement.LEGEND:
            position = (x2 + 20, y1 + 50)
            size = (80, 100)
            properties = {"items": ["Series 1", "Series 2"], "orientation": "vertical"}
        
        elif element_type == SketchElement.DATA_POINTS:
            position = (x1, y1)
            size = (width, height)
            properties = {"chart_type": "data_visualization", "interactive": False}
        
        elif element_type == SketchElement.GRID_LINES:
            position = (x1, y1)
            size = (width, height)
            properties = {"style": "dashed", "color": "gray", "opacity": 0.3}
        
        elif element_type == SketchElement.COLOR_SCALE:
            position = (x2 + 20, y1 + height // 2 - 50)
            size = (20, 100)
            properties = {"orientation": "vertical", "min_value": 0, "max_value": 100}
        
        else:
            # Default positioning
            position = (x1, y1)
            size = (50, 20)
            properties = {}
        
        return SketchComponent(
            id=f"comp_{comp_id:03d}",
            element_type=element_type,
            position=position,
            size=size,
            properties=properties,
            confidence=0.9
        )
    
    async def _should_include_component(self, element_type: SketchElement, query: str) -> bool:
        """Determine if an optional component should be included."""
        
        query_lower = query.lower()
        
        if element_type == SketchElement.TITLE:
            return True  # Always include title
        
        elif element_type == SketchElement.LEGEND:
            return any(word in query_lower for word in ['multiple', 'series', 'categories', 'groups'])
        
        elif element_type == SketchElement.GRID_LINES:
            return any(word in query_lower for word in ['precise', 'detailed', 'grid'])
        
        elif element_type == SketchElement.ANNOTATIONS:
            return any(word in query_lower for word in ['annotate', 'highlight', 'mark', 'important'])
        
        elif element_type == SketchElement.INTERACTIONS:
            return any(word in query_lower for word in ['interactive', 'zoom', 'brush', 'tooltip'])
        
        elif element_type == SketchElement.COLOR_SCALE:
            return any(word in query_lower for word in ['color', 'heatmap', 'intensity', 'scale'])
        
        else:
            return False
    
    async def _generate_sketch_image(self, sketch: ContextualSketch) -> Optional[str]:
        """Generate a visual representation of the sketch."""
        
        if not (PIL_AVAILABLE or MATPLOTLIB_AVAILABLE):
            return None
        
        try:
            if MATPLOTLIB_AVAILABLE:
                return await self._generate_matplotlib_sketch(sketch)
            elif PIL_AVAILABLE:
                return await self._generate_pil_sketch(sketch)
        except Exception as e:
            logger.error(f"Error generating sketch image: {e}")
            return None
    
    async def _generate_matplotlib_sketch(self, sketch: ContextualSketch) -> str:
        """Generate sketch using matplotlib."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, sketch.canvas_size[0])
        ax.set_ylim(0, sketch.canvas_size[1])
        ax.invert_yaxis()  # Invert Y axis to match screen coordinates
        
        # Draw components
        for component in sketch.components:
            x, y = component.position
            w, h = component.size
            
            if component.element_type == SketchElement.TITLE:
                ax.text(x, y, component.properties.get("text", "Title"), 
                       ha='center', va='center', fontsize=14, weight='bold')
            
            elif component.element_type == SketchElement.X_AXIS:
                ax.plot([x, x + w], [y, y], 'k-', linewidth=2)
                ax.text(x + w//2, y + 20, component.properties.get("label", "X Axis"), 
                       ha='center', va='center')
            
            elif component.element_type == SketchElement.Y_AXIS:
                ax.plot([x, x], [y, y + h], 'k-', linewidth=2)
                ax.text(x - 20, y + h//2, component.properties.get("label", "Y Axis"), 
                       ha='center', va='center', rotation=90)
            
            elif component.element_type == SketchElement.DATA_POINTS:
                # Draw placeholder data area
                rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                       edgecolor='blue', facecolor='lightblue', alpha=0.3)
                ax.add_patch(rect)
                ax.text(x + w//2, y + h//2, 'Data Area', ha='center', va='center')
            
            elif component.element_type == SketchElement.LEGEND:
                rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                       edgecolor='gray', facecolor='white')
                ax.add_patch(rect)
                ax.text(x + w//2, y + 10, 'Legend', ha='center', va='center', fontsize=10)
            
            elif component.element_type == SketchElement.GRID_LINES:
                # Draw grid pattern
                for i in range(5):
                    grid_x = x + i * (w // 4)
                    ax.plot([grid_x, grid_x], [y, y + h], 'k--', alpha=0.3, linewidth=0.5)
                for i in range(5):
                    grid_y = y + i * (h // 4)
                    ax.plot([x, x + w], [grid_y, grid_y], 'k--', alpha=0.3, linewidth=0.5)
        
        # Remove matplotlib axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    async def _generate_pil_sketch(self, sketch: ContextualSketch) -> str:
        """Generate sketch using PIL."""
        
        # Create image
        img = Image.new('RGB', sketch.canvas_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        # Draw components
        for component in sketch.components:
            x, y = component.position
            w, h = component.size
            
            if component.element_type == SketchElement.TITLE:
                text = component.properties.get("text", "Title")
                draw.text((x, y), text, fill='black', font=title_font)
            
            elif component.element_type == SketchElement.X_AXIS:
                draw.line([(x, y), (x + w, y)], fill='black', width=2)
                label = component.properties.get("label", "X Axis")
                draw.text((x + w//2, y + 5), label, fill='black', font=font)
            
            elif component.element_type == SketchElement.Y_AXIS:
                draw.line([(x, y), (x, y + h)], fill='black', width=2)
                label = component.properties.get("label", "Y Axis")
                draw.text((x - 30, y + h//2), label, fill='black', font=font)
            
            elif component.element_type == SketchElement.DATA_POINTS:
                draw.rectangle([x, y, x + w, y + h], outline='blue', fill=None)
                draw.text((x + w//2, y + h//2), "Data Area", fill='blue', font=font)
            
            elif component.element_type == SketchElement.LEGEND:
                draw.rectangle([x, y, x + w, y + h], outline='gray', fill=None)
                draw.text((x + 5, y + 5), "Legend", fill='black', font=font)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return image_base64
    
    async def validate_context(self, reasoning_results: Dict) -> Dict[str, Any]:
        """Validate context by predicting missing sketch components."""
        logger.info("Validating context through component prediction...")
        
        if not self.current_sketch:
            return {"error": "No current sketch to validate"}
        
        step_number = len(self.context_validations) + 1
        
        # Check if it's time for validation
        if step_number % self.validation_interval != 0:
            return {"skipped": True, "next_validation_step": step_number + 1}
        
        # Predict missing components
        missing_components = await self._predict_missing_components()
        
        # Generate predicted components
        predicted_components = []
        for element_type in missing_components:
            predicted_comp = await self._predict_component(element_type)
            if predicted_comp:
                predicted_components.append(predicted_comp)
        
        # Calculate confidence score
        confidence_score = await self._calculate_prediction_confidence(
            missing_components, predicted_components
        )
        
        # Determine if context is maintained
        context_maintained = confidence_score >= self.context_threshold
        
        # Create validation record
        validation = ContextValidation(
            step_number=step_number,
            missing_components=missing_components,
            predicted_components=predicted_components,
            confidence_score=confidence_score,
            context_maintained=context_maintained,
            validation_time=datetime.now()
        )
        
        self.context_validations.append(validation)
        
        # Update statistics
        if predicted_components:
            self.successful_predictions += 1
        if context_maintained:
            self.context_maintained_count += 1
        
        # Update sketch context score
        if self.current_sketch:
            self.current_sketch.context_score = confidence_score
        
        return {
            "step_number": step_number,
            "missing_components": [comp.value for comp in missing_components],
            "predicted_components_count": len(predicted_components),
            "confidence_score": confidence_score,
            "context_maintained": context_maintained,
            "validation_time": validation.validation_time.isoformat()
        }
    
    async def _predict_missing_components(self) -> List[SketchElement]:
        """Predict what components are missing from the current sketch."""
        
        if not self.current_sketch:
            return []
        
        current_elements = {comp.element_type for comp in self.current_sketch.components}
        sketch_type = self.current_sketch.sketch_type
        
        # Get expected components for this sketch type
        template = self.sketch_templates.get(sketch_type, {})
        required = set(template.get("required_components", []))
        optional = set(template.get("optional_components", []))
        
        # Find missing required components
        missing_required = required - current_elements
        
        # Predict some optional components that should be present
        missing_optional = []
        for element in optional:
            if element not in current_elements:
                # Use simple heuristics to predict if this should be present
                should_predict = await self._should_predict_component(element)
                if should_predict:
                    missing_optional.append(element)
        
        return list(missing_required) + missing_optional
    
    async def _should_predict_component(self, element_type: SketchElement) -> bool:
        """Determine if we should predict this component as missing."""
        
        # Simple prediction rules
        if element_type == SketchElement.TITLE:
            return True  # Titles are usually expected
        elif element_type == SketchElement.LEGEND:
            return len(self.current_sketch.components) > 3  # For complex charts
        elif element_type == SketchElement.GRID_LINES:
            return False  # Less critical
        elif element_type == SketchElement.ANNOTATIONS:
            return False  # Optional
        else:
            return False
    
    async def _predict_component(self, element_type: SketchElement) -> Optional[SketchComponent]:
        """Predict a missing component."""
        
        if not self.current_sketch:
            return None
        
        # Find suitable position and size for the predicted component
        canvas_width, canvas_height = self.current_sketch.canvas_size
        
        # Simple positioning logic
        if element_type == SketchElement.TITLE:
            position = (canvas_width // 2 - 100, 20)
            size = (200, 30)
            properties = {"text": "Predicted Title", "align": "center"}
        
        elif element_type == SketchElement.LEGEND:
            position = (canvas_width - 120, 100)
            size = (100, 150)
            properties = {"items": ["Predicted Series"], "orientation": "vertical"}
        
        elif element_type == SketchElement.ANNOTATIONS:
            position = (canvas_width // 2, canvas_height // 2)
            size = (100, 20)
            properties = {"text": "Predicted Annotation"}
        
        else:
            position = (50, 50)
            size = (100, 50)
            properties = {}
        
        return SketchComponent(
            id=f"pred_{len(self.current_sketch.components) + 1:03d}",
            element_type=element_type,
            position=position,
            size=size,
            properties=properties,
            confidence=0.7,
            predicted=True
        )
    
    async def _calculate_prediction_confidence(self, missing_components: List[SketchElement], 
                                             predicted_components: List[SketchComponent]) -> float:
        """Calculate confidence score for predictions."""
        
        if not missing_components:
            return 1.0  # Perfect if nothing missing
        
        if not predicted_components:
            return 0.0  # No predictions made
        
        # Simple confidence calculation
        predicted_count = len(predicted_components)
        missing_count = len(missing_components)
        
        # Base confidence on prediction coverage
        coverage = predicted_count / missing_count if missing_count > 0 else 1.0
        
        # Average component confidence
        avg_component_confidence = np.mean([comp.confidence for comp in predicted_components])
        
        # Combined confidence
        confidence = (coverage * 0.6 + avg_component_confidence * 0.4)
        
        return min(1.0, confidence)
    
    def is_ready(self) -> bool:
        """Check if the sketching module is ready."""
        return self.ready
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        
        context_maintenance_rate = 0.0
        if self.context_validations:
            context_maintenance_rate = self.context_maintained_count / len(self.context_validations)
        
        prediction_success_rate = 0.0
        if self.context_validations:
            prediction_success_rate = self.successful_predictions / len(self.context_validations)
        
        return {
            "ready": self.ready,
            "total_sketches": self.total_sketches,
            "current_sketch_id": self.current_sketch.id if self.current_sketch else None,
            "total_validations": len(self.context_validations),
            "context_maintenance_rate": context_maintenance_rate,
            "prediction_success_rate": prediction_success_rate,
            "pil_available": PIL_AVAILABLE,
            "matplotlib_available": MATPLOTLIB_AVAILABLE
        }
    
    def get_current_sketch_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current sketch."""
        
        if not self.current_sketch:
            return None
        
        return {
            "sketch_id": self.current_sketch.id,
            "sketch_type": self.current_sketch.sketch_type.value,
            "component_count": len(self.current_sketch.components),
            "context_score": self.current_sketch.context_score,
            "creation_time": self.current_sketch.creation_time.isoformat(),
            "has_image": self.current_sketch.image_data is not None,
            "components": [
                {
                    "id": comp.id,
                    "type": comp.element_type.value,
                    "position": comp.position,
                    "size": comp.size,
                    "predicted": comp.predicted
                }
                for comp in self.current_sketch.components
            ]
        }