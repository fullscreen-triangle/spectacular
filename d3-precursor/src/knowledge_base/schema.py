from enum import Enum, auto
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass


class ApiElementType(Enum):
    """Types of API elements that can be stored in the knowledge base."""
    FUNCTION = auto()
    METHOD = auto()
    CLASS = auto()
    MODULE = auto()
    PROPERTY = auto()
    EVENT = auto()
    CONSTANT = auto()


class ParameterType(Enum):
    """Types of parameters for API elements."""
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    ARRAY = auto()
    OBJECT = auto()
    FUNCTION = auto()
    ANY = auto()


class VisualizationType(Enum):
    """Types of visualizations associated with usage patterns."""
    BAR_CHART = auto()
    LINE_CHART = auto()
    SCATTER_PLOT = auto()
    NETWORK_GRAPH = auto()
    TREE = auto()
    HIERARCHY = auto()
    MAP = auto()
    CUSTOM = auto()


@dataclass
class Parameter:
    """Represents a parameter for an API element."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class ReturnValue:
    """Represents the return value of an API element."""
    type: Union[ParameterType, List[ParameterType]]
    description: str
    example: Optional[Any] = None


@dataclass
class Example:
    """Code example showing how to use an API element."""
    code: str
    description: str
    context: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None


@dataclass
class UsagePattern:
    """Common patterns of usage for an API element."""
    name: str
    description: str
    code_template: str
    visualization_type: Optional[VisualizationType] = None
    contexts: List[str] = None


@dataclass
class ApiElement:
    """Represents an API element in the D3 library."""
    name: str
    type: ApiElementType
    description: str
    parameters: List[Parameter] = None
    return_value: Optional[ReturnValue] = None
    examples: List[Example] = None
    usage_patterns: List[UsagePattern] = None
    related_elements: List[str] = None
    source_url: Optional[str] = None
    version_introduced: Optional[str] = None
    version_deprecated: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None 