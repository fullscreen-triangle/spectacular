"""
Triple Validation Framework for Spectacular AI System

This module implements the triple validation approach:
1. Pugachev-Cobra (Ridiculous Solution Validation)
2. Intent Analysis (User Intent Recognition) 
3. Reasoning Validation (AI Understanding Verification)
"""

from .core.triple_validator import TripleValidator, TripleValidationResult
from .core.pugachev_cobra import PugachevCobraGenerator, RidiculousPlot
from .core.intent_analyzer import IntentAnalyzer, IntentPlot
from .core.reasoning_validator import ReasoningValidator, ReasoningPlot

__all__ = [
    'TripleValidator',
    'TripleValidationResult', 
    'PugachevCobraGenerator',
    'RidiculousPlot',
    'IntentAnalyzer',
    'IntentPlot',
    'ReasoningValidator',
    'ReasoningPlot'
]

__version__ = "1.0.0"
