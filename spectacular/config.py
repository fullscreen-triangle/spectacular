"""
Configuration module for the Spectacular system.

This module provides centralized configuration management for all
components of the Spectacular metacognitive visualization system.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetacognitiveConfig:
    """Configuration for the Metacognitive Orchestrator."""
    confidence_threshold: float = 0.8
    max_refinement_cycles: int = 3
    reasoning_weights: Dict[str, float] = field(default_factory=lambda: {
        'fuzzy': 0.2,
        'bayesian': 0.25,
        'mdp': 0.2,
        'contextual': 0.2,
        'hf_models': 0.15
    })


@dataclass
class PretoriaConfig:
    """Configuration for the Pretoria fuzzy logic engine."""
    max_rules: int = 500
    confidence_threshold: float = 0.7
    learning_rate: float = 0.01


@dataclass
class MzekezekeConfig:
    """Configuration for the Mzekezeke Bayesian network."""
    max_evidence_nodes: int = 1000
    max_hypothesis_nodes: int = 100
    confidence_threshold: float = 0.8
    learning_rate: float = 0.01
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'accuracy': 0.4,
        'consistency': 0.3,
        'novelty': 0.2,
        'efficiency': 0.1
    })


@dataclass
class ZengezaConfig:
    """Configuration for the Zengeza MDP module."""
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.99


@dataclass
class NicotineConfig:
    """Configuration for the Nicotine sketching module."""
    canvas_width: int = 800
    canvas_height: int = 600
    validation_interval: int = 3
    context_threshold: float = 0.7


@dataclass
class HuggingFaceConfig:
    """Configuration for the HuggingFace integration hub."""
    huggingface_api_key: Optional[str] = None
    api_base_url: str = 'https://api-inference.huggingface.co/models/'
    max_concurrent_requests: int = 5
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class SpectacularConfig:
    """Main configuration for the Spectacular system."""
    # Module configurations
    metacognitive: MetacognitiveConfig = field(default_factory=MetacognitiveConfig)
    pretoria: PretoriaConfig = field(default_factory=PretoriaConfig)
    mzekezeke: MzekezekeConfig = field(default_factory=MzekezekeConfig)
    zengeza: ZengezaConfig = field(default_factory=ZengezaConfig)
    nicotine: NicotineConfig = field(default_factory=NicotineConfig)
    hf_integration: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    # System-wide settings
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_gpu: bool = False
    max_concurrent_queries: int = 10
    
    # Data and model paths
    model_cache_dir: str = "./models"
    data_dir: str = "./data"
    output_dir: str = "./output"
    
    @classmethod
    def from_env(cls) -> 'SpectacularConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # HuggingFace API key from environment
        hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        if hf_api_key:
            config.hf_integration.huggingface_api_key = hf_api_key
        
        # Debug mode
        if os.getenv('DEBUG', '').lower() in ('true', '1', 'yes'):
            config.debug_mode = True
            config.log_level = "DEBUG"
        
        # GPU enabling
        if os.getenv('ENABLE_GPU', '').lower() in ('true', '1', 'yes'):
            config.enable_gpu = True
        
        # Paths
        config.model_cache_dir = os.getenv('MODEL_CACHE_DIR', config.model_cache_dir)
        config.data_dir = os.getenv('DATA_DIR', config.data_dir)
        config.output_dir = os.getenv('OUTPUT_DIR', config.output_dir)
        
        # Confidence thresholds
        if os.getenv('CONFIDENCE_THRESHOLD'):
            threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))
            config.metacognitive.confidence_threshold = threshold
            config.mzekezeke.confidence_threshold = threshold
            config.nicotine.context_threshold = threshold
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SpectacularConfig':
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
        
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SpectacularConfig':
        """Create configuration from a dictionary."""
        config = cls()
        
        # Update module configurations
        if 'metacognitive' in config_dict:
            config.metacognitive = MetacognitiveConfig(**config_dict['metacognitive'])
        
        if 'pretoria' in config_dict:
            config.pretoria = PretoriaConfig(**config_dict['pretoria'])
        
        if 'mzekezeke' in config_dict:
            config.mzekezeke = MzekezekeConfig(**config_dict['mzekezeke'])
        
        if 'zengeza' in config_dict:
            config.zengeza = ZengezaConfig(**config_dict['zengeza'])
        
        if 'nicotine' in config_dict:
            config.nicotine = NicotineConfig(**config_dict['nicotine'])
        
        if 'hf_integration' in config_dict:
            config.hf_integration = HuggingFaceConfig(**config_dict['hf_integration'])
        
        # Update system-wide settings
        for key in ['debug_mode', 'log_level', 'enable_gpu', 'max_concurrent_queries',
                   'model_cache_dir', 'data_dir', 'output_dir']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'metacognitive': {
                'confidence_threshold': self.metacognitive.confidence_threshold,
                'max_refinement_cycles': self.metacognitive.max_refinement_cycles,
                'reasoning_weights': self.metacognitive.reasoning_weights
            },
            'pretoria': {
                'max_rules': self.pretoria.max_rules,
                'confidence_threshold': self.pretoria.confidence_threshold,
                'learning_rate': self.pretoria.learning_rate
            },
            'mzekezeke': {
                'max_evidence_nodes': self.mzekezeke.max_evidence_nodes,
                'max_hypothesis_nodes': self.mzekezeke.max_hypothesis_nodes,
                'confidence_threshold': self.mzekezeke.confidence_threshold,
                'learning_rate': self.mzekezeke.learning_rate,
                'objective_weights': self.mzekezeke.objective_weights
            },
            'zengeza': {
                'learning_rate': self.zengeza.learning_rate,
                'discount_factor': self.zengeza.discount_factor,
                'exploration_rate': self.zengeza.exploration_rate,
                'exploration_decay': self.zengeza.exploration_decay
            },
            'nicotine': {
                'canvas_width': self.nicotine.canvas_width,
                'canvas_height': self.nicotine.canvas_height,
                'validation_interval': self.nicotine.validation_interval,
                'context_threshold': self.nicotine.context_threshold
            },
            'hf_integration': {
                'huggingface_api_key': self.hf_integration.huggingface_api_key,
                'api_base_url': self.hf_integration.api_base_url,
                'max_concurrent_requests': self.hf_integration.max_concurrent_requests,
                'timeout_seconds': self.hf_integration.timeout_seconds,
                'retry_attempts': self.hf_integration.retry_attempts
            },
            'debug_mode': self.debug_mode,
            'log_level': self.log_level,
            'enable_gpu': self.enable_gpu,
            'max_concurrent_queries': self.max_concurrent_queries,
            'model_cache_dir': self.model_cache_dir,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to a JSON file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
        
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
    
    def validate(self) -> bool:
        """Validate the configuration."""
        issues = []
        
        # Check confidence thresholds
        for threshold in [self.metacognitive.confidence_threshold,
                         self.mzekezeke.confidence_threshold,
                         self.nicotine.context_threshold]:
            if not 0.0 <= threshold <= 1.0:
                issues.append(f"Confidence threshold {threshold} not in range [0.0, 1.0]")
        
        # Check learning rates
        for lr in [self.pretoria.learning_rate,
                  self.mzekezeke.learning_rate,
                  self.zengeza.learning_rate]:
            if not 0.0 < lr <= 1.0:
                issues.append(f"Learning rate {lr} not in range (0.0, 1.0]")
        
        # Check discount factor
        if not 0.0 <= self.zengeza.discount_factor <= 1.0:
            issues.append(f"Discount factor {self.zengeza.discount_factor} not in range [0.0, 1.0]")
        
        # Check reasoning weights sum to 1.0
        weights_sum = sum(self.metacognitive.reasoning_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            issues.append(f"Reasoning weights sum to {weights_sum}, should be 1.0")
        
        # Check objective weights sum to 1.0
        obj_weights_sum = sum(self.mzekezeke.objective_weights.values())
        if abs(obj_weights_sum - 1.0) > 0.01:
            issues.append(f"Objective weights sum to {obj_weights_sum}, should be 1.0")
        
        # Check canvas size
        if self.nicotine.canvas_width <= 0 or self.nicotine.canvas_height <= 0:
            issues.append("Canvas dimensions must be positive")
        
        # Check max values
        if self.pretoria.max_rules <= 0:
            issues.append("Max rules must be positive")
        
        if self.mzekezeke.max_evidence_nodes <= 0 or self.mzekezeke.max_hypothesis_nodes <= 0:
            issues.append("Max nodes must be positive")
        
        if issues:
            for issue in issues:
                logger.error(f"Configuration validation error: {issue}")
            return False
        
        logger.info("Configuration validation passed")
        return True


# Default configuration instance
default_config = SpectacularConfig()


def get_config() -> SpectacularConfig:
    """Get the current configuration."""
    # Try to load from file first, then environment, then default
    config_file = os.getenv('SPECTACULAR_CONFIG', 'config/spectacular.json')
    
    if os.path.exists(config_file):
        logger.info(f"Loading configuration from {config_file}")
        config = SpectacularConfig.from_file(config_file)
    else:
        logger.info("Loading configuration from environment")
        config = SpectacularConfig.from_env()
    
    # Validate configuration
    if not config.validate():
        logger.warning("Configuration validation failed, using default configuration")
        config = SpectacularConfig()
    
    return config


def setup_logging(config: SpectacularConfig):
    """Set up logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific loggers
    if config.debug_mode:
        logging.getLogger('spectacular').setLevel(logging.DEBUG)
    else:
        logging.getLogger('spectacular').setLevel(logging.INFO)
    
    # Suppress some verbose libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING) 