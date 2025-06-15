"""
HuggingFace Integration Hub for Spectacular

This module manages the integration with multiple HuggingFace models for
different aspects of visualization generation, including code synthesis,
vision-language understanding, and domain-specific reasoning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from datetime import datetime

# HuggingFace dependencies
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        pipeline, T5ForConditionalGeneration, T5Tokenizer
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available. Install with: pip install requests")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of HuggingFace models available."""
    CODE_GENERATION = "code_generation"
    VISION_LANGUAGE = "vision_language"
    NATURAL_LANGUAGE = "natural_language"
    DOMAIN_SPECIFIC = "domain_specific"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"


class ModelSize(Enum):
    """Model size categories."""
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    XL = "xl"


@dataclass
class ModelConfig:
    """Configuration for a HuggingFace model."""
    name: str
    model_id: str
    model_type: ModelType
    size: ModelSize
    local_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    use_local: bool = False
    use_api: bool = True
    priority: float = 1.0
    specialized_for: List[str] = field(default_factory=list)


@dataclass
class ModelInvocation:
    """Record of a model invocation."""
    model_name: str
    input_data: Any
    output_data: Any
    tokens_used: int
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultiModelResponse:
    """Response from multiple model invocations."""
    primary_response: Any
    all_responses: Dict[str, Any]
    consensus_score: float
    best_model: str
    confidence: float
    processing_time_ms: float


class HuggingFaceHub:
    """
    Advanced HuggingFace model integration hub.
    
    This class manages:
    1. Multiple model configurations and selection
    2. Local and API-based model inference
    3. Multi-model consensus and ranking
    4. Specialized model routing
    5. Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the HuggingFace integration hub."""
        self.config = config or {}
        self.ready = False
        
        # Model management
        self.available_models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # API configuration
        self.api_key = config.get('huggingface_api_key')
        self.api_base_url = config.get('api_base_url', 'https://api-inference.huggingface.co/models/')
        
        # Performance tracking
        self.invocation_history: List[ModelInvocation] = []
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Configuration
        self.max_concurrent_requests = config.get('max_concurrent_requests', 5)
        self.timeout_seconds = config.get('timeout_seconds', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        
        # Initialize the hub
        asyncio.create_task(self._initialize_hub())
        
        logger.info("HuggingFace Integration Hub initialized")
    
    async def _initialize_hub(self):
        """Initialize the HuggingFace hub."""
        try:
            # Register available models
            await self._register_models()
            
            # Load high-priority local models
            await self._load_priority_models()
            
            # Test API connectivity
            await self._test_api_connectivity()
            
            self.ready = True
            logger.info("HuggingFace hub initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing HuggingFace hub: {e}")
            self.ready = False
    
    async def _register_models(self):
        """Register available HuggingFace models."""
        
        models = [
            # Code generation models
            ModelConfig(
                name="CodeT5-base",
                model_id="Salesforce/codet5-base",
                model_type=ModelType.CODE_GENERATION,
                size=ModelSize.BASE,
                max_tokens=512,
                specialized_for=["javascript", "d3js", "visualization"]
            ),
            ModelConfig(
                name="CodeT5-large",
                model_id="Salesforce/codet5-large",
                model_type=ModelType.CODE_GENERATION,
                size=ModelSize.LARGE,
                max_tokens=1024,
                specialized_for=["complex_visualizations", "advanced_d3"]
            ),
            ModelConfig(
                name="InCoder",
                model_id="facebook/incoder-1B",
                model_type=ModelType.CODE_GENERATION,
                size=ModelSize.BASE,
                max_tokens=512,
                specialized_for=["code_completion", "infilling"]
            ),
            
            # Vision-language models
            ModelConfig(
                name="CLIP",
                model_id="openai/clip-vit-base-patch32",
                model_type=ModelType.VISION_LANGUAGE,
                size=ModelSize.BASE,
                specialized_for=["image_understanding", "sketch_analysis"]
            ),
            ModelConfig(
                name="BLIP-2",
                model_id="Salesforce/blip2-opt-2.7b",
                model_type=ModelType.VISION_LANGUAGE,
                size=ModelSize.LARGE,
                specialized_for=["image_captioning", "visual_qa"]
            ),
            
            # Natural language models
            ModelConfig(
                name="T5-base",
                model_id="t5-base",
                model_type=ModelType.NATURAL_LANGUAGE,
                size=ModelSize.BASE,
                max_tokens=512,
                specialized_for=["text_generation", "summarization"]
            ),
            
            # Domain-specific models (simulated)
            ModelConfig(
                name="Plot-BERT",
                model_id="mock/plot-bert-base",  # This would be a custom trained model
                model_type=ModelType.DOMAIN_SPECIFIC,
                size=ModelSize.BASE,
                specialized_for=["visualization_terminology", "chart_understanding"],
                use_api=False,  # Would use local implementation
                use_local=True
            ),
            
            # Embedding models
            ModelConfig(
                name="SentenceTransformer",
                model_id="all-MiniLM-L6-v2",
                model_type=ModelType.EMBEDDING,
                size=ModelSize.SMALL,
                specialized_for=["semantic_similarity", "query_embedding"]
            )
        ]
        
        for model in models:
            self.available_models[model.name] = model
        
        logger.info(f"Registered {len(models)} HuggingFace models")
    
    async def _load_priority_models(self):
        """Load high-priority models locally."""
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, skipping local model loading")
            return
        
        priority_models = [
            name for name, config in self.available_models.items()
            if config.priority > 0.8 and config.use_local
        ]
        
        for model_name in priority_models:
            try:
                await self._load_model_locally(model_name)
            except Exception as e:
                logger.warning(f"Failed to load {model_name} locally: {e}")
    
    async def _load_model_locally(self, model_name: str):
        """Load a model locally."""
        
        model_config = self.available_models.get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if model_name == "Plot-BERT":
            # Mock implementation for domain-specific model
            self.loaded_models[model_name] = {"type": "mock", "ready": True}
            self.tokenizers[model_name] = {"type": "mock"}
            logger.info(f"Loaded mock model: {model_name}")
            return
        
        # Load actual model and tokenizer
        try:
            if model_config.model_type == ModelType.CODE_GENERATION:
                tokenizer = T5Tokenizer.from_pretrained(model_config.model_id)
                model = T5ForConditionalGeneration.from_pretrained(model_config.model_id)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
                model = AutoModel.from_pretrained(model_config.model_id)
            
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Loaded model locally: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            raise
    
    async def _test_api_connectivity(self):
        """Test connectivity to HuggingFace API."""
        
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests not available, skipping API connectivity test")
            return
        
        # Simple test with a small model
        test_model = "microsoft/DialoGPT-medium"
        test_url = f"{self.api_base_url}{test_model}"
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                test_url,
                headers=headers,
                json={"inputs": "Hello"},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("HuggingFace API connectivity confirmed")
            else:
                logger.warning(f"API test returned status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"API connectivity test failed: {e}")
    
    async def invoke_model(self, model_name: str, query_context, reasoning_results: Dict) -> Dict[str, Any]:
        """Invoke a specific model for processing."""
        logger.info(f"Invoking model: {model_name}")
        
        model_config = self.available_models.get(model_name)
        if not model_config:
            return {"error": f"Model {model_name} not found"}
        
        start_time = datetime.now()
        
        try:
            # Prepare input based on model type
            model_input = await self._prepare_model_input(model_config, query_context, reasoning_results)
            
            # Invoke model
            if model_config.use_local and model_name in self.loaded_models:
                result = await self._invoke_local_model(model_name, model_input)
            elif model_config.use_api:
                result = await self._invoke_api_model(model_name, model_input)
            else:
                return {"error": f"No inference method available for {model_name}"}
            
            # Calculate metrics
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Record invocation
            invocation = ModelInvocation(
                model_name=model_name,
                input_data=model_input,
                output_data=result,
                tokens_used=result.get('tokens_used', 0),
                latency_ms=latency_ms,
                success=True,
                timestamp=start_time
            )
            self.invocation_history.append(invocation)
            
            # Update performance metrics
            await self._update_performance_metrics(model_name, invocation)
            
            return {
                "model_name": model_name,
                "result": result,
                "latency_ms": latency_ms,
                "tokens_used": result.get('tokens_used', 0),
                "confidence": result.get('confidence', 0.7)
            }
            
        except Exception as e:
            logger.error(f"Error invoking {model_name}: {e}")
            
            # Record failed invocation
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            invocation = ModelInvocation(
                model_name=model_name,
                input_data=query_context.query if hasattr(query_context, 'query') else str(query_context),
                output_data=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
                timestamp=start_time
            )
            self.invocation_history.append(invocation)
            
            return {"error": str(e), "model_name": model_name}
    
    async def _prepare_model_input(self, model_config: ModelConfig, query_context, reasoning_results: Dict) -> str:
        """Prepare input for a specific model."""
        
        if model_config.model_type == ModelType.CODE_GENERATION:
            # Prepare code generation input
            query = getattr(query_context, 'query', str(query_context))
            prompt = f"Generate D3.js code for: {query}\n\n// D3.js Code:"
            return prompt
        
        elif model_config.model_type == ModelType.VISION_LANGUAGE:
            # Prepare vision-language input
            query = getattr(query_context, 'query', str(query_context))
            # Would include image data from sketches if available
            return f"Analyze this visualization request: {query}"
        
        elif model_config.model_type == ModelType.NATURAL_LANGUAGE:
            # Prepare natural language input
            query = getattr(query_context, 'query', str(query_context))
            return f"Improve this visualization description: {query}"
        
        elif model_config.model_type == ModelType.DOMAIN_SPECIFIC:
            # Prepare domain-specific input
            query = getattr(query_context, 'query', str(query_context))
            return f"Extract visualization concepts from: {query}"
        
        else:
            # Default preparation
            return getattr(query_context, 'query', str(query_context))
    
    async def _invoke_local_model(self, model_name: str, model_input: str) -> Dict[str, Any]:
        """Invoke a locally loaded model."""
        
        model = self.loaded_models[model_name]
        tokenizer = self.tokenizers[model_name]
        model_config = self.available_models[model_name]
        
        if model_name == "Plot-BERT":
            # Mock domain-specific model
            return {
                "output": f"Domain analysis: {model_input[:50]}...",
                "confidence": 0.8,
                "tokens_used": len(model_input.split()),
                "concepts_extracted": ["chart", "visualization", "data"]
            }
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available for local inference")
        
        # Tokenize input
        inputs = tokenizer(
            model_input,
            max_length=model_config.max_tokens,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate output
        with torch.no_grad():
            if model_config.model_type == ModelType.CODE_GENERATION:
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=model_config.max_tokens,
                    temperature=model_config.temperature,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                outputs = model(**inputs)
                # For non-generative models, return embeddings or classification
                generated_text = f"Model output processed (shape: {outputs.last_hidden_state.shape})"
        
        return {
            "output": generated_text,
            "confidence": 0.8,
            "tokens_used": len(inputs.input_ids[0])
        }
    
    async def _invoke_api_model(self, model_name: str, model_input: str) -> Dict[str, Any]:
        """Invoke a model via HuggingFace API."""
        
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("Requests not available for API calls")
        
        model_config = self.available_models[model_name]
        api_url = f"{self.api_base_url}{model_config.model_id}"
        
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare API payload
        payload = {
            "inputs": model_input,
            "parameters": {
                "max_new_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "return_full_text": False
            }
        }
        
        # Make API request
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse response based on model type
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    output = result[0]["generated_text"]
                else:
                    output = str(result[0])
            else:
                output = str(result)
            
            return {
                "output": output,
                "confidence": 0.7,
                "tokens_used": len(model_input.split()) + len(output.split())
            }
        else:
            raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")
    
    async def multi_model_inference(self, query_context, reasoning_results: Dict, 
                                  model_names: Optional[List[str]] = None) -> MultiModelResponse:
        """Perform inference with multiple models and return consensus."""
        logger.info("Performing multi-model inference...")
        
        if model_names is None:
            # Select models based on query characteristics
            model_names = await self._select_models_for_query(query_context)
        
        start_time = datetime.now()
        
        # Invoke all models concurrently
        tasks = []
        for model_name in model_names:
            task = asyncio.create_task(
                self.invoke_model(model_name, query_context, reasoning_results)
            )
            tasks.append((model_name, task))
        
        # Collect results
        all_responses = {}
        successful_responses = {}
        
        for model_name, task in tasks:
            try:
                result = await task
                all_responses[model_name] = result
                if "error" not in result:
                    successful_responses[model_name] = result
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                all_responses[model_name] = {"error": str(e)}
        
        # Calculate consensus and select best response
        if successful_responses:
            best_model, primary_response, consensus_score = await self._calculate_consensus(
                successful_responses
            )
            confidence = np.mean([r.get('confidence', 0.5) for r in successful_responses.values()])
        else:
            best_model = model_names[0] if model_names else "none"
            primary_response = {"error": "All models failed"}
            consensus_score = 0.0
            confidence = 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MultiModelResponse(
            primary_response=primary_response,
            all_responses=all_responses,
            consensus_score=consensus_score,
            best_model=best_model,
            confidence=confidence,
            processing_time_ms=processing_time
        )
    
    async def _select_models_for_query(self, query_context) -> List[str]:
        """Select appropriate models based on query characteristics."""
        
        selected_models = []
        
        # Always include a code generation model
        if hasattr(query_context, 'complexity_score') and query_context.complexity_score > 0.7:
            selected_models.append("CodeT5-large")
        else:
            selected_models.append("CodeT5-base")
        
        # Add domain-specific model
        selected_models.append("Plot-BERT")
        
        # Add vision-language model if visual complexity is high
        if hasattr(query_context, 'visual_complexity') and query_context.visual_complexity > 0.6:
            selected_models.append("CLIP")
        
        # Add natural language model for query refinement
        selected_models.append("T5-base")
        
        return selected_models
    
    async def _calculate_consensus(self, responses: Dict[str, Dict]) -> Tuple[str, Dict, float]:
        """Calculate consensus among model responses."""
        
        # Simple consensus calculation
        # In practice, this would be much more sophisticated
        
        scores = {}
        for model_name, response in responses.items():
            confidence = response.get('confidence', 0.5)
            latency = response.get('latency_ms', 1000)
            
            # Score based on confidence and speed
            score = confidence * 0.7 + (1000 / max(latency, 100)) * 0.3
            scores[model_name] = score
        
        # Select best model
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        primary_response = responses[best_model]
        
        # Calculate consensus score
        avg_confidence = np.mean([r.get('confidence', 0.5) for r in responses.values()])
        consensus_score = avg_confidence * (len(responses) / len(self.available_models))
        
        return best_model, primary_response, consensus_score
    
    async def _update_performance_metrics(self, model_name: str, invocation: ModelInvocation):
        """Update performance metrics for a model."""
        
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                "total_invocations": 0,
                "successful_invocations": 0,
                "avg_latency_ms": 0.0,
                "avg_tokens_per_second": 0.0,
                "avg_confidence": 0.0
            }
        
        metrics = self.model_performance[model_name]
        metrics["total_invocations"] += 1
        
        if invocation.success:
            metrics["successful_invocations"] += 1
            
            # Update moving averages
            alpha = 0.1  # Learning rate for moving average
            metrics["avg_latency_ms"] = (
                (1 - alpha) * metrics["avg_latency_ms"] +
                alpha * invocation.latency_ms
            )
            
            if invocation.latency_ms > 0 and invocation.tokens_used > 0:
                tokens_per_second = (invocation.tokens_used / invocation.latency_ms) * 1000
                metrics["avg_tokens_per_second"] = (
                    (1 - alpha) * metrics["avg_tokens_per_second"] +
                    alpha * tokens_per_second
                )
            
            # Update confidence if available in output
            if isinstance(invocation.output_data, dict) and "confidence" in invocation.output_data:
                confidence = invocation.output_data["confidence"]
                metrics["avg_confidence"] = (
                    (1 - alpha) * metrics["avg_confidence"] +
                    alpha * confidence
                )
    
    def is_ready(self) -> bool:
        """Check if the HuggingFace hub is ready."""
        return self.ready
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        
        total_invocations = len(self.invocation_history)
        successful_invocations = sum(1 for inv in self.invocation_history if inv.success)
        
        return {
            "ready": self.ready,
            "total_models_registered": len(self.available_models),
            "models_loaded_locally": len(self.loaded_models),
            "total_invocations": total_invocations,
            "successful_invocations": successful_invocations,
            "success_rate": successful_invocations / max(1, total_invocations),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "api_available": REQUESTS_AVAILABLE and bool(self.api_key)
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        return self.model_performance.copy()
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        return {
            name: {
                "model_id": config.model_id,
                "model_type": config.model_type.value,
                "size": config.size.value,
                "specialized_for": config.specialized_for,
                "use_local": config.use_local,
                "use_api": config.use_api,
                "loaded_locally": name in self.loaded_models
            }
            for name, config in self.available_models.items()
        }