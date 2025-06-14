from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import json


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    tokens_used: int
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """Create a response from a dictionary."""
        return cls(
            content=data["content"],
            model=data["model"],
            tokens_used=data["tokens_used"],
            metadata=data.get("metadata")
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the LLM provider.
        
        Args:
            api_key: The API key for the LLM provider.
            model: The default model to use.
        """
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM.
            **kwargs: Additional arguments for the provider.
            
        Returns:
            The response from the LLM.
        """
        pass
    
    @abstractmethod
    async def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate a JSON response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM.
            json_schema: The JSON schema for the expected output.
            **kwargs: Additional arguments for the provider.
            
        Returns:
            The JSON response from the LLM.
        """
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM in chat format.
        
        Args:
            messages: The messages to send to the LLM, each with "role" and "content".
            **kwargs: Additional arguments for the provider.
            
        Returns:
            The response from the LLM.
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The number of tokens.
        """
        pass
    
    def format_json_prompt(self, prompt: str, schema: Dict[str, Any]) -> str:
        """Format a prompt to instruct the model to return JSON.
        
        Args:
            prompt: The base prompt.
            schema: The JSON schema for the expected output.
            
        Returns:
            The formatted prompt.
        """
        schema_str = json.dumps(schema, indent=2)
        return f"""{prompt}

Your response must be valid JSON that conforms to the following schema:
{schema_str}

Respond with only the JSON object, no other text."""

    def get_system_prompt(self) -> str:
        """Get the default system prompt for the provider.
        
        Returns:
            The default system prompt.
        """
        return """You are an AI assistant specialized in the D3.js library, 
helping with code generation, documentation, and API knowledge. 
Provide clear, concise, and helpful responses."""
    
    def get_model_context_size(self, model: Optional[str] = None) -> int:
        """Get the context size (max tokens) for the specified model.
        
        Args:
            model: The model to get the context size for. If None, the default model is used.
            
        Returns:
            The maximum number of tokens the model can handle.
        """
        # Default implementation returns a conservative value
        return 4096 