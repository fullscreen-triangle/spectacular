import os
import json
from typing import Dict, List, Optional, Any, Union
import aiohttp
import asyncio
import re

from .base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic's Claude API."""
    
    # Model context sizes
    MODEL_CONTEXT_SIZES = {
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-2.1": 200000,
        "claude-2.0": 100000,
        "claude-instant-1.2": 100000,
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize the Anthropic provider.
        
        Args:
            api_key: The Anthropic API key. If None, reads from ANTHROPIC_API_KEY environment variable.
            model: The Anthropic model to use.
        """
        super().__init__(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            model=model
        )
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the Anthropic API.
        
        Args:
            prompt: The prompt to send to the API.
            **kwargs: Additional arguments for the API call.
            
        Returns:
            The response from the API.
        """
        system_prompt = kwargs.get("system_prompt", self.get_system_prompt())
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        return await self.chat(
            messages, 
            system=system_prompt,
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000)
        )
    
    async def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate a JSON response from the Anthropic API.
        
        Args:
            prompt: The prompt to send to the API.
            json_schema: The JSON schema for the expected output.
            **kwargs: Additional arguments for the API call.
            
        Returns:
            The JSON response from the API.
        """
        formatted_prompt = self.format_json_prompt(prompt, json_schema)
        system_prompt = kwargs.get(
            "system_prompt", 
            "You are a helpful assistant that responds only with valid JSON."
        )
        
        response = await self.generate(
            formatted_prompt,
            system_prompt=system_prompt,
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", 0.2),  # Lower temperature for more deterministic JSON
            max_tokens=kwargs.get("max_tokens", 2000)
        )
        
        # Parse the JSON response
        try:
            # Clean the response if it has markdown code block formatting
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Response: {response.content}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the Anthropic API in chat format.
        
        Args:
            messages: The messages to send to the API, each with "role" and "content".
            **kwargs: Additional arguments for the API call.
            
        Returns:
            The response from the API.
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system = kwargs.get("system", self.get_system_prompt())
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters if provided
        for param in ["top_p", "top_k", "stop_sequences"]:
            if param in kwargs:
                payload[param] = kwargs[param]
                
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_info = await resp.text()
                    raise Exception(f"Anthropic API error: {resp.status} - {error_info}")
                
                data = await resp.json()
                
                # Extract response content
                content = data["content"][0]["text"]
                
                # Estimate tokens used (Anthropic doesn't provide token counts directly in v1)
                # Roughly 4 characters per token for English text
                input_text = system + "\n" + "\n".join(msg["content"] for msg in messages)
                input_tokens = self.count_tokens(input_text)
                output_tokens = self.count_tokens(content)
                total_tokens = input_tokens + output_tokens
                
                return LLMResponse(
                    content=content,
                    model=model,
                    tokens_used=total_tokens,
                    metadata={
                        "message_id": data.get("id"),
                        "estimated_input_tokens": input_tokens,
                        "estimated_output_tokens": output_tokens
                    }
                )
    
    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the given text.
        
        This is a rough estimate as Anthropic doesn't provide a public tokenizer.
        For English text, ~4 characters per token is a reasonable estimate.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The estimated number of tokens.
        """
        # Remove whitespace to get a more accurate character count
        text = re.sub(r'\s+', ' ', text).strip()
        
        # About 4 characters per token for English text
        return max(1, len(text) // 4)
    
    def get_model_context_size(self, model: Optional[str] = None) -> int:
        """Get the context size (max tokens) for the specified model.
        
        Args:
            model: The model to get the context size for. If None, the default model is used.
            
        Returns:
            The maximum number of tokens the model can handle.
        """
        model = model or self.model
        return self.MODEL_CONTEXT_SIZES.get(model, 100000)  # Default to 100k for unknown models 