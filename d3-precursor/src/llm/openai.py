import os
import json
from typing import Dict, List, Optional, Any, Union
import tiktoken
import aiohttp
import asyncio

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI's API."""
    
    # Model context sizes
    MODEL_CONTEXT_SIZES = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4-32k": 32768,
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: The OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
            model: The OpenAI model to use.
        """
        super().__init__(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model=model
        )
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the OpenAI API.
        
        Args:
            prompt: The prompt to send to the API.
            **kwargs: Additional arguments for the API call.
            
        Returns:
            The response from the API.
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        # Convert to chat format
        messages = [
            {"role": "system", "content": kwargs.get("system_prompt", self.get_system_prompt())},
            {"role": "user", "content": prompt}
        ]
        
        return await self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    
    async def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate a JSON response from the OpenAI API.
        
        Args:
            prompt: The prompt to send to the API.
            json_schema: The JSON schema for the expected output.
            **kwargs: Additional arguments for the API call.
            
        Returns:
            The JSON response from the API.
        """
        formatted_prompt = self.format_json_prompt(prompt, json_schema)
        
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.2)  # Lower temperature for more deterministic JSON
        max_tokens = kwargs.get("max_tokens", 2000)
        
        response = await self.generate(
            formatted_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=kwargs.get("system_prompt", "You are a helpful assistant that responds only with valid JSON.")
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
        """Generate a response from the OpenAI API in chat format.
        
        Args:
            messages: The messages to send to the API, each with "role" and "content".
            **kwargs: Additional arguments for the API call.
            
        Returns:
            The response from the API.
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters if provided
        for param in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_info = await resp.text()
                    raise Exception(f"OpenAI API error: {resp.status} - {error_info}")
                
                data = await resp.json()
                
                # Extract response content
                content = data["choices"][0]["message"]["content"]
                
                # Calculate tokens used
                prompt_tokens = data["usage"]["prompt_tokens"]
                completion_tokens = data["usage"]["completion_tokens"]
                total_tokens = data["usage"]["total_tokens"]
                
                return LLMResponse(
                    content=content,
                    model=model,
                    tokens_used=total_tokens,
                    metadata={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "finish_reason": data["choices"][0]["finish_reason"]
                    }
                )
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The number of tokens.
        """
        # Use tiktoken to count tokens
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Default to cl100k_base encoding for newer models
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    def get_model_context_size(self, model: Optional[str] = None) -> int:
        """Get the context size (max tokens) for the specified model.
        
        Args:
            model: The model to get the context size for. If None, the default model is used.
            
        Returns:
            The maximum number of tokens the model can handle.
        """
        model = model or self.model
        return self.MODEL_CONTEXT_SIZES.get(model, 4096)  # Default to 4096 for unknown models 