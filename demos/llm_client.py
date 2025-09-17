"""
Simple OpenAI API client for the demo

This handles communication with ChatGPT (free tier) and provides
clean error handling and retry logic.
"""

import openai
import asyncio
import time
from typing import Optional, Dict, Any

class SimpleLLMClient:
    """Simple OpenAI client for demo purposes"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up OpenAI client
        openai.api_key = api_key
        
        # Simple rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests for free tier
    
    async def get_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get completion from OpenAI API with error handling"""
        
        # Simple rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            print(f"🤖 Querying {self.model}...")
            
            # Use the synchronous API (OpenAI library handles async internally)
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.last_request_time = time.time()
            
            content = response.choices[0].message.content.strip()
            print(f"✅ Response received ({len(content)} characters)")
            
            return content
            
        except openai.error.RateLimitError as e:
            print(f"⚠️ Rate limit hit, waiting 60 seconds...")
            await asyncio.sleep(60)
            return await self.get_completion(prompt, system_prompt)  # Retry
            
        except openai.error.AuthenticationError as e:
            error_msg = "❌ Authentication failed. Please check your OpenAI API key."
            print(error_msg)
            return f"Error: {error_msg}"
            
        except openai.error.APIError as e:
            error_msg = f"❌ OpenAI API error: {str(e)}"
            print(error_msg)
            return f"Error: {error_msg}"
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(error_msg)
            return f"Error: {error_msg}"
    
    def test_connection(self) -> bool:
        """Test if API key and connection work"""
        
        try:
            print("🔍 Testing OpenAI API connection...")
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'test' if you can hear me."}],
                max_tokens=10,
                temperature=0
            )
            
            content = response.choices[0].message.content.strip().lower()
            
            if "test" in content:
                print("✅ OpenAI API connection successful!")
                return True
            else:
                print(f"⚠️ Unexpected response: {content}")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False
