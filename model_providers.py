import os
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import requests
import json

logger = logging.getLogger("multi_model_assistant.models")

class ModelProvider(ABC):
    """Abstract base class for all model providers."""
    
    @abstractmethod
    def generate_response(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a response for the given query and context.
        
        Args:
            query: The user's input query
            context: Optional context information
            
        Returns:
            Response dictionary with text and metadata
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass

class ClaudeProvider(ModelProvider):
    """Provider for Anthropic's Claude models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """Initialize the Claude provider.
        
        Args:
            api_key: Anthropic API key (defaults to environment variable)
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        if not self.api_key:
            logger.warning("No Claude API key provided. Claude provider will not function.")
        
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    
    def generate_response(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a response using Claude."""
        if not self.api_key:
            return {"text": "Claude API key not configured", "error": True}
        
        try:
            # Prepare system prompt based on context
            system_prompt = self._build_system_prompt(context)
            
            # Prepare memory context if available
            memory_text = ""
            if context and "memory" in context:
                memory_text = "\n\nRelevant past information:\n" + context["memory"]
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": query + memory_text}
                ],
                "max_tokens": 1000
            }
            
            # Make the API request
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text
            response_text = result["content"][0]["text"]
            
            # Calculate token usage
            input_tokens = self.get_token_count(query + memory_text + system_prompt)
            output_tokens = self.get_token_count(response_text)
            
            return {
                "text": response_text,
                "tokens_used": input_tokens + output_tokens,
                "model": self.model,
                "provider": "claude"
            }
            
        except Exception as e:
            logger.error(f"Error generating Claude response: {e}")
            return {"text": f"Error generating response: {str(e)}", "error": True}
    
    def _build_system_prompt(self, context: Optional[Dict]) -> str:
        """Build a system prompt based on context."""
        base_prompt = "You are a helpful, harmless, and honest AI assistant with expertise in reasoning and memory."
        
        # Add emotion context if available
        if context and "emotion" in context:
            base_prompt += f"\n\nThe user's current emotional state appears to be: {context['emotion']}. Please be mindful of this in your response."
        
        return base_prompt
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count for Claude."""
        # Simple estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

class DeepSeekProvider(ModelProvider):
    """Provider for DeepSeek models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """Initialize the DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key (defaults to environment variable)
            model: DeepSeek model to use
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("No DeepSeek API key provided. DeepSeek provider will not function.")
        
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a response using DeepSeek."""
        if not self.api_key:
            return {"text": "DeepSeek API key not configured", "error": True}
        
        try:
            # Prepare system prompt based on context
            system_prompt = self._build_system_prompt(context)
            
            # Prepare memory context if available
            memory_text = ""
            if context and "memory" in context:
                memory_text = "\n\nRelevant past information:\n" + context["memory"]
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query + memory_text}
                ],
                "max_tokens": 1000
            }
            
            # Make the API request
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text
            response_text = result["choices"][0]["message"]["content"]
            
            # Calculate token usage
            input_tokens = result.get("usage", {}).get("prompt_tokens", self.get_token_count(query + memory_text + system_prompt))
            output_tokens = result.get("usage", {}).get("completion_tokens", self.get_token_count(response_text))
            
            return {
                "text": response_text,
                "tokens_used": input_tokens + output_tokens,
                "model": self.model,
                "provider": "deepseek"
            }
            
        except Exception as e:
            logger.error(f"Error generating DeepSeek response: {e}")
            return {"text": f"Error generating response: {str(e)}", "error": True}
    
    def _build_system_prompt(self, context: Optional[Dict]) -> str:
        """Build a system prompt based on context."""
        base_prompt = "You are a helpful AI assistant with expertise in planning, logic, and internet search."
        
        # Add emotion context if available
        if context and "emotion" in context:
            base_prompt += f"\n\nThe user's current emotional state appears to be: {context['emotion']}. Please be mindful of this in your response."
        
        return base_prompt
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count for DeepSeek."""
        # Simple estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

class ChatGPTProvider(ModelProvider):
    """Provider for OpenAI's ChatGPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize the ChatGPT provider.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: ChatGPT model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. ChatGPT provider will not function.")
        
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a response using ChatGPT."""
        if not self.api_key:
            return {"text": "OpenAI API key not configured", "error": True}
        
        try:
            # Prepare system prompt based on context
            system_prompt = self._build_system_prompt(context)
            
            # Prepare memory context if available
            memory_text = ""
            if context and "memory" in context:
                memory_text = "\n\nRelevant past information:\n" + context["memory"]
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query + memory_text}
                ],
                "max_tokens": 1000
            }
            
            # Make the API request
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text
            response_text = result["choices"][0]["message"]["content"]
            
            # Calculate token usage
            input_tokens = result.get("usage", {}).get("prompt_tokens", self.get_token_count(query + memory_text + system_prompt))
            output_tokens = result.get("usage", {}).get("completion_tokens", self.get_token_count(response_text))
            
            return {
                "text": response_text,
                "tokens_used": input_tokens + output_tokens,
                "model": self.model,
                "provider": "chatgpt"
            }
            
        except Exception as e:
            logger.error(f"Error generating ChatGPT response: {e}")
            return {"text": f"Error generating response: {str(e)}", "error": True}
    
    def _build_system_prompt(self, context: Optional[Dict]) -> str:
        """Build a system prompt based on context."""
        base_prompt = "You are a helpful AI assistant with a human-like conversational style."
        
        # Add emotion context if available
        if context and "emotion" in context:
            base_prompt += f"\n\nThe user's current emotional state appears to be: {context['emotion']}. Please be mindful of this in your response."
        
        return base_prompt
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count for ChatGPT."""
        # Simple estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

class ModelProviderFactory:
    """Factory for creating model providers."""
    
    @staticmethod
    def create_provider(provider_type: str, api_key: Optional[str] = None) -> ModelProvider:
        """Create a model provider of the specified type.
        
        Args:
            provider_type: Type of provider ('claude', 'deepseek', 'chatgpt')
            api_key: Optional API key
            
        Returns:
            ModelProvider instance
        """
        if provider_type == "claude":
            return ClaudeProvider(api_key)
        elif provider_type == "deepseek":
            return DeepSeekProvider(api_key)
        elif provider_type == "chatgpt":
            return ChatGPTProvider(api_key)
        else:
            logger.error(f"Unknown provider type: {provider_type}")
            raise ValueError(f"Unknown provider type: {provider_type}")

# Update the MultiModelAssistant._initialize_models method to use this factory
def initialize_models_for_assistant(assistant):
    """Initialize model providers for the assistant.
    
    Args:
        assistant: MultiModelAssistant instance
    """
    for model_type, model_config in assistant.config["models"].items():
        provider_type = model_config["provider"]
        api_key_env = model_config.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        
        try:
            provider = ModelProviderFactory.create_provider(provider_type, api_key)
            assistant.models[provider_type] = provider
            logger.info(f"Initialized {provider_type} provider for {model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize {provider_type} provider: {e}")
            
    # Initialize fallback providers if not already initialized
    for model_type, model_config in assistant.config["models"].items():
        fallback = model_config.get("fallback")
        if fallback and fallback not in assistant.models:
            api_key_env = assistant.config["models"].get(fallback, {}).get("api_key_env")
            api_key = os.environ.get(api_key_env) if api_key_env else None
            
            try:
                provider = ModelProviderFactory.create_provider(fallback, api_key)
                assistant.models[fallback] = provider
                logger.info(f"Initialized {fallback} provider as fallback")
            except Exception as e:
                logger.error(f"Failed to initialize {fallback} fallback provider: {e}")