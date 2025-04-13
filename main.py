import os
import json
import logging
from typing import Dict, List, Optional, Any
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_model_assistant")

class MultiModelAssistant:
    """Main controller for the multi-model AI assistant that integrates multiple LLM services."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the assistant with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.memory_system = None
        self.voice_system = None
        self.emotion_detector = None
        self.image_generator = None
        self.credit_tracker = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Multi-model assistant initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Use default configuration if file doesn't exist
                logger.warning(f"Config file {config_path} not found. Using default configuration.")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "models": {
                "reasoning": {
                    "provider": "claude",
                    "fallback": "deepseek",
                    "api_key_env": "CLAUDE_API_KEY"
                },
                "planning": {
                    "provider": "deepseek",
                    "fallback": None,
                    "api_key_env": "DEEPSEEK_API_KEY"
                },
                "conversation": {
                    "provider": "chatgpt",
                    "fallback": "deepseek",
                    "api_key_env": "OPENAI_API_KEY"
                }
            },
            "memory": {
                "provider": "chroma",  # Free vector DB
                "collection_name": "assistant_memory"
            },
            "voice": {
                "provider": "coqui",  # Free voice cloning
                "voice_id": "default"
            },
            "image": {
                "provider": "stable-diffusion"  # Free image generation
            },
            "credit_limits": {
                "claude": 1000,
                "deepseek": 5000,
                "chatgpt": 2000
            }
        }
    
    def _initialize_components(self):
        """Initialize all components based on configuration."""
        # Initialize model providers
        self._initialize_models()
        
        # Initialize memory system
        self._initialize_memory()
        
        # Initialize voice system
        self._initialize_voice()
        
        # Initialize emotion detection
        self._initialize_emotion_detection()
        
        # Initialize image generation
        self._initialize_image_generation()
    
    def _initialize_models(self):
        """Initialize AI model providers."""
        logger.info("Initializing AI models")
        from model_providers import initialize_models_for_assistant
        initialize_models_for_assistant(self)
    
    def _initialize_memory(self):
        """Initialize memory and vector database."""
        logger.info("Initializing memory system")
        from memory_system import initialize_memory_for_assistant
        initialize_memory_for_assistant(self)
    
    def _initialize_voice(self):
        """Initialize voice synthesis and recognition."""
        logger.info("Initializing voice system")
        from voice_system import initialize_voice_for_assistant
        initialize_voice_for_assistant(self)
    
    def _initialize_emotion_detection(self):
        """Initialize emotion detection system."""
        logger.info("Initializing emotion detection")
        from voice_system import initialize_emotion_detection_for_assistant
        initialize_emotion_detection_for_assistant(self)
    
    def _initialize_image_generation(self):
        """Initialize image generation system."""
        logger.info("Initializing image generation")
        from image_generation import initialize_image_generation_for_assistant
        initialize_image_generation_for_assistant(self)
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user query through the appropriate models.
        
        Args:
            query: The user's input query
            context: Optional context information
            
        Returns:
            Response dictionary with text and any additional data
        """
        logger.info(f"Processing query: {query[:50]}...")
        
        # Detect emotion if voice input
        emotion = None
        if self.emotion_detector and context and context.get('input_type') == 'voice':
            emotion = self.emotion_detector.detect_emotion(query)
            logger.info(f"Detected emotion: {emotion}")
        
        # Determine which model to use based on query type
        model_type = self._determine_model_type(query, context)
        
        # Get response from appropriate model
        response = self._get_model_response(model_type, query, context, emotion)
        
        # Save to memory
        if self.memory_system:
            self.memory_system.save_interaction(query, response)
        
        return response
    
    def _determine_model_type(self, query: str, context: Optional[Dict]) -> str:
        """Determine which type of model should handle this query.
        
        Args:
            query: The user's input query
            context: Optional context information
            
        Returns:
            String indicating model type ('reasoning', 'planning', or 'conversation')
        """
        # Simple keyword-based routing for now
        # This could be enhanced with a classifier model later
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['why', 'explain', 'understand', 'reason']):
            return 'reasoning'
        elif any(word in query_lower for word in ['plan', 'steps', 'organize', 'search', 'find', 'look up']):
            return 'planning'
        else:
            return 'conversation'
    
    def _get_model_response(self, model_type: str, query: str, 
                           context: Optional[Dict], emotion: Optional[str]) -> Dict[str, Any]:
        """Get response from the appropriate model based on type.
        
        Args:
            model_type: Type of model to use ('reasoning', 'planning', 'conversation')
            query: The user's input query
            context: Optional context information
            emotion: Detected emotion, if available
            
        Returns:
            Response dictionary with text and any additional data
        """
        # Check if primary model has available credits
        model_config = self.config['models'].get(model_type, self.config['models']['conversation'])
        provider = model_config['provider']
        
        # Check if we need to use fallback
        if self._check_credits(provider):
            # Use primary model
            model = self.models.get(provider)
        else:
            # Use fallback model
            fallback = model_config.get('fallback')
            if fallback and self._check_credits(fallback):
                model = self.models.get(fallback)
                logger.info(f"Using fallback model {fallback} for {model_type}")
            else:
                # Default to deepseek as final fallback
                model = self.models.get('deepseek')
                logger.warning(f"Using deepseek as emergency fallback for {model_type}")
        
        # Get relevant memory if available
        memory_context = None
        if self.memory_system:
            memory_context = self.memory_system.retrieve_relevant(query)
        
        # Combine all context
        full_context = {}
        if context:
            full_context.update(context)
        if memory_context:
            full_context['memory'] = memory_context
        if emotion:
            full_context['emotion'] = emotion
        
        # Get response from model
        response = model.generate_response(query, full_context)
        
        # Track credit usage
        self._update_credits(provider, response.get('tokens_used', 0))
        
        return response
    
    def _check_credits(self, provider: str) -> bool:
        """Check if the provider has available credits.
        
        Args:
            provider: The model provider name
            
        Returns:
            Boolean indicating if credits are available
        """
        limit = self.config['credit_limits'].get(provider, 0)
        used = self.credit_tracker.get(provider, 0)
        return used < limit
    
    def _update_credits(self, provider: str, tokens_used: int):
        """Update the credit usage for a provider.
        
        Args:
            provider: The model provider name
            tokens_used: Number of tokens used in this interaction
        """
        current = self.credit_tracker.get(provider, 0)
        self.credit_tracker[provider] = current + tokens_used
        logger.info(f"Updated {provider} credits: {current + tokens_used}/{self.config['credit_limits'].get(provider, 0)}")

    def process_voice_input(self, audio_data):
        """Process voice input, transcribe it, and handle the query.
        
        Args:
            audio_data: Audio data from the user
            
        Returns:
            Response dictionary with text and audio response
        """
        if not self.voice_system:
            return {"error": "Voice system not initialized"}
        
        # Transcribe audio to text
        text = self.voice_system.transcribe(audio_data)
        
        # Process the text query
        context = {"input_type": "voice"}
        response = self.process_query(text, context)
        
        # Convert response to speech
        audio_response = self.voice_system.synthesize(response["text"])
        
        # Add audio to response
        response["audio"] = audio_response
        
        return response

    def generate_image(self, prompt: str, size: str = "512x512") -> Dict[str, Any]:
        """Generate an image based on a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            size: Size of the image to generate
            
        Returns:
            Dictionary with image data and metadata
        """
        logger.info(f"Generating image for prompt: {prompt[:50]}...")
        
        if not self.image_generator:
            return {"error": "Image generation not initialized", "image_data": None}
        
        # Generate image
        result = self.image_generator.generate_image(prompt, size)
        
        # Save to memory if available
        if self.memory_system and not result.get("error"):
            memory_text = f"Generated image for prompt: {prompt}"
            self.memory_system.save_interaction(prompt, {"text": memory_text})
        
        return result

    def process_internet_search(self, query: str) -> Dict[str, Any]:
        """Process an internet search query.
        
        Args:
            query: The search query
            
        Returns:
            Response dictionary with search results
        """
        logger.info(f"Processing internet search: {query[:50]}...")
        
        # Use DeepSeek for internet search
        model_config = self.config['models'].get('planning')
        provider = model_config['provider']
        
        # Check if we need to use fallback
        if self._check_credits(provider):
            # Use primary model
            model = self.models.get(provider)
        else:
            # Use fallback model
            fallback = model_config.get('fallback')
            if fallback and self._check_credits(fallback):
                model = self.models.get(fallback)
                logger.info(f"Using fallback model {fallback} for internet search")
            else:
                # Default to deepseek as final fallback
                model = self.models.get('deepseek')
                logger.warning(f"Using deepseek as emergency fallback for internet search")
        
        # Add search context
        context = {"task": "internet_search", "query": query}
        
        # Get response from model
        response = model.generate_response(f"Search the internet for: {query}", context)
        
        # Track credit usage
        self._update_credits(provider, response.get('tokens_used', 0))
        
        return response

# Example usage
if __name__ == "__main__":
    assistant = MultiModelAssistant()
    
    # Example text query
    response = assistant.process_query("Why is the sky blue?")
    print(f"Response: {response['text']}")
    
    # Example internet search
    # search_response = assistant.process_internet_search("Latest news about AI")
    # print(f"Search results: {search_response['text']}")
    
    # Example image generation
    # image_result = assistant.generate_image("A beautiful sunset over mountains")
    # if not image_result.get("error"):
    #     print(f"Image generated and saved to: {image_result['image_path']}")
    
    # This would be replaced with actual audio data in a real application
    # response = assistant.process_voice_input(audio_data)