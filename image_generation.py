import os
import logging
import requests
import base64
import tempfile
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger("multi_model_assistant.image")

class ImageGenerator(ABC):
    """Abstract base class for image generation systems."""
    
    @abstractmethod
    def generate_image(self, prompt: str, size: str = "512x512") -> Dict[str, Any]:
        """Generate an image based on a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            size: Size of the image to generate
            
        Returns:
            Dictionary with image data and metadata
        """
        pass

class StableDiffusionGenerator(ImageGenerator):
    """Image generator using Stable Diffusion via free API."""
    
    def __init__(self):
        """Initialize Stable Diffusion image generator."""
        # Using Hugging Face Inference API (free tier)
        self.api_key = os.environ.get("HF_API_KEY")
        self.use_api = self.api_key is not None
        
        if not self.use_api:
            logger.warning("No Hugging Face API key provided. Image generation will be limited.")
    
    def generate_image(self, prompt: str, size: str = "512x512") -> Dict[str, Any]:
        """Generate an image using Stable Diffusion."""
        if self.use_api:
            return self._generate_with_api(prompt, size)
        else:
            return {"error": "No API key provided for image generation", "image_data": None}
    
    def _generate_with_api(self, prompt: str, size: str) -> Dict[str, Any]:
        """Generate image using Hugging Face Inference API."""
        try:
            # Using a free Stable Diffusion model on Hugging Face
            API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "width": width,
                    "height": height
                }
            }
            
            # Make request
            response = requests.post(API_URL, headers=headers, json=payload)
            
            # Handle response
            if response.status_code == 200:
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file.write(response.content)
                    image_path = temp_file.name
                
                # Convert to base64 for easier handling
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                
                return {
                    "image_data": response.content,
                    "image_base64": image_base64,
                    "image_path": image_path,
                    "prompt": prompt,
                    "size": size
                }
            else:
                logger.error(f"Error generating image: {response.status_code} - {response.text}")
                return {"error": f"Error generating image: {response.status_code}", "image_data": None}
                
        except Exception as e:
            logger.error(f"Error generating image with API: {e}")
            return {"error": str(e), "image_data": None}

class DalleMiniGenerator(ImageGenerator):
    """Image generator using DALL-E Mini (free alternative)."""
    
    def __init__(self):
        """Initialize DALL-E Mini image generator."""
        # Using public API for DALL-E Mini
        pass
    
    def generate_image(self, prompt: str, size: str = "256x256") -> Dict[str, Any]:
        """Generate an image using DALL-E Mini."""
        try:
            # Using a free DALL-E Mini API
            API_URL = "https://bf.dallemini.ai/generate"
            
            # Prepare payload
            payload = {"prompt": prompt}
            
            # Make request
            response = requests.post(API_URL, json=payload)
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                if "images" in result and len(result["images"]) > 0:
                    # Get first image
                    image_base64 = result["images"][0]
                    image_data = base64.b64decode(image_base64)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file.write(image_data)
                        image_path = temp_file.name
                    
                    return {
                        "image_data": image_data,
                        "image_base64": image_base64,
                        "image_path": image_path,
                        "prompt": prompt,
                        "size": size
                    }
                else:
                    logger.error("No images returned from API")
                    return {"error": "No images returned", "image_data": None}
            else:
                logger.error(f"Error generating image: {response.status_code} - {response.text}")
                return {"error": f"Error generating image: {response.status_code}", "image_data": None}
                
        except Exception as e:
            logger.error(f"Error generating image with DALL-E Mini: {e}")
            return {"error": str(e), "image_data": None}

def create_image_generator(config: Dict) -> ImageGenerator:
    """Create an image generator based on configuration.
    
    Args:
        config: Image generation configuration dictionary
        
    Returns:
        ImageGenerator instance
    """
    provider = config.get("provider", "stable-diffusion")
    
    if provider == "stable-diffusion":
        return StableDiffusionGenerator()
    elif provider == "dalle-mini":
        return DalleMiniGenerator()
    else:
        logger.warning(f"Unknown image provider: {provider}. Using Stable Diffusion as fallback.")
        return StableDiffusionGenerator()

# Update the MultiModelAssistant initialization methods
def initialize_image_generation_for_assistant(assistant):
    """Initialize image generation for the assistant.
    
    Args:
        assistant: MultiModelAssistant instance
    """
    try:
        image_config = assistant.config.get("image", {"provider": "stable-diffusion"})
        assistant.image_generator = create_image_generator(image_config)
        logger.info(f"Initialized image generator: {image_config['provider']}")
    except Exception as e:
        logger.error(f"Failed to initialize image generator: {e}")
        assistant.image_generator = None