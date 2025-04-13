import torch
import numpy as np
from transformers import pipeline
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger("multi_model_assistant.emotion")

class EmotionDetector:
    """Class for detecting emotions from voice and text input."""
    
    def __init__(self):
        """Initialize emotion detection models."""
        try:
            # Initialize text emotion detection pipeline
            self.text_emotion = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=3
            )
            
            # Initialize voice emotion detection model
            # Using a lightweight model for voice emotion detection
            self.voice_emotion = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-speech-commands-v2",
                top_k=3
            )
            
            logger.info("Emotion detection models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing emotion detection models: {e}")
            self.text_emotion = None
            self.voice_emotion = None
    
    def detect_from_text(self, text: str) -> List[Dict[str, float]]:
        """Detect emotions from text input.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing emotion labels and scores
        """
        try:
            if not self.text_emotion:
                return [{"label": "neutral", "score": 1.0}]
            
            results = self.text_emotion(text)
            return [{
                "label": result["label"],
                "score": float(result["score"])
            } for result in results]
        except Exception as e:
            logger.error(f"Error detecting emotions from text: {e}")
            return [{"label": "error", "score": 1.0}]
    
    def detect_from_voice(self, audio_data: bytes) -> List[Dict[str, float]]:
        """Detect emotions from voice input.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            List of dictionaries containing emotion labels and scores
        """
        try:
            if not self.voice_emotion:
                return [{"label": "neutral", "score": 1.0}]
            
            # Convert audio data to format expected by model
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            results = self.voice_emotion(temp_path)
            
            # Clean up temporary file
            import os
            os.unlink(temp_path)
            
            return [{
                "label": result["label"],
                "score": float(result["score"])
            } for result in results]
        except Exception as e:
            logger.error(f"Error detecting emotions from voice: {e}")
            return [{"label": "error", "score": 1.0}]
    
    def combine_emotions(self, text_emotions: List[Dict[str, float]], 
                        voice_emotions: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine emotions detected from text and voice.
        
        Args:
            text_emotions: Emotions detected from text
            voice_emotions: Emotions detected from voice
            
        Returns:
            Dictionary containing combined emotion scores
        """
        # Combine scores with weights (0.6 for text, 0.4 for voice)
        combined = {}
        
        # Process text emotions
        for emotion in text_emotions:
            combined[emotion["label"]] = emotion["score"] * 0.6
        
        # Process voice emotions
        for emotion in voice_emotions:
            if emotion["label"] in combined:
                combined[emotion["label"]] += emotion["score"] * 0.4
            else:
                combined[emotion["label"]] = emotion["score"] * 0.4
        
        return combined