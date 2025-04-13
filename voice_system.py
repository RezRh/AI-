import os
import logging
from typing import Dict, List, Optional, Any
import tempfile
import base64
import requests
from abc import ABC, abstractmethod
from emotion_detector import EmotionDetector

logger = logging.getLogger("multi_model_assistant.voice")

class VoiceSystem(ABC):
    """Abstract base class for voice systems."""
    
    @abstractmethod
    def transcribe(self, audio_data) -> str:
        """Transcribe audio data to text.
        
        Args:
            audio_data: Audio data in supported format
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def synthesize(self, text: str):
        """Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data in appropriate format
        """
        pass

class CoquiVoiceSystem(VoiceSystem):
    """Voice system using Coqui TTS and STT."""
    
    def __init__(self, voice_id: str = "default"):
        """Initialize Coqui voice system.
        
        Args:
            voice_id: Voice ID to use for synthesis
        """
        self.voice_id = voice_id
        self.api_key = os.environ.get("COQUI_API_KEY")
        
        # Initialize emotion detector
        self.emotion_detector = EmotionDetector()
        
        # Check if we have an API key
        if not self.api_key:
            logger.warning("No Coqui API key provided. Using local fallback if available.")
            self._setup_local_fallback()
        else:
            self.use_api = True
            logger.info("Using Coqui API for voice processing")
    
    def _setup_local_fallback(self):
        """Set up local fallback using offline libraries if available."""
        try:
            # Try to import necessary libraries
            import torch
            from TTS.api import TTS
            
            # Initialize TTS
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            
            # Try to import STT libraries
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            
            self.use_api = False
            logger.info("Using local TTS/STT libraries for voice processing")
        except ImportError:
            logger.error("Local TTS/STT libraries not available and no API key provided.")
            self.use_api = False
    
    def transcribe(self, audio_data) -> Dict[str, Any]:
        """Transcribe audio data to text and detect emotions using Coqui or local fallback.
        
        Returns:
            Dictionary containing transcribed text and detected emotions
        """
        # Transcribe audio to text
        if self.use_api and self.api_key:
            text = self._transcribe_with_api(audio_data)
        else:
            text = self._transcribe_locally(audio_data)
            
        # Detect emotions from both voice and text
        voice_emotions = self.emotion_detector.detect_from_voice(audio_data)
        text_emotions = self.emotion_detector.detect_from_text(text)
        
        # Combine emotions
        combined_emotions = self.emotion_detector.combine_emotions(text_emotions, voice_emotions)
        
        return {
            "text": text,
            "emotions": combined_emotions
        }
    
    def _transcribe_with_api(self, audio_data) -> str:
        """Transcribe audio using Coqui API."""
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Prepare API request
            url = "https://api.coqui.ai/v1/transcribe"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, headers=headers, files=files)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                logger.error(f"Coqui API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Error transcribing with Coqui API: {e}")
            return ""
    
    def _transcribe_locally(self, audio_data) -> str:
        """Transcribe audio using local libraries."""
        try:
            if not hasattr(self, 'recognizer'):
                logger.error("Local STT not available")
                return "[Transcription not available]"
            
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Transcribe using speech_recognition
            import speech_recognition as sr
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)  # Using Google's API as fallback
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return text
        except Exception as e:
            logger.error(f"Error transcribing locally: {e}")
            return "[Transcription failed]"
    
    def synthesize(self, text: str):
        """Synthesize text to speech using Coqui or local fallback."""
        if self.use_api and self.api_key:
            return self._synthesize_with_api(text)
        else:
            return self._synthesize_locally(text)
    
    def _synthesize_with_api(self, text: str):
        """Synthesize text using Coqui API."""
        try:
            # Prepare API request
            url = "https://api.coqui.ai/v1/synthesize"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "voice_id": self.voice_id
            }
            
            # Make request
            response = requests.post(url, headers=headers, json=payload)
            
            # Process response
            if response.status_code == 200:
                # Response contains audio data as base64
                result = response.json()
                audio_base64 = result.get("audio")
                if audio_base64:
                    return base64.b64decode(audio_base64)
                else:
                    logger.error("No audio data in Coqui API response")
                    return None
            else:
                logger.error(f"Coqui API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error synthesizing with Coqui API: {e}")
            return None
    
    def _synthesize_locally(self, text: str):
        """Synthesize text using local TTS library."""
        try:
            if not hasattr(self, 'tts'):
                logger.error("Local TTS not available")
                return None
            
            # Generate audio with local TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            self.tts.tts_to_file(text=text, file_path=temp_path)
            
            # Read the file
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio_data
        except Exception as e:
            logger.error(f"Error synthesizing locally: {e}")
            return None

class EmotionDetector:
    """Detect emotions in audio or text."""
    
    def __init__(self):
        """Initialize emotion detector."""
        self.api_key = os.environ.get("EMOTION_API_KEY")
        
        # Check if we have an API key
        if not self.api_key:
            logger.warning("No Emotion API key provided. Using simple keyword-based detection.")
            self.use_api = False
        else:
            self.use_api = True
            logger.info("Using Emotion API for emotion detection")
    
    def detect_emotion(self, input_data) -> Optional[str]:
        """Detect emotion in audio or text.
        
        Args:
            input_data: Audio data or text
            
        Returns:
            Detected emotion or None
        """
        if isinstance(input_data, str):
            # Text input
            return self._detect_emotion_from_text(input_data)
        else:
            # Audio input
            return self._detect_emotion_from_audio(input_data)
    
    def _detect_emotion_from_text(self, text: str) -> Optional[str]:
        """Detect emotion from text."""
        if self.use_api and self.api_key:
            return self._detect_emotion_from_text_api(text)
        else:
            return self._detect_emotion_from_text_keywords(text)
    
    def _detect_emotion_from_text_api(self, text: str) -> Optional[str]:
        """Detect emotion from text using API."""
        try:
            # This is a placeholder for a real emotion API
            # Replace with actual API integration
            url = "https://api.emotion-detection.com/v1/detect"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"text": text}
            
            # Make request
            response = requests.post(url, headers=headers, json=payload)
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                return result.get("emotion")
            else:
                logger.error(f"Emotion API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error detecting emotion from text with API: {e}")
            return None
    
    def _detect_emotion_from_text_keywords(self, text: str) -> Optional[str]:
        """Detect emotion from text using simple keyword matching."""
        text_lower = text.lower()
        
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "happy": ["happy", "joy", "delighted", "glad", "pleased", "excited", "thrilled"],
            "sad": ["sad", "unhappy", "depressed", "down", "miserable", "upset", "tearful"],
            "angry": ["angry", "mad", "furious", "outraged", "annoyed", "irritated", "frustrated"],
            "fearful": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
            "surprised": ["surprised", "shocked", "amazed", "astonished", "stunned"],
            "disgusted": ["disgusted", "revolted", "nauseated", "repulsed"],
            "neutral": ["fine", "okay", "ok", "alright", "neutral"]
        }
        
        # Count occurrences of emotion keywords
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_counts[emotion] += 1
        
        # Find emotion with highest count
        max_count = 0
        detected_emotion = "neutral"
        for emotion, count in emotion_counts.items():
            if count > max_count:
                max_count = count
                detected_emotion = emotion
        
        return detected_emotion if max_count > 0 else "neutral"
    
    def _detect_emotion_from_audio(self, audio_data) -> Optional[str]:
        """Detect emotion from audio."""
        if self.use_api and self.api_key:
            return self._detect_emotion_from_audio_api(audio_data)
        else:
            # Fallback to transcribing and then detecting from text
            # This would require a transcription service
            logger.warning("Audio emotion detection not available without API key")
            return "neutral"
    
    def _detect_emotion_from_audio_api(self, audio_data) -> Optional[str]:
        """Detect emotion from audio using API."""
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Prepare API request
            # This is a placeholder for a real emotion API
            url = "https://api.emotion-detection.com/v1/detect-audio"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, headers=headers, files=files)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                return result.get("emotion")
            else:
                logger.error(f"Emotion API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error detecting emotion from audio with API: {e}")
            return None

# Update the MultiModelAssistant initialization methods
def initialize_voice_for_assistant(assistant):
    """Initialize voice system for the assistant.
    
    Args:
        assistant: MultiModelAssistant instance
    """
    try:
        voice_config = assistant.config.get("voice", {})
        provider = voice_config.get("provider", "coqui")
        voice_id = voice_config.get("voice_id", "default")
        
        if provider == "coqui":
            assistant.voice_system = CoquiVoiceSystem(voice_id)
            logger.info(f"Initialized Coqui voice system with voice ID: {voice_id}")
        else:
            logger.warning(f"Unknown voice provider: {provider}. Using Coqui as fallback.")
            assistant.voice_system = CoquiVoiceSystem("default")
    except Exception as e:
        logger.error(f"Failed to initialize voice system: {e}")
        assistant.voice_system = None

def initialize_emotion_detection_for_assistant(assistant):
    """Initialize emotion detection for the assistant.
    
    Args:
        assistant: MultiModelAssistant instance
    """
    try:
        assistant.emotion_detector = EmotionDetector()
        logger.info("Initialized emotion detector")
    except Exception as e:
        logger.error(f"Failed to initialize emotion detector: {e}")
        assistant.emotion_detector = None