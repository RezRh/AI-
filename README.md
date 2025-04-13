# Multi-Model AI Assistant

A powerful voice assistant that combines multiple AI models (Claude, DeepSeek, ChatGPT) for specialized tasks, with voice capabilities, emotion detection, and memory integration.

## Features

- **Multi-Model Architecture**: Uses different AI models for specific tasks:
  - Claude for reasoning and long-term memory
  - DeepSeek for planning, logic, and internet search
  - ChatGPT for human-like conversation
  - Automatic fallback to DeepSeek when credits run out

- **Voice Capabilities**:
  - Speech recognition and synthesis using Coqui TTS (free voice cloning)
  - Emotion detection from voice and text input

- **Memory System**:
  - Vector-based memory storage using ChromaDB (free)
  - Retrieves relevant past conversations for context

- **Image Generation**:
  - Free image generation using Stable Diffusion (via Hugging Face API)
  - Fallback to DALL-E Mini when needed

- **Credit Management**:
  - Tracks usage of paid API services
  - Automatically switches to fallback models when limits are reached

- **Flexible Configuration**:
  - Easy configuration via JSON file
  - Environment variables for API keys

## Project Structure

```
├── main.py                 # Main assistant controller
├── model_providers.py      # AI model integration (Claude, DeepSeek, ChatGPT)
├── memory_system.py        # Vector memory using ChromaDB
├── voice_system.py         # Voice processing with Coqui
├── emotion_detector.py     # Emotion detection from voice and text
├── image_generation.py     # Image generation with Stable Diffusion
├── config.json             # Configuration file
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/multi-model-ai-assistant.git
   cd multi-model-ai-assistant
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up API keys as environment variables:
   ```
   CLAUDE_API_KEY=your_claude_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   OPENAI_API_KEY=your_openai_api_key
   COQUI_API_KEY=your_coqui_api_key (optional)
   HF_API_KEY=your_huggingface_api_key (optional, for image generation)
   ```

## Configuration

The system is configured via `config.json`. You can modify this file to change model preferences, memory settings, and credit limits.

```json
{
    "models": {
        "reasoning": {
            "provider": "claude",
            "fallback": "deepseek",
            "api_key_env": "CLAUDE_API_KEY"
        },
        "planning": {
            "provider": "deepseek",
            "fallback": null,
            "api_key_env": "DEEPSEEK_API_KEY"
        },
        "conversation": {
            "provider": "chatgpt",
            "fallback": "deepseek",
            "api_key_env": "OPENAI_API_KEY"
        }
    },
    "memory": {
        "provider": "chroma",
        "collection_name": "assistant_memory",
        "persist_directory": "./chroma_db"
    },
    "voice": {
        "provider": "coqui",
        "voice_id": "default"
    },
    "image": {
        "provider": "stable-diffusion"
    },
    "credit_limits": {
        "claude": 1000,
        "deepseek": 5000,
        "chatgpt": 2000
    }
}
```

## Usage

### Basic Usage

```python
from main import MultiModelAssistant

# Initialize the assistant
assistant = MultiModelAssistant()

# Process a text query
response = assistant.process_query("Why is the sky blue?")
print(response['text'])

# Generate an image
image_result = assistant.generate_image("A beautiful sunset over mountains")
print(f"Image saved to: {image_result['image_path']}")

# Process an internet search
search_response = assistant.process_internet_search("Latest news about AI")
print(search_response['text'])
```

### Voice Processing

```python
# Process voice input (audio_data would be raw audio bytes)
response = assistant.process_voice_input(audio_data)
print(response['text'])

# The response includes synthesized audio
audio_response = response['audio']
# Play or save the audio as needed
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.