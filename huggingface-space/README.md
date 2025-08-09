---
title: Dia TTS - Nari Voice
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
suggested_hardware: a10g-small
suggested_storage: large
---

# Dia TTS - Nari Voice Generator

This Hugging Face Space provides a web interface for the Dia text-to-speech model, enabling you to convert text into natural, human-like speech.

## Features

- 🎙️ High-quality text-to-speech generation
- 🎛️ Adjustable generation parameters (temperature, top_p, max_tokens)
- 🔊 Direct audio playback and download
- 📱 Responsive web interface
- ⚡ GPU acceleration support

## Usage

1. Enter your text in the input field
2. Adjust generation parameters if needed
3. Click "Generate Speech" 
4. Listen to or download the generated audio

## Model Details

- **Model**: `nari-labs/Dia-1.6B-0626`
- **Type**: Text-to-Speech
- **Output**: WAV audio files
- **Quality**: High-fidelity speech synthesis

## API Integration

Once deployed, you can integrate this Space with your applications by making HTTP requests to the Gradio API endpoints.

Example integration:
```python
import requests

response = requests.post(
    "https://your-space-name.hf.space/api/predict",
    json={
        "data": ["Your text here", 3072, 0.7, 0.9]
    }
)
```