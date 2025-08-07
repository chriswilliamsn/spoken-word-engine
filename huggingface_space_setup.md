# Hugging Face Space Deployment for Dia TTS

## Steps to deploy on Hugging Face Spaces:

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose "Docker" as the SDK
4. Upload these files:

### Dockerfile
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

### app.py (modified for HF Spaces)
```python
import gradio as gr
import torch
from dia import Dia
import numpy as np
import tempfile
import os

# Load model
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626")

def generate_speech(text, max_tokens=3072, temperature=0.7, top_p=0.9):
    if not text.strip():
        return None, "Please enter some text"
    
    try:
        # Generate audio
        audio_array = model.generate(
            text=text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            model.save_audio(f.name, audio_array)
            return f.name, f"Generated speech for: {text[:50]}..."
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_speech,
    inputs=[
        gr.Textbox(label="Text", placeholder="Enter text to convert to speech"),
        gr.Slider(1024, 4096, value=3072, label="Max Tokens"),
        gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, label="Top P")
    ],
    outputs=[
        gr.Audio(label="Generated Speech"),
        gr.Textbox(label="Status")
    ],
    title="Dia TTS - Nari Voice",
    description="Convert text to speech using the Dia model"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
```

### requirements.txt
```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
gradio>=4.0.0
huggingface-hub>=0.16.0
dac>=1.0.0
```

The Space will give you a public URL like: `https://yourname-dia-tts.hf.space`