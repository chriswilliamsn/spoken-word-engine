# Running Dia TTS Locally

## Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ storage space

## Installation
```bash
# Clone the model
git clone https://huggingface.co/nari-labs/Dia-1.6B-0626

# Install dependencies
pip install torch torchaudio transformers huggingface_hub

# Install the Dia library
pip install dac
```

## Usage
```python
from dia import Dia

# Load the model
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626")

# Generate speech
audio = model.generate(
    text="Hello world",
    max_tokens=3072,
    temperature=0.7,
    top_p=0.9
)

# Save audio
model.save_audio("output.wav", audio)
```

## FastAPI Server
You can use the existing `python/app.py` from your project to create a local server.