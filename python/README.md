# Dia Text-to-Speech Model

This directory contains the Python implementation of the Dia TTS model from the nari-labs/dia repository.

## Structure

- `dia/` - Main Python package
  - `__init__.py` - Package initialization  
  - `config.py` - Configuration management with Pydantic
  - `model.py` - Main Dia model implementation
  - `layers.py` - Neural network layers and components
  - `state.py` - Inference state management
  - `audio.py` - Audio processing utilities

## Installation

```bash
cd python
pip install -r requirements.txt
```

## Usage

```python
from dia import Dia

# Load pretrained model
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626")

# Generate speech from text
audio = model.generate("Hello world!")
```

## Docker Support

The repository includes both CPU and GPU Dockerfiles for containerized deployment.