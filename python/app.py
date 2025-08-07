#!/usr/bin/env python3
"""
Dia TTS Web Server
Provides HTTP API for text-to-speech generation using the Dia model
"""

import os
import io
import base64
import logging
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from dia import Dia

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class TTSRequest(BaseModel):
    text: str
    max_tokens: Optional[int] = 3072
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class TTSResponse(BaseModel):
    audio_content: str  # base64 encoded audio
    message: str

# Global model instance
model: Optional[Dia] = None

# Initialize FastAPI app
app = FastAPI(
    title="Dia TTS Server",
    description="Text-to-Speech API using the Dia model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model():
    """Load the Dia model on startup"""
    global model
    try:
        logger.info("Loading Dia model...")
        
        # Try to load model with memory optimization
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
            
        model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626",
            device=device,
            compute_dtype="float16" if torch.cuda.is_available() else "float32"
        )
        logger.info("Dia model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load Dia model: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Don't fail startup, just log the error
        model = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """Generate speech from text using the Dia model"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Dia model not loaded. Please check server logs."
        )
    
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        # Generate audio using Dia model
        audio_array = model.generate(
            text=request.text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Convert numpy array to audio file in memory
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Save audio using the model's save_audio method
            model.save_audio(temp_file.name, audio_array)
            
            # Read the audio file and encode as base64
            with open(temp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Clean up temp file
            os.unlink(temp_file.name)
        
        logger.info("Speech generation completed successfully")
        
        return TTSResponse(
            audio_content=audio_base64,
            message=f"Generated speech for: '{request.text[:50]}...'"
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate speech: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Dia TTS Server",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Generate speech from text (POST)",
            "/docs": "API documentation"
        },
        "model_status": "loaded" if model else "not loaded"
    }

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting Dia TTS Server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )