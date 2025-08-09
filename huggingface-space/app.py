import gradio as gr
import torch
import numpy as np
import tempfile
import os
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

def load_dia_model():
    """Load the Dia model"""
    global model
    try:
        logger.info("Loading Dia model...")
        from dia import Dia
        
        # Load with appropriate device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_dtype = "float16" if torch.cuda.is_available() else "float32"
        
        model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626",
            device=device,
            compute_dtype=compute_dtype,
            load_dac=True
        )
        logger.info(f"Dia model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to load Dia model: {e}")
        return False

def generate_speech(
    text: str, 
    max_tokens: int = 3072, 
    temperature: float = 0.7, 
    top_p: float = 0.9
) -> Tuple[Optional[str], str]:
    """Generate speech from text using Dia model"""
    
    if not text or not text.strip():
        return None, "❌ Please enter some text to convert to speech"
    
    if model is None:
        return None, "❌ Model not loaded. Please refresh the page and try again."
    
    try:
        logger.info(f"Generating speech for text: {text[:50]}...")
        
        # Generate audio using Dia model
        audio_array = model.generate(
            text=text.strip(),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            model.save_audio(temp_file.name, audio_array)
            
            logger.info("Speech generation completed successfully")
            return temp_file.name, f"✅ Generated speech for: '{text[:50]}{'...' if len(text) > 50 else ''}'"
            
    except Exception as e:
        error_msg = f"❌ Error generating speech: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

# Load model on startup
model_loaded = load_dia_model()

# Create Gradio interface
with gr.Blocks(
    title="Dia TTS - Nari Voice Generator",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🎙️ Dia TTS - Nari Voice Generator
    
    Convert your text into natural, human-like speech using the advanced Dia text-to-speech model.
    
    **Model**: `nari-labs/Dia-1.6B-0626`
    """)
    
    if not model_loaded:
        gr.Markdown("⚠️ **Warning**: Model failed to load. Some functionality may not work.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="📝 Text Input",
                placeholder="Enter the text you want to convert to speech...",
                lines=4,
                max_lines=10
            )
            
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=512,
                    maximum=4096,
                    value=3072,
                    step=128,
                    label="🎯 Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="🎲 Top P"
                )
            
            generate_btn = gr.Button(
                "🎵 Generate Speech",
                variant="primary",
                size="lg"
            )
        
        with gr.Column():
            audio_output = gr.Audio(
                label="🔊 Generated Speech",
                type="filepath"
            )
            status_output = gr.Textbox(
                label="📊 Status",
                interactive=False,
                lines=2
            )
    
    # Event handlers
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, max_tokens, temperature, top_p],
        outputs=[audio_output, status_output],
        show_progress=True
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["Transform your text into natural, human-like speech with our advanced AI technology.", 3072, 0.7, 0.9],
            ["The quick brown fox jumps over the lazy dog. This is a test of the Dia text-to-speech system.", 2048, 0.8, 0.9],
            ["Welcome to the future of voice synthesis. Experience the power of AI-generated speech.", 3072, 0.6, 0.8],
        ],
        inputs=[text_input, max_tokens, temperature, top_p],
        outputs=[audio_output, status_output],
        fn=generate_speech,
        cache_examples=False
    )
    
    gr.Markdown("""
    ---
    
    ### 📚 Usage Tips:
    - **Max Tokens**: Controls the length of generated audio (higher = longer)
    - **Temperature**: Controls randomness (0.1 = conservative, 1.0 = creative)
    - **Top P**: Controls diversity of word selection (0.1 = focused, 1.0 = diverse)
    
    ### ⚙️ Technical Details:
    - Model: Dia-1.6B-0626 by Nari Labs
    - Output Format: WAV audio
    - Recommended Text Length: 50-500 characters for best results
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )