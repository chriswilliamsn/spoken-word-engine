"""Configuration management module for the Dia model.

This module provides comprehensive configuration management for the Dia model,
including separate configurations for encoder and decoder components, and their
combination into a master configuration object.

Classes:
    EncoderConfig: Configuration specific to the encoder component.
    DecoderConfig: Configuration specific to the decoder component.
    DiaConfig: Master configuration combining all components.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EncoderConfig(BaseModel, frozen=True):
    """Configuration for the encoder component of the Dia model.

    Attributes:
        model_type: Type of the model, defaults to "dia_encoder".
        vocab_size: Vocabulary size for the encoder, defaults to 128256.
        intermediate_size: Size of the "intermediate" (i.e., feed-forward) layer in the encoder, defaults to 4096.
        num_hidden_layers: Number of hidden layers in the encoder, defaults to 32.
        num_attention_heads: Number of attention heads for each attention layer, defaults to 32.
        num_key_value_heads: Number of key-value heads for each attention layer.
            Defaults to the same value as num_attention_heads.
        hidden_size: The dimensionality of the encoder hidden states, defaults to 4096.
        max_position_embeddings: Maximum position embeddings length, defaults to 8192.
        rope_theta: Base theta for RoPE embeddings, defaults to 500000.0.
        attention_dropout: Attention dropout probability, defaults to 0.0.
        hidden_dropout: Hidden layer dropout probability, defaults to 0.0.
        rms_norm_eps: Epsilon for RMS normalization, defaults to 1e-5.
        vocab_size: Size of the vocabulary, defaults to 128256.
        eos_token_id: End-of-sequence token ID, defaults to 128001.
        pad_token_id: Padding token ID, defaults to 128001.
    """

    hidden_size: int = Field(default=4096, gt=0)
    intermediate_size: int = Field(default=4096, gt=0)
    max_position_embeddings: int = Field(default=8192, gt=0)
    model_type: str = Field(default="dia_encoder")
    num_attention_heads: int = Field(default=32, gt=0)
    num_hidden_layers: int = Field(default=32, gt=0)
    num_key_value_heads: int = Field(default=32, gt=0)
    rope_theta: float = Field(default=500000.0, gt=0.0)
    attention_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    hidden_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    rms_norm_eps: float = Field(default=1e-5, gt=0.0)
    vocab_size: int = Field(default=128256, gt=0)
    eos_token_id: int = Field(default=128001, ge=0)
    pad_token_id: int = Field(default=128001, ge=0)


class DecoderConfig(BaseModel, frozen=True):
    """Configuration for the decoder component of the Dia model.

    Attributes:
        model_type: Type of the model, defaults to "dia_decoder".
        vocab_size: Size of vocabulary for the decoder, defaults to 1026.
        intermediate_size: Size of the "intermediate" (i.e., feed-forward) layer in the decoder, defaults to 8192.
        num_hidden_layers: Number of hidden layers in the decoder, defaults to 24.
        num_attention_heads: Number of attention heads for each attention layer, defaults to 32.
        num_key_value_heads: Number of key-value heads for each attention layer.
            Defaults to the same value as num_attention_heads.
        hidden_size: The dimensionality of the decoder hidden states, defaults to 4096.
        max_position_embeddings: Maximum position embeddings length, defaults to 8192.
        rope_theta: Base theta for RoPE embeddings, defaults to 500000.0.
        attention_dropout: Attention dropout probability, defaults to 0.0.
        hidden_dropout: Hidden layer dropout probability, defaults to 0.0.
        rms_norm_eps: Epsilon for RMS normalization, defaults to 1e-5.
        vocab_size: Size of the vocabulary, defaults to 1026.
        eos_token_id: End-of-sequence token ID, defaults to 1025.
        pad_token_id: Padding token ID, defaults to 1025.
        audio_vocab_size: Size of the audio vocabulary, defaults to 1024.
        bos_token_id: Beginning-of-sequence token ID, defaults to 1024.
    """

    hidden_size: int = Field(default=4096, gt=0)
    intermediate_size: int = Field(default=8192, gt=0)
    max_position_embeddings: int = Field(default=8192, gt=0)
    model_type: str = Field(default="dia_decoder")
    num_attention_heads: int = Field(default=32, gt=0)
    num_hidden_layers: int = Field(default=24, gt=0)
    num_key_value_heads: int = Field(default=32, gt=0)
    rope_theta: float = Field(default=500000.0, gt=0.0)
    attention_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    hidden_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    rms_norm_eps: float = Field(default=1e-5, gt=0.0)
    vocab_size: int = Field(default=1026, gt=0)
    eos_token_id: int = Field(default=1025, ge=0)
    pad_token_id: int = Field(default=1025, ge=0)
    audio_vocab_size: int = Field(default=1024, gt=0)
    bos_token_id: int = Field(default=1024, ge=0)


class DiaConfig(BaseModel, frozen=True):
    """Main configuration container for the Dia model architecture.

    Attributes:
        model_type: Type of the model, defaults to "dia".
        encoder: Configuration for the encoder component.
        decoder: Configuration for the decoder component.
        cross_attention_hidden_size: Hidden size for cross-attention, defaults to 4096.
        guidance_scale: Scale for classifier-free guidance, defaults to 4.0.
        delay_pattern: Delay pattern for audio generation, defaults to [0, 1, 2, 3, 4, 5, 6, 7, 8].
        use_cache: Whether to use caching for generation, defaults to True.
        is_encoder_decoder: Whether the model is an encoder-decoder, defaults to True.
        torch_dtype: PyTorch data type, defaults to "float32".
        transformers_version: Version of transformers library used, defaults to "4.47.0.dev0".
        id2label: Mapping from label IDs to label names.
        label2id: Mapping from label names to label IDs.
        architectures: List of model architectures, defaults to ["DiaForConditionalGeneration"].
    """

    architectures: list[str] = Field(default_factory=lambda: ["DiaForConditionalGeneration"])
    cross_attention_hidden_size: int = Field(default=4096, gt=0)
    delay_pattern: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8])
    guidance_scale: float = Field(default=4.0, gt=0.0)
    is_encoder_decoder: bool = Field(default=True)
    model_type: str = Field(default="dia")
    torch_dtype: str = Field(default="float32")
    transformers_version: str = Field(default="4.47.0.dev0")
    use_cache: bool = Field(default=True)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    id2label: dict[str, str] = Field(default_factory=dict)
    label2id: dict[str, str] = Field(default_factory=dict)

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "validate_default": True,
        "str_strip_whitespace": True,
        "frozen": True,
    }

    @classmethod
    def load(cls, path: str) -> "DiaConfig | None":
        """Load and validate a Dia configuration from a JSON file.

        Args:
            path: The file path to the JSON configuration file.

        Returns:
            A validated DiaConfig instance if the file exists and is valid,
            None if the file doesn't exist.

        Raises:
            pydantic.ValidationError: If the JSON content fails validation against the DiaConfig schema.
        """
        config_path = Path(path)
        if not config_path.exists():
            return None

        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)

        return cls.model_validate(config_data)