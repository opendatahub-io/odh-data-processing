"""Encoder modules for embedding generation."""

from .bge import UnifiedBGEEncoder
from .openai import OpenAIEncoder
from .nvembed import NVEmbedEncoder
from .arctic import ArcticEmbedEncoder
from .qwen2 import Qwen2EmbedEncoder
from .sentence import SentenceEncoder
from .sfr_mistral import SFRMistralEncoder

# Encoder registry
ENCODERS = {
    "bge": UnifiedBGEEncoder,
    "openai": OpenAIEncoder,
    "nvembed": NVEmbedEncoder,
    "arctic": ArcticEmbedEncoder,
    "qwen2": Qwen2EmbedEncoder,
    "sentence": SentenceEncoder,
    "sfr_mistral": SFRMistralEncoder,
}

def get_encoder_class(encoder_type: str):
    """Get encoder class by type name."""
    if encoder_type not in ENCODERS:
        available = ", ".join(ENCODERS.keys())
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {available}")
    return ENCODERS[encoder_type]

def list_encoders():
    """List available encoder types."""
    return list(ENCODERS.keys())

__all__ = [
    "UnifiedBGEEncoder", "OpenAIEncoder", "NVEmbedEncoder", 
    "ArcticEmbedEncoder", "Qwen2EmbedEncoder", "SentenceEncoder", 
    "SFRMistralEncoder", "get_encoder_class", "list_encoders"
]