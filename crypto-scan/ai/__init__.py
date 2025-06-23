"""
AI Module for Crypto Scan
Moduł AI zawierający CLIP i inne komponenty uczenia maszynowego
"""

from .clip_model import CLIPWrapper, get_clip_model, get_clip_image_embedding

__all__ = [
    'CLIPWrapper',
    'get_clip_model',
    'get_clip_image_embedding'
]