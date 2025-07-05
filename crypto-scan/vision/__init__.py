"""
Vision AI Module - AI-EYE System
Comprehensive visual pattern recognition combining CLIP and GPT analysis
"""

# Import only if modules are available
try:
    from .ai_label_pipeline import prepare_ai_label
    AI_PIPELINE_AVAILABLE = True
except ImportError:
    AI_PIPELINE_AVAILABLE = False
    
try:
    from .vision_scoring import score_from_ai_label
    VISION_SCORING_AVAILABLE = True
except ImportError:
    VISION_SCORING_AVAILABLE = False

# Safe exports
__all__ = []
if AI_PIPELINE_AVAILABLE:
    __all__.append('prepare_ai_label')
if VISION_SCORING_AVAILABLE:
    __all__.append('score_from_ai_label')