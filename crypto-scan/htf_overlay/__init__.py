"""
HTF Overlay Module - Higher Timeframe Market Structure Awareness
Adds macro context analysis to enhance AI-EYE decisions with HTF phase detection
"""

# Import main HTF overlay functions
try:
    from .phase_detector import detect_htf_phase
    from .overlay_score import score_from_htf_overlay
    HTF_OVERLAY_AVAILABLE = True
except ImportError:
    HTF_OVERLAY_AVAILABLE = False

# Optional support/resistance module
try:
    from .htf_support_resistance import detect_htf_levels
    HTF_SR_AVAILABLE = True
except ImportError:
    HTF_SR_AVAILABLE = False

__all__ = []
if HTF_OVERLAY_AVAILABLE:
    __all__.extend(['detect_htf_phase', 'score_from_htf_overlay'])
if HTF_SR_AVAILABLE:
    __all__.append('detect_htf_levels')