"""
Trap Detector Module - Module 3 of Advanced Vision-AI System

Wykrywa fałszywe wybicia, bull/bear trapy i pułapki FOMO
"""

from .trap_patterns import (
    detect_fake_breakout,
    detect_failed_breakout,
    detect_bear_trap,
    detect_exhaustion_spike,
    comprehensive_trap_analysis
)

from .trap_scorer import (
    score_from_trap_detector,
    create_trap_detector_summary,
    validate_trap_scoring_input
)

__all__ = [
    'detect_fake_breakout',
    'detect_failed_breakout', 
    'detect_bear_trap',
    'detect_exhaustion_spike',
    'comprehensive_trap_analysis',
    'score_from_trap_detector',
    'create_trap_detector_summary',
    'validate_trap_scoring_input'
]

__version__ = "1.0.0"
__author__ = "Crypto Scanner AI Team"
__description__ = "Advanced trap detection for cryptocurrency trading patterns"