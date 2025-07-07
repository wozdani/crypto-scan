"""
PrePump Engine v2 – Stealth AI
System analizy sygnałów pre-pump bez polegania na wykresach

Główne komponenty:
- StealthEngine: Główny silnik scoringu
- StealthSignalDetector: Detekcja sygnałów z otoczenia rynku  
- StealthWeightManager: Zarządzanie wagami sygnałów
- StealthFeedbackSystem: System uczenia przez feedback loop
"""

from .stealth_engine import StealthEngine, StealthResult, analyze_token_stealth, get_stealth_engine
from .stealth_signals import StealthSignalDetector
from .stealth_weights import StealthWeightManager
from .stealth_feedback import StealthFeedbackSystem

__version__ = "2.0.0"
__author__ = "Crypto Scanner Team"

# Convenience imports dla łatwego użycia
__all__ = [
    'StealthEngine',
    'StealthResult', 
    'StealthSignalDetector',
    'StealthWeightManager',
    'StealthFeedbackSystem',

    'analyze_token_stealth',
    'get_stealth_engine'
]