"""
Modern Detector Logging System - Integration Layer
Bridges modern detectors with unified log_utils system
"""

from utils.log_utils import (
    log_whaleclip, log_diamond, log_californium, log_rl_agent, 
    log_consensus, log_alert_system, log_fusion_decision,
    log_stealth_signal, log_detector_summary, initialize_logging_system
)


def initialize_modern_logging():
    """Initialize modern detector logging system"""
    initialize_logging_system()
    print("🔧 [MODERN LOGGING] AI detector logging system initialized")


def log_detector_activity(detector_type: str, symbol: str, **kwargs):
    """Universal detector logging dispatcher"""
    
    if detector_type == 'whaleclip':
        score = kwargs.get('score', 0.0)
        trust = kwargs.get('trust', 0.0) 
        pattern = kwargs.get('pattern', 'UNKNOWN')
        log_whaleclip(symbol, score, trust, pattern)
        
    elif detector_type == 'diamond':
        pattern = kwargs.get('pattern', 'UNKNOWN')
        score = kwargs.get('score', 0.0)
        anomaly = kwargs.get('anomaly', 0.0)
        log_diamond(symbol, pattern, score, anomaly)
        
    elif detector_type == 'californium':
        mastermind_addr = kwargs.get('mastermind_addr', '0x...')
        score = kwargs.get('score', 0.0)
        boost = kwargs.get('boost', 0.0)
        log_californium(symbol, mastermind_addr, score, boost)
        
    elif detector_type == 'rl_agent':
        state = kwargs.get('state', [])
        action = kwargs.get('action', 'UNKNOWN')
        reward = kwargs.get('reward', 0.0)
        log_rl_agent(symbol, state, action, reward)
        
    elif detector_type == 'consensus':
        votes = kwargs.get('votes', {})
        decision = kwargs.get('decision', 'UNKNOWN')
        log_consensus(symbol, votes, decision)
        
    elif detector_type == 'fusion':
        detector_scores = kwargs.get('detector_scores', {})
        weighted_score = kwargs.get('weighted_score', 0.0)
        confidence = kwargs.get('confidence', 'UNKNOWN')
        log_fusion_decision(symbol, detector_scores, weighted_score, confidence)
        
    elif detector_type == 'stealth':
        signal_name = kwargs.get('signal_name', 'UNKNOWN')
        strength = kwargs.get('strength', 0.0)
        active = kwargs.get('active', False)
        details = kwargs.get('details', None)
        log_stealth_signal(symbol, signal_name, strength, active, details)
        
    else:
        print(f"[MODERN LOGGING] Unknown detector type: {detector_type}")


def log_comprehensive_analysis(symbol: str, all_results: dict):
    """Log comprehensive analysis summary for a symbol"""
    log_detector_summary(symbol, all_results)


# Export all logging functions for easy import
__all__ = [
    'initialize_modern_logging',
    'log_detector_activity', 
    'log_comprehensive_analysis',
    'log_whaleclip',
    'log_diamond', 
    'log_californium',
    'log_rl_agent',
    'log_consensus',
    'log_alert_system',
    'log_fusion_decision',
    'log_stealth_signal',
    'log_detector_summary'
]