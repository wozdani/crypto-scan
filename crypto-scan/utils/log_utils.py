"""
Unified Logging System for Crypto Scanner AI Detectors
Centralized logging utilities for all modern AI detection components
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


def get_log_prefix(detector_name: str) -> str:
    """Get standardized log prefix for detector"""
    prefixes = {
        'whaleclip': '[WHALECLIP]',
        'diamond': '[DIAMOND]',
        'californium': '[CALIFORNIUM]',
        'rl_agent': '[RL AGENT]',
        'consensus': '[CONSENSUS]',
        'alert_system': '[ALERT SYSTEM]',
        'stealth': '[STEALTH]',
        'fusion': '[FUSION]'
    }
    return prefixes.get(detector_name.lower(), f'[{detector_name.upper()}]')


def log_whaleclip(symbol: str, score: float, trust: float, pattern: str) -> None:
    """Log WhaleCLIP detector activity"""
    prefix = get_log_prefix('whaleclip')
    print(f"{prefix} {symbol} - Score: {score:.2f} | Trust: {trust:.2f} | Pattern: {pattern}")


def log_diamond(symbol: str, pattern: str, score: float, anomaly: float) -> None:
    """Log DiamondWhale detector activity"""
    prefix = get_log_prefix('diamond')
    print(f"{prefix} {symbol} - Detected pattern: {pattern} | Score: {score:.2f} | Subgraph anomaly: {anomaly:.2f}")


def log_californium(symbol: str, mastermind_addr: str, score: float, boost: float) -> None:
    """Log CaliforniumWhale detector activity"""
    prefix = get_log_prefix('californium')
    addr_short = mastermind_addr[:10] + "..." if len(mastermind_addr) > 10 else mastermind_addr
    print(f"{prefix} {symbol} - Mastermind traced: {addr_short} | Score: {score:.2f} | EWMA Boost: {boost:.2f}")


def log_rl_agent(symbol: str, state: List[float], action: str, reward: float) -> None:
    """Log RLAgentV3 activity"""
    prefix = get_log_prefix('rl_agent')
    state_short = [round(s, 3) for s in state[:3]]  # Show first 3 state elements
    print(f"{prefix} Update for {symbol} - State: {state_short}... | Action: {action} | Reward: {reward}")


def log_consensus(symbol: str, votes: Dict[str, Any], decision: str) -> None:
    """Log ConsensusAgentSystem activity"""
    prefix = get_log_prefix('consensus')
    vote_summary = {k: round(v, 3) if isinstance(v, float) else v for k, v in votes.items()}
    print(f"{prefix} {symbol} - Votes: {vote_summary} | Final Decision: {decision}")


def log_alert_system(symbol: str, triggering_detector: str, final_score: float) -> None:
    """Log central alert system activity"""
    prefix = get_log_prefix('alert_system')
    print(f"{prefix} Final alert for {symbol} | Source: {triggering_detector} | Score: {final_score:.2f}")


def log_fusion_decision(symbol: str, detector_scores: Dict[str, float], 
                       weighted_score: float, confidence: str) -> None:
    """Log Fusion Engine multi-detector decisions"""
    prefix = get_log_prefix('fusion')
    scores_str = ", ".join([f"{k}:{v:.2f}" for k, v in detector_scores.items()])
    print(f"{prefix} {symbol} - Detectors: [{scores_str}] | Weighted: {weighted_score:.2f} | Confidence: {confidence}")


def log_stealth_signal(symbol: str, signal_name: str, strength: float, 
                      active: bool, details: Optional[str] = None) -> None:
    """Log stealth signal detection with enhanced formatting"""
    prefix = get_log_prefix('stealth')
    status = "ACTIVE" if active else "INACTIVE"
    detail_str = f" | {details}" if details else ""
    print(f"{prefix} {symbol} [{signal_name}] - {status} (strength: {strength:.3f}){detail_str}")


def log_detector_summary(symbol: str, detector_results: Dict[str, Dict[str, Any]]) -> None:
    """Log comprehensive detector summary for a symbol"""
    print(f"\n🎯 [DETECTOR SUMMARY] {symbol}")
    for detector, results in detector_results.items():
        score = results.get('score', 0.0)
        confidence = results.get('confidence', 'UNKNOWN')
        active = results.get('active', False)
        status = "✅" if active else "❌"
        print(f"   {status} {detector}: {score:.3f} ({confidence})")
    print()


def disable_legacy_tjde_logs() -> None:
    """Set environment variables to disable legacy TJDE logging"""
    os.environ['TJDE_DEBUG'] = '0'
    os.environ['DISABLE_TJDE_LOGS'] = '1'
    os.environ['LEGACY_MODE'] = '0'


def log_debug(message: str, level: str = 'info', component: str = 'SYSTEM') -> None:
    """Controlled debug logging with level filtering"""
    debug_enabled = os.getenv('DEBUG', '0') == '1'
    tjde_debug = os.getenv('TJDE_DEBUG', '0') == '1'
    
    # Skip TJDE logs if disabled
    if 'TJDE' in component and not tjde_debug:
        return
        
    # Only show debug level if DEBUG is enabled
    if level == 'debug' and not debug_enabled:
        return
        
    timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f"[{timestamp}] [{component}] {message}")


# Convenience functions for common logging patterns
def log_detector_start(detector_name: str, symbol: str) -> None:
    """Log detector analysis start"""
    prefix = get_log_prefix(detector_name)
    print(f"{prefix} Starting analysis for {symbol}")


def log_detector_complete(detector_name: str, symbol: str, score: float, active: bool) -> None:
    """Log detector analysis completion"""
    prefix = get_log_prefix(detector_name)
    status = "TRIGGERED" if active else "NO SIGNAL"
    print(f"{prefix} Complete for {symbol} - Score: {score:.3f} | Status: {status}")


def log_error_with_context(detector_name: str, symbol: str, error: str, context: Dict[str, Any] = None) -> None:
    """Log errors with additional context"""
    prefix = get_log_prefix(detector_name)
    context_str = f" | Context: {context}" if context else ""
    print(f"❌ {prefix} ERROR {symbol}: {error}{context_str}")


# Initialize logging system
def initialize_logging_system() -> None:
    """Initialize the unified logging system"""
    disable_legacy_tjde_logs()
    print("🔧 [LOG SYSTEM] Unified AI detector logging initialized")
    print("🔧 [LOG SYSTEM] Legacy TJDE logs disabled")
    print("🔧 [LOG SYSTEM] Modern detector logs enabled")
    print("🔧 [LOG SYSTEM] Available detectors: WhaleCLIP, DiamondWhale, CaliforniumWhale, RLAgent, Consensus, Fusion")
    
    # Log the standardized prefixes
    print("🔧 [LOG PREFIXES] [WHALECLIP], [DIAMOND], [CALIFORNIUM], [RL AGENT], [CONSENSUS], [ALERT SYSTEM], [FUSION], [STEALTH]")


if __name__ == "__main__":
    # Test the logging system
    initialize_logging_system()
    
    # Test all detector log functions
    log_whaleclip("BTCUSDT", 0.85, 0.92, "ACCUMULATION")
    log_diamond("ETHUSDT", "TEMPORAL_ANOMALY", 0.78, 0.65)
    log_californium("ADAUSDT", "0x1234567890abcdef", 0.91, 0.15)
    log_rl_agent("SOLUSDT", [0.1, 0.2, 0.3, 0.4], "BUY", 1.5)
    log_consensus("BNBUSDT", {"diamond": 0.8, "whale": 0.6}, "STRONG_BUY")
    log_alert_system("DOTUSDT", "DiamondWhale", 0.87)
    log_fusion_decision("LINKUSDT", {"diamond": 0.7, "californium": 0.8}, 0.75, "HIGH")