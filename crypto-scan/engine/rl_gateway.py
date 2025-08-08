"""
PUNKT 9 FIX: RL/Enhanced RL Gateway - raz i tylko "borderline"
Centralny gateway dla wszystkich wywołań Enhanced RL z ograniczeniem do borderline cases
"""

import time
from typing import Dict, Any, Optional
from functools import wraps

# Global tracking dla @once_per_scan
_rl_scan_cache = {}

def once_per_scan(category: str, subcategory: str):
    """
    PUNKT 9 FIX: Dekorator zapewniający że Enhanced RL odpala się maksymalnie raz na scan
    
    Args:
        category: Kategoria (np. "AI")
        subcategory: Podkategoria (np. "engine")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Klucz dla cache z timestampem
            current_time = time.time()
            cache_key = f"{category}_{subcategory}"
            
            # Sprawdź czy nie wywoływano już w tym scanie (15-minutowe okno)
            if cache_key in _rl_scan_cache:
                last_call_time = _rl_scan_cache[cache_key]
                if current_time - last_call_time < 900:  # 15 minut
                    return {"status": "skipped", "reason": "already_called_this_scan"}
            
            # Wywołaj funkcję i zapisz timestamp
            result = func(*args, **kwargs)
            _rl_scan_cache[cache_key] = current_time
            return result
        return wrapper
    return decorator

@once_per_scan("AI", "engine")
def maybe_run_rl(symbol: str, p_raw: float, consensus: str, stealth_data: Dict, market_data: Dict) -> Dict[str, Any]:
    """
    PUNKT 9 FIX: RL odpalaj raz i tylko "borderline"
    
    Args:
        symbol: Symbol tokena
        p_raw: Raw probability z aggregatora (0-1)
        consensus: Consensus decision ("BUY", "HOLD", "AVOID", "WATCH")
        stealth_data: Dane ze stealth engine
        market_data: Dane rynkowe
        
    Returns:
        Dict z wynikiem RL lub powodem skip
    """
    # PUNKT 9 FIX: Borderline checking - tylko 0.65 <= p_raw <= 0.80 i consensus != "BUY"
    if not (0.65 <= p_raw <= 0.80):
        return {"status": "skipped", "reason": "non-borderline", "p_raw": p_raw, "range": "0.65-0.80"}
    
    if consensus == "BUY":
        return {"status": "skipped", "reason": "consensus_already_buy", "consensus": consensus}
    
    # Borderline case - uruchom Enhanced RL
    try:
        print(f"[RL GATEWAY] {symbol}: BORDERLINE detected - p_raw={p_raw:.3f}, consensus={consensus}")
        return run_rl(symbol, stealth_data, market_data)
    except Exception as e:
        print(f"[RL GATEWAY ERROR] {symbol}: {e}")
        return {"status": "error", "reason": str(e)}

def run_rl(symbol: str, stealth_data: Dict, market_data: Dict) -> Dict[str, Any]:
    """
    PUNKT 9 FIX: Faktyczne uruchomienie Enhanced RL
    
    Args:
        symbol: Symbol tokena
        stealth_data: Dane ze stealth engine
        market_data: Dane rynkowe
        
    Returns:
        Wynik Enhanced RL
    """
    try:
        # Import Enhanced RL system
        from enhanced_rl_integration import process_stealth_with_enhanced_rl
        
        # Uruchom Enhanced RL
        enhanced_result = process_stealth_with_enhanced_rl(symbol, stealth_data, market_data)
        
        print(f"[RL GATEWAY] {symbol}: Enhanced RL completed - decision={enhanced_result.get('decision')}, score={enhanced_result.get('weighted_score', 0):.3f}")
        
        return {
            "status": "completed",
            "enhanced_result": enhanced_result,
            "decision": enhanced_result.get('decision'),
            "weighted_score": enhanced_result.get('weighted_score', 0.0),
            "adaptive_threshold": enhanced_result.get('adaptive_threshold', 0.0)
        }
        
    except ImportError as e:
        print(f"[RL GATEWAY] {symbol}: Enhanced RL not available: {e}")
        return {"status": "unavailable", "reason": "enhanced_rl_not_available"}
    except Exception as e:
        print(f"[RL GATEWAY] {symbol}: Enhanced RL failed: {e}")
        return {"status": "failed", "reason": str(e)}

def get_rl_scan_stats() -> Dict[str, Any]:
    """Zwraca statystyki wywołań RL w tym scanie"""
    current_time = time.time()
    active_calls = {}
    
    for key, timestamp in _rl_scan_cache.items():
        if current_time - timestamp < 900:  # Ostatnie 15 minut
            active_calls[key] = {
                "timestamp": timestamp,
                "minutes_ago": (current_time - timestamp) / 60
            }
    
    return {
        "active_scan_calls": active_calls,
        "total_cached_calls": len(_rl_scan_cache)
    }

def clear_rl_scan_cache():
    """Wyczyść cache RL (dla testów lub force reset)"""
    global _rl_scan_cache
    _rl_scan_cache.clear()
    print("[RL GATEWAY] Scan cache cleared")