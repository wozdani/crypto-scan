"""
Smart Money Detection Module
Implements strict criteria for identifying smart money addresses
"""

from typing import Set, Optional
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH
from utils.dynamic_whale_thresholds import calculate_dynamic_whale_threshold, validate_whale_strength

def is_exchange(addr: str, known_exchanges: Set[str]) -> bool:
    """Check if address is a known exchange"""
    return addr.lower() in known_exchanges

def is_smart_money(addr: str, trust: float, preds: int, usd: float, 
                   repeats_7d: int, known_exchanges: Set[str]) -> bool:
    """
    Determine if address qualifies as smart money
    
    Args:
        addr: Address to check
        trust: Trust score (0-1)
        preds: Number of predictions
        usd: USD value of transaction
        repeats_7d: Number of repeat transactions in 7 days
        known_exchanges: Set of known exchange addresses
    
    Returns:
        bool: True if address is smart money
    """
    # Exclude exchange addresses if configured
    if STEALTH["SM_EXCLUDE_EXCHANGES"] and is_exchange(addr, known_exchanges):
        return False
    
    # Criteria 1: High trust and sufficient predictions
    if trust >= STEALTH["SM_TRUST_MIN"] and preds >= STEALTH["SM_PRED_MIN"]:
        return True
    
    # Criteria 2: Repeat whale with sufficient value
    if repeats_7d >= STEALTH["SM_REPEAT_MIN_7D"] and usd >= STEALTH["SM_WHALE_MIN_USD"]:
        return True
    
    return False

def check_trust_ok(addresses: list, known_exchanges: Set[str], min_trust: float = 0.6) -> bool:
    """
    PUNKT 5 FIX: Sprawdź czy jest co najmniej 1 adres z trust>=min_trust i nie-CEX
    
    Args:
        addresses: Lista słowników z adresami i ich parametrami (trust, addr)
        known_exchanges: Set znanych adresów giełd CEX
        min_trust: Minimalny wymagany trust score (domyślnie 0.6)
    
    Returns:
        bool: True jeśli co najmniej 1 adres spełnia warunki
    """
    for addr_info in addresses:
        addr = addr_info.get("addr", "")
        trust = addr_info.get("trust", 0.0)
        
        # Sprawdź: trust >= 0.6 AND nie jest CEX
        if trust >= min_trust and not is_exchange(addr, known_exchanges):
            return True
    
    return False

def apply_smart_money_boost(sig: dict, trust_ok: bool) -> dict:
    """
    PUNKT 5 FIX: Jednoznaczny boost smart-money (bez karuzeli 3.0→1.0)
    Tylko raz i tylko jeśli trust_ok
    
    Args:
        sig: Signal dictionary with 'strength' and 'meta' fields
        trust_ok: True if at least 1 address has trust>=0.6 and is not CEX
    
    Returns:
        dict: Modified signal with smart money boost if applicable
    """
    # Tylko jeśli trust_ok, dodaj jednolity mały boost
    if not trust_ok:
        return sig
    
    s = sig.get("strength", 0.0)
    sig["strength"] = min(1.0, s + 0.2)  # Jednolity, mały boost +0.2
    sig["meta"] = {**sig.get("meta", {}), "smart_money_boost": True}
    
    print(f"[SMART MONEY] Applied fixed boost +0.2 (trust_ok=True)")
    
    return sig