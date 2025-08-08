"""
Dynamic Whale Threshold Calculator
Calculates realistic whale thresholds based on token volume and market conditions
"""

from typing import Dict
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH

def calculate_dynamic_whale_threshold(volume_24h_usd: float) -> float:
    """
    Calculate dynamic whale threshold based on 24h volume
    
    For EDUUSDT case: $1.65M volume → threshold should be ~$33k, not $3.7k
    
    Args:
        volume_24h_usd: 24h volume in USD
        
    Returns:
        float: Minimum USD value for whale classification
    """
    # Base calculation: percentage of 24h volume
    dynamic_threshold = volume_24h_usd * STEALTH["WHALE_THRESHOLD_MULTIPLIER"]
    
    # Apply min/max bounds
    threshold = max(
        STEALTH["WHALE_THRESHOLD_MIN"], 
        min(dynamic_threshold, STEALTH["WHALE_THRESHOLD_MAX"])
    )
    
    print(f"[WHALE THRESHOLD] Volume=${volume_24h_usd:,.0f} → Dynamic threshold=${threshold:,.0f}")
    return threshold

def validate_whale_strength(whale_usd_values: list, volume_24h_usd: float) -> float:
    """
    Validate whale transactions and calculate realistic strength
    
    Args:
        whale_usd_values: List of whale transaction values in USD
        volume_24h_usd: 24h volume in USD
        
    Returns:
        float: Whale strength [0,1] based on realistic thresholds
    """
    if not whale_usd_values or volume_24h_usd <= 0:
        return 0.0
    
    # Get dynamic threshold
    threshold = calculate_dynamic_whale_threshold(volume_24h_usd)
    
    # Filter whales that meet dynamic threshold
    valid_whales = [w for w in whale_usd_values if w >= threshold]
    
    if not valid_whales:
        print(f"[WHALE VALIDATION] No whales above ${threshold:,.0f} threshold")
        return 0.0
    
    # Calculate strength based on whale size relative to volume
    total_whale_usd = sum(valid_whales)
    whale_ratio = min(1.0, total_whale_usd / (volume_24h_usd * 0.1))  # 10% of volume = 1.0 strength
    
    print(f"[WHALE VALIDATION] {len(valid_whales)} valid whales, total=${total_whale_usd:,.0f}, strength={whale_ratio:.3f}")
    return whale_ratio

def get_whale_context_info(symbol: str, whale_usd_values: list, volume_24h_usd: float) -> Dict:
    """
    Get contextual information about whale activity for logging/debugging
    
    Args:
        symbol: Token symbol
        whale_usd_values: List of whale transaction values
        volume_24h_usd: 24h volume in USD
        
    Returns:
        Dict with whale analysis context
    """
    threshold = calculate_dynamic_whale_threshold(volume_24h_usd)
    valid_whales = [w for w in whale_usd_values if w >= threshold]
    
    return {
        "symbol": symbol,
        "volume_24h_usd": volume_24h_usd,
        "dynamic_threshold_usd": threshold,
        "total_whale_count": len(whale_usd_values),
        "valid_whale_count": len(valid_whales),
        "largest_whale_usd": max(whale_usd_values) if whale_usd_values else 0,
        "total_whale_usd": sum(valid_whales),
        "whale_strength": validate_whale_strength(whale_usd_values, volume_24h_usd),
        "ratio_to_volume": sum(valid_whales) / volume_24h_usd if volume_24h_usd > 0 else 0
    }