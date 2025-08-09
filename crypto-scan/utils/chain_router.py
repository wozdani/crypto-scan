"""
Chain Router System - Consistent chain/contract mapping across modules
Eliminates chain mismatch between whale_ping, dex_inflow, and contract detection
"""

from typing import Dict, Tuple, Optional
from utils.contracts import get_contract

# Cache for chain routing to ensure consistency within scan round
_chain_cache = {}

def chain_router(symbol: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Consistent chain/contract router for entire scan round
    
    Args:
        symbol: Trading symbol (e.g., 'WIFUSDT')
        
    Returns:
        tuple: (chain, contract_address, status)
        status: 'found'|'chain_mismatch'|'not_found'|'cached'
    """
    
    # Check cache first for consistency within scan round
    if symbol in _chain_cache:
        cached = _chain_cache[symbol]
        return cached['chain'], cached['contract'], 'cached'
    
    # Get contract info from contracts module
    try:
        contract_info = get_contract(symbol)
        
        if contract_info and contract_info.get('address') and contract_info.get('chain'):
            chain = contract_info['chain']
            address = contract_info['address']
            
            # Validate chain/symbol compatibility
            compatibility_status = validate_chain_symbol_compatibility(symbol, chain)
            
            # Cache for consistency
            _chain_cache[symbol] = {
                'chain': chain,
                'contract': address,
                'status': compatibility_status
            }
            
            print(f"[CHAIN ROUTER] {symbol}: {chain}/{address[:10]}... ({compatibility_status})")
            return chain, address, compatibility_status
            
    except Exception as e:
        print(f"[CHAIN ROUTER ERROR] {symbol}: {e}")
    
    # No contract found
    _chain_cache[symbol] = {
        'chain': None,
        'contract': None,
        'status': 'not_found'
    }
    
    print(f"[CHAIN ROUTER] {symbol}: No contract found")
    return None, None, 'not_found'


def validate_chain_symbol_compatibility(symbol: str, chain: str) -> str:
    """
    Validate if symbol and chain are compatible
    
    Args:
        symbol: Trading symbol
        chain: Blockchain chain
        
    Returns:
        'found'|'chain_mismatch'
    """
    
    # Known chain mismatches (CEX symbols vs actual blockchain)
    known_mismatches = {
        'WIFUSDT': ['ethereum'],  # WIF is typically Solana but might have ethereum bridge
        'BONKUSDT': ['ethereum'], # BONK is typically Solana but might have ethereum bridge
        'RAYUSDT': ['ethereum'],  # RAY is typically Solana but might have ethereum bridge
    }
    
    # Check for known mismatches
    symbol_base = symbol.replace('USDT', '').replace('USDC', '').replace('BUSD', '')
    
    if symbol in known_mismatches and chain in known_mismatches[symbol]:
        print(f"[CHAIN COMPATIBILITY] {symbol}: Potential mismatch - {symbol_base} typically not on {chain}")
        return 'chain_mismatch'
    
    # Additional validation rules
    if chain == 'ethereum' and symbol_base in ['WIF', 'BONK', 'RAY', 'SOL']:
        print(f"[CHAIN COMPATIBILITY] {symbol}: Solana token found on ethereum (likely bridge)")
        return 'chain_mismatch'
    
    return 'found'


def get_chain_status_for_dex(symbol: str) -> Tuple[bool, str]:
    """
    Determine if DEX inflow should be enabled for this symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        tuple: (enabled, reason)
    """
    
    chain, contract, status = chain_router(symbol)
    
    if status == 'not_found':
        return False, f"no_contract_found"
    
    if status == 'chain_mismatch':
        return False, f"chain_mismatch_{chain}"
    
    if status in ['found', 'cached']:
        return True, f"enabled_{chain}"
    
    return False, f"unknown_status_{status}"


def clear_chain_cache():
    """Clear chain cache for new scan round"""
    global _chain_cache
    _chain_cache = {}
    print("[CHAIN ROUTER] Cache cleared for new scan round")


def get_cached_chain_info(symbol: str) -> Optional[Dict]:
    """Get cached chain info for debugging"""
    return _chain_cache.get(symbol)


def debug_chain_routing(symbol: str) -> Dict:
    """Debug chain routing for specific symbol"""
    
    chain, contract, status = chain_router(symbol)
    dex_enabled, dex_reason = get_chain_status_for_dex(symbol)
    cached_info = get_cached_chain_info(symbol)
    
    return {
        'symbol': symbol,
        'chain': chain,
        'contract': contract[:10] + '...' if contract else None,
        'status': status,
        'dex_enabled': dex_enabled,
        'dex_reason': dex_reason,
        'cached': cached_info is not None,
        'cache_details': cached_info
    }


# Export functions
__all__ = [
    'chain_router',
    'get_chain_status_for_dex', 
    'clear_chain_cache',
    'debug_chain_routing',
    'validate_chain_symbol_compatibility'
]