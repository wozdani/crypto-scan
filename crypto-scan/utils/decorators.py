"""
Reusable decorators for crypto scanner system
"""

import functools
from typing import Dict, Any, Callable


# Global scan cache - clears when new scan starts
_scan_cache: Dict[str, Any] = {}


def clear_scan_cache():
    """Clear the scan cache for new scan cycle"""
    global _scan_cache
    _scan_cache.clear()


def once_per_scan(namespace: str, module: str):
    """
    Cache function result for the duration of a single scan cycle
    
    Args:
        namespace: Namespace for the cache key (e.g., "FEATURES", "AI")
        module: Module for the cache key (e.g., "engine", "detector")
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create unique cache key
            cache_key = f"{namespace}.{module}.{func.__name__}.{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Return cached result if available
            if cache_key in _scan_cache:
                return _scan_cache[cache_key]
            
            # Calculate result and cache it
            result = func(*args, **kwargs)
            _scan_cache[cache_key] = result
            return result
        
        return wrapper
    return decorator