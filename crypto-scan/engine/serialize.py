"""
PUNKT 11: Serializacja do stealth_result/explore
Zunifikowany format bez rozjazdów whale/dex między detekcją a explore
"""

from typing import Dict, List, Any


def serialize_detection_result(ctx: Dict[str, Any], signals: Dict[str, Any], 
                             aggregator: Dict[str, Any], consensus: Dict[str, Any]) -> Dict[str, Any]:
    """
    PUNKT 11: Serializacja wyniku detekcji do zunifikowanego formatu
    
    Args:
        ctx: Context z price_ref i innymi danymi
        signals: Pełny dict sygnałów z active/strength/meta
        aggregator: Wynik agregatora {"z_raw", "p_raw", "contrib"}
        consensus: Wynik consensusu
        
    Returns:
        Zunifikowany format dla stealth_result i explore
    """
    # Nigdy nie nadpisuj istniejących sygnałów „fixami"
    out = {
        "price_ref": ctx.get("price_ref", 0.0),
        "signals": signals,            # pełny dict z active/strength/meta
        "active_signals": [name for name, data in signals.items() 
                          if isinstance(data, dict) and data.get("active", False)],
        "aggregator": aggregator,      # {"z_raw","p_raw","contrib"}
        "consensus": consensus,
    }
    
    return out


def save_stealth_result(symbol: str, result: Dict[str, Any], cache_dir: str = "crypto-scan/cache") -> None:
    """
    Zapisz wynik detekcji do stealth_result
    
    Args:
        symbol: Symbol tokena
        result: Zunifikowany wynik z serialize_detection_result
        cache_dir: Katalog cache
    """
    import json
    import os
    from datetime import datetime
    
    os.makedirs(f"{cache_dir}/stealth_result", exist_ok=True)
    
    # Dodaj timestamp i symbol
    result["timestamp"] = datetime.now().isoformat()
    result["symbol"] = symbol
    
    filepath = f"{cache_dir}/stealth_result/{symbol}_latest.json"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[SERIALIZE] Saved stealth result for {symbol}")
    except Exception as e:
        print(f"[SERIALIZE ERROR] Failed to save stealth result for {symbol}: {e}")


def load_stealth_result(symbol: str, cache_dir: str = "crypto-scan/cache") -> Dict[str, Any]:
    """
    Wczytaj wynik detekcji ze stealth_result
    
    Args:
        symbol: Symbol tokena
        cache_dir: Katalog cache
        
    Returns:
        Zunifikowany wynik lub pusty dict
    """
    import json
    import os
    
    filepath = f"{cache_dir}/stealth_result/{symbol}_latest.json"
    
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                result = json.load(f)
            return result
        else:
            return {}
    except Exception as e:
        print(f"[SERIALIZE ERROR] Failed to load stealth result for {symbol}: {e}")
        return {}


def save_explore_data(symbol: str, result: Dict[str, Any], cache_dir: str = "crypto-scan/cache") -> None:
    """
    Zapisz dane do explore mode - używa tego samego formatu co stealth_result
    
    Args:
        symbol: Symbol tokena
        result: Zunifikowany wynik z serialize_detection_result
        cache_dir: Katalog cache
    """
    import json
    import os
    from datetime import datetime
    
    os.makedirs(f"{cache_dir}/explore", exist_ok=True)
    
    # Dodaj timestamp i symbol
    result["timestamp"] = datetime.now().isoformat()
    result["symbol"] = symbol
    result["mode"] = "explore"
    
    filepath = f"{cache_dir}/explore/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[SERIALIZE] Saved explore data for {symbol}")
    except Exception as e:
        print(f"[SERIALIZE ERROR] Failed to save explore data for {symbol}: {e}")


def ensure_compatibility(legacy_signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Zapewnij kompatybilność z legacy formatami sygnałów
    
    Args:
        legacy_signals: Sygnały w starym formacie
        
    Returns:
        Sygnały w nowym zunifikowanym formacie
    """
    normalized = {}
    
    for name, data in legacy_signals.items():
        if isinstance(data, dict):
            # Już w poprawnym formacie
            normalized[name] = data
        elif isinstance(data, (int, float, bool)):
            # Legacy format - konwertuj do nowego
            normalized[name] = {
                "active": bool(data),
                "strength": float(data) if isinstance(data, (int, float)) else 1.0,
                "meta": {"converted_from_legacy": True}
            }
        else:
            # Nieznany format - stwórz podstawowy
            normalized[name] = {
                "active": False,
                "strength": 0.0,
                "meta": {"unknown_format": str(type(data))}
            }
    
    return normalized