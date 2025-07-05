"""
Invalid Symbol Filter
Eliminuje tokeny z błędnymi symbolami TradingView z całego pipeline TJDE
"""

import os
import json
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

def is_invalid_symbol(symbol: str) -> bool:
    """
    Sprawdza czy token ma status 'Invalid symbol' na podstawie metadanych GPT
    
    Args:
        symbol: Symbol tokena do sprawdzenia
        
    Returns:
        True jeśli token ma błędny symbol lub nie powinien być analizowany
    """
    try:
        # Sprawdź metadata w folderze training_data/charts/
        chart_files = []
        charts_dir = "training_data/charts"
        
        if os.path.exists(charts_dir):
            for file in os.listdir(charts_dir):
                if file.startswith(f"{symbol}_") and file.endswith("_metadata.json"):
                    chart_files.append(os.path.join(charts_dir, file))
        
        # Sprawdź też legacy metadata folder
        legacy_meta_path = f"metadata/{symbol}_metadata.json"
        if os.path.exists(legacy_meta_path):
            chart_files.append(legacy_meta_path)
            
        # Analizuj wszystkie znalezione pliki metadata
        for meta_path in chart_files:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                # Sprawdź GPT analysis
                gpt_analysis = meta.get("gpt_analysis", "").lower()
                setup_label = meta.get("setup_label", "").lower()
                
                # Flagowanie invalid symbols
                invalid_indicators = [
                    "invalid symbol",
                    "symbol not found", 
                    "no data available",
                    "chart not available",
                    "data unavailable"
                ]
                
                for indicator in invalid_indicators:
                    if indicator in gpt_analysis:
                        logger.info(f"[INVALID FILTER] {symbol}: Found '{indicator}' in GPT analysis")
                        return True
                
                # Sprawdź setup_label - problematyczne labels
                problematic_setups = [
                    "setup_analysis",
                    "unknown",
                    "no_clear_pattern",
                    "invalid_data"
                ]
                
                if setup_label in problematic_setups:
                    logger.info(f"[INVALID FILTER] {symbol}: Problematic setup_label '{setup_label}'")
                    return True
                    
            except Exception as e:
                logger.warning(f"[INVALID FILTER] Error reading {meta_path}: {e}")
                continue
                
        return False
        
    except Exception as e:
        logger.error(f"[INVALID FILTER] Error checking {symbol}: {e}")
        return False

def is_invalid_symbol_from_result(result: Dict) -> bool:
    """
    Sprawdza czy result z skanowania zawiera oznaki invalid symbol
    
    Args:
        result: Słownik z rezultatem skanowania tokena
        
    Returns:
        True jeśli token jest invalid
    """
    try:
        symbol = result.get("symbol", "")
        
        # Sprawdź flagę invalid_symbol jeśli istnieje
        if result.get("invalid_symbol", False):
            return True
            
        # Sprawdź chart_path
        chart_path = result.get("chart_path", "")
        if "TRADINGVIEW_FAILED" in chart_path:
            return True
            
        # Sprawdź GPT setup_label
        setup_label = result.get("setup_label", "").lower()
        problematic_setups = ["setup_analysis", "unknown", "no_clear_pattern", "invalid_data"]
        if setup_label in problematic_setups:
            return True
            
        # Sprawdź AI label
        ai_label = result.get("ai_label", {})
        if isinstance(ai_label, dict):
            ai_pattern = ai_label.get("label", "").lower()
            if ai_pattern in ["unknown", "no_pattern", "invalid"]:
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"[INVALID FILTER] Error checking result: {e}")
        return False

def filter_valid_symbols(results: List[Dict]) -> List[Dict]:
    """
    Filtruje listę rezultatów usuwając invalid symbols
    
    Args:
        results: Lista słowników z rezultatami skanowania
        
    Returns:
        Przefiltrowana lista bez invalid symbols
    """
    valid_results = []
    filtered_count = 0
    
    for result in results:
        symbol = result.get("symbol", "unknown")
        
        # Sprawdź czy symbol jest invalid
        if is_invalid_symbol(symbol) or is_invalid_symbol_from_result(result):
            filtered_count += 1
            logger.info(f"[INVALID FILTER] Filtered out {symbol} - invalid symbol")
            continue
            
        valid_results.append(result)
    
    if filtered_count > 0:
        logger.info(f"[INVALID FILTER] Filtered {filtered_count} invalid symbols, {len(valid_results)} valid symbols remain")
    
    return valid_results

def should_skip_symbol_analysis(symbol: str) -> bool:
    """
    Sprawdza czy symbol powinien być pominięty w analizie TJDE
    
    Args:
        symbol: Symbol do sprawdzenia
        
    Returns:
        True jeśli symbol powinien być pominięty
    """
    # Szybkie sprawdzenie na podstawie nazwy
    suspicious_patterns = [
        "1000000",  # Bardzo długie liczby
        "10000",    # Długie liczby
        "PEIPEI",   # Znane problematyczne tokeny
        "COQU",
        "LADIES"
    ]
    
    for pattern in suspicious_patterns:
        if pattern in symbol:
            logger.info(f"[INVALID FILTER] Suspicious symbol pattern in {symbol}")
            
    # Pełne sprawdzenie przez metadata
    return is_invalid_symbol(symbol)

def log_invalid_symbol_detection(symbol: str, reason: str):
    """
    Loguje wykrycie invalid symbol z powodem
    
    Args:
        symbol: Symbol tokena
        reason: Powód uznania za invalid
    """
    logger.warning(f"[INVALID SYMBOL] {symbol}: {reason}")

def get_invalid_symbols_stats() -> Dict:
    """
    Zwraca statystyki invalid symbols z folderu charts
    
    Returns:
        Słownik ze statystykami
    """
    try:
        stats = {
            "total_symbols": 0,
            "invalid_symbols": 0,
            "invalid_list": []
        }
        
        charts_dir = "training_data/charts"
        if not os.path.exists(charts_dir):
            return stats
            
        symbols = set()
        for file in os.listdir(charts_dir):
            if file.endswith("_metadata.json"):
                symbol = file.split("_")[0]
                symbols.add(symbol)
        
        stats["total_symbols"] = len(symbols)
        
        for symbol in symbols:
            if is_invalid_symbol(symbol):
                stats["invalid_symbols"] += 1
                stats["invalid_list"].append(symbol)
        
        return stats
        
    except Exception as e:
        logger.error(f"[INVALID FILTER] Error getting stats: {e}")
        return {"error": str(e)}

# Test function
def test_invalid_symbol_filter():
    """Test invalid symbol detection"""
    print("🧪 Testing Invalid Symbol Filter:")
    
    test_symbols = ["BTCUSDT", "1000000PEIPEIUSDT", "10000COQUSDT", "ETHUSDT"]
    
    for symbol in test_symbols:
        is_invalid = is_invalid_symbol(symbol)
        print(f"  {symbol}: {'❌ INVALID' if is_invalid else '✅ VALID'}")
    
    stats = get_invalid_symbols_stats()
    print(f"📊 Stats: {stats['invalid_symbols']}/{stats['total_symbols']} invalid symbols")
    
if __name__ == "__main__":
    test_invalid_symbol_filter()