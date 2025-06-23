"""
Bybit Symbols Cache Manager - podobny do mechanizmu CoinGecko cache
Automatyczne sprawdzanie, odnawianie i walidacja cache symboli Bybit
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Cache paths
BYBIT_CACHE_PATHS = [
    "data/cache/bybit_symbols.json",
    "utils/data/cache/bybit_symbols.json"
]
BYBIT_CACHE_MAIN = "data/cache/bybit_symbols.json"
# Cache expiry removed - only check if empty

def should_rebuild_bybit_cache() -> bool:
    """
    Sprawdza czy cache symboli Bybit powinien zostać odświeżony
    Odświeża tylko gdy plik nie istnieje lub jest pusty
    
    Returns:
        bool: True jeśli cache powinien być odświeżony
    """
    try:
        # Sprawdź czy główny plik cache istnieje
        if not os.path.exists(BYBIT_CACHE_MAIN):
            logger.info("Bybit cache nie istnieje - wymagane odświeżenie")
            return True
        
        # Sprawdź czy plik nie jest pusty
        try:
            with open(BYBIT_CACHE_MAIN, 'r') as f:
                symbols = json.load(f)
                
            if not symbols or len(symbols) == 0:
                logger.info("Bybit cache jest pusty - wymagane odświeżenie")
                return True
                
            if len(symbols) < 50:  # Zbyt mało symboli - prawdopodobnie błąd
                logger.info(f"Bybit cache zawiera tylko {len(symbols)} symboli - wymagane odświeżenie")
                return True
                
            logger.info(f"Bybit cache jest aktualny - zawiera {len(symbols)} symboli")
            return False
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Błąd odczytu Bybit cache: {e} - wymagane odświeżenie")
            return True
            
    except Exception as e:
        logger.error(f"Błąd sprawdzania Bybit cache: {e} - wymuszam odświeżenie")
        return True

def build_bybit_symbols_cache() -> bool:
    """
    Buduje cache symboli Bybit podobnie do build_coingecko_cache()
    
    Returns:
        bool: True jeśli cache został pomyślnie zbudowany
    """
    try:
        logger.info("🔄 Budowanie cache symboli Bybit...")
        
        # Pobierz symbole z API
        symbols = fetch_bybit_symbols_from_api()
        
        if not symbols:
            logger.error("❌ Nie udało się pobrać symboli z Bybit API")
            return False
        
        # Upewnij się że katalog istnieje
        os.makedirs(os.path.dirname(BYBIT_CACHE_MAIN), exist_ok=True)
        
        # Zapisz do głównego cache
        with open(BYBIT_CACHE_MAIN, 'w') as f:
            json.dump(symbols, f, indent=2)
        
        # Synchronizuj z alternatywnymi lokalizacjami
        for cache_path in BYBIT_CACHE_PATHS:
            if cache_path != BYBIT_CACHE_MAIN:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'w') as f:
                        json.dump(symbols, f, indent=2)
                except Exception as e:
                    logger.warning(f"Nie udało się synchronizować cache do {cache_path}: {e}")
        
        logger.info(f"✅ Cache symboli Bybit zbudowany pomyślnie - {len(symbols)} symboli")
        return True
        
    except Exception as e:
        logger.error(f"❌ Błąd budowania cache symboli Bybit: {e}")
        return False

def fetch_bybit_symbols_from_api() -> List[str]:
    """
    Pobiera symbole z Bybit API - logika z utils/bybit_symbols.py
    
    Returns:
        List[str]: Lista symboli USDT
    """
    try:
        logger.info("📡 Pobieranie symboli z Bybit API...")
        
        url = "https://api.bybit.com/v5/market/instruments-info"
        params = {
            "category": "linear"  # Linear futures (perpetual)
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("retCode") != 0:
            logger.error(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
            return []
        
        instruments = data.get("result", {}).get("list", [])
        
        # Filtruj tylko USDT perpetual contracts które są aktywne
        usdt_symbols = []
        for instrument in instruments:
            symbol = instrument.get("symbol", "")
            status = instrument.get("status", "")
            
            if (symbol.endswith("USDT") and 
                status == "Trading" and 
                instrument.get("contractType") == "LinearPerpetual"):
                usdt_symbols.append(symbol)
        
        logger.info(f"✅ Pobrano {len(usdt_symbols)} aktywnych symboli USDT z Bybit")
        return sorted(usdt_symbols)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Bybit symbols: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching Bybit symbols: {e}")
        return []

def get_bybit_symbols_cached() -> List[str]:
    """
    Pobiera symbole z cache z automatycznym odświeżaniem
    Podobne do get_symbols_cached() w data_fetchers
    
    Returns:
        List[str]: Lista symboli z cache
    """
    try:
        # Sprawdź czy cache wymaga odświeżenia
        if should_rebuild_bybit_cache():
            logger.info("🔄 Odświeżanie cache symboli Bybit...")
            build_success = build_bybit_symbols_cache()
            
            if not build_success:
                logger.warning("⚠️ Nie udało się odświeżyć cache - próbuję załadować stary cache")
        
        # Załaduj symbole z cache
        for cache_path in BYBIT_CACHE_PATHS:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        symbols = json.load(f)
                    
                    if symbols and len(symbols) > 0:
                        logger.info(f"📋 Załadowano {len(symbols)} symboli z cache: {cache_path}")
                        return symbols
                        
                except Exception as e:
                    logger.warning(f"Błąd ładowania cache z {cache_path}: {e}")
                    continue
        
        # Fallback - spróbuj pobrać bezpośrednio z API
        logger.warning("⚠️ Wszystkie cache są niedostępne - pobieranie bezpośrednio z API")
        symbols = fetch_bybit_symbols_from_api()
        
        if symbols:
            # Zapisz do cache dla przyszłych użyć
            try:
                os.makedirs(os.path.dirname(BYBIT_CACHE_MAIN), exist_ok=True)
                with open(BYBIT_CACHE_MAIN, 'w') as f:
                    json.dump(symbols, f, indent=2)
                logger.info("💾 Zapisano symbole do cache")
            except Exception as e:
                logger.warning(f"Nie udało się zapisać do cache: {e}")
        
        return symbols
        
    except Exception as e:
        logger.error(f"❌ Krytyczny błąd pobierania symboli Bybit: {e}")
        return []

def validate_bybit_cache() -> Dict[str, any]:
    """
    Waliduje stan cache symboli Bybit
    
    Returns:
        Dict: Raport stanu cache
    """
    report = {
        "cache_files": {},
        "total_symbols": 0,
        "cache_age_hours": 0,
        "is_valid": False,
        "recommendations": []
    }
    
    try:
        main_cache_exists = os.path.exists(BYBIT_CACHE_MAIN)
        
        if main_cache_exists:
            # Sprawdź wiek (dla informacji)
            file_time = datetime.fromtimestamp(os.path.getmtime(BYBIT_CACHE_MAIN))
            age = datetime.now() - file_time
            report["cache_age_hours"] = age.total_seconds() / 3600
            
            # Sprawdź zawartość
            try:
                with open(BYBIT_CACHE_MAIN, 'r') as f:
                    symbols = json.load(f)
                report["total_symbols"] = len(symbols) if symbols else 0
                
                if report["total_symbols"] > 50:
                    report["is_valid"] = True
                else:
                    if report["total_symbols"] <= 50:
                        report["recommendations"].append("Zbyt mało symboli - odśwież cache")
                        
            except Exception as e:
                report["recommendations"].append(f"Błąd odczytu cache: {e}")
        else:
            report["recommendations"].append("Główny cache nie istnieje - zbuduj cache")
        
        # Sprawdź wszystkie lokalizacje cache
        for cache_path in BYBIT_CACHE_PATHS:
            exists = os.path.exists(cache_path)
            size = 0
            if exists:
                try:
                    with open(cache_path, 'r') as f:
                        symbols = json.load(f)
                    size = len(symbols) if symbols else 0
                except:
                    size = -1  # Error reading
            
            report["cache_files"][cache_path] = {
                "exists": exists,
                "symbols_count": size
            }
        
        return report
        
    except Exception as e:
        report["recommendations"].append(f"Błąd walidacji: {e}")
        return report

if __name__ == "__main__":
    # Test cache manager
    print("🧪 Testing Bybit Cache Manager...")
    
    # Sprawdź stan cache
    validation = validate_bybit_cache()
    print(f"Cache validation: {validation}")
    
    # Test pobierania symboli
    symbols = get_bybit_symbols_cached()
    print(f"Retrieved {len(symbols)} symbols")
    
    if symbols:
        print("Sample symbols:", symbols[:10])