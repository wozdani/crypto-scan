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
    Pobiera symbole z Bybit API używając istniejącego systemu autoryzacji
    
    Returns:
        List[str]: Lista symboli USDT
    """
    try:
        logger.info("📡 Pobieranie symboli z Bybit API...")
        
        # Użyj istniejącego systemu z data_fetchers.py
        try:
            from utils.data_fetchers import get_symbols_cached as get_symbols_from_data_fetchers
            symbols = get_symbols_from_data_fetchers()
            
            if symbols and len(symbols) > 0:
                logger.info(f"✅ Pobrano {len(symbols)} symboli z data_fetchers")
                return symbols
        except Exception as e:
            logger.warning(f"Nie udało się użyć data_fetchers: {e}")
        
        # Fallback - próbuj z autoryzacją Bybit
        try:
            import hmac
            import hashlib
            
            api_key = os.getenv('BYBIT_API_KEY')
            secret_key = os.getenv('BYBIT_SECRET_KEY')
            
            if api_key and secret_key:
                logger.info("Używam autoryzowanych zapytań Bybit")
                
                timestamp = str(int(time.time() * 1000))
                recv_window = "5000"
                
                params = {
                    "category": "linear"
                }
                
                param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                raw_str = timestamp + api_key + recv_window + param_str
                
                signature = hmac.new(
                    secret_key.encode('utf-8'),
                    raw_str.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                headers = {
                    "X-BAPI-API-KEY": api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-SIGN-TYPE": "2",
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": recv_window,
                    "Content-Type": "application/json"
                }
                
                url = "https://api.bybit.com/v5/market/tickers"
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("retCode") != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                    return get_fallback_symbols()
                
                tickers = data.get("result", {}).get("list", [])
                
                usdt_symbols = []
                for ticker in tickers:
                    symbol = ticker.get("symbol", "")
                    if symbol.endswith("USDT"):
                        usdt_symbols.append(symbol)
                
                if len(usdt_symbols) < 50:
                    logger.warning(f"Otrzymano tylko {len(usdt_symbols)} symboli - używam fallback")
                    return get_fallback_symbols()
                
                logger.info(f"✅ Pobrano {len(usdt_symbols)} symboli USDT z autoryzowanym API")
                return sorted(usdt_symbols)
                
        except Exception as e:
            logger.error(f"Błąd autoryzowanego zapytania: {e}")
        
        # Ostateczny fallback
        return get_fallback_symbols()
        
    except Exception as e:
        logger.error(f"Unexpected error fetching Bybit symbols: {e}")
        return get_fallback_symbols()

def get_fallback_symbols() -> List[str]:
    """
    Fallback lista głównych symboli USDT gdy API nie działa
    
    Returns:
        List[str]: Lista podstawowych symboli
    """
    logger.info("📋 Używam fallback listy symboli")
    
    # Lista głównych kryptowalut które zwykle są dostępne na Bybit
    fallback_symbols = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT",
        "SOLUSDT", "AVAXUSDT", "LUNAUSDT", "MATICUSDT", "FTMUSDT", "ATOMUSDT",
        "NEARUSDT", "ALGOUSDT", "VETUSDT", "ICPUSDT", "AXSUSDT", "SANDUSDT",
        "MANAUSDT", "ENJUSDT", "CHZUSDT", "FLOWUSDT", "GALAUSDT", "GMTUSDT",
        "APEUSDT", "OPUSDT", "ARBUSDT", "LDOUSDT", "MAGICUSDT", "BLZUSDT",
        "UNFIUSDT", "CTKUSDT", "KLAYUSDT", "ROSEUSDT", "DARUSDT", "PEOPLEUSDT",
        "JASMYUSDT", "WOOUSDT", "BELUSDT", "SFPUSDT", "ANCUSDT", "WAXUSDT",
        "TLMUSDT", "CVXUSDT", "STGUSDT", "GALUSDT", "LRCUSDT", "BATUSDT",
        "IOTXUSDT", "ONTUSDT", "ZENUSDT", "QTUMUSDT", "ZILUSDT", "RLCUSDT",
        "C98USDT", "CELRUSDT", "BAKEUSDT", "OCEANUSDT", "SKLUSDT", "GRTUSDT",
        "1INCHUSDT", "CKBUSDT", "RENUSDT", "RSRUSDT", "REEFUSDT", "STORJUSDT",
        "COTIUSDT", "DENTUSDT", "CHRUSDT", "PENDLEUSDT", "AMBUSDT", "SUSHIUSDT",
        "YFIUSDT", "AAVEUSDT", "COMPUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT",
        "BALUSDT", "UNIUSDT", "SXPUSDT", "BANDUSDT", "ALPHAUSDT", "KAVAUSDT",
        "RUNEUSDT", "YFIIUSDT", "DEFIUSDT", "AUDIOUSDT", "CTSIUSDT", "OGNUSDT",
        "HNTUSDT", "OMGUSDT", "IOSTUSDT", "DUSKUSDT", "FLMUSDT", "SCRTUSDT",
        "MDTUSDT", "WINUSDT", "TROYUSDT", "VITEUSDT", "ONEUSDT", "FTOUSDT",
        "HOTUSDT", "DREPUSDT", "TCOUSDT", "FISUSDT", "ORNUSDT", "CKBUSDT"
    ]
    
    return sorted(list(set(fallback_symbols)))  # Remove duplicates and sort

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