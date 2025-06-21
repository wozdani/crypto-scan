"""
Bybit Symbols Fetcher - pobiera wszystkie symbole z Bybit API jednym zapytaniem
Podobne do implementacji w pump-analysis
"""

import requests
import json
import time
from typing import List, Dict, Optional

def get_all_bybit_symbols() -> List[str]:
    """
    Pobiera wszystkie dostępne symbole USDT z Bybit API
    
    Returns:
        List[str]: Lista symboli zakończonych na USDT
    """
    try:
        url = "https://api.bybit.com/v5/market/instruments-info"
        params = {
            "category": "linear"  # Linear futures (perpetual)
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("retCode") != 0:
            print(f"❌ Bybit API error: {data.get('retMsg', 'Unknown error')}")
            return []
        
        instruments = data.get("result", {}).get("list", [])
        
        # Filter only USDT perpetual contracts that are trading
        usdt_symbols = []
        for instrument in instruments:
            symbol = instrument.get("symbol", "")
            status = instrument.get("status", "")
            
            if (symbol.endswith("USDT") and 
                status == "Trading" and 
                instrument.get("contractType") == "LinearPerpetual"):
                usdt_symbols.append(symbol)
        
        print(f"✅ Pobrano {len(usdt_symbols)} aktywnych symboli USDT z Bybit")
        return sorted(usdt_symbols)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching Bybit symbols: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error fetching Bybit symbols: {e}")
        return []

def get_symbol_details(symbols: List[str]) -> Dict[str, Dict]:
    """
    Pobiera szczegółowe informacje o symbolach (ceny, wolumeny, itp.)
    
    Args:
        symbols: Lista symboli do sprawdzenia
        
    Returns:
        Dict: Słownik z danymi symboli {symbol: {price, volume, change24h, ...}}
    """
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {
            "category": "linear"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("retCode") != 0:
            print(f"❌ Bybit tickers API error: {data.get('retMsg', 'Unknown error')}")
            return {}
        
        tickers = data.get("result", {}).get("list", [])
        
        # Create symbol details dictionary
        symbol_details = {}
        for ticker in tickers:
            symbol = ticker.get("symbol", "")
            if symbol in symbols:
                symbol_details[symbol] = {
                    "price": float(ticker.get("lastPrice", 0)),
                    "volume24h": float(ticker.get("volume24h", 0)),
                    "turnover24h": float(ticker.get("turnover24h", 0)),
                    "price24hPcnt": float(ticker.get("price24hPcnt", 0)),
                    "highPrice24h": float(ticker.get("highPrice24h", 0)),
                    "lowPrice24h": float(ticker.get("lowPrice24h", 0)),
                    "bid1Price": float(ticker.get("bid1Price", 0)),
                    "ask1Price": float(ticker.get("ask1Price", 0))
                }
        
        print(f"✅ Pobrano szczegóły dla {len(symbol_details)} symboli")
        return symbol_details
        
    except Exception as e:
        print(f"❌ Error fetching symbol details: {e}")
        return {}

def get_filtered_symbols(min_volume_usdt: float = 1000000, 
                        min_price: float = 0.001, 
                        max_price: float = 1000,
                        max_spread_pct: float = 2.0) -> List[str]:
    """
    Pobiera i filtruje symbole według kryteriów jakości
    
    Args:
        min_volume_usdt: Minimalny wolumen 24h w USDT
        min_price: Minimalna cena
        max_price: Maksymalna cena  
        max_spread_pct: Maksymalny spread w %
        
    Returns:
        List[str]: Lista przefiltrowanych symboli
    """
    print(f"🔍 Fetching and filtering Bybit symbols...")
    print(f"   Min volume: ${min_volume_usdt:,.0f}")
    print(f"   Price range: ${min_price} - ${max_price}")
    print(f"   Max spread: {max_spread_pct}%")
    
    # Get all symbols
    all_symbols = get_all_bybit_symbols()
    if not all_symbols:
        return []
    
    # Get details for filtering
    symbol_details = get_symbol_details(all_symbols)
    if not symbol_details:
        return all_symbols  # Return unfiltered if details fetch failed
    
    # Apply filters
    filtered_symbols = []
    for symbol in all_symbols:
        details = symbol_details.get(symbol)
        if not details:
            continue
            
        price = details["price"]
        volume_usdt = details["turnover24h"]
        bid = details["bid1Price"]
        ask = details["ask1Price"]
        
        # Skip if price out of range
        if price < min_price or price > max_price:
            continue
            
        # Skip if volume too low
        if volume_usdt < min_volume_usdt:
            continue
            
        # Skip if spread too wide
        if bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / ask) * 100
            if spread_pct > max_spread_pct:
                continue
        
        filtered_symbols.append(symbol)
    
    print(f"✅ Filtered {len(filtered_symbols)}/{len(all_symbols)} symbols passed quality filters")
    return sorted(filtered_symbols)

if __name__ == "__main__":
    # Test the functions
    print("Testing Bybit symbols fetcher...")
    
    symbols = get_filtered_symbols(
        min_volume_usdt=500000,  # $500K min volume
        min_price=0.01,          # $0.01 min price
        max_price=100,           # $100 max price
        max_spread_pct=1.0       # 1% max spread
    )
    
    print(f"Total quality symbols: {len(symbols)}")
    if symbols:
        print("Sample symbols:", symbols[:10])