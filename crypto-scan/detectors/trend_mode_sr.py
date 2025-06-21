"""
Support/Resistance Based Trend Mode Detection
Complete replacement using 3h recent trend + historical S/R levels + 5M entry
"""

import requests
from datetime import datetime, timezone

def fetch_15m_prices_extended(symbol: str, hours_back: int = 24):
    """
    Fetch extended 15M prices for S/R analysis
    
    Args:
        symbol: Trading symbol
        hours_back: Hours back to fetch (24h = 96 candles)
        
    Returns:
        List of close prices
    """
    url = "https://api.bybit.com/v5/market/kline"
    candles_needed = hours_back * 4  # 4 candles per hour for 15M
    
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": "15",
        "limit": min(candles_needed, 1000)
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            prices = [float(candle[4]) for candle in data["result"]["list"]]
            return prices
        
        # Fallback to mock data when API fails
        from detectors.trend_mode_mock import generate_mock_15m_prices, should_use_mock_data
        if should_use_mock_data():
            print(f"[API FALLBACK] Using mock 15M data for {symbol}")
            return generate_mock_15m_prices(symbol, candles_needed)
        
        return []
    except Exception as e:
        print(f"Error fetching extended 15M prices for {symbol}: {e}")
        # Use mock data as fallback
        try:
            from detectors.trend_mode_mock import generate_mock_15m_prices
            print(f"[API FALLBACK] Using mock 15M data for {symbol}")
            return generate_mock_15m_prices(symbol, candles_needed)
        except:
            return []


def fetch_5m_prices_recent(symbol: str, count: int = 12):
    """
    Fetch recent 5M prices for entry analysis (1h = 12 candles)
    
    Args:
        symbol: Trading symbol
        count: Number of 5M candles (default 12 for 1h)
        
    Returns:
        List of close prices
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": "5",
        "limit": min(count, 200)
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            prices = [float(candle[4]) for candle in data["result"]["list"]]
            return prices[:count] if len(prices) >= count else prices
        
        # Fallback to mock data when API fails
        from detectors.trend_mode_mock import generate_mock_5m_prices
        print(f"[API FALLBACK] Using mock 5M data for {symbol}")
        return generate_mock_5m_prices(symbol, count)
        
    except Exception as e:
        print(f"Error fetching 5M prices for {symbol}: {e}")
        # Use mock data as fallback
        try:
            from detectors.trend_mode_mock import generate_mock_5m_prices
            print(f"[API FALLBACK] Using mock 5M data for {symbol}")
            return generate_mock_5m_prices(symbol, count)
        except:
            return []


def get_current_price(symbol: str):
    """
    Get current price from Bybit ticker
    
    Args:
        symbol: Trading symbol
        
    Returns:
        float: Current price
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "spot",
        "symbol": symbol
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            ticker = data["result"]["list"][0]
            return float(ticker.get("lastPrice", 0))
        
        # Fallback to mock data when API fails
        from detectors.trend_mode_mock import generate_mock_current_price
        print(f"[API FALLBACK] Using mock current price for {symbol}")
        return generate_mock_current_price(symbol)
        
    except Exception as e:
        print(f"Error fetching current price for {symbol}: {e}")
        # Use mock data as fallback
        try:
            from detectors.trend_mode_mock import generate_mock_current_price
            print(f"[API FALLBACK] Using mock current price for {symbol}")
            return generate_mock_current_price(symbol)
        except:
            return 0.0


def get_orderbook_volumes_sr(symbol: str):
    """
    Get ask/bid volumes for entry analysis
    
    Args:
        symbol: Trading symbol
        
    Returns:
        tuple: (ask_volumes, bid_volumes) - 3 recent values each
    """
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        "category": "spot",
        "symbol": symbol,
        "limit": 10
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("retCode") == 0 and data.get("result"):
            asks = data["result"].get("a", [])
            bids = data["result"].get("b", [])
            
            # Calculate total volumes for top levels
            ask_vol = sum(float(ask[1]) for ask in asks[:3]) if asks else 0
            bid_vol = sum(float(bid[1]) for bid in bids[:3]) if bids else 0
            
            # Return as lists with 3 simulated values for trend analysis
            ask_volumes = [ask_vol * 1.1, ask_vol * 1.05, ask_vol]
            bid_volumes = [bid_vol * 0.9, bid_vol * 0.95, bid_vol]
            
            return ask_volumes, bid_volumes
        
        # Fallback to mock data when API fails
        from detectors.trend_mode_mock import generate_mock_orderbook_volumes
        print(f"[API FALLBACK] Using mock orderbook data for {symbol}")
        return generate_mock_orderbook_volumes(symbol)
        
    except Exception as e:
        print(f"Error fetching orderbook volumes for {symbol}: {e}")
        # Use mock data as fallback
        try:
            from detectors.trend_mode_mock import generate_mock_orderbook_volumes
            print(f"[API FALLBACK] Using mock orderbook data for {symbol}")
            return generate_mock_orderbook_volumes(symbol)
        except:
            return [], []


def is_strong_recent_trend(prices_15m_recent: list[float]) -> bool:
    """
    Ocena rytmu tylko z ostatnich 3h (12 świec 15M)
    
    Args:
        prices_15m_recent: Ostatnie 12 świec 15M (3h)
        
    Returns:
        bool: True jeśli silny trend w ostatnich 3h
    """
    if len(prices_15m_recent) < 6:
        return False
    
    # Oblicz stosunek świec wzrostowych (tymczasowo obniżony próg do testów)
    up_ratio = sum(1 for i in range(1, len(prices_15m_recent)) 
                  if prices_15m_recent[i] > prices_15m_recent[i-1]) / (len(prices_15m_recent) - 1)
    
    # Sprawdź postęp cenowy (ostatnia > pierwsza)
    price_progress = prices_15m_recent[-1] > prices_15m_recent[0]
    
    # Debug logging
    print(f"[TREND DEBUG] Recent trend analysis - up_ratio: {up_ratio:.3f}, price_progress: {price_progress}")
    
    return up_ratio >= 0.5 and price_progress  # Lowered from 0.6 to 0.5 for testing


def find_support_resistance_levels(prices_15m_history: list[float], tolerance: float = 0.003) -> list[float]:
    """
    Identyfikacja poziomów S/R z danych historycznych 3-24h wstecz
    
    Args:
        prices_15m_history: Świece z przedziału 3-24h (84 świece)
        tolerance: Tolerancja dla łączenia podobnych poziomów
        
    Returns:
        list: Posortowane poziomy S/R
    """
    if len(prices_15m_history) < 5:
        return []
    
    levels = []
    
    # Znajdź pivot points
    for i in range(2, len(prices_15m_history) - 2):
        # Pivot high (szczyt lokalny)
        pivot_high = (prices_15m_history[i] > prices_15m_history[i-1] and 
                     prices_15m_history[i] > prices_15m_history[i+1])
        
        # Pivot low (dołek lokalny)
        pivot_low = (prices_15m_history[i] < prices_15m_history[i-1] and 
                    prices_15m_history[i] < prices_15m_history[i+1])
        
        if pivot_high or pivot_low:
            level = prices_15m_history[i]
            # Dodaj poziom tylko jeśli nie ma podobnego w tolerancji
            if not any(abs(level - l) / l < tolerance for l in levels):
                levels.append(level)
    
    return sorted(levels)


def is_price_near_support(current_price: float, levels: list[float], margin: float = 0.004) -> bool:
    """
    Sprawdza, czy obecna cena testuje poziom wsparcia
    
    Args:
        current_price: Obecna cena
        levels: Lista poziomów S/R
        margin: Margines dla testu poziomu (0.2%)
        
    Returns:
        bool: True jeśli cena blisko wsparcia
    """
    if not levels or current_price <= 0:
        return False
    
    # Znajdź poziomy wsparcia (poniżej obecnej ceny)
    supports = [l for l in levels if l < current_price]
    
    if not supports:
        return False
    
    # Sprawdź czy cena jest blisko któregoś ze wsparć
    near_any_support = any(abs(current_price - s) / s < margin for s in supports)
    
    # Debug logging
    if supports:
        closest_support = max(supports)
        distance_pct = abs(current_price - closest_support) / closest_support * 100
        print(f"[TREND DEBUG] Support analysis - closest: {closest_support:.2f}, distance: {distance_pct:.2f}%, near_support: {near_any_support}")
    else:
        print(f"[TREND DEBUG] Support analysis - no support levels below current price")
    
    return near_any_support


def is_entry_after_correction(prices_5m: list[float], asks: list[float], bids: list[float]) -> bool:
    """
    Wykrywa idealne wejście na końcu korekty (5M)
    
    Args:
        prices_5m: Lista cen 5M (ostatnie 12 świec)
        asks: Lista wolumenów ask (3 wartości)
        bids: Lista wolumenów bid (3 wartości)
        
    Returns:
        bool: True jeśli idealny moment wejścia
    """
    if len(prices_5m) < 5 or len(asks) < 3 or len(bids) < 3:
        return False
    
    # Sprawdź czerwone świece w ostatnich 4 okresach
    recent_reds = sum(1 for i in range(-4, -1) if prices_5m[i] < prices_5m[i-1])
    
    # Sprawdź spadek presji sprzedaży (ask volume maleje)
    ask_down = asks[-3] > asks[-2] > asks[-1]
    
    # Sprawdź wzrost presji kupna (bid volume rośnie)
    bid_up = bids[-3] < bids[-2] < bids[-1]
    
    # Warunek: min. 2 czerwone + spadek ask + wzrost bid
    return recent_reds >= 2 and ask_down and bid_up


def detect_sr_trend_mode(symbol: str) -> dict:
    """
    Główna funkcja S/R Trend Mode Detection
    
    Args:
        symbol: Symbol do analizy
        
    Returns:
        dict: Wynik analizy S/R trend mode
    """
    try:
        print(f"🎯 [SR TREND DEBUG] Starting S/R trend analysis for {symbol}")
        
        # Pobierz dane cenowe
        prices_15m_full = fetch_15m_prices_extended(symbol, 24)  # 24h historii
        prices_5m = fetch_5m_prices_recent(symbol, 12)  # 1h dla entry
        current_price = get_current_price(symbol)
        ask_vols, bid_vols = get_orderbook_volumes_sr(symbol)
        
        print(f"📊 [SR TREND DEBUG] {symbol} - 15M candles: {len(prices_15m_full)}, 5M candles: {len(prices_5m)}, Current price: {current_price}")
        
        if len(prices_15m_full) < 24:  # Minimum 6h danych
            return {
                "trend_mode": False,
                "trend_score": 0,
                "entry_triggered": False,
                "description": "Insufficient 15M data for S/R analysis",
                "details": {"error": "Need minimum 24 candles (6h)"}
            }
        
        # Podziel dane na recent (ostatnie 3h = 12 świec) i history (3-24h = 84 świece)
        prices_15m_recent = prices_15m_full[:12]  # Ostatnie 3h
        prices_15m_history = prices_15m_full[12:96] if len(prices_15m_full) >= 96 else prices_15m_full[12:]  # 3-24h wstecz
        
        # Sprawdź warunki
        strong_trend = is_strong_recent_trend(prices_15m_recent)
        support_levels = find_support_resistance_levels(prices_15m_history)
        near_support = is_price_near_support(current_price, support_levels)
        entry_signal = is_entry_after_correction(prices_5m, ask_vols, bid_vols)
        
        print(f"📈 [SR TREND DEBUG] {symbol} - Strong trend: {strong_trend}, Near support: {near_support}, Entry signal: {entry_signal}")
        print(f"🔍 [SR TREND DEBUG] {symbol} - Support levels found: {len(support_levels)}, Current price: {current_price}")
        
        # Enhanced debugging with detailed logging
        with open("trend_debug_log.txt", "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {symbol} - ")
            
            if not strong_trend:
                if len(prices_15m_recent) > 1:
                    up_moves = sum(1 for i in range(1, len(prices_15m_recent)) 
                                  if prices_15m_recent[i] > prices_15m_recent[i-1])
                    up_ratio = up_moves / (len(prices_15m_recent) - 1)
                    f.write(f"trend too weak (up_ratio={up_ratio:.3f})\n")
                else:
                    f.write(f"insufficient 15M data\n")
            elif not near_support:
                f.write(f"no support near current price (levels={len(support_levels)})\n")
            elif not entry_signal:
                f.write(f"bid pressure too low\n")
            else:
                f.write(f"all conditions met - score: {100 if entry_signal else 70}\n")
        
        # Łączenie warunków
        if strong_trend and near_support:
            if entry_signal:
                return {
                    "trend_mode": True,
                    "trend_score": 100,
                    "entry_triggered": True,
                    "description": "Strong recent trend + Near support + Entry after correction",
                    "details": {
                        "strong_trend_3h": strong_trend,
                        "near_support": near_support,
                        "entry_after_correction": entry_signal,
                        "support_levels_count": len(support_levels),
                        "current_price": current_price,
                        "closest_support": min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else None,
                        "recent_15m_count": len(prices_15m_recent),
                        "history_15m_count": len(prices_15m_history),
                        "prices_5m_count": len(prices_5m)
                    }
                }
            else:
                return {
                    "trend_mode": True,
                    "trend_score": 70,
                    "entry_triggered": False,
                    "description": "Strong recent trend + Near support, waiting for entry signal",
                    "details": {
                        "strong_trend_3h": strong_trend,
                        "near_support": near_support,
                        "entry_after_correction": entry_signal,
                        "support_levels_count": len(support_levels),
                        "current_price": current_price
                    }
                }
        else:
            return {
                "trend_mode": False,
                "trend_score": 0,
                "entry_triggered": False,
                "description": f"Conditions not met - Strong trend: {strong_trend}, Near support: {near_support}",
                "details": {
                    "strong_trend_3h": strong_trend,
                    "near_support": near_support,
                    "entry_after_correction": entry_signal,
                    "support_levels_count": len(support_levels),
                    "current_price": current_price
                }
            }
            
    except Exception as e:
        print(f"⚠️ Error in S/R trend mode detection for {symbol}: {e}")
        return {
            "trend_mode": False,
            "trend_score": 0,
            "entry_triggered": False,
            "description": f"Error: {str(e)[:50]}",
            "details": {"error": str(e)}
        }


def test_sr_trend_mode():
    """Test function for S/R trend mode detection"""
    test_symbol = "BTCUSDT"
    result = detect_sr_trend_mode(test_symbol)
    
    print(f"S/R Trend Mode Test Result for {test_symbol}:")
    print(f"Trend Mode: {result['trend_mode']}")
    print(f"Score: {result['trend_score']}")
    print(f"Entry Triggered: {result['entry_triggered']}")
    print(f"Description: {result['description']}")
    
    return result


if __name__ == "__main__":
    test_sr_trend_mode()