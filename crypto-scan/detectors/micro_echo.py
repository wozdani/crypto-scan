#!/usr/bin/env python3
"""
Micro-Timeframe Echo Detector - 8th Layer Flow Analysis
Wykrywa powtarzające się mikroimpulsy wzrostowe na interwale 1m w ciągu 30-45 minut
"""

from typing import List, Tuple, Dict, Optional
import requests
import json
import time

def detect_micro_echo(prices_1m: List[float]) -> Tuple[bool, str, Dict]:
    """
    Wykrywa 3+ mikroimpulsy wzrostowe na interwale 1m w ciągu ostatnich 30–45 min
    
    Args:
        prices_1m: Lista cen z interwału 1-minutowego
        
    Returns:
        Tuple: (detected, description, details)
    """
    try:
        if len(prices_1m) < 20:
            return False, "Za mało danych 1m dla analizy micro echo (wymagane min 20, otrzymano {})".format(len(prices_1m)), {
                "prices_count": len(prices_1m),
                "impulse_count": 0,
                "impulse_sequences": [],
                "echo_strength": "insufficient_data"
            }
        
        # Wykryj mikroimpulsy (3+ świec wzrostowych z rzędu)
        impulse_count = 0
        impulse_sequences = []
        total_impulse_strength = 0
        
        i = 0
        while i < len(prices_1m) - 3:
            p0, p1, p2, p3 = prices_1m[i:i+4]
            
            # Sprawdź czy mamy 3 świece wzrostowe z rzędu
            if p1 > p0 and p2 > p1 and p3 > p2:
                # Oblicz siłę impulsu
                impulse_strength = ((p3 - p0) / p0) * 100  # % wzrost
                
                # Sprawdź czy to znaczący impuls (min 0.1% wzrost)
                if impulse_strength >= 0.1:
                    impulse_count += 1
                    total_impulse_strength += impulse_strength
                    
                    impulse_sequences.append({
                        "start_index": i,
                        "end_index": i + 3,
                        "start_price": p0,
                        "end_price": p3,
                        "strength_pct": round(impulse_strength, 4),
                        "duration_minutes": 3
                    })
                    
                    i += 4  # Przeskocz ten impuls aby uniknąć nakładania
                else:
                    i += 1
            else:
                i += 1
        
        # Oblicz metryki echo
        avg_impulse_strength = total_impulse_strength / max(impulse_count, 1)
        echo_frequency = impulse_count / (len(prices_1m) / 10)  # impulsy na 10-minutowy segment
        
        # Klasyfikacja siły echo
        echo_strength = "none"
        if impulse_count >= 5:
            echo_strength = "very_strong"
        elif impulse_count >= 4:
            echo_strength = "strong"
        elif impulse_count >= 3:
            echo_strength = "moderate"
        elif impulse_count >= 2:
            echo_strength = "weak"
        
        # Detekcja micro echo (≥3 mikroimpulsy)
        detected = impulse_count >= 3
        
        details = {
            "prices_count": len(prices_1m),
            "impulse_count": impulse_count,
            "avg_impulse_strength": round(avg_impulse_strength, 4),
            "total_impulse_strength": round(total_impulse_strength, 4),
            "echo_frequency": round(echo_frequency, 3),
            "echo_strength": echo_strength,
            "impulse_sequences": impulse_sequences,
            "analysis_timeframe": f"{len(prices_1m)} minutes",
            "detection_threshold": 3
        }
        
        if detected:
            if impulse_count >= 5:
                description = f"Bardzo silne micro echo - {impulse_count} mikroimpulsów wzrostowych (średnia siła: {avg_impulse_strength:.2f}%)"
            elif impulse_count >= 4:
                description = f"Silne micro echo - {impulse_count} mikroimpulsy wzrostowe (średnia siła: {avg_impulse_strength:.2f}%)"
            else:
                description = f"Umiarkowane micro echo - {impulse_count} mikroimpulsy wzrostowe (średnia siła: {avg_impulse_strength:.2f}%)"
        else:
            if impulse_count == 2:
                description = f"Słabe micro echo - tylko {impulse_count} mikroimpulsy (wymagane ≥3)"
            elif impulse_count == 1:
                description = f"Pojedynczy mikroimpuls - brak trendu fraktalnego"
            else:
                description = f"Brak micro echo - brak mikroimpulsów wzrostowych"
        
        return detected, description, details
        
    except Exception as e:
        return False, f"Błąd analizy micro echo: {str(e)}", {
            "error": str(e),
            "impulse_count": 0,
            "echo_strength": "error"
        }

def calculate_micro_echo_score(echo_result: Tuple[bool, str, Dict]) -> int:
    """
    Oblicza score dla micro echo (0-10 punktów)
    
    Args:
        echo_result: Wynik z detect_micro_echo
        
    Returns:
        Score 0-10 punktów
    """
    detected, description, details = echo_result
    
    if not detected:
        return 0
    
    impulse_count = details.get("impulse_count", 0)
    avg_impulse_strength = details.get("avg_impulse_strength", 0)
    echo_frequency = details.get("echo_frequency", 0)
    
    base_score = 0
    
    # Score bazowy na podstawie liczby impulse'ów
    if impulse_count >= 6:
        base_score = 10  # Wyjątkowo silne echo
    elif impulse_count >= 5:
        base_score = 9   # Bardzo silne echo
    elif impulse_count >= 4:
        base_score = 7   # Silne echo
    elif impulse_count >= 3:
        base_score = 5   # Umiarkowane echo
    
    # Bonus za siłę impulse'ów
    if avg_impulse_strength >= 0.5:
        base_score += 1  # Silne mikroimpulsy
    elif avg_impulse_strength >= 0.3:
        base_score = min(base_score + 1, 10)  # Umiarkowane mikroimpulsy
    
    # Bonus za częstotliwość
    if echo_frequency >= 0.8:
        base_score = min(base_score + 1, 10)  # Wysoka częstotliwość
    
    return min(base_score, 10)

def analyze_micro_echo_detailed(prices_1m: List[float]) -> Dict:
    """
    Szczegółowa analiza micro echo z dodatkowymi metrykami
    
    Args:
        prices_1m: Lista cen 1-minutowych
        
    Returns:
        Szczegółowa analiza micro echo patterns
    """
    detected, description, details = detect_micro_echo(prices_1m)
    score = calculate_micro_echo_score((detected, description, details))
    
    # Dodatkowe analizy
    if len(prices_1m) >= 20:
        # Analiza momentum
        recent_momentum = calculate_momentum_trend(prices_1m[-15:])  # ostatnie 15 minut
        overall_momentum = calculate_momentum_trend(prices_1m)
        
        # Analiza volatility
        price_volatility = calculate_price_volatility(prices_1m)
        
        # Analiza consistency
        impulse_consistency = analyze_impulse_consistency(details.get("impulse_sequences", []))
        
        return {
            "basic_analysis": {
                "detected": detected,
                "description": description,
                "score": score,
                "details": details
            },
            "advanced_metrics": {
                "recent_momentum": round(recent_momentum, 4),
                "overall_momentum": round(overall_momentum, 4),
                "price_volatility": round(price_volatility, 4),
                "impulse_consistency": impulse_consistency,
                "trend_direction": "bullish" if overall_momentum > 0 else "bearish" if overall_momentum < 0 else "sideways"
            },
            "interpretation": {
                "echo_quality": "high" if score >= 8 else "medium" if score >= 5 else "low" if score > 0 else "none",
                "fractal_strength": "strong" if details.get("impulse_count", 0) >= 4 else "moderate" if details.get("impulse_count", 0) >= 3 else "weak",
                "trend_confirmation": "confirmed" if detected and recent_momentum > 0 else "mixed" if detected else "unconfirmed"
            }
        }
    
    return {
        "basic_analysis": {
            "detected": detected,
            "description": description,
            "score": score,
            "details": details
        },
        "error": "Insufficient data for advanced analysis"
    }

def calculate_momentum_trend(prices: List[float]) -> float:
    """Oblicza momentum trendu (% zmiana first -> last)"""
    if len(prices) < 2:
        return 0.0
    return ((prices[-1] - prices[0]) / prices[0]) * 100

def calculate_price_volatility(prices: List[float]) -> float:
    """Oblicza zmienność cen (standard deviation)"""
    if len(prices) < 2:
        return 0.0
    
    mean_price = sum(prices) / len(prices)
    variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
    return variance ** 0.5

def analyze_impulse_consistency(impulse_sequences: List[Dict]) -> float:
    """Analizuje konsystencję impulse'ów (podobieństwo siły)"""
    if len(impulse_sequences) < 2:
        return 0.0
    
    strengths = [seq["strength_pct"] for seq in impulse_sequences]
    avg_strength = sum(strengths) / len(strengths)
    
    # Oblicz coefficient of variation (lower = more consistent)
    if avg_strength == 0:
        return 0.0
    
    variance = sum((strength - avg_strength) ** 2 for strength in strengths) / len(strengths)
    std_dev = variance ** 0.5
    cv = std_dev / avg_strength
    
    # Konwersja na consistency score (1.0 = perfect consistency, 0.0 = high variation)
    return max(0.0, 1.0 - cv)

def fetch_1m_prices_bybit(symbol: str, limit: int = 45) -> List[float]:
    """
    Pobiera ceny 1-minutowe z Bybit API
    
    Args:
        symbol: Symbol tradingowy (np. 'BTCUSDT')
        limit: Liczba świec do pobrania (domyślnie 45 minut)
        
    Returns:
        Lista cen zamknięcia z interwału 1m
    """
    try:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "1",
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("retCode") == 0 and "result" in data and "list" in data["result"]:
                klines = data["result"]["list"]
                
                # Extracting close prices (index 4 in kline data)
                prices = [float(kline[4]) for kline in reversed(klines)]  # reverse to get chronological order
                return prices
        
        return []
        
    except Exception as e:
        print(f"⚠️ Error fetching 1m prices for {symbol}: {e}")
        return []

def create_mock_bullish_1m_prices():
    """Tworzy mock ceny 1m z bullish micro echo dla testów"""
    base_price = 50000.0
    prices = [base_price]  # Start price
    
    # Create explicit 3-candle bullish impulses
    current_price = base_price
    
    # Impulse 1: Minutes 1-4 (3 bullish + 1 pullback)
    current_price += 25  # 50025
    prices.append(current_price)
    current_price += 30  # 50055 
    prices.append(current_price)
    current_price += 20  # 50075
    prices.append(current_price)
    current_price -= 10  # Small pullback to 50065
    prices.append(current_price)
    
    # Sideways/consolidation: Minutes 5-8
    for _ in range(4):
        current_price += (-5 + (len(prices) % 3) * 5)  # Small oscillation
        prices.append(current_price)
    
    # Impulse 2: Minutes 9-12 (3 bullish + 1 pullback)
    current_price += 28  # Strong move up
    prices.append(current_price)
    current_price += 35  
    prices.append(current_price)
    current_price += 22  
    prices.append(current_price)
    current_price -= 12  # Small pullback
    prices.append(current_price)
    
    # Sideways: Minutes 13-16
    for _ in range(4):
        current_price += (-3 + (len(prices) % 2) * 6)
        prices.append(current_price)
    
    # Impulse 3: Minutes 17-20 (3 bullish + 1 pullback)
    current_price += 32
    prices.append(current_price)
    current_price += 26
    prices.append(current_price)
    current_price += 29
    prices.append(current_price)
    current_price -= 8  # Small pullback
    prices.append(current_price)
    
    # More consolidation and another impulse
    # Sideways: Minutes 21-24
    for _ in range(4):
        current_price += (-4 + (len(prices) % 3) * 4)
        prices.append(current_price)
    
    # Impulse 4: Minutes 25-28 (3 bullish + 1 pullback)
    current_price += 24
    prices.append(current_price)
    current_price += 31
    prices.append(current_price)
    current_price += 18
    prices.append(current_price)
    current_price -= 6  # Small pullback
    prices.append(current_price)
    
    # Fill remaining minutes with slight upward bias
    while len(prices) < 45:
        current_price += (-2 + (len(prices) % 4) * 2)
        prices.append(current_price)
    
    return prices[:45]  # Ensure exactly 45 prices

def create_mock_sideways_1m_prices():
    """Tworzy mock ceny 1m z sideways movement (brak micro echo)"""
    base_price = 50000.0
    prices = []
    
    for i in range(45):
        # Sideways z małymi oscylacjami
        oscillation = 5 * (0.5 - (i % 10) / 10)  # -2.5 do +2.5
        noise = 2 * (0.5 - (i % 3) / 3)  # dodatkowy szum
        price = base_price + oscillation + noise
        prices.append(price)
    
    return prices

def create_mock_mixed_1m_prices():
    """Tworzy mock ceny 1m z mixed signals (2 mikroimpulsy - poniżej progu)"""
    base_price = 50000.0
    prices = []
    
    for i in range(45):
        if i < 20:
            # Sideways
            price = base_price + (i % 5) - 2
        elif i < 35:
            # Jeden mikroimpuls
            if (i - 20) % 4 < 3:
                price = base_price + (i - 20) * 1.5 + ((i - 20) % 4) * 1.2
            else:
                price = base_price + (i - 20) * 1.5 - 0.8
        else:
            # Drugi mikroimpuls (słabszy)
            base_segment = base_price + 18
            if (i - 35) % 4 < 3:
                price = base_segment + (i - 35) * 1 + ((i - 35) % 4) * 0.8
            else:
                price = base_segment + (i - 35) * 1 - 0.5
        
        prices.append(price)
    
    return prices

def main():
    """Test funkcji micro echo detection"""
    print("🧪 Testing Micro-Timeframe Echo Detector\n")
    
    # Test 1: Bullish micro echo
    bullish_prices = create_mock_bullish_1m_prices()
    detected1, desc1, details1 = detect_micro_echo(bullish_prices)
    score1 = calculate_micro_echo_score((detected1, desc1, details1))
    
    print(f"📈 Bullish Micro Echo Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Impulse count: {details1['impulse_count']}")
    print(f"   Avg impulse strength: {details1['avg_impulse_strength']:.3f}%")
    print(f"   Echo frequency: {details1['echo_frequency']:.3f}")
    print(f"   Echo strength: {details1['echo_strength']}")
    print(f"   Score: {score1}/10")
    
    # Test 2: Sideways movement
    sideways_prices = create_mock_sideways_1m_prices()
    detected2, desc2, details2 = detect_micro_echo(sideways_prices)
    score2 = calculate_micro_echo_score((detected2, desc2, details2))
    
    print(f"\n↔️ Sideways Movement Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Impulse count: {details2['impulse_count']}")
    print(f"   Score: {score2}/10")
    
    # Test 3: Mixed signals
    mixed_prices = create_mock_mixed_1m_prices()
    detected3, desc3, details3 = detect_micro_echo(mixed_prices)
    score3 = calculate_micro_echo_score((detected3, desc3, details3))
    
    print(f"\n🔀 Mixed Signals Test:")
    print(f"   Detected: {detected3}")
    print(f"   Description: {desc3}")
    print(f"   Impulse count: {details3['impulse_count']}")
    print(f"   Score: {score3}/10")
    
    # Test 4: Szczegółowa analiza
    print(f"\n🔍 Detailed Analysis - Bullish Echo:")
    detailed_analysis = analyze_micro_echo_detailed(bullish_prices)
    print(f"   Echo quality: {detailed_analysis['interpretation']['echo_quality']}")
    print(f"   Fractal strength: {detailed_analysis['interpretation']['fractal_strength']}")
    print(f"   Trend confirmation: {detailed_analysis['interpretation']['trend_confirmation']}")
    if 'advanced_metrics' in detailed_analysis:
        adv_metrics = detailed_analysis['advanced_metrics']
        print(f"   Overall momentum: {adv_metrics['overall_momentum']:.3f}%")
        print(f"   Trend direction: {adv_metrics['trend_direction']}")
    
    print("\n✅ Micro-Timeframe Echo Detector tests completed!")

if __name__ == "__main__":
    main()