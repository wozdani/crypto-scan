#!/usr/bin/env python3
"""
One-Sided Pressure Detector - 7th Layer Flow Analysis
Wykrywa przewagę kupujących w strukturze orderbooku przez analizę stosunku bid/ask
"""

from typing import Dict, List, Tuple, Optional
import json

def detect_one_sided_pressure(orderbook: Dict) -> Tuple[bool, str, Dict]:
    """
    Wykrywa przewagę kupujących – silny bid, słaby ask
    
    Args:
        orderbook: Dict z kluczami 'bids' i 'asks'
        
    Returns:
        Tuple: (detected, description, details)
    """
    try:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Sprawdź minimalne wymagania
        if len(bids) < 3 or len(asks) < 3:
            return False, "Za mało poziomów orderbooka (wymagane min 3 bid i 3 ask)", {
                "bid_levels": len(bids),
                "ask_levels": len(asks),
                "pressure_ratio": 0.0,
                "bid_depth_score": 0,
                "ask_depth_score": 0,
                "dominance_type": "insufficient_data"
            }
        
        # Analizuj top 5 poziomów
        top_bids = bids[:5]
        top_asks = asks[:5]
        
        # Oblicz całkowity wolumen
        total_bid_vol = sum([float(bid[1]) for bid in top_bids if len(bid) >= 2])
        total_ask_vol = sum([float(ask[1]) for ask in top_asks if len(ask) >= 2])
        
        if total_ask_vol == 0:
            return False, "Brak wolumenu ask - nieprawidłowe dane orderbooka", {
                "bid_levels": len(bids),
                "ask_levels": len(asks),
                "total_bid_vol": total_bid_vol,
                "total_ask_vol": total_ask_vol,
                "pressure_ratio": 0.0,
                "dominance_type": "invalid_data"
            }
        
        # Oblicz stosunek pressure
        pressure_ratio = total_bid_vol / total_ask_vol
        
        # Analizuj głębokość (poziomy z wolumenem > średnia/2)
        avg_bid_vol = total_bid_vol / len(top_bids)
        avg_ask_vol = total_ask_vol / len(top_asks)
        
        significant_bid_levels = sum(1 for bid in top_bids if float(bid[1]) > avg_bid_vol / 2)
        significant_ask_levels = sum(1 for ask in top_asks if float(ask[1]) > avg_ask_vol / 2)
        
        # Oblicz depth scores
        bid_depth_score = significant_bid_levels
        ask_depth_score = significant_ask_levels
        depth_ratio = bid_depth_score / max(ask_depth_score, 1)
        
        # Klasyfikacja dominacji
        dominance_type = "neutral"
        if pressure_ratio >= 2.0:
            dominance_type = "strong_bid_dominance"
        elif pressure_ratio >= 1.5:
            dominance_type = "moderate_bid_dominance"
        elif pressure_ratio <= 0.5:
            dominance_type = "ask_dominance"
        elif pressure_ratio <= 0.67:
            dominance_type = "weak_ask_dominance"
        
        # Detekcja one-sided pressure (przewaga kupujących)
        # Główny warunek: pressure ratio > 1.5x
        # Dodatkowy warunek: depth ratio >= 0.8 (złagodzony próg)
        detected = pressure_ratio > 1.5 and depth_ratio >= 0.8
        
        details = {
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "total_bid_vol": round(total_bid_vol, 4),
            "total_ask_vol": round(total_ask_vol, 4),
            "pressure_ratio": round(pressure_ratio, 3),
            "bid_depth_score": bid_depth_score,
            "ask_depth_score": ask_depth_score,
            "depth_ratio": round(depth_ratio, 3),
            "dominance_type": dominance_type,
            "significant_bid_levels": significant_bid_levels,
            "significant_ask_levels": significant_ask_levels
        }
        
        if detected:
            if pressure_ratio >= 2.0:
                description = f"Silna przewaga kupujących - bid dominuje {pressure_ratio:.1f}x nad ask (głębokość bid: {bid_depth_score}/5)"
            else:
                description = f"Umiarkowana przewaga kupujących - bid przewyższa ask {pressure_ratio:.1f}x (głębokość bid: {bid_depth_score}/5)"
        else:
            if pressure_ratio < 1.0:
                description = f"Przewaga sprzedających - ask dominuje {1/pressure_ratio:.1f}x nad bid"
            else:
                if pressure_ratio > 1.5:
                    description = f"⚖️ Balanced orderbook - pressure ratio {pressure_ratio:.1f}x spełniony, ale niewystarczająca głębokość bid (depth ratio: {depth_ratio:.1f}, wymagane ≥0.8)"
                else:
                    description = f"Brak one-sided pressure - stosunek {pressure_ratio:.1f}x niewystarczający (wymagane >1.5x)"
        
        return detected, description, details
        
    except Exception as e:
        return False, f"Błąd analizy one-sided pressure: {str(e)}", {
            "error": str(e),
            "pressure_ratio": 0.0,
            "dominance_type": "error"
        }

def calculate_one_sided_pressure_score(pressure_result: Tuple[bool, str, Dict]) -> int:
    """
    Oblicza score dla one-sided pressure (0-20 punktów)
    
    Args:
        pressure_result: Wynik z detect_one_sided_pressure
        
    Returns:
        Score 0-20 punktów
    """
    detected, description, details = pressure_result
    
    if not detected:
        return 0
    
    pressure_ratio = details.get("pressure_ratio", 0.0)
    depth_ratio = details.get("depth_ratio", 0.0)
    dominance_type = details.get("dominance_type", "neutral")
    
    base_score = 0
    
    # Score bazowy na podstawie pressure ratio
    if pressure_ratio >= 3.0:
        base_score = 20  # Ekstremalnie silna przewaga
    elif pressure_ratio >= 2.5:
        base_score = 18  # Bardzo silna przewaga
    elif pressure_ratio >= 2.0:
        base_score = 15  # Silna przewaga
    elif pressure_ratio >= 1.5:
        base_score = 12  # Umiarkowana przewaga
    
    # Bonus za głębokość
    if depth_ratio >= 2.0:
        base_score += 2  # Bardzo głębokie bidy
    elif depth_ratio >= 1.5:
        base_score += 1  # Głębokie bidy
    
    # Ograniczenie do maksymalnego score
    return min(base_score, 20)

def analyze_one_sided_pressure_detailed(orderbook: Dict) -> Dict:
    """
    Szczegółowa analiza one-sided pressure z dodatkowymi metrykami
    
    Args:
        orderbook: Dict z danymi orderbooka
        
    Returns:
        Szczegółowa analiza pressure patterns
    """
    detected, description, details = detect_one_sided_pressure(orderbook)
    score = calculate_one_sided_pressure_score((detected, description, details))
    
    # Dodatkowe metryki
    bids = orderbook.get("bids", [])[:5]
    asks = orderbook.get("asks", [])[:5]
    
    # Analiza koncentracji wolumenu
    if bids and asks:
        # Top bid vs top ask
        top_bid_vol = float(bids[0][1]) if bids else 0
        top_ask_vol = float(asks[0][1]) if asks else 0
        top_level_ratio = (top_bid_vol / max(top_ask_vol, 0.001))
        
        # Średni wolumen per poziom
        avg_bid_vol = sum(float(b[1]) for b in bids) / len(bids) if bids else 0
        avg_ask_vol = sum(float(a[1]) for a in asks) / len(asks) if asks else 0
        
        # Współczynnik koncentracji (top level / średnia)
        bid_concentration = (top_bid_vol / max(avg_bid_vol, 0.001)) if avg_bid_vol > 0 else 0
        ask_concentration = (top_ask_vol / max(avg_ask_vol, 0.001)) if avg_ask_vol > 0 else 0
        
        return {
            "basic_analysis": {
                "detected": detected,
                "description": description,
                "score": score,
                "details": details
            },
            "advanced_metrics": {
                "top_level_ratio": round(top_level_ratio, 3),
                "avg_bid_vol": round(avg_bid_vol, 4),
                "avg_ask_vol": round(avg_ask_vol, 4),
                "bid_concentration": round(bid_concentration, 3),
                "ask_concentration": round(ask_concentration, 3),
                "concentration_advantage": "bid" if bid_concentration > ask_concentration * 1.2 else "ask" if ask_concentration > bid_concentration * 1.2 else "balanced"
            },
            "interpretation": {
                "pressure_strength": "strong" if score >= 15 else "moderate" if score >= 10 else "weak" if score > 0 else "none",
                "market_sentiment": "bullish_pressure" if detected else "neutral_or_bearish",
                "continuation_probability": "high" if score >= 15 and details.get("depth_ratio", 0) >= 1.5 else "moderate" if score >= 10 else "low"
            }
        }
    
    return {
        "basic_analysis": {
            "detected": detected,
            "description": description,
            "score": score,
            "details": details
        },
        "error": "Insufficient orderbook data for advanced analysis"
    }

def create_mock_strong_bid_orderbook():
    """Tworzy mock orderbook z silną przewagą bid dla testów"""
    return {
        "bids": [
            [50000.0, 2.5],  # Silny bid
            [49995.0, 2.0],
            [49990.0, 1.8],
            [49985.0, 1.5],
            [49980.0, 1.2]
        ],
        "asks": [
            [50005.0, 0.8],  # Słaby ask
            [50010.0, 0.6],
            [50015.0, 0.5],
            [50020.0, 0.4],
            [50025.0, 0.3]
        ]
    }

def create_mock_balanced_orderbook():
    """Tworzy mock orderbook ze zrównoważonymi poziomami dla testów"""
    return {
        "bids": [
            [50000.0, 1.2],
            [49995.0, 1.1],
            [49990.0, 1.0],
            [49985.0, 0.9],
            [49980.0, 0.8]
        ],
        "asks": [
            [50005.0, 1.1],
            [50010.0, 1.0],
            [50015.0, 0.9],
            [50020.0, 0.8],
            [50025.0, 0.7]
        ]
    }

def create_mock_ask_dominant_orderbook():
    """Tworzy mock orderbook z przewagą ask dla testów"""
    return {
        "bids": [
            [50000.0, 0.5],  # Słaby bid
            [49995.0, 0.4],
            [49990.0, 0.3],
            [49985.0, 0.3],
            [49980.0, 0.2]
        ],
        "asks": [
            [50005.0, 2.0],  # Silny ask
            [50010.0, 1.8],
            [50015.0, 1.5],
            [50020.0, 1.2],
            [50025.0, 1.0]
        ]
    }

def main():
    """Test funkcji one-sided pressure detection"""
    print("🧪 Testing One-Sided Pressure Detector\n")
    
    # Test 1: Silna przewaga bid
    strong_bid_book = create_mock_strong_bid_orderbook()
    detected1, desc1, details1 = detect_one_sided_pressure(strong_bid_book)
    score1 = calculate_one_sided_pressure_score((detected1, desc1, details1))
    
    print(f"💪 Strong Bid Dominance Test:")
    print(f"   Detected: {detected1}")
    print(f"   Description: {desc1}")
    print(f"   Pressure ratio: {details1['pressure_ratio']}")
    print(f"   Bid depth: {details1['bid_depth_score']}/5")
    print(f"   Ask depth: {details1['ask_depth_score']}/5")
    print(f"   Score: {score1}/20")
    
    # Test 2: Zrównoważony orderbook
    balanced_book = create_mock_balanced_orderbook()
    detected2, desc2, details2 = detect_one_sided_pressure(balanced_book)
    score2 = calculate_one_sided_pressure_score((detected2, desc2, details2))
    
    print(f"\n⚖️ Balanced Orderbook Test:")
    print(f"   Detected: {detected2}")
    print(f"   Description: {desc2}")
    print(f"   Pressure ratio: {details2['pressure_ratio']}")
    print(f"   Score: {score2}/20")
    
    # Test 3: Przewaga ask
    ask_dominant_book = create_mock_ask_dominant_orderbook()
    detected3, desc3, details3 = detect_one_sided_pressure(ask_dominant_book)
    score3 = calculate_one_sided_pressure_score((detected3, desc3, details3))
    
    print(f"\n📉 Ask Dominance Test:")
    print(f"   Detected: {detected3}")
    print(f"   Description: {desc3}")
    print(f"   Pressure ratio: {details3['pressure_ratio']}")
    print(f"   Score: {score3}/20")
    
    # Test 4: Szczegółowa analiza
    print(f"\n🔍 Detailed Analysis - Strong Bid:")
    detailed_analysis = analyze_one_sided_pressure_detailed(strong_bid_book)
    print(f"   Pressure strength: {detailed_analysis['interpretation']['pressure_strength']}")
    print(f"   Market sentiment: {detailed_analysis['interpretation']['market_sentiment']}")
    print(f"   Continuation probability: {detailed_analysis['interpretation']['continuation_probability']}")
    if 'advanced_metrics' in detailed_analysis:
        adv_metrics = detailed_analysis['advanced_metrics']
        print(f"   Top level ratio: {adv_metrics['top_level_ratio']}")
        print(f"   Concentration advantage: {adv_metrics['concentration_advantage']}")
    
    print("\n✅ One-Sided Pressure Detector tests completed!")

if __name__ == "__main__":
    main()