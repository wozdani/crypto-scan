"""
Orderbook Sentiment Detection Module
Ocenia nastroje rynkowe przez bid/ask bez klasycznych wskaźników ani twardych filtrów
"""

def detect_orderbook_sentiment(orderbook):
    """
    Ocenia nastroje rynkowe przez bid/ask. Nie używa klasycznych wskaźników ani twardych filtrów.
    
    Zachowanie:
    - Bidy dominują wolumenowo
    - Ask cofają się (spread wąski)
    - Te same bidy pojawiają się ponownie (reloading)
    
    Args:
        orderbook: Dict with 'bids' and 'asks' arrays [[price, volume], ...]
        
    Returns:
        tuple: (bool, str) - (czy_wykryto_pozytywny_sentiment, opis_sytuacji)
    """
    bids = orderbook.get("bids", [])[:10]
    asks = orderbook.get("asks", [])[:10]

    if not bids or not asks:
        return False, "Brak danych orderbook"

    # Analiza przewagi popytu
    total_bid = sum(float(b[1]) for b in bids)
    total_ask = sum(float(a[1]) for a in asks)
    if total_bid < total_ask * 0.8:
        return False, "Brak przewagi popytu"

    # Analiza spreadu - gotowość do transakcji
    ask_prices = [float(a[0]) for a in asks]
    bid_prices = [float(b[0]) for b in bids]
    if ask_prices[0] - bid_prices[0] > bid_prices[0] * 0.003:
        return False, "Spread zbyt szeroki – brak gotowości do transakcji"

    # Analiza re-akumulacji na bidach
    bid_levels = [round(float(b[0]), 4) for b in bids]
    level_counts = {lvl: bid_levels.count(lvl) for lvl in bid_levels}
    reloaded = any(count > 1 for count in level_counts.values())
    if not reloaded:
        return False, "Brak sygnałów re-akumulacji na bidach"

    return True, "Ask cofają się, bidy stabilne, presja popytu aktywna"

def analyze_orderbook_details(orderbook):
    """
    Szczegółowa analiza orderbook dla debugging i logging
    
    Args:
        orderbook: Dict with orderbook data
        
    Returns:
        dict: Szczegółowe dane analityczne
    """
    bids = orderbook.get("bids", [])[:10]
    asks = orderbook.get("asks", [])[:10]
    
    if not bids or not asks:
        return {"error": "Insufficient orderbook data"}
    
    # Analiza wolumenów
    total_bid = sum(float(b[1]) for b in bids)
    total_ask = sum(float(a[1]) for a in asks)
    bid_ask_ratio = total_bid / total_ask if total_ask > 0 else 0
    
    # Analiza spreadu
    best_bid = float(bids[0][0]) if bids else 0
    best_ask = float(asks[0][0]) if asks else 0
    spread = best_ask - best_bid if best_ask and best_bid else 0
    spread_pct = (spread / best_ask * 100) if best_ask else 0
    
    # Analiza poziomów bid
    bid_levels = [round(float(b[0]), 4) for b in bids]
    level_counts = {lvl: bid_levels.count(lvl) for lvl in bid_levels}
    max_reload_count = max(level_counts.values()) if level_counts else 0
    
    return {
        "total_bid_volume": total_bid,
        "total_ask_volume": total_ask,
        "bid_ask_ratio": bid_ask_ratio,
        "spread_usd": spread,
        "spread_percent": spread_pct,
        "bid_levels_analyzed": len(bid_levels),
        "max_reload_count": max_reload_count,
        "bid_dominance": bid_ask_ratio >= 0.8,
        "tight_spread": spread_pct <= 0.3,
        "reloading_detected": max_reload_count > 1
    }

def get_orderbook_sentiment_summary(orderbook):
    """
    Podsumowanie sentymentu orderbook w formacie czytelnym dla człowieka
    
    Args:
        orderbook: Dict with orderbook data
        
    Returns:
        dict: Podsumowanie sentymentu
    """
    detected, description = detect_orderbook_sentiment(orderbook)
    details = analyze_orderbook_details(orderbook)
    
    if "error" in details:
        return {
            "sentiment_detected": False,
            "description": "Brak danych orderbook",
            "confidence": 0,
            "details": details
        }
    
    # Oblicz poziom pewności na podstawie wskaźników
    confidence_factors = []
    
    if details["bid_dominance"]:
        confidence_factors.append(30)  # Dominacja bidów
    
    if details["tight_spread"]:
        confidence_factors.append(25)  # Wąski spread
        
    if details["reloading_detected"]:
        confidence_factors.append(35)  # Re-akumulacja
        
    if float(details["bid_ask_ratio"]) > 1.2:
        confidence_factors.append(10)  # Silna przewaga bidów
    
    confidence = min(sum(confidence_factors), 100)
    
    return {
        "sentiment_detected": detected,
        "description": description,
        "confidence": confidence,
        "key_factors": {
            "bid_dominance": details["bid_dominance"],
            "tight_spread": details["tight_spread"], 
            "reloading_detected": details["reloading_detected"],
            "bid_ask_ratio": round(float(details["bid_ask_ratio"]), 2)
        },
        "details": details
    }