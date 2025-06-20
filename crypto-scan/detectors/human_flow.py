#!/usr/bin/env python3
"""
Human-Like Flow Detector - 9th Layer Flow Analysis
Wykrywa sekwencję: cofka → impuls → pauza → kontynuacja (human pattern)
Psychologiczny podpis decyzyjny tradingu ludzkiego vs botowego
"""

from typing import List, Tuple, Dict

def detect_human_like_flow(prices: List[float]) -> Tuple[bool, str, Dict]:
    """
    Wykrywa sekwencję: cofka → impuls → pauza → kontynuacja (human pattern).
    
    Args:
        prices: Lista cen z ostatnich 90 minut (np. 18 punktów po 5m)
        
    Returns:
        Tuple: (detected, description, details)
    """
    if not prices or len(prices) < 10:
        return False, "Za mało danych dla analizy human flow (wymagane min 10 punktów)", {
            "pattern_count": 0,
            "best_pattern": None,
            "flow_quality": "insufficient_data"
        }
    
    patterns_found = []
    best_pattern = None
    max_strength = 0
    
    # Szukaj wzorców human-like flow w całej sekwencji
    for i in range(0, len(prices) - 7):
        try:
            # Oblicz 4 nogi wzorca psychologicznego
            leg0 = (prices[i+1] - prices[i]) / prices[i]     # cofka (zwątpienie)
            leg1 = (prices[i+3] - prices[i+1]) / prices[i+1]  # impuls (decyzja)
            leg2 = abs((prices[i+5] - prices[i+3]) / prices[i+3])  # pauza (zawahanie)
            leg3 = (prices[i+7] - prices[i+5]) / prices[i+5]  # kontynuacja (commitment)
            
            # Debug output for first pattern
            if i == 0:
                print(f"Debug - Pattern at position {i}:")
                print(f"  leg0 (cofka): {leg0:.6f} (need < -0.002)")
                print(f"  leg1 (impuls): {leg1:.6f} (need > 0.004)")  
                print(f"  leg2 (pauza): {leg2:.6f} (need < 0.0015)")
                print(f"  leg3 (kontynuacja): {leg3:.6f} (need > 0.004)")
            
            # Warunki wzorca psychologicznego
            cofka_valid = leg0 < -0.002        # cofka min -0.2%
            impuls_valid = leg1 > 0.004        # impuls min +0.4%
            pauza_valid = leg2 < 0.0015        # pauza max ±0.15%
            kontynuacja_valid = leg3 > 0.004   # kontynuacja min +0.4%
            
            if cofka_valid and impuls_valid and pauza_valid and kontynuacja_valid:
                # Oblicz siłę wzorca
                pattern_strength = abs(leg0) + leg1 + leg3 - leg2  # Im wyższa, tym lepiej
                
                pattern_data = {
                    "position": i,
                    "cofka_pct": leg0 * 100,
                    "impuls_pct": leg1 * 100,
                    "pauza_pct": leg2 * 100,
                    "kontynuacja_pct": leg3 * 100,
                    "pattern_strength": pattern_strength,
                    "sequence_clarity": (leg1 + leg3) / (abs(leg0) + leg2 + 0.001)  # Clarity ratio
                }
                
                patterns_found.append(pattern_data)
                
                if pattern_strength > max_strength:
                    max_strength = pattern_strength
                    best_pattern = pattern_data
        
        except (ZeroDivisionError, IndexError):
            continue
    
    # Analiza wyników
    pattern_count = len(patterns_found)
    detected = pattern_count > 0
    
    if detected:
        # Klasyfikacja jakości wzorca
        if pattern_count >= 3:
            flow_quality = "multiple_patterns"
            description = f"Silny human flow - {pattern_count} wzorców psychologicznych (najlepszy: {max_strength:.4f})"
        elif pattern_count == 2:
            flow_quality = "strong_pattern"
            description = f"Wyraźny human flow - {pattern_count} wzorce psychologiczne (siła: {max_strength:.4f})"
        else:
            flow_quality = "single_pattern"
            description = f"Human flow wykryty - 1 wzorzec psychologiczny (siła: {max_strength:.4f})"
    else:
        flow_quality = "no_pattern"
        description = "Brak human flow - sekwencja nie przypomina ludzkiego wzorca decyzyjnego"
    
    details = {
        "pattern_count": pattern_count,
        "best_pattern": best_pattern,
        "all_patterns": patterns_found,
        "flow_quality": flow_quality,
        "max_strength": max_strength,
        "analysis_window": len(prices)
    }
    
    return detected, description, details

def calculate_human_flow_score(flow_result: Tuple[bool, str, Dict]) -> int:
    """
    Oblicza score dla human flow (0-15 punktów)
    
    Args:
        flow_result: Wynik z detect_human_like_flow
        
    Returns:
        Score 0-15 punktów
    """
    detected, description, details = flow_result
    
    if not detected:
        return 0
    
    pattern_count = details.get("pattern_count", 0)
    max_strength = details.get("max_strength", 0)
    flow_quality = details.get("flow_quality", "no_pattern")
    
    # Bazowy score na podstawie liczby wzorców
    if pattern_count >= 3:
        base_score = 15  # Maksymalny score
    elif pattern_count == 2:
        base_score = 12  # Silny score
    elif pattern_count == 1:
        base_score = 10  # Podstawowy score
    else:
        return 0
    
    # Bonus za siłę wzorca
    if max_strength > 0.02:  # Bardzo silny wzorzec
        strength_bonus = 3
    elif max_strength > 0.015:  # Silny wzorzec
        strength_bonus = 2
    elif max_strength > 0.01:  # Średni wzorzec
        strength_bonus = 1
    else:
        strength_bonus = 0
    
    # Sprawdź clarity najlepszego wzorca
    best_pattern = details.get("best_pattern", {})
    if best_pattern and best_pattern.get("sequence_clarity", 0) > 3.0:
        clarity_bonus = 1
    else:
        clarity_bonus = 0
    
    final_score = min(base_score + strength_bonus + clarity_bonus, 15)
    return final_score

def analyze_human_flow_detailed(prices: List[float]) -> Dict:
    """
    Szczegółowa analiza human flow z dodatkowymi metrykami
    
    Args:
        prices: Lista cen do analizy
        
    Returns:
        Szczegółowa analiza human flow patterns
    """
    basic_result = detect_human_like_flow(prices)
    basic_score = calculate_human_flow_score(basic_result)
    
    detected, description, details = basic_result
    
    analysis = {
        "basic_analysis": {
            "detected": detected,
            "score": basic_score,
            "description": description
        }
    }
    
    if detected and details.get("all_patterns"):
        patterns = details["all_patterns"]
        
        # Zaawansowane metryki
        avg_pattern_strength = sum(p["pattern_strength"] for p in patterns) / len(patterns)
        avg_clarity = sum(p["sequence_clarity"] for p in patterns) / len(patterns)
        
        # Analiza konsystencji wzorców
        cofka_values = [p["cofka_pct"] for p in patterns]
        impuls_values = [p["impuls_pct"] for p in patterns]
        kontynuacja_values = [p["kontynuacja_pct"] for p in patterns]
        
        cofka_consistency = calculate_consistency(cofka_values)
        impuls_consistency = calculate_consistency(impuls_values)
        kontynuacja_consistency = calculate_consistency(kontynuacja_values)
        
        analysis["advanced_metrics"] = {
            "avg_pattern_strength": avg_pattern_strength,
            "avg_sequence_clarity": avg_clarity,
            "cofka_consistency": cofka_consistency,
            "impuls_consistency": impuls_consistency,
            "kontynuacja_consistency": kontynuacja_consistency,
            "pattern_frequency": len(patterns) / (len(prices) - 7) if len(prices) > 7 else 0
        }
        
        # Interpretacja psychologiczna
        if avg_pattern_strength > 0.02:
            psychology_strength = "very_strong"
        elif avg_pattern_strength > 0.015:
            psychology_strength = "strong"
        elif avg_pattern_strength > 0.01:
            psychology_strength = "moderate"
        else:
            psychology_strength = "weak"
        
        analysis["interpretation"] = {
            "psychology_strength": psychology_strength,
            "decision_clarity": "high" if avg_clarity > 3.0 else "moderate" if avg_clarity > 2.0 else "low",
            "human_confidence": "confirmed" if len(patterns) >= 2 and avg_pattern_strength > 0.015 else "probable" if len(patterns) >= 1 else "uncertain"
        }
    
    return analysis

def calculate_consistency(values: List[float]) -> float:
    """Oblicza konsystencję wartości (niższa odchylka = wyższa konsystencja)"""
    if len(values) < 2:
        return 1.0
    
    avg = sum(values) / len(values)
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5
    
    # Normalizuj do 0-1 (1 = bardzo konsystentne)
    consistency = max(0, 1 - (std_dev / (abs(avg) + 0.001)))
    return consistency

def create_mock_human_flow_prices():
    """Tworzy mock ceny z wyraźnym human flow dla testów"""
    base_price = 50000.0
    prices = [base_price]
    current_price = base_price
    
    # Wzorzec 1: cofka → impuls → pauza → kontynuacja (pozycja 0-7)
    # Pozycja 0 (start)
    prices.append(current_price)
    
    # Pozycja 1: Cofka (zwątpienie) - wymagane < -0.2%
    current_price *= (1 - 0.0025)  # -0.25%
    prices.append(current_price)
    
    # Pozycja 2: Kontynuacja cofki
    current_price *= (1 - 0.001)   # Dodatkowa mała cofka
    prices.append(current_price)
    
    # Pozycja 3: Impuls (decyzja) - wymagane > +0.4%
    current_price *= (1 + 0.005)   # +0.5%
    prices.append(current_price)
    
    # Pozycja 4: Kontynuacja impulsu
    current_price *= (1 + 0.002)   # Dodatkowy wzrost
    prices.append(current_price)
    
    # Pozycja 5: Pauza (zawahanie) - wymagane < ±0.15%
    current_price *= (1 + 0.0008)  # +0.08% (bardzo mała zmiana)
    prices.append(current_price)
    
    # Pozycja 6: Kontynuacja pauzy
    current_price *= (1 - 0.0005)  # -0.05% (minimalna oscylacja)
    prices.append(current_price)
    
    # Pozycja 7: Kontynuacja (commitment) - wymagane > +0.4%
    current_price *= (1 + 0.0045)  # +0.45%
    prices.append(current_price)
    
    # Wzorzec 2: Drugi human flow pattern (pozycja 8-15)
    # Pozycja 8-9: Druga cofka
    current_price *= (1 - 0.003)   # -0.3%
    prices.append(current_price)
    current_price *= (1 - 0.0008)  # Dodatkowa cofka
    prices.append(current_price)
    
    # Pozycja 10-11: Drugi impuls  
    current_price *= (1 + 0.0048)  # +0.48%
    prices.append(current_price)
    current_price *= (1 + 0.0015)  # Kontynuacja
    prices.append(current_price)
    
    # Pozycja 12-13: Druga pauza
    current_price *= (1 + 0.0012)  # +0.12% (w limicie)
    prices.append(current_price)
    current_price *= (1 - 0.0007)  # Mała oscylacja
    prices.append(current_price)
    
    # Pozycja 14-15: Druga kontynuacja
    current_price *= (1 + 0.0042)  # +0.42%
    prices.append(current_price)
    current_price *= (1 + 0.001)   # Potwierdzenie
    prices.append(current_price)
    
    # Dodaj 2 więcej punktów do 18 total
    for i in range(2):
        current_price *= (1 + (-0.001 + i * 0.0015))  # Lekka oscylacja
        prices.append(current_price)
    
    return prices

def create_mock_bot_flow_prices():
    """Tworzy mock ceny z chaotycznym bot flow (brak wzorca ludzkiego)"""
    import random
    base_price = 50000.0
    prices = [base_price]
    current_price = base_price
    
    # Chaotyczne ruchy bez wzorca psychologicznego
    random.seed(42)  # Dla powtarzalności testów
    for i in range(17):
        change_pct = random.uniform(-0.008, 0.008)  # ±0.8%
        current_price += current_price * change_pct
        prices.append(current_price)
    
    return prices

def main():
    """Test funkcji human flow detection"""
    print("🧪 Testing Human-Like Flow Detector\n")
    
    # Test 1: Human flow
    human_prices = create_mock_human_flow_prices()
    detected, desc, details = detect_human_like_flow(human_prices)
    score = calculate_human_flow_score((detected, desc, details))
    
    print("🧠 Human Flow Test:")
    print(f"   Detected: {detected}")
    print(f"   Description: {desc}")
    print(f"   Pattern count: {details.get('pattern_count', 0)}")
    print(f"   Max strength: {details.get('max_strength', 0):.4f}")
    print(f"   Flow quality: {details.get('flow_quality', 'unknown')}")
    print(f"   Score: {score}/15\n")
    
    # Test 2: Bot flow
    bot_prices = create_mock_bot_flow_prices()
    detected, desc, details = detect_human_like_flow(bot_prices)
    score = calculate_human_flow_score((detected, desc, details))
    
    print("🤖 Bot Flow Test:")
    print(f"   Detected: {detected}")
    print(f"   Description: {desc}")
    print(f"   Pattern count: {details.get('pattern_count', 0)}")
    print(f"   Score: {score}/15\n")
    
    # Test 3: Detailed analysis
    print("🔍 Detailed Analysis - Human Flow:")
    detailed_analysis = analyze_human_flow_detailed(human_prices)
    
    basic = detailed_analysis['basic_analysis']
    print(f"   Detection: {basic['detected']}")
    print(f"   Score: {basic['score']}/15")
    print(f"   Description: {basic['description']}")
    
    if 'advanced_metrics' in detailed_analysis:
        advanced = detailed_analysis['advanced_metrics']
        print(f"   Avg pattern strength: {advanced['avg_pattern_strength']:.4f}")
        print(f"   Avg sequence clarity: {advanced['avg_sequence_clarity']:.3f}")
        print(f"   Pattern frequency: {advanced['pattern_frequency']:.3f}")
    
    if 'interpretation' in detailed_analysis:
        interpretation = detailed_analysis['interpretation']
        print(f"   Psychology strength: {interpretation['psychology_strength']}")
        print(f"   Decision clarity: {interpretation['decision_clarity']}")
        print(f"   Human confidence: {interpretation['human_confidence']}")
    
    print("\n✅ Human-Like Flow Detector tests completed!")

if __name__ == "__main__":
    main()