import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

def calculate_stage_minus1_combo_score(signals):
    """
    Calculate Stage -1 combo score based on powerful signal combinations
    Returns score for combo-based activation
    """
    score = 0

    if signals.get("whale_activity") and signals.get("volume_spike") and signals.get("compressed"):
        score += 15
        print(f"[COMBO] Whale+Volume+Compressed: +15")

    if signals.get("dex_inflow") and signals.get("compressed") and signals.get("RSI_flatline"):
        score += 15
        print(f"[COMBO] DEX+Compressed+RSI: +15")

    if signals.get("orderbook_anomaly") and signals.get("spoofing") and signals.get("heatmap_exhaustion"):
        score += 12
        print(f"[COMBO] Orderbook+Spoofing+Heatmap: +12")

    if signals.get("volume_spike") and signals.get("compressed") and signals.get("vwap_pinning"):
        score += 12
        print(f"[COMBO] Volume+Compressed+VWAP: +12")

    if signals.get("whale_activity") and signals.get("dex_inflow") and signals.get("compressed"):
        score += 16
        print(f"[COMBO] Whale+DEX+Compressed: +16")

    return score

def check_stage_minus1_activation(signals, stage_2_1_active):
    """
    Check if Stage -1 should activate based on combo score
    Returns True if combo score >= 12
    """
    combo_score = calculate_stage_minus1_combo_score(signals)
    
    print(f"[STAGE -1] Combo score: {combo_score}")
    
    if combo_score >= 12:
        print(f"[STAGE -1] Combo-based activation. Score: {combo_score}")
        return True

    return False

def detect_stage_minus1_compression(symbol, stage_minus2_1_signals):
    """
    Stage -1: Compression Filter (PPWCS 2.6) - Enhanced with Combo-based Activation
    Aktywuje się na podstawie:
    1. Tradycyjnej logiki ≥1 sygnał z Stage -2.1
    2. NOWEJ combo-based logiki dla mocnych kombinacji sygnałów
    
    Args:
        symbol: symbol tokena
        stage_minus2_1_signals: dictionary z sygnałami z Stage -2.1
    
    Returns:
        bool: True jeśli kompresja aktywna
    """
    try:
        print(f"[DEBUG] {symbol} Stage -1 analysis START")
        
        # 1. TRADYCYJNA LOGIKA - policz aktywne sygnały z Stage -2.1
        active_signals = sum([
            stage_minus2_1_signals.get("whale_activity", False),
            stage_minus2_1_signals.get("dex_inflow", False),
            stage_minus2_1_signals.get("orderbook_anomaly", False),
            stage_minus2_1_signals.get("volume_spike", False),
            stage_minus2_1_signals.get("vwap_pinning", False),
            stage_minus2_1_signals.get("spoofing", False),
            stage_minus2_1_signals.get("cluster_slope", False),
            stage_minus2_1_signals.get("heatmap_exhaustion", False),
            stage_minus2_1_signals.get("social_spike", False),
        ])
        
        traditional_active = active_signals >= 1
        print(f"[DEBUG] {symbol} traditional compression - signals: {active_signals}, active: {traditional_active}")
        
        # 2. NOWA COMBO-BASED LOGIKA
        combo_active = check_stage_minus1_activation(stage_minus2_1_signals, active_signals)
        
        # Stage -1 aktywny jeśli którakolwiek metoda się powiedzie
        compression_active = traditional_active or combo_active
        
        activation_reason = []
        if traditional_active:
            activation_reason.append(f"traditional({active_signals} signals)")
        if combo_active:
            activation_reason.append("combo-based")
            
        if compression_active:
            reason = " + ".join(activation_reason)
            logger.info(f"✅ Stage -1 aktywny dla {symbol}: {reason}")
            print(f"[STAGE -1] {symbol} ACTIVATED via {reason}")
        else:
            logger.debug(f"🚫 Stage -1 nieaktywny dla {symbol}: {active_signals} signals, combo failed")
            print(f"[STAGE -1] {symbol} NOT ACTIVATED")
            
        return compression_active
        
    except Exception as e:
        logger.error(f"❌ Błąd w Stage -1 dla {symbol}: {e}")
        return False

def get_compression_quality_score(stage_minus2_1_signals):
    """
    Oblicza jakość kompresji na podstawie kombinacji sygnałów
    
    Returns:
        int: score jakości kompresji (0-15)
    """
    try:
        quality_score = 0
        
        # Bonus za kluczowe kombinacje
        if (stage_minus2_1_signals.get("whale_activity") and 
            stage_minus2_1_signals.get("dex_inflow")):
            quality_score += 5  # Silna akumulacja
            
        if (stage_minus2_1_signals.get("volume_spike") and 
            stage_minus2_1_signals.get("orderbook_anomaly")):
            quality_score += 4  # Manipulacja + presja
            
        if (stage_minus2_1_signals.get("vwap_pinning") and 
            stage_minus2_1_signals.get("heatmap_exhaustion")):
            quality_score += 4  # Kontrola ceny + wyczerpanie podaży
            
        # Dodatkowe punkty za każdy aktywny sygnał
        active_count = sum(stage_minus2_1_signals.values())
        quality_score += min(active_count, 6)  # Maksymalnie 6 punktów
        
        return min(quality_score, 15)  # Cap na 15 punktów
        
    except Exception as e:
        logger.error(f"❌ Błąd w obliczeniu quality score: {e}")
        return 0

def analyze_compression_timing(symbol):
    """
    Analizuje timing kompresji - czy sygnały pojawiły się w odpowiednim czasie
    
    Returns:
        dict: analiza timingu kompresji
    """
    try:
        current_time = datetime.now(timezone.utc)
        
        # W produkcji będzie analizować historię sygnałów
        timing_analysis = {
            "signals_in_hour": True,  # Czy sygnały w tej samej godzinie
            "market_hours": 6 <= current_time.hour <= 18,  # Godziny rynkowe
            "weekend_penalty": current_time.weekday() >= 5,  # Weekend
            "timing_score": 0
        }
        
        # Oblicz timing score
        if timing_analysis["signals_in_hour"]:
            timing_analysis["timing_score"] += 3
            
        if timing_analysis["market_hours"]:
            timing_analysis["timing_score"] += 2
        else:
            timing_analysis["timing_score"] -= 1
            
        if timing_analysis["weekend_penalty"]:
            timing_analysis["timing_score"] -= 2
            
        return timing_analysis
        
    except Exception as e:
        logger.error(f"❌ Błąd w analizie timingu dla {symbol}: {e}")
        return {"timing_score": 0}