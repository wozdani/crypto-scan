import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

def detect_stage_minus1_compression(symbol, stage_minus2_1_signals):
    """
    Stage -1: Compression Filter (PPWCS 2.6)
    Aktywuje się gdy ≥2 sygnały z Stage -2.1 pojawiły się w tej samej godzinie
    
    Args:
        symbol: symbol tokena
        stage_minus2_1_signals: dictionary z sygnałami z Stage -2.1
    
    Returns:
        bool: True jeśli kompresja aktywna
    """
    try:
        # Policz aktywne sygnały z Stage -2.1
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
        
        # ZŁAGODZONE WARUNKI: kompresja aktywna gdy ≥1 sygnał (zamiast ≥2)
        # lub specjalne kombinacje wysokiej wartości
        compression_active = active_signals >= 1
        
        # Bonus dla silnych kombinacji
        strong_combo = (
            (stage_minus2_1_signals.get("whale_activity", False) and stage_minus2_1_signals.get("dex_inflow", False)) or
            (stage_minus2_1_signals.get("volume_spike", False) and stage_minus2_1_signals.get("orderbook_anomaly", False))
        )
        
        print(f"[DEBUG] {symbol} compression - signals: {active_signals}, strong_combo: {strong_combo}")
        
        if compression_active:
            logger.info(f"✅ Stage -1 kompresja aktywna dla {symbol}: {active_signals} sygnałów")
        else:
            logger.debug(f"🚫 Stage -1 kompresja nieaktywna dla {symbol}: tylko {active_signals} sygnałów")
            
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