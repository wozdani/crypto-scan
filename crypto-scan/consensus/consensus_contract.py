"""
Consensus Contract - Silent Coordination Layer
Wspólne reguły konsensusu dla wszystkich agentów (read-only)
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ConsensusContract:
    """
    Kontrakt konsensusu - wspólne zasady dla wszystkich agentów
    Agenci znają te reguły ale nie widzą wzajemnie swoich głosów
    """
    
    # Przestrzeń głosów (HOLD → NO_OP)
    VOTE_SPACE = ["BUY", "SELL", "NO_OP", "ABSTAIN"]
    
    # Wymagany margin dla decyzji (BUY - SELL ≥ margin)
    REQUIRED_MARGIN = 2
    
    # Minimalna liczba ważnych głosów (po filtrach)
    MIN_VALID_VOTES = 3
    
    # Próg confidence (głosy poniżej są ignorowane)
    CONFIDENCE_FLOOR = 0.62
    
    # Polityka ABSTAIN
    ABSTAIN_POLICY = {
        "no_data": "Użyj ABSTAIN gdy brak danych dla detektora",
        "inactive_detector": "Użyj ABSTAIN gdy detektor nieaktywny dla tokena",
        "stale_data": "Użyj ABSTAIN gdy dane przestarzałe (>5min)"
    }
    
    # Semantyka głosów
    VOTE_SEMANTICS = {
        "BUY": "Edge spełnia progi kontraktu, wysokie prawdopodobieństwo wzrostu",
        "SELL": "Negatywny edge, prawdopodobieństwo spadku",
        "NO_OP": "Dane są, ale edge poniżej progu - rezygnacja z impulsywnego działania",
        "ABSTAIN": "Brak danych lub detektor nieaktywny"
    }
    
    # Tie-break policy
    TIE_BREAK_POLICY = {
        "enabled": True,
        "min_gap_for_escalation": 1,  # Gap < 2 → eskalacja do Debatera
        "debater_weight_multiplier": 1.5  # Debater ma większą wagę przy tie-break
    }
    
    # Wagi domyślne (dopóki nie nauczone)
    DEFAULT_AGENT_WEIGHT = 1.0
    
    # Guard rails dla adaptacji
    ADAPTATION_GUARDS = {
        "max_margin_change_per_update": 0.5,
        "max_confidence_floor_change": 0.05,
        "cooldown_between_updates_hours": 2,
        "min_samples_for_update": 50
    }
    
    @classmethod
    def to_prompt_context(cls) -> str:
        """
        Generuje kontekst dla promptów agentów
        """
        return f"""
KONTRAKT KONSENSUSU (Silent Coordination):
- Przestrzeń głosów: BUY / SELL / NO_OP / ABSTAIN
- Wymagany margin: ≥{cls.REQUIRED_MARGIN} (BUY - SELL)
- Min ważnych głosów: ≥{cls.MIN_VALID_VOTES}
- Confidence floor: ≥{cls.CONFIDENCE_FLOOR}

SEMANTYKA GŁOSÓW:
- BUY: Twój edge spełnia progi, prawdopodobieństwo wzrostu
- SELL: Negatywny edge, prawdopodobieństwo spadku  
- NO_OP: Dane są, ale edge poniżej progu (dawne HOLD)
- ABSTAIN: Brak danych lub detektor nieaktywny

KLUCZOWE: Głosuj świadomie progów. Jeśli wiesz że margin ≥2 jest wymagany,
nie forsuj BUY gdy edge jest słaby. Użyj NO_OP gdy dane są ale nieprzekonujące.
"""
    
    @classmethod
    def validate_vote(cls, vote: str, confidence: float) -> tuple[bool, str]:
        """
        Waliduje pojedynczy głos względem kontraktu
        
        Returns:
            (is_valid, reason)
        """
        if vote not in cls.VOTE_SPACE:
            return False, f"Invalid vote: {vote}, must be one of {cls.VOTE_SPACE}"
        
        if vote != "ABSTAIN" and confidence < cls.CONFIDENCE_FLOOR:
            return False, f"Confidence {confidence:.3f} below floor {cls.CONFIDENCE_FLOOR}"
        
        return True, "Valid"
    
    @classmethod
    def should_escalate(cls, buy_count: int, sell_count: int) -> bool:
        """
        Sprawdza czy potrzebna eskalacja do Debatera
        """
        if not cls.TIE_BREAK_POLICY["enabled"]:
            return False
            
        gap = abs(buy_count - sell_count)
        return gap < cls.REQUIRED_MARGIN and gap >= cls.TIE_BREAK_POLICY["min_gap_for_escalation"]