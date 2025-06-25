"""
CLIP-GPT Mapping System
Maps GPT commentary to CLIP labels and enhances decision making
"""
import re
from typing import Dict, Optional, List, Tuple

class CLIPGPTMapper:
    """Maps GPT commentary to CLIP labels and provides enhanced scoring"""
    
    def __init__(self):
        # GPT commentary patterns to CLIP labels mapping
        self.gpt_to_clip_patterns = {
            "pullback-in-trend": [
                r"pullback.*trend", r"cofnięcie.*trend", r"korekta.*wzrost",
                r"pullback.*squeeze", r"cofnięcie.*ściśnięcie", 
                r"retest.*support", r"test.*wsparcie", r"bounce.*dip"
            ],
            "breakout-continuation": [
                r"breakout.*continuation", r"wybicie.*kontynuacja",
                r"momentum.*breakout", r"strong.*breakout", r"volume.*breakout",
                r"trend.*acceleration", r"impulse.*move"
            ],
            "trend-reversal": [
                r"reversal.*pattern", r"odwrócenie.*trend", 
                r"exhaustion.*pattern", r"wyczerpanie.*trend",
                r"double.*top", r"double.*bottom", r"head.*shoulders"
            ],
            "range-accumulation": [
                r"range.*bound", r"consolidation", r"accumulation",
                r"sideways.*movement", r"boczny.*ruch", r"akumulacja"
            ],
            "support-bounce": [
                r"support.*bounce", r"wsparcie.*odbicie",
                r"bounce.*from.*support", r"test.*support.*hold",
                r"strong.*support", r"key.*support.*level"
            ],
            "resistance-rejection": [
                r"resistance.*rejection", r"opór.*odrzucenie",
                r"failed.*breakout", r"fake.*breakout", r"false.*break",
                r"rejection.*resistance"
            ],
            "volume-backed": [
                r"volume.*increase", r"wolumen.*wzrost",
                r"high.*volume", r"volume.*confirmation",
                r"buying.*pressure", r"volume.*spike"
            ],
            "exhaustion-pattern": [
                r"exhaustion", r"wyczerpanie", r"tired.*move",
                r"weakening.*momentum", r"divergence", r"blow.*off.*top"
            ]
        }
        
        # CLIP confidence boosts based on patterns
        self.pattern_confidence_boosts = {
            "pullback-in-trend": 0.05,
            "breakout-continuation": 0.08,
            "support-bounce": 0.04,
            "volume-backed": 0.06,
            "trend-reversal": -0.03,  # Lower confidence for reversals
            "exhaustion-pattern": -0.05,
            "resistance-rejection": -0.04
        }
        
        # Scoring modifiers based on CLIP+GPT consensus
        self.consensus_modifiers = {
            "high_confidence_bullish": 0.10,    # CLIP + GPT both bullish, high confidence
            "medium_confidence_bullish": 0.05,
            "neutral_consensus": 0.00,
            "medium_confidence_bearish": -0.03,
            "high_confidence_bearish": -0.06,
            "pattern_conflict": -0.02           # CLIP and GPT disagree
        }
    
    def map_gpt_to_clip_label(self, gpt_commentary: str, current_clip_label: str = "unknown") -> str:
        """
        Map GPT commentary to appropriate CLIP label
        
        Args:
            gpt_commentary: GPT generated commentary
            current_clip_label: Current CLIP label (if any)
            
        Returns:
            Enhanced or corrected CLIP label
        """
        if not gpt_commentary or current_clip_label != "unknown":
            return current_clip_label
            
        commentary_lower = gpt_commentary.lower()
        
        # Check each pattern category
        for label, patterns in self.gpt_to_clip_patterns.items():
            for pattern in patterns:
                if re.search(pattern, commentary_lower):
                    print(f"[CLIP-GPT MAP] GPT→CLIP: '{pattern}' → {label}")
                    return label
        
        # Fallback: basic sentiment analysis
        bullish_terms = ["breakout", "wybicie", "momentum", "strong", "bounce", "support"]
        bearish_terms = ["rejection", "odrzucenie", "exhaustion", "weak", "resistance"]
        
        bullish_score = sum(1 for term in bullish_terms if term in commentary_lower)
        bearish_score = sum(1 for term in bearish_terms if term in commentary_lower)
        
        if bullish_score > bearish_score:
            return "pullback-in-trend"  # Conservative bullish
        elif bearish_score > bullish_score:
            return "resistance-rejection"  # Conservative bearish
        else:
            return "range-accumulation"  # Neutral
    
    def analyze_clip_gpt_consensus(self, clip_info: Dict, gpt_commentary: str) -> Dict:
        """
        Analyze consensus between CLIP prediction and GPT commentary
        
        Args:
            clip_info: CLIP prediction information
            gpt_commentary: GPT commentary text
            
        Returns:
            Consensus analysis with scoring modifiers
        """
        clip_label = clip_info.get("trend_label", "unknown")
        clip_confidence = clip_info.get("clip_confidence", 0.0)
        
        # Map GPT to CLIP if needed
        if clip_label == "unknown" and gpt_commentary:
            mapped_label = self.map_gpt_to_clip_label(gpt_commentary, clip_label)
            if mapped_label != "unknown":
                clip_label = mapped_label
                print(f"[CLIP-GPT ENHANCE] Mapped GPT→CLIP: {mapped_label}")
        
        # Analyze sentiment alignment
        commentary_lower = gpt_commentary.lower() if gpt_commentary else ""
        
        # Calculate consensus strength
        consensus_strength = self._calculate_consensus_strength(clip_label, commentary_lower, clip_confidence)
        
        # Determine consensus category
        consensus_category = self._determine_consensus_category(consensus_strength, clip_confidence)
        
        # Calculate final modifier
        base_modifier = self.consensus_modifiers.get(consensus_category, 0.0)
        pattern_boost = self.pattern_confidence_boosts.get(clip_label, 0.0)
        
        total_modifier = base_modifier + (pattern_boost * clip_confidence)
        
        return {
            "enhanced_clip_label": clip_label,
            "consensus_strength": consensus_strength,
            "consensus_category": consensus_category,
            "scoring_modifier": total_modifier,
            "confidence_boost": pattern_boost,
            "gpt_mapped": clip_info.get("trend_label", "unknown") == "unknown",
            "analysis_details": {
                "original_clip": clip_info.get("trend_label", "unknown"),
                "gpt_patterns_found": self._extract_gpt_patterns(commentary_lower),
                "clip_confidence": clip_confidence
            }
        }
    
    def _calculate_consensus_strength(self, clip_label: str, commentary_lower: str, clip_confidence: float) -> float:
        """Calculate strength of CLIP-GPT consensus"""
        
        # Define label sentiment mapping
        bullish_labels = ["breakout-continuation", "pullback-in-trend", "support-bounce", "volume-backed"]
        bearish_labels = ["trend-reversal", "resistance-rejection", "exhaustion-pattern"]
        neutral_labels = ["range-accumulation", "consolidation"]
        
        # Determine CLIP sentiment
        clip_sentiment = 0.0
        if clip_label in bullish_labels:
            clip_sentiment = 1.0
        elif clip_label in bearish_labels:
            clip_sentiment = -1.0
        
        # Analyze GPT sentiment
        bullish_terms = ["breakout", "momentum", "strong", "bounce", "support", "trend", "acceleration"]
        bearish_terms = ["rejection", "exhaustion", "weak", "resistance", "reversal", "failed"]
        
        gpt_bullish = sum(1 for term in bullish_terms if term in commentary_lower)
        gpt_bearish = sum(1 for term in bearish_terms if term in commentary_lower)
        
        if gpt_bullish + gpt_bearish == 0:
            gpt_sentiment = 0.0
        else:
            gpt_sentiment = (gpt_bullish - gpt_bearish) / (gpt_bullish + gpt_bearish)
        
        # Calculate alignment
        if clip_sentiment == 0.0 and gpt_sentiment == 0.0:
            return 0.5  # Both neutral
        elif clip_sentiment * gpt_sentiment > 0:
            return 0.8 * clip_confidence  # Aligned sentiment
        elif clip_sentiment * gpt_sentiment < 0:
            return 0.2 * clip_confidence  # Conflicting sentiment
        else:
            return 0.4 * clip_confidence  # One neutral, one directional
    
    def _determine_consensus_category(self, consensus_strength: float, clip_confidence: float) -> str:
        """Determine consensus category for modifier lookup"""
        
        if consensus_strength > 0.7 and clip_confidence > 0.6:
            return "high_confidence_bullish"
        elif consensus_strength > 0.5 and clip_confidence > 0.4:
            return "medium_confidence_bullish"
        elif consensus_strength < 0.3 and clip_confidence > 0.4:
            if consensus_strength < 0.15:
                return "high_confidence_bearish"
            else:
                return "medium_confidence_bearish"
        elif consensus_strength < 0.35:
            return "pattern_conflict"
        else:
            return "neutral_consensus"
    
    def _extract_gpt_patterns(self, commentary_lower: str) -> List[str]:
        """Extract recognized patterns from GPT commentary"""
        patterns_found = []
        
        for label, patterns in self.gpt_to_clip_patterns.items():
            for pattern in patterns:
                if re.search(pattern, commentary_lower):
                    patterns_found.append(f"{label}:{pattern}")
        
        return patterns_found

# Global instance
clip_gpt_mapper = CLIPGPTMapper()