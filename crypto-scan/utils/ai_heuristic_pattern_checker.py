"""
AI Heuristic Pattern Checker
Sprawdza kombinacje cech które historycznie oznaczały silny trend
Umożliwia alerty nawet przy niskim scoringu jeśli pattern pasuje
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import centralized error logging system
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from utils.scan_error_reporter import log_warning
except ImportError:
    def log_warning(label, exception=None, additional_info=None):
        """Fallback log_warning if import fails"""
        if exception:
            print(f"⚠️ [{label}] {exception} - {additional_info or ''}")
        else:
            print(f"⚠️ [{label}] {additional_info or ''}")

logger = logging.getLogger(__name__)

class AIHeuristicPatternChecker:
    """Checker dla AI heuristic patterns"""
    
    def __init__(self, patterns_file: str = "data/ai_successful_patterns.json"):
        """Initialize pattern checker"""
        self.patterns_file = Path(patterns_file)
        self.patterns = self._load_success_patterns()
        
        logger.info(f"AI Heuristic Pattern Checker initialized with {len(self.patterns)} patterns")
    
    def _load_success_patterns(self) -> List[Dict]:
        """Load success patterns from JSON file"""
        try:
            if not self.patterns_file.exists():
                logger.warning(f"Patterns file not found: {self.patterns_file}")
                self._create_default_patterns()
            
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            
            logger.info(f"Loaded {len(patterns)} AI success patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading success patterns: {e}")
            return self._get_default_patterns()
    
    def _create_default_patterns(self):
        """Create default patterns file if it doesn't exist"""
        try:
            self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
            
            default_patterns = self._get_default_patterns()
            
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(default_patterns, f, indent=2)
            
            logger.info(f"Created default patterns file: {self.patterns_file}")
            
        except Exception as e:
            logger.error(f"Error creating default patterns file: {e}")
    
    def _get_default_patterns(self) -> List[Dict]:
        """Get default success patterns"""
        return [
            {
                "features": [
                    "volume_behavior=buying_volume_increase",
                    "psych_flags=liquidity_grab",
                    "trend_strength>0.4"
                ],
                "label": "buy_volume_liquidity_combo",
                "success_rate": 0.86,
                "min_score": 0.35,
                "description": "Strong buying volume with liquidity grab - high success rate"
            },
            {
                "features": [
                    "volume_behavior=low_volume_after_breakout",
                    "psych_flags=pinning_detected",
                    "pullback_quality>0.6"
                ],
                "label": "hidden_accumulation_pattern",
                "success_rate": 0.81,
                "min_score": 0.33,
                "description": "Hidden accumulation with pinning after breakout"
            },
            {
                "features": [
                    "trend_strength>0.7",
                    "support_reaction>0.6",
                    "volume_behavior=volume_spike"
                ],
                "label": "strong_trend_support_volume",
                "success_rate": 0.78,
                "min_score": 0.4,
                "description": "Strong trend with support reaction and volume spike"
            },
            {
                "features": [
                    "psych_flags=fakeout_rejection",
                    "pullback_quality>0.7",
                    "htf_supportive_score>0.5"
                ],
                "label": "fakeout_rejection_htf_support",
                "success_rate": 0.74,
                "min_score": 0.38,
                "description": "Fakeout rejection with strong pullback and HTF support"
            },
            {
                "features": [
                    "liquidity_pattern_score>0.6",
                    "psych_flags=momentum_confirmed",
                    "market_phase=trending-up"
                ],
                "label": "momentum_liquidity_trending",
                "success_rate": 0.82,
                "min_score": 0.42,
                "description": "Confirmed momentum with liquidity in uptrend"
            },
            {
                "features": [
                    "volume_behavior=buying_volume_increase",
                    "trend_strength>0.6",
                    "psych_flags=bounce_confirmed"
                ],
                "label": "volume_trend_bounce_combo",
                "success_rate": 0.79,
                "min_score": 0.36,
                "description": "Volume increase with strong trend and confirmed bounce"
            }
        ]
    
    def feature_matches_condition(self, feature_dict: Dict, condition: str) -> bool:
        """
        Match single condition string like `trend_strength>0.4`
        
        Args:
            feature_dict: Dictionary with feature values
            condition: Condition string to match
            
        Returns:
            True if condition is met
        """
        try:
            condition = condition.strip()
            
            # Handle equality conditions (key=value)
            if "=" in condition and ">" not in condition and "<" not in condition:
                key, val = condition.split("=", 1)
                key = key.strip()
                val = val.strip().lower()
                
                # Check in multiple possible locations
                feature_val = None
                if key in feature_dict:
                    feature_val = feature_dict[key]
                elif 'psych_flags' in feature_dict and isinstance(feature_dict['psych_flags'], dict):
                    if key.replace('psych_flags=', '') in feature_dict['psych_flags']:
                        return feature_dict['psych_flags'][key.replace('psych_flags=', '')]
                
                if feature_val is not None:
                    if isinstance(feature_val, dict):
                        # For dict values, check if the key exists and is True
                        return feature_val.get(val, False)
                    else:
                        return str(feature_val).lower() == val
                
                # Handle special psych_flags format
                if condition.startswith('psych_flags='):
                    flag_name = condition.replace('psych_flags=', '')
                    psych_flags = feature_dict.get('psych_flags', {})
                    if isinstance(psych_flags, dict):
                        return psych_flags.get(flag_name, False)
                    else:
                        return flag_name in str(psych_flags).lower()
                
                return False
            
            # Handle greater than conditions (key>value)
            elif ">" in condition:
                key, val = condition.split(">", 1)
                key = key.strip()
                val = float(val.strip())
                
                feature_val = feature_dict.get(key, -999)
                if isinstance(feature_val, (int, float)):
                    return feature_val > val
                return False
            
            # Handle less than conditions (key<value)
            elif "<" in condition:
                key, val = condition.split("<", 1)
                key = key.strip()
                val = float(val.strip())
                
                feature_val = feature_dict.get(key, 999)
                if isinstance(feature_val, (int, float)):
                    return feature_val < val
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error matching condition '{condition}': {e}")
            return False
    
    def check_known_success_patterns(self, feature_dict: Dict, current_score: float) -> Optional[Dict]:
        """
        Check if features match any known success patterns
        
        Args:
            feature_dict: Dictionary with current features
            current_score: Current scoring value
            
        Returns:
            Pattern match information or None
        """
        try:
            for pattern in self.patterns:
                # Skip if score is already above minimum threshold
                min_score = pattern.get("min_score", 0.0)
                if current_score >= min_score:
                    continue
                
                # Check if all conditions in pattern are met
                pattern_features = pattern.get("features", [])
                if not pattern_features:
                    continue
                
                matches = []
                all_match = True
                
                for condition in pattern_features:
                    match_result = self.feature_matches_condition(feature_dict, condition)
                    matches.append({
                        "condition": condition,
                        "matched": match_result
                    })
                    
                    if not match_result:
                        all_match = False
                        break
                
                if all_match:
                    logger.info(f"AI Pattern matched: {pattern.get('label', 'unknown')} with {len(pattern_features)} conditions")
                    
                    return {
                        "label": pattern.get("label", "unknown-pattern"),
                        "confidence": pattern.get("success_rate", 0.5),
                        "features_matched": pattern_features,
                        "description": pattern.get("description", ""),
                        "min_score": min_score,
                        "current_score": current_score,
                        "condition_details": matches
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking success patterns: {e}")
            return None
    
    def add_new_pattern(self, features: List[str], label: str, success_rate: float, min_score: float, description: str = ""):
        """Add new success pattern to the collection"""
        try:
            new_pattern = {
                "features": features,
                "label": label,
                "success_rate": success_rate,
                "min_score": min_score,
                "description": description
            }
            
            self.patterns.append(new_pattern)
            
            # Save updated patterns
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, indent=2)
            
            logger.info(f"Added new AI pattern: {label}")
            
        except Exception as e:
            logger.error(f"Error adding new pattern: {e}")
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns"""
        return {
            "total_patterns": len(self.patterns),
            "pattern_labels": [p.get("label", "unknown") for p in self.patterns],
            "avg_success_rate": sum(p.get("success_rate", 0) for p in self.patterns) / len(self.patterns) if self.patterns else 0,
            "min_scores": [p.get("min_score", 0) for p in self.patterns]
        }


# Global instance
_global_pattern_checker = None

def get_pattern_checker() -> AIHeuristicPatternChecker:
    """Get global pattern checker instance"""
    global _global_pattern_checker
    if _global_pattern_checker is None:
        _global_pattern_checker = AIHeuristicPatternChecker()
    return _global_pattern_checker

def check_known_success_patterns(feature_dict: Dict, current_score: float) -> Optional[Dict]:
    """
    Convenience function to check success patterns
    
    Args:
        feature_dict: Dictionary with features
        current_score: Current score
        
    Returns:
        Pattern match or None
    """
    checker = get_pattern_checker()
    return checker.check_known_success_patterns(feature_dict, current_score)

def main():
    """Test AI heuristic pattern checker"""
    # Testing AI Heuristic Pattern Checker
    
    checker = AIHeuristicPatternChecker()
    
    # Get pattern statistics
    stats = checker.get_pattern_stats()
    
    # Test pattern matching
    test_features = [
        {
            "name": "Buy Volume + Liquidity Grab",
            "features": {
                "volume_behavior": "buying_volume_increase",
                "psych_flags": {"liquidity_grab": True},
                "trend_strength": 0.45,
                "pullback_quality": 0.35
            },
            "score": 0.30
        },
        {
            "name": "Hidden Accumulation",
            "features": {
                "volume_behavior": "low_volume_after_breakout",
                "psych_flags": {"pinning_detected": True},
                "pullback_quality": 0.65,
                "trend_strength": 0.40
            },
            "score": 0.32
        },
        {
            "name": "No Pattern Match",
            "features": {
                "volume_behavior": "normal",
                "psych_flags": {},
                "trend_strength": 0.25,
                "pullback_quality": 0.30
            },
            "score": 0.28
        }
    ]
    
    # Testing Pattern Matching
    
    for i, test_case in enumerate(test_features, 1):
        match = checker.check_known_success_patterns(test_case['features'], test_case['score'])
    
    # AI Heuristic Pattern Checker test completed

if __name__ == "__main__":
    main()