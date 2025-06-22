#!/usr/bin/env python3
"""
Update Advanced Weights - Samouczący się feedback loop

Automatycznie aktualizuje wagi scoringowe na podstawie skuteczności alertów
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List


def update_weights_based_on_performance(
    performance_logs: List[Dict],
    weights_path: str = "data/scoring/advanced_weights.json",
    learning_rate: float = 0.01,
    min_weight: float = 0.05,
    max_weight: float = 0.4
) -> Dict[str, float]:
    """
    Aktualizuje wagi na podstawie analizy skuteczności
    
    Args:
        performance_logs: Logi z analyze_alert_performance
        weights_path: Ścieżka do pliku z wagami
        learning_rate: Szybkość uczenia (0.01 = 1% adjustment per update)
        min_weight: Minimalna waga
        max_weight: Maksymalna waga
        
    Returns:
        Dict z nowymi wagami
    """
    try:
        # Load current weights
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                weights_data = json.load(f)
                weights = {k: v for k, v in weights_data.items() if isinstance(v, (int, float))}
        else:
            # Default weights
            weights = {
                "trend_strength": 0.25,
                "pullback_quality": 0.2,
                "support_reaction": 0.15,
                "liquidity_pattern_score": 0.1,
                "psych_score": 0.1,
                "htf_supportive_score": 0.1,
                "market_phase_modifier": 0.1
            }
        
        if not performance_logs:
            print("⚠️ No performance logs provided for weight update")
            return weights
        
        print(f"🧠 Updating weights based on {len(performance_logs)} performance logs...")
        
        # Calculate feature effectiveness
        feature_success_impact = _calculate_feature_impact(performance_logs)
        
        # Apply weight adjustments
        adjustments = {}
        for feature_name in weights.keys():
            impact = feature_success_impact.get(feature_name, 0.0)
            
            # Positive impact = increase weight, negative = decrease
            adjustment = impact * learning_rate
            adjustments[feature_name] = adjustment
            
            # Apply adjustment
            new_weight = weights[feature_name] + adjustment
            weights[feature_name] = max(min_weight, min(new_weight, max_weight))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] = weights[key] / total_weight
        
        # Round for readability
        for key in weights:
            weights[key] = round(weights[key], 4)
        
        # Save updated weights
        _save_updated_weights(weights, adjustments, performance_logs, weights_path)
        
        print(f"✅ Weights updated successfully")
        _print_weight_changes(adjustments, weights)
        
        return weights
        
    except Exception as e:
        print(f"❌ Error updating weights: {e}")
        return weights


def _calculate_feature_impact(logs: List[Dict]) -> Dict[str, float]:
    """Oblicza wpływ każdej cechy na sukces alertów"""
    feature_impact = {}
    
    try:
        successful_logs = [log for log in logs if log.get("success", False)]
        failed_logs = [log for log in logs if not log.get("success", False)]
        
        feature_names = ["trend_strength", "pullback_quality", "support_reaction",
                        "liquidity_pattern_score", "psych_score", "htf_supportive_score", 
                        "market_phase_modifier"]
        
        for feature_name in feature_names:
            # Average feature value in successful alerts
            success_values = []
            for log in successful_logs:
                features = log.get("features_used", {})
                if feature_name in features:
                    success_values.append(features[feature_name])
            
            # Average feature value in failed alerts
            fail_values = []
            for log in failed_logs:
                features = log.get("features_used", {})
                if feature_name in features:
                    fail_values.append(features[feature_name])
            
            # Calculate impact
            if success_values and fail_values:
                success_avg = sum(success_values) / len(success_values)
                fail_avg = sum(fail_values) / len(fail_values)
                impact = success_avg - fail_avg
            elif success_values:
                # Only successful data available
                success_avg = sum(success_values) / len(success_values)
                impact = success_avg - 0.5  # Compare to neutral
            elif fail_values:
                # Only failed data available
                fail_avg = sum(fail_values) / len(fail_values)
                impact = 0.5 - fail_avg  # Negative impact
            else:
                impact = 0.0
            
            feature_impact[feature_name] = round(impact, 4)
    
    except Exception as e:
        print(f"⚠️ Error calculating feature impact: {e}")
    
    return feature_impact


def _save_updated_weights(
    weights: Dict[str, float],
    adjustments: Dict[str, float],
    performance_logs: List[Dict],
    weights_path: str
):
    """Zapisuje zaktualizowane wagi z metadanymi"""
    try:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        
        # Calculate success rate from logs
        successful = sum(1 for log in performance_logs if log.get("success", False))
        success_rate = successful / len(performance_logs) if performance_logs else 0.0
        
        weights_data = {
            **weights,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "version": "2.0_adaptive",
            "total_adjustments": len(performance_logs),
            "success_rate": round(success_rate, 4),
            "adjustments_applied": adjustments,
            "performance_summary": {
                "total_alerts": len(performance_logs),
                "successful_alerts": successful,
                "success_rate": round(success_rate, 4)
            }
        }
        
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Error saving weights: {e}")


def _print_weight_changes(adjustments: Dict[str, float], new_weights: Dict[str, float]):
    """Wyświetla zmiany wag"""
    print(f"\n📊 WEIGHT ADJUSTMENTS:")
    print(f"{'Feature':<25} {'Adjustment':<12} {'New Weight':<12}")
    print("-" * 50)
    
    for feature in new_weights:
        adjustment = adjustments.get(feature, 0.0)
        new_weight = new_weights[feature]
        
        adjustment_str = f"{adjustment:+.4f}" if adjustment != 0 else "0.0000"
        arrow = "📈" if adjustment > 0 else "📉" if adjustment < 0 else "➡️"
        
        print(f"{feature:<25} {adjustment_str:<12} {new_weight:<12.4f} {arrow}")


def get_advanced_weights(weights_path: str = "data/scoring/advanced_weights.json") -> Dict[str, float]:
    """
    Pobiera aktualne wagi adaptacyjne
    Używane w simulate_trader_decision_advanced()
    """
    try:
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                weights_data = json.load(f)
                
            # Extract only numeric weights
            weights = {}
            for key, value in weights_data.items():
                if isinstance(value, (int, float)) and key in [
                    "trend_strength", "pullback_quality", "support_reaction",
                    "liquidity_pattern_score", "psych_score", "htf_supportive_score", 
                    "market_phase_modifier"
                ]:
                    weights[key] = float(value)
            
            # Validate weights sum close to 1.0
            total = sum(weights.values())
            if 0.9 <= total <= 1.1:  # Allow small floating point errors
                return weights
            else:
                print(f"⚠️ Invalid weights sum ({total:.3f}), using defaults")
        
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}")
    
    # Return default weights
    return {
        "trend_strength": 0.25,
        "pullback_quality": 0.2,
        "support_reaction": 0.15,
        "liquidity_pattern_score": 0.1,
        "psych_score": 0.1,
        "htf_supportive_score": 0.1,
        "market_phase_modifier": 0.1
    }


def run_feedback_loop(
    performance_logs_path: str = "logs/tjde_performance_analysis.json",
    weights_path: str = "data/scoring/advanced_weights.json"
) -> bool:
    """
    Uruchamia pełny feedback loop
    
    Returns:
        bool: True jeśli weights zostały zaktualizowane
    """
    try:
        print("🔄 Running adaptive feedback loop...")
        
        # Load performance logs
        if not os.path.exists(performance_logs_path):
            print(f"⚠️ Performance logs not found: {performance_logs_path}")
            return False
        
        with open(performance_logs_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        
        performance_logs = analysis_data.get("detailed_logs", [])
        
        if not performance_logs:
            print("⚠️ No performance logs found")
            return False
        
        # Update weights
        new_weights = update_weights_based_on_performance(performance_logs, weights_path)
        
        print(f"✅ Feedback loop complete - weights updated based on {len(performance_logs)} alerts")
        return True
        
    except Exception as e:
        print(f"❌ Error in feedback loop: {e}")
        return False


if __name__ == "__main__":
    # Test weight update system
    print("🧪 Testing Advanced Weights Update System...")
    
    # Create test performance logs
    test_logs = [
        {
            "symbol": "BTCUSDT",
            "success": True,
            "features_used": {"trend_strength": 0.8, "pullback_quality": 0.7, "psych_score": 0.2}
        },
        {
            "symbol": "ETHUSDT",
            "success": False,
            "features_used": {"trend_strength": 0.3, "pullback_quality": 0.4, "psych_score": 0.8}
        },
        {
            "symbol": "ADAUSDT",
            "success": True,
            "features_used": {"trend_strength": 0.9, "pullback_quality": 0.6, "psych_score": 0.1}
        }
    ]
    
    # Test weight update
    updated_weights = update_weights_based_on_performance(test_logs, "data/scoring/test_weights.json")
    
    print(f"\n📊 Test Results:")
    for feature, weight in updated_weights.items():
        print(f"  {feature}: {weight:.4f}")
    
    # Test weight retrieval
    retrieved_weights = get_advanced_weights("data/scoring/test_weights.json")
    print(f"\n✅ Weight retrieval test: {len(retrieved_weights)} weights loaded")
    
    # Cleanup
    if os.path.exists("data/scoring/test_weights.json"):
        os.remove("data/scoring/test_weights.json")
    
    print("✅ Advanced Weights Update test complete")