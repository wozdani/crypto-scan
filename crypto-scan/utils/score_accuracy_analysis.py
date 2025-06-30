#!/usr/bin/env python3
"""
Score Accuracy Analysis - Analiza skuteczności alertów TJDE

Analizuje performance alertów i generuje feedback dla systemu self-learning
"""

import json
import os
import random as system_random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional


def analyze_alert_performance(
    alerts_path: str = "data/alerts/alerts_history.json",
    candle_loader=None,
    hours_to_check: int = 2,
    success_threshold: float = 2.0
) -> Tuple[List[Dict], float]:
    """
    Analizuje skuteczność historycznych alertów TJDE
    
    Args:
        alerts_path: Ścieżka do pliku z historią alertów
        candle_loader: Funkcja do pobierania świec (symbol, after, duration_hours)
        hours_to_check: Ile godzin po alercie sprawdzać
        success_threshold: Próg sukcesu w % (domyślnie 2%)
        
    Returns:
        Tuple[logs, success_rate]: Lista logów i wskaźnik sukcesu
    """
    try:
        if not os.path.exists(alerts_path):
            print(f"⚠️ Alert history file not found: {alerts_path}")
            return [], 0.0
        
        with open(alerts_path, "r", encoding="utf-8") as f:
            alerts_data = json.load(f)
        
        alerts = alerts_data.get("alerts", []) if isinstance(alerts_data, dict) else alerts_data
        
        if not alerts:
            print("⚠️ No alerts found in history")
            return [], 0.0
        
        success_count = 0
        total_count = 0
        logs = []
        
        print(f"📊 Analyzing {len(alerts)} historical alerts...")
        
        for alert in alerts:
            try:
                symbol = alert.get("symbol", "UNKNOWN")
                alert_time_str = alert.get("timestamp", alert.get("created_at", ""))
                score = alert.get("final_score", alert.get("tjde_final_score", 0))
                
                if not alert_time_str:
                    continue
                
                # Parse timestamp
                try:
                    if "T" in alert_time_str:
                        alert_time = datetime.fromisoformat(alert_time_str.replace("Z", "+00:00"))
                    else:
                        alert_time = datetime.fromisoformat(alert_time_str)
                except:
                    continue
                
                # Use candle loader if provided
                if candle_loader:
                    try:
                        candles = candle_loader(symbol, after=alert_time, duration_hours=hours_to_check)
                        if not candles or len(candles) < 2:
                            continue
                        
                        start_price = float(candles[0]["close"])
                        end_price = float(candles[-1]["close"])
                        pct_change = (end_price - start_price) / start_price * 100
                        
                    except Exception as e:
                        print(f"⚠️ Candle loading failed for {symbol}: {e}")
                        continue
                else:
                    # Simulated performance for testing (remove in production)
                    pct_change = _simulate_performance_based_on_score(score)
                
                alert_success = pct_change >= success_threshold
                
                log_entry = {
                    "symbol": symbol,
                    "timestamp": alert_time_str,
                    "score": score,
                    "price_change_pct": round(pct_change, 2),
                    "success": alert_success,
                    "hours_checked": hours_to_check,
                    "features_used": alert.get("used_features", {}),
                    "context_modifiers": alert.get("context_modifiers", [])
                }
                
                logs.append(log_entry)
                
                if alert_success:
                    success_count += 1
                total_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing alert: {e}")
                continue
        
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        print(f"📈 Analysis complete: {success_count}/{total_count} successful ({success_rate:.1%})")
        
        return logs, success_rate
        
    except Exception as e:
        print(f"❌ Error in alert performance analysis: {e}")
        return [], 0.0


def analyze_feature_effectiveness(logs: List[Dict]) -> Dict[str, float]:
    """
    Analizuje skuteczność poszczególnych cech w predykcji sukcesu
    
    Args:
        logs: Lista logów z analyze_alert_performance
        
    Returns:
        Dict z effectiveness score dla każdej cechy
    """
    feature_effectiveness = {}
    
    try:
        if not logs:
            return feature_effectiveness
        
        # Collect feature values for successful vs failed alerts
        successful_features = []
        failed_features = []
        
        for log in logs:
            features = log.get("features_used", {})
            if log.get("success", False):
                successful_features.append(features)
            else:
                failed_features.append(features)
        
        # Calculate effectiveness for each feature
        for feature_name in ["trend_strength", "pullback_quality", "support_reaction", 
                           "liquidity_pattern_score", "psych_score", "htf_supportive_score", 
                           "market_phase_modifier"]:
            
            if not successful_features and not failed_features:
                continue
            
            # Average feature value in successful alerts
            success_avg = 0.0
            if successful_features:
                success_values = [f.get(feature_name, 0) for f in successful_features]
                success_avg = sum(success_values) / len(success_values)
            
            # Average feature value in failed alerts
            fail_avg = 0.0
            if failed_features:
                fail_values = [f.get(feature_name, 0) for f in failed_features]
                fail_avg = sum(fail_values) / len(fail_values)
            
            # Effectiveness = difference between success and failure averages
            effectiveness = success_avg - fail_avg
            feature_effectiveness[feature_name] = round(effectiveness, 4)
        
        print(f"🧠 Feature effectiveness analysis complete")
        
    except Exception as e:
        print(f"❌ Error in feature effectiveness analysis: {e}")
    
    return feature_effectiveness


def _simulate_performance_based_on_score(score: float) -> float:
    """
    Symuluje performance na podstawie score (do testów bez live data)
    W produkcji należy usunąć i używać prawdziwych danych cenowych
    """
    # Higher scores tend to perform better (but with randomness)
    base_performance = score * 10  # Convert 0-1 score to 0-10% range
    
    # Add some randomness
    random_factor = system_random.uniform(-3, 3)
    
    # Simulate market conditions
    market_noise = system_random.uniform(-2, 2)
    
    return base_performance + random_factor + market_noise


def save_performance_analysis(logs: List[Dict], success_rate: float, output_path: str = "logs/tjde_performance_analysis.json"):
    """Zapisuje analizę performance do pliku"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        analysis_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_alerts": len(logs),
            "successful_alerts": sum(1 for log in logs if log.get("success", False)),
            "success_rate": round(success_rate, 4),
            "feature_effectiveness": analyze_feature_effectiveness(logs),
            "detailed_logs": logs
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Performance analysis saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving performance analysis: {e}")


if __name__ == "__main__":
    # Test analysis system
    print("🧪 Testing Score Accuracy Analysis System...")
    
    # Create test alert data
    test_alerts = [
        {
            "symbol": "BTCUSDT",
            "timestamp": "2025-06-22T18:00:00Z",
            "final_score": 0.75,
            "used_features": {"trend_strength": 0.8, "pullback_quality": 0.7}
        },
        {
            "symbol": "ETHUSDT", 
            "timestamp": "2025-06-22T17:00:00Z",
            "final_score": 0.45,
            "used_features": {"trend_strength": 0.4, "pullback_quality": 0.5}
        }
    ]
    
    # Test with simulated data
    os.makedirs("data/alerts", exist_ok=True)
    test_path = "data/alerts/test_alerts.json"
    
    with open(test_path, "w") as f:
        json.dump(test_alerts, f)
    
    # Run analysis
    logs, success_rate = analyze_alert_performance(test_path)
    
    print(f"📊 Test Results:")
    print(f"  Alerts analyzed: {len(logs)}")
    print(f"  Success rate: {success_rate:.1%}")
    
    # Save analysis
    save_performance_analysis(logs, success_rate, "logs/test_performance.json")
    
    # Cleanup
    os.remove(test_path)
    
    print("✅ Score Accuracy Analysis test complete")