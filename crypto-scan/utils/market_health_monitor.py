#!/usr/bin/env python3
"""
Market Health Monitor - Monitoring i analiza jakości sygnałów rynkowych
Wykrywa okresy słabych sygnałów i generuje alerty o stanie rynku
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import statistics

class MarketHealthMonitor:
    """Monitor zdrowia rynku i jakości sygnałów TJDE"""
    
    def __init__(self):
        self.data_dir = "data/market_health"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def record_scan_health(self, scan_results: List[Dict], timestamp: str = None) -> Dict:
        """
        Zapisuje stan zdrowia rynku po skanie
        
        Args:
            scan_results: Wyniki skanowania tokenów
            timestamp: Opcjonalny timestamp
            
        Returns:
            Statystyki zdrowia rynku
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Wyciągnij scores z wyników
        scores = []
        decisions = {"long": 0, "short": 0, "wait": 0, "avoid": 0, "skip": 0}
        quality_levels = {"high": 0, "medium": 0, "low": 0, "negative": 0}
        
        for result in scan_results:
            if result and isinstance(result, dict):
                score = result.get('tjde_score', 0.0)
                decision = result.get('tjde_decision', 'unknown')
                
                if score is not None and score != 0.0:
                    scores.append(score)
                    
                    # Klasyfikacja jakości
                    if score >= 0.6:
                        quality_levels["high"] += 1
                    elif score >= 0.3:
                        quality_levels["medium"] += 1
                    elif score > 0.0:
                        quality_levels["low"] += 1
                    else:
                        quality_levels["negative"] += 1
                    
                    # Zlicz decyzje
                    if decision in decisions:
                        decisions[decision] += 1
        
        # Oblicz statystyki
        health_stats = {
            "timestamp": timestamp,
            "total_tokens": len(scan_results),
            "valid_scores": len(scores),
            "score_stats": {
                "max": max(scores) if scores else 0.0,
                "mean": statistics.mean(scores) if scores else 0.0,
                "median": statistics.median(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0
            },
            "quality_distribution": quality_levels,
            "decision_distribution": decisions,
            "market_condition": self._assess_market_condition(scores, quality_levels),
            "chart_generation_eligible": quality_levels["high"] + quality_levels["medium"]
        }
        
        # Zapisz histogram scores
        if scores:
            health_stats["score_histogram"] = self._create_score_histogram(scores)
        
        # Zapisz do pliku
        filename = f"{self.data_dir}/health_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(health_stats, f, indent=2)
        
        # Sprawdź czy trzeba wysłać alert
        alert_info = self._check_market_alerts(health_stats)
        if alert_info:
            health_stats["alert"] = alert_info
            print(f"🚨 [MARKET ALERT] {alert_info['message']}")
        
        return health_stats
    
    def _assess_market_condition(self, scores: List[float], quality_levels: Dict) -> str:
        """Ocenia ogólny stan rynku na podstawie scores"""
        if not scores:
            return "no_signals"
        
        max_score = max(scores)
        high_quality = quality_levels["high"]
        medium_quality = quality_levels["medium"]
        total_valid = quality_levels["high"] + quality_levels["medium"] + quality_levels["low"]
        
        if max_score >= 0.7:
            return "excellent"
        elif max_score >= 0.5 and high_quality >= 3:
            return "good"
        elif max_score >= 0.3 and (high_quality + medium_quality) >= 5:
            return "moderate"
        elif max_score >= 0.15:
            return "weak"
        else:
            return "very_weak"
    
    def _create_score_histogram(self, scores: List[float]) -> Dict:
        """Tworzy histogram scores dla analiz"""
        bins = {
            "excellent_0.7+": len([s for s in scores if s >= 0.7]),
            "good_0.5-0.7": len([s for s in scores if 0.5 <= s < 0.7]),
            "moderate_0.3-0.5": len([s for s in scores if 0.3 <= s < 0.5]),
            "weak_0.1-0.3": len([s for s in scores if 0.1 <= s < 0.3]),
            "very_weak_0.0-0.1": len([s for s in scores if 0.0 <= s < 0.1]),
            "negative_<0.0": len([s for s in scores if s < 0.0])
        }
        return bins
    
    def _check_market_alerts(self, health_stats: Dict) -> Optional[Dict]:
        """Sprawdza czy wysłać alert o stanie rynku"""
        condition = health_stats["market_condition"]
        max_score = health_stats["score_stats"]["max"]
        high_quality = health_stats["quality_distribution"]["high"]
        
        # Alert: Bardzo słaby rynek przez długi czas
        if condition in ["very_weak", "weak"] and max_score < 0.15:
            recent_files = self._get_recent_health_files(hours=4)
            weak_periods = 0
            
            for file_path in recent_files:
                try:
                    with open(file_path, 'r') as f:
                        past_health = json.load(f)
                        if past_health.get("market_condition") in ["very_weak", "weak"]:
                            weak_periods += 1
                except:
                    continue
            
            if weak_periods >= 8:  # 4 godziny słabych sygnałów (co 30 min)
                return {
                    "type": "prolonged_weakness",
                    "message": f"[NO SIGNAL MARKET] TrendMode in full risk-off - {weak_periods} consecutive weak periods",
                    "severity": "high",
                    "recommendation": "Consider switching to accumulation or wait mode"
                }
        
        # Alert: Kompletny brak wysokiej jakości sygnałów
        if high_quality == 0 and max_score < 0.2:
            return {
                "type": "no_quality_signals",
                "message": f"[QUALITY DROUGHT] No high-quality signals detected (max: {max_score:.3f})",
                "severity": "medium",
                "recommendation": "Maintain defensive posture"
            }
        
        return None
    
    def _get_recent_health_files(self, hours: int = 4) -> List[str]:
        """Pobiera pliki health z ostatnich X godzin"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_files = []
        
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.startswith("health_") and filename.endswith(".json"):
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time >= cutoff_time:
                            recent_files.append(file_path)
                    except:
                        continue
        
        return sorted(recent_files)
    
    def get_market_summary(self, hours: int = 24) -> Dict:
        """Generuje podsumowanie stanu rynku z ostatnich X godzin"""
        recent_files = self._get_recent_health_files(hours)
        
        if not recent_files:
            return {"error": "No recent health data available"}
        
        all_conditions = []
        all_max_scores = []
        total_charts_eligible = 0
        
        for file_path in recent_files:
            try:
                with open(file_path, 'r') as f:
                    health = json.load(f)
                    all_conditions.append(health.get("market_condition", "unknown"))
                    all_max_scores.append(health.get("score_stats", {}).get("max", 0.0))
                    total_charts_eligible += health.get("chart_generation_eligible", 0)
            except:
                continue
        
        # Analiza trendu
        condition_counts = {}
        for condition in all_conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        dominant_condition = max(condition_counts.items(), key=lambda x: x[1])[0] if condition_counts else "unknown"
        
        return {
            "period_hours": hours,
            "scans_analyzed": len(recent_files),
            "dominant_condition": dominant_condition,
            "condition_distribution": condition_counts,
            "max_score_peak": max(all_max_scores) if all_max_scores else 0.0,
            "avg_max_score": statistics.mean(all_max_scores) if all_max_scores else 0.0,
            "total_chart_eligible_tokens": total_charts_eligible,
            "trend_assessment": self._assess_trend(all_max_scores)
        }
    
    def _assess_trend(self, max_scores: List[float]) -> str:
        """Ocenia trend jakości sygnałów"""
        if len(max_scores) < 3:
            return "insufficient_data"
        
        recent_avg = statistics.mean(max_scores[-3:])
        older_avg = statistics.mean(max_scores[:-3]) if len(max_scores) > 3 else recent_avg
        
        if recent_avg > older_avg * 1.2:
            return "improving"
        elif recent_avg < older_avg * 0.8:
            return "deteriorating"
        else:
            return "stable"

# Global instance
market_monitor = MarketHealthMonitor()

def log_scan_health(scan_results: List[Dict]) -> Dict:
    """Helper function to log scan health from main pipeline"""
    return market_monitor.record_scan_health(scan_results)

def check_market_condition() -> Dict:
    """Helper function to get current market condition"""
    return market_monitor.get_market_summary(hours=6)