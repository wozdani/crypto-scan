"""
Trend Mode Alert Engine - Trailing Scoring & Alert System

System zarządzania historią scoringu i generowania alertów dla Trend Mode
z integracją pullback_flow_pattern i innych detektorów.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


class TrendModeAlertEngine:
    """Zarządza trailing scoring i alertami dla Trend Mode"""
    
    def __init__(self, history_file="trend_score_history.json", max_history=5):
        self.history_file = history_file
        self.max_history = max_history
        self.score_history = self._load_history()
        
    def _load_history(self) -> Dict:
        """Ładuje historię scoringu z pliku"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading trend score history: {e}")
        return {}
    
    def _save_history(self):
        """Zapisuje historię scoringu do pliku"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.score_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving trend score history: {e}")
    
    def update_score_history(self, symbol: str, score: int, details: Dict):
        """
        Aktualizuje historię scoringu dla symbolu
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            score: Aktualny trend score
            details: Szczegóły scoringu (aktywne moduły, triggers)
        """
        if symbol not in self.score_history:
            self.score_history[symbol] = []
        
        # Dodaj nowy wpis
        entry = {
            "score": score,
            "timestamp": int(time.time()),
            "details": details
        }
        
        self.score_history[symbol].append(entry)
        
        # Ogranicz historię do max_history wpisów
        if len(self.score_history[symbol]) > self.max_history:
            self.score_history[symbol] = self.score_history[symbol][-self.max_history:]
        
        self._save_history()
    
    def get_score_change(self, symbol: str, current_score: int) -> Tuple[int, int]:
        """
        Oblicza zmianę score względem poprzedniego skanu
        
        Args:
            symbol: Trading symbol
            current_score: Aktualny score
            
        Returns:
            tuple: (previous_score, score_change)
        """
        if symbol not in self.score_history or len(self.score_history[symbol]) == 0:
            return 0, current_score
        
        previous_entry = self.score_history[symbol][-1]
        previous_score = previous_entry["score"]
        score_change = current_score - previous_score
        
        return previous_score, score_change
    
    def should_trigger_alert_trend_mode_2(self, symbol: str, current_score: int, alert_level: int, score_breakdown: Dict) -> Tuple[bool, str]:
        """
        Trend Mode 2.0 trailing logic - alert tylko przy wzroście score +5 AND przekroczeniu progu
        
        Args:
            symbol: Trading symbol
            current_score: Aktualny Trend Mode 2.0 score
            alert_level: Poziom alertu (1/2/3)
            score_breakdown: Breakdown scoringu
            
        Returns:
            tuple: (should_alert, reason)
        """
        previous_score, score_change = self.get_score_change(symbol, current_score)
        
        # Brak alertu jeśli level = 0
        if alert_level == 0:
            return False, f"No alert level (score: {current_score}, change: {score_change:+d})"
        
        # Trailing logic: Score musi wzrosnąć o min. +5 AND przekroczyć próg alertu
        score_increased = score_change >= 5
        
        # Progi dla różnych poziomów
        level_thresholds = {1: 25, 2: 40, 3: 60}
        threshold = level_thresholds.get(alert_level, 25)
        threshold_crossed = current_score >= threshold and previous_score < threshold
        
        # Alert tylko gdy BOTH warunki spełnione
        if score_increased and threshold_crossed:
            reason = f"Level {alert_level}: Score +{score_change} AND crossed {threshold} ({previous_score}→{current_score})"
            return True, reason
        
        # Alternatywnie: duży wzrost score (+15) nawet bez przekroczenia nowego progu
        if score_change >= 15 and current_score >= threshold:
            reason = f"Level {alert_level}: Major score increase +{score_change} (current: {current_score})"
            return True, reason
        
        # Brak warunków alertu
        if not score_increased:
            return False, f"Score change too small: {score_change:+d} (need +5)"
        elif not threshold_crossed:
            return False, f"Threshold not crossed: {current_score} (need {threshold}+)"
        else:
            return False, f"No alert conditions met (score: {current_score}, change: {score_change:+d})"

    def should_trigger_alert(self, symbol: str, current_score: int, details: Dict) -> Tuple[bool, str]:
        """
        Legacy compatibility - określa czy powinien zostać wysłany alert
        
        Args:
            symbol: Trading symbol
            current_score: Aktualny trend score
            details: Szczegóły scoringu
            
        Returns:
            tuple: (should_alert, reason)
        """
        previous_score, score_change = self.get_score_change(symbol, current_score)
        
        # Warunek 1: Score wzrósł o min. +10 punktów
        if score_change >= 10:
            reason = f"Score wzrósł o +{score_change} ({previous_score}→{current_score})"
            return True, reason
        
        # Warunek 2: Score przekroczył stały próg ≥70
        if current_score >= 70 and previous_score < 70:
            reason = f"Score przekroczył próg 70 ({previous_score}→{current_score})"
            return True, reason
        
        # Warunek 3: Wysoki score z pullback trigger (dodatkowy warunek)
        if current_score >= 65 and details.get("pullback_trigger", False):
            if previous_score < 60:  # Znaczący wzrost z pullback
                reason = f"Pullback trigger + score wzrost ({previous_score}→{current_score})"
                return True, reason
        
        return False, f"Brak warunków alertu (zmiana: {score_change:+d}, score: {current_score})"
    
    def compute_enhanced_trend_score(self, symbol: str, base_score: int, modules_data: Dict) -> Tuple[int, Dict]:
        """
        Oblicza wzmocniony trend score z integracją wszystkich modułów
        
        Args:
            symbol: Trading symbol
            base_score: Bazowy score z compute_trend_mode_score
            modules_data: Dane z różnych modułów (pullback, flow, orderbook)
            
        Returns:
            tuple: (enhanced_score, score_breakdown)
        """
        enhanced_score = base_score
        score_breakdown = {"base_score": base_score}
        active_modules = []
        
        # 1. Pullback Flow Pattern (+10 punktów)
        pullback_data = modules_data.get("pullback_flow", {})
        if pullback_data.get("pullback_trigger", False):
            pullback_bonus = 10
            enhanced_score += pullback_bonus
            score_breakdown["pullback_flow"] = pullback_bonus
            active_modules.append(f"pullback_flow({pullback_data.get('confidence_score', 0)})")
        
        # 2. Flow Consistency (+5 punktów)
        flow_consistency = modules_data.get("flow_consistency", 0)
        if flow_consistency > 70:  # >70% consistency
            flow_bonus = 5
            enhanced_score += flow_bonus
            score_breakdown["flow_consistency"] = flow_bonus
            active_modules.append(f"flow_consistency({flow_consistency}%)")
        
        # 3. Bullish Orderbook (+5 punktów)
        orderbook_data = modules_data.get("orderbook", {})
        bid_dominance = orderbook_data.get("bid_ask_ratio", 0) >= 1.2
        if bid_dominance:
            orderbook_bonus = 5
            enhanced_score += orderbook_bonus
            score_breakdown["orderbook_bullish"] = orderbook_bonus
            active_modules.append(f"bid_dominance({orderbook_data.get('bid_ask_ratio', 0):.2f})")
        
        # 4. One-Sided Pressure (+3 punkty)
        pressure_detected = modules_data.get("one_sided_pressure", False)
        if pressure_detected:
            pressure_bonus = 3
            enhanced_score += pressure_bonus
            score_breakdown["one_sided_pressure"] = pressure_bonus
            active_modules.append("one_sided_pressure")
        
        # 5. Heatmap Vacuum (+3 punkty)
        vacuum_detected = modules_data.get("heatmap_vacuum", False)
        if vacuum_detected:
            vacuum_bonus = 3
            enhanced_score += vacuum_bonus
            score_breakdown["heatmap_vacuum"] = vacuum_bonus
            active_modules.append("heatmap_vacuum")
        
        # Ograniczenie do maksymalnego score (np. 100)
        enhanced_score = min(enhanced_score, 100)
        
        score_breakdown["final_score"] = enhanced_score
        score_breakdown["active_modules"] = active_modules
        
        return enhanced_score, score_breakdown
    
    def generate_trend_mode_2_alert(self, symbol: str, score: int, alert_level: int, score_breakdown: Dict, reason: str) -> Dict:
        """
        Generuje alert dla Trend Mode 2.0
        
        Args:
            symbol: Trading symbol
            score: Final trend score
            alert_level: Poziom alertu (1/2/3)
            score_breakdown: Breakdown scoringu
            reason: Powód alertu
            
        Returns:
            dict: Alert data
        """
        level_names = {1: "WATCHLIST", 2: "ACTIVE ENTRY", 3: "CONFIRMED TREND"}
        level_name = level_names.get(alert_level, "UNKNOWN")
        
        # Formatuj wiadomość alertu
        alert_message = f"🔥 TREND MODE 2.0 - {level_name}: {symbol}\n"
        alert_message += f"📊 Score: {score}/100 (Level {alert_level}/3)\n"
        alert_message += f"🎯 {reason}\n"
        
        # Core detectors
        if score_breakdown.get('active_core'):
            alert_message += f"🔵 Core: {', '.join(score_breakdown['active_core'])}\n"
        
        # Helper detectors
        if score_breakdown.get('active_helper'):
            alert_message += f"🟢 Helper: {', '.join(score_breakdown['active_helper'])}\n"
        
        # Negative detectors (warning)
        if score_breakdown.get('active_negative'):
            alert_message += f"🔴 Warning: {', '.join(score_breakdown['active_negative'])}\n"
        
        # Points breakdown
        alert_message += f"📈 Points: Core +{score_breakdown.get('core_points', 0)}, Helper +{score_breakdown.get('helper_points', 0)}"
        if score_breakdown.get('negative_points', 0) < 0:
            alert_message += f", Negative {score_breakdown['negative_points']}"
        
        # Timestamp
        alert_message += f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        
        alert_data = {
            "symbol": symbol,
            "alert_type": "trend_mode_2",
            "alert_level": alert_level,
            "level_name": level_name,
            "score": score,
            "reason": reason,
            "message": alert_message,
            "score_breakdown": score_breakdown,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": self._calculate_trend_mode_2_priority(alert_level, score, score_breakdown)
        }
        
        return alert_data
    
    def _calculate_trend_mode_2_priority(self, alert_level: int, score: int, score_breakdown: Dict) -> str:
        """Oblicza priorytet alertu dla Trend Mode 2.0"""
        core_count = len(score_breakdown.get('active_core', []))
        negative_count = len(score_breakdown.get('active_negative', []))
        
        # HIGH priority
        if alert_level >= 3 or (score >= 70 and core_count >= 2):
            return "HIGH"
        
        # MEDIUM priority
        if alert_level == 2 or (score >= 40 and core_count >= 1):
            return "MEDIUM"
        
        # LOW priority but warning if negatives
        if negative_count >= 2:
            return "LOW-WARNING"
        
        return "LOW"

    def generate_trend_alert(self, symbol: str, score: int, score_breakdown: Dict, reason: str) -> Dict:
        """
        Generuje alert dla Trend Mode
        
        Args:
            symbol: Trading symbol
            score: Final trend score
            score_breakdown: Breakdown scoringu
            reason: Powód alertu
            
        Returns:
            dict: Alert data
        """
        active_modules = score_breakdown.get("active_modules", [])
        
        # Formatuj wiadomość alertu
        alert_message = f"🔥 TREND MODE ALERT: {symbol}\n"
        alert_message += f"📊 Score: {score}/100 ({reason})\n"
        alert_message += f"🎯 Aktywne moduły: {', '.join(active_modules)}\n"
        
        # Dodaj szczegóły pullback jeśli aktywny
        if "pullback_flow" in score_breakdown:
            alert_message += f"📉➡️📈 Pullback trigger aktywny\n"
        
        # Dodaj timestamp
        alert_message += f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        
        alert_data = {
            "symbol": symbol,
            "alert_type": "trend_mode",
            "score": score,
            "reason": reason,
            "message": alert_message,
            "score_breakdown": score_breakdown,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": self._calculate_priority(score, active_modules)
        }
        
        return alert_data
    
    def _calculate_priority(self, score: int, active_modules: List[str]) -> str:
        """Oblicza priorytet alertu"""
        if score >= 85 or len(active_modules) >= 4:
            return "HIGH"
        elif score >= 70 or len(active_modules) >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def process_trend_mode_2_alert(self, symbol: str, trend_mode_2_result: Dict) -> Optional[Dict]:
        """
        Główna funkcja przetwarzania alertów Trend Mode 2.0
        
        Args:
            symbol: Trading symbol
            trend_mode_2_result: Wynik z TrendMode2ScoreEngine.process_symbol_trend_mode_2()
            
        Returns:
            dict lub None: Alert data jeśli warunki spełnione
        """
        try:
            score = trend_mode_2_result.get('trend_mode_2_score', 0)
            alert_level = trend_mode_2_result.get('alert_level', 0)
            score_breakdown = trend_mode_2_result.get('score_breakdown', {})
            
            # Sprawdź warunki alertu z Trend Mode 2.0 trailing logic
            should_alert, reason = self.should_trigger_alert_trend_mode_2(
                symbol, score, alert_level, score_breakdown
            )
            
            # Aktualizuj historię
            details = {
                'alert_level': alert_level,
                'score_breakdown': score_breakdown,
                'trend_mode_2': True
            }
            self.update_score_history(symbol, score, details)
            
            # Generuj alert jeśli potrzeba
            if should_alert:
                alert_data = self.generate_trend_mode_2_alert(
                    symbol, score, alert_level, score_breakdown, reason
                )
                print(f"🚨 [TREND MODE 2.0] {symbol}: {reason}")
                return alert_data
            else:
                level_name = {0: "No Alert", 1: "Watchlist", 2: "Active Entry", 3: "Confirmed"}[alert_level]
                print(f"📊 [TREND MODE 2.0] {symbol}: {score}/100 - {level_name} - {reason}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error processing Trend Mode 2.0 alert for {symbol}: {e}")
            return None

    def process_trend_mode_alert(self, symbol: str, base_score: int, modules_data: Dict) -> Optional[Dict]:
        """
        Główna funkcja przetwarzania alertów Trend Mode
        
        Args:
            symbol: Trading symbol
            base_score: Bazowy trend score
            modules_data: Dane z modułów
            
        Returns:
            dict lub None: Alert data jeśli warunki spełnione
        """
        try:
            # 1. Oblicz wzmocniony score
            enhanced_score, score_breakdown = self.compute_enhanced_trend_score(
                symbol, base_score, modules_data
            )
            
            # 2. Sprawdź warunki alertu
            should_alert, reason = self.should_trigger_alert(symbol, enhanced_score, score_breakdown)
            
            # 3. Aktualizuj historię
            self.update_score_history(symbol, enhanced_score, score_breakdown)
            
            # 4. Generuj alert jeśli potrzeba
            if should_alert:
                alert_data = self.generate_trend_alert(symbol, enhanced_score, score_breakdown, reason)
                print(f"🚨 [TREND ALERT] {symbol}: {reason}")
                return alert_data
            else:
                print(f"📊 [TREND SCORE] {symbol}: {enhanced_score}/100 - {reason}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error processing trend mode alert for {symbol}: {e}")
            return None
    
    def get_trending_symbols(self, min_score: int = 60) -> List[Dict]:
        """
        Zwraca symbole z wysokim trend score
        
        Args:
            min_score: Minimalny score do uwzględnienia
            
        Returns:
            list: Lista symboli z ich scores
        """
        trending = []
        
        for symbol, history in self.score_history.items():
            if history:
                latest = history[-1]
                if latest["score"] >= min_score:
                    trending.append({
                        "symbol": symbol,
                        "score": latest["score"],
                        "timestamp": latest["timestamp"],
                        "active_modules": latest["details"].get("active_modules", [])
                    })
        
        # Sortuj po score (malejąco)
        trending.sort(key=lambda x: x["score"], reverse=True)
        return trending


# Globalna instancja dla użycia w całym systemie
trend_alert_engine = TrendModeAlertEngine()