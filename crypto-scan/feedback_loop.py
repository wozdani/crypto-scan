#!/usr/bin/env python3
"""
TJDE Feedback Loop System - Adaptive Learning Engine
System uczący TJDE na podstawie skuteczności alertów i decyzji tradingowych.

Główne funkcje:
- Zapisuje wynik każdego alertu (score, decision, phase, result_after_n_hours)
- Porównuje przewidywania z faktycznym wynikiem rynku
- Zwiększa wagę komponentów z trafnych alertów
- Redukuje wagę komponentów z fałszywych alertów
- Aktualizuje profile scoringu automatycznie
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

# Configuration
PROFILE_DIR = "data/weights/"
FEEDBACK_LOG = "data/feedback_results.json"
PERFORMANCE_LOG = "data/performance_history.json"
BACKUP_DIR = "data/backups/"
LEARNING_RATE = 0.03  # Jak szybko system adaptuje
MIN_SAMPLES = 10  # Minimum alertów przed adaptacją
PERFORMANCE_THRESHOLD = 0.60  # 60% success rate minimum

class TJDEFeedbackLoop:
    """
    Adaptive learning system for TJDE engine
    Learns from trading results and optimizes component weights
    """
    
    def __init__(self):
        self.ensure_directories()
        self.feedback_data = self.load_feedback_data()
        self.performance_history = self.load_performance_history()
        print("[FEEDBACK LOOP] TJDE Adaptive Learning System initialized")
    
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(PROFILE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        os.makedirs("data", exist_ok=True)
    
    def load_feedback_data(self) -> List[Dict]:
        """Load existing feedback data"""
        if os.path.exists(FEEDBACK_LOG):
            try:
                with open(FEEDBACK_LOG, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[FEEDBACK LOOP] Error loading feedback data: {e}")
                return []
        return []
    
    def load_performance_history(self) -> List[Dict]:
        """Load performance history"""
        if os.path.exists(PERFORMANCE_LOG):
            try:
                with open(PERFORMANCE_LOG, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[FEEDBACK LOOP] Error loading performance history: {e}")
                return []
        return []
    
    def save_feedback_data(self):
        """Save feedback data to file"""
        try:
            with open(FEEDBACK_LOG, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            print(f"[FEEDBACK LOOP] Error saving feedback data: {e}")
    
    def save_performance_history(self):
        """Save performance history"""
        try:
            with open(PERFORMANCE_LOG, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            print(f"[FEEDBACK LOOP] Error saving performance history: {e}")
    
    def record_alert_result(
        self, 
        symbol: str, 
        phase: str, 
        score: float,
        decision: str,
        score_components: Dict[str, float],
        entry_price: float,
        alert_time: str,
        was_successful: Optional[bool] = None,
        exit_price: Optional[float] = None,
        profit_loss_pct: Optional[float] = None,
        hours_tracked: int = 2
    ):
        """
        Record trading alert result for learning
        
        Args:
            symbol: Trading symbol
            phase: Market phase (pre-pump, trend, breakout, consolidation)
            score: Final TJDE score
            decision: TJDE decision (enter, avoid, scalp_entry, wait)
            score_components: Individual component scores
            entry_price: Price when alert triggered
            alert_time: ISO timestamp of alert
            was_successful: True if profitable, False if loss, None if pending
            exit_price: Exit price (if trade completed)
            profit_loss_pct: Profit/loss percentage
            hours_tracked: Hours to track performance (default 2h)
        """
        
        feedback_entry = {
            "symbol": symbol,
            "phase": phase,
            "score": score,
            "decision": decision,
            "score_components": score_components,
            "entry_price": entry_price,
            "alert_time": alert_time,
            "was_successful": was_successful,
            "exit_price": exit_price,
            "profit_loss_pct": profit_loss_pct,
            "hours_tracked": hours_tracked,
            "recorded_at": datetime.now().isoformat(),
            "status": "pending" if was_successful is None else "completed"
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback_data()
        
        print(f"[FEEDBACK LOOP] Recorded alert result: {symbol} {phase} {decision} ({'SUCCESS' if was_successful else 'FAILURE' if was_successful is not None else 'PENDING'})")
    
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetch current price for a symbol using Bybit API
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            
        Returns:
            Current price or None if failed
        """
        try:
            import requests
            
            # Use Bybit ticker API
            url = f"https://api.bybit.com/v5/market/tickers"
            params = {
                "category": "spot",
                "symbol": symbol
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    ticker = data["result"]["list"][0]
                    price = float(ticker.get("lastPrice", 0))
                    return price if price > 0 else None
                    
        except Exception as e:
            print(f"[FEEDBACK LOOP] Error fetching price for {symbol}: {e}")
        
        return None
    
    def update_pending_results(self):
        """
        Update pending results by checking current prices
        This should be called periodically to evaluate ongoing trades
        """
        
        updated_count = 0
        
        for entry in self.feedback_data:
            if entry["status"] == "pending":
                try:
                    # Check if tracking period has elapsed
                    alert_time = datetime.fromisoformat(entry["alert_time"])
                    hours_elapsed = (datetime.now() - alert_time).total_seconds() / 3600
                    
                    if hours_elapsed >= entry["hours_tracked"]:
                        # Get current price
                        current_price = self.fetch_current_price(entry["symbol"])
                        if current_price:
                            entry_price = entry["entry_price"]
                            price_change_pct = ((current_price - entry_price) / entry_price) * 100
                            
                            # Determine success based on decision type
                            if entry["decision"] in ["enter", "scalp_entry"]:
                                # For buy signals, success = positive price movement
                                was_successful = price_change_pct > 2.0  # 2% threshold
                            else:
                                # For avoid/wait signals, success = avoiding loss
                                was_successful = price_change_pct < -1.0  # Avoided >1% loss
                            
                            # Update entry
                            entry["exit_price"] = current_price
                            entry["profit_loss_pct"] = price_change_pct
                            entry["was_successful"] = was_successful
                            entry["status"] = "completed"
                            entry["completed_at"] = datetime.now().isoformat()
                            
                            updated_count += 1
                            print(f"[FEEDBACK LOOP] Updated {entry['symbol']}: {price_change_pct:.2f}% ({'SUCCESS' if was_successful else 'FAILURE'})")
                            
                except Exception as e:
                    print(f"[FEEDBACK LOOP] Error updating {entry.get('symbol', 'unknown')}: {e}")
        
        if updated_count > 0:
            self.save_feedback_data()
            print(f"[FEEDBACK LOOP] Updated {updated_count} pending results")
    
    def load_profile(self, profile_name: str) -> Dict[str, float]:
        """Load scoring profile"""
        profile_path = os.path.join(PROFILE_DIR, profile_name)
        try:
            with open(profile_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[FEEDBACK LOOP] Error loading profile {profile_name}: {e}")
            return {}
    
    def save_profile(self, profile_name: str, profile: Dict[str, float]):
        """Save scoring profile with backup"""
        profile_path = os.path.join(PROFILE_DIR, profile_name)
        
        # Create backup
        if os.path.exists(profile_path):
            backup_name = f"{profile_name}.backup_{int(time.time())}"
            backup_path = os.path.join(BACKUP_DIR, backup_name)
            try:
                with open(profile_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                print(f"[FEEDBACK LOOP] Created backup: {backup_name}")
            except Exception as e:
                print(f"[FEEDBACK LOOP] Backup failed: {e}")
        
        # Save new profile
        try:
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2)
            print(f"[FEEDBACK LOOP] Saved updated profile: {profile_name}")
        except Exception as e:
            print(f"[FEEDBACK LOOP] Error saving profile {profile_name}: {e}")
    
    def calculate_component_performance(self, decision_results: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate performance statistics for each component
        
        Returns:
            Dict with component performance metrics
        """
        if not decision_results:
            return {}
        
        stats = {}
        
        # Initialize stats for all components
        all_components = set()
        for result in decision_results:
            all_components.update(result.get("score_components", {}).keys())
        
        for component in all_components:
            stats[component] = {
                "tp_weight": 0.0,  # True positive weight sum
                "fp_weight": 0.0,  # False positive weight sum
                "tp_count": 0,     # True positive count
                "fp_count": 0,     # False positive count
                "success_rate": 0.0,
                "avg_contribution": 0.0
            }
        
        # Calculate statistics
        for result in decision_results:
            if result.get("was_successful") is None:
                continue  # Skip pending results
            
            components = result.get("score_components", {})
            is_successful = result["was_successful"]
            
            for component, weight in components.items():
                if is_successful:
                    stats[component]["tp_weight"] += weight
                    stats[component]["tp_count"] += 1
                else:
                    stats[component]["fp_weight"] += weight
                    stats[component]["fp_count"] += 1
        
        # Calculate derived metrics
        for component, data in stats.items():
            total_count = data["tp_count"] + data["fp_count"]
            total_weight = data["tp_weight"] + data["fp_weight"]
            
            if total_count > 0:
                data["success_rate"] = data["tp_count"] / total_count
                data["avg_contribution"] = total_weight / total_count
            
        return stats
    
    def apply_feedback(self, profile_name: str, decision_results: List[Dict]) -> bool:
        """
        Apply feedback learning to update profile weights
        
        Args:
            profile_name: Name of profile to update
            decision_results: List of completed trading results
            
        Returns:
            True if profile was updated, False otherwise
        """
        
        if len(decision_results) < MIN_SAMPLES:
            print(f"[FEEDBACK LOOP] Insufficient samples for {profile_name}: {len(decision_results)} < {MIN_SAMPLES}")
            return False
        
        # Calculate overall success rate
        completed_results = [r for r in decision_results if r.get("was_successful") is not None]
        if not completed_results:
            print(f"[FEEDBACK LOOP] No completed results for {profile_name}")
            return False
        
        success_rate = sum(1 for r in completed_results if r["was_successful"]) / len(completed_results)
        print(f"[FEEDBACK LOOP] Success rate for {profile_name}: {success_rate:.2%}")
        
        # Only apply learning if performance is reasonable
        if success_rate < 0.3:  # Less than 30% success - too low to learn from
            print(f"[FEEDBACK LOOP] Success rate too low for learning: {success_rate:.2%}")
            return False
        
        profile = self.load_profile(profile_name)
        if not profile:
            print(f"[FEEDBACK LOOP] Could not load profile: {profile_name}")
            return False
        
        # Calculate component performance
        component_stats = self.calculate_component_performance(completed_results)
        
        # Apply adaptive learning rate based on performance
        adaptive_rate = LEARNING_RATE
        if success_rate > 0.7:
            adaptive_rate *= 1.5  # Learn faster from good performance
        elif success_rate < 0.5:
            adaptive_rate *= 0.5  # Learn slower from poor performance
        
        print(f"[FEEDBACK LOOP] Applying learning rate: {adaptive_rate:.3f}")
        
        # Update weights based on component performance
        adjustments = {}
        for component in profile.keys():
            if component in component_stats:
                stats = component_stats[component]
                
                # Calculate adjustment based on success rate and contribution
                if stats["tp_count"] + stats["fp_count"] > 0:
                    # Weight adjustment: (success_rate - 0.5) * contribution * learning_rate
                    success_factor = (stats["success_rate"] - 0.5) * 2  # -1 to +1
                    contribution_factor = stats["avg_contribution"]
                    adjustment = success_factor * contribution_factor * adaptive_rate
                    
                    adjustments[component] = adjustment
                    new_weight = max(0.01, min(0.8, profile[component] + adjustment))  # Bounds: 1% to 80%
                    profile[component] = new_weight
                    
                    print(f"[FEEDBACK LOOP] {component}: {stats['success_rate']:.2%} success → {adjustment:+.3f} → {new_weight:.3f}")
        
        # Normalize weights to sum to 1.0
        total_weight = sum(profile.values())
        if total_weight > 0:
            for component in profile.keys():
                profile[component] /= total_weight
        
        # Save updated profile
        self.save_profile(profile_name, profile)
        
        # Record performance update
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "profile": profile_name,
            "samples": len(completed_results),
            "success_rate": success_rate,
            "adjustments": adjustments,
            "final_weights": profile.copy()
        })
        self.save_performance_history()
        
        return True
    
    def run_feedback_loop(self) -> Dict[str, Any]:
        """
        Main feedback loop execution
        
        Returns:
            Summary of learning results
        """
        print("[FEEDBACK LOOP] Starting adaptive learning cycle...")
        
        # Update pending results first
        self.update_pending_results()
        
        # Group results by phase
        completed_results = [r for r in self.feedback_data if r.get("was_successful") is not None]
        
        if not completed_results:
            print("[FEEDBACK LOOP] No completed results to learn from")
            return {"status": "no_data", "message": "No completed trading results available"}
        
        phase_groups = {}
        for result in completed_results:
            phase = result["phase"]
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(result)
        
        # Apply learning to each phase profile
        results = {}
        profile_mapping = {
            "pre-pump": "tjde_pre_pump_profile.json",
            "trend": "tjde_trend_following_profile.json", 
            "trend-following": "tjde_trend_following_profile.json",
            "consolidation": "tjde_consolidation_profile.json",
            "breakout": "tjde_breakout_profile.json"
        }
        
        total_updates = 0
        for phase, results_list in phase_groups.items():
            profile_name = profile_mapping.get(phase)
            if profile_name:
                updated = self.apply_feedback(profile_name, results_list)
                if updated:
                    total_updates += 1
                results[phase] = {
                    "samples": len(results_list),
                    "updated": updated,
                    "profile": profile_name
                }
            else:
                print(f"[FEEDBACK LOOP] No profile mapping for phase: {phase}")
        
        summary = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(completed_results),
            "profiles_updated": total_updates,
            "phase_results": results
        }
        
        print(f"[FEEDBACK LOOP] Learning cycle completed: {total_updates} profiles updated from {len(completed_results)} samples")
        return summary

def run_feedback_loop():
    """Standalone execution function"""
    feedback_system = TJDEFeedbackLoop()
    return feedback_system.run_feedback_loop()

def record_trading_result(
    symbol: str,
    phase: str, 
    score: float,
    decision: str,
    score_components: Dict[str, float],
    entry_price: float,
    was_successful: bool = None,
    profit_loss_pct: float = None
):
    """
    Convenience function to record trading result
    Can be called from other modules
    """
    feedback_system = TJDEFeedbackLoop()
    feedback_system.record_alert_result(
        symbol=symbol,
        phase=phase,
        score=score,
        decision=decision,
        score_components=score_components,
        entry_price=entry_price,
        alert_time=datetime.now().isoformat(),
        was_successful=was_successful,
        profit_loss_pct=profit_loss_pct
    )

if __name__ == "__main__":
    # Test/demo execution
    result = run_feedback_loop()
    print(f"[FEEDBACK LOOP] Execution result: {result}")