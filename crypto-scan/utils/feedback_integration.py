"""
Feedback Integration Module
Automatyzuje proces logowania wyników alertów i uruchamiania feedback loop
Integruje alert system z ewaluacją +6h i zapisem do feedback_results.json
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import requests
from utils.log_feedback_result import log_feedback_result

# File paths
PENDING_ALERTS_FILE = "data/pending_feedback_alerts.json"
FEEDBACK_RESULTS_FILE = "data/feedback_results.json"

class FeedbackIntegration:
    """Manages automatic feedback collection for TJDE alerts"""
    
    def __init__(self):
        self.pending_alerts = self._load_pending_alerts()
        
    def _load_pending_alerts(self) -> List[Dict]:
        """Load pending alerts waiting for evaluation"""
        if not os.path.exists(PENDING_ALERTS_FILE):
            return []
        
        try:
            with open(PENDING_ALERTS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_pending_alerts(self):
        """Save pending alerts to file"""
        os.makedirs(os.path.dirname(PENDING_ALERTS_FILE), exist_ok=True)
        try:
            with open(PENDING_ALERTS_FILE, "w") as f:
                json.dump(self.pending_alerts, f, indent=2)
        except Exception as e:
            print(f"[FEEDBACK INTEGRATION ERROR] Failed to save pending alerts: {e}")
    
    def record_alert_for_feedback(
        self, 
        symbol: str, 
        score_components: Dict[str, float], 
        phase: str, 
        alert_time: str = None,
        entry_price: float = None,
        additional_data: Dict[str, Any] = None
    ):
        """
        Records a new TJDE alert for later feedback evaluation
        
        Args:
            symbol: Trading symbol (e.g., PEPEUSDT)
            score_components: Component scores from simulate_trader_decision_advanced
            phase: Market phase (pre-pump, trend, breakout, consolidation)
            alert_time: ISO timestamp of alert
            entry_price: Entry price at alert time
            additional_data: Additional context data
        """
        
        alert_record = {
            "symbol": symbol,
            "score_components": score_components,
            "phase": phase,
            "alert_time": alert_time or datetime.now(timezone.utc).isoformat(),
            "entry_price": entry_price,
            "evaluation_time": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
            "status": "pending",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "additional_data": additional_data or {}
        }
        
        # Add to pending alerts
        self.pending_alerts.append(alert_record)
        self._save_pending_alerts()
        
        print(f"[FEEDBACK INTEGRATION] Recorded alert for {symbol} - evaluation at +6h")
        print(f"[FEEDBACK INTEGRATION] Entry price: {entry_price}, Phase: {phase}")
        
    def evaluate_pending_alerts(self) -> int:
        """
        Evaluates all pending alerts that are due for +6h evaluation
        
        Returns:
            Number of alerts evaluated
        """
        current_time = datetime.now(timezone.utc)
        evaluated_count = 0
        remaining_alerts = []
        
        for alert in self.pending_alerts:
            try:
                evaluation_time = datetime.fromisoformat(alert["evaluation_time"].replace('Z', '+00:00'))
                
                if current_time >= evaluation_time:
                    # Time for evaluation
                    success = self._evaluate_single_alert(alert)
                    if success:
                        evaluated_count += 1
                        print(f"[FEEDBACK EVALUATION] Completed: {alert['symbol']}")
                    else:
                        # Keep for retry later
                        remaining_alerts.append(alert)
                        print(f"[FEEDBACK EVALUATION] Failed: {alert['symbol']} - will retry")
                else:
                    # Not yet due for evaluation
                    remaining_alerts.append(alert)
                    
            except Exception as e:
                print(f"[FEEDBACK EVALUATION ERROR] {alert.get('symbol', 'unknown')}: {e}")
                # Keep problematic alerts for manual review
                alert["error"] = str(e)
                remaining_alerts.append(alert)
        
        # Update pending alerts
        self.pending_alerts = remaining_alerts
        self._save_pending_alerts()
        
        if evaluated_count > 0:
            print(f"[FEEDBACK INTEGRATION] Evaluated {evaluated_count} alerts, {len(remaining_alerts)} remaining")
        
        return evaluated_count
    
    def _evaluate_single_alert(self, alert: Dict) -> bool:
        """
        Evaluate single alert performance and log feedback result
        
        Args:
            alert: Alert record to evaluate
            
        Returns:
            True if evaluation successful, False otherwise
        """
        try:
            symbol = alert["symbol"]
            entry_price = alert.get("entry_price")
            
            if not entry_price:
                print(f"[FEEDBACK EVALUATION] {symbol}: No entry price recorded")
                return False
            
            # Get current price
            current_price = self._fetch_current_price(symbol)
            if not current_price:
                print(f"[FEEDBACK EVALUATION] {symbol}: Could not fetch current price")
                return False
            
            # Calculate performance
            profit_loss_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Determine success criteria (adjust based on phase)
            phase = alert.get("phase", "unknown")
            success_threshold = self._get_success_threshold(phase)
            was_successful = profit_loss_pct >= success_threshold
            
            # Log feedback result
            log_feedback_result(
                symbol=symbol,
                score_components=alert["score_components"],
                phase=phase,
                was_successful=was_successful,
                entry_price=entry_price,
                exit_price=current_price,
                profit_loss_pct=profit_loss_pct,
                alert_time=alert["alert_time"],
                additional_data={
                    **alert.get("additional_data", {}),
                    "evaluation_hours": 6,
                    "success_threshold": success_threshold,
                    "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            print(f"[FEEDBACK RESULT] {symbol}: {'SUCCESS' if was_successful else 'FAILURE'} "
                  f"({profit_loss_pct:+.2f}% vs {success_threshold:+.2f}% threshold)")
            
            return True
            
        except Exception as e:
            print(f"[FEEDBACK EVALUATION ERROR] {alert.get('symbol', 'unknown')}: {e}")
            return False
    
    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetch current price using Bybit API
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if failed
        """
        try:
            # Use Bybit API to get current price
            url = f"https://api.bybit.com/v5/market/tickers"
            params = {
                "category": "spot",
                "symbol": symbol
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                ticker = data["result"]["list"][0]
                return float(ticker.get("lastPrice", 0))
            
            print(f"[PRICE FETCH] {symbol}: No price data in API response")
            return None
            
        except Exception as e:
            print(f"[PRICE FETCH ERROR] {symbol}: {e}")
            return None
    
    def _get_success_threshold(self, phase: str) -> float:
        """
        Get success threshold based on market phase
        
        Args:
            phase: Market phase
            
        Returns:
            Success threshold percentage
        """
        thresholds = {
            "pre-pump": 3.0,      # 3% for pre-pump alerts
            "trend": 2.0,         # 2% for trend-following
            "breakout": 4.0,      # 4% for breakout alerts
            "consolidation": 1.5, # 1.5% for consolidation
            "unknown": 2.0        # Default 2%
        }
        
        return thresholds.get(phase, 2.0)
    
    def get_pending_count(self) -> int:
        """Get number of pending alerts"""
        return len(self.pending_alerts)
    
    def get_due_count(self) -> int:
        """Get number of alerts due for evaluation"""
        current_time = datetime.now(timezone.utc)
        due_count = 0
        
        for alert in self.pending_alerts:
            try:
                evaluation_time = datetime.fromisoformat(alert["evaluation_time"].replace('Z', '+00:00'))
                if current_time >= evaluation_time:
                    due_count += 1
            except Exception:
                continue
                
        return due_count
    
    def cleanup_old_pending_alerts(self, days_old: int = 7):
        """
        Remove pending alerts older than specified days
        
        Args:
            days_old: Days after which to remove old alerts
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_old)
        original_count = len(self.pending_alerts)
        
        self.pending_alerts = [
            alert for alert in self.pending_alerts
            if datetime.fromisoformat(alert.get("recorded_at", "1970-01-01T00:00:00+00:00").replace('Z', '+00:00')) >= cutoff_time
        ]
        
        removed_count = original_count - len(self.pending_alerts)
        if removed_count > 0:
            self._save_pending_alerts()
            print(f"[FEEDBACK CLEANUP] Removed {removed_count} old pending alerts")

# Global instance for easy integration
feedback_integration = FeedbackIntegration()

def record_tjde_alert_for_feedback(
    symbol: str, 
    tjde_result: Dict, 
    market_data: Dict = None
) -> bool:
    """
    Convenience function to record TJDE alert for feedback evaluation
    
    Args:
        symbol: Trading symbol
        tjde_result: Result from TJDE analysis with score_components
        market_data: Market data with current price
        
    Returns:
        True if successfully recorded
    """
    try:
        # Extract data from TJDE result
        score_components = tjde_result.get("score_components", {})
        phase = tjde_result.get("market_phase", "unknown")
        
        # Get entry price from market data
        entry_price = None
        if market_data and "price_usd" in market_data:
            entry_price = float(market_data["price_usd"])
        elif market_data and "candles_15m" in market_data:
            candles = market_data["candles_15m"]
            if candles and len(candles) > 0:
                # Use close price of last candle
                entry_price = float(candles[-1][4] if isinstance(candles[-1], list) else candles[-1].get("close", 0))
        
        # Record alert
        feedback_integration.record_alert_for_feedback(
            symbol=symbol,
            score_components=score_components,
            phase=phase,
            entry_price=entry_price,
            additional_data={
                "tjde_score": tjde_result.get("final_score", 0),
                "tjde_decision": tjde_result.get("decision", "unknown"),
                "confidence": tjde_result.get("confidence", 0)
            }
        )
        
        return True
        
    except Exception as e:
        print(f"[FEEDBACK RECORD ERROR] {symbol}: {e}")
        return False

def run_feedback_evaluation():
    """Run feedback evaluation for all due alerts"""
    try:
        evaluated = feedback_integration.evaluate_pending_alerts()
        
        # If we evaluated alerts, trigger feedback loop learning
        if evaluated > 0:
            try:
                from feedback_loop import TJDEFeedbackLoop
                feedback_loop = TJDEFeedbackLoop()
                
                # Run learning for all phases
                phases = ["pre-pump", "trend", "breakout", "consolidation"]
                for phase in phases:
                    try:
                        feedback_loop.learn_from_feedback(phase)
                        print(f"[FEEDBACK LEARNING] Updated weights for {phase} phase")
                    except Exception as e:
                        print(f"[FEEDBACK LEARNING ERROR] {phase}: {e}")
                        
            except Exception as e:
                print(f"[FEEDBACK LOOP ERROR] Failed to run learning: {e}")
        
        return evaluated
        
    except Exception as e:
        print(f"[FEEDBACK EVALUATION ERROR] {e}")
        return 0

def get_feedback_stats():
    """Get feedback system statistics"""
    return {
        "pending_alerts": feedback_integration.get_pending_count(),
        "due_for_evaluation": feedback_integration.get_due_count(),
        "total_feedback_results": len(feedback_integration._load_pending_alerts())
    }

# Auto-cleanup on import
try:
    feedback_integration.cleanup_old_pending_alerts()
except Exception:
    pass