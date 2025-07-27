#!/usr/bin/env python3
"""
Pump Verification System dla Agent Learning
Sprawdza czy pumps rzeczywiście wystąpiły po explore mode alerts
Aktualizuje agent models na podstawie rzeczywistych wyników
"""

import json
import os
import sys
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional

# Add crypto-scan to path
sys.path.append('/home/runner/workspace/crypto-scan')

class PumpVerificationSystem:
    def __init__(self):
        self.explore_file = "/home/runner/workspace/crypto-scan/cache/explore_learning_data.json"
        self.verification_file = "/home/runner/workspace/crypto-scan/cache/pump_verification_results.json"
        self.agent_learning_file = "/home/runner/workspace/crypto-scan/cache/agent_learning_history.json"
        
    def load_explore_data(self) -> List[Dict]:
        """Załaduj explore mode data from all files"""
        explore_data = []
        try:
            explore_dir = "/home/runner/workspace/crypto-scan/cache/explore_mode"
            print(f"[PUMP VERIFICATION DEBUG] Looking for explore data in: {explore_dir}")
            
            if not os.path.exists(explore_dir):
                print(f"[PUMP VERIFICATION] No explore directory found: {explore_dir}")
                return []
            
            files = os.listdir(explore_dir)
            json_files = [f for f in files if f.endswith("_explore.json")]
            print(f"[PUMP VERIFICATION DEBUG] Found {len(json_files)} explore files")
            
            for filename in json_files:
                filepath = os.path.join(explore_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        # Check timestamp
                        timestamp = datetime.fromisoformat(data.get('timestamp', ''))
                        hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
                        print(f"[PUMP VERIFICATION DEBUG] {filename}: {hours_ago:.1f} hours ago")
                        explore_data.append(data)
                except Exception as file_error:
                    print(f"[PUMP VERIFICATION ERROR] Loading {filename}: {file_error}")
                        
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] Loading explore data: {e}")
            
        print(f"[PUMP VERIFICATION DEBUG] Loaded {len(explore_data)} explore records")
        return explore_data
    
    def load_verification_results(self) -> List[Dict]:
        """Załaduj previous verification results"""
        try:
            with open(self.verification_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] Loading verification results: {e}")
            return []
    
    def save_verification_results(self, results: List[Dict]):
        """Zapisz verification results"""
        try:
            with open(self.verification_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[PUMP VERIFICATION] Saved {len(results)} verification results")
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] Saving results: {e}")
    
    def get_price_data(self, symbol: str, start_time: datetime, hours_after: int = 6) -> Optional[Dict]:
        """Pobierz price data dla verification (6h po explore alert)"""
        try:
            # Użyj Bybit API dla historical price data
            end_time = start_time + timedelta(hours=hours_after)
            
            # Convert to timestamp
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": "15",  # 15min candles
                "start": start_ts,
                "end": end_ts,
                "limit": 24  # 6 hours of 15min candles
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    candles = data["result"]["list"]
                    
                    # Pierwszy candle (start price) i ostatni candle (end price)
                    start_price = float(candles[-1][1])  # open price pierwszego candle
                    end_price = float(candles[0][4])     # close price ostatniego candle
                    
                    # Maksymalna cena w tym okresie
                    max_price = max(float(candle[2]) for candle in candles)
                    
                    # Oblicz pump percentage
                    pump_percentage = ((max_price - start_price) / start_price) * 100
                    end_percentage = ((end_price - start_price) / start_price) * 100
                    
                    return {
                        "start_price": start_price,
                        "end_price": end_price,
                        "max_price": max_price,
                        "pump_percentage": pump_percentage,
                        "end_percentage": end_percentage,
                        "candles_count": len(candles),
                        "period_hours": hours_after
                    }
            
            print(f"[PUMP VERIFICATION] Failed to get price data for {symbol}: {response.status_code}")
            return None
            
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] Getting price data for {symbol}: {e}")
            return None
    
    def classify_pump_result(self, price_data: Dict, original_score: float) -> Dict:
        """Klasyfikuj czy był pump i jak strong"""
        pump_percentage = price_data["pump_percentage"]
        end_percentage = price_data["end_percentage"]
        
        # Pump classification thresholds
        if pump_percentage >= 10.0:
            pump_level = "STRONG_PUMP"
            agent_should_have_voted = "BUY"
        elif pump_percentage >= 5.0:
            pump_level = "MEDIUM_PUMP" 
            agent_should_have_voted = "BUY"
        elif pump_percentage >= 2.0:
            pump_level = "WEAK_PUMP"
            agent_should_have_voted = "BUY" if original_score > 2.0 else "HOLD"
        elif pump_percentage >= -2.0:
            pump_level = "NO_PUMP"
            agent_should_have_voted = "HOLD"
        else:
            pump_level = "DUMP"
            agent_should_have_voted = "AVOID"
        
        return {
            "pump_level": pump_level,
            "agent_should_have_voted": agent_should_have_voted,
            "pump_percentage": pump_percentage,
            "end_percentage": end_percentage,
            "sustained_pump": end_percentage > 1.0  # Czy pump był sustained
        }
    
    def verify_pending_alerts(self) -> List[Dict]:
        """Sprawdź pending explore alerts które są ready dla verification"""
        explore_data = self.load_explore_data()
        verification_results = self.load_verification_results()
        
        # Get already verified symbols
        verified_symbols = {r["symbol"] + "_" + r["timestamp"] for r in verification_results}
        
        now = datetime.now()
        new_verifications = []
        
        for entry in explore_data:
            entry_key = entry["symbol"] + "_" + entry["timestamp"]
            if entry_key in verified_symbols:
                continue  # Already verified
            
            # Check if 6+ hours passed
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
            except:
                continue  # Invalid timestamp format
                
            hours_passed = (now - entry_time).total_seconds() / 3600
            
            if hours_passed >= 6.0:  # 6+ hours passed - ready for verification
                print(f"[PUMP VERIFICATION] Verifying {entry['symbol']} from {entry['timestamp']} ({hours_passed:.1f}h ago)")
                
                # Get price data for verification
                price_data = self.get_price_data(entry["symbol"], entry_time, hours_after=6)
                
                if price_data:
                    # Classify pump result
                    pump_result = self.classify_pump_result(price_data, entry.get("stealth_score", entry.get("score", 0.0)))
                    
                    verification = {
                        "symbol": entry["symbol"],
                        "timestamp": entry["timestamp"],
                        "original_score": entry.get("stealth_score", entry.get("score", 0.0)),
                        "trigger_reason": entry.get("explore_reason", "unknown"),
                        "agents_decision": entry.get("consensus_decision", "UNKNOWN"),
                        "agents_votes": entry.get("agent_votes", {}),
                        "detectors": entry.get("active_signals", []),
                        "verification_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "hours_after": round(hours_passed, 1),
                        "price_data": price_data,
                        "pump_classification": pump_result,
                        "agents_accuracy": self.calculate_agent_accuracy(entry, pump_result)
                    }
                    
                    new_verifications.append(verification)
                    print(f"[PUMP VERIFICATION] {entry['symbol']}: {pump_result['pump_level']} ({pump_result['pump_percentage']:.1f}%)")
                else:
                    print(f"[PUMP VERIFICATION] Could not get price data for {entry['symbol']}")
        
        return new_verifications
    
    def calculate_agent_accuracy(self, entry: Dict, pump_result: Dict) -> Dict:
        """Oblicz accuracy każdego agenta dla tego prediction"""
        agents_decision = entry.get("agents_decision", "NO_CONSENSUS")
        should_have_voted = pump_result["agent_should_have_voted"]
        
        # Sprawdź czy agents decision był correct
        correct_decision = agents_decision == should_have_voted
        
        return {
            "agents_voted": agents_decision,
            "should_have_voted": should_have_voted,
            "correct_decision": correct_decision,
            "accuracy_score": 1.0 if correct_decision else 0.0
        }
    
    def update_agent_learning(self, verifications: List[Dict]):
        """Aktualizuj agent learning na podstawie verification results"""
        if not verifications:
            return
        
        # Load existing learning history
        try:
            with open(self.agent_learning_file, 'r') as f:
                learning_history = json.load(f)
        except FileNotFoundError:
            learning_history = {
                "total_verifications": 0,
                "correct_predictions": 0,
                "overall_accuracy": 0.0,
                "by_pump_level": {},
                "by_detector": {},
                "learning_updates": []
            }
        except Exception as e:
            print(f"[AGENT LEARNING ERROR] Loading history: {e}")
            return
        
        # Update statistics
        for verification in verifications:
            learning_history["total_verifications"] += 1
            
            if verification["agents_accuracy"]["correct_decision"]:
                learning_history["correct_predictions"] += 1
            
            # Update by pump level
            pump_level = verification["pump_classification"]["pump_level"]
            if pump_level not in learning_history["by_pump_level"]:
                learning_history["by_pump_level"][pump_level] = {"total": 0, "correct": 0}
            
            learning_history["by_pump_level"][pump_level]["total"] += 1
            if verification["agents_accuracy"]["correct_decision"]:
                learning_history["by_pump_level"][pump_level]["correct"] += 1
            
            # Update by detector
            for detector in verification["detectors"]:
                if detector not in learning_history["by_detector"]:
                    learning_history["by_detector"][detector] = {"total": 0, "correct": 0}
                
                learning_history["by_detector"][detector]["total"] += 1
                if verification["agents_accuracy"]["correct_decision"]:
                    learning_history["by_detector"][detector]["correct"] += 1
            
            # Add learning update entry
            learning_update = {
                "timestamp": verification["verification_time"],
                "symbol": verification["symbol"],
                "pump_level": pump_level,
                "agents_accuracy": verification["agents_accuracy"]["accuracy_score"],
                "lesson_learned": self.generate_lesson(verification)
            }
            learning_history["learning_updates"].append(learning_update)
        
        # Calculate overall accuracy
        if learning_history["total_verifications"] > 0:
            learning_history["overall_accuracy"] = learning_history["correct_predictions"] / learning_history["total_verifications"]
        
        # Keep only last 1000 learning updates
        if len(learning_history["learning_updates"]) > 1000:
            learning_history["learning_updates"] = learning_history["learning_updates"][-1000:]
        
        # Save updated learning history
        try:
            with open(self.agent_learning_file, 'w') as f:
                json.dump(learning_history, f, indent=2)
            print(f"[AGENT LEARNING] Updated with {len(verifications)} new verifications")
            print(f"[AGENT LEARNING] Overall accuracy: {learning_history['overall_accuracy']:.1%} ({learning_history['correct_predictions']}/{learning_history['total_verifications']})")
        except Exception as e:
            print(f"[AGENT LEARNING ERROR] Saving learning history: {e}")
    
    def generate_lesson(self, verification: Dict) -> str:
        """Generate lesson learned z verification"""
        symbol = verification["symbol"]
        pump_level = verification["pump_classification"]["pump_level"]
        agents_voted = verification["agents_accuracy"]["agents_voted"]
        should_have_voted = verification["agents_accuracy"]["should_have_voted"]
        correct = verification["agents_accuracy"]["correct_decision"]
        
        if correct:
            return f"✅ Correct: {symbol} agents voted {agents_voted}, pump was {pump_level}"
        else:
            return f"❌ Wrong: {symbol} agents voted {agents_voted}, should have voted {should_have_voted} (pump: {pump_level})"
    
    def run_verification_cycle(self):
        """Run complete verification cycle"""
        print("[PUMP VERIFICATION] Starting verification cycle...")
        
        # Verify pending alerts
        new_verifications = self.verify_pending_alerts()
        
        if new_verifications:
            # Save verification results
            existing_results = self.load_verification_results()
            all_results = existing_results + new_verifications
            self.save_verification_results(all_results)
            
            # Update agent learning
            self.update_agent_learning(new_verifications)
            
            print(f"[PUMP VERIFICATION] Completed verification of {len(new_verifications)} alerts")
        else:
            print("[PUMP VERIFICATION] No alerts ready for verification")
        
        return new_verifications

def main():
    """Main verification function"""
    verifier = PumpVerificationSystem()
    return verifier.run_verification_cycle()

if __name__ == "__main__":
    main()