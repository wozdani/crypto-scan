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
        self.cooldown_file = "/home/runner/workspace/crypto-scan/cache/pump_verification_cooldowns.json"
        self.cooldown_days = 7  # Don't verify same token within 7 days
        self.cleanup_days = 7   # Remove old explore data after 7 days
        
        # Ensure cache directory exists
        os.makedirs("/home/runner/workspace/crypto-scan/cache/explore_mode", exist_ok=True)
        
    def load_explore_data(self) -> List[Dict]:
        """Załaduj explore mode data from all files"""
        explore_data = []
        try:
            # FIXED: Use correct nested path where files are actually saved
            explore_dir_nested = "/home/runner/workspace/crypto-scan/crypto-scan/cache/explore_mode"
            explore_dir_simple = "/home/runner/workspace/crypto-scan/cache/explore_mode"
            explore_dir_relative = "crypto-scan/cache/explore_mode"
            
            # Try all possible paths to find where files are actually stored
            if os.path.exists(explore_dir_nested) and len(os.listdir(explore_dir_nested)) > 0:
                working_dir = explore_dir_nested
                print(f"[PUMP VERIFICATION DEBUG] Using nested path: {explore_dir_nested}")
            elif os.path.exists(explore_dir_simple) and len(os.listdir(explore_dir_simple)) > 0:
                working_dir = explore_dir_simple
                print(f"[PUMP VERIFICATION DEBUG] Using simple path: {explore_dir_simple}")
            elif os.path.exists(explore_dir_relative) and len(os.listdir(explore_dir_relative)) > 0:
                working_dir = explore_dir_relative
                print(f"[PUMP VERIFICATION DEBUG] Using relative path: {explore_dir_relative}")
            else:
                print(f"[PUMP VERIFICATION ERROR] No explore files found in any path:")
                print(f"  - Nested: {explore_dir_nested} (exists: {os.path.exists(explore_dir_nested)})")
                print(f"  - Simple: {explore_dir_simple} (exists: {os.path.exists(explore_dir_simple)})")
                print(f"  - Relative: {explore_dir_relative} (exists: {os.path.exists(explore_dir_relative)})")
                return []
            
            explore_dir = working_dir
            
            if not os.path.exists(explore_dir):
                print(f"[PUMP VERIFICATION] No explore directory found: {explore_dir}")
                return []
            
            files = os.listdir(explore_dir)
            # UPDATED: Accept both formats - legacy "_explore.json" and new "TOKEN_YYYYMMDD_HHMMSS_DETECTORS.json"
            json_files = [f for f in files if f.endswith("_explore.json") or (f.endswith(".json") and "_" in f and any(c.isdigit() for c in f))]
            print(f"[PUMP VERIFICATION DEBUG] Found {len(json_files)} explore files (legacy + enhanced format)")
            
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
    
    def load_cooldowns(self) -> Dict[str, str]:
        """Załaduj token cooldown timestamps"""
        try:
            with open(self.cooldown_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] Loading cooldowns: {e}")
            return {}
    
    def save_cooldowns(self, cooldowns: Dict[str, str]):
        """Zapisz token cooldown timestamps"""
        try:
            with open(self.cooldown_file, 'w') as f:
                json.dump(cooldowns, f, indent=2)
            print(f"[PUMP VERIFICATION] Updated cooldowns for {len(cooldowns)} tokens")
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] Saving cooldowns: {e}")
    
    def is_token_in_cooldown(self, symbol: str, cooldowns: Dict[str, str]) -> bool:
        """Sprawdź czy token jest w 7-day cooldown"""
        if symbol not in cooldowns:
            return False
        
        try:
            last_verification = datetime.fromisoformat(cooldowns[symbol])
            days_since = (datetime.now() - last_verification).days
            
            if days_since < self.cooldown_days:
                print(f"[PUMP VERIFICATION COOLDOWN] {symbol}: {days_since} days since last verification (need {self.cooldown_days})")
                return True
            else:
                print(f"[PUMP VERIFICATION COOLDOWN] {symbol}: {days_since} days passed - cooldown expired")
                return False
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] Checking cooldown for {symbol}: {e}")
            return False
    
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
        
        # Pump classification thresholds - Updated: >=5% = STRONG_PUMP
        if pump_percentage >= 5.0:  # >=5% = STRONG_PUMP
            pump_level = "STRONG_PUMP"
            agent_should_have_voted = "BUY"
        elif pump_percentage >= 2.0:
            pump_level = "WEAK_SIGNAL"
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
        cooldowns = self.load_cooldowns()
        
        # Get already verified symbols
        verified_symbols = {r["symbol"] + "_" + r["timestamp"] for r in verification_results}
        
        now = datetime.now()
        new_verifications = []
        cooldown_skipped = 0
        
        for entry in explore_data:
            entry_key = entry["symbol"] + "_" + entry["timestamp"]
            if entry_key in verified_symbols:
                continue  # Already verified
            
            # Check 7-day cooldown for this symbol
            if self.is_token_in_cooldown(entry["symbol"], cooldowns):
                cooldown_skipped += 1
                continue  # Skip - in cooldown period
            
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
                    
                    # Extract enhanced features from enriched explore data
                    enhanced_features = self.extract_enhanced_features(entry)
                    
                    verification = {
                        "symbol": entry["symbol"],
                        "timestamp": entry["timestamp"],
                        "alert_timestamp": entry.get("timestamp", entry.get("source_timestamp")),
                        "original_score": entry.get("stealth_score", entry.get("score", 0.0)),
                        "trigger_reason": entry.get("explore_reason", "unknown"),
                        "agents_decision": entry.get("consensus_decision", "UNKNOWN"),
                        "agents_votes": entry.get("agent_votes", {}),
                        "detectors": entry.get("active_signals", []),
                        "verification_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "hours_after": round(hours_passed, 1),
                        "price_data": price_data,
                        "pump_classification": pump_result,
                        "agents_accuracy": self.calculate_agent_accuracy(entry, pump_result),
                        "enhanced_features": enhanced_features,
                        "ai_detector_inputs": enhanced_features.get("ai_scores", {}),
                        "mastermind_patterns": enhanced_features.get("mastermind_data", {}),
                        "graph_analysis": enhanced_features.get("graph_features", {}),
                        "enriched_data_quality": enhanced_features.get("data_quality_score", 0.0)
                    }
                    
                    new_verifications.append(verification)
                    
                    # Update cooldown for this token
                    cooldowns[entry["symbol"]] = now.isoformat()
                    
                    # Enhanced success/fail logging with percentage
                    pump_percentage = pump_result['pump_percentage']
                    result_status = "SUCCESS" if pump_percentage >= 2.0 else "FAIL"
                    print(f"[EVALUATION RESULT] {entry['symbol']}: {pump_percentage:+.2f}% → {result_status}")
                    print(f"[PUMP VERIFICATION] {entry['symbol']}: {pump_result['pump_level']} ({pump_result['pump_percentage']:.1f}%)")
                else:
                    print(f"[PUMP VERIFICATION] Could not get price data for {entry['symbol']}")
        
        # Save updated cooldowns
        if new_verifications:
            self.save_cooldowns(cooldowns)
        
        # Log summary
        if cooldown_skipped > 0:
            print(f"[PUMP VERIFICATION COOLDOWN] Skipped {cooldown_skipped} tokens due to 7-day cooldown")
        
        print(f"[PUMP VERIFICATION] Processed {len(new_verifications)} new verifications")
        return new_verifications
    
    def cleanup_old_explore_data(self):
        """Usuń explore data starsze niż 7 dni"""
        try:
            explore_dir = "/home/runner/workspace/crypto-scan/cache/explore_mode"
            if not os.path.exists(explore_dir):
                return
            
            cutoff_date = datetime.now() - timedelta(days=self.cleanup_days)
            files_removed = 0
            
            files = os.listdir(explore_dir)
            json_files = [f for f in files if f.endswith("_explore.json")]
            
            for filename in json_files:
                filepath = os.path.join(explore_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        file_timestamp = datetime.fromisoformat(data.get('timestamp', ''))
                        
                        if file_timestamp < cutoff_date:
                            os.remove(filepath)
                            files_removed += 1
                            print(f"[CLEANUP] Removed old explore file: {filename} ({(datetime.now() - file_timestamp).days} days old)")
                            
                except Exception as file_error:
                    print(f"[CLEANUP ERROR] Processing {filename}: {file_error}")
            
            if files_removed > 0:
                print(f"[CLEANUP] Removed {files_removed} old explore files (older than {self.cleanup_days} days)")
            else:
                print(f"[CLEANUP] No old explore files to remove")
                
        except Exception as e:
            print(f"[CLEANUP ERROR] During explore data cleanup: {e}")
    
    def cleanup_old_verification_results(self):
        """Usuń verification results starsze niż 30 dni żeby nie zabierały miejsca"""
        try:
            verification_results = self.load_verification_results()
            if not verification_results:
                return
            
            cutoff_date = datetime.now() - timedelta(days=30)  # Keep verification results for 30 days
            original_count = len(verification_results)
            
            # Filter out old results
            filtered_results = []
            for result in verification_results:
                try:
                    verification_time = datetime.strptime(result.get('verification_time', ''), '%Y-%m-%d %H:%M:%S')
                    if verification_time >= cutoff_date:
                        filtered_results.append(result)
                except:
                    # Keep results with invalid timestamp format
                    filtered_results.append(result)
            
            removed_count = original_count - len(filtered_results)
            
            if removed_count > 0:
                self.save_verification_results(filtered_results)
                print(f"[CLEANUP] Removed {removed_count} old verification results (older than 30 days)")
            else:
                print(f"[CLEANUP] No old verification results to remove")
                
        except Exception as e:
            print(f"[CLEANUP ERROR] During verification results cleanup: {e}")

    def run_verification_cycle(self) -> List[Dict]:
        """Main verification cycle - sprawdź explore alerts i update agent learning"""
        try:
            print(f"[PUMP VERIFICATION] === Starting verification cycle at {datetime.now()} ===")
            
            # First cleanup old data
            self.cleanup_old_explore_data()       # Remove old explore files (7+ days)
            self.cleanup_old_verification_results()  # Remove old verification results (30+ days)
            
            # Get new verifications
            new_verifications = self.verify_pending_alerts()
            
            if not new_verifications:
                print("[PUMP VERIFICATION] No new pumps ready for verification")
                return []
            
            # Load existing results
            all_results = self.load_verification_results()
            
            # Add new verifications to results
            all_results.extend(new_verifications)
            
            # Save updated results
            self.save_verification_results(all_results)
            
            # CRITICAL FIX: Update agent learning with new verification results
            self.update_agent_learning(new_verifications)
            
            print(f"[PUMP VERIFICATION] === Completed cycle: {len(new_verifications)} new verifications ===")
            print(f"[PUMP VERIFICATION] Agent learning updated with {len(new_verifications)} new results")
            return new_verifications
            
        except Exception as e:
            print(f"[PUMP VERIFICATION ERROR] During verification cycle: {e}")
            return []
    
    def extract_enhanced_features(self, entry: Dict) -> Dict:
        """Extract enhanced features from enriched explore data dla lepszego agent learning"""
        enhanced_features = {
            "ai_scores": {},
            "mastermind_data": {},
            "graph_features": {},
            "data_quality_score": 0.0,
            "feature_count": 0
        }
        
        try:
            # Extract confidence sources (AI detector scores)
            if "confidence_sources" in entry:
                cs = entry["confidence_sources"]
                enhanced_features["ai_scores"] = {
                    "diamondwhale_ai": cs.get("diamondwhale_ai", 0.0),
                    "californiumwhale_ai": cs.get("californiumwhale_ai", 0.0),
                    "whale_clip": cs.get("whale_clip", 0.0),
                    "mastermind_tracing": cs.get("mastermind_tracing", 0.0)
                }
                enhanced_features["feature_count"] += len([v for v in enhanced_features["ai_scores"].values() if v > 0])
            
            # Extract detector scores
            if "detector_scores" in entry:
                ds = entry["detector_scores"]
                enhanced_features["detector_scores"] = ds
                enhanced_features["feature_count"] += len([v for v in ds.values() if v > 0])
            
            # Extract mastermind tracing data
            if "mastermind_tracing" in entry:
                mt = entry["mastermind_tracing"]
                if isinstance(mt, dict):
                    enhanced_features["mastermind_data"] = {
                        "sequence_detected": mt.get("sequence_detected", False),
                        "sequence_length": mt.get("sequence_length", 0),
                        "confidence": mt.get("confidence", 0.0),
                        "actors_count": len(mt.get("actors", []))
                    }
                    if mt.get("sequence_detected", False):
                        enhanced_features["feature_count"] += 1
            
            # Extract graph features
            if "graph_features" in entry:
                gf = entry["graph_features"]
                enhanced_features["graph_features"] = {
                    "accumulation_subgraph_score": gf.get("accumulation_subgraph_score", 0.0),
                    "temporal_pattern_shift": gf.get("temporal_pattern_shift", 0.0),
                    "whale_loop_probability": gf.get("whale_loop_probability", 0.0),
                    "unique_addresses": gf.get("unique_addresses", 0),
                    "avg_tx_interval_seconds": gf.get("avg_tx_interval_seconds", 0.0)
                }
                enhanced_features["feature_count"] += len([v for v in gf.values() if isinstance(v, (int, float)) and v > 0])
            
            # Calculate overall data quality score
            max_features = 20  # Estimated max features available
            enhanced_features["data_quality_score"] = min(enhanced_features["feature_count"] / max_features, 1.0)
            
            print(f"[ENHANCED FEATURES] {entry.get('symbol', 'UNKNOWN')}: {enhanced_features['feature_count']} features, quality: {enhanced_features['data_quality_score']:.2f}")
            
        except Exception as e:
            print(f"[ENHANCED FEATURES ERROR] {entry.get('symbol', 'UNKNOWN')}: {e}")
        
        return enhanced_features
    
    def calculate_agent_accuracy(self, entry: Dict, pump_result: Dict) -> Dict:
        """Oblicz accuracy każdego agenta dla tego prediction z ULTRA-FIXED logiką"""
        agents_decision = entry.get("consensus_decision", entry.get("agents_decision", "NO_CONSENSUS"))
        should_have_voted = pump_result["agent_should_have_voted"]
        pump_percentage = pump_result["pump_percentage"]
        
        # ULTRA-FIXED accuracy logic - akceptuj UNKNOWN/WATCH jako valid dla NO_PUMP
        correct_decision = False
        
        if pump_percentage >= 2.0:  # Was a pump (≥2%)
            # Should have been BUY - correct if agents voted BUY
            correct_decision = agents_decision == "BUY"
        elif pump_percentage >= -2.0:  # No significant movement
            # Should have been HOLD - correct if NOT BUY (HOLD/NO_CONSENSUS/UNKNOWN/WATCH all OK)
            correct_decision = agents_decision in ["HOLD", "NO_CONSENSUS", "UNKNOWN", "WATCH", "SKIP"]
        else:  # Was a dump (<-2%)
            # Should have been AVOID - correct if NOT BUY
            correct_decision = agents_decision in ["AVOID", "HOLD", "UNKNOWN", "WATCH", "SKIP"]
        
        return {
            "agents_voted": agents_decision,
            "should_have_voted": should_have_voted,
            "correct_decision": correct_decision,
            "accuracy_score": 1.0 if correct_decision else 0.0,
            "pump_percentage": pump_percentage,
            "accuracy_logic": f"pump={pump_percentage:.1f}%, agents={agents_decision}, correct={correct_decision}"
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
        
        # Update statistics with enhanced learning
        enhanced_patterns = {"ai_detector_patterns": {}, "mastermind_patterns": {}, "graph_patterns": {}}
        
        for verification in verifications:
            learning_history["total_verifications"] += 1
            correct = verification["agents_accuracy"]["correct_decision"]
            
            if correct:
                learning_history["correct_predictions"] += 1
            
            # Update by pump level
            pump_level = verification["pump_classification"]["pump_level"]
            if pump_level not in learning_history["by_pump_level"]:
                learning_history["by_pump_level"][pump_level] = {"total": 0, "correct": 0}
            
            learning_history["by_pump_level"][pump_level]["total"] += 1
            if correct:
                learning_history["by_pump_level"][pump_level]["correct"] += 1
            
            # Update by detector
            for detector in verification["detectors"]:
                if detector not in learning_history["by_detector"]:
                    learning_history["by_detector"][detector] = {"total": 0, "correct": 0}
                
                learning_history["by_detector"][detector]["total"] += 1
                if correct:
                    learning_history["by_detector"][detector]["correct"] += 1
            
            # Enhanced learning - AI detector patterns
            enhanced_features = verification.get("enhanced_features", {})
            ai_scores = enhanced_features.get("ai_scores", {})
            
            for ai_detector, score in ai_scores.items():
                if score > 0:  # Only track active AI detectors
                    if ai_detector not in enhanced_patterns["ai_detector_patterns"]:
                        enhanced_patterns["ai_detector_patterns"][ai_detector] = {
                            "total": 0, "correct": 0, "score_sum": 0.0, "pump_correlations": {}
                        }
                    
                    enhanced_patterns["ai_detector_patterns"][ai_detector]["total"] += 1
                    enhanced_patterns["ai_detector_patterns"][ai_detector]["score_sum"] += score
                    
                    if correct:
                        enhanced_patterns["ai_detector_patterns"][ai_detector]["correct"] += 1
                    
                    # Track pump level correlations
                    if pump_level not in enhanced_patterns["ai_detector_patterns"][ai_detector]["pump_correlations"]:
                        enhanced_patterns["ai_detector_patterns"][ai_detector]["pump_correlations"][pump_level] = {"total": 0, "correct": 0}
                    
                    enhanced_patterns["ai_detector_patterns"][ai_detector]["pump_correlations"][pump_level]["total"] += 1
                    if correct:
                        enhanced_patterns["ai_detector_patterns"][ai_detector]["pump_correlations"][pump_level]["correct"] += 1
            
            # Enhanced learning - Mastermind patterns
            mastermind_data = enhanced_features.get("mastermind_data", {})
            if mastermind_data.get("sequence_detected", False):
                seq_length = mastermind_data.get("sequence_length", 0)
                confidence = mastermind_data.get("confidence", 0.0)
                
                pattern_key = f"seq_length_{seq_length}"
                if pattern_key not in enhanced_patterns["mastermind_patterns"]:
                    enhanced_patterns["mastermind_patterns"][pattern_key] = {
                        "total": 0, "correct": 0, "confidence_sum": 0.0
                    }
                
                enhanced_patterns["mastermind_patterns"][pattern_key]["total"] += 1
                enhanced_patterns["mastermind_patterns"][pattern_key]["confidence_sum"] += confidence
                if correct:
                    enhanced_patterns["mastermind_patterns"][pattern_key]["correct"] += 1
            
            # Enhanced learning - Graph analysis patterns
            graph_features = enhanced_features.get("graph_features", {})
            if graph_features:
                whale_loop_prob = graph_features.get("whale_loop_probability", 0.0)
                accumulation_score = graph_features.get("accumulation_subgraph_score", 0.0)
                
                # Categorize graph patterns
                if whale_loop_prob > 0.8:
                    pattern_type = "high_whale_loop"
                elif accumulation_score > 0.7:
                    pattern_type = "high_accumulation"
                else:
                    pattern_type = "standard_graph"
                
                if pattern_type not in enhanced_patterns["graph_patterns"]:
                    enhanced_patterns["graph_patterns"][pattern_type] = {"total": 0, "correct": 0}
                
                enhanced_patterns["graph_patterns"][pattern_type]["total"] += 1
                if correct:
                    enhanced_patterns["graph_patterns"][pattern_type]["correct"] += 1
            
            # Add learning update entry
            learning_update = {
                "timestamp": verification["verification_time"],
                "symbol": verification["symbol"],
                "pump_level": pump_level,
                "agents_accuracy": verification["agents_accuracy"]["accuracy_score"],
                "lesson_learned": self.generate_lesson(verification)
            }
            learning_history["learning_updates"].append(learning_update)
        
        # Save enhanced patterns to learning history
        learning_history["enhanced_patterns"] = enhanced_patterns
        
        # Calculate overall accuracy
        if learning_history["total_verifications"] > 0:
            learning_history["overall_accuracy"] = learning_history["correct_predictions"] / learning_history["total_verifications"]
        
        # Calculate enhanced statistics
        learning_history["enhanced_statistics"] = self.calculate_enhanced_statistics(enhanced_patterns)
        
        # Add accuracy trend analysis
        recent_verifications = learning_history["learning_updates"][-10:] if len(learning_history["learning_updates"]) >= 10 else learning_history["learning_updates"]
        if recent_verifications:
            recent_accuracy = sum(v["agents_accuracy"] for v in recent_verifications) / len(recent_verifications)
            learning_history["recent_accuracy_trend"] = f"Last {len(recent_verifications)} predictions: {recent_accuracy:.1%}"
        
        # Keep only last 1000 learning updates
        if len(learning_history["learning_updates"]) > 1000:
            learning_history["learning_updates"] = learning_history["learning_updates"][-1000:]
        
        # Save updated learning history
        try:
            with open(self.agent_learning_file, 'w') as f:
                json.dump(learning_history, f, indent=2)
            print(f"[AGENT LEARNING] Updated with {len(verifications)} new verifications")
            print(f"[AGENT LEARNING] Overall accuracy: {learning_history['overall_accuracy']:.1%} ({learning_history['correct_predictions']}/{learning_history['total_verifications']})")
            
            # Print enhanced learning insights
            enhanced_stats = learning_history.get("enhanced_statistics", {})
            if enhanced_stats:
                print(f"[ENHANCED LEARNING] Best AI detector: {enhanced_stats.get('best_ai_detector', 'None')}")
                print(f"[ENHANCED LEARNING] Best pattern type: {enhanced_stats.get('best_pattern_type', 'None')}")
                
        except Exception as e:
            print(f"[AGENT LEARNING ERROR] Saving learning history: {e}")
    
    def calculate_enhanced_statistics(self, enhanced_patterns: Dict) -> Dict:
        """Calculate enhanced statistics from patterns for better agent insights"""
        stats = {
            "best_ai_detector": "none",
            "best_ai_accuracy": 0.0,
            "best_pattern_type": "none", 
            "best_pattern_accuracy": 0.0,
            "total_enhanced_features": 0,
            "ai_detector_rankings": {}
        }
        
        try:
            # Analyze AI detector performance
            ai_patterns = enhanced_patterns.get("ai_detector_patterns", {})
            for detector, data in ai_patterns.items():
                if data["total"] > 0:
                    accuracy = data["correct"] / data["total"]
                    avg_score = data["score_sum"] / data["total"]
                    
                    stats["ai_detector_rankings"][detector] = {
                        "accuracy": accuracy,
                        "avg_score": avg_score,
                        "total_predictions": data["total"]
                    }
                    
                    if accuracy > stats["best_ai_accuracy"] and data["total"] >= 3:  # Min 3 predictions for reliability
                        stats["best_ai_detector"] = detector
                        stats["best_ai_accuracy"] = accuracy
            
            # Analyze pattern types
            pattern_types = ["mastermind_patterns", "graph_patterns"]
            for pattern_type in pattern_types:
                patterns = enhanced_patterns.get(pattern_type, {})
                for pattern_name, data in patterns.items():
                    if data["total"] > 0:
                        accuracy = data["correct"] / data["total"]
                        if accuracy > stats["best_pattern_accuracy"] and data["total"] >= 2:
                            stats["best_pattern_type"] = f"{pattern_type}:{pattern_name}"
                            stats["best_pattern_accuracy"] = accuracy
            
            # Count total enhanced features used
            stats["total_enhanced_features"] = sum(
                len(enhanced_patterns.get(pattern_type, {})) 
                for pattern_type in ["ai_detector_patterns", "mastermind_patterns", "graph_patterns"]
            )
            
        except Exception as e:
            print(f"[ENHANCED STATS ERROR] {e}")
        
        return stats
    
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

def main():
    """Main verification function"""
    verifier = PumpVerificationSystem()
    return verifier.run_verification_cycle()

if __name__ == "__main__":
    main()