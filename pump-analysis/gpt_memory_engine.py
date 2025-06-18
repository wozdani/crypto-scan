"""
GPT Memory Engine - Advanced Integration System

This module provides advanced GPT learning capabilities with:
- Function detector memory and analysis
- Crypto-scan service integration 
- Pattern recognition across pump events
- Meta-detector generation
- Performance tracking and optimization
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class GPTMemoryEngine:
    """Advanced GPT learning system with memory and cross-system integration"""
    
    def __init__(self):
        self.memory_file = "gpt_memory.json"
        self.detectors_dir = Path("detectors")
        self.crypto_scan_data_dir = Path("../crypto-scan/data")
        
        # Ensure directories exist
        self.detectors_dir.mkdir(exist_ok=True)
        
        # Initialize memory
        self.memory = self._load_memory()
        
        logger.info("🧠 GPT Memory Engine initialized")
    
    def _load_memory(self) -> Dict:
        """Load GPT memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading GPT memory: {e}")
        
        # Initialize empty memory structure
        return {
            "detectors": {},
            "pump_events": {},
            "crypto_scan_signals": {},
            "meta_patterns": {},
            "performance_metrics": {
                "total_detectors": 0,
                "successful_predictions": 0,
                "false_positives": 0,
                "accuracy_score": 0.0
            },
            "learning_insights": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _save_memory(self):
        """Save GPT memory to JSON file"""
        try:
            self.memory["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2, default=str)
            logger.info("💾 GPT memory saved successfully")
        except Exception as e:
            logger.error(f"Error saving GPT memory: {e}")
    
    def register_detector_function(self, symbol: str, date: str, function_code: str, 
                                 pump_data: Dict, pre_pump_analysis: Dict, 
                                 crypto_scan_signals: Optional[Dict] = None) -> str:
        """
        Register a new detector function with full context
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            date: Date in YYYYMMDD format
            function_code: Generated Python function code
            pump_data: Pump event details (price_increase, duration, etc.)
            pre_pump_analysis: Pre-pump technical analysis
            crypto_scan_signals: Real-time signals from crypto-scan if available
            
        Returns:
            Function filename
        """
        
        function_id = f"{symbol}_{date}"
        filename = f"detect_{symbol.lower()}_{date}_preconditions.py"
        filepath = self.detectors_dir / filename
        
        # Save function to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(function_code)
            logger.info(f"💾 Detector function saved: {filename}")
        except Exception as e:
            logger.error(f"Error saving detector function: {e}")
            return ""
        
        # Register in memory
        detector_data = {
            "function_id": function_id,
            "symbol": symbol,
            "date": date,
            "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pump_data": pump_data,
            "pre_pump_analysis": self._extract_key_features(pre_pump_analysis),
            "crypto_scan_signals": crypto_scan_signals or {},
            "performance": {
                "tested_cases": 0,
                "successful_detections": 0,
                "false_positives": 0,
                "accuracy": 0.0
            },
            "refinements": [],
            "meta_patterns": []
        }
        
        self.memory["detectors"][function_id] = detector_data
        self.memory["performance_metrics"]["total_detectors"] += 1
        
        # Link pump event
        pump_id = f"{symbol}_{date}"
        self.memory["pump_events"][pump_id] = {
            "symbol": symbol,
            "date": date,
            "price_increase_pct": pump_data.get("price_increase_pct", 0),
            "duration_minutes": pump_data.get("duration_minutes", 0),
            "volume_spike": pump_data.get("volume_spike", 0),
            "detector_function": function_id,
            "crypto_scan_detected": bool(crypto_scan_signals),
            "ppwcs_score": crypto_scan_signals.get("ppwcs_score") if crypto_scan_signals else None
        }
        
        self._save_memory()
        return filename
    
    def _extract_key_features(self, pre_pump_analysis: Dict) -> Dict:
        """Extract key features from pre-pump analysis for pattern recognition"""
        return {
            "rsi": pre_pump_analysis.get("rsi", 0),
            "price_trend": pre_pump_analysis.get("price_trend", "unknown"),
            "volume_trend": pre_pump_analysis.get("volume_trend", "unknown"),
            "compression_detected": pre_pump_analysis.get("compression", {}).get("detected", False),
            "volume_spikes_count": len(pre_pump_analysis.get("volume_spikes", [])),
            "fake_rejects_count": len(pre_pump_analysis.get("fake_rejects", [])),
            "vwap_position": pre_pump_analysis.get("vwap", {}).get("above_vwap", False)
        }
    
    def get_similar_detectors(self, current_analysis: Dict, limit: int = 5) -> List[Dict]:
        """
        Find similar detector functions based on pre-pump analysis patterns
        
        Args:
            current_analysis: Current pre-pump analysis to compare
            limit: Maximum number of similar detectors to return
            
        Returns:
            List of similar detector data with similarity scores
        """
        
        current_features = self._extract_key_features(current_analysis)
        similarities = []
        
        for detector_id, detector_data in self.memory["detectors"].items():
            stored_features = detector_data["pre_pump_analysis"]
            
            # Calculate similarity score
            similarity = self._calculate_similarity(current_features, stored_features)
            
            if similarity > 0.3:  # Minimum similarity threshold
                similarities.append({
                    "detector_id": detector_id,
                    "similarity_score": similarity,
                    "detector_data": detector_data
                })
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:limit]
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity score between two feature sets"""
        score = 0.0
        total_weight = 0.0
        
        # Define feature weights
        weights = {
            "rsi": 0.2,
            "price_trend": 0.15,
            "volume_trend": 0.15,
            "compression_detected": 0.15,
            "volume_spikes_count": 0.1,
            "fake_rejects_count": 0.1,
            "vwap_position": 0.15
        }
        
        for feature, weight in weights.items():
            if feature in features1 and feature in features2:
                total_weight += weight
                
                if feature == "rsi":
                    # RSI similarity (closer values = higher score)
                    rsi_diff = abs(features1[feature] - features2[feature])
                    score += weight * max(0, (50 - rsi_diff) / 50)
                elif feature in ["volume_spikes_count", "fake_rejects_count"]:
                    # Count similarity
                    count_diff = abs(features1[feature] - features2[feature])
                    score += weight * max(0, (5 - count_diff) / 5)
                elif feature in ["price_trend", "volume_trend"]:
                    # Exact match for trends
                    if features1[feature] == features2[feature]:
                        score += weight
                else:
                    # Boolean features
                    if features1[feature] == features2[feature]:
                        score += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def generate_context_for_gpt(self, current_analysis: Dict, pump_data: Dict) -> str:
        """
        Generate rich context for GPT including similar detectors and patterns
        
        Args:
            current_analysis: Current pre-pump analysis
            pump_data: Current pump event data
            
        Returns:
            Formatted context string for GPT
        """
        
        similar_detectors = self.get_similar_detectors(current_analysis)
        
        context = f"""
=== GPT MEMORY CONTEXT ===

📊 SYSTEM STATISTICS:
• Total detectors created: {self.memory['performance_metrics']['total_detectors']}
• Overall accuracy: {self.memory['performance_metrics']['accuracy_score']:.1%}
• Successful predictions: {self.memory['performance_metrics']['successful_predictions']}

🔍 CURRENT PUMP ANALYSIS:
• Symbol: {pump_data.get('symbol', 'UNKNOWN')}
• Price increase: +{pump_data.get('price_increase_pct', 0):.1f}%
• Duration: {pump_data.get('duration_minutes', 0)} minutes
• Volume spike: {pump_data.get('volume_spike', 0):.1f}x

📈 PRE-PUMP FEATURES:
• RSI: {current_analysis.get('rsi', 0):.1f}
• Price trend: {current_analysis.get('price_trend', 'unknown')}
• Volume trend: {current_analysis.get('volume_trend', 'unknown')}
• Compression detected: {current_analysis.get('compression', {}).get('detected', False)}
• Volume spikes: {len(current_analysis.get('volume_spikes', []))}
• Fake rejects: {len(current_analysis.get('fake_rejects', []))}
"""

        if similar_detectors:
            context += "\n🧠 SIMILAR PATTERNS FOUND:\n"
            for i, similar in enumerate(similar_detectors[:3], 1):
                detector = similar["detector_data"]
                similarity = similar["similarity_score"]
                
                context += f"""
{i}. {detector['symbol']} ({detector['date']}) - Similarity: {similarity:.1%}
   • Price increase: +{detector['pump_data'].get('price_increase_pct', 0):.1f}%
   • RSI: {detector['pre_pump_analysis'].get('rsi', 0):.1f}
   • Accuracy: {detector['performance'].get('accuracy', 0):.1%}
"""

        # Add meta-patterns if available
        if self.memory["meta_patterns"]:
            context += "\n🎯 DISCOVERED META-PATTERNS:\n"
            for pattern_id, pattern in self.memory["meta_patterns"].items():
                context += f"• {pattern['description']} (Confidence: {pattern['confidence']:.1%})\n"

        # Add recent insights
        recent_insights = self.memory["learning_insights"][-3:] if self.memory["learning_insights"] else []
        if recent_insights:
            context += "\n💡 RECENT LEARNING INSIGHTS:\n"
            for insight in recent_insights:
                context += f"• {insight['description']} ({insight['date']})\n"

        context += """

=== INSTRUCTIONS FOR IMPROVEMENT ===
Based on the similar patterns above:
1. Identify what worked well in previous successful detectors
2. Avoid patterns that led to false positives
3. Consider combining successful elements from multiple detectors
4. Generate more precise thresholds based on historical data
5. Look for unique aspects of this case that might require new logic

Focus on creating a detector that learns from past successes while adapting to new patterns.
"""

        return context
    
    def update_detector_performance(self, detector_id: str, test_result: Dict):
        """Update detector performance metrics based on test results"""
        if detector_id in self.memory["detectors"]:
            perf = self.memory["detectors"][detector_id]["performance"]
            perf["tested_cases"] += 1
            
            if test_result.get("detected", False):
                if test_result.get("correct_prediction", True):  # Assume correct unless specified
                    perf["successful_detections"] += 1
                    self.memory["performance_metrics"]["successful_predictions"] += 1
                else:
                    perf["false_positives"] += 1
                    self.memory["performance_metrics"]["false_positives"] += 1
            
            # Update accuracy
            if perf["tested_cases"] > 0:
                perf["accuracy"] = perf["successful_detections"] / perf["tested_cases"]
            
            # Update overall accuracy
            total_tests = sum(d["performance"]["tested_cases"] for d in self.memory["detectors"].values())
            total_successes = sum(d["performance"]["successful_detections"] for d in self.memory["detectors"].values())
            
            if total_tests > 0:
                self.memory["performance_metrics"]["accuracy_score"] = total_successes / total_tests
            
            self._save_memory()
            logger.info(f"📊 Updated performance for {detector_id}: {perf['accuracy']:.1%} accuracy")
    
    def discover_meta_patterns(self) -> List[Dict]:
        """
        Analyze all detectors to discover meta-patterns across successful cases
        
        Returns:
            List of discovered meta-patterns
        """
        
        successful_detectors = [
            d for d in self.memory["detectors"].values() 
            if d["performance"]["accuracy"] > 0.7 and d["performance"]["tested_cases"] >= 3
        ]
        
        if len(successful_detectors) < 3:
            return []
        
        patterns = []
        
        # Pattern 1: RSI range analysis
        rsi_values = [d["pre_pump_analysis"]["rsi"] for d in successful_detectors]
        if rsi_values:
            avg_rsi = sum(rsi_values) / len(rsi_values)
            rsi_std = (sum((x - avg_rsi) ** 2 for x in rsi_values) / len(rsi_values)) ** 0.5
            
            if rsi_std < 15:  # Low variance indicates pattern
                patterns.append({
                    "type": "rsi_range",
                    "description": f"Successful pumps often have RSI around {avg_rsi:.1f} ±{rsi_std:.1f}",
                    "confidence": min(0.95, len(successful_detectors) / 10),
                    "rule": f"rsi >= {avg_rsi - rsi_std:.1f} and rsi <= {avg_rsi + rsi_std:.1f}"
                })
        
        # Pattern 2: Volume spike frequency
        volume_spike_counts = [d["pre_pump_analysis"]["volume_spikes_count"] for d in successful_detectors]
        most_common_spikes = max(set(volume_spike_counts), key=volume_spike_counts.count)
        spike_frequency = volume_spike_counts.count(most_common_spikes) / len(volume_spike_counts)
        
        if spike_frequency > 0.6:
            patterns.append({
                "type": "volume_pattern",
                "description": f"Successful pumps typically have {most_common_spikes} volume spike(s)",
                "confidence": spike_frequency,
                "rule": f"volume_spikes_count == {most_common_spikes}"
            })
        
        # Update memory with discovered patterns
        for pattern in patterns:
            pattern_id = f"{pattern['type']}_{datetime.now().strftime('%Y%m%d')}"
            self.memory["meta_patterns"][pattern_id] = pattern
        
        if patterns:
            self.memory["learning_insights"].append({
                "date": datetime.now().strftime('%Y-%m-%d'),
                "description": f"Discovered {len(patterns)} new meta-patterns",
                "details": [p["description"] for p in patterns]
            })
        
        self._save_memory()
        return patterns
    
    def get_crypto_scan_integration_data(self) -> Optional[Dict]:
        """
        Load recent crypto-scan signals and performance data for integration
        
        Returns:
            Dictionary with crypto-scan performance metrics and recent signals
        """
        
        try:
            # Try to load recent alerts from crypto-scan
            alerts_file = self.crypto_scan_data_dir / "alerts.json"
            reports_file = self.crypto_scan_data_dir / "signal_reports.json"
            
            integration_data = {
                "recent_alerts": [],
                "ppwcs_performance": {},
                "stage_activation_stats": {},
                "last_updated": None
            }
            
            if alerts_file.exists():
                with open(alerts_file, 'r', encoding='utf-8') as f:
                    alerts_data = json.load(f)
                    integration_data["recent_alerts"] = alerts_data.get("alerts", [])[-50:]  # Last 50 alerts
                    integration_data["last_updated"] = alerts_data.get("last_updated")
            
            if reports_file.exists():
                with open(reports_file, 'r', encoding='utf-8') as f:
                    reports_data = json.load(f)
                    integration_data["ppwcs_performance"] = reports_data.get("performance_summary", {})
                    integration_data["stage_activation_stats"] = reports_data.get("stage_stats", {})
            
            return integration_data
            
        except Exception as e:
            logger.warning(f"Could not load crypto-scan integration data: {e}")
            return None
    
    def suggest_crypto_scan_improvements(self) -> List[str]:
        """
        Analyze pump_analysis results vs crypto-scan signals to suggest improvements
        
        Returns:
            List of improvement suggestions
        """
        
        suggestions = []
        
        # Analyze pump events that were missed by crypto-scan
        missed_pumps = [
            event for event in self.memory["pump_events"].values()
            if not event["crypto_scan_detected"] and event["price_increase_pct"] > 15
        ]
        
        if len(missed_pumps) > 5:
            suggestions.append(
                f"Consider lowering PPWCS thresholds - {len(missed_pumps)} significant pumps "
                f"(>{15}%) were not detected by crypto-scan"
            )
        
        # Analyze false positives (high PPWCS but no pump)
        crypto_scan_data = self.get_crypto_scan_integration_data()
        if crypto_scan_data and crypto_scan_data["recent_alerts"]:
            high_ppwcs_alerts = [
                alert for alert in crypto_scan_data["recent_alerts"]
                if alert.get("ppwcs_score", 0) > 80
            ]
            
            # Check which ones didn't result in pumps (would need follow-up data)
            if len(high_ppwcs_alerts) > 10:
                suggestions.append(
                    "Consider refining Stage 1g quality scoring - many high PPWCS alerts detected. "
                    "Review if they resulted in actual pumps."
                )
        
        # Analyze RSI patterns from successful detectors
        successful_rsi_patterns = [
            d["pre_pump_analysis"]["rsi"] for d in self.memory["detectors"].values()
            if d["performance"]["accuracy"] > 0.8
        ]
        
        if successful_rsi_patterns:
            avg_successful_rsi = sum(successful_rsi_patterns) / len(successful_rsi_patterns)
            suggestions.append(
                f"Successful pumps often have RSI around {avg_successful_rsi:.1f}. "
                f"Consider adjusting RSI filters in crypto-scan accordingly."
            )
        
        return suggestions
    
    def generate_meta_detector(self, patterns: List[Dict]) -> str:
        """
        Generate a meta-detector function based on discovered patterns
        
        Args:
            patterns: List of meta-patterns discovered
            
        Returns:
            Python code for meta-detector function
        """
        
        function_code = '''
def meta_detector_universal_pre_pump(df):
    """
    Universal meta-detector based on learned patterns across multiple pump events
    
    This function combines successful patterns discovered from historical analysis:
'''
        
        for pattern in patterns:
            function_code += f"    # Pattern: {pattern['description']} (Confidence: {pattern['confidence']:.1%})\n"
        
        function_code += '''
    
    Args:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
    Returns:
        dict: Detection result with confidence score and triggered patterns
    """
    
    import pandas as pd
    import numpy as np
    
    if len(df) < 4:
        return {"detected": False, "confidence": 0.0, "reason": "Insufficient data"}
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['volume_ma'] = df['volume'].rolling(window=4).mean()
    df['volume_spike'] = df['volume'] / df['volume_ma']
    
    current_rsi = df['rsi'].iloc[-1]
    volume_spikes = (df['volume_spike'] > 2.5).sum()
    
    triggered_patterns = []
    confidence_score = 0.0
    
'''
        
        # Add pattern-specific logic
        for pattern in patterns:
            if pattern['type'] == 'rsi_range':
                function_code += f'''
    # RSI Range Pattern
    if {pattern['rule']}:
        triggered_patterns.append("rsi_range")
        confidence_score += {pattern['confidence']:.2f}
'''
            elif pattern['type'] == 'volume_pattern':
                function_code += f'''
    # Volume Pattern  
    if {pattern['rule']}:
        triggered_patterns.append("volume_pattern")
        confidence_score += {pattern['confidence']:.2f}
'''
        
        function_code += '''
    
    # Combine patterns for final decision
    detected = len(triggered_patterns) >= 2 and confidence_score > 1.0
    
    return {
        "detected": detected,
        "confidence": min(confidence_score, 1.0),
        "triggered_patterns": triggered_patterns,
        "rsi": current_rsi,
        "volume_spikes": volume_spikes
    }

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)
'''
        
        return function_code
    
    def get_memory_summary(self) -> Dict:
        """Get comprehensive summary of GPT memory state"""
        return {
            "total_detectors": len(self.memory["detectors"]),
            "total_pump_events": len(self.memory["pump_events"]),
            "overall_accuracy": self.memory["performance_metrics"]["accuracy_score"],
            "meta_patterns_discovered": len(self.memory["meta_patterns"]),
            "learning_insights_count": len(self.memory["learning_insights"]),
            "recent_detectors": list(self.memory["detectors"].keys())[-5:],
            "top_performing_detectors": [
                (k, v["performance"]["accuracy"]) 
                for k, v in self.memory["detectors"].items()
                if v["performance"]["tested_cases"] > 0
            ]
        }