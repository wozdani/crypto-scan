"""
GPT Learning System for Pump Analysis
Advanced self-improvement mechanism for generated detector functions
"""

import json
import os
import re
import importlib.util
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FunctionPerformance:
    """Data class for function performance tracking"""
    function_name: str
    symbol: str
    pump_date: str
    active_signals: List[str]
    retrospective_test_passed: bool
    detection_timestamp: Optional[str]
    miss_timestamp: Optional[str]
    detection_accuracy_minutes: Optional[int]  # How many minutes before/after pump
    confidence_score: float
    version: int

class LearningSystem:
    """Main learning system for GPT function evolution"""
    
    def __init__(self):
        self.base_dir = "pump-analysis"
        self.functions_dir = os.path.join(self.base_dir, "generated_functions")
        self.deprecated_dir = os.path.join(self.base_dir, "deprecated_functions")
        self.logs_file = os.path.join(self.base_dir, "function_logs.json")
        self.recommendations_file = os.path.join(self.base_dir, "gpt_recommendations.json")
        
        # Create directories
        os.makedirs(self.functions_dir, exist_ok=True)
        os.makedirs(self.deprecated_dir, exist_ok=True)
        
        # Initialize logs if not exists
        if not os.path.exists(self.logs_file):
            self._initialize_logs()
    
    def _initialize_logs(self):
        """Initialize function logs JSON file"""
        initial_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_functions": 0,
                "total_tests": 0,
                "last_evolution": None
            },
            "functions": {},
            "evolution_history": [],
            "performance_stats": {
                "avg_accuracy": 0.0,
                "best_functions": [],
                "deprecated_count": 0
            }
        }
        
        with open(self.logs_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
        
        logger.info(f"📊 Initialized learning system logs: {self.logs_file}")
    
    def save_gpt_function(self, function_code: str, symbol: str, pump_date: str, 
                         active_signals: List[str], pre_pump_data: Dict) -> str:
        """
        Save GPT-generated function with metadata
        
        Args:
            function_code: Generated Python function code
            symbol: Trading symbol (e.g., 'BTCUSDT')
            pump_date: Date of pump in YYYYMMDD format
            active_signals: List of detected signals
            pre_pump_data: Pre-pump analysis data
            
        Returns:
            Path to saved function file
        """
        # Generate function name and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        function_name = f"detect_pre_pump_{symbol.lower()}_{pump_date}"
        filename = f"{function_name}_{timestamp}.py"
        filepath = os.path.join(self.functions_dir, filename)
        
        # Extract pump event details for metadata
        pump_event = pre_pump_data.get('pump_event', {})
        
        # Create comprehensive function file with metadata
        function_content = f'''"""
GPT Generated Detector Function
Generated: {datetime.now().isoformat()}
Symbol: {symbol}
Pump Date: {pump_date}
Active Signals: {', '.join(active_signals)}
Pump Increase: {pump_event.get('price_increase_pct', 'N/A')}%
Duration: {pump_event.get('duration_minutes', 'N/A')} minutes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

{function_code}

# Metadata for learning system
FUNCTION_METADATA = {{
    "function_name": "{function_name}",
    "symbol": "{symbol}",
    "pump_date": "{pump_date}",
    "active_signals": {active_signals},
    "generated_timestamp": "{datetime.now().isoformat()}",
    "pump_increase_pct": {pump_event.get('price_increase_pct', 0)},
    "pump_duration_minutes": {pump_event.get('duration_minutes', 0)},
    "version": 1
}}
'''
        
        # Save function file
        with open(filepath, 'w') as f:
            f.write(function_content)
        
        # Log function creation
        self._log_function_creation(function_name, symbol, pump_date, active_signals, 
                                  filepath, pre_pump_data)
        
        logger.info(f"💾 Saved GPT function: {filename}")
        return filepath
    
    def _log_function_creation(self, function_name: str, symbol: str, pump_date: str,
                              active_signals: List[str], filepath: str, pre_pump_data: Dict):
        """Log new function creation to function_logs.json"""
        
        # Load existing logs
        with open(self.logs_file, 'r') as f:
            logs = json.load(f)
        
        # Create function entry
        function_entry = {
            "created": datetime.now().isoformat(),
            "symbol": symbol,
            "pump_date": pump_date,
            "active_signals": active_signals,
            "filepath": filepath,
            "version": 1,
            "tests": [],
            "performance": {
                "total_tests": 0,
                "successful_detections": 0,
                "false_positives": 0,
                "accuracy_score": 0.0,
                "avg_detection_time_minutes": None
            },
            "evolution": {
                "parent_function": None,
                "improvements": [],
                "deprecated": False,
                "deprecation_reason": None
            },
            "pump_details": {
                "price_increase_pct": pre_pump_data.get('pump_event', {}).get('price_increase_pct', 0),
                "duration_minutes": pre_pump_data.get('pump_event', {}).get('duration_minutes', 0),
                "volume_spike": pre_pump_data.get('analysis', {}).get('volume_spikes', []),
                "trend": pre_pump_data.get('analysis', {}).get('trend', 'unknown')
            }
        }
        
        # Add to logs
        logs["functions"][function_name] = function_entry
        logs["metadata"]["total_functions"] += 1
        logs["metadata"]["last_evolution"] = datetime.now().isoformat()
        
        # Save updated logs
        with open(self.logs_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def test_functions_on_new_pump(self, pump_data: Dict, pre_pump_candles: pd.DataFrame) -> Dict:
        """
        Test all existing functions on a new pump discovery
        
        Args:
            pump_data: New pump event data
            pre_pump_candles: Pre-pump candle data for testing
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"🧪 Testing existing functions on new pump: {pump_data.get('symbol', 'Unknown')}")
        
        test_results = {
            "pump_symbol": pump_data.get('symbol'),
            "pump_date": pump_data.get('start_time'),
            "functions_tested": 0,
            "successful_detections": [],
            "failed_detections": [],
            "close_detections": [],  # Functions that were close but missed
            "recommendations": []
        }
        
        # Load all existing functions
        function_files = [f for f in os.listdir(self.functions_dir) if f.endswith('.py')]
        
        for function_file in function_files:
            try:
                # Load and test function
                result = self._test_single_function(function_file, pre_pump_candles, pump_data)
                
                test_results["functions_tested"] += 1
                
                if result["detected"]:
                    test_results["successful_detections"].append(result)
                    logger.info(f"✅ Function {result['function_name']} detected pump")
                elif result["close_detection"]:
                    test_results["close_detections"].append(result)
                    logger.info(f"🔶 Function {result['function_name']} was close (missed by {result['miss_margin_minutes']} min)")
                else:
                    test_results["failed_detections"].append(result)
                
                # Update function logs with test result
                self._update_function_test_log(result)
                
            except Exception as e:
                logger.error(f"❌ Error testing function {function_file}: {e}")
        
        # Generate recommendations based on test results
        recommendations = self._generate_test_recommendations(test_results)
        test_results["recommendations"] = recommendations
        
        # Save test results
        self._save_test_results(test_results)
        
        return test_results
    
    def _test_single_function(self, function_file: str, candle_data: pd.DataFrame, 
                             pump_data: Dict) -> Dict:
        """Test a single function on candle data"""
        
        function_path = os.path.join(self.functions_dir, function_file)
        
        # Load function dynamically
        spec = importlib.util.spec_from_file_location("detector_module", function_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get function metadata
        metadata = getattr(module, 'FUNCTION_METADATA', {})
        function_name = metadata.get('function_name', function_file.replace('.py', ''))
        
        # Find the main detection function
        detector_func = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and attr_name.startswith('detect_'):
                detector_func = attr
                break
        
        if not detector_func:
            return {
                "function_name": function_name,
                "function_file": function_file,
                "detected": False,
                "error": "No detector function found",
                "close_detection": False
            }
        
        # Test function on pre-pump data
        pump_start_time = pd.to_datetime(pump_data.get('start_time'))
        
        # Test different time windows before pump
        detection_results = []
        
        for minutes_before in [60, 45, 30, 15, 10, 5]:
            test_time = pump_start_time - pd.Timedelta(minutes=minutes_before)
            
            # Get data up to test time
            test_data = candle_data[candle_data.index <= test_time].copy()
            
            if len(test_data) < 20:  # Need minimum data
                continue
                
            try:
                # Call detector function
                result = detector_func(test_data)
                
                if result and result.get('signal_detected', False):
                    detection_results.append({
                        "detected_at": test_time,
                        "minutes_before_pump": minutes_before,
                        "confidence": result.get('confidence', 0.5),
                        "signals": result.get('active_signals', [])
                    })
                    
            except Exception as e:
                logger.debug(f"Function {function_name} failed at {minutes_before}min: {e}")
        
        # Analyze results
        if detection_results:
            best_detection = min(detection_results, key=lambda x: x['minutes_before_pump'])
            return {
                "function_name": function_name,
                "function_file": function_file,
                "detected": True,
                "detection_time": best_detection["detected_at"].isoformat(),
                "minutes_before_pump": best_detection["minutes_before_pump"],
                "confidence": best_detection["confidence"],
                "signals": best_detection["signals"],
                "close_detection": False,
                "all_detections": detection_results
            }
        
        # Check for close detections (after pump started)
        for minutes_after in [5, 10, 15, 30]:
            test_time = pump_start_time + pd.Timedelta(minutes=minutes_after)
            test_data = candle_data[candle_data.index <= test_time].copy()
            
            try:
                result = detector_func(test_data)
                if result and result.get('signal_detected', False):
                    return {
                        "function_name": function_name,
                        "function_file": function_file,
                        "detected": False,
                        "close_detection": True,
                        "miss_margin_minutes": minutes_after,
                        "confidence": result.get('confidence', 0.5)
                    }
            except:
                continue
        
        return {
            "function_name": function_name,
            "function_file": function_file,
            "detected": False,
            "close_detection": False
        }
    
    def _update_function_test_log(self, test_result: Dict):
        """Update function performance logs with test result"""
        
        with open(self.logs_file, 'r') as f:
            logs = json.load(f)
        
        function_name = test_result["function_name"]
        
        if function_name in logs["functions"]:
            func_log = logs["functions"][function_name]
            
            # Add test entry
            test_entry = {
                "timestamp": datetime.now().isoformat(),
                "detected": test_result["detected"],
                "close_detection": test_result.get("close_detection", False),
                "detection_time_minutes": test_result.get("minutes_before_pump"),
                "confidence": test_result.get("confidence", 0.0)
            }
            
            func_log["tests"].append(test_entry)
            func_log["performance"]["total_tests"] += 1
            
            if test_result["detected"]:
                func_log["performance"]["successful_detections"] += 1
            
            # Recalculate accuracy
            total_tests = func_log["performance"]["total_tests"]
            successful = func_log["performance"]["successful_detections"]
            func_log["performance"]["accuracy_score"] = successful / total_tests if total_tests > 0 else 0.0
            
            # Save updated logs
            with open(self.logs_file, 'w') as f:
                json.dump(logs, f, indent=2)
    
    def _generate_test_recommendations(self, test_results: Dict) -> List[Dict]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        successful_count = len(test_results["successful_detections"])
        close_count = len(test_results["close_detections"])
        total_tested = test_results["functions_tested"]
        
        if successful_count == 0 and close_count > 0:
            # Some functions were close - recommend evolution
            recommendations.append({
                "type": "evolve_functions",
                "priority": "high",
                "message": f"{close_count} functions were close to detecting the pump. Consider evolving them.",
                "functions_to_evolve": [f["function_name"] for f in test_results["close_detections"]]
            })
        
        elif successful_count == 0:
            # No detection at all - recommend new function
            recommendations.append({
                "type": "create_new_function",
                "priority": "high",
                "message": "No existing functions detected this pump. Create new specialized function.",
                "pump_characteristics": test_results.get("pump_symbol")
            })
        
        if successful_count > 0:
            # Some functions worked - recommend promotion
            best_functions = sorted(
                test_results["successful_detections"],
                key=lambda x: x.get("minutes_before_pump", 0),
                reverse=True
            )[:3]
            
            recommendations.append({
                "type": "promote_functions",
                "priority": "medium",
                "message": f"{successful_count} functions successfully detected the pump.",
                "best_functions": [f["function_name"] for f in best_functions]
            })
        
        return recommendations
    
    def _save_test_results(self, test_results: Dict):
        """Save test results to file"""
        results_file = os.path.join(self.base_dir, "test_results", 
                                   f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
    
    def evolve_function(self, function_name: str, improvement_reason: str, 
                       new_function_code: str) -> str:
        """
        Evolve an existing function to create improved version
        
        Args:
            function_name: Name of function to evolve
            improvement_reason: Reason for evolution
            new_function_code: Improved function code
            
        Returns:
            Path to new evolved function
        """
        logger.info(f"🧬 Evolving function: {function_name}")
        
        # Load function logs to get metadata
        with open(self.logs_file, 'r') as f:
            logs = json.load(f)
        
        if function_name not in logs["functions"]:
            logger.error(f"Function {function_name} not found in logs")
            return None
        
        original_func = logs["functions"][function_name]
        current_version = original_func["version"]
        new_version = current_version + 1
        
        # Create new function name with version
        base_name = function_name.replace(f"_v{current_version}", "") if f"_v{current_version}" in function_name else function_name
        new_function_name = f"{base_name}_v{new_version}"
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{new_function_name}_{timestamp}.py"
        filepath = os.path.join(self.functions_dir, filename)
        
        # Create evolved function content
        evolved_content = f'''"""
GPT Evolved Detector Function (Version {new_version})
Evolved from: {function_name}
Evolution reason: {improvement_reason}
Generated: {datetime.now().isoformat()}
Symbol: {original_func["symbol"]}
Original pump date: {original_func["pump_date"]}
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

{new_function_code}

# Metadata for learning system
FUNCTION_METADATA = {{
    "function_name": "{new_function_name}",
    "symbol": "{original_func["symbol"]}",
    "pump_date": "{original_func["pump_date"]}",
    "active_signals": {original_func["active_signals"]},
    "generated_timestamp": "{datetime.now().isoformat()}",
    "version": {new_version},
    "parent_function": "{function_name}",
    "evolution_reason": "{improvement_reason}"
}}
'''
        
        # Save evolved function
        with open(filepath, 'w') as f:
            f.write(evolved_content)
        
        # Log evolution
        self._log_function_evolution(new_function_name, function_name, improvement_reason,
                                   filepath, original_func, new_version)
        
        logger.info(f"🚀 Created evolved function: {new_function_name}")
        return filepath
    
    def _log_function_evolution(self, new_function_name: str, parent_function: str,
                               improvement_reason: str, filepath: str, original_func: Dict,
                               new_version: int):
        """Log function evolution to logs"""
        
        with open(self.logs_file, 'r') as f:
            logs = json.load(f)
        
        # Create entry for evolved function
        evolved_entry = original_func.copy()
        evolved_entry.update({
            "created": datetime.now().isoformat(),
            "filepath": filepath,
            "version": new_version,
            "tests": [],  # Reset tests for new version
            "performance": {
                "total_tests": 0,
                "successful_detections": 0,
                "false_positives": 0,
                "accuracy_score": 0.0,
                "avg_detection_time_minutes": None
            },
            "evolution": {
                "parent_function": parent_function,
                "improvements": [improvement_reason],
                "deprecated": False,
                "deprecation_reason": None
            }
        })
        
        # Add to logs
        logs["functions"][new_function_name] = evolved_entry
        logs["metadata"]["total_functions"] += 1
        
        # Add evolution history entry
        evolution_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "evolution",
            "parent_function": parent_function,
            "new_function": new_function_name,
            "reason": improvement_reason,
            "version": new_version
        }
        
        logs["evolution_history"].append(evolution_entry)
        logs["metadata"]["last_evolution"] = datetime.now().isoformat()
        
        # Save updated logs
        with open(self.logs_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def retrospective_test_suite(self, recent_pumps_data: List[Dict]) -> Dict:
        """
        Run retrospective tests on last 20 pumps for all functions
        Used for periodic evaluation (every 12h)
        """
        logger.info(f"🔬 Running retrospective test suite on {len(recent_pumps_data)} pumps")
        
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "pumps_tested": len(recent_pumps_data),
            "functions_performance": {},
            "overall_stats": {
                "avg_detection_rate": 0.0,
                "best_performing_functions": [],
                "functions_to_deprecate": []
            },
            "recommendations": []
        }
        
        # Load all functions
        function_files = [f for f in os.listdir(self.functions_dir) if f.endswith('.py')]
        
        for function_file in function_files:
            function_performance = self._test_function_on_multiple_pumps(
                function_file, recent_pumps_data
            )
            
            function_name = function_performance["function_name"]
            test_results["functions_performance"][function_name] = function_performance
        
        # Calculate overall statistics
        self._calculate_retrospective_stats(test_results)
        
        # Generate recommendations
        recommendations = self._generate_retrospective_recommendations(test_results)
        test_results["recommendations"] = recommendations
        
        # Save retrospective results
        retro_file = os.path.join(self.base_dir, "retrospective_tests",
                                 f"retro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(retro_file), exist_ok=True)
        
        with open(retro_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"📊 Retrospective test completed. Results saved to: {retro_file}")
        return test_results
    
    def _test_function_on_multiple_pumps(self, function_file: str, pumps_data: List[Dict]) -> Dict:
        """Test a single function on multiple pump events"""
        
        function_path = os.path.join(self.functions_dir, function_file)
        
        # Load function metadata
        try:
            spec = importlib.util.spec_from_file_location("detector_module", function_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            metadata = getattr(module, 'FUNCTION_METADATA', {})
            function_name = metadata.get('function_name', function_file.replace('.py', ''))
        except Exception as e:
            return {
                "function_name": function_file.replace('.py', ''),
                "error": f"Failed to load function: {e}",
                "detections": 0,
                "total_tests": 0,
                "accuracy": 0.0
            }
        
        detections = 0
        valid_tests = 0
        detection_times = []
        
        for pump_data in pumps_data:
            try:
                # Here you would load pre-pump candle data for each pump
                # For now, we'll simulate this
                result = self._simulate_function_test(function_name, pump_data)
                
                valid_tests += 1
                
                if result["detected"]:
                    detections += 1
                    if result.get("minutes_before_pump"):
                        detection_times.append(result["minutes_before_pump"])
                        
            except Exception as e:
                logger.debug(f"Error testing {function_name} on pump {pump_data.get('symbol')}: {e}")
        
        accuracy = detections / valid_tests if valid_tests > 0 else 0.0
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else None
        
        return {
            "function_name": function_name,
            "function_file": function_file,
            "detections": detections,
            "total_tests": valid_tests,
            "accuracy": accuracy,
            "avg_detection_time_minutes": avg_detection_time,
            "detection_times": detection_times
        }
    
    def _simulate_function_test(self, function_name: str, pump_data: Dict) -> Dict:
        """Simulate function test (placeholder for real implementation)"""
        # This is a simplified simulation
        # In real implementation, you would load actual candle data and test
        
        import random
        
        # Simulate detection based on function characteristics
        detected = random.random() > 0.7  # 30% detection rate simulation
        minutes_before = random.randint(5, 60) if detected else None
        
        return {
            "detected": detected,
            "minutes_before_pump": minutes_before,
            "confidence": random.random()
        }
    
    def _calculate_retrospective_stats(self, test_results: Dict):
        """Calculate overall statistics from retrospective tests"""
        
        performances = test_results["functions_performance"]
        
        if not performances:
            return
        
        # Calculate average detection rate
        total_accuracy = sum(p["accuracy"] for p in performances.values() if "accuracy" in p)
        avg_detection_rate = total_accuracy / len(performances) if performances else 0.0
        
        test_results["overall_stats"]["avg_detection_rate"] = avg_detection_rate
        
        # Find best performing functions
        best_functions = sorted(
            [(name, perf) for name, perf in performances.items() if "accuracy" in perf],
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )[:5]
        
        test_results["overall_stats"]["best_performing_functions"] = [
            {"name": name, "accuracy": perf["accuracy"]} for name, perf in best_functions
        ]
        
        # Find functions to potentially deprecate (low accuracy)
        functions_to_deprecate = [
            name for name, perf in performances.items()
            if "accuracy" in perf and perf["accuracy"] < 0.1 and perf["total_tests"] >= 5
        ]
        
        test_results["overall_stats"]["functions_to_deprecate"] = functions_to_deprecate
    
    def _generate_retrospective_recommendations(self, test_results: Dict) -> List[Dict]:
        """Generate recommendations based on retrospective test results"""
        recommendations = []
        
        overall_stats = test_results["overall_stats"]
        avg_detection_rate = overall_stats["avg_detection_rate"]
        
        if avg_detection_rate < 0.3:
            recommendations.append({
                "type": "system_improvement",
                "priority": "high",
                "message": f"Overall detection rate is low ({avg_detection_rate:.1%}). Consider reviewing function generation strategy."
            })
        
        best_functions = overall_stats["best_performing_functions"]
        if best_functions:
            recommendations.append({
                "type": "promote_best_functions",
                "priority": "medium",
                "message": f"Top performing functions should be used as templates for new generations.",
                "functions": [f["name"] for f in best_functions[:3]]
            })
        
        functions_to_deprecate = overall_stats["functions_to_deprecate"]
        if functions_to_deprecate:
            recommendations.append({
                "type": "deprecate_functions",
                "priority": "low",
                "message": f"{len(functions_to_deprecate)} functions show poor performance and should be deprecated.",
                "functions": functions_to_deprecate
            })
        
        return recommendations
    
    def deprecate_function(self, function_name: str, reason: str):
        """Move function to deprecated folder and update logs"""
        
        with open(self.logs_file, 'r') as f:
            logs = json.load(f)
        
        if function_name not in logs["functions"]:
            logger.error(f"Function {function_name} not found")
            return
        
        func_data = logs["functions"][function_name]
        original_path = func_data["filepath"]
        
        if os.path.exists(original_path):
            # Move to deprecated folder
            filename = os.path.basename(original_path)
            deprecated_path = os.path.join(self.deprecated_dir, filename)
            
            os.rename(original_path, deprecated_path)
            
            # Update logs
            func_data["evolution"]["deprecated"] = True
            func_data["evolution"]["deprecation_reason"] = reason
            func_data["filepath"] = deprecated_path
            
            logs["performance_stats"]["deprecated_count"] += 1
            
            with open(self.logs_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"📁 Deprecated function {function_name}: {reason}")
    
    def get_learning_summary(self) -> Dict:
        """Get comprehensive learning system summary"""
        
        with open(self.logs_file, 'r') as f:
            logs = json.load(f)
        
        # Calculate summary statistics
        total_functions = logs["metadata"]["total_functions"]
        active_functions = len([f for f in logs["functions"].values() 
                              if not f.get("evolution", {}).get("deprecated", False)])
        deprecated_functions = logs["performance_stats"]["deprecated_count"]
        
        # Find best performing functions
        best_functions = []
        for name, func in logs["functions"].items():
            if func["performance"]["total_tests"] > 0:
                best_functions.append({
                    "name": name,
                    "accuracy": func["performance"]["accuracy_score"],
                    "tests": func["performance"]["total_tests"]
                })
        
        best_functions.sort(key=lambda x: x["accuracy"], reverse=True)
        
        return {
            "total_functions": total_functions,
            "active_functions": active_functions,
            "deprecated_functions": deprecated_functions,
            "avg_accuracy": logs["performance_stats"]["avg_accuracy"],
            "best_functions": best_functions[:10],
            "last_evolution": logs["metadata"]["last_evolution"],
            "evolution_count": len(logs["evolution_history"])
        }