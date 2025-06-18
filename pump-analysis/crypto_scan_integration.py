"""
Crypto-Scan Integration Module

This module provides integration between pump-analysis and crypto-scan systems
to enable GPT learning from real-time pre-pump signals and historical pump data.
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class CryptoScanIntegration:
    """Handles integration with crypto-scan system data"""
    
    def __init__(self):
        self.crypto_scan_dir = Path("../crypto-scan")
        self.data_dir = self.crypto_scan_dir / "data"
        
        logger.info("🔗 Crypto-Scan Integration initialized")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """
        Get recent alerts from crypto-scan system
        
        Args:
            hours: Number of hours back to look for alerts
            
        Returns:
            List of recent alert data
        """
        
        alerts_file = self.data_dir / "alerts.json"
        if not alerts_file.exists():
            logger.warning("No alerts file found in crypto-scan")
            return []
        
        try:
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts_data = json.load(f)
            
            alerts = alerts_data.get("alerts", [])
            
            # Filter by time if needed
            cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
            recent_alerts = [
                alert for alert in alerts
                if alert.get("timestamp", 0) > cutoff_time
            ]
            
            logger.info(f"📊 Found {len(recent_alerts)} recent alerts from crypto-scan")
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Error loading crypto-scan alerts: {e}")
            return []
    
    def get_ppwcs_performance_stats(self) -> Dict:
        """
        Get PPWCS performance statistics from crypto-scan
        
        Returns:
            Dictionary with performance metrics
        """
        
        reports_file = self.data_dir / "signal_reports.json"
        if not reports_file.exists():
            return {}
        
        try:
            with open(reports_file, 'r', encoding='utf-8') as f:
                reports_data = json.load(f)
            
            return {
                "total_alerts": len(reports_data.get("reports", [])),
                "avg_ppwcs_score": self._calculate_avg_ppwcs(reports_data),
                "stage_activation_stats": self._analyze_stage_activations(reports_data),
                "detector_frequency": self._analyze_detector_frequency(reports_data)
            }
            
        except Exception as e:
            logger.error(f"Error loading crypto-scan performance stats: {e}")
            return {}
    
    def _calculate_avg_ppwcs(self, reports_data: Dict) -> float:
        """Calculate average PPWCS score from reports"""
        reports = reports_data.get("reports", [])
        if not reports:
            return 0.0
        
        scores = [r.get("ppwcs_score", 0) for r in reports if r.get("ppwcs_score")]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _analyze_stage_activations(self, reports_data: Dict) -> Dict:
        """Analyze which stages are most frequently activated"""
        reports = reports_data.get("reports", [])
        
        stage_counts = {
            "stage_minus2_1": 0,
            "stage_minus2_2": 0,
            "stage_minus1": 0,
            "stage_1g": 0
        }
        
        for report in reports:
            for stage in stage_counts.keys():
                if report.get(stage + "_active", False):
                    stage_counts[stage] += 1
        
        return stage_counts
    
    def _analyze_detector_frequency(self, reports_data: Dict) -> Dict:
        """Analyze frequency of different detector activations"""
        reports = reports_data.get("reports", [])
        detector_counts = {}
        
        for report in reports:
            signals = report.get("signals", {})
            for detector, active in signals.items():
                if active:
                    detector_counts[detector] = detector_counts.get(detector, 0) + 1
        
        return detector_counts
    
    def get_symbol_signal_history(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Get signal history for a specific symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            days: Number of days back to analyze
            
        Returns:
            List of signal events for the symbol
        """
        
        reports_file = self.data_dir / "signal_reports.json"
        if not reports_file.exists():
            return []
        
        try:
            with open(reports_file, 'r', encoding='utf-8') as f:
                reports_data = json.load(f)
            
            reports = reports_data.get("reports", [])
            cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
            
            symbol_reports = [
                report for report in reports
                if report.get("symbol") == symbol and report.get("timestamp", 0) > cutoff_time
            ]
            
            logger.info(f"📈 Found {len(symbol_reports)} signal events for {symbol}")
            return symbol_reports
            
        except Exception as e:
            logger.error(f"Error loading signal history for {symbol}: {e}")
            return []
    
    def check_symbol_pre_pump_detected(self, symbol: str, pump_time: datetime, 
                                     window_hours: int = 2) -> Optional[Dict]:
        """
        Check if crypto-scan detected pre-pump signals before a pump event
        
        Args:
            symbol: Trading symbol
            pump_time: When the pump occurred
            window_hours: Hours before pump to check for signals
            
        Returns:
            Dict with pre-pump signal data if found, None otherwise
        """
        
        start_time = pump_time.timestamp() - (window_hours * 3600)
        end_time = pump_time.timestamp()
        
        signal_history = self.get_symbol_signal_history(symbol, days=1)
        
        pre_pump_signals = [
            signal for signal in signal_history
            if start_time <= signal.get("timestamp", 0) <= end_time
        ]
        
        if pre_pump_signals:
            # Return the highest PPWCS signal within the window
            best_signal = max(pre_pump_signals, key=lambda x: x.get("ppwcs_score", 0))
            logger.info(f"✅ Found pre-pump signal for {symbol}: PPWCS {best_signal.get('ppwcs_score', 0)}")
            return best_signal
        
        logger.info(f"❌ No pre-pump signals found for {symbol}")
        return None
    
    def get_successful_prediction_rate(self) -> Dict:
        """
        Calculate prediction success rate by correlating alerts with subsequent pumps
        Note: This requires historical pump data to be meaningful
        
        Returns:
            Dictionary with success rate metrics
        """
        
        # This would need integration with pump detection results
        # For now, return basic structure
        return {
            "total_alerts": 0,
            "verified_pumps": 0,
            "success_rate": 0.0,
            "false_positive_rate": 0.0,
            "avg_time_to_pump": 0.0
        }
    
    def suggest_improvements_for_crypto_scan(self, pump_analysis_results: Dict) -> List[str]:
        """
        Generate improvement suggestions for crypto-scan based on pump analysis results
        
        Args:
            pump_analysis_results: Results from pump analysis system
            
        Returns:
            List of improvement suggestions
        """
        
        suggestions = []
        
        # Analyze missed pumps
        missed_pumps = pump_analysis_results.get("pumps_not_detected_by_crypto_scan", [])
        if len(missed_pumps) > 3:
            suggestions.append(
                f"Consider lowering alert thresholds - {len(missed_pumps)} significant pumps were missed"
            )
        
        # Analyze false positives (would need follow-up data)
        recent_alerts = self.get_recent_alerts(hours=24)
        high_ppwcs_alerts = [a for a in recent_alerts if a.get("ppwcs_score", 0) > 80]
        
        if len(high_ppwcs_alerts) > 10:
            suggestions.append(
                "High number of strong alerts detected - review if they resulted in actual pumps"
            )
        
        # Analyze detector patterns
        detector_stats = self.get_ppwcs_performance_stats().get("detector_frequency", {})
        
        # Find underutilized detectors
        total_reports = sum(detector_stats.values()) if detector_stats else 0
        if total_reports > 0:
            underused = [
                detector for detector, count in detector_stats.items()
                if count / total_reports < 0.05  # Less than 5% activation
            ]
            
            if underused:
                suggestions.append(
                    f"Consider reviewing sensitivity of underused detectors: {', '.join(underused[:3])}"
                )
        
        return suggestions
    
    def export_training_data_for_gpt(self, output_file: str = "crypto_scan_training_data.json"):
        """
        Export crypto-scan data in format suitable for GPT training
        
        Args:
            output_file: Output filename for training data
        """
        
        try:
            training_data = {
                "recent_alerts": self.get_recent_alerts(hours=48),
                "performance_stats": self.get_ppwcs_performance_stats(),
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_summary": {
                    "total_alerts": len(self.get_recent_alerts(hours=48)),
                    "unique_symbols": len(set(
                        alert.get("symbol") for alert in self.get_recent_alerts(hours=48)
                    )),
                    "avg_ppwcs": self.get_ppwcs_performance_stats().get("avg_ppwcs_score", 0)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📄 Training data exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
    
    def get_integration_summary(self) -> Dict:
        """Get comprehensive summary of crypto-scan integration status"""
        
        data_files_status = {
            "alerts_file": (self.data_dir / "alerts.json").exists(),
            "reports_file": (self.data_dir / "signal_reports.json").exists(),
            "cache_files": len(list(self.data_dir.glob("cache/*.json"))) if (self.data_dir / "cache").exists() else 0
        }
        
        recent_alerts = self.get_recent_alerts(hours=24)
        
        return {
            "integration_status": "active" if any(data_files_status.values()) else "inactive",
            "data_files": data_files_status,
            "recent_activity": {
                "alerts_last_24h": len(recent_alerts),
                "latest_alert_time": max([a.get("timestamp", 0) for a in recent_alerts]) if recent_alerts else None,
                "active_symbols": len(set(a.get("symbol") for a in recent_alerts))
            },
            "performance_overview": self.get_ppwcs_performance_stats()
        }