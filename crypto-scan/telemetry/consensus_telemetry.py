#!/usr/bin/env python3
"""
Consensus Telemetry - Logging and metrics for multi-agent decisions
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from contracts.agent_contracts import TelemetryRecord, ConsensusResult
from llm.stable_client import stable_client

logger = logging.getLogger(__name__)

class ConsensusTelemetry:
    """Tracks consensus decisions and performance metrics"""
    
    def __init__(self, storage_path: str = "state/consensus_telemetry.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_records = []
        self.session_start = datetime.utcnow()
        
    def record_consensus(self, symbol: str, consensus: ConsensusResult, 
                        agent_count: int, processing_time_ms: int,
                        api_calls_count: int = 4) -> TelemetryRecord:
        """Record a consensus decision with telemetry"""
        
        # Estimate API cost (approximate)
        cost_stats = stable_client.get_cost_stats()
        estimated_cost = api_calls_count * cost_stats.get("avg_cost_per_call", 0.02)
        
        record = TelemetryRecord(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            final_probs=consensus.final_probs,
            entropy=consensus.entropy,
            confidence=consensus.confidence,
            dominant_action=consensus.dominant_action,
            agent_count=agent_count,
            processing_time_ms=processing_time_ms,
            api_calls_count=api_calls_count,
            cost_estimate_usd=estimated_cost
        )
        
        # Add to daily records
        self.daily_records.append(record.dict())
        
        # Log key metrics
        logger.info(f"[TELEMETRY] {symbol}: {consensus.dominant_action} "
                   f"(conf={consensus.confidence:.3f}, H={consensus.entropy:.2f}, "
                   f"time={processing_time_ms}ms, cost=${estimated_cost:.4f})")
        
        # Save periodically
        if len(self.daily_records) % 10 == 0:
            self._save_daily_records()
            
        return record
    
    def _save_daily_records(self):
        """Save daily telemetry records"""
        if not self.daily_records:
            return
            
        try:
            # Create daily file
            today = datetime.utcnow().strftime("%Y%m%d")
            daily_file = self.storage_path.parent / f"telemetry_daily_{today}.json"
            
            # Load existing records if any
            existing_records = []
            if daily_file.exists():
                with open(daily_file, 'r') as f:
                    existing_records = json.load(f)
            
            # Append new records
            all_records = existing_records + self.daily_records
            
            # Save combined records
            with open(daily_file, 'w') as f:
                json.dump(all_records, f, indent=2)
                
            logger.debug(f"Saved {len(self.daily_records)} telemetry records to {daily_file}")
            self.daily_records = []  # Clear after saving
            
        except Exception as e:
            logger.error(f"Failed to save telemetry records: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        if not self.daily_records:
            return {"message": "No telemetry data available"}
        
        # Aggregate stats
        total_decisions = len(self.daily_records)
        total_cost = sum(r.get("cost_estimate_usd", 0) for r in self.daily_records)
        avg_processing_time = sum(r.get("processing_time_ms", 0) for r in self.daily_records) / total_decisions
        avg_entropy = sum(r.get("entropy", 0) for r in self.daily_records) / total_decisions
        avg_confidence = sum(r.get("confidence", 0) for r in self.daily_records) / total_decisions
        
        # Action distribution
        actions = [r.get("dominant_action", "UNKNOWN") for r in self.daily_records]
        action_counts = {action: actions.count(action) for action in ["BUY", "HOLD", "AVOID", "ABSTAIN"]}
        
        # Cost stats
        cost_stats = stable_client.get_cost_stats()
        
        return {
            "session_duration_mins": (datetime.utcnow() - self.session_start).total_seconds() / 60,
            "total_decisions": total_decisions,
            "total_cost_usd": round(total_cost, 4),
            "avg_processing_time_ms": round(avg_processing_time, 1),
            "avg_entropy": round(avg_entropy, 3),
            "avg_confidence": round(avg_confidence, 3),
            "action_distribution": action_counts,
            "api_stats": cost_stats,
            "decisions_per_minute": round(total_decisions / max(1, (datetime.utcnow() - self.session_start).total_seconds() / 60), 2)
        }
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent consensus decisions"""
        recent = self.daily_records[-limit:] if self.daily_records else []
        
        # Format for readability
        formatted = []
        for record in recent:
            formatted.append({
                "symbol": record.get("symbol", "unknown"),
                "action": record.get("dominant_action", "unknown"),
                "confidence": round(record.get("confidence", 0), 3),
                "entropy": round(record.get("entropy", 0), 2),
                "time_ago_mins": self._time_ago_minutes(record.get("timestamp")),
                "cost_usd": round(record.get("cost_estimate_usd", 0), 4)
            })
            
        return formatted
    
    def _time_ago_minutes(self, timestamp_str: str) -> int:
        """Calculate minutes since timestamp"""
        try:
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                delta = datetime.utcnow() - timestamp.replace(tzinfo=None)
                return int(delta.total_seconds() / 60)
            return 0
        except:
            return 0
    
    def export_daily_summary(self, date_str: str = None) -> Dict[str, Any]:
        """Export daily summary for analysis"""
        if date_str is None:
            date_str = datetime.utcnow().strftime("%Y%m%d")
            
        daily_file = self.storage_path.parent / f"telemetry_daily_{date_str}.json"
        
        if not daily_file.exists():
            return {"error": f"No data for date {date_str}"}
        
        try:
            with open(daily_file, 'r') as f:
                records = json.load(f)
                
            if not records:
                return {"error": "Empty data file"}
            
            # Aggregate daily stats
            total_decisions = len(records)
            total_cost = sum(r.get("cost_estimate_usd", 0) for r in records)
            avg_entropy = sum(r.get("entropy", 0) for r in records) / total_decisions
            avg_confidence = sum(r.get("confidence", 0) for r in records) / total_decisions
            
            # Action distribution
            actions = [r.get("dominant_action", "UNKNOWN") for r in records]
            action_dist = {action: actions.count(action) for action in ["BUY", "HOLD", "AVOID", "ABSTAIN"]}
            
            # High confidence decisions
            high_conf_decisions = [r for r in records if r.get("confidence", 0) > 0.6]
            
            return {
                "date": date_str,
                "total_decisions": total_decisions,
                "total_cost_usd": round(total_cost, 4),
                "avg_entropy": round(avg_entropy, 3),
                "avg_confidence": round(avg_confidence, 3),
                "action_distribution": action_dist,
                "high_confidence_count": len(high_conf_decisions),
                "high_confidence_rate": round(len(high_conf_decisions) / total_decisions * 100, 1),
                "unique_symbols": len(set(r.get("symbol", "unknown") for r in records))
            }
            
        except Exception as e:
            return {"error": f"Failed to export summary: {e}"}
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """Clean up old telemetry files"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime("%Y%m%d")
            
            pattern = "telemetry_daily_*.json"
            deleted_count = 0
            
            for file_path in self.storage_path.parent.glob(pattern):
                # Extract date from filename
                date_part = file_path.stem.split("_")[-1]
                if date_part < cutoff_str:
                    file_path.unlink()
                    deleted_count += 1
                    
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old telemetry files (keeping {days_to_keep} days)")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old telemetry files: {e}")

# Global instance
consensus_telemetry = ConsensusTelemetry()