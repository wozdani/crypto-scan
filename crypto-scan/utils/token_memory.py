"""
Token Memory System for Historical Scoring Analysis
Tracks token behavior patterns over 4 days for adaptive decision making
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

MEMORY_PATH = "data/token_profile_store.json"
MEMORY_LOOKBACK_HOURS = 96  # 4 days

def load_token_memory() -> Dict[str, List[Dict]]:
    """Load token memory from persistent storage with enhanced error handling"""
    try:
        if not os.path.exists(MEMORY_PATH):
            return {}
            
        # FIX 3: Enhanced JSON loading with corruption detection
        with open(MEMORY_PATH, "r") as f:
            content = f.read().strip()
            
        if not content:
            print(f"[MEMORY] Empty file {MEMORY_PATH}, initializing new memory")
            return {}
            
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                print(f"[MEMORY ERROR] Invalid data structure in {MEMORY_PATH}, reinitializing")
                _backup_corrupted_file(MEMORY_PATH)
                return {}
            return data
            
        except json.JSONDecodeError as je:
            print(f"[MEMORY ERROR] JSON corruption in {MEMORY_PATH}: {je}")
            print(f"[MEMORY ERROR] Error at line {je.lineno}, column {je.colno}: {je.msg}")
            _backup_corrupted_file(MEMORY_PATH)
            return {}
            
    except Exception as e:
        logging.error(f"Failed to load token memory: {e}")
        return {}


def _backup_corrupted_file(file_path: str):
    """Backup corrupted file and create fresh one"""
    try:
        import shutil
        backup_path = f"{file_path}.corrupted.{int(datetime.now().timestamp())}"
        shutil.copy2(file_path, backup_path)
        print(f"[MEMORY BACKUP] Corrupted file backed up to: {backup_path}")
        
        # Initialize fresh file
        with open(file_path, "w") as f:
            json.dump({}, f, indent=2)
        print(f"[MEMORY] Created fresh memory file: {file_path}")
        
    except Exception as e:
        print(f"[MEMORY BACKUP ERROR] Failed to backup corrupted file: {e}")

def save_token_memory(data: Dict[str, List[Dict]]):
    """Save token memory to persistent storage"""
    try:
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        with open(MEMORY_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save token memory: {e}")

def update_token_memory(symbol: str, entry: Dict[str, Any]):
    """
    Update token memory with new scoring entry
    
    Args:
        symbol: Trading symbol
        entry: Dictionary containing scoring data and decision info
    """
    try:
        data = load_token_memory()
        token_mem = data.get(symbol, [])

        # Add timestamp as ISO format
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry["symbol"] = symbol  # Ensure symbol is stored

        token_mem.append(entry)

        # Remove data older than MEMORY_LOOKBACK_HOURS
        cutoff = datetime.now(timezone.utc) - timedelta(hours=MEMORY_LOOKBACK_HOURS)
        token_mem = [
            e for e in token_mem
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]

        data[symbol] = token_mem
        save_token_memory(data)
        
        logging.debug(f"Updated token memory for {symbol}: {len(token_mem)} entries")
        
    except Exception as e:
        logging.error(f"Failed to update token memory for {symbol}: {e}")

def get_token_memory(symbol: str) -> List[Dict[str, Any]]:
    """
    Get historical memory entries for a token
    
    Args:
        symbol: Trading symbol
        
    Returns:
        List of historical entries for the token
    """
    try:
        data = load_token_memory()
        return data.get(symbol, [])
    except Exception as e:
        logging.error(f"Failed to get token memory for {symbol}: {e}")
        return []

def analyze_token_behavior(symbol: str) -> Dict[str, Any]:
    """
    Analyze token behavior patterns from memory
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with behavior analysis results
    """
    try:
        history = get_token_memory(symbol)
        
        if not history:
            return {
                "total_entries": 0,
                "recent_bad_decisions": 0,
                "success_rate": 0.0,
                "avg_score": 0.0,
                "behavior_modifier": 0.0,
                "common_phases": [],
                "analysis": "No historical data available"
            }
        
        # Analyze decision patterns
        total_entries = len(history)
        entry_decisions = [h.get("decision", "unknown") for h in history]
        scores = [h.get("tjde_score", 0) for h in history if h.get("tjde_score")]
        
        # Count failed "consider_entry" decisions
        recent_bad_decisions = [
            h for h in history
            if h.get("decision") == "consider_entry" and h.get("result_after_2h") == "fail"
        ]
        
        # Calculate success rate for evaluated entries
        evaluated_entries = [h for h in history if h.get("result_after_2h") is not None]
        success_rate = 0.0
        if evaluated_entries:
            successful = [h for h in evaluated_entries if h.get("result_after_2h") == "success"]
            success_rate = len(successful) / len(evaluated_entries)
        
        # Calculate behavior modifier (penalty for repeated failures)
        behavior_modifier = -0.05 * len(recent_bad_decisions)
        
        # Find common phases/setups
        phases = [h.get("phase", "unknown") for h in history if h.get("phase")]
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        common_phases = sorted(phase_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "total_entries": total_entries,
            "recent_bad_decisions": len(recent_bad_decisions),
            "success_rate": success_rate,
            "avg_score": avg_score,
            "behavior_modifier": behavior_modifier,
            "common_phases": common_phases,
            "last_24h_entries": len([h for h in history if 
                                   datetime.fromisoformat(h["timestamp"]) > 
                                   datetime.now(timezone.utc) - timedelta(hours=24)]),
            "analysis": f"Token shows {'poor' if success_rate < 0.4 else 'good' if success_rate > 0.7 else 'mixed'} historical performance"
        }
        
    except Exception as e:
        logging.error(f"Failed to analyze token behavior for {symbol}: {e}")
        return {"error": str(e), "behavior_modifier": 0.0}

def set_result_for_last_entry(symbol: str, result_label: str):
    """
    Update the result for the most recent entry (used by feedback loop)
    
    Args:
        symbol: Trading symbol
        result_label: "success" or "fail"
    """
    try:
        data = load_token_memory()
        if symbol not in data or not data[symbol]:
            logging.debug(f"No memory entries found for {symbol}")
            return
            
        # Update the most recent entry
        data[symbol][-1]["result_after_2h"] = result_label
        data[symbol][-1]["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        
        save_token_memory(data)
        logging.debug(f"Updated result for {symbol}: {result_label}")
        
    except Exception as e:
        logging.error(f"Failed to set result for {symbol}: {e}")

def get_memory_stats() -> Dict[str, Any]:
    """Get statistics about token memory usage"""
    try:
        data = load_token_memory()
        
        total_symbols = len(data)
        total_entries = sum(len(entries) for entries in data.values())
        
        # Find most active tokens
        most_active = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        
        # Count evaluated entries
        evaluated_count = 0
        for entries in data.values():
            evaluated_count += len([e for e in entries if e.get("result_after_2h") is not None])
        
        return {
            "total_symbols": total_symbols,
            "total_entries": total_entries,
            "evaluated_entries": evaluated_count,
            "evaluation_rate": evaluated_count / total_entries if total_entries > 0 else 0,
            "most_active_tokens": [(symbol, len(entries)) for symbol, entries in most_active],
            "memory_file_size": os.path.getsize(MEMORY_PATH) if os.path.exists(MEMORY_PATH) else 0
        }
        
    except Exception as e:
        logging.error(f"Failed to get memory stats: {e}")
        return {"error": str(e)}

def cleanup_token_memory(max_entries_per_token: int = 100):
    """
    Clean up token memory by limiting entries per token
    
    Args:
        max_entries_per_token: Maximum entries to keep per token
    """
    try:
        data = load_token_memory()
        cleaned_count = 0
        
        for symbol in data:
            if len(data[symbol]) > max_entries_per_token:
                # Keep only the most recent entries
                data[symbol] = data[symbol][-max_entries_per_token:]
                cleaned_count += 1
        
        if cleaned_count > 0:
            save_token_memory(data)
            logging.info(f"Cleaned token memory for {cleaned_count} symbols")
            
    except Exception as e:
        logging.error(f"Failed to cleanup token memory: {e}")

def export_token_profile(symbol: str) -> Optional[Dict]:
    """Export complete profile for a specific token"""
    try:
        history = get_token_memory(symbol)
        analysis = analyze_token_behavior(symbol)
        
        return {
            "symbol": symbol,
            "history": history,
            "analysis": analysis,
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logging.error(f"Failed to export token profile for {symbol}: {e}")
        return None