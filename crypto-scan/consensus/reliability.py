# consensus/reliability.py
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EMA_ALPHA = 0.1
_store = defaultdict(lambda: 0.6)  # domyślna wiarygodność
_storage_path = Path("state/agent_reliability.json")

def _load_store():
    """Load reliability store from disk"""
    global _store
    try:
        if _storage_path.exists():
            with open(_storage_path, 'r') as f:
                data = json.load(f)
                _store.update(data.get("reliability", {}))
    except Exception as e:
        print(f"Warning: Could not load reliability store: {e}")

def _save_store():
    """Save reliability store to disk"""
    try:
        _storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "reliability": dict(_store),
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
        with open(_storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save reliability store: {e}")

def get(agent_name: str) -> float:
    """Get current reliability for agent"""
    if not _store:
        _load_store()
    return _store[agent_name]

def update(agent_name: str, outcome: str):
    """Update agent reliability based on outcome
    
    Args:
        agent_name: Name of agent (Analyzer, Reasoner, Voter, Debater)
        outcome: One of {"TP","FP","TN","FN"} for True/False Positive/Negative
    """
    if not _store:
        _load_store()
    
    # outcome ∈ {"TP","FP","TN","FN"} mapowane do targetu 0..1
    target = {"TP": 1.0, "TN": 0.8, "FP": 0.2, "FN": 0.4}.get(outcome, 0.6)
    prev = _store[agent_name]
    _store[agent_name] = (1 - EMA_ALPHA) * prev + EMA_ALPHA * target
    
    # Persist to disk
    _save_store()
    
    print(f"[RELIABILITY] {agent_name}: {prev:.3f} → {_store[agent_name]:.3f} (outcome: {outcome})")

def get_all_reliabilities() -> dict:
    """Get all current reliabilities"""
    if not _store:
        _load_store()
    return dict(_store)

def reset_agent(agent_name: str):
    """Reset agent reliability to default"""
    if not _store:
        _load_store()
    _store[agent_name] = 0.6
    _save_store()
    print(f"[RELIABILITY] {agent_name}: reset to 0.6")

# Initialize on import
_load_store()