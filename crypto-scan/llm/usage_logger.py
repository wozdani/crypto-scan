# llm/usage_logger.py
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

USAGE_LOG_PATH = Path("state/llm_usage.jsonl")

def log_usage(resp, agent_name: str, token: str = "unknown", model: str = "gpt-4o"):
    """Log LLM usage statistics for cost tracking"""
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            
            # Calculate estimated cost (rough estimates for gpt-4o)
            cost_per_prompt_token = 0.0000025  # $2.50 per 1M tokens
            cost_per_completion_token = 0.00001  # $10.00 per 1M tokens
            estimated_cost = (prompt_tokens * cost_per_prompt_token) + (completion_tokens * cost_per_completion_token)
            
            # Log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent_name": agent_name,
                "token": token,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(estimated_cost, 6)
            }
            
            # Append to JSONL file
            USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(USAGE_LOG_PATH, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            
            print(f"[LLM USAGE] {agent_name} prompt={prompt_tokens} completion={completion_tokens} "
                  f"total={total_tokens} cost=${estimated_cost:.6f} token={token}")
    except Exception as e:
        print(f"[LLM USAGE] Error logging usage: {e}")

def get_daily_usage_summary(date_str: Optional[str] = None) -> dict:
    """Get daily usage summary for dashboard"""
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    try:
        if not USAGE_LOG_PATH.exists():
            return {"error": "No usage data available"}
        
        total_tokens = 0
        total_cost = 0.0
        agent_breakdown = {}
        
        with open(USAGE_LOG_PATH, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("timestamp", "").startswith(date_str):
                        total_tokens += entry.get("total_tokens", 0)
                        total_cost += entry.get("estimated_cost_usd", 0)
                        
                        agent = entry.get("agent_name", "unknown")
                        if agent not in agent_breakdown:
                            agent_breakdown[agent] = {"tokens": 0, "cost": 0.0, "calls": 0}
                        
                        agent_breakdown[agent]["tokens"] += entry.get("total_tokens", 0)
                        agent_breakdown[agent]["cost"] += entry.get("estimated_cost_usd", 0)
                        agent_breakdown[agent]["calls"] += 1
                except:
                    continue
        
        return {
            "date": date_str,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "agent_breakdown": agent_breakdown,
            "avg_cost_per_call": round(total_cost / max(sum(a["calls"] for a in agent_breakdown.values()), 1), 6)
        }
    except Exception as e:
        return {"error": f"Failed to get usage summary: {e}"}

def cleanup_old_logs(days_to_keep: int = 7):
    """Cleanup old usage logs to manage disk space"""
    try:
        if not USAGE_LOG_PATH.exists():
            return
        
        cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
        cutoff_str = cutoff_date.isoformat()
        
        temp_path = USAGE_LOG_PATH.with_suffix('.tmp')
        kept_entries = 0
        
        with open(USAGE_LOG_PATH, 'r') as infile, open(temp_path, 'w') as outfile:
            for line in infile:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("timestamp", "") >= cutoff_str:
                        outfile.write(line)
                        kept_entries += 1
                except:
                    continue
        
        temp_path.replace(USAGE_LOG_PATH)
        print(f"[USAGE CLEANUP] Kept {kept_entries} entries, removed logs older than {days_to_keep} days")
    except Exception as e:
        print(f"[USAGE CLEANUP] Error: {e}")