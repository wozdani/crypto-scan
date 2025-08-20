"""
Main runner for last 10 tokens with cache and budget management
"""
import json
import os
import time
from typing import List, Dict, Any, Optional
from pipeline.last10_batch_runner import build_items_from_last10, _cache_key
from llm.multi_detector_last10_fixed import run_last10_all_detectors

# Global counter for multi-detector tokens (≥2 detectors)
_multi_detector_counter = 0
_counter_file_path = "state/multi_detector_counter.json"

def _load_counter() -> int:
    """Load counter from disk"""
    global _multi_detector_counter
    if os.path.exists(_counter_file_path):
        try:
            with open(_counter_file_path, 'r') as f:
                data = json.load(f)
                _multi_detector_counter = data.get("count", 0)
        except (json.JSONDecodeError, FileNotFoundError):
            _multi_detector_counter = 0
    return _multi_detector_counter

def _save_counter() -> None:
    """Save counter to disk"""
    global _multi_detector_counter
    os.makedirs(os.path.dirname(_counter_file_path), exist_ok=True)
    with open(_counter_file_path, 'w') as f:
        json.dump({"count": _multi_detector_counter, "last_update": time.time()}, f)

def increment_multi_detector_counter() -> int:
    """Increment counter for tokens with ≥2 detectors and return new count"""
    global _multi_detector_counter
    _load_counter()  # Ensure we have latest from disk
    _multi_detector_counter += 1
    _save_counter()
    return _multi_detector_counter

def check_and_reset_counter(force_reset: bool = False) -> int:
    """Check counter and optionally reset it. Returns current count before reset."""
    global _multi_detector_counter
    _load_counter()
    current = _multi_detector_counter
    if force_reset or current >= 10:
        _multi_detector_counter = 0
        _save_counter()
    return current

# Initialize counter on import
_load_counter()

class Last10Runner:
    """Runner with cache and budget management for last 10 tokens"""
    
    def __init__(self, cache_path: str = "cache/last10_llm_cache.json", budget_file: str = "cache/llm_budget.json"):
        self.cache_path = cache_path
        self.budget_file = budget_file
        self._cache = {}
        self._budget_data = {}
        self._load_cache()
        self._load_budget()
    
    def _load_cache(self) -> None:
        """Load LLM cache from disk"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self._cache, f, indent=2)
    
    def _load_budget(self) -> None:
        """Load budget tracking from disk"""
        if os.path.exists(self.budget_file):
            try:
                with open(self.budget_file, 'r') as f:
                    self._budget_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._budget_data = {}
        
        # Initialize default budget if not exists
        if "daily_calls" not in self._budget_data:
            self._budget_data = {
                "daily_calls": 0,
                "daily_limit": 100,  # Conservative daily limit
                "last_reset": time.strftime("%Y-%m-%d")
            }
    
    def _save_budget(self) -> None:
        """Save budget to disk"""
        os.makedirs(os.path.dirname(self.budget_file), exist_ok=True)
        with open(self.budget_file, 'w') as f:
            json.dump(self._budget_data, f, indent=2)
    
    def _check_budget(self) -> bool:
        """Check if budget allows new call"""
        today = time.strftime("%Y-%m-%d")
        
        # Reset daily counter if new day
        if self._budget_data["last_reset"] != today:
            self._budget_data["daily_calls"] = 0
            self._budget_data["last_reset"] = today
            self._save_budget()
        
        return self._budget_data["daily_calls"] < self._budget_data["daily_limit"]
    
    def _use_budget(self) -> None:
        """Increment budget usage"""
        self._budget_data["daily_calls"] += 1
        self._save_budget()
    
    def run_one_call_for_last10(self, min_trust: float = 0.0, min_liq_usd: float = 0.0) -> List[Dict[str, Any]]:
        """
        Run analysis for last 10 tokens with single LLM call
        
        Args:
            min_trust: Minimum trust score filter
            min_liq_usd: Minimum liquidity USD filter
        
        Returns:
            List of results for last 10 tokens only
        """
        # Build items from last 10 tokens only
        items = build_items_from_last10(min_trust, min_liq_usd)
        
        if not items:
            print("[LAST10 RUNNER] No items from last 10 tokens")
            return []
        
        print(f"[LAST10 RUNNER] ✅ BATCH PROCESSING: {len(items)} items from last 10 tokens")
        
        # Show breakdown by detector
        detector_counts = {}
        for item in items:
            det = item["det"]
            detector_counts[det] = detector_counts.get(det, 0) + 1
        print(f"[LAST10 RUNNER] Detector breakdown: {detector_counts}")
        
        # Show symbols being processed
        symbols = list(set(item["s"] for item in items))
        print(f"[LAST10 RUNNER] Processing symbols: {symbols}")
        
        # DISABLE CACHE for true batch processing - wszystkie items w jednym API call
        to_query = items  # Wszystkie items idą do batch processing
        cache_hits = {}   # Nie używamy cache dla LAST10
        
        print(f"[LAST10 RUNNER] BATCH MODE: Processing all {len(to_query)} items in single API call")
        print(f"[LAST10 RUNNER] Cache DISABLED for consistent batch processing")
        
        # Check budget for new queries
        if to_query and not self._check_budget():
            print("[LAST10 RUNNER] Budget exhausted - returning DEFER for new queries")
            # Return cache hits + DEFER for new queries
            results = list(cache_hits.values())
            for item in to_query:
                results.append({
                    "s": item["s"],
                    "det": item["det"],
                    "result": {
                        "d": "DEFER",
                        "c": 0.0,
                        "cl": {"ok": 0, "warn": 1},
                        "dbg": {"a": [], "p": [], "n": ["budget_exhausted"]}
                    }
                })
            return [r["result"] for r in results]
        
        # Make single LLM call for items to query
        llm_results = []
        if to_query:
            try:
                print(f"[LAST10 RUNNER] Making single LLM call for {len(to_query)} items")
                llm_response = run_last10_all_detectors(to_query)
                llm_results = llm_response.get("results", [])
                
                # Use budget
                self._use_budget()
                
                # Cache results
                result_map = {}
                for result in llm_results:
                    key = f"{result['s']}_{result['det']}"
                    result_map[key] = result
                
                # Cache by original cache key
                for item in to_query:
                    cache_key = _cache_key(item)
                    lookup_key = f"{item['s']}_{item['det']}"
                    if lookup_key in result_map:
                        self._cache[cache_key] = result_map[lookup_key]
                
                self._save_cache()
                
            except Exception as e:
                print(f"[LAST10 RUNNER] LLM call failed: {e}")
                # Return DEFER for failed queries
                for item in to_query:
                    llm_results.append({
                        "s": item["s"],
                        "det": item["det"],
                        "d": "DEFER",
                        "c": 0.0,
                        "cl": {"ok": 0, "warn": 1},
                        "dbg": {"a": [], "p": [], "n": [f"llm_error:{str(e)}"]}
                    })
        
        # Merge cache hits and LLM results
        all_results = []
        
        # Add cache hits
        for cache_result in cache_hits.values():
            all_results.append(cache_result["result"])
        
        # Add LLM results
        all_results.extend(llm_results)
        
        print(f"[LAST10 RUNNER] Returning {len(all_results)} total results")
        return all_results
    
    def aggregate_per_token(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate results per token (optional helper)
        
        Args:
            results: List of individual detector results
        
        Returns:
            Dict of {symbol: {decision, confidence, detectors}}
        """
        token_groups = {}
        
        # Group by symbol
        for result in results:
            symbol = result["s"]
            if symbol not in token_groups:
                token_groups[symbol] = []
            token_groups[symbol].append(result)
        
        # Aggregate per token
        aggregated = {}
        for symbol, symbol_results in token_groups.items():
            # Apply rule: AVOID > BUY > HOLD
            decisions = [r["d"] for r in symbol_results]
            confidences = [r["c"] for r in symbol_results]
            detectors = [r["det"] for r in symbol_results]
            
            # Decision priority
            if "AVOID" in decisions:
                final_decision = "AVOID"
            elif "BUY" in decisions:
                # Promote BUY only if confidence >= 0.7
                buy_results = [r for r in symbol_results if r["d"] == "BUY"]
                max_buy_conf = max([r["c"] for r in buy_results]) if buy_results else 0.0
                if max_buy_conf >= 0.7:
                    final_decision = "BUY"
                else:
                    final_decision = "HOLD"
            else:
                final_decision = "HOLD"
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            aggregated[symbol] = {
                "decision": final_decision,
                "confidence": round(avg_confidence, 3),
                "detectors": detectors,
                "detector_count": len(detectors)
            }
        
        return aggregated

# Global singleton
_last10_runner = None

def get_last10_runner() -> Last10Runner:
    """Get global Last10Runner instance"""
    global _last10_runner
    if _last10_runner is None:
        _last10_runner = Last10Runner()
    return _last10_runner