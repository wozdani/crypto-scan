"""
TOP 5 TJDE Token Selector
Centralizes selection logic for training data generation
"""

from typing import List, Dict, Set
import json
import os
from datetime import datetime

class TOP5TokenSelector:
    """Centralized TOP 5 token selection for Vision-AI training"""
    
    def __init__(self):
        self._current_top5: Set[str] = set()
        self._session_id = None
        self._selection_timestamp = None
    
    def select_top5_tokens(self, tjde_results: List[Dict], prefer_complete_data: bool = True) -> List[Dict]:
        """
        Select TOP 5 tokens by TJDE score for training data generation
        
        Args:
            tjde_results: List of scan results with TJDE scores
            prefer_complete_data: Prefer tokens with complete 15M+5M candle data
            
        Returns:
            Top 5 results sorted by TJDE score
        """
        # Filter valid results with TJDE scores and exclude invalid symbols
        valid_results = [
            r for r in tjde_results 
            if r.get('tjde_score', 0) > 0 
            and not r.get('invalid_symbol', False)  # 🔒 CRITICAL: Block invalid symbols from TOP5
        ]
        
        if not valid_results:
            print("[TOP5 SELECTOR] No valid TJDE results for training data")
            return []
        
        # Prefer tokens with complete data if enabled
        if prefer_complete_data:
            complete_data_results = [r for r in valid_results if not r.get('partial_data', False)]
            if len(complete_data_results) >= 5:
                valid_results = complete_data_results
                print(f"[TOP5 SELECTOR] Using {len(complete_data_results)} tokens with complete 15M+5M data")
            else:
                incomplete_count = len(valid_results) - len(complete_data_results)
                print(f"[TOP5 SELECTOR] ⚠️ Including {incomplete_count} tokens with partial data (insufficient complete tokens)")
        
        # Sort by TJDE score descending and take TOP 5
        top5 = sorted(valid_results, key=lambda x: x.get('tjde_score', 0), reverse=True)[:5]
        
        # Update current selection
        self._current_top5 = {result['symbol'] for result in top5}
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._selection_timestamp = datetime.now().isoformat()
        
        print(f"[TOP5 SELECTOR] Selected TOP 5 tokens for training data:")
        for i, result in enumerate(top5, 1):
            symbol = result['symbol']
            tjde_score = result['tjde_score']
            decision = result.get('tjde_decision', 'unknown')
            print(f"  {i}. {symbol}: TJDE {tjde_score:.3f} ({decision})")
        
        # Save selection for reference
        self._save_selection_log(top5)
        
        return top5
    
    def is_token_in_top5(self, symbol: str) -> bool:
        """
        Check if token is in current TOP 5 selection
        
        Args:
            symbol: Token symbol to check
            
        Returns:
            True if token is in TOP 5
        """
        return symbol in self._current_top5
    
    def get_current_top5_symbols(self) -> Set[str]:
        """Get current TOP 5 symbol set"""
        return self._current_top5.copy()
    
    def should_generate_training_data(self, symbol: str, tjde_score: float = None) -> bool:
        """
        Determine if training data should be generated for this token
        
        Args:
            symbol: Token symbol
            tjde_score: Optional TJDE score for additional validation
            
        Returns:
            True if training data should be generated
        """
        if not self._current_top5:
            print(f"[TOP5 SELECTOR] ⚠️ No TOP 5 selection available - skipping {symbol}")
            return False
            
        is_in_top5 = symbol in self._current_top5
        
        if not is_in_top5:
            print(f"[TOP5 SELECTOR] ⏭️ Skipping training chart for {symbol} (not in TOP 5 TJDE)")
            return False
            
        print(f"[TOP5 SELECTOR] ✅ {symbol} is in TOP 5 - generating training data")
        return True
    
    def _save_selection_log(self, top5_results: List[Dict]):
        """Save TOP 5 selection to log file"""
        try:
            log_dir = "data/top5_selections"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = f"{log_dir}/selection_{self._session_id}.json"
            
            selection_data = {
                "session_id": self._session_id,
                "timestamp": self._selection_timestamp,
                "top5_tokens": [
                    {
                        "rank": i + 1,
                        "symbol": result['symbol'],
                        "tjde_score": result['tjde_score'],
                        "tjde_decision": result.get('tjde_decision', 'unknown'),
                        "market_phase": result.get('market_phase', 'unknown')
                    }
                    for i, result in enumerate(top5_results)
                ],
                "total_candidates": len(top5_results)
            }
            
            with open(log_file, 'w') as f:
                json.dump(selection_data, f, indent=2)
                
            print(f"[TOP5 SELECTOR] Selection logged: {log_file}")
            
        except Exception as e:
            print(f"[TOP5 SELECTOR ERROR] Failed to save selection log: {e}")

# Global instance
_top5_selector = TOP5TokenSelector()

def get_top5_selector() -> TOP5TokenSelector:
    """Get global TOP5 selector instance"""
    return _top5_selector

def select_top5_tjde_tokens(tjde_results: List[Dict]) -> List[Dict]:
    """
    Convenience function to select TOP 5 TJDE tokens
    
    Args:
        tjde_results: List of scan results with TJDE scores
        
    Returns:
        Top 5 results sorted by TJDE score
    """
    return _top5_selector.select_top5_tokens(tjde_results)

def should_generate_training_data(symbol: str, tjde_score: float = None) -> bool:
    """
    Convenience function to check if training data should be generated
    
    Args:
        symbol: Token symbol
        tjde_score: Optional TJDE score
        
    Returns:
        True if token is in TOP 5 and should generate training data
    """
    return _top5_selector.should_generate_training_data(symbol, tjde_score)

def warn_about_non_top5_generation(symbol: str, context: str = ""):
    """
    Log warning about training data generation for non-TOP5 token
    
    Args:
        symbol: Token symbol
        context: Additional context about where this occurred
    """
    print(f"🧨 [TOP5 VIOLATION] {symbol}: Training data generated outside TOP 5 selection! Context: {context}")
    print(f"    Current TOP 5: {_top5_selector.get_current_top5_symbols()}")
    print(f"    This reduces dataset quality and wastes disk space.")

def get_top5_status_report() -> Dict:
    """Get current TOP 5 selection status"""
    return {
        "current_top5": list(_top5_selector.get_current_top5_symbols()),
        "session_id": _top5_selector._session_id,
        "selection_timestamp": _top5_selector._selection_timestamp,
        "has_selection": bool(_top5_selector._current_top5)
    }