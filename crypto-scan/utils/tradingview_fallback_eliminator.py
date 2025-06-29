"""
TradingView Fallback Eliminator - Complete Matplotlib Fallback Removal
Ensures no matplotlib charts are generated when TradingView fails
"""

import os
import logging
from typing import Optional, Dict
from datetime import datetime

class TradingViewFallbackEliminator:
    """Eliminates all matplotlib fallbacks ensuring TradingView-only system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failed_charts_log = "data/failed_tv_charts.json"
        
    def handle_tradingview_failure(
        self, 
        symbol: str, 
        phase: str = "unknown", 
        setup: str = "unknown", 
        score: float = 0.0,
        error_reason: str = "unknown_error",
        output_dir: str = "training_data/charts"
    ) -> Optional[str]:
        """
        Handle TradingView generation failure WITHOUT matplotlib fallback
        Creates placeholder file instead of synthetic chart
        
        Args:
            symbol: Trading symbol
            phase: Market phase
            setup: Setup type  
            score: TJDE score
            error_reason: Reason for failure
            output_dir: Output directory
            
        Returns:
            Path to placeholder file or None
        """
        try:
            # Log the failure
            self._log_failed_generation(symbol, error_reason, phase, setup, score)
            
            # Create placeholder file instead of matplotlib fallback
            placeholder_path = self._create_failure_placeholder(
                symbol, phase, setup, score, error_reason, output_dir
            )
            
            # Ensure NO matplotlib fallback is called
            self.logger.warning(f"🚫 NO MATPLOTLIB FALLBACK - TradingView-only system enforced")
            
            return placeholder_path
            
        except Exception as e:
            self.logger.error(f"[FALLBACK ELIMINATOR] {symbol} → {e}")
            return None
    
    def _create_failure_placeholder(
        self, 
        symbol: str, 
        phase: str, 
        setup: str, 
        score: float, 
        error_reason: str,
        output_dir: str
    ) -> str:
        """Create text placeholder for failed TradingView generation"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Proper filename format as requested
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{phase}-{setup}_score-{int(score*1000)}_TRADINGVIEW_FAILED_placeholder.txt"
            placeholder_path = os.path.join(output_dir, filename)
            
            # Create placeholder content
            content = f"""TRADINGVIEW SCREENSHOT FAILED
Symbol: {symbol}
Phase: {phase}
Setup: {setup}  
Score: {score:.3f}
Error: {error_reason}
Timestamp: {datetime.now().isoformat()}

Note: This is a placeholder file created because TradingView screenshot generation failed.
No matplotlib fallback was used to maintain dataset quality.
"""
            
            with open(placeholder_path, 'w') as f:
                f.write(content)
                
            self.logger.info(f"⚠️ TradingView FAILED - Created placeholder: {filename}")
            return placeholder_path
            
        except Exception as e:
            self.logger.error(f"[PLACEHOLDER] {symbol} → {e}")
            return None
    
    def _log_failed_generation(
        self, 
        symbol: str, 
        error_reason: str, 
        phase: str, 
        setup: str, 
        score: float
    ):
        """Log failed TradingView generation for analysis"""
        try:
            import json
            
            failure_entry = {
                "symbol": symbol,
                "error_reason": error_reason,
                "phase": phase,
                "setup": setup,
                "score": score,
                "timestamp": datetime.now().isoformat(),
                "fallback_prevented": True
            }
            
            # Load existing failures
            failures = []
            if os.path.exists(self.failed_charts_log):
                with open(self.failed_charts_log, 'r') as f:
                    failures = json.load(f)
            
            # Add new failure
            failures.append(failure_entry)
            
            # Keep only last 1000 failures
            if len(failures) > 1000:
                failures = failures[-1000:]
            
            # Save updated failures
            os.makedirs(os.path.dirname(self.failed_charts_log), exist_ok=True)
            with open(self.failed_charts_log, 'w') as f:
                json.dump(failures, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"[FAILURE LOG] {symbol} → {e}")
    
    def prevent_matplotlib_fallback(self, symbol: str) -> bool:
        """
        Explicitly prevent any matplotlib chart generation
        Returns False to indicate no chart should be generated
        """
        self.logger.warning(f"🚫 [MATPLOTLIB BLOCKED] {symbol} → Matplotlib fallback prevented")
        return False
    
    def get_failure_statistics(self) -> Dict:
        """Get statistics about TradingView generation failures"""
        try:
            import json
            
            if not os.path.exists(self.failed_charts_log):
                return {"total_failures": 0, "error_types": {}}
            
            with open(self.failed_charts_log, 'r') as f:
                failures = json.load(f)
            
            # Analyze error types
            error_types = {}
            for failure in failures:
                error = failure.get('error_reason', 'unknown')
                error_types[error] = error_types.get(error, 0) + 1
            
            return {
                "total_failures": len(failures),
                "error_types": error_types,
                "recent_failures": failures[-10:] if failures else []
            }
            
        except Exception as e:
            self.logger.error(f"[FAILURE STATS] → {e}")
            return {"total_failures": 0, "error_types": {}}

# Global instance
_fallback_eliminator = None

def get_fallback_eliminator() -> TradingViewFallbackEliminator:
    """Get global fallback eliminator instance"""
    global _fallback_eliminator
    if _fallback_eliminator is None:
        _fallback_eliminator = TradingViewFallbackEliminator()
    return _fallback_eliminator

def handle_tradingview_failure_safe(
    symbol: str, 
    error_reason: str, 
    phase: str = "unknown", 
    setup: str = "unknown", 
    score: float = 0.0
) -> Optional[str]:
    """
    Safe handler for TradingView failures - prevents matplotlib fallback
    
    Args:
        symbol: Trading symbol
        error_reason: Reason for TradingView failure
        phase: Market phase
        setup: Setup type
        score: TJDE score
        
    Returns:
        Path to placeholder file or None
    """
    eliminator = get_fallback_eliminator()
    return eliminator.handle_tradingview_failure(
        symbol, phase, setup, score, error_reason
    )

def prevent_matplotlib_fallback(symbol: str) -> bool:
    """Prevent any matplotlib chart generation"""
    eliminator = get_fallback_eliminator()
    return eliminator.prevent_matplotlib_fallback(symbol)