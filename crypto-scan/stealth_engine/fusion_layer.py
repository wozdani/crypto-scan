#!/usr/bin/env python3
"""
Multi-Detector Fusion Engine - Stage 5/7
Unified decision system combining CaliforniumWhale AI, DiamondWhale AI, and WhaleCLIP
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusionEngine:
    """
    Multi-Detector Fusion Engine combining three advanced AI detectors
    """
    
    def __init__(self, config_path: str = "crypto-scan/cache"):
        """Initialize Fusion Engine with configurable weights"""
        self.config_path = config_path
        self.weights_file = os.path.join(config_path, "fusion_weights.json")
        self.fusion_history_file = os.path.join(config_path, "fusion_history.json")
        
        # Default weights (can be adjusted by feedback loop)
        self.default_weights = {
            "californium": 0.4,  # Temporal graph + QIRL analysis
            "diamond": 0.35,     # DiamondWhale AI temporal patterns  
            "whaleclip": 0.25    # Behavioral analysis
        }
        
        # Default alert threshold
        self.default_threshold = 0.65
        
        # Load or initialize weights
        self.weights = self.load_weights()
        self.threshold = self.load_threshold()
        
        # Ensure cache directory exists
        os.makedirs(config_path, exist_ok=True)
    
    def load_weights(self) -> Dict[str, float]:
        """Load fusion weights from file or use defaults"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    weights_data = json.load(f)
                    weights = weights_data.get('weights', self.default_weights)
                    logger.info(f"[FUSION] Loaded weights: {weights}")
                    return weights
        except Exception as e:
            logger.warning(f"[FUSION] Error loading weights: {e}")
        
        logger.info(f"[FUSION] Using default weights: {self.default_weights}")
        return self.default_weights.copy()
    
    def load_threshold(self) -> float:
        """Load fusion threshold from file or use default"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    weights_data = json.load(f)
                    threshold = weights_data.get('threshold', self.default_threshold)
                    logger.info(f"[FUSION] Loaded threshold: {threshold}")
                    return threshold
        except Exception as e:
            logger.warning(f"[FUSION] Error loading threshold: {e}")
        
        logger.info(f"[FUSION] Using default threshold: {self.default_threshold}")
        return self.default_threshold
    
    def save_weights(self):
        """Save current weights and threshold to file"""
        try:
            weights_data = {
                "weights": self.weights,
                "threshold": self.threshold,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "version": "5.7"
            }
            
            with open(self.weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
                
            logger.info(f"[FUSION] Saved weights and threshold to {self.weights_file}")
        except Exception as e:
            logger.error(f"[FUSION] Error saving weights: {e}")
    
    def get_detector_scores(self, symbol: str) -> Tuple[float, float, float]:
        """
        Get scores from all three detectors
        
        Args:
            symbol: Token symbol (e.g., 'BTCUSDT')
            
        Returns:
            Tuple of (californium_score, diamond_score, whaleclip_score)
        """
        californium_score = 0.0
        diamond_score = 0.0
        whaleclip_score = 0.0
        
        try:
            # CaliforniumWhale AI Score
            from stealth_engine.stealth_engine import californium_whale_score
            californium_score = californium_whale_score(symbol)
            logger.info(f"[FUSION] {symbol}: CaliforniumWhale score = {californium_score:.3f}")
        except Exception as e:
            logger.warning(f"[FUSION] {symbol}: CaliforniumWhale error: {e}")
        
        try:
            # DiamondWhale AI Score - using existing function from stealth_engine
            from stealth.diamond.diamond_whale_detect import run_diamond_detector
            from stealth_engine.stealth_engine import get_contract
            
            # Get contract for symbol
            contract_info = get_contract(symbol)
            if contract_info and contract_info.get('address'):
                diamond_result = run_diamond_detector(
                    contract_info['address'], 
                    contract_info.get('chain', 'ethereum')
                )
                diamond_score = diamond_result.get('diamond_score', 0.0)
                logger.info(f"[FUSION] {symbol}: DiamondWhale score = {diamond_score:.3f}")
            else:
                logger.info(f"[FUSION] {symbol}: No contract found for DiamondWhale analysis")
        except Exception as e:
            logger.warning(f"[FUSION] {symbol}: DiamondWhale error: {e}")
        
        try:
            # WhaleCLIP Score - mock implementation (placeholder for future integration)
            # This would integrate with actual WhaleCLIP behavioral analysis
            whaleclip_score = 0.0  # Placeholder - to be implemented
            logger.info(f"[FUSION] {symbol}: WhaleCLIP score = {whaleclip_score:.3f}")
        except Exception as e:
            logger.warning(f"[FUSION] {symbol}: WhaleCLIP error: {e}")
        
        return californium_score, diamond_score, whaleclip_score
    
    def calculate_fusion_score(self, californium_score: float, diamond_score: float, 
                             whaleclip_score: float) -> float:
        """
        Calculate weighted fusion score from individual detector scores
        
        Args:
            californium_score: CaliforniumWhale AI score (0.0-1.0)
            diamond_score: DiamondWhale AI score (0.0-1.0) 
            whaleclip_score: WhaleCLIP score (0.0-1.0)
            
        Returns:
            Weighted fusion score (0.0-1.0)
        """
        fusion_score = (
            self.weights["californium"] * californium_score +
            self.weights["diamond"] * diamond_score +
            self.weights["whaleclip"] * whaleclip_score
        )
        
        # Ensure score is within valid range
        fusion_score = max(0.0, min(1.0, fusion_score))
        
        logger.info(f"[FUSION] Weighted calculation:")
        logger.info(f"  Californium: {californium_score:.3f} × {self.weights['californium']:.2f} = {californium_score * self.weights['californium']:.3f}")
        logger.info(f"  Diamond: {diamond_score:.3f} × {self.weights['diamond']:.2f} = {diamond_score * self.weights['diamond']:.3f}")
        logger.info(f"  WhaleCLIP: {whaleclip_score:.3f} × {self.weights['whaleclip']:.2f} = {whaleclip_score * self.weights['whaleclip']:.3f}")
        logger.info(f"  Final Fusion Score: {fusion_score:.3f}")
        
        return fusion_score
    
    def should_alert(self, fusion_score: float) -> bool:
        """
        Determine if fusion score triggers alert
        
        Args:
            fusion_score: Calculated fusion score
            
        Returns:
            True if alert should be triggered
        """
        should_alert = fusion_score >= self.threshold
        logger.info(f"[FUSION] Alert decision: {should_alert} (score: {fusion_score:.3f}, threshold: {self.threshold:.3f})")
        return should_alert
    
    def fusion_decision(self, symbol: str) -> Dict[str, Any]:
        """
        Main fusion decision function combining all three detectors
        
        Args:
            symbol: Token symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with fusion analysis results
        """
        logger.info(f"[FUSION] Starting multi-detector analysis for {symbol}")
        
        try:
            # Get scores from all detectors
            californium_score, diamond_score, whaleclip_score = self.get_detector_scores(symbol)
            
            # Calculate weighted fusion score
            fusion_score = self.calculate_fusion_score(californium_score, diamond_score, whaleclip_score)
            
            # Determine alert decision
            alert_decision = self.should_alert(fusion_score)
            
            # Determine confidence level
            confidence = self.get_confidence_level(fusion_score)
            
            # Create result
            result = {
                "symbol": symbol,
                "fusion_score": round(fusion_score, 3),
                "should_alert": alert_decision,
                "confidence": confidence,
                "threshold": self.threshold,
                "details": {
                    "californium": round(californium_score, 3),
                    "diamond": round(diamond_score, 3),
                    "whaleclip": round(whaleclip_score, 3)
                },
                "weights": self.weights.copy(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save to history
            self.save_fusion_history(result)
            
            logger.info(f"[FUSION] {symbol}: Decision complete - Alert: {alert_decision}, Score: {fusion_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"[FUSION] Error in fusion_decision for {symbol}: {e}")
            return {
                "symbol": symbol,
                "fusion_score": 0.0,
                "should_alert": False,
                "confidence": "ERROR",
                "error": str(e),
                "details": {"californium": 0.0, "diamond": 0.0, "whaleclip": 0.0}
            }
    
    def get_confidence_level(self, fusion_score: float) -> str:
        """
        Determine confidence level based on fusion score
        
        Args:
            fusion_score: Calculated fusion score
            
        Returns:
            Confidence level string
        """
        if fusion_score >= 0.8:
            return "VERY_HIGH"
        elif fusion_score >= 0.7:
            return "HIGH"
        elif fusion_score >= 0.5:
            return "MEDIUM"
        elif fusion_score >= 0.3:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def save_fusion_history(self, result: Dict[str, Any]):
        """Save fusion result to history file"""
        try:
            # Load existing history
            history = []
            if os.path.exists(self.fusion_history_file):
                try:
                    with open(self.fusion_history_file, 'r') as f:
                        history = json.load(f)
                except:
                    history = []
            
            # Add new result
            history.append(result)
            
            # Keep only last 1000 results
            if len(history) > 1000:
                history = history[-1000:]
            
            # Save updated history
            with open(self.fusion_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"[FUSION] Error saving history: {e}")
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion system statistics"""
        try:
            if not os.path.exists(self.fusion_history_file):
                return {
                    "total_decisions": 0,
                    "alerts_triggered": 0,
                    "avg_fusion_score": 0.0,
                    "confidence_distribution": {},
                    "detector_avg_scores": {"californium": 0.0, "diamond": 0.0, "whaleclip": 0.0}
                }
            
            with open(self.fusion_history_file, 'r') as f:
                history = json.load(f)
            
            if not history:
                return {
                    "total_decisions": 0,
                    "alerts_triggered": 0,
                    "avg_fusion_score": 0.0,
                    "confidence_distribution": {},
                    "detector_avg_scores": {"californium": 0.0, "diamond": 0.0, "whaleclip": 0.0}
                }
            
            # Calculate statistics
            total_decisions = len(history)
            alerts_triggered = sum(1 for h in history if h.get('should_alert', False))
            avg_fusion_score = sum(h.get('fusion_score', 0) for h in history) / total_decisions
            
            # Confidence distribution
            confidence_dist = {}
            for h in history:
                conf = h.get('confidence', 'UNKNOWN')
                confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
            
            # Detector average scores
            detector_avgs = {
                "californium": sum(h.get('details', {}).get('californium', 0) for h in history) / total_decisions,
                "diamond": sum(h.get('details', {}).get('diamond', 0) for h in history) / total_decisions,
                "whaleclip": sum(h.get('details', {}).get('whaleclip', 0) for h in history) / total_decisions
            }
            
            return {
                "total_decisions": total_decisions,
                "alerts_triggered": alerts_triggered,
                "alert_rate": round(alerts_triggered / total_decisions * 100, 2) if total_decisions > 0 else 0,
                "avg_fusion_score": round(avg_fusion_score, 3),
                "confidence_distribution": confidence_dist,
                "detector_avg_scores": {k: round(v, 3) for k, v in detector_avgs.items()},
                "current_weights": self.weights,
                "current_threshold": self.threshold
            }
            
        except Exception as e:
            logger.error(f"[FUSION] Error getting statistics: {e}")
            return {"error": str(e)}
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update fusion weights"""
        # Validate weights
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"[FUSION] Weights don't sum to 1.0: {total_weight}")
            # Normalize weights
            for key in new_weights:
                new_weights[key] = new_weights[key] / total_weight
        
        self.weights.update(new_weights)
        self.save_weights()
        logger.info(f"[FUSION] Updated weights: {self.weights}")
    
    def update_threshold(self, new_threshold: float):
        """Update fusion alert threshold"""
        self.threshold = max(0.0, min(1.0, new_threshold))
        self.save_weights()
        logger.info(f"[FUSION] Updated threshold: {self.threshold}")

# Global instance
_fusion_engine = None

def get_fusion_engine() -> FusionEngine:
    """Get singleton FusionEngine instance"""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = FusionEngine()
    return _fusion_engine

def fusion_decision(symbol: str) -> Dict[str, Any]:
    """
    Convenience function for multi-detector fusion decision
    
    Args:
        symbol: Token symbol (e.g., 'BTCUSDT')
        
    Returns:
        Fusion analysis result
    """
    engine = get_fusion_engine()
    return engine.fusion_decision(symbol)

def format_fusion_alert_message(result: Dict[str, Any]) -> str:
    """
    Format fusion alert message for Telegram
    
    Args:
        result: Fusion decision result
        
    Returns:
        Formatted Telegram message
    """
    symbol = result.get('symbol', 'UNKNOWN')
    fusion_score = result.get('fusion_score', 0.0)
    confidence = result.get('confidence', 'UNKNOWN')
    details = result.get('details', {})
    
    # Confidence emoji
    conf_emoji = {
        "VERY_HIGH": "🔥",
        "HIGH": "⚡",
        "MEDIUM": "💧", 
        "LOW": "❄️",
        "VERY_LOW": "🌫️"
    }.get(confidence, "❓")
    
    message = f"🚨 *FUSION ALERT* - {symbol}\n"
    message += f"🛰️ *Multi-Detector Signal Detected!*\n\n"
    
    message += f"📊 *Fusion Score:* `{fusion_score:.3f}`\n"
    message += f"{conf_emoji} *Confidence:* {confidence}\n\n"
    
    message += f"🔬 *Detector Breakdown:*\n"
    message += f"• 🧠 CaliforniumWhale AI: `{details.get('californium', 0):.3f}`\n"
    message += f"• 💎 DiamondWhale AI: `{details.get('diamond', 0):.3f}`\n"
    message += f"• 🐋 WhaleCLIP: `{details.get('whaleclip', 0):.3f}`\n\n"
    
    message += f"⚡ *Unified Intelligence*\n"
    message += f"🎯 *Action:* IMMEDIATE ATTENTION\n"
    message += f"🚀 *Signal Quality:* INSTITUTIONAL GRADE\n\n"
    
    # Timestamp
    utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    message += f"🕒 *UTC:* {utc_time}"
    
    return message

def test_fusion_engine():
    """Test Fusion Engine functionality"""
    print("🧪 Testing Multi-Detector Fusion Engine...")
    
    engine = get_fusion_engine()
    
    # Test fusion decision
    test_symbol = "ETHUSDT"
    result = fusion_decision(test_symbol)
    
    print(f"📊 Fusion Decision for {test_symbol}:")
    print(f"   Fusion Score: {result.get('fusion_score', 0):.3f}")
    print(f"   Should Alert: {result.get('should_alert', False)}")
    print(f"   Confidence: {result.get('confidence', 'UNKNOWN')}")
    print(f"   Details: {result.get('details', {})}")
    
    # Test alert message formatting
    if result.get('should_alert', False):
        message = format_fusion_alert_message(result)
        print("\n📱 Sample Fusion Alert Message:")
        print(message)
    
    # Test statistics
    stats = engine.get_fusion_statistics()
    print(f"\n📈 Fusion Statistics: {stats}")
    
    print("✅ Fusion Engine test completed")

if __name__ == "__main__":
    test_fusion_engine()