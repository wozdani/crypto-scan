"""
Cluster Integration with TJDE System
Integruje predykcje klastrów z systemem TJDE dla enhanced decision making
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from predict_cluster import get_cluster_predictor
from generate_embeddings import get_embedding_generator

logger = logging.getLogger(__name__)

class TJDEClusterIntegration:
    """Integration klastrów z systemem TJDE"""
    
    def __init__(self):
        """Initialize cluster integration"""
        self.cluster_predictor = get_cluster_predictor()
        self.embedding_generator = get_embedding_generator()
        
        # Cluster-based score modifiers
        self.cluster_modifiers = {
            "high_performance": 0.05,    # Boost for high-performing clusters
            "medium_performance": 0.02,  # Small boost for medium clusters
            "low_performance": -0.03,    # Penalty for low-performing clusters
            "unknown": 0.0               # Neutral for unknown clusters
        }
        
        logger.info("TJDE cluster integration initialized")
    
    def enhance_tjde_with_cluster(
        self, 
        symbol: str, 
        tjde_result: Dict[str, Any], 
        image_path: Optional[str] = None,
        gpt_comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance TJDE decision with cluster analysis
        
        Args:
            symbol: Trading symbol
            tjde_result: Original TJDE result
            image_path: Optional chart image path
            gpt_comment: Optional GPT commentary
            
        Returns:
            Enhanced TJDE result with cluster information
        """
        try:
            # Generate embedding for current setup
            embedding = self.embedding_generator.generate_combined_embedding(
                symbol, image_path, tjde_result, gpt_comment
            )
            
            if embedding is None:
                logger.warning(f"Could not generate embedding for {symbol}")
                return self._add_cluster_info(tjde_result, None)
            
            # Predict cluster and setup quality
            quality_prediction = self.cluster_predictor.predict_setup_quality(embedding, symbol)
            
            # Calculate cluster-based score modifier
            cluster_modifier = self._calculate_cluster_modifier(quality_prediction)
            
            # Apply modifier to TJDE score
            original_score = tjde_result.get("final_score", 0)
            enhanced_score = original_score + cluster_modifier
            enhanced_score = max(0.0, min(1.0, enhanced_score))  # Clamp to [0, 1]
            
            # Update decision if score change is significant
            original_decision = tjde_result.get("decision", "unknown")
            new_decision = self._update_decision_based_on_score(enhanced_score, original_decision)
            
            # Create enhanced result
            enhanced_result = tjde_result.copy()
            enhanced_result.update({
                "final_score": enhanced_score,
                "decision": new_decision,
                "cluster_enhanced": True,
                "cluster_info": {
                    "cluster": quality_prediction.get("cluster", -1),
                    "quality_score": quality_prediction.get("quality_score", 0.5),
                    "similar_symbols": quality_prediction.get("similar_symbols", []),
                    "recommendation": quality_prediction.get("recommendation", "neutral"),
                    "confidence": quality_prediction.get("confidence", 0.0),
                    "score_modifier": cluster_modifier,
                    "original_score": original_score
                }
            })
            
            # Add cluster reasoning to decision reasons
            if "decision_reasons" in enhanced_result:
                cluster_reason = f"Cluster analysis: {quality_prediction.get('recommendation', 'neutral')} (modifier: {cluster_modifier:+.3f})"
                enhanced_result["decision_reasons"].append(cluster_reason)
            
            logger.info(f"Enhanced {symbol} TJDE with cluster: {original_score:.3f} → {enhanced_score:.3f}")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error enhancing TJDE with cluster for {symbol}: {e}")
            return self._add_cluster_info(tjde_result, {"error": str(e)})
    
    def _calculate_cluster_modifier(self, quality_prediction: Dict[str, Any]) -> float:
        """Calculate score modifier based on cluster prediction"""
        try:
            quality_score = quality_prediction.get("quality_score", 0.5)
            confidence = quality_prediction.get("confidence", 0.0)
            recommendation = quality_prediction.get("recommendation", "neutral")
            
            # Base modifier from quality score
            base_modifier = (quality_score - 0.5) * 0.1  # Scale to ±0.05 range
            
            # Apply confidence weighting
            confidence_weighted_modifier = base_modifier * confidence
            
            # Additional boost/penalty based on recommendation
            recommendation_modifiers = {
                "consider": 0.02,
                "neutral": 0.0,
                "avoid": -0.02
            }
            
            recommendation_modifier = recommendation_modifiers.get(recommendation, 0.0)
            
            # Combine modifiers
            total_modifier = confidence_weighted_modifier + recommendation_modifier
            
            # Clamp to reasonable range
            total_modifier = max(-0.1, min(0.1, total_modifier))
            
            return total_modifier
            
        except Exception as e:
            logger.error(f"Error calculating cluster modifier: {e}")
            return 0.0
    
    def _update_decision_based_on_score(self, enhanced_score: float, original_decision: str) -> str:
        """Update decision based on enhanced score"""
        try:
            # Decision thresholds
            if enhanced_score >= 0.75:
                return "join_trend"
            elif enhanced_score >= 0.6:
                return "consider_entry"
            else:
                return "avoid"
                
        except Exception:
            return original_decision
    
    def _add_cluster_info(self, tjde_result: Dict[str, Any], cluster_info: Optional[Dict]) -> Dict[str, Any]:
        """Add cluster info to TJDE result"""
        result = tjde_result.copy()
        result["cluster_enhanced"] = False
        result["cluster_info"] = cluster_info or {"error": "No cluster analysis available"}
        return result
    
    def get_cluster_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Get cluster-based recommendations for symbol"""
        try:
            # Check if symbol has historical cluster assignment
            if hasattr(self.cluster_predictor, 'symbol_clusters') and self.cluster_predictor.symbol_clusters:
                cluster = self.cluster_predictor.symbol_clusters.get(symbol)
                
                if cluster is not None:
                    similar_symbols = self.cluster_predictor.get_similar_symbols(cluster, limit=10)
                    
                    return {
                        "symbol": symbol,
                        "cluster": cluster,
                        "similar_symbols": similar_symbols,
                        "recommendation": "Historical cluster analysis available"
                    }
            
            return {
                "symbol": symbol,
                "cluster": None,
                "similar_symbols": [],
                "recommendation": "No historical cluster data"
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster recommendations for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e)
            }


# Global instance
_global_cluster_integration = None

def get_cluster_integration() -> TJDEClusterIntegration:
    """Get global cluster integration instance"""
    global _global_cluster_integration
    if _global_cluster_integration is None:
        _global_cluster_integration = TJDEClusterIntegration()
    return _global_cluster_integration

def enhance_tjde_with_cluster(symbol: str, tjde_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to enhance TJDE with cluster analysis
    
    Args:
        symbol: Trading symbol
        tjde_result: Original TJDE result
        
    Returns:
        Enhanced TJDE result
    """
    integration = get_cluster_integration()
    return integration.enhance_tjde_with_cluster(symbol, tjde_result)

def main():
    """Test cluster integration"""
    print("Testing TJDE Cluster Integration")
    print("=" * 40)
    
    integration = TJDEClusterIntegration()
    
    # Test with mock TJDE result
    mock_tjde = {
        "symbol": "BTCUSDT",
        "decision": "consider_entry",
        "final_score": 0.685,
        "quality_grade": "moderate",
        "score_breakdown": {
            "trend_strength": 0.75,
            "pullback_quality": 0.68,
            "support_reaction": 0.72,
            "liquidity_pattern_score": 0.65,
            "psych_score": 0.80,
            "htf_supportive_score": 0.55,
            "market_phase_modifier": 0.05
        },
        "decision_reasons": ["Initial TJDE analysis"]
    }
    
    print(f"Original TJDE Result:")
    print(f"   Symbol: {mock_tjde['symbol']}")
    print(f"   Decision: {mock_tjde['decision']}")
    print(f"   Score: {mock_tjde['final_score']}")
    
    # Enhance with cluster analysis
    enhanced_result = integration.enhance_tjde_with_cluster(
        "BTCUSDT", 
        mock_tjde,
        gpt_comment="pullback-in-trend | trending-up | volume-backed"
    )
    
    print(f"\nEnhanced TJDE Result:")
    print(f"   Decision: {enhanced_result['decision']}")
    print(f"   Score: {enhanced_result['final_score']}")
    print(f"   Cluster Enhanced: {enhanced_result['cluster_enhanced']}")
    
    if enhanced_result.get("cluster_info"):
        cluster_info = enhanced_result["cluster_info"]
        print(f"   Cluster Info:")
        print(f"     Cluster: {cluster_info.get('cluster', 'N/A')}")
        print(f"     Quality Score: {cluster_info.get('quality_score', 0):.3f}")
        print(f"     Recommendation: {cluster_info.get('recommendation', 'N/A')}")
        print(f"     Score Modifier: {cluster_info.get('score_modifier', 0):+.3f}")
    
    # Test cluster recommendations
    recommendations = integration.get_cluster_recommendations("BTCUSDT")
    print(f"\nCluster Recommendations:")
    print(f"   {recommendations}")
    
    print(f"\n✅ Cluster integration test completed")

if __name__ == "__main__":
    main()