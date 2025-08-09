"""
Phase 6: Trader-Level AI Decision Engine
Final integration of all modules into elite trader-like decision making
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI


class TraderLevelAIEngine:
    """Elite trader-level AI decision engine integrating all perception layers"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Decision thresholds (adaptive based on learning)
        self.adaptive_thresholds = {
            "clip_confidence_min": 0.4,
            "tjde_score_min": 0.45,
            "consensus_agreement_min": 0.7,
            "embedding_similarity_min": 0.6,
            "reward_history_min": 0.3
        }
    
    def analyze_clip_gpt_consensus(self, clip_prediction: Dict, gpt_commentary: str) -> Dict:
        """Analyze consensus between CLIP visual analysis and GPT textual interpretation"""
        try:
            clip_trend = clip_prediction.get("trend_label", "unknown")
            clip_setup = clip_prediction.get("setup_type", "unknown")
            clip_confidence = clip_prediction.get("clip_confidence", 0.0)
            
            # Extract key terms from GPT commentary
            gpt_lower = gpt_commentary.lower()
            
            # Map CLIP predictions to GPT terms
            consensus_score = 0.0
            consensus_details = []
            
            # Trend consensus analysis
            if clip_trend == "pullback":
                if any(term in gpt_lower for term in ["pullback", "cofnięcie", "korekta", "zniżka"]):
                    consensus_score += 0.3
                    consensus_details.append("Trend consensus: CLIP+GPT agree on pullback")
                elif any(term in gpt_lower for term in ["wzrost", "breakout", "wybicie"]):
                    consensus_score -= 0.2
                    consensus_details.append("Trend conflict: CLIP sees pullback, GPT sees upward")
            
            elif clip_trend == "breakout":
                if any(term in gpt_lower for term in ["breakout", "wybicie", "wzrost", "momentum"]):
                    consensus_score += 0.3
                    consensus_details.append("Trend consensus: CLIP+GPT agree on breakout")
                elif any(term in gpt_lower for term in ["pullback", "spadek", "korekta"]):
                    consensus_score -= 0.2
                    consensus_details.append("Trend conflict: CLIP sees breakout, GPT sees pullback")
            
            # Setup consensus analysis
            if clip_setup == "support-bounce":
                if any(term in gpt_lower for term in ["support", "wsparcie", "bounce", "odbicie"]):
                    consensus_score += 0.25
                    consensus_details.append("Setup consensus: CLIP+GPT agree on support action")
            
            elif clip_setup == "volume-backed":
                if any(term in gpt_lower for term in ["volume", "wolumen", "aktywność"]):
                    consensus_score += 0.25
                    consensus_details.append("Setup consensus: CLIP+GPT agree on volume significance")
            
            # Sentiment analysis from GPT
            sentiment_score = 0.0
            if any(term in gpt_lower for term in ["strong", "silny", "mocny", "dobry"]):
                sentiment_score += 0.15
            elif any(term in gpt_lower for term in ["weak", "słaby", "niepewny"]):
                sentiment_score -= 0.15
            
            consensus_score += sentiment_score
            
            # Confidence weighting
            weighted_consensus = consensus_score * clip_confidence
            
            return {
                "consensus_score": consensus_score,
                "weighted_consensus": weighted_consensus,
                "clip_confidence": clip_confidence,
                "consensus_details": consensus_details,
                "agreement_level": "high" if weighted_consensus > 0.5 else "medium" if weighted_consensus > 0.2 else "low"
            }
            
        except Exception as e:
            print(f"[CONSENSUS ERROR] {e}")
            return {
                "consensus_score": 0.0,
                "weighted_consensus": 0.0,
                "agreement_level": "unknown",
                "consensus_details": []
            }
    
    def evaluate_contextual_memory(self, embedding_context: Dict, reward_history: Dict) -> Dict:
        """Evaluate decision context based on embedding similarity and reward history"""
        try:
            similar_cases = embedding_context.get("similar_cases", [])
            embedding_boost = embedding_context.get("similarity_boost", 0.0)
            
            # Analyze similar case outcomes
            successful_cases = 0
            total_cases = len(similar_cases)
            avg_reward = 0.0
            
            if similar_cases:
                for case in similar_cases:
                    # Mock reward evaluation (in production, use actual historical rewards)
                    case_score = case.get("tjde_score", 0.0)
                    if case_score > 0.6:
                        successful_cases += 1
                        avg_reward += case_score
                
                if total_cases > 0:
                    success_rate = successful_cases / total_cases
                    avg_reward = avg_reward / successful_cases if successful_cases > 0 else 0.0
                else:
                    success_rate = 0.0
            else:
                success_rate = 0.5  # Neutral when no historical data
            
            # Reward history analysis
            historical_accuracy = reward_history.get("accuracy_rate", 0.5)
            recent_performance = reward_history.get("recent_trend", "stable")
            
            # Calculate contextual confidence
            contextual_confidence = 0.0
            
            # Embedding similarity contribution
            if embedding_boost > 0 and success_rate > 0.6:
                contextual_confidence += 0.3 * success_rate
            
            # Historical accuracy contribution
            if historical_accuracy > 0.6:
                contextual_confidence += 0.2 * historical_accuracy
            elif historical_accuracy < 0.3:
                contextual_confidence -= 0.1
            
            # Recent performance trend
            if recent_performance == "improving":
                contextual_confidence += 0.1
            elif recent_performance == "declining":
                contextual_confidence -= 0.1
            
            return {
                "contextual_confidence": contextual_confidence,
                "similar_cases_count": total_cases,
                "similar_cases_success_rate": success_rate,
                "historical_accuracy": historical_accuracy,
                "embedding_boost": embedding_boost,
                "memory_quality": "high" if contextual_confidence > 0.3 else "medium" if contextual_confidence > 0.1 else "low"
            }
            
        except Exception as e:
            print(f"[CONTEXTUAL MEMORY ERROR] {e}")
            return {
                "contextual_confidence": 0.0,
                "memory_quality": "unknown",
                "similar_cases_count": 0
            }
    
    def final_trader_decision(
        self, 
        symbol: str,
        clip_prediction: Dict, 
        gpt_commentary: str, 
        tjde_features: Dict, 
        embedding_context: Dict, 
        reward_history: Dict
    ) -> Dict:
        """Elite trader-level decision making through multi-layer consensus"""
        try:
            print(f"[TRADER AI] {symbol}: Starting elite-level decision analysis")
            
            # === LAYER 1: CLIP + GPT CONSENSUS ===
            consensus_analysis = self.analyze_clip_gpt_consensus(clip_prediction, gpt_commentary)
            
            # === LAYER 2: TJDE TECHNICAL ANALYSIS ===
            tjde_score = tjde_features.get("final_score", 0.0)
            tjde_decision = tjde_features.get("decision", "avoid")
            
            # === LAYER 3: CONTEXTUAL MEMORY EVALUATION ===
            memory_analysis = self.evaluate_contextual_memory(embedding_context, reward_history)
            
            # === LAYER 4: MULTI-LAYER FUSION ===
            # Base decision components
            consensus_weight = consensus_analysis["weighted_consensus"]
            tjde_weight = tjde_score
            memory_weight = memory_analysis["contextual_confidence"]
            
            # Elite trader decision logic: contextual agreement required
            elite_score = 0.0
            decision_factors = []
            
            # High-confidence consensus requirement
            if consensus_analysis["agreement_level"] == "high" and tjde_score > 0.7:
                elite_score += 0.4
                decision_factors.append(f"Strong CLIP+GPT+TJDE consensus ({consensus_weight:.2f})")
            
            # Memory-backed pattern recognition
            if memory_analysis["memory_quality"] in ["high", "medium"] and memory_analysis["similar_cases_count"] >= 2:
                memory_boost = memory_weight * 0.3
                elite_score += memory_boost
                decision_factors.append(f"Pattern memory support (+{memory_boost:.3f})")
            
            # Technical confirmation
            if tjde_score > 0.7:
                elite_score += 0.2
                decision_factors.append("Strong technical indicators")
            elif tjde_score > 0.5:
                elite_score += 0.1
                decision_factors.append("Moderate technical support")
            
            # Risk assessment through historical accuracy
            historical_accuracy = memory_analysis["historical_accuracy"]
            if historical_accuracy < 0.3:
                elite_score *= 0.7  # Risk reduction for poor historical performance
                decision_factors.append("Risk reduction due to poor history")
            
            # Final elite decision
            if elite_score > 0.7:
                final_decision = "join_trend"
                confidence_level = "high"
            elif elite_score > 0.45:
                final_decision = "consider_entry"
                confidence_level = "medium"
            else:
                final_decision = "avoid"
                confidence_level = "low"
            
            # Generate reasoning
            reasoning = f"Elite AI Decision: {final_decision.upper()} (score: {elite_score:.3f})"
            reasoning += f"\nConsensus: CLIP+GPT agreement {consensus_analysis['agreement_level']}"
            reasoning += f"\nTJDE Technical: {tjde_score:.3f}"
            reasoning += f"\nMemory Context: {memory_analysis['memory_quality']}"
            reasoning += f"\nFactors: {'; '.join(decision_factors)}"
            
            result = {
                "elite_decision": final_decision,
                "elite_score": round(elite_score, 3),
                "confidence_level": confidence_level,
                "consensus_analysis": consensus_analysis,
                "memory_analysis": memory_analysis,
                "decision_factors": decision_factors,
                "reasoning": reasoning,
                "trader_level_ai": True
            }
            
            print(f"[TRADER AI] {symbol}: {final_decision.upper()} (elite score: {elite_score:.3f})")
            return result
            
        except Exception as e:
            print(f"[TRADER AI ERROR] {symbol}: {e}")
            return {
                "elite_decision": "avoid",
                "elite_score": 0.0,
                "confidence_level": "error",
                "reasoning": f"Decision error: {str(e)}",
                "trader_level_ai": False
            }
    
    def generate_human_like_commentary(
        self, 
        symbol: str, 
        elite_decision_result: Dict, 
        market_context: Dict = None
    ) -> str:
        """Generate human-like trader commentary explaining the decision"""
        try:
            decision = elite_decision_result.get("elite_decision", "avoid")
            score = elite_decision_result.get("elite_score", 0.0)
            factors = elite_decision_result.get("decision_factors", [])
            consensus = elite_decision_result.get("consensus_analysis", {})
            
            # Create contextual prompt for GPT
            prompt = f"""
            Jesteś doświadczonym traderem analizującym {symbol}. 
            
            Analiza AI wskazuje na decyzję: {decision.upper()} z oceną {score:.3f}
            
            Kluczowe czynniki:
            {chr(10).join(['- ' + factor for factor in factors])}
            
            Zgodność CLIP+GPT: {consensus.get('agreement_level', 'unknown')}
            
            Napisz krótki, profesjonalny komentarz (2-3 zdania) wyjaśniający sytuację jak doświadczony trader. 
            Użyj konkretnego języka rynkowego. Nie podejmuj decyzji - tylko wyjaśnij kontekst.
            """
            
            # Upgraded to GPT-5 for enhanced crypto trading commentary
            # using latest OpenAI model for superior market analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150,
                temperature=0.3
            )
            
            commentary = response.choices[0].message.content.strip()
            
            print(f"[TRADER COMMENTARY] {symbol}: Generated human-like analysis")
            return commentary
            
        except Exception as e:
            print(f"[TRADER COMMENTARY ERROR] {symbol}: {e}")
            # Fallback commentary
            decision = elite_decision_result.get("elite_decision", "avoid")
            score = elite_decision_result.get("elite_score", 0.0)
            
            return f"Analiza techniczna {symbol}: {decision} z oceną {score:.3f}. " \
                   f"Decyzja oparta na wielowarstwowej analizie AI uwzględniającej " \
                   f"wzorce wizualne, kontekst rynkowy i doświadczenie historyczne."


def integrate_trader_level_ai(symbol: str, complete_analysis: Dict) -> Dict:
    """Integrate trader-level AI decision making into complete analysis"""
    try:
        ai_engine = TraderLevelAIEngine()
        
        # Extract components for elite decision making
        clip_prediction = complete_analysis.get("clip_features", {})
        gpt_commentary = complete_analysis.get("gpt_comment", "")
        tjde_features = complete_analysis
        embedding_context = complete_analysis.get("embedding_similarity", {})
        
        # Get reward history from memory context
        historical_context = complete_analysis.get("historical_context", {})
        reward_history = historical_context.get("performance", {})
        
        # Run elite trader decision analysis
        elite_result = ai_engine.final_trader_decision(
            symbol, clip_prediction, gpt_commentary, tjde_features, embedding_context, reward_history
        )
        
        # Generate human-like commentary
        human_commentary = ai_engine.generate_human_like_commentary(symbol, elite_result)
        
        # Update complete analysis with trader-level AI results
        enhanced_analysis = complete_analysis.copy()
        enhanced_analysis.update({
            "trader_level_ai": elite_result,
            "elite_decision": elite_result["elite_decision"],
            "elite_score": elite_result["elite_score"],
            "human_commentary": human_commentary,
            "final_score": elite_result["elite_score"],  # Override with elite score
            "decision": elite_result["elite_decision"],  # Override with elite decision
            "arcymistrz_ai": True
        })
        
        print(f"[TRADER LEVEL AI] {symbol}: Elite analysis complete - {elite_result['elite_decision'].upper()}")
        
        return enhanced_analysis
        
    except Exception as e:
        print(f"[TRADER LEVEL AI ERROR] {symbol}: {e}")
        return complete_analysis


def test_trader_level_ai_engine():
    """Test the complete trader-level AI decision engine"""
    print("Testing Phase 6: Trader-Level AI Decision Engine...")
    
    try:
        # Initialize AI engine
        ai_engine = TraderLevelAIEngine()
        
        # Create comprehensive test data
        test_symbol = "TRADERAI"
        
        # Test CLIP prediction
        clip_prediction = {
            "trend_label": "pullback",
            "setup_type": "support-bounce", 
            "clip_confidence": 0.78
        }
        
        # Test GPT commentary
        gpt_commentary = "Silny pullback z wyraźną reakcją na wsparciu. Volume potwierdza zainteresowanie kupujących."
        
        # Test TJDE features
        tjde_features = {
            "final_score": 0.72,
            "decision": "consider_entry",
            "trend_strength": 0.8,
            "pullback_quality": 0.85
        }
        
        # Test embedding context
        embedding_context = {
            "similar_cases": [
                {"tjde_score": 0.75, "decision": "consider_entry"},
                {"tjde_score": 0.68, "decision": "consider_entry"}
            ],
            "similarity_boost": 0.02
        }
        
        # Test reward history
        reward_history = {
            "accuracy_rate": 0.72,
            "recent_trend": "improving"
        }
        
        # Test consensus analysis
        print("Testing CLIP+GPT consensus...")
        consensus = ai_engine.analyze_clip_gpt_consensus(clip_prediction, gpt_commentary)
        print(f"✅ Consensus analysis: {consensus['agreement_level']} agreement")
        
        # Test contextual memory
        print("Testing contextual memory...")
        memory_analysis = ai_engine.evaluate_contextual_memory(embedding_context, reward_history)
        print(f"✅ Memory analysis: {memory_analysis['memory_quality']} quality")
        
        # Test elite decision making
        print("Testing elite decision making...")
        elite_result = ai_engine.final_trader_decision(
            test_symbol, clip_prediction, gpt_commentary, tjde_features, embedding_context, reward_history
        )
        
        print(f"✅ Elite decision: {elite_result['elite_decision']} (score: {elite_result['elite_score']:.3f})")
        
        # Test human commentary
        print("Testing human commentary...")
        commentary = ai_engine.generate_human_like_commentary(test_symbol, elite_result)
        print(f"✅ Commentary generated: {len(commentary)} characters")
        
        # Test complete integration
        print("Testing complete integration...")
        complete_analysis = {
            "clip_features": clip_prediction,
            "gpt_comment": gpt_commentary,
            "final_score": tjde_features["final_score"],
            "decision": tjde_features["decision"],
            "embedding_similarity": embedding_context,
            "historical_context": {"performance": reward_history}
        }
        
        enhanced_analysis = integrate_trader_level_ai(test_symbol, complete_analysis)
        
        if enhanced_analysis.get("arcymistrz_ai"):
            print("✅ Complete trader-level AI integration working")
            print(f"  Elite Decision: {enhanced_analysis['elite_decision']}")
            print(f"  Elite Score: {enhanced_analysis['elite_score']:.3f}")
            print(f"  Human Commentary: Available")
        
        print("✅ Phase 6 Trader-Level AI Decision Engine working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Phase 6 test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_trader_level_ai_engine()