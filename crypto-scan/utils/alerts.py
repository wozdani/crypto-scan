"""
Enhanced TJDE Telegram Alert System
Sends detailed TOP 5 token analysis with scoring breakdowns and feedback loop information
"""

import os
import logging
from typing import List, Dict, Any
from utils.telegram_bot import send_trend_alert

# Configure logging
logging.basicConfig(
    filename='logs/debug.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def send_tjde_telegram_summary(top5_results: List[Dict[str, Any]], feedback_data: Dict = None):
    """
    Send enhanced TJDE alerts for TOP 5 scoring tokens
    
    Args:
        top5_results: List of top 5 TJDE analysis results
        feedback_data: Optional feedback loop data with weight changes
    """
    try:
        print(f"[ALERT DEBUG] Preparing to send alerts for {len(top5_results)} results")
        logging.debug(f"[ALERT DEBUG] Starting alert preparation for {len(top5_results)} results")
        
        if not top5_results:
            print("[TJDE ALERTS] No results to send")
            logging.warning("[ALERT DEBUG] No results provided for alert sending")
            return False
        
        for i, result in enumerate(top5_results, 1):
            symbol = result.get('symbol', 'UNKNOWN')
            score = result.get('final_score', 0.0)
            decision = result.get('decision', 'avoid')
            confidence = result.get('confidence', 0.0)
            grade = result.get('grade', 'unknown')
            
            print(f"[ALERT DEBUG] Sending alert for {symbol}: score={score:.3f}, decision={decision}")
            logging.debug(f"[ALERT DEBUG] Processing alert {i}/{len(top5_results)} for {symbol}: score={score:.3f}, decision={decision}")
            
            # Get market context
            market_context = result.get('market_context', {})
            market_phase = market_context.get('phase', 'unknown') if isinstance(market_context, dict) else str(market_context)
            
            # Get CLIP analysis if available
            clip_label = result.get('clip_label', 'unknown')
            clip_confidence = result.get('clip_confidence', 0.0)
            
            # Get vision analysis if available (legacy)
            vision_analysis = result.get('vision_analysis', {})
            vision_confidence = vision_analysis.get('confidence', 0) if vision_analysis else 0
            
            # Format decision reasons
            reasons = result.get("decision_reasons", [])
            reasons_str = "→ " + "\n→ ".join(reasons[:5]) if reasons else "→ Standard analysis applied"
            
            # Format score breakdown
            breakdown = result.get("score_breakdown", {})
            breakdown_lines = []
            for k, v in breakdown.items():
                if v > 0:  # Only show non-zero components
                    display_name = k.replace('_', ' ').title()
                    breakdown_lines.append(f"• {display_name}: {round(v, 3)}")
            breakdown_str = "\n".join(breakdown_lines) if breakdown_lines else "• Standard scoring applied"
            
            # Format weights used
            weights = result.get("weights_used", {})
            weights_lines = []
            for k, v in weights.items():
                display_name = k.replace('_', ' ').title()
                weights_lines.append(f"{display_name}: {round(v, 4)}")
            weights_str = "\n".join(weights_lines) if weights_lines else "Standard weights"
            
            # Generate setup explanation
            setup_explanation = generate_setup_explanation(result)
            
            # Decision emoji mapping
            decision_emoji = {
                'join_trend': '🟢 LONG',
                'consider_entry': '🟡 WATCH', 
                'wait': '⏳ WAIT',
                'avoid': '🔴 AVOID'
            }
            
            decision_display = decision_emoji.get(decision, f"❓ {decision.upper()}")
            
            # Add vision analysis if available
            vision_section = ""
            if vision_analysis and vision_confidence > 0.3:
                vision_phase = vision_analysis.get('phase', 'unknown')
                vision_setup = vision_analysis.get('setup', 'unknown')
                vision_section = f"""

🎯 Computer Vision Analysis:
• Pattern: {vision_phase.replace('-', ' ').title()}
• Setup: {vision_setup.replace('-', ' ').title()}  
• AI Confidence: {round(vision_confidence*100, 1)}%
• Description: {vision_analysis.get('phase_description', 'Visual pattern detected')}"""

            # Compose main alert message
            message = f"""#{i} ⚡️ TJDE Alert: {symbol}

📊 Score: {round(score, 3)} | Confidence: {round(confidence*100, 1)}%
{decision_display} | Grade: {grade.upper()}
📈 Phase: {market_phase}{vision_section}

🧠 Score Breakdown:
{breakdown_str}

⚖️ Adaptive Weights:
{weights_str}

💡 Setup Analysis:
{setup_explanation}

🔍 Decision Logic:
{reasons_str}

📸 CLIP Label: {clip_label} ({clip_confidence:.3f})"""

            # Send individual token alert
            success = send_trend_alert(message)
            if success:
                print(f"📤 [TJDE ALERT] Sent #{i}: {symbol} ({decision}, {score:.3f})")
                
                # Log alert to history for feedback loop analysis
                from utils.alert_utils import log_alert_history
                
                log_alert_history(
                    symbol=symbol,
                    score=score,
                    decision=decision,
                    breakdown=result.get("score_breakdown", {})
                )
            else:
                print(f"❌ [TJDE ALERT] Failed to send #{i}: {symbol}")
        
        # Send feedback loop summary if weight changes occurred
        if feedback_data and feedback_data.get("success"):
            send_feedback_summary(feedback_data)
        
        return True
        
    except Exception as e:
        print(f"❌ [TJDE ALERTS] Error sending summary: {e}")
        return False


def generate_setup_explanation(result: Dict[str, Any]) -> str:
    """
    Generate intelligent setup explanation based on scoring components
    
    Args:
        result: TJDE analysis result
        
    Returns:
        Human-readable setup explanation
    """
    try:
        breakdown = result.get("score_breakdown", {})
        decision = result.get('decision', 'avoid')
        
        explanations = []
        
        # Analyze dominant scoring factors
        if breakdown.get('trend_strength', 0) > 0.15:
            explanations.append("Strong trending momentum detected")
        
        if breakdown.get('pullback_quality', 0) > 0.12:
            explanations.append("Quality pullback to support levels")
        
        if breakdown.get('support_reaction', 0) > 0.10:
            explanations.append("Positive reaction at key support")
        
        if breakdown.get('liquidity_pattern_score', 0) > 0.08:
            explanations.append("Favorable liquidity patterns observed")
        
        if breakdown.get('htf_supportive_score', 0) > 0.05:
            explanations.append("Higher timeframe alignment")
        
        # Market context factors
        market_context = result.get('market_context', {})
        if isinstance(market_context, dict):
            if market_context.get('volatility') == 'high':
                explanations.append("High volatility environment")
            if market_context.get('volume_profile') == 'above_average':
                explanations.append("Above-average volume participation")
        
        # Decision-specific explanations
        if decision == 'join_trend':
            explanations.append("All conditions align for trend continuation")
        elif decision == 'consider_entry':
            explanations.append("Setup developing but requires confirmation")
        elif decision == 'wait':
            explanations.append("Market conditions uncertain, patience advised")
        else:
            explanations.append("Risk factors outweigh potential rewards")
        
        return " | ".join(explanations[:3]) if explanations else "Standard technical analysis applied"
        
    except Exception as e:
        return "Analysis framework applied"


def send_feedback_summary(feedback_data: Dict[str, Any]):
    """
    Send feedback loop weight changes summary
    
    Args:
        feedback_data: Feedback analysis results with weight changes
    """
    try:
        weights_before = feedback_data.get("weights_before", {})
        weights_after = feedback_data.get("weights_after", {})
        explanations = feedback_data.get("explanations", {})
        performance = feedback_data.get("performance", {})
        
        # Build weight changes summary
        changes = []
        for key in weights_after:
            before = weights_before.get(key, 0)
            after = weights_after[key]
            delta = after - before
            
            if abs(delta) >= 0.01:  # Significant changes only
                arrow = "🔼" if delta > 0 else "🔽"
                explanation = explanations.get(key, "Automatic adjustment based on performance")
                display_name = key.replace('_', ' ').title()
                
                changes.append(f"{arrow} {display_name}: {before:.3f} → {after:.3f}")
                changes.append(f"   Reason: {explanation}")
        
        if changes:
            success_rate = performance.get("success_rate", 0) * 100
            total_analyzed = performance.get("total_analyzed", 0)
            
            feedback_message = f"""🔧 Feedback Loop Update

📈 Performance: {success_rate:.1f}% success rate
📊 Analyzed: {total_analyzed} recent alerts

⚖️ Weight Adjustments:
{chr(10).join(changes)}

🧠 The system is continuously learning from market outcomes to improve prediction accuracy."""
            
            send_trend_alert(feedback_message)
            print("📤 [FEEDBACK] Weight changes summary sent")
        
    except Exception as e:
        print(f"❌ [FEEDBACK] Error sending summary: {e}")


def format_decision_reasons(reasons: List[str], max_reasons: int = 5) -> str:
    """
    Format decision reasons for display
    
    Args:
        reasons: List of decision reasons
        max_reasons: Maximum number of reasons to display
        
    Returns:
        Formatted reasons string
    """
    if not reasons:
        return "→ Standard analysis framework applied"
    
    formatted_reasons = []
    for reason in reasons[:max_reasons]:
        # Clean up reason text
        clean_reason = reason.strip()
        if not clean_reason.startswith('→'):
            clean_reason = f"→ {clean_reason}"
        formatted_reasons.append(clean_reason)
    
    return "\n".join(formatted_reasons)