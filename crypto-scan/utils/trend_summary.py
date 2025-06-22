#!/usr/bin/env python3
"""
Trend Summary - Ciągły system alertów na Telegramie

Pobiera wyniki TJDE, sortuje według final_score i wysyła Top 5 tokenów 
z kompletnym breakdown i uzasadnieniem decyzji na Telegram.
"""

import json
import os
from typing import List, Dict, Any
from operator import itemgetter
from datetime import datetime


def send_top_trendmode_alerts_to_telegram(results: List[Dict], top_n: int = 5) -> bool:
    """
    Wysyła Top N tokenów z TJDE na Telegram z kompletną analizą
    
    Args:
        results: Lista wyników z simulate_trader_decision_advanced()
        top_n: Ilość top tokenów do wysłania
        
    Returns:
        bool: True jeśli wysłanie się powiodło
    """
    try:
        if not results:
            print("[TREND SUMMARY] No TJDE results to send")
            return False
        
        # Filtruj tylko tokeny z decyzją join_trend lub consider_entry
        valid_results = [
            r for r in results 
            if r.get("decision") in ["join_trend", "consider_entry"] 
            and r.get("final_score", 0) > 0.4
        ]
        
        if not valid_results:
            print("[TREND SUMMARY] No valid TJDE decisions found for alerts")
            return False
        
        # Sortuj według final_score
        sorted_results = sorted(valid_results, key=itemgetter("final_score"), reverse=True)
        top_results = sorted_results[:top_n]
        
        print(f"[TREND SUMMARY] Sending Top {len(top_results)} TJDE alerts to Telegram")
        
        # Wyślij każdy token osobno
        success_count = 0
        for i, result in enumerate(top_results, 1):
            if send_single_trend_alert(result, rank=i):
                success_count += 1
        
        # Wyślij podsumowanie skanu
        if success_count > 0:
            send_scan_summary(len(results), len(valid_results), success_count)
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ [TREND SUMMARY] Error sending alerts: {e}")
        return False


def send_single_trend_alert(result: Dict, rank: int = 1) -> bool:
    """
    Wysyła pojedynczy alert TJDE na Telegram
    
    Args:
        result: Wynik z simulate_trader_decision_advanced()
        rank: Pozycja w rankingu
        
    Returns:
        bool: True jeśli wysłanie się powiodło
    """
    try:
        symbol = result.get("symbol", "UNKNOWN")
        score = result.get("final_score", 0.0)
        confidence = result.get("confidence", 0.0)
        grade = result.get("quality_grade", "N/A")
        decision = result.get("decision", "N/A")
        market_phase = result.get("market_phase", "unknown")
        context_modifiers = result.get("context_modifiers", [])
        
        # Score breakdown z used_features lub score_breakdown
        breakdown = result.get("used_features", result.get("score_breakdown", {}))
        
        # Przygotuj formatowane reasons
        reasons = _generate_decision_reasons(result, breakdown)
        
        # Emoji dla decision
        decision_emoji = {
            "join_trend": "🚀",
            "consider_entry": "⚡",
            "avoid": "❌"
        }.get(decision, "❓")
        
        # Emoji dla grade
        grade_emoji = {
            "strong": "💪",
            "moderate": "👌", 
            "weak": "⚠️"
        }.get(grade, "❓")
        
        # Formatuj wiadomość
        message = f"""#{rank} {decision_emoji} *TJDE Alert: {symbol}*
        
📊 *Score:* `{score:.3f}` | *Confidence:* `{confidence:.3f}`
{grade_emoji} *Grade:* `{grade}` | *Decision:* `{decision.upper()}`
📈 *Phase:* `{market_phase}`

🧠 *Score Breakdown:*
{_format_score_breakdown(breakdown)}

💡 *Analysis:*
{chr(10).join([f"→ {reason}" for reason in reasons[:3]])}"""

        # Dodaj context modifiers jeśli istnieją
        if context_modifiers:
            modifiers_text = ", ".join(context_modifiers)
            message += f"\n\n🔧 *Context:* `{modifiers_text}`"
        
        message += f"\n\n⏰ `{datetime.now().strftime('%H:%M:%S')}`"
        
        # Wyślij na Telegram
        from utils.telegram_bot import send_trend_alert
        return send_trend_alert(message.strip())
        
    except Exception as e:
        print(f"❌ [TREND SUMMARY] Error sending single alert: {e}")
        return False


def send_scan_summary(total_analyzed: int, valid_decisions: int, alerts_sent: int) -> bool:
    """
    Wysyła podsumowanie skanu na Telegram
    
    Args:
        total_analyzed: Całkowita liczba analizowanych tokenów
        valid_decisions: Liczba tokenów z pozytywnymi decyzjami
        alerts_sent: Liczba wysłanych alertów
        
    Returns:
        bool: True jeśli wysłanie się powiodło
    """
    try:
        summary_message = f"""📊 *Scan Summary* - {datetime.now().strftime('%H:%M')}

🔍 Analyzed: `{total_analyzed}` tokens
✅ Valid decisions: `{valid_decisions}` tokens  
📤 Alerts sent: `{alerts_sent}` tokens

Next scan: `15 minutes`"""
        
        from utils.telegram_bot import send_trend_alert
        return send_trend_alert(summary_message)
        
    except Exception as e:
        print(f"❌ [TREND SUMMARY] Error sending summary: {e}")
        return False


def _format_score_breakdown(breakdown: Dict) -> str:
    """Formatuje breakdown scoringu dla Telegram"""
    try:
        if not breakdown:
            return "• No breakdown available"
        
        formatted_lines = []
        for key, value in breakdown.items():
            if isinstance(value, (int, float)):
                # Formatuj nazwy cech
                display_name = key.replace("_", " ").title()
                formatted_lines.append(f"• {display_name}: `{value:.3f}`")
        
        return "\n".join(formatted_lines[:6])  # Limit to 6 lines
        
    except Exception:
        return "• Error formatting breakdown"


def _generate_decision_reasons(result: Dict, breakdown: Dict) -> List[str]:
    """
    Generuje uzasadnienia decyzji na podstawie wyniku TJDE
    
    Args:
        result: Wynik TJDE
        breakdown: Score breakdown
        
    Returns:
        Lista z uzasadnieniami
    """
    try:
        reasons = []
        decision = result.get("decision", "")
        score = result.get("final_score", 0.0)
        grade = result.get("quality_grade", "")
        
        # Główne uzasadnienie na podstawie decision i score
        if decision == "join_trend":
            if score >= 0.8:
                reasons.append("High confidence trend breakout detected")
            elif score >= 0.7:
                reasons.append("Strong trend continuation opportunity")
            else:
                reasons.append("Moderate trend join signal")
        elif decision == "consider_entry":
            reasons.append("Potential entry point with moderate confidence")
        
        # Analizuj najsilniejsze komponenty
        if breakdown:
            sorted_components = sorted(breakdown.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
            
            for component, value in sorted_components[:2]:
                if isinstance(value, (int, float)) and value > 0.6:
                    if "trend" in component.lower():
                        reasons.append(f"Strong trend momentum ({value:.2f})")
                    elif "support" in component.lower():
                        reasons.append(f"Solid support level reaction ({value:.2f})")
                    elif "liquidity" in component.lower():
                        reasons.append(f"Positive liquidity patterns ({value:.2f})")
                    elif "pullback" in component.lower():
                        reasons.append(f"Clean pullback structure ({value:.2f})")
        
        # Context modifiers
        context_modifiers = result.get("context_modifiers", [])
        if "volume_backed_breakout" in context_modifiers:
            reasons.append("Volume confirms breakout")
        if "htf_alignment_boost" in context_modifiers:
            reasons.append("Higher timeframe alignment")
        
        return reasons[:4]  # Limit to 4 reasons
        
    except Exception as e:
        print(f"❌ Error generating reasons: {e}")
        return ["Analysis completed"]


def log_trend_summary(results: List[Dict]) -> bool:
    """
    Loguje podsumowanie trend analysis do pliku
    
    Args:
        results: Lista wyników TJDE
        
    Returns:
        bool: True jeśli zapis się powiódł
    """
    try:
        os.makedirs("logs", exist_ok=True)
        
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_analyzed": len(results),
            "valid_decisions": len([r for r in results if r.get("decision") in ["join_trend", "consider_entry"]]),
            "top_scores": [
                {
                    "symbol": r.get("symbol"),
                    "score": r.get("final_score"),
                    "decision": r.get("decision")
                }
                for r in sorted(results, key=itemgetter("final_score"), reverse=True)[:10]
            ]
        }
        
        with open("logs/trend_summary.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(summary_data, ensure_ascii=False)}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error logging trend summary: {e}")
        return False


if __name__ == "__main__":
    # Test trend summary system
    print("🧪 Testing Trend Summary System...")
    
    # Create test TJDE results
    test_results = [
        {
            "symbol": "BTCUSDT",
            "final_score": 0.78,
            "confidence": 0.85,
            "quality_grade": "strong",
            "decision": "join_trend",
            "market_phase": "breakout-continuation",
            "used_features": {
                "trend_strength": 0.9,
                "pullback_quality": 0.7,
                "support_reaction": 0.8
            },
            "context_modifiers": ["volume_backed_breakout"]
        },
        {
            "symbol": "ETHUSDT", 
            "final_score": 0.65,
            "confidence": 0.72,
            "quality_grade": "moderate",
            "decision": "consider_entry",
            "market_phase": "range-accumulation",
            "used_features": {
                "trend_strength": 0.6,
                "liquidity_pattern_score": 0.7,
                "psych_score": 0.5
            },
            "context_modifiers": []
        }
    ]
    
    # Test formatting
    print("Testing single alert formatting...")
    success = send_single_trend_alert(test_results[0], rank=1)
    print(f"Single alert test: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Test summary logging
    log_success = log_trend_summary(test_results)
    print(f"Summary logging: {'✅ SUCCESS' if log_success else '❌ FAILED'}")
    
    print("✅ Trend Summary System test complete")