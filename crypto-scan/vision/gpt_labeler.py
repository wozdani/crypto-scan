"""
GPT Labeler - Context-Aware Pattern Recognition
Analyzes trading charts using GPT-4o with market context integration
"""

import json
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

def build_vision_prompt(context: Dict) -> str:
    """
    Build comprehensive GPT prompt for visual pattern analysis
    
    Args:
        context: Market context dictionary containing volume, price, HTF data
        
    Returns:
        Formatted prompt string for GPT analysis
    """
    
    prompt = f"""
Jesteś profesjonalnym traderem analizującym setup rynkowy na podstawie kontekstu.

📊 KONTEKST RYNKOWY:
- Wolumen: {context.get('volume_info', 'nieznany')}
- Pozycja ceny: {context.get('price_position', 'nieznana')}
- HTF kontekst: {context.get('htf_context', 'brak danych')}
- CLIP widzi: {context.get('clip_label', 'nieznany')} (pewność: {context.get('clip_confidence', 0.0)})

🎯 ZADANIE:
Na podstawie powyższego kontekstu określ:

1. **LABEL** - główny pattern (wybierz jeden):
   - pullback (korekta w trendzie)
   - breakout (przebicie)
   - range (konsolidacja)
   - accumulation (akumulacja)
   - exhaustion (wyczerpanie)
   - retest (test wsparcia/oporu)
   - early_trend (początek trendu)
   - chaos (brak jasnego patternu)

2. **PHASE** - faza rynku (wybierz jedną):
   - trend (silny kierunkowy ruch)
   - consolidation (konsolidacja)
   - reversal (odwrócenie)
   - accumulation (akumulacja przed ruchem)
   - distribution (dystrybucja)

3. **CONFIDENCE** - pewność analizy (0.00-1.00):
   - 0.90+ gdy wszystkie sygnały się zgadzają
   - 0.70-0.89 gdy większość sygnałów potwierdza
   - 0.50-0.69 gdy sygnały mieszane
   - 0.30-0.49 gdy niepewność dominuje
   - <0.30 gdy brak jasnych sygnałów

📋 ODPOWIEDŹ (tylko JSON):
{{
    "label": "nazwa_patternu",
    "phase": "nazwa_fazy", 
    "confidence": 0.XX,
    "reasoning": "krótkie uzasadnienie 1-2 zdania"
}}
"""
    
    return prompt

def gpt_label_with_context(context: Dict) -> Dict:
    """
    Generate AI label using GPT-4o with market context
    
    Args:
        context: Market context dictionary
        
    Returns:
        Dictionary with label, phase, confidence, and reasoning
    """
    try:
        # Import OpenAI if available
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            logger.warning("[GPT VISION] ❌ OpenAI library not available")
            return _fallback_analysis(context)
        except Exception as e:
            logger.error(f"[GPT VISION] ❌ OpenAI initialization failed: {e}")
            return _fallback_analysis(context)
        
        # Build prompt
        prompt = build_vision_prompt(context)
        
        logger.info("[GPT VISION] 🧠 Analyzing market context with GPT-4o...")
        
        # Call GPT-4o for reliable technical analysis
        response = client.chat.completions.create(
            model="gpt-4o",  # Reliable model for consistent visual understanding
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś ekspertem od analizy technicznej. Zawsze odpowiadaj w formacie JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
            max_completion_tokens=300
        )
        
        # Parse response
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Validate and clean result
        cleaned_result = _validate_gpt_result(result)
        
        logger.info(f"[GPT VISION] ✅ Analysis complete: {cleaned_result.get('label')} "
                   f"(phase: {cleaned_result.get('phase')}, confidence: {cleaned_result.get('confidence')})")
        
        return cleaned_result
        
    except json.JSONDecodeError as e:
        logger.error(f"[GPT VISION] ❌ JSON parsing failed: {e}")
        return _fallback_analysis(context)
    except Exception as e:
        logger.error(f"[GPT VISION] ❌ GPT analysis failed: {e}")
        return _fallback_analysis(context)

def _validate_gpt_result(result: Dict) -> Dict:
    """Validate and clean GPT result"""
    
    # Valid labels
    valid_labels = [
        "pullback", "breakout", "range", "accumulation", 
        "exhaustion", "retest", "early_trend", "chaos"
    ]
    
    # Valid phases
    valid_phases = [
        "trend", "consolidation", "reversal", 
        "accumulation", "distribution"
    ]
    
    # Clean and validate
    cleaned = {
        "label": result.get("label", "unknown"),
        "phase": result.get("phase", "unknown"),
        "confidence": float(result.get("confidence", 0.5)),
        "reasoning": result.get("reasoning", "No reasoning provided")
    }
    
    # Validate label
    if cleaned["label"] not in valid_labels:
        cleaned["label"] = "chaos"
        
    # Validate phase
    if cleaned["phase"] not in valid_phases:
        cleaned["phase"] = "consolidation"
        
    # Validate confidence
    cleaned["confidence"] = max(0.0, min(1.0, cleaned["confidence"]))
    
    return cleaned

def _fallback_analysis(context: Dict) -> Dict:
    """Fallback analysis when GPT is not available"""
    
    # Simple heuristic analysis based on CLIP
    clip_label = context.get("clip_label", "unknown")
    clip_confidence = context.get("clip_confidence", 0.0)
    
    # Map CLIP labels to simplified patterns
    label_mapping = {
        "pullback": "pullback",
        "breakout": "breakout", 
        "range": "range",
        "accumulation": "accumulation",
        "exhaustion": "exhaustion",
        "retest": "retest",
        "early_trend": "early_trend"
    }
    
    fallback_label = label_mapping.get(clip_label, "chaos")
    fallback_confidence = min(0.6, clip_confidence)  # Cap fallback confidence
    
    logger.info(f"[GPT VISION] 🔄 Using fallback analysis: {fallback_label} (confidence: {fallback_confidence})")
    
    return {
        "label": fallback_label,
        "phase": "consolidation",  # Safe default
        "confidence": fallback_confidence,
        "reasoning": f"Fallback analysis based on CLIP prediction: {clip_label}"
    }

def test_gpt_labeler():
    """Test GPT labeler functionality"""
    
    # Test context
    test_context = {
        "volume_info": "volume increasing",
        "price_position": "recent breakout",
        "htf_context": "bullish trend on 4H",
        "clip_label": "breakout",
        "clip_confidence": 0.85
    }
    
    result = gpt_label_with_context(test_context)
    print(f"Test result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_gpt_labeler()