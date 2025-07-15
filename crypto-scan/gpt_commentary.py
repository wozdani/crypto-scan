"""
GPT Commentary System for Trend-Mode
Provides intelligent chart analysis, CLIP feedback, and alert commentary
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Optional, Any
from openai import OpenAI

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Convert image to base64 for GPT-4 Vision analysis"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"[GPT COMMENTARY] Failed to encode image {image_path}: {e}")
        return None


def extract_primary_label_from_commentary(commentary: str) -> str:
    """
    Extract primary setup label from GPT commentary
    
    Args:
        commentary: GPT commentary text
        
    Returns:
        Primary setup label based on content analysis
    """
    if not commentary:
        return "unknown"
    
    commentary_lower = commentary.lower()
    
    # Pattern detection with priority order
    if any(keyword in commentary_lower for keyword in ["pullback", "cofnięcie", "korekta", "retest"]):
        if any(trend in commentary_lower for trend in ["trend", "wzrostowy", "býczy"]):
            return "pullback_in_trend"
        else:
            return "pullback_correction"
    
    elif any(keyword in commentary_lower for keyword in ["breakout", "wybicie", "przebicie", "breakthrough"]):
        if any(cont in commentary_lower for cont in ["kontynuacja", "continuation", "follow-through"]):
            return "breakout_continuation"
        else:
            return "breakout_initial"
    
    elif any(keyword in commentary_lower for keyword in ["squeeze", "ściskanie", "compression", "consolidat"]):
        return "squeeze_pattern"
    
    elif any(keyword in commentary_lower for keyword in ["support", "wsparcie", "odbicie", "bounce"]):
        return "support_bounce"
    
    elif any(keyword in commentary_lower for keyword in ["resistance", "opór", "test", "rejection"]):
        return "resistance_test"
    
    elif any(keyword in commentary_lower for keyword in ["trend", "trendu", "momentum", "wzrostowy"]):
        if any(strong in commentary_lower for strong in ["silny", "strong", "mocny"]):
            return "trending_up"
        else:
            return "trend_following"
    
    elif any(keyword in commentary_lower for keyword in ["volume", "wolumen", "aktywność", "activity"]):
        return "volume_backed_move"
    
    elif any(keyword in commentary_lower for keyword in ["accumulation", "akumulacja", "gromadzenie"]):
        return "range_accumulation"
    
    elif any(keyword in commentary_lower for keyword in ["exhaustion", "wyczerpanie", "reversal", "odwrócenie"]):
        return "exhaustion_pattern"
    
    elif any(keyword in commentary_lower for keyword in ["consolidation", "konsolidacja", "sideways"]):
        return "consolidation"
    
    else:
        # Fallback based on general sentiment
        if any(positive in commentary_lower for positive in ["pozytywny", "positive", "bullish", "silny"]):
            return "bullish_setup"
        elif any(negative in commentary_lower for negative in ["negatywny", "negative", "bearish", "słaby"]):
            return "bearish_setup"
        else:
            return "neutral_pattern"


def extract_primary_label(gpt_commentary: str) -> str:
    """
    Ekstrahuje główną etykietę trendu/setupu z komentarza GPT.
    Zwraca nazwę labela w formacie: trend_setup_type
    
    Args:
        gpt_commentary: GPT commentary text
        
    Returns:
        Setup label in format: trend_setup_type
    """
    if not gpt_commentary:
        return "unknown"
    
    gpt_commentary = gpt_commentary.lower()

    # Detailed pattern detection for CLIP training (Polish patterns)
    if "pullback" in gpt_commentary and any(word in gpt_commentary for word in ["reakcja", "reakcją", "odbicie", "odbicia"]):
        return "trend_pullback_reacted"
    elif "pullback" in gpt_commentary and any(word in gpt_commentary for word in ["brak reakcji", "brak odbicia", "bez reakcji"]):
        return "trend_pullback_failed"
    elif any(phrase in gpt_commentary for phrase in ["kontynuacja trendu", "dynamiczna kontynuacja", "kontynuacji trendu"]):
        return "trend_continuation"
    elif any(phrase in gpt_commentary for phrase in ["wyczerpanie trendu", "osłabienie trendu", "wyczerpaniu trendu"]):
        return "trend_exhaustion"
    elif "fakeout" in gpt_commentary and any(word in gpt_commentary for word in ["opór", "oporu", "oporze"]):
        return "fakeout_on_resistance"
    elif "fakeout" in gpt_commentary and any(word in gpt_commentary for word in ["wsparcie", "wsparcia", "wsparciu"]):
        return "fakeout_on_support"
    elif any(word in gpt_commentary for word in ["konsolidacja", "konsolidacji", "konsolidacją"]):
        return "range_consolidation"
    elif any(phrase in gpt_commentary for phrase in ["brak trendu", "bez trendu", "brak wyraźnego trendu"]):
        return "no_trend"
    
    # English patterns for broader compatibility
    elif "pullback" in gpt_commentary and ("reaction" in gpt_commentary or "bounce" in gpt_commentary):
        return "trend_pullback_reacted"
    elif "pullback" in gpt_commentary and ("failed" in gpt_commentary or "no reaction" in gpt_commentary):
        return "trend_pullback_failed"
    elif "trend continuation" in gpt_commentary or "momentum continuation" in gpt_commentary:
        return "trend_continuation"
    elif "trend exhaustion" in gpt_commentary or "weakening trend" in gpt_commentary:
        return "trend_exhaustion"
    elif "fake breakout" in gpt_commentary or "false breakout" in gpt_commentary:
        return "fakeout_breakout"
    elif "consolidation" in gpt_commentary or "ranging" in gpt_commentary:
        return "range_consolidation"
    elif "no clear trend" in gpt_commentary or "sideways" in gpt_commentary:
        return "no_trend"
    else:
        return "unknown"


def rename_chart_files_with_gpt_label(original_png_path: str, gpt_commentary: str) -> tuple[str, str]:
    """
    Rename chart files (.png and .json) with GPT-extracted label
    
    Args:
        original_png_path: Original PNG file path
        gpt_commentary: GPT commentary for label extraction
        
    Returns:
        Tuple of (new_png_path, new_json_path)
    """
    try:
        # Extract label from GPT commentary
        extracted_label = extract_primary_label(gpt_commentary)
        
        # Skip renaming if no meaningful label extracted
        if extracted_label == "unknown":
            return original_png_path, original_png_path.replace('.png', '.json')
        
        # Parse original filename
        import os
        from pathlib import Path
        
        original_path = Path(original_png_path)
        filename_parts = original_path.stem.split('_')
        
        # Expected format: SYMBOL_timestamp_phase_decision_tjde.png
        if len(filename_parts) >= 2:
            symbol = filename_parts[0]
            timestamp = filename_parts[1]
            
            # Create new filename with GPT label
            new_filename = f"{symbol}_{timestamp}_{extracted_label}"
            new_png_path = str(original_path.parent / f"{new_filename}.png")
            new_json_path = str(original_path.parent / f"{new_filename}.json")
            
            # Rename files if they exist
            if os.path.exists(original_png_path):
                os.rename(original_png_path, new_png_path)
                print(f"[GPT RENAME] PNG: {original_path.name} → {Path(new_png_path).name}")
                
            original_json_path = original_png_path.replace('.png', '.json')
            if os.path.exists(original_json_path):
                os.rename(original_json_path, new_json_path)
                print(f"[GPT RENAME] JSON: {Path(original_json_path).name} → {Path(new_json_path).name}")
                
            return new_png_path, new_json_path
        else:
            print(f"[GPT RENAME] Cannot parse filename: {original_path.name}")
            return original_png_path, original_png_path.replace('.png', '.json')
            
    except Exception as e:
        print(f"[GPT RENAME ERROR] Failed to rename files: {e}")
        return original_png_path, original_png_path.replace('.png', '.json')


def generate_chart_commentary(image_path: str, tjde_score: float, decision: str, 
                            clip_prediction: Optional[Dict] = None, symbol: str = "UNKNOWN") -> Optional[str]:
    """
    GPT as chart commentator - analyzes chart image with TJDE and CLIP context
    
    Args:
        image_path: Path to chart image
        tjde_score: TJDE final score
        decision: TJDE decision
        clip_prediction: Optional CLIP prediction data
        symbol: Trading symbol
        
    Returns:
        GPT commentary text or None if failed
    """
    if not openai_client or not os.path.exists(image_path):
        return None
    
    try:
        # Encode chart image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        # Build context-aware prompt
        clip_context = ""
        if clip_prediction:
            clip_context = f"\nCLIP Model sugeruje: {clip_prediction.get('decision', 'unknown')} z pewnością {clip_prediction.get('confidence', 0):.2f}"
        
        prompt = f"""Analizuj wykres {symbol} w kontekście trendów rynkowych jako doświadczony analityk techniczny.

KONTEKST SCORINGU:
- TJDE Score: {tjde_score:.3f}
- Decyzja systemu: {decision}{clip_context}

ZADANIE:
Opisz setup, fazę rynku, zachowanie wolumenu, i potencjalne zagrożenia/możliwości. 
Skoncentruj się na obiektywnej analizie wzorców świecowych, poziomów wsparcia/oporu, i momentum.

WAŻNE: Nie podejmuj decyzji tradingowych - tylko opisuj to co widzisz na wykresie.

Format odpowiedzi:
- Setup: [opis wzorca]
- Faza rynku: [trend/konsolidacja/reversal]
- Wolumen: [analiza aktywności]
- Kluczowe poziomy: [wsparcie/opór]
- Potencjalne zagrożenia: [co może się zmienić]"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Latest model with vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        commentary = response.choices[0].message.content
        
        # Extract primary label from commentary
        extracted_label = extract_primary_label_from_commentary(commentary)
        
        # Save commentary to .gpt.json file
        gpt_json_path = image_path.replace('.png', '.gpt.json')
        gpt_data = {
            "commentary": commentary,
            "extracted_label": extracted_label,
            "tjde_score": tjde_score,
            "decision": decision,
            "clip_prediction": clip_prediction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "chart_commentary"
        }
        
        with open(gpt_json_path, 'w', encoding='utf-8') as f:
            json.dump(gpt_data, f, indent=2, ensure_ascii=False)
        
        print(f"[GPT COMMENTARY] Chart analysis saved: {gpt_json_path}")
        return commentary
        
    except Exception as e:
        print(f"[GPT COMMENTARY ERROR] {e}")
        return None


def explain_clip_misclassification(image_path: str, expected_setup: str, 
                                 actual_prediction: str, symbol: str = "UNKNOWN") -> Optional[str]:
    """
    GPT as CLIP error analyst - explains why CLIP made incorrect prediction
    
    Args:
        image_path: Path to chart image
        expected_setup: Correct setup based on actual outcome
        actual_prediction: CLIP's incorrect prediction
        symbol: Trading symbol
        
    Returns:
        GPT explanation or None if failed
    """
    if not openai_client or not os.path.exists(image_path):
        return None
    
    try:
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        prompt = f"""Analizuj błąd klasyfikacji modelu Vision-AI na wykresie {symbol}.

PROBLEM:
- Model CLIP zaklasyfikował jako: {actual_prediction}
- Rzeczywisty setup to: {expected_setup}

ZADANIE:
Wyjaśnij możliwe przyczyny błędnej klasyfikacji:
1. Jakie wzorce na wykresie mogły zmylić model?
2. Czy brakuje charakterystycznych cech dla prawidłowej klasyfikacji?
3. Jakie elementy wykresu są niejednoznaczne?
4. Rekomendacje dla poprawy modelu

Bądź konkretny i wskaż specific visual patterns."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        
        explanation = response.choices[0].message.content
        
        # Save to .clip_feedback.json
        feedback_path = image_path.replace('.png', '.clip_feedback.json')
        feedback_data = {
            "explanation": explanation,
            "expected_setup": expected_setup,
            "actual_prediction": actual_prediction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "clip_error_analysis"
        }
        
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        print(f"[GPT CLIP ANALYSIS] Error explanation saved: {feedback_path}")
        return explanation
        
    except Exception as e:
        print(f"[GPT CLIP ANALYSIS ERROR] {e}")
        return None


def audit_scoring_chart_consistency(image_path: str, scoring_data: Dict, 
                                   symbol: str = "UNKNOWN") -> Optional[str]:
    """
    GPT as scoring auditor - checks consistency between chart and TJDE scoring
    
    Args:
        image_path: Path to chart image
        scoring_data: TJDE scoring breakdown
        symbol: Trading symbol
        
    Returns:
        GPT audit report or None if failed
    """
    if not openai_client or not os.path.exists(image_path):
        return None
    
    try:
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        # Extract key scoring components
        trend_strength = scoring_data.get('trend_strength', 0.0)
        pullback_quality = scoring_data.get('pullback_quality', 0.0)
        volume_score = scoring_data.get('volume_behavior_score', 0.0)
        final_score = scoring_data.get('final_score', 0.0)
        
        prompt = f"""Audytuj zgodność między scoringiem algorytmicznym a rzeczywistym wykresem {symbol}.

SCORING ALGORYTMICZNY:
- Siła trendu: {trend_strength:.3f}
- Jakość pullbacku: {pullback_quality:.3f}
- Score wolumenu: {volume_score:.3f}
- Wynik końcowy: {final_score:.3f}

ZADANIE AUDYTU:
1. Czy scoring trend_strength odpowiada temu co widzisz na wykresie?
2. Czy pullback_quality jest adekwatny do wzorca świecowego?
3. Czy volume_score zgadza się z aktywnością na wykresie?
4. Zidentyfikuj największe niespójności
5. Sugerowane korekty algorytmu

Bądź krytyczny i wskaż konkretne problemy."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=700
        )
        
        audit_report = response.choices[0].message.content
        
        # Save to .gpt_audit.json
        audit_path = image_path.replace('.png', '.gpt_audit.json')
        audit_data = {
            "audit_report": audit_report,
            "scoring_data": scoring_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "scoring_audit"
        }
        
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)
        
        print(f"[GPT AUDIT] Scoring audit saved: {audit_path}")
        return audit_report
        
    except Exception as e:
        print(f"[GPT AUDIT ERROR] {e}")
        return None


def generate_synthetic_description(image_path: str, symbol: str = "UNKNOWN") -> Optional[str]:
    """
    GPT as synthetic descriptor - creates CLIP training descriptions for charts without classification
    
    Args:
        image_path: Path to chart image
        symbol: Trading symbol
        
    Returns:
        GPT-generated description or None if failed
    """
    if not openai_client or not os.path.exists(image_path):
        return None
    
    try:
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        prompt = f"""Analizuj wykres {symbol} bez dostępnej klasyfikacji CLIP i wygeneruj opis treningowy.

ZADANIE:
Opisz najbardziej prawdopodobny setup, fazę rynku i momentum w formacie przydatnym dla treningu modeli Vision-AI.

Format opisu:
- Setup_Type: [breakout/pullback/reversal/consolidation]
- Market_Phase: [trending/accumulation/distribution/transition]
- Momentum: [bullish/bearish/neutral]
- Volume_Pattern: [increasing/decreasing/stable]
- Key_Features: [lista 3-4 kluczowych wzorców wizualnych]

Bądź precyzyjny i używaj terminologii technicznej."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        description = response.choices[0].message.content
        
        # Save to .gpt_synthetic.json
        synthetic_path = image_path.replace('.png', '.gpt_synthetic.json')
        synthetic_data = {
            "synthetic_description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "synthetic_labeling"
        }
        
        with open(synthetic_path, 'w', encoding='utf-8') as f:
            json.dump(synthetic_data, f, indent=2, ensure_ascii=False)
        
        print(f"[GPT SYNTHETIC] Description saved: {synthetic_path}")
        return description
        
    except Exception as e:
        print(f"[GPT SYNTHETIC ERROR] {e}")
        return None


def generate_telegram_alert_commentary(symbol: str, tjde_data: Dict, 
                                     market_data: Dict = None) -> str:
    """
    GPT as alert commentator - creates human-readable Telegram alert summaries
    
    Args:
        symbol: Trading symbol
        tjde_data: TJDE analysis results
        market_data: Optional market data context
        
    Returns:
        Human-readable alert commentary
    """
    if not openai_client:
        # Fallback to basic template
        return f"Alert {symbol}: Score {tjde_data.get('final_score', 0):.2f} - {tjde_data.get('decision', 'unknown')}"
    
    try:
        # Extract key information
        score = tjde_data.get('final_score', 0.0)
        decision = tjde_data.get('decision', 'unknown')
        phase = tjde_data.get('market_phase', 'unknown')
        clip_info = tjde_data.get('clip_decision')
        
        prompt = f"""Wygeneruj czytelny komentarz dla alertu Telegram bez decyzji tradingowych.

DANE:
- Symbol: {symbol}
- Score TJDE: {score:.3f}
- Decyzja systemu: {decision}
- Faza rynku: {phase}
{f"- CLIP sugeruje: {clip_info}" if clip_info else ""}

WYMAGANIA:
- Maksymalnie 2-3 zdania
- Język zrozumiały dla użytkownika
- Opisz sytuację rynkową bez rekomendacji
- Skoncentruj się na faktach: trend, wsparcie, wolumen
- Dodaj kontekst czy sygnał jest silny/słaby

Przykład: "Obserwujemy cofnięcie do strefy wsparcia w silnym trendzie. Wolumen stabilny. Sygnał pozostaje neutralny – obserwacja." """

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        commentary = response.choices[0].message.content.strip()
        print(f"[GPT TELEGRAM] Generated commentary for {symbol}")
        return commentary
        
    except Exception as e:
        print(f"[GPT TELEGRAM ERROR] {e}")
        # Fallback commentary
        return f"Alert {symbol}: Score {score:.2f} - system suggest {decision}. Market phase: {phase}."


def run_comprehensive_gpt_analysis(image_path: str, symbol: str, tjde_data: Dict, 
                                 clip_prediction: Optional[Dict] = None) -> Dict[str, str]:
    """
    Run comprehensive GPT analysis pipeline for a chart
    
    Args:
        image_path: Path to chart image
        symbol: Trading symbol
        tjde_data: TJDE analysis results
        clip_prediction: Optional CLIP prediction data
        
    Returns:
        Dictionary with all GPT analysis results
    """
    results = {}
    
    print(f"[GPT COMPREHENSIVE] Starting analysis for {symbol}")
    
    # 1. Chart commentary
    commentary = generate_chart_commentary(
        image_path, 
        tjde_data.get('final_score', 0), 
        tjde_data.get('decision', 'unknown'),
        clip_prediction,
        symbol
    )
    if commentary:
        results['chart_commentary'] = commentary
    
    # 2. Scoring audit (if significant inconsistencies detected)
    if should_audit_scoring(tjde_data):
        audit = audit_scoring_chart_consistency(image_path, tjde_data, symbol)
        if audit:
            results['scoring_audit'] = audit
    
    # 3. Synthetic description (if no CLIP prediction available)
    if not clip_prediction:
        synthetic = generate_synthetic_description(image_path, symbol)
        if synthetic:
            results['synthetic_description'] = synthetic
    
    # 4. Telegram commentary
    telegram_comment = generate_telegram_alert_commentary(symbol, tjde_data)
    results['telegram_commentary'] = telegram_comment
    
    print(f"[GPT COMPREHENSIVE] Completed {len(results)} analyses for {symbol}")
    return results


def should_audit_scoring(tjde_data: Dict) -> bool:
    """Determine if scoring audit is needed based on potential inconsistencies"""
    score = tjde_data.get('final_score', 0.0)
    decision = tjde_data.get('decision', '')
    
    # Check for major score-decision mismatches
    if score > 0.6 and decision == 'avoid':
        return True
    if score < 0.4 and decision in ['consider_entry', 'join_trend']:
        return True
    
    return False


def test_gpt_label_extraction():
    """Test GPT label extraction and file renaming system"""
    
    # Test cases for label extraction
    test_cases = [
        ("Market pokazuje silny pullback z reakcją na poziomie wsparcia", "trend_pullback_reacted"),
        ("Widzimy pullback ale brak reakcji kupujących", "trend_pullback_failed"),
        ("Kontynuacja trendu wzrostowego z dynamiczną kontynuacją", "trend_continuation"),
        ("Trend pokazuje wyczerpanie trendu i osłabienie", "trend_exhaustion"),
        ("Fakeout na poziomie oporu z odrzuceniem", "fakeout_on_resistance"),
        ("Konsolidacja boczna bez wyraźnego kierunku", "range_consolidation"),
        ("Pullback shows strong reaction and bounce from support level", "trend_pullback_reacted"),
        ("Trend continuation with momentum continuation pattern", "trend_continuation"),
        ("Fake breakout above resistance level failed", "fakeout_breakout"),
        ("No clear trend direction, sideways movement", "no_trend"),
        ("Random commentary without specific patterns", "unknown")
    ]
    
    print("[GPT LABEL TEST] Testing label extraction...")
    
    for i, (commentary, expected) in enumerate(test_cases, 1):
        extracted = extract_primary_label(commentary)
        status = "✅" if extracted == expected else "❌"
        print(f"{status} Test {i}: '{commentary[:50]}...' → '{extracted}' (expected: '{expected}')")
        
    # Test file renaming function
    print("\n[FILE RENAME TEST] Testing automatic renaming...")
    
    import tempfile
    import os
    
    try:
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test PNG and JSON files
            test_png = os.path.join(temp_dir, "BTCUSDT_2025-06-25_20:30:00_phase_decision_tjde.png")
            test_json = os.path.join(temp_dir, "BTCUSDT_2025-06-25_20:30:00_phase_decision_tjde.json")
            
            # Create dummy files
            with open(test_png, 'w') as f:
                f.write("dummy png content")
            with open(test_json, 'w') as f:
                f.write('{"symbol": "BTCUSDT", "setup": "unknown"}')
            
            # Test renaming with GPT commentary
            test_commentary = "Market shows strong pullback with reaction from support level"
            new_png, new_json = rename_chart_files_with_gpt_label(test_png, test_commentary)
            
            # Check if files were renamed correctly
            expected_label = extract_primary_label(test_commentary)
            if expected_label in new_png:
                print(f"✅ File renaming successful: {os.path.basename(new_png)}")
            else:
                print(f"❌ File renaming failed: expected '{expected_label}' in filename")
                
    except Exception as e:
        print(f"❌ File rename test error: {e}")
    
    print("\n[GPT LABEL TEST] Testing completed")


def main():
    """Test GPT commentary system"""
    print("Testing GPT Commentary System with Label Extraction...")
    
    # Run label extraction tests
    test_gpt_label_extraction()
    
    print("\nGPT Commentary System ready for production integration")


if __name__ == "__main__":
    main()