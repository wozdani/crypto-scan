import os
import json
from datetime import datetime, timedelta, timezone
import openai
from dotenv import load_dotenv

# Załaduj API Key z .env
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Szablon promptu
PROMPT_TEMPLATE = """
Oceń jakość sygnału pre-pump dla tokena {symbol}. Dane wejściowe:
- PPWCS: {score}
- Tags: {tags}
- Compressed: {compressed}
- Stage 1g: {stage1g}

Oceń strukturę: czy wygląda na dojrzałą do wybicia? Czy jest ryzyko false breakout? Czy zalecasz wejście, obserwację czy wstrzymanie się?

Zwróć 2–4 zdania w języku polskim, ton ekspercki, zwięzły, ale stanowczy.
"""

def send_report_to_chatgpt(symbol: str, tags: list[str], score: float, compressed: bool, stage1g: bool) -> str:
    try:
        prompt = PROMPT_TEMPLATE.format(
            symbol=symbol,
            score=score,
            tags=", ".join(tags),
            compressed="tak" if compressed else "nie",
            stage1g="tak" if stage1g else "nie"
        )

        response = client.chat.completions.create(
            model="gpt-5",  # Upgraded to GPT-5 for enhanced crypto analysis capabilities
            messages=[
                {"role": "system", "content": "Jesteś ekspertem rynku kryptowalut specjalizującym się w analizie sygnałów pre-pump."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )

        return response.choices[0].message.content.strip() if response.choices[0].message.content else "No response"

    except Exception as e:
        print(f"GPT error: {e}")
        return "⚠️ Błąd podczas generowania analizy GPT."

def score_gpt_feedback(gpt_text):
    """
    GPT Feedback Auto-Scorer - automatycznie ocenia jakość alertu
    na podstawie odpowiedzi GPT i zwraca feedback_score (0-100)
    
    Args:
        gpt_text: tekst odpowiedzi z GPT
    
    Returns:
        int: score od 0 do 100
    """
    if not gpt_text or len(gpt_text.strip()) < 10:
        return 50  # Neutral score dla pustych odpowiedzi
    
    text = gpt_text.lower()
    
    # Podstawowy score na podstawie siły sygnału
    base_score = 60  # Neutralny punkt startowy
    
    # Silne sygnały (polskie i angielskie frazy)
    strong_signals = [
        "silny sygnał", "mocny sygnał", "bardzo obiecujący", "doskonały",
        "strong signal", "very likely", "excellent", "robust", "solid"
    ]
    
    # Umiarkowane sygnały
    moderate_signals = [
        "umiarkowany", "dobry", "przyzwoity", "solidny",
        "moderate", "good", "decent", "reasonable"
    ]
    
    # Słabe sygnały
    weak_signals = [
        "słaby", "ryzykowny", "niepewny", "wątpliwy",
        "weak", "risky", "uncertain", "doubtful"
    ]
    
    # Ustal podstawowy score
    if any(phrase in text for phrase in strong_signals):
        base_score = 85
    elif any(phrase in text for phrase in moderate_signals):
        base_score = 70
    elif any(phrase in text for phrase in weak_signals):
        base_score = 50
    
    # Bonusy za pozytywne wskaźniki
    positive_indicators = [
        "niskie ryzyko", "solidna struktura", "dobra konfirmacja", "stabilny wzrost",
        "low risk", "solid structure", "good confirmation", "stable growth"
    ]
    if any(phrase in text for phrase in positive_indicators):
        base_score += 10
    
    # Wysokie prawdopodobieństwo kontynuacji
    continuation_indicators = [
        "wysokie prawdopodobieństwo", "duża szansa", "kontynuacja prawdopodobna",
        "high probability", "likely to continue", "strong momentum"
    ]
    if any(phrase in text for phrase in continuation_indicators):
        base_score += 8
    
    # Pozytywne wskaźniki techniczne
    technical_positive = [
        "przełamanie oporu", "wzrost wolumenu", "akumulacja", "wybicie",
        "breakout", "volume increase", "accumulation", "bullish pattern"
    ]
    if any(phrase in text for phrase in technical_positive):
        base_score += 5
    
    # Kary za negatywne wskaźniki
    negative_indicators = [
        "fakeout", "brak konfirmacji", "wysokie ryzyko", "możliwy spadek",
        "lack of confirmation", "high risk", "potential decline", "false signal"
    ]
    if any(phrase in text for phrase in negative_indicators):
        base_score -= 15
    
    # Kary za pump and dump ryzyko
    pnd_indicators = [
        "pump and dump", "sztuczny wzrost", "manipulacja", "social hype",
        "artificial pump", "manipulation", "unsustainable"
    ]
    if any(phrase in text for phrase in pnd_indicators):
        base_score -= 10
    
    # Kary za niepewność
    uncertainty_indicators = [
        "niepewność", "trudne do przewidzenia", "mieszane sygnały",
        "uncertainty", "hard to predict", "mixed signals", "unclear"
    ]
    if any(phrase in text for phrase in uncertainty_indicators):
        base_score -= 8
    
    # Bonus za szczegółową analizę (długość tekstu)
    if len(gpt_text) > 200:
        base_score += 3
    
    # Clamp do zakresu 0-100
    return max(0, min(100, base_score))

def categorize_feedback_score(score):
    """
    Kategoryzuje feedback score na poziomy jakości
    
    Args:
        score: numeryczny score 0-100
    
    Returns:
        tuple: (kategoria, opis, emoji)
    """
    if score >= 85:
        return ("ULTRA_CLEAN", "Ultra clean signal, high continuation probability", "🔵")
    elif score >= 70:
        return ("STRONG", "Strong signal with good potential", "🟢")
    elif score >= 55:
        return ("MODERATE", "Moderate signal with some risk", "🟡")
    elif score >= 40:
        return ("WEAK", "Weak signal, higher risk", "🟠")
    else:
        return ("POOR", "Poor signal, possible fakeout", "🔴")

def get_feedback_statistics(feedback_scores):
    """
    Oblicza statystyki dla listy feedback scores
    
    Args:
        feedback_scores: lista numerycznych scores
    
    Returns:
        dict: statystyki feedback scores
    """
    if not feedback_scores:
        return {
            "avg_score": 0,
            "max_score": 0,
            "min_score": 0,
            "ultra_clean_count": 0,
            "strong_count": 0,
            "total_count": 0
        }
    
    avg_score = sum(feedback_scores) / len(feedback_scores)
    ultra_clean_count = sum(1 for score in feedback_scores if score >= 85)
    strong_count = sum(1 for score in feedback_scores if score >= 70)
    
    return {
        "avg_score": round(avg_score, 1),
        "max_score": max(feedback_scores),
        "min_score": min(feedback_scores),
        "ultra_clean_count": ultra_clean_count,
        "strong_count": strong_count,
        "total_count": len(feedback_scores)
    }

def send_report_to_gpt(symbol: str, data: dict, tp_forecast: dict, alert_level: str = "strong") -> str:
    """Enhanced GPT feedback for high-confidence signals (PPWCS >= 80)"""
    try:
        ppwcs = data.get("ppwcs_score", 0)
        whale = data.get("whale_activity", False)
        inflow = data.get("dex_inflow_usd", 0)
        compressed = data.get("compressed", False)
        stage1g = data.get("stage1g_active", False)
        pure_acc = data.get("pure_accumulation", False)
        # social_spike removed - handled by Stage -2.2 tags
        heatmap_exhaustion = data.get("heatmap_exhaustion", False)
        sector_cluster = data.get("sector_clustered", False)
        spoofing = data.get("spoofing_suspected", False)
        vwap_pinned = data.get("vwap_pinned", False)
        vol_slope = data.get("volume_slope_up", False)

        timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")

        prompt = f"""You are an expert crypto analyst. Evaluate the following pre-pump signal:

Token: ${symbol.upper()}
PPWCS: {ppwcs}
Alert Level: {alert_level}
Detected at: {timestamp}

Stage –2.1:
• Whale Activity: {whale}
• DEX Inflow (USD): {inflow}
• News/Tag Analysis: {data.get("event_tag", "none")}
• Sector Time Clustering Active: {sector_cluster}

Stage –1:
• Compressed Structure: {compressed}

Stage 1g:
• Active: {stage1g}
• Pure Accumulation (No Social): {pure_acc}

Structural Detectors:
• Heatmap Exhaustion: {heatmap_exhaustion}
• Spoofing Suspected: {spoofing}
• VWAP Pinned: {vwap_pinned}
• Volume Slope Up: {vol_slope}

TP Forecast:
• TP1: +{int(tp_forecast['TP1'] * 100)}%
• TP2: +{int(tp_forecast['TP2'] * 100)}%
• TP3: +{int(tp_forecast['TP3'] * 100)}%
• Trailing: {tp_forecast['trailing_tp']}

Evaluate the quality and strength of this signal. Provide a confident but concise assessment in 3 short sentences, including any risk factors and probability of continuation. Reply in Polish."""

        response = client.chat.completions.create(
            model="gpt-5",  # Upgraded to GPT-5 for enhanced crypto signal evaluation capabilities
            messages=[
                {"role": "system", "content": "You are a crypto signal quality evaluator. Respond in Polish."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )

        return response.choices[0].message.content.strip() if response.choices[0].message.content else "No response"

    except Exception as e:
        print(f"❌ GPT feedback error: {e}")
        return f"[GPT ERROR] {str(e)}"

def get_recent_gpt_analyses(hours=24, limit=10):
    """
    Get recent ChatGPT analyses
    """
    try:
        analysis_file = "data/gpt_analysis/gpt_reports.json"
        if not os.path.exists(analysis_file):
            return []
            
        with open(analysis_file, 'r') as f:
            analyses = json.load(f)
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_analyses = []
        for analysis in analyses:
            try:
                analysis_time = datetime.fromisoformat(analysis.get('timestamp', ''))
                if analysis_time > cutoff_time:
                    recent_analyses.append(analysis)
            except:
                continue
                
        recent_analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return recent_analyses[:limit]
        
    except Exception as e:
        print(f"❌ Error getting recent GPT analyses: {e}")
        return []

def test_openai_connection():
    """
    Test OpenAI API connection
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": "Hello, respond with 'Connection successful'"}],
            max_completion_tokens=50
        )
        
        if response.choices[0].message.content:
            return True, "Connection successful"
        else:
            return False, "No response received"
            
    except Exception as e:
        return False, str(e)

