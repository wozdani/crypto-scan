import os
import json
from datetime import datetime, timedelta
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
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
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

def send_report_to_gpt(symbol: str, data: dict, tp_forecast: dict, alert_level: str = "strong") -> str:
    """Enhanced GPT feedback for high-confidence signals (PPWCS >= 80)"""
    try:
        ppwcs = data.get("ppwcs_score", 0)
        whale = data.get("whale_activity", False)
        inflow = data.get("dex_inflow_usd", 0)
        compressed = data.get("compressed", False)
        stage1g = data.get("stage1g_active", False)
        pure_acc = data.get("pure_accumulation", False)
        social_spike = data.get("social_spike", False)
        heatmap_exhaustion = data.get("heatmap_exhaustion", False)
        sector_cluster = data.get("sector_clustered", False)
        spoofing = data.get("spoofing_suspected", False)
        vwap_pinned = data.get("vwap_pinned", False)
        vol_slope = data.get("volume_slope_up", False)

        timestamp = datetime.utcnow().strftime("%H:%M UTC")

        prompt = f"""You are an expert crypto analyst. Evaluate the following pre-pump signal:

Token: ${symbol.upper()}
PPWCS: {ppwcs}
Alert Level: {alert_level}
Detected at: {timestamp}

Stage –2.1:
• Whale Activity: {whale}
• DEX Inflow (USD): {inflow}
• Social Spike Detected: {social_spike}
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
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
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
            
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
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
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, respond with 'Connection successful'"}],
            max_tokens=50
        )
        
        if response.choices[0].message.content:
            return True, "Connection successful"
        else:
            return False, "No response received"
            
    except Exception as e:
        return False, str(e)

