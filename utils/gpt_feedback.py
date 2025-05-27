import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def send_report_to_chatgpt(symbol, signals, score):
    """
    Send detailed analysis report to ChatGPT for high-score symbols (≥80)
    """
    if not openai_client:
        print("⚠️ OpenAI API key not configured")
        return None
        
    try:
        # Prepare analysis prompt
        prompt = create_analysis_prompt(symbol, signals, score)
        
        # Send to ChatGPT
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert cryptocurrency analyst specializing in pre-pump detection. Provide concise, actionable analysis based on the technical data provided. Respond in JSON format."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.3
        )
        
        # Parse and save response
        analysis = json.loads(response.choices[0].message.content)
        save_gpt_analysis(symbol, analysis, score)
        
        print(f"🤖 ChatGPT analysis completed for {symbol}")
        return analysis
        
    except Exception as e:
        print(f"❌ Error sending report to ChatGPT for {symbol}: {e}")
        return None

def create_analysis_prompt(symbol, signals, score):
    """
    Create detailed analysis prompt for ChatGPT
    """
    prompt = f"""
    Analyze the following cryptocurrency pre-pump detection data for {symbol}:

    PPWCS Score: {score}/100

    Technical Signals:
    {json.dumps(signals, indent=2)}

    Current timestamp: {datetime.utcnow().isoformat()}

    Please provide analysis in the following JSON format:
    {{
        "symbol": "{symbol}",
        "risk_assessment": "low|medium|high",
        "confidence_level": 0-100,
        "key_indicators": ["list", "of", "important", "signals"],
        "price_prediction": "bullish|bearish|neutral",
        "time_horizon": "short_term|medium_term|long_term",
        "entry_recommendation": "immediate|wait|avoid",
        "risk_factors": ["potential", "risks"],
        "supporting_evidence": ["evidence", "points"],
        "summary": "Brief 2-3 sentence summary"
    }}

    Focus on:
    1. Technical signal strength and reliability
    2. Market timing considerations
    3. Risk/reward assessment
    4. Entry and exit strategies
    5. Potential false positive indicators
    """
    
    return prompt

def save_gpt_analysis(symbol, analysis, score):
    """
    Save ChatGPT analysis to file
    """
    try:
        analysis_entry = {
            'symbol': symbol,
            'score': score,
            'analysis': analysis,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Ensure directory exists
        os.makedirs("data/gpt_analysis", exist_ok=True)
        
        # Load existing analyses or create new list
        analysis_file = "data/gpt_analysis/gpt_reports.json"
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analyses = json.load(f)
        else:
            analyses = []
            
        analyses.append(analysis_entry)
        
        # Keep only last 500 analyses
        if len(analyses) > 500:
            analyses = analyses[-500:]
            
        with open(analysis_file, 'w') as f:
            json.dump(analyses, f, indent=2)
            
        # Also save individual report file
        report_filename = f"data/gpt_analysis/{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(analysis_entry, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error saving GPT analysis for {symbol}: {e}")

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
            
        from datetime import timedelta
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_analyses = []
        for analysis in analyses:
            try:
                analysis_time = datetime.fromisoformat(analysis.get('timestamp', ''))
                if analysis_time > cutoff_time:
                    recent_analyses.append(analysis)
            except:
                continue
                
        # Sort by timestamp (newest first) and limit results
        recent_analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return recent_analyses[:limit]
        
    except Exception as e:
        print(f"❌ Error getting recent GPT analyses: {e}")
        return []

def analyze_market_sentiment(symbol_list):
    """
    Analyze overall market sentiment for multiple symbols
    """
    if not openai_client:
        print("⚠️ OpenAI API key not configured")
        return None
        
    try:
        prompt = f"""
        Analyze the current cryptocurrency market sentiment for these symbols: {', '.join(symbol_list[:20])}
        
        Consider:
        1. Overall market trends
        2. Sector performance
        3. Risk factors
        4. Opportunities
        
        Provide analysis in JSON format:
        {{
            "overall_sentiment": "bullish|bearish|neutral",
            "market_phase": "accumulation|distribution|trending|consolidation",
            "risk_level": "low|medium|high",
            "top_opportunities": ["symbol1", "symbol2"],
            "symbols_to_avoid": ["symbol1", "symbol2"],
            "key_factors": ["factor1", "factor2"],
            "summary": "Brief market overview"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional cryptocurrency market analyst. Provide objective market analysis based on current trends."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.4
        )
        
        market_analysis = json.loads(response.choices[0].message.content)
        
        # Save market sentiment analysis
        sentiment_entry = {
            'analysis': market_analysis,
            'symbols_analyzed': symbol_list[:20],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        os.makedirs("data/market_sentiment", exist_ok=True)
        sentiment_file = f"data/market_sentiment/sentiment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(sentiment_file, 'w') as f:
            json.dump(sentiment_entry, f, indent=2)
            
        return market_analysis
        
    except Exception as e:
        print(f"❌ Error analyzing market sentiment: {e}")
        return None

def test_openai_connection():
    """
    Test OpenAI API connection
    """
    if not openai_client:
        return False, "API key not configured"
        
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, respond with 'Connection successful'"}],
            max_tokens=50
        )
        
        if response.choices[0].message.content:
            return True, "Connection successful"
        else:
            return False, "No response received"
            
    except Exception as e:
        return False, str(e)
