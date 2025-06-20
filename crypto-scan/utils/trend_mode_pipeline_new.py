"""
Complete Trend Mode Pipeline - Advanced 15M/5M Analysis
Replaces old trend detection with sophisticated multi-timeframe system
"""

from .trend_stage_minus1 import detect_stage_minus1
from .orderbook_sentiment import detect_orderbook_sentiment
from .bybit_orderbook import get_orderbook_with_fallback
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# Removed old directional_flow_detector - replaced with advanced trend mode
import json
from datetime import datetime, timezone

def fetch_5m_prices_bybit(symbol: str, count: int = 24):
    """
    Fetch 5-minute close prices from Bybit API
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        count: Number of 5-minute candles to fetch
        
    Returns:
        List of close prices
    """
    import requests
    
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": "5",
        "limit": min(count, 200)
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            prices = [float(candle[4]) for candle in data["result"]["list"]]
            return prices[:count] if len(prices) >= count else prices
        
        return []
    
    except Exception as e:
        print(f"⚠️ Error fetching 5m prices for {symbol}: {str(e)}")
        return []

def compute_trend_mode_score(symbol: str, prices_5m: list, prices_1m: list, orderbook_data: dict) -> tuple:
    """
    Advanced Trend Mode scoring using 15M/5M analysis
    
    Args:
        symbol: Symbol do analizy
        prices_5m: Lista cen 5-minutowych (zachowana dla kompatybilności)
        prices_1m: Lista cen 1-minutowych (zachowana dla kompatybilności)
        orderbook_data: Dane orderbook (zachowane dla kompatybilności)
        
    Returns:
        tuple: (score, reasons) gdzie score 0-100+ i reasons to lista opisów
    """
    try:
        print(f"🎯 [TREND DEBUG] Starting advanced trend mode analysis for {symbol}")
        
        # Use advanced trend mode detection
        from detectors.trend_mode_advanced import detect_advanced_trend_mode
        
        result = detect_advanced_trend_mode(symbol)
        
        score = result.get("trend_score", 0)
        
        if result.get("trend_mode", False):
            reasons = [result.get("description", "Advanced trend mode detected")]
            print(f"🚀 [TREND DEBUG] {symbol} - {result['description']} ({score} points)")
        else:
            reasons = []
            print(f"📊 [TREND DEBUG] {symbol} - {result['description']}")
        
        # Add fallback detection for other signals (if advanced detection fails)
        if score == 0:
            print(f"📈 [TREND DEBUG] {symbol} - Running fallback detectors...")
            
            # Import fallback detector functions  
            try:
                from detectors.flow_consistency import detect_flow_consistency_index
                from detectors.vwap_pinning import detect_vwap_pinning, calculate_vwap_values
                from detectors.human_flow import detect_human_like_flow
                
                # Run selected fallback detectors with reduced scoring
                if prices_5m:
                    # Flow Consistency (2 punkty)
                    try:
                        consistency_result = detect_flow_consistency_index(prices_5m)
                        if consistency_result[0]:
                            score += 2
                            reasons.append("flow consistency – stable trend momentum")
                    except Exception:
                        pass
                    
                    # VWAP Pinning (3 punkty)
                    try:
                        if len(prices_5m) >= 20:
                            vwap_values = calculate_vwap_values(prices_5m, [1000] * len(prices_5m))
                            vwap_result = detect_vwap_pinning(prices_5m, vwap_values)
                            if vwap_result[0]:
                                score += 3
                                reasons.append("vwap pinning – price anchored to VWAP")
                    except Exception:
                        pass
                    
                    # Human Flow (3 punkty)
                    try:
                        if len(prices_5m) >= 15:
                            human_result = detect_human_like_flow(prices_5m)
                            if human_result[0]:
                                score += 3
                                reasons.append("human flow – psychological trading pattern")
                    except Exception:
                        pass
                
            except ImportError:
                print(f"⚠️ [TREND DEBUG] {symbol} - Fallback detectors not available")
        
        print(f"🎯 [TREND DEBUG] {symbol} - Final score: {score}/100+, Active detectors: {len(reasons)}")
        if reasons:
            print(f"🔍 [TREND DEBUG] Active signals: {', '.join(reasons)}")
        
        return score, reasons
        
    except Exception as e:
        print(f"⚠️ Error in advanced trend mode scoring for {symbol}: {str(e)}")
        return 0, [f"Error: {str(e)[:50]}"]

def detect_trend_mode(symbol, data, get_orderbook_func):
    """
    Advanced trend mode detection using 15M/5M analysis
    
    Parametry:
        symbol (str): Symbol tokena (np. 'PEPEUSDT')
        data (list): Świece 15M: [timestamp, open, high, low, close, volume]
        get_orderbook_func (callable): Funkcja pobierająca orderbook z API

    Zwraca:
        (bool, str): (Czy trend aktywny, Powód)
    """
    try:
        from detectors.trend_mode_advanced import detect_advanced_trend_mode
        
        result = detect_advanced_trend_mode(symbol)
        
        trend_active = result.get("trend_mode", False)
        description = result.get("description", "No trend detected")
        
        return trend_active, description
        
    except Exception as e:
        print(f"⚠️ Error in trend mode detection for {symbol}: {e}")
        return False, f"Error: {str(e)[:50]}"

def detect_trend_mode_extended(symbol, candle_data):
    """
    Extended trend mode detection with additional details
    
    Args:
        symbol: Symbol trading pair (np. 'BTCUSDT')
        candle_data: Lista świec OHLCV
        
    Returns:
        tuple: (bool, str, dict) - (trend_aktywny, opis, szczegóły)
    """
    try:
        from detectors.trend_mode_advanced import detect_advanced_trend_mode
        
        result = detect_advanced_trend_mode(symbol)
        
        trend_active = result.get("trend_mode", False)
        description = result.get("description", "No trend detected")
        details = result.get("details", {})
        
        return trend_active, description, details
        
    except Exception as e:
        print(f"⚠️ Error in extended trend mode detection for {symbol}: {e}")
        return False, f"Error: {str(e)[:50]}", {"error": str(e)}

def save_trend_mode_alert(alert_data):
    """
    Zapisuje alert trend mode do pliku JSON
    
    Args:
        alert_data: Dictionary z danymi alertu
    """
    try:
        # Save to main alerts file
        alerts_file = "data/trend_mode_alerts.json"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Load existing alerts
        try:
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            alerts = []
        
        # Add new alert
        alerts.append(alert_data)
        
        # Keep only last 100 alerts
        alerts = alerts[-100:]
        
        # Save back to file
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
            
    except Exception as e:
        print(f"⚠️ Error saving trend mode alert: {e}")

def save_trend_mode_report(symbol, alert_data):
    """
    Zapisuje raport trend mode do osobnego pliku w folderze reports
    
    Args:
        symbol: Trading symbol
        alert_data: Dictionary z danymi alertu
    """
    try:
        # Create reports directory
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create daily report file
        today = datetime.now().strftime("%Y%m%d")
        report_file = f"{reports_dir}/trend_mode_alerts_{today}.json"
        
        # Load existing reports for today
        try:
            with open(report_file, 'r') as f:
                reports = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            reports = []
        
        # Add new report
        reports.append(alert_data)
        
        # Save back to file
        with open(report_file, 'w') as f:
            json.dump(reports, f, indent=2, default=str)
            
    except Exception as e:
        print(f"⚠️ Error saving trend mode report: {e}")

def get_trend_mode_statistics():
    """
    Pobiera statystyki trend mode z ostatnich alertów
    
    Returns:
        dict: Statystyki trend mode
    """
    try:
        alerts_file = "data/trend_mode_alerts.json"
        
        try:
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"total_alerts": 0, "avg_score": 0}
        
        if not alerts:
            return {"total_alerts": 0, "avg_score": 0}
        
        total_alerts = len(alerts)
        scores = [alert.get("comprehensive_score", 0) for alert in alerts]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "total_alerts": total_alerts,
            "avg_score": round(avg_score, 2),
            "max_score": max(scores) if scores else 0,
            "recent_alerts": alerts[-5:]  # Last 5 alerts
        }
        
    except Exception as e:
        print(f"⚠️ Error getting trend mode statistics: {e}")
        return {"error": str(e)}

def analyze_trend_mode_patterns(symbol_list, candle_data_dict):
    """
    Analizuje wzorce trend mode dla listy symboli
    
    Args:
        symbol_list: Lista symboli do analizy
        candle_data_dict: Dict z danymi świec dla każdego symbolu
        
    Returns:
        dict: Wyniki analizy trend mode
    """
    results = {}
    
    for symbol in symbol_list:
        try:
            candle_data = candle_data_dict.get(symbol, [])
            trend_active, description, details = detect_trend_mode_extended(symbol, candle_data)
            
            results[symbol] = {
                "trend_active": trend_active,
                "description": description,
                "details": details,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            results[symbol] = {
                "trend_active": False,
                "description": f"Error: {str(e)[:50]}",
                "details": {"error": str(e)},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    return results