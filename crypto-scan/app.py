from flask import Flask, render_template, jsonify, request
import os
import json
from datetime import datetime, timedelta, timezone
from utils.scoring import get_top_performers, get_symbol_stats
from utils.reports import get_latest_reports
from utils.gpt_feedback import get_recent_gpt_analyses
from utils.telegram_bot import test_telegram_connection
from utils.gpt_feedback import test_openai_connection

app = Flask(__name__)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    try:
        # Test connections
        telegram_ok, telegram_msg = test_telegram_connection()
        openai_ok, openai_msg = test_openai_connection()
        
        # Get data file stats
        data_stats = get_data_file_stats()
        
        return jsonify({
            'status': 'running',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'connections': {
                'telegram': {'status': 'ok' if telegram_ok else 'error', 'message': telegram_msg},
                'openai': {'status': 'ok' if openai_ok else 'error', 'message': openai_msg}
            },
            'data_stats': data_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top-performers')
def get_top_performers_api():
    """Get top performing symbols"""
    try:
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 10))
        
        performers = get_top_performers(limit=limit, hours=hours)
        
        return jsonify({
            'performers': performers,
            'period_hours': hours,
            'total_count': len(performers)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent-alerts')
def get_recent_alerts_api():
    """Get recent alerts"""
    try:
        hours = int(request.args.get('hours', 24))
        
        alerts_file = "data/alerts/alerts_history.json"
        if not os.path.exists(alerts_file):
            return jsonify({'alerts': [], 'total_count': 0})
            
        with open(alerts_file, 'r') as f:
            all_alerts = json.load(f)
            
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in all_alerts:
            try:
                alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
                if alert_time > cutoff_time:
                    recent_alerts.append(alert)
            except:
                continue
                
        # Sort by timestamp (newest first)
        recent_alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'alerts': recent_alerts[:50],  # Limit to 50 most recent
            'total_count': len(recent_alerts),
            'period_hours': hours
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbol/<symbol>')
def get_symbol_info(symbol):
    """Get detailed information for a specific symbol"""
    try:
        symbol = symbol.upper()
        
        # Get symbol stats
        stats = get_symbol_stats(symbol)
        
        # Get recent scores
        scores_file = "data/scores/ppwcs_scores.json"
        recent_scores = []
        
        if os.path.exists(scores_file):
            with open(scores_file, 'r') as f:
                all_scores = json.load(f)
                
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
            
            for score in all_scores:
                try:
                    score_time = datetime.fromisoformat(score.get('timestamp', ''))
                    if score_time > cutoff_time and score.get('symbol') == symbol:
                        recent_scores.append(score)
                except:
                    continue
                    
        # Sort by timestamp
        recent_scores.sort(key=lambda x: x.get('timestamp', ''))
        
        return jsonify({
            'symbol': symbol,
            'stats': stats,
            'recent_scores': recent_scores[-50:]  # Last 50 scores
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gpt-analyses')
def get_gpt_analyses_api():
    """Get recent GPT analyses"""
    try:
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 10))
        
        analyses = get_recent_gpt_analyses(hours=hours, limit=limit)
        
        return jsonify({
            'analyses': analyses,
            'total_count': len(analyses),
            'period_hours': hours
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports')
def get_reports_api():
    """Get latest reports"""
    try:
        report_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 10))
        
        reports = get_latest_reports(report_type=report_type, limit=limit)
        
        return jsonify({
            'reports': reports,
            'total_count': len(reports),
            'type': report_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stage-minus1-alerts')
def get_stage_minus1_alerts_api():
    """Get recent Stage -1 alerts"""
    try:
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 10))
        
        alerts_file = os.path.join("data", "stage_minus1_alerts.json")
        stage_minus1_alerts = []
        
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r') as f:
                all_alerts = json.load(f)
            
            # Filter by time period
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            for alert in all_alerts:
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
                    if alert_time >= cutoff_time:
                        stage_minus1_alerts.append({
                            'symbol': alert['symbol'],
                            'market_tension': alert.get('market_tension', 'NIEZNANE'),
                            'rhythm_description': alert.get('rhythm_description', ''),
                            'timestamp': alert['timestamp'],
                            'alert_type': 'stage_minus1',
                            'tension_level': alert.get('tension_level', 'STANDARD')
                        })
                except (KeyError, ValueError):
                    continue
            
            # Sort by timestamp (newest first) and limit results
            stage_minus1_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            stage_minus1_alerts = stage_minus1_alerts[:limit]
        
        return jsonify({
            'stage_minus1_alerts': stage_minus1_alerts,
            'total_count': len(stage_minus1_alerts),
            'period_hours': hours
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-overview')
def get_market_overview():
    """Get market overview data"""
    try:
        # Get recent performers
        top_performers = get_top_performers(limit=5, hours=24)
        
        # Get alert statistics
        alerts_file = "data/alerts/alerts_history.json"
        alert_count_24h = 0
        
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
                
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            for alert in alerts:
                try:
                    alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
                    if alert_time > cutoff_time:
                        alert_count_24h += 1
                except:
                    continue
                    
        # Get score statistics
        scores_file = "data/scores/ppwcs_scores.json"
        avg_score_24h = 0
        high_score_count = 0
        
        if os.path.exists(scores_file):
            with open(scores_file, 'r') as f:
                scores = json.load(f)
                
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_scores = []
            
            for score in scores:
                try:
                    score_time = datetime.fromisoformat(score.get('timestamp', ''))
                    if score_time > cutoff_time:
                        score_value = score.get('score', 0)
                        recent_scores.append(score_value)
                        if score_value >= 80:
                            high_score_count += 1
                except:
                    continue
                    
            if recent_scores:
                avg_score_24h = sum(recent_scores) / len(recent_scores)
                
        return jsonify({
            'overview': {
                'alert_count_24h': alert_count_24h,
                'avg_score_24h': round(avg_score_24h, 1),
                'high_score_count_24h': high_score_count,
                'top_performers': top_performers,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_data_file_stats():
    """Get statistics about data files"""
    try:
        stats = {}
        
        data_files = {
            'alerts': 'data/alerts/alerts_history.json',
            'scores': 'data/scores/ppwcs_scores.json',
            'symbols': 'data/cache/symbols.json'
        }
        
        for name, filepath in data_files.items():
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                modification_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            record_count = len(data)
                        elif isinstance(data, dict):
                            record_count = len(data.get('symbols', [])) if name == 'symbols' else 1
                        else:
                            record_count = 1
                except:
                    record_count = 0
                    
                stats[name] = {
                    'exists': True,
                    'size_bytes': file_size,
                    'last_modified': modification_time.isoformat(),
                    'record_count': record_count
                }
            else:
                stats[name] = {'exists': False}
                
        return stats
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/api/whale-priority')
def get_whale_priority_api():
    """Get whale priority information"""
    try:
        priority_file = "data/priority/whale_priority_current.json"
        
        if not os.path.exists(priority_file):
            return jsonify({
                'priority_tokens': [],
                'total_count': 0,
                'last_update': None,
                'status': 'no_data'
            })
        
        with open(priority_file, 'r', encoding='utf-8') as f:
            priority_data = json.load(f)
        
        # Get priority tokens from the data (handle both old and new formats)
        if 'priority_tokens' in priority_data:
            # New format with priority_tokens array
            priority_tokens = priority_data['priority_tokens']
        else:
            # Old format with tokens object
            priority_tokens = []
            for symbol, info in priority_data.get('tokens', {}).items():
                priority_tokens.append({
                    'symbol': symbol,
                    'priority_score': info['priority_score'],
                    'whale_count': info['whale_count'],
                    'minutes_ago': info['minutes_ago'],
                    'under_watch': info.get('under_watch', False),
                    'whale_pattern': info.get('whale_pattern'),
                    'whale_score_boost': info.get('whale_score_boost', 0),
                    'last_whale': info.get('last_whale')
                })
            
            # Sort by priority score
            priority_tokens.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return jsonify({
            'priority_tokens': priority_tokens,
            'total_count': len(priority_tokens),
            'last_update': priority_data.get('timestamp'),
            'status': 'active' if priority_tokens else 'no_priority'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("data/alerts", exist_ok=True)
    os.makedirs("data/scores", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("🌐 Starting Crypto Scanner Web Interface...")
    print("🔗 Access dashboard at: http://localhost:5000")
    
    # Import request here to avoid circular import
    from flask import request
    
    app.run(host='0.0.0.0', port=5000, debug=False)
