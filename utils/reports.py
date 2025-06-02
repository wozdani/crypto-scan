import os
import json
import zipfile
from datetime import datetime, timedelta, timezone

def save_stage_signal(symbol, score, stage2_pass, compressed, stage1g_active):
    """
    Save stage detection results to individual symbol file
    """
    try:
        signal_data = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ppwcs_score': score,
            'stage_minus2': stage2_pass,
            'stage_minus1': compressed,
            'stage_1g': stage1g_active
        }
        
        # Ensure reports directory exists
        os.makedirs("reports/signals", exist_ok=True)
        
        # Save to daily file
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        signals_file = f"reports/signals/signals_{date_str}.json"
        
        # Load existing signals or create new list
        if os.path.exists(signals_file):
            with open(signals_file, 'r') as f:
                signals = json.load(f)
        else:
            signals = []
            
        signals.append(signal_data)
        
        with open(signals_file, 'w') as f:
            json.dump(signals, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error saving stage signal for {symbol}: {e}")

def save_conditional_reports():
    """
    Generate and save conditional reports based on current data
    """
    try:
        # High score symbols report
        generate_high_score_report()
        
        # Daily summary report
        generate_daily_summary()
        
        # Alert frequency report
        generate_alert_frequency_report()
        
        print("📊 Conditional reports generated")
        
    except Exception as e:
        print(f"❌ Error generating conditional reports: {e}")

def generate_high_score_report():
    """
    Generate report for symbols with high PPWCS scores
    """
    try:
        from utils.scoring import get_top_performers
        
        top_performers = get_top_performers(limit=20, hours=24)
        
        if not top_performers:
            return
            
        report = {
            'report_type': 'high_score_symbols',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'criteria': 'Top 20 symbols by PPWCS score in last 24 hours',
            'symbols': top_performers,
            'summary': {
                'total_symbols': len(top_performers),
                'avg_score': sum(s['score'] for s in top_performers) / len(top_performers),
                'max_score': max(s['score'] for s in top_performers),
                'min_score': min(s['score'] for s in top_performers)
            }
        }
        
        # Save report
        os.makedirs("reports/high_scores", exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
        report_file = f"reports/high_scores/high_scores_{date_str}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error generating high score report: {e}")

def generate_daily_summary():
    """
    Generate daily summary report
    """
    try:
        # Get today's data
        today = datetime.now(timezone.utc).date()
        
        # Load today's signals
        signals_file = f"reports/signals/signals_{today.strftime('%Y%m%d')}.json"
        if os.path.exists(signals_file):
            with open(signals_file, 'r') as f:
                signals = json.load(f)
        else:
            signals = []
            
        # Load today's alerts
        from utils.scoring import get_recent_alerts
        alerts = []
        try:
            alerts_file = "data/alerts/alerts_history.json"
            if os.path.exists(alerts_file):
                with open(alerts_file, 'r') as f:
                    all_alerts = json.load(f)
                    
                # Filter for today's alerts
                for alert in all_alerts:
                    try:
                        alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
                        if alert_time.date() == today:
                            alerts.append(alert)
                    except:
                        continue
        except:
            pass
            
        # Generate summary
        summary = {
            'report_type': 'daily_summary',
            'date': today.isoformat(),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'statistics': {
                'total_scans': len(signals),
                'total_alerts': len(alerts),
                'unique_symbols_scanned': len(set(s.get('symbol', '') for s in signals)),
                'unique_symbols_alerted': len(set(extract_symbol_from_alert(a.get('message', '')) for a in alerts)),
                'avg_score': sum(s.get('ppwcs_score', 0) for s in signals) / len(signals) if signals else 0,
                'max_score': max(s.get('ppwcs_score', 0) for s in signals) if signals else 0,
                'high_score_count': sum(1 for s in signals if s.get('ppwcs_score', 0) >= 80)
            },
            'top_performers': sorted(
                [s for s in signals if s.get('ppwcs_score', 0) > 0],
                key=lambda x: x.get('ppwcs_score', 0),
                reverse=True
            )[:10]
        }
        
        # Save summary
        os.makedirs("reports/daily", exist_ok=True)
        summary_file = f"reports/daily/summary_{today.strftime('%Y%m%d')}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error generating daily summary: {e}")

def generate_alert_frequency_report():
    """
    Generate report on alert frequency by symbol
    """
    try:
        from utils.scoring import get_recent_alerts
        
        # Get alerts from last 7 days
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        alerts_file = "data/alerts/alerts_history.json"
        if not os.path.exists(alerts_file):
            return
            
        with open(alerts_file, 'r') as f:
            all_alerts = json.load(f)
            
        # Filter recent alerts
        recent_alerts = []
        for alert in all_alerts:
            try:
                alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
                if alert_time > cutoff_time:
                    recent_alerts.append(alert)
            except:
                continue
                
        # Count alerts by symbol
        symbol_counts = {}
        for alert in recent_alerts:
            symbol = extract_symbol_from_alert(alert.get('message', ''))
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
        # Generate report
        frequency_report = {
            'report_type': 'alert_frequency',
            'period': '7_days',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_alerts': len(recent_alerts),
            'unique_symbols': len(symbol_counts),
            'symbol_frequencies': sorted(
                symbol_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            'high_frequency_symbols': [
                symbol for symbol, count in symbol_counts.items() if count >= 5
            ]
        }
        
        # Save report
        os.makedirs("reports/frequency", exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        frequency_file = f"reports/frequency/frequency_{date_str}.json"
        
        with open(frequency_file, 'w') as f:
            json.dump(frequency_report, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error generating alert frequency report: {e}")

def extract_symbol_from_alert(message):
    """
    Extract symbol from alert message
    """
    try:
        if not message:
            return None
            
        # Look for pattern like "*SYMBOL*"
        import re
        match = re.search(r'\*([A-Z0-9]+)\*', message)
        if match:
            return match.group(1)
            
        return None
        
    except:
        return None

def compress_reports_to_zip():
    """
    Compress old reports to zip files to save space
    """
    try:
        # Compress reports older than 7 days
        cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=7)
        
        report_dirs = ['signals', 'daily', 'frequency', 'high_scores']
        
        for report_dir in report_dirs:
            dir_path = f"reports/{report_dir}"
            if not os.path.exists(dir_path):
                continue
                
            # Find old files
            old_files = []
            for filename in os.listdir(dir_path):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(dir_path, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time.date() < cutoff_date:
                    old_files.append(file_path)
                    
            if old_files:
                # Create zip file
                zip_filename = f"reports/{report_dir}_archive_{cutoff_date.strftime('%Y%m%d')}.zip"
                
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in old_files:
                        zipf.write(file_path, os.path.basename(file_path))
                        os.remove(file_path)  # Remove original file
                        
                print(f"📦 Compressed {len(old_files)} old {report_dir} files to {zip_filename}")
                
    except Exception as e:
        print(f"❌ Error compressing reports: {e}")

def get_latest_reports(report_type='all', limit=10):
    """
    Get latest reports of specified type
    """
    try:
        reports = []
        
        if report_type in ['all', 'daily']:
            daily_dir = "reports/daily"
            if os.path.exists(daily_dir):
                for filename in sorted(os.listdir(daily_dir), reverse=True)[:limit]:
                    if filename.endswith('.json'):
                        file_path = os.path.join(daily_dir, filename)
                        with open(file_path, 'r') as f:
                            report = json.load(f)
                            report['file_path'] = file_path
                            reports.append(report)
                            
        if report_type in ['all', 'high_scores']:
            scores_dir = "reports/high_scores"
            if os.path.exists(scores_dir):
                for filename in sorted(os.listdir(scores_dir), reverse=True)[:limit]:
                    if filename.endswith('.json'):
                        file_path = os.path.join(scores_dir, filename)
                        with open(file_path, 'r') as f:
                            report = json.load(f)
                            report['file_path'] = file_path
                            reports.append(report)
                            
        return reports[:limit]
        
    except Exception as e:
        print(f"❌ Error getting latest reports: {e}")
        return []

def cleanup_old_reports(days_to_keep=30):
    """
    Clean up reports older than specified days
    """
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        report_paths = [
            "reports/signals",
            "reports/daily", 
            "reports/frequency",
            "reports/high_scores"
        ]
        
        total_removed = 0
        
        for report_path in report_paths:
            if not os.path.exists(report_path):
                continue
                
            for filename in os.listdir(report_path):
                file_path = os.path.join(report_path, filename)
                
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        total_removed += 1
                        
        if total_removed > 0:
            print(f"🧹 Cleaned up {total_removed} old report files")
            
    except Exception as e:
        print(f"❌ Error cleaning up old reports: {e}")
