"""
GPT Feedback Integration Module for Pump Analysis

This module integrates GPT feedback from crypto-scan system to enhance pump analysis
with recent AI insights for tokens being analyzed.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

class GPTFeedbackIntegration:
    """Handles integration with crypto-scan GPT feedback system"""
    
    def __init__(self):
        self.crypto_scan_path = "../crypto-scan"
        self.gpt_reports_file = "data/gpt_analysis/gpt_reports.json"
        self.feedback_file = "data/feedback"
        
    def get_recent_gpt_feedback(self, symbol: str, hours: int = 2) -> Optional[Dict]:
        """
        Get recent GPT feedback for a specific symbol from crypto-scan
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            hours: Number of hours back to search (default 2)
            
        Returns:
            Dict with recent GPT feedback or None if not found
        """
        try:
            # Clean symbol format for matching
            clean_symbol = symbol.replace('USDT', '').upper()
            
            # Check main GPT reports file
            gpt_reports_path = os.path.join(self.crypto_scan_path, self.gpt_reports_file)
            if os.path.exists(gpt_reports_path):
                feedback = self._search_gpt_reports(gpt_reports_path, clean_symbol, hours)
                if feedback:
                    return feedback
            
            # Check individual feedback files
            feedback_dir = os.path.join(self.crypto_scan_path, self.feedback_file)
            if os.path.exists(feedback_dir):
                feedback = self._search_feedback_files(feedback_dir, clean_symbol, hours)
                if feedback:
                    return feedback
                    
            return None
            
        except Exception as e:
            print(f"⚠️ Error getting GPT feedback for {symbol}: {e}")
            return None
    
    def _search_gpt_reports(self, file_path: str, symbol: str, hours: int) -> Optional[Dict]:
        """Search GPT reports file for recent feedback"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reports = json.load(f)
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Search for matching symbol in recent reports
            for report in reports:
                try:
                    report_symbol = report.get('symbol', '').upper()
                    timestamp_str = report.get('timestamp', '')
                    
                    if report_symbol == symbol:
                        # Parse timestamp
                        report_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
                        if report_time > cutoff_time:
                            return {
                                'source': 'gpt_reports',
                                'symbol': symbol,
                                'timestamp': timestamp_str,
                                'age_hours': (datetime.now(timezone.utc) - report_time).total_seconds() / 3600,
                                'score': report.get('score', 0),
                                'analysis': report.get('analysis', {}),
                                'gpt_text': report.get('analysis', {}).get('summary', 'No summary available')
                            }
                except Exception as e:
                    continue
                    
            return None
            
        except Exception as e:
            print(f"⚠️ Error searching GPT reports: {e}")
            return None
    
    def _search_feedback_files(self, feedback_dir: str, symbol: str, hours: int) -> Optional[Dict]:
        """Search individual feedback files for recent feedback"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Look for feedback files matching the symbol
            for filename in os.listdir(feedback_dir):
                if filename.endswith('.json') and symbol.lower() in filename.lower():
                    file_path = os.path.join(feedback_dir, filename)
                    
                    try:
                        # Check file modification time
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)
                        
                        if file_time > cutoff_time:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                feedback_data = json.load(f)
                            
                            return {
                                'source': 'feedback_file',
                                'symbol': symbol,
                                'filename': filename,
                                'timestamp': file_time.isoformat(),
                                'age_hours': (datetime.now(timezone.utc) - file_time).total_seconds() / 3600,
                                'feedback_data': feedback_data,
                                'gpt_text': feedback_data.get('gpt_analysis', 'No GPT analysis available')
                            }
                    except Exception as e:
                        continue
                        
            return None
            
        except Exception as e:
            print(f"⚠️ Error searching feedback files: {e}")
            return None
    
    def get_all_recent_feedback(self, hours: int = 2) -> List[Dict]:
        """
        Get all recent GPT feedback from crypto-scan system
        
        Args:
            hours: Number of hours back to search
            
        Returns:
            List of recent feedback entries
        """
        try:
            all_feedback = []
            
            # Get from GPT reports
            gpt_reports_path = os.path.join(self.crypto_scan_path, self.gpt_reports_file)
            if os.path.exists(gpt_reports_path):
                feedback_list = self._get_all_from_gpt_reports(gpt_reports_path, hours)
                all_feedback.extend(feedback_list)
            
            # Get from feedback files
            feedback_dir = os.path.join(self.crypto_scan_path, self.feedback_file)
            if os.path.exists(feedback_dir):
                feedback_list = self._get_all_from_feedback_files(feedback_dir, hours)
                all_feedback.extend(feedback_list)
            
            # Sort by timestamp (newest first)
            all_feedback.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return all_feedback
            
        except Exception as e:
            print(f"⚠️ Error getting all recent feedback: {e}")
            return []
    
    def _get_all_from_gpt_reports(self, file_path: str, hours: int) -> List[Dict]:
        """Get all recent feedback from GPT reports file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reports = json.load(f)
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_feedback = []
            
            for report in reports:
                try:
                    timestamp_str = report.get('timestamp', '')
                    report_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    if report_time > cutoff_time:
                        recent_feedback.append({
                            'source': 'gpt_reports',
                            'symbol': report.get('symbol', '').upper(),
                            'timestamp': timestamp_str,
                            'age_hours': (datetime.now(timezone.utc) - report_time).total_seconds() / 3600,
                            'score': report.get('score', 0),
                            'analysis': report.get('analysis', {}),
                            'gpt_text': report.get('analysis', {}).get('summary', 'No summary available')
                        })
                except Exception as e:
                    continue
                    
            return recent_feedback
            
        except Exception as e:
            print(f"⚠️ Error getting all from GPT reports: {e}")
            return []
    
    def _get_all_from_feedback_files(self, feedback_dir: str, hours: int) -> List[Dict]:
        """Get all recent feedback from individual files"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_feedback = []
            
            for filename in os.listdir(feedback_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(feedback_dir, filename)
                    
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)
                        
                        if file_time > cutoff_time:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                feedback_data = json.load(f)
                            
                            # Extract symbol from filename
                            symbol = filename.replace('.json', '').upper()
                            
                            recent_feedback.append({
                                'source': 'feedback_file',
                                'symbol': symbol,
                                'filename': filename,
                                'timestamp': file_time.isoformat(),
                                'age_hours': (datetime.now(timezone.utc) - file_time).total_seconds() / 3600,
                                'feedback_data': feedback_data,
                                'gpt_text': feedback_data.get('gpt_analysis', 'No GPT analysis available')
                            })
                    except Exception as e:
                        continue
                        
            return recent_feedback
            
        except Exception as e:
            print(f"⚠️ Error getting all from feedback files: {e}")
            return []
    
    def format_feedback_for_pump_analysis(self, feedback: Dict) -> str:
        """
        Format GPT feedback for inclusion in pump analysis
        
        Args:
            feedback: Feedback dictionary from get_recent_gpt_feedback
            
        Returns:
            Formatted string for pump analysis context
        """
        if not feedback:
            return ""
        
        try:
            symbol = feedback.get('symbol', 'Unknown')
            age = feedback.get('age_hours', 0)
            source = feedback.get('source', 'unknown')
            
            # Format age
            if age < 1:
                age_str = f"{int(age * 60)} minut temu"
            else:
                age_str = f"{age:.1f}h temu"
            
            # Get GPT text
            gpt_text = feedback.get('gpt_text', 'Brak analizy GPT')
            
            # Get score if available
            score = feedback.get('score', 0)
            score_text = f" (Score: {score})" if score > 0 else ""
            
            formatted = f"""
📊 RECENT GPT FEEDBACK dla {symbol} ({age_str}){score_text}:
{gpt_text}

Źródło: {source}
Timestamp: {feedback.get('timestamp', 'Unknown')}
"""
            
            return formatted.strip()
            
        except Exception as e:
            print(f"⚠️ Error formatting feedback: {e}")
            return f"⚠️ Błąd formatowania feedback dla {feedback.get('symbol', 'Unknown')}"
    
    def get_feedback_summary(self, hours: int = 2) -> Dict:
        """
        Get summary of recent GPT feedback activity
        
        Args:
            hours: Number of hours back to analyze
            
        Returns:
            Summary statistics
        """
        try:
            all_feedback = self.get_all_recent_feedback(hours)
            
            if not all_feedback:
                return {
                    'total_feedback': 0,
                    'unique_symbols': 0,
                    'avg_score': 0,
                    'high_score_count': 0,
                    'symbols_analyzed': []
                }
            
            # Calculate statistics
            scores = [f.get('score', 0) for f in all_feedback if f.get('score', 0) > 0]
            avg_score = sum(scores) / len(scores) if scores else 0
            high_score_count = len([s for s in scores if s >= 80])
            
            symbols = list(set([f.get('symbol', '') for f in all_feedback]))
            symbols = [s for s in symbols if s]  # Remove empty strings
            
            return {
                'total_feedback': len(all_feedback),
                'unique_symbols': len(symbols),
                'avg_score': round(avg_score, 1),
                'high_score_count': high_score_count,
                'symbols_analyzed': symbols[:10]  # Top 10 symbols
            }
            
        except Exception as e:
            print(f"⚠️ Error getting feedback summary: {e}")
            return {
                'total_feedback': 0,
                'unique_symbols': 0,
                'avg_score': 0,
                'high_score_count': 0,
                'symbols_analyzed': []
            }

def test_gpt_feedback_integration():
    """Test function for GPT feedback integration"""
    print("🧪 Testing GPT Feedback Integration...")
    
    integration = GPTFeedbackIntegration()
    
    # Test getting recent feedback summary
    print("\n📊 Recent feedback summary:")
    summary = integration.get_feedback_summary(hours=24)
    print(f"Total feedback entries: {summary['total_feedback']}")
    print(f"Unique symbols: {summary['unique_symbols']}")
    print(f"Average score: {summary['avg_score']}")
    print(f"High scores (≥80): {summary['high_score_count']}")
    print(f"Symbols: {', '.join(summary['symbols_analyzed'])}")
    
    # Test getting all recent feedback
    print("\n📋 All recent feedback (last 2 hours):")
    all_feedback = integration.get_all_recent_feedback(hours=2)
    for feedback in all_feedback[:3]:  # Show first 3
        print(f"- {feedback['symbol']} ({feedback['age_hours']:.1f}h ago) - Score: {feedback.get('score', 'N/A')}")
    
    # Test specific symbol lookup
    test_symbols = ['BTC', 'ETH', 'BTCUSDT', 'ETHUSDT']
    print(f"\n🔍 Testing specific symbol lookups:")
    for symbol in test_symbols:
        feedback = integration.get_recent_gpt_feedback(symbol, hours=24)
        if feedback:
            formatted = integration.format_feedback_for_pump_analysis(feedback)
            print(f"\n✅ Found feedback for {symbol}:")
            print(formatted[:200] + "..." if len(formatted) > 200 else formatted)
        else:
            print(f"❌ No recent feedback for {symbol}")

if __name__ == "__main__":
    test_gpt_feedback_integration()